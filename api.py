from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import json          # ← 新增：用于序列化 413 响应
import asyncio       # ← 新增：用于 handler 缓存的 Lock

# 必须在 paddle 加载前设置，否则 PIR 执行器 bug (ConvertPirAttribute2RuntimeAttribute) 会造成页面崩溃
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import tempfile
import logging
from ocr_pdf import PDFOCRHandler
from device_utils import detect_device

# ---------------------------------------------------------------------------
# PDFOCRHandler 缓存 + 并发控制
# 实现：(model, device, lang, model_size, optimize_pdf, optimize_level, grayscale)
#       七元组到 handler 实例的映射，避免每次请求重复初始化（30s–2min）
# 注意：PaddleOCR 实例非线程安全，每个 handler 绑定一个 asyncio.Lock
# ---------------------------------------------------------------------------
_HANDLER_CACHE: dict[tuple, "PDFOCRHandler"] = {}
_HANDLER_LOCKS: dict[tuple, asyncio.Lock] = {}


def get_handler(
    output_dir: str,
    model: str,
    device: str,
    lang: str,
    model_size: str,
    optimize_pdf: bool,
    optimize_level: str,
    grayscale: bool,
) -> "PDFOCRHandler":
    """获取或创建 PDFOCRHandler 实例（按模型参数缓存）

    Cache key = (model, device, lang, model_size, optimize_pdf, optimize_level, grayscale)
    output_dir 不参与 key（因每次请求不同），在返回缓存实例时覆盖。
    """
    key = (model, device, lang, model_size, optimize_pdf, optimize_level, grayscale)
    handler = _HANDLER_CACHE.get(key)
    if handler is not None:
        # 缓存命中：更新 output_dir（每次请求不同临时目录）
        handler.output_dir = output_dir
        return handler
    # 缓存未命中：创建新 handler 并缓存
    handler = PDFOCRHandler(
        output_dir, model,
        device=device,
        lang=lang,
        model_size=model_size,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale,
    )
    _HANDLER_CACHE[key] = handler
    return handler


def clear_handler_cache():
    """清空 handler 缓存（测试环境使用，避免 mock 实例跨测试污染）"""
    _HANDLER_CACHE.clear()
    _HANDLER_LOCKS.clear()


# PP-ChatOCRv4 运行时补丁：修复 LLM JSON 数组/裸字符串解析 + 注入 few-shot 示例
try:
    import chatocr_patch  # noqa: F401  (patches apply on import)

    chatocr_patch
except ImportError:
    pass

# 配置日志级别映射
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 设置默认日志级别
DEFAULT_LOG_LEVEL = os.environ.get('LOG_LEVEL', 'info').lower()
log_level = LOG_LEVELS.get(DEFAULT_LOG_LEVEL, logging.INFO)

# 设置日志配置
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 设置paddleocr相关日志器的级别
for logger_name in ['paddleocr', 'paddle', 'ppocr', 'paddlex']:
    logging.getLogger(logger_name).setLevel(log_level)

logger.info(f"日志级别已设置为：{DEFAULT_LOG_LEVEL}")

# 创建FastAPI应用
app = FastAPI(
    title="PDF OCR API服务",
    description="提供PDF文档的OCR识别服务，支持多种模型",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# 上传大小限制中间件（必须在 CORS 之前注册，作为最外层先检查大小）
# ---------------------------------------------------------------------------
_MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "200"))
_MAX_UPLOAD_SIZE = _MAX_UPLOAD_SIZE_MB * 1024 * 1024


class MaxBodySizeMiddleware:
    """ASGI 中间件：限制请求体大小，超过时返回 413 Payload Too Large

    检查 Content-Length 头，若超过阈值则在请求到达路由之前直接拒绝。
    注册为最外层中间件，超大 body 不会进入 CORS 和路由处理。
    """

    def __init__(self, app, max_size: int = _MAX_UPLOAD_SIZE):
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 读取 Content-Length 头（bytes 格式）
        content_length = 0
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    content_length = int(value)
                except (ValueError, TypeError):
                    pass
                break

        if content_length > self.max_size:
            body = json.dumps({
                "detail": f"文件大小超过限制，最大允许 {_MAX_UPLOAD_SIZE_MB} MB"
            }).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        await self.app(scope, receive, send)


# 添加中间件（注意：FastAPI 中间件为 LIFO 顺序——后注册的最先执行）
# MaxBodySize 后注册，作为最外层，请求先检查大小再进入 CORS 和路由
app.add_middleware(MaxBodySizeMiddleware, max_size=_MAX_UPLOAD_SIZE)

# CORS 中间件（允许 Web 前端跨域调用）
# 安全说明：
# - 当 CORS_ALLOW_ORIGINS 环境变量未设置时，使用 ["*"] + allow_credentials=False
# - 当设置了 CORS_ALLOW_ORIGINS（逗号分隔），使用白名单 + allow_credentials=True
_cors_origins_str = os.environ.get("CORS_ALLOW_ORIGINS", "")
if _cors_origins_str:
    _cors_allow_origins = [origin.strip() for origin in _cors_origins_str.split(",") if origin.strip()]
else:
    _cors_allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=bool(_cors_origins_str),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 根路径
@app.get("/")
def root():
    return {
        "message": "PDF OCR API服务",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/device",
            "/ocr/pdf"
        ]
    }

# 健康检查
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"],
        "device_modes": ["auto", "gpu", "cpu"],
        "default_device": "auto"
    }

# 设备检测接口
@app.get("/device")
def device_info():
    """检测当前计算设备信息"""
    info = detect_device()
    return {
        "status": "success",
        "device": info.get("device", "unknown"),
        "device_type": info.get("device_type", "unknown"),
        "paddlex_device": info.get("paddlex_device", "cpu"),
        "cuda_version": info.get("cuda_version") or None,
        "gpu_info": info.get("gpu_info") or None,
        "detail": info.get("detail", ""),
        "supported_devices": ["auto", "gpu", "cpu"],
        "note": "device=auto 时自动检测; device=gpu 强制使用 GPU; device=cpu 强制使用 CPU"
    }

# OCR处理接口
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="pp-ocrv6", description="OCR模型选择: pp-ocrv6 (默认), pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4"),
    device: Optional[str] = Form(default="auto", description="推理设备: auto(自动检测), gpu(强制GPU), cpu(强制CPU)"),
    lang: Optional[str] = Form(default="ch", description="语言 (PP-OCRv6/v5): ch, chinese_cht, en, japan, korean, latin"),
    model_size: Optional[str] = Form(default="medium", description="PP-OCRv6 模型尺寸: medium, small, tiny"),
    optimize_pdf: Optional[bool] = Form(default=False, description="是否优化PDF文件"),
    optimize_level: Optional[str] = Form(default="medium", description="PDF优化级别: low, medium, high"),
    grayscale: Optional[bool] = Form(default=False, description="是否使用灰度渲染")
):
    """
    处理PDF文件的OCR识别

    Args:
        file: 上传的PDF文件
        model: OCR模型选择，可选值: pp-ocrv6, pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4
        lang: 语言 (PP-OCRv6/v5)
        model_size: PP-OCRv6 模型尺寸 (medium/small/tiny)
        optimize_pdf: 是否优化PDF
        optimize_level: PDF优化级别，可选值: low, medium, high
        grayscale: 是否使用灰度渲染

    Returns:
        识别结果
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传PDF文件")
    
    # 验证模型选择
    valid_models = ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"模型选择错误，请选择以下模型之一: {', '.join(valid_models)}")
    
    # 验证优化级别
    valid_optimize_levels = ["low", "medium", "high"]
    if optimize_level not in valid_optimize_levels:
        raise HTTPException(status_code=400, detail=f"优化级别选择错误，请选择以下级别之一: {', '.join(valid_optimize_levels)}")
    
    # 验证设备参数
    valid_devices = ["auto", "gpu", "cpu"]
    if device not in valid_devices:
        raise HTTPException(status_code=400, detail=f"设备参数错误，请选择以下之一: {', '.join(valid_devices)}")
    
    try:
        # 创建临时目录保存上传的PDF文件
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存上传的PDF文件
            pdf_path = os.path.join(tmp_dir, file.filename)
            with open(pdf_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # 创建临时输出目录
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取或创建OCR处理器（工厂函数，带缓存+并发锁）
            logger.info(f"获取OCR处理器，使用模型: {model}")
            ocr_handler = get_handler(
                output_dir, model,
                device=device,
                lang=lang,
                model_size=model_size,
                optimize_pdf=optimize_pdf,
                optimize_level=optimize_level,
                grayscale=grayscale,
            )

            # 处理PDF文件（同一 handler 的并发调用通过 asyncio.Lock 串行化）
            logger.info(f"开始处理PDF文件: {file.filename}")
            lock = _HANDLER_LOCKS.setdefault(
                (model, device, lang, model_size, optimize_pdf, optimize_level, grayscale),
                asyncio.Lock(),
            )
            async with lock:
                success = ocr_handler.process_pdf(pdf_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="PDF文件处理失败")
            
            # 读取识别结果
            txt_filename = os.path.splitext(file.filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            if not os.path.exists(txt_path):
                raise HTTPException(status_code=500, detail="识别结果文件生成失败")
            
            with open(txt_path, "r", encoding="utf-8") as f:
                ocr_result = f.read()
            
            # 检测实际使用的设备
            device_info = detect_device(device if device != "auto" else None)

            # 返回识别结果
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "filename": file.filename,
                    "model": model,
                    "lang": lang,
                    "model_size": model_size,
                    "device": device,
                    "device_info": device_info,
                    "result": ocr_result
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理PDF文件时发生错误: {str(e)}")
        import traceback
        logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理PDF文件时发生错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("UVICORN_RELOAD", "0") == "1"
    )
