from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional
import os
import tempfile
import logging
from ocr_pdf import PDFOCRHandler

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

# 根路径
@app.get("/")
def root():
    return {
        "message": "PDF OCR API服务",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/ocr/pdf"
        ]
    }

# 健康检查
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]
    }

# OCR处理接口
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="pp-ocrv5", description="OCR模型选择: pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4"),
    optimize_pdf: Optional[bool] = Form(default=False, description="是否优化PDF文件"),
    optimize_level: Optional[str] = Form(default="medium", description="PDF优化级别: low, medium, high"),
    grayscale: Optional[bool] = Form(default=False, description="是否使用灰度渲染")
):
    """
    处理PDF文件的OCR识别
    
    Args:
        file: 上传的PDF文件
        model: OCR模型选择，可选值: pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4
        optimize_pdf: 是否优化PDF文件
        optimize_level: PDF优化级别，可选值: low, medium, high
        grayscale: 是否使用灰度渲染
    
    Returns:
        识别结果
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传PDF文件")
    
    # 验证模型选择
    valid_models = ["pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"模型选择错误，请选择以下模型之一: {', '.join(valid_models)}")
    
    # 验证优化级别
    valid_optimize_levels = ["low", "medium", "high"]
    if optimize_level not in valid_optimize_levels:
        raise HTTPException(status_code=400, detail=f"优化级别选择错误，请选择以下级别之一: {', '.join(valid_optimize_levels)}")
    
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
            
            # 初始化OCR处理器
            logger.info(f"初始化OCR处理器，使用模型: {model}")
            ocr_handler = PDFOCRHandler(
                output_dir, 
                model,
                optimize_pdf=optimize_pdf,
                optimize_level=optimize_level,
                grayscale=grayscale
            )
            
            # 处理PDF文件
            logger.info(f"开始处理PDF文件: {file.filename}")
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
            
            # 返回识别结果
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "filename": file.filename,
                    "model": model,
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

# 运行API服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
