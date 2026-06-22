#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# 模块信息
# ----------------------------------------------------------------------
# @Author  : Prog.le
# @Email   : Prog.le@outlook.com
# @Time    : 2026-06-22
# @FileName: ocr_pdf.py
# @Version : 1.3.0
# ----------------------------------------------------------------------



# 导入必要的库
import os

# 必须在 paddle 加载前设置，否则 PIR 执行器 bug (ConvertPirAttribute2RuntimeAttribute) 会造成页面崩溃
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLEX_HOME", os.path.join(os.getcwd(), ".paddlex"))

import time
import argparse
import logging
import sys
import re
from PyPDF2 import PdfReader, PdfWriter

# PP-ChatOCRv4 运行时补丁：修复 LLM JSON 数组/裸字符串解析 + 注入 few-shot 示例
# 无论当前是否使用 pp-chatocrv4 模型，提前 import 确保补丁就位
try:
    import chatocr_patch  # noqa: F401  (patches apply on import)

    chatocr_patch  # silence linter
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

# 设置日志配置
# 创建logs目录
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 设置日志文件路径（包含日期信息）
log_filename = os.path.join(logs_dir, f"ocr_pdf_{time.strftime('%Y-%m-%d')}.log")

# 配置日志同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,  # 初始默认值，后续会根据命令行参数更新
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(log_filename, encoding='utf-8')  # 文件输出
    ]
)
logger = logging.getLogger(__name__)

# 创建自定义的paddlex.utils.cache模块
class CustomCacheModule:
    def __init__(self):
        # 设置自定义的目录路径
        self.paddlex_dir = os.path.join(os.getcwd(), '.paddlex')
        self.temp_dir = os.path.join(self.paddlex_dir, 'temp')
        self.model_dir = os.path.join(self.paddlex_dir, 'models')
        
        # 创建目录结构
        for dir_path in [self.paddlex_dir, self.temp_dir, self.model_dir, 
                        os.path.join(self.paddlex_dir, "func_ret"), 
                        os.path.join(self.paddlex_dir, "locks"),
                        os.path.join(self.paddlex_dir, "official_models")]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"成功创建目录: {dir_path}")
                except Exception as e:
                    logger.error(f"创建目录失败: {dir_path}, 错误: {e}")
        
        # 设置模块变量
        self.DEFAULT_CACHE_DIR = self.paddlex_dir
        self.CACHE_DIR = self.paddlex_dir
        self.FUNC_CACHE_DIR = os.path.join(self.paddlex_dir, "func_ret")
        self.FILE_LOCK_DIR = os.path.join(self.paddlex_dir, "locks")
        self.TEMP_DIR = self.temp_dir
    
    def create_cache_dir(self, *args, **kwargs):
        """create cache dir"""
        pass
    
    def get_cache_dir(self, *args, **kwargs):
        """get cache dir"""
        return self.CACHE_DIR

# 创建模块实例
custom_cache_module = CustomCacheModule()

# 将自定义模块注入到sys.modules中
sys.modules['paddlex.utils.cache'] = custom_cache_module

# PaddleX 3.7.1 兼容补丁：paddlepaddle-gpu 2.6.2 的 AnalysisConfig 缺少 set_optimization_level
# PaddleX 内部调用该方法设置优化级别，但该 API 在 2.6.2 中不存在
try:
    import paddle.base.libpaddle  # noqa: F401
    paddle.base.libpaddle.AnalysisConfig.set_optimization_level = (
        lambda self, level: None
    )
except Exception:
    pass

# 现在导入其他模块
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pypdfium2 as pdfium
import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL
from device_utils import detect_device, verify_paddle_device
from tqdm import tqdm

class PDFOCRHandler:
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
                 lang='ch', model_size='medium',
                 output_formats=None,
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
        """
        初始化 PDF OCR 处理器

        Args:
            output_dir: 输出目录
            model: OCR 模型选择
                - 'paddleocr-vl': 0.9B 视觉语言多模态模型
                - 'pp-ocrv6': PP-OCRv6 通用文字识别 (默认，50 种语言)
                - 'pp-ocrv5': PP-OCRv5 旧版通用文字识别
                - 'pp-structurev3': 复杂文档结构化解析
                - 'pp-chatocrv4': 信息抽取 (需 API key)
            device: 推理设备 auto/gpu/cpu
            lang: PP-OCRv6/v5 语言选择
                - 'ch': 简体中文 (默认)
                - 'chinese_cht': 繁体中文
                - 'en': 英语
                - 'japan': 日语
                - 'korean': 韩语
                - 'latin': 拉丁语系
            model_size: PP-OCRv6 模型尺寸
                - 'medium': PP-OCRv6_medium  (默认, 最高精度)
                - 'small':  PP-OCRv6_small   (平衡精度与速度)
                - 'tiny':   PP-OCRv6_tiny    (极小体积, 边缘设备)
            output_formats: 输出格式列表, 默认 ['markdown','json','img','pdf']
            optimize_pdf: 是否优化 PDF
            optimize_level: PDF 优化级别
            grayscale: 是否使用灰度渲染
        """
        self.output_dir = output_dir
        self.model = model
        self.optimize_pdf_flag = optimize_pdf
        self.optimize_level = optimize_level
        self.grayscale = grayscale
        self.device = device
        self.lang = lang
        self.model_size = model_size

        # 输出格式选择
        if output_formats is None:
            output_formats = ['markdown', 'json', 'img', 'pdf']
        self.output_formats = output_formats

        # 设备检测
        self.device_info = detect_device(device)
        paddlex_device = self.device_info['paddlex_device']

        # 验证 PaddlePaddle 是否能实际使用该设备
        verify_result = verify_paddle_device(paddlex_device)
        if verify_result['fallback'] == 'true':
            corrected_device = verify_result['paddlex_device']
            logger.warning(f"设备从 {paddlex_device} 回退到 {corrected_device}: "
                           f"{verify_result['message']}")
            self.device_info['paddlex_device'] = corrected_device
            self.device_info['device_type'] = 'cpu (fallback from gpu)'
            paddlex_device = corrected_device

        logger.info(f"推理设备: {self.device_info['device_type']} "
                    f"(传入 PaddleOCR: {paddlex_device})")

        # 根据选择的模型配置PaddleOCR
        logger.info(f"正在初始化{model}模型...")
        if model == 'paddleocr-vl':
            # PaddleOCR-VL模型配置
            self.ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device,
                use_queues=False,  # 禁用多线程队列模式: 默认 YAML 中 use_queues=True, 但 Paddle 的 VLM 模型在多线程下不线程安全, 会抛出空异常
                enable_mkldnn=False  # 禁用 MKLDNN 避免 PIR+oneDNN 属性转换 bug (ConvertPirAttribute2RuntimeAttribute)
            )
        elif model == 'pp-structurev3':
            # PP-StructureV3模型配置
            self.ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device,
                enable_mkldnn=False  # 禁用 MKLDNN 避免 PIR+oneDNN 属性转换 bug
            )
        elif model == 'pp-chatocrv4':
            # PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用
            logger.error(f"PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用")
            raise ValueError(f"{model}模型需要额外的API配置，暂不支持直接使用")
        elif model in ('pp-ocrv6', 'pp-ocrv5'):
            # PP-OCRv6/v5 通用文字识别
            # PP-OCRv6 是 PaddleOCR 3.7+ 默认模型
            # 通过显式指定 text_*_model_name 锁定到 v6/v5,
            # 避免 paddleocr 升级时自动变更默认
            if model == 'pp-ocrv6':
                det_name = f"PP-OCRv6_{model_size}_det"
                rec_name = f"PP-OCRv6_{model_size}_rec"
            else:
                # 旧版 v5 仍保留兼容入口
                det_name = f"PP-OCRv5_{model_size}_det"
                rec_name = f"PP-OCRv5_{model_size}_rec"

            logger.info(f"使用 {model} ({model_size}) 模型, "
                        f"det={det_name}, rec={rec_name}, lang={lang}")
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device,
                lang=lang,
                text_detection_model_name=det_name,
                text_recognition_model_name=rec_name,
                enable_mkldnn=False  # 禁用 MKLDNN 避免 PIR+oneDNN 属性转换 bug (ConvertPirAttribute2RuntimeAttribute)
            )
        else:
            # 兜底: 未知模型走 PaddleOCR 默认 (v6 medium, ch)
            logger.warning(f"未知模型 '{model}', 使用 PaddleOCR 默认配置 (PP-OCRv6_medium, lang=ch)")
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device,
                lang='ch',
                text_detection_model_name='PP-OCRv6_medium_det',
                text_recognition_model_name='PP-OCRv6_medium_rec',
                enable_mkldnn=False  # 禁用 MKLDNN 避免 PIR+oneDNN 属性转换 bug (ConvertPirAttribute2RuntimeAttribute)
            )
        logger.info(f"{model}模型初始化完成")

        logger.info(f"使用OCR模型: {model} (lang={lang}, size={model_size})")
        logger.info(f"PDF优化: {'开启' if self.optimize_pdf_flag else '关闭'}")
        if self.optimize_pdf_flag:
            logger.info(f"优化级别: {self.optimize_level}")
        logger.info(f"灰度渲染: {'开启' if self.grayscale else '关闭'}")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize_pdf(self, pdf_path):
        """
        优化PDF文件，包括压缩、去重、结构优化等
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            str: 优化后的PDF文件路径
        """
        logger.info(f"开始优化PDF文件: {pdf_path}")
        
        # 生成优化后的文件名
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        optimized_pdf_path = os.path.join(self.output_dir, f"{filename}_optimized.pdf")
        
        try:
            # 读取原始PDF
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            
            # 复制页面并优化
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                
                # 添加页面到输出
                writer.add_page(page)
            
            # 设置压缩参数
            compression_level = 0
            if self.optimize_level == 'low':
                compression_level = 1
            elif self.optimize_level == 'medium':
                compression_level = 3
            elif self.optimize_level == 'high':
                compression_level = 5
            
            # 写入优化后的PDF
            with open(optimized_pdf_path, 'wb') as f:
                writer.write(f)
            
            # 获取文件大小，计算压缩率
            original_size = os.path.getsize(pdf_path)
            optimized_size = os.path.getsize(optimized_pdf_path)
            compression_ratio = (1 - optimized_size / original_size) * 100
            
            logger.info(f"PDF优化完成")
            logger.info(f"原始大小: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"优化后大小: {optimized_size / 1024 / 1024:.2f} MB")
            logger.info(f"压缩率: {compression_ratio:.2f}%")
            
            return optimized_pdf_path
            
        except Exception as e:
            logger.warning(f"PDF优化失败（将使用原始文件进行OCR识别）: {str(e)}")
            import traceback
            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
            # 如果优化失败，返回原始文件路径
            return pdf_path
    
    def process_pdf(self, pdf_path):
        """处理单个PDF文件的OCR识别"""
        import time  # 确保time模块可用
        import os  # 确保os模块在方法内可用
        start_time = time.time()
        success = False
        file_size = 0
        file_size_mb = 0  # 初始化文件大小变量，避免NameError
        total_pages = 0
        pdf_output_dir = None
        output_path_str = "N/A"
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(pdf_path)
            file_size_mb = file_size / (1024 * 1024)  # 转换为MB
            
            logger.info(f"开始处理PDF文件: {pdf_path}")
            logger.info(f"文件信息: 文件名={os.path.basename(pdf_path)}, 大小={file_size_mb:.2f}MB")
            
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            # MinerU 风格输出目录结构:
            #   {output_dir}/{filename}/
            #     images/page_N.jpg
            #     output.md
            #     layout.pdf
            pdf_output_dir = os.path.join(self.output_dir, filename)
            images_dir = os.path.join(pdf_output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # 优化PDF文件
            if self.optimize_pdf_flag:
                pdf_path = self.optimize_pdf(pdf_path)
            
            # 打开PDF文件
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            logger.info(f"PDF文件总页数: {total_pages}")
            
            # 识别结果
            ocr_results = []
            page_raw_results = []
            
            # 逐页处理（带进度条）
            for page_num in tqdm(range(total_pages), desc=f"OCR {filename}", unit="页", leave=True):
                
                try:
                    # 获取页面
                    page = pdf[page_num]
                    
                    # 将页面转换为图像
                    # 降低初始渲染分辨率，减少内存占用
                    bitmap = page.render(
                        scale=1.0,  # 降低初始缩放比例以提高性能
                        rotation=0,
                        # 使用灰度渲染可以进一步减少内存使用
                        grayscale=self.grayscale
                    )
                    
                    # 转换为numpy数组
                    img = bitmap.to_numpy()
                    
                    # 转换为OpenCV格式（BGR）
                    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # 检查图像尺寸，如果过大则进行缩放
                    max_size = 6000  # 降低最大尺寸以提高处理速度
                    target_resolution = 300  # 设置目标分辨率
                    
                    height, width = img_cv.shape[:2]
                    current_resolution = width * height
                    
                    if height > max_size or width > max_size:
                        # 计算缩放比例
                        scale_factor = max_size / max(height, width)
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        
                        logger.info(f"图像尺寸过大 ({width}x{height})，将缩放到 {new_width}x{new_height}")
                        
                        # 缩放图像
                        img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        height, width = img_cv.shape[:2]
                    
                    # 检查分辨率，如果过高则进一步降低
                    desired_max_resolution = 2000 * 2000  # 400万像素
                    if width * height > desired_max_resolution:
                        resolution_scale = (desired_max_resolution / (width * height)) ** 0.5
                        new_width = int(width * resolution_scale)
                        new_height = int(height * resolution_scale)
                        
                        logger.info(f"图像分辨率过高 ({width}x{height})，将缩放到 {new_width}x{new_height}")
                        
                        # 缩放图像
                        img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 执行OCR识别
                    logger.info(f"开始识别第 {page_num + 1} 页内容...")
                    
                    # 限制识别时间，避免长时间无响应
                    import time
                    start_ocr_time = time.time()
                    try:
                        result = self.ocr.predict(img_cv)
                        ocr_time = time.time() - start_ocr_time
                        logger.info(f"第 {page_num + 1} 页识别完成，耗时: {ocr_time:.2f}秒")
                    except Exception as e:
                        logger.error(f"第 {page_num + 1} 页识别失败: {str(e)}")
                        # 释放资源
                        del img_cv, img
                        if 'bitmap' in locals():
                            del bitmap
                        continue
                    
                    # 提取识别文本
                    page_text = []
                    try:
                        # 尝试处理predict方法的返回结果
                        if result:
                            # 将生成器转换为列表以便处理
                            result_list = list(result)
                            page_raw_results.append(result_list)
                            
                            # 根据模型类型处理不同的输出格式
                            if self.model == 'pp-structurev3':
                                # 处理PP-StructureV3模型的输出格式
                                for res in result_list:
                                    if hasattr(res, 'print') and callable(res.print):
                                        # 对于PP-StructureV3的结果对象
                                        # 尝试保存为markdown以获取结构化内容
                                        import json
                                        import tempfile
                                        import os
                                        
                                        # 创建临时目录保存结果
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            try:
                                                # 保存为JSON和Markdown
                                                if hasattr(res, 'save_to_json'):
                                                    res.save_to_json(save_path=tmpdir)
                                                    
                                                if hasattr(res, 'save_to_markdown'):
                                                    res.save_to_markdown(save_path=tmpdir)
                                                    
                                                # 读取Markdown结果
                                                markdown_files = [f for f in os.listdir(tmpdir) if f.endswith('.md')]
                                                if markdown_files:
                                                    markdown_path = os.path.join(tmpdir, markdown_files[0])
                                                    with open(markdown_path, 'r', encoding='utf-8') as f:
                                                        md_content = f.read()
                                                        page_text.append(md_content)
                                                
                                                # 如果没有Markdown，尝试读取JSON
                                                elif os.listdir(tmpdir):
                                                    json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
                                                    if json_files:
                                                        json_path = os.path.join(tmpdir, json_files[0])
                                                        with open(json_path, 'r', encoding='utf-8') as f:
                                                            json_content = json.load(f)
                                                            # 从JSON中提取文本
                                                            if isinstance(json_content, list):
                                                                for item in json_content:
                                                                    if isinstance(item, dict):
                                                                        if 'text' in item:
                                                                            page_text.append(item['text'])
                                                                    elif isinstance(item, str):
                                                                        page_text.append(item)
                                            except Exception as e:
                                                logger.error(f"处理PP-StructureV3结果时出错: {str(e)}")
                                    else:
                                        # 尝试直接提取文本
                                        if isinstance(res, dict):
                                            if 'text' in res:
                                                page_text.append(res['text'])
                                        elif isinstance(res, (list, tuple)):
                                            # 递归提取文本
                                            for item in res:
                                                if isinstance(item, dict) and 'text' in item:
                                                    page_text.append(item['text'])
                                                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                                    if isinstance(item[-1], str):
                                                        page_text.append(item[-1])
                                            
                            elif self.model == 'paddleocr-vl':
                                # 处理PaddleOCR-VL模型的输出格式
                                for res in result_list:
                                    if hasattr(res, 'print') and callable(res.print):
                                        # 对于PaddleOCR-VL的结果对象
                                        import tempfile
                                        import os
                                        import json
                                        
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            try:
                                                if hasattr(res, 'save_to_json'):
                                                    res.save_to_json(save_path=tmpdir)
                                                    
                                                # 读取JSON结果
                                                json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
                                                if json_files:
                                                    json_path = os.path.join(tmpdir, json_files[0])
                                                    with open(json_path, 'r', encoding='utf-8') as f:
                                                        json_content = json.load(f)
                                                        # 从JSON中提取文本
                                                        if isinstance(json_content, list):
                                                            for item in json_content:
                                                                if isinstance(item, dict):
                                                                    # 检查是否有parsing_res_list字段
                                                                    if 'parsing_res_list' in item:
                                                                        parsing_res_list = item['parsing_res_list']
                                                                        # 解析文本内容
                                                                        for parsing_item in parsing_res_list:
                                                                            if isinstance(parsing_item, str):
                                                                                # 查找content字段
                                                                                content_start = parsing_item.find('content:')
                                                                                if content_start != -1:
                                                                                    # 提取content字段内容
                                                                                    content = parsing_item[content_start + len('content:'):].strip()
                                                                                    if content:
                                                                                        page_text.append(content)
                                                                    elif 'text' in item:
                                                                        page_text.append(item['text'])
                                                                elif isinstance(item, str):
                                                                    page_text.append(item)
                                            except Exception as e:
                                                logger.error(f"处理PaddleOCR-VL结果时出错: {str(e)}")
                                    else:
                                        # 尝试直接提取文本
                                        if isinstance(res, dict):
                                            # 检查是否有parsing_res_list字段
                                            if 'parsing_res_list' in res:
                                                parsing_res_list = res['parsing_res_list']
                                                # 解析文本内容
                                                for parsing_item in parsing_res_list:
                                                    if isinstance(parsing_item, str):
                                                        # 查找content字段
                                                        content_start = parsing_item.find('content:')
                                                        if content_start != -1:
                                                            # 提取content字段内容
                                                            content = parsing_item[content_start + len('content:'):].strip()
                                                            if content:
                                                                page_text.append(content)
                                            elif 'text' in res:
                                                page_text.append(res['text'])
                                        elif isinstance(res, (list, tuple)):
                                            for item in res:
                                                if isinstance(item, dict):
                                                    if 'parsing_res_list' in item:
                                                        parsing_res_list = item['parsing_res_list']
                                                        # 解析文本内容
                                                        for parsing_item in parsing_res_list:
                                                            if isinstance(parsing_item, str):
                                                                # 查找content字段
                                                                content_start = parsing_item.find('content:')
                                                                if content_start != -1:
                                                                    # 提取content字段内容
                                                                    content = parsing_item[content_start + len('content:'):].strip()
                                                                    if content:
                                                                        page_text.append(content)
                                                    elif 'text' in item:
                                                        page_text.append(item['text'])
                                                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                                    if isinstance(item[-1], str):
                                                        page_text.append(item[-1])
                            else:
                                # 处理PP-OCRv5模型的输出格式
                                if result_list and isinstance(result_list[0], dict):
                                    # 如果是字典格式，检查是否有rec_texts字段
                                    if 'rec_texts' in result_list[0]:
                                        rec_texts = result_list[0]['rec_texts']
                                        page_text.extend(rec_texts)
                                    else:
                                        # 记录返回格式以便调试
                                        logger.debug(f"识别结果格式(字典)：{result_list[0].keys()}")
                                        # 尝试从其他可能的字段提取文本
                                        for item in result_list:
                                            if 'text' in item:
                                                page_text.append(item['text'])
                                elif result_list:
                                    # 如果不是字典格式，尝试其他方式提取
                                    logger.debug(f"识别结果格式(非字典)：{type(result_list[0])}")
                                    # 对于列表或元组格式，尝试提取文本
                                    for item in result_list:
                                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                                            # 可能是[(box, text), ...]格式
                                            page_text.extend([text for box, text in item])
                    except Exception as e:
                        logger.error(f"处理第 {page_num + 1} 页识别结果时出错: {str(e)}")
                        import traceback
                        logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
                        # 释放资源
                        del img_cv, img
                        if 'bitmap' in locals():
                            del bitmap
                        continue
                    
                    # 如果没有提取到文本，尝试使用备用方法
                    if not page_text:
                        logger.warning(f"第 {page_num + 1} 页未提取到文本，尝试使用备用方法")
                        # 尝试直接从result_list中提取文本
                        try:
                            # 检查是否是paddleocr-vl模型的结果格式
                            if self.model == 'paddleocr-vl':
                                for res in result_list:
                                    if isinstance(res, dict):
                                        # 检查是否有parsing_res_list字段
                                        if 'parsing_res_list' in res:
                                            parsing_res_list = res['parsing_res_list']
                                            # 直接解析parsing_res_list中的文本内容
                                            for parsing_item in parsing_res_list:
                                                if isinstance(parsing_item, str):
                                                    # 查找content字段
                                                    content_start = parsing_item.find('content:')
                                                    if content_start != -1:
                                                        # 提取content字段内容直到下一个分隔符
                                                        content_end = parsing_item.find('#################', content_start)
                                                        if content_end != -1:
                                                            content = parsing_item[content_start + len('content:'):content_end].strip()
                                                        else:
                                                            content = parsing_item[content_start + len('content:'):].strip()
                                                        if content:
                                                            page_text.append(content)
                            
                            # 如果还是没有提取到文本，使用最后的备用方法
                            if not page_text:
                                text_content = str(result_list)
                                if len(text_content) > 0:
                                    # 尝试从字符串中提取content字段内容
                                    import re
                                    content_pattern = r'content:\s*(.*?)\s*#################'
                                    content_matches = re.findall(content_pattern, text_content, re.DOTALL)
                                    if content_matches:
                                        page_text.extend(content_matches)
                                    else:
                                        page_text.append(text_content)
                        except Exception as e:
                            logger.error(f"备用方法提取文本失败: {str(e)}")
                            import traceback
                            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
                    
                    # 添加页面分隔符和识别结果
                    ocr_results.append(f"=== 第 {page_num + 1} 页 ===")
                    ocr_results.extend(page_text)
                    
                    # 保存页面图像为低质量 JPG（用于 output.md 和 layout.pdf）
                    try:
                        jpg_path = os.path.join(images_dir, f"page_{page_num + 1}.jpg")
                        cv2.imwrite(jpg_path, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    except Exception as e:
                        logger.warning(f"保存第 {page_num + 1} 页图像失败: {e}")
                    
                    # 释放当前页的资源，避免内存泄漏
                    del img_cv, img
                    if 'bitmap' in locals():
                        del bitmap
                    if 'result_list' in locals():
                        del result_list
                    if 'result' in locals():
                        del result
                    if 'rec_texts' in locals():
                        del rec_texts
                    
                    # 强制进行垃圾回收
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"处理第 {page_num + 1} 页时出错: {str(e)}")
                    # 继续处理下一页，而不是整个文件失败
                    continue
            
            # MinerU 风格输出（即使部分页面处理失败）
            if ocr_results:
                # 1) output.md — 每页一张图片引用，后接 OCR 文本
                md_path = os.path.join(pdf_output_dir, "output.md")
                self._build_markdown(ocr_results, images_dir, md_path, total_pages)
                logger.info(f"output.md 已生成: {md_path}")
                
                # 2) layout.pdf — 彩色标注框叠加 PDF
                pdf_path = os.path.join(pdf_output_dir, "layout.pdf")
                self._create_layout_pdf(images_dir, page_raw_results, pdf_path, total_pages)
                if os.path.exists(pdf_path):
                    logger.info(f"layout.pdf 已生成: {pdf_path}")
                
                # 3) output.json — 结构化 JSON 输出（如果启用）
                if 'json' in self.output_formats:
                    json_path = os.path.join(pdf_output_dir, "output.json")
                    self._build_json(ocr_results, json_path, total_pages, filename)
                
                logger.info(f"PDF 文件处理完成: {pdf_output_dir}")
                success = True
                return True
            else:
                logger.warning(f"PDF文件处理完成，但未识别到任何文本: {pdf_path}")
                success = False
                return False
            
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {pdf_path}，错误信息: {str(e)}")
            import traceback
            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
            return False
            
        finally:
            # 计算处理耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 记录综合日志
            logger.info("=" * 50)
            logger.info("OCR识别完成日志")
            logger.info(f"日期时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"文件名: {os.path.basename(pdf_path)}")
            logger.info(f"文件大小: {file_size_mb:.2f}MB")
            logger.info(f"总页数: {total_pages}")
            logger.info(f"处理耗时: {elapsed_time:.2f}秒")
            logger.info(f"处理结果: {'成功' if success else '失败'}")
            logger.info(f"输出路径: {pdf_output_dir if success else 'N/A'}")
            logger.info("=" * 50)
            
            # 将日志内容以表格形式输出到本地md文件
            log_md_path = os.path.join(os.getcwd(), 'ocr_logs.md')
            
            # 检查文件是否存在，如果不存在则创建并添加表头
            if not os.path.exists(log_md_path):
                with open(log_md_path, 'w', encoding='utf-8') as f:
                    f.write('# OCR识别日志\n\n')
                    f.write('| 日期时间 | 文件名 | 文件大小 | 总页数 | 处理耗时 | 处理结果 | 输出路径 |\n')
                    f.write('|---------|-------|---------|-------|---------|---------|---------|\n')
            
            # 准备日志数据行
            log_time = time.strftime('%Y-%m-%d %H:%M:%S')
            file_name = os.path.basename(pdf_path)
            file_size_str = f"{file_size_mb:.2f}MB"
            pages_str = str(total_pages)
            elapsed_str = f"{elapsed_time:.2f}秒"
            result_str = "成功" if success else "失败"
            output_path_str = pdf_output_dir if success else "N/A"
            
            # 生成Markdown表格行
            log_row = f"| {log_time} | {file_name} | {file_size_str} | {pages_str} | {elapsed_str} | {result_str} | {output_path_str} |\n"
            
            # 将日志行追加到文件
            with open(log_md_path, 'a', encoding='utf-8') as f:
                f.write(log_row)
            
            logger.info(f"日志已记录到Markdown文件: {log_md_path}")

    # ------------------------------------------------------------------
    # MinerU 风格输出辅助方法
    # ------------------------------------------------------------------

    def _build_markdown(self, ocr_results, images_dir, md_path, total_pages):
        """构建 output.md：每页一张图片引用 + OCR 文本"""
        lines = []
        for page_num in range(total_pages):
            lines.append(f"![page_{page_num + 1}](images/page_{page_num + 1}.jpg)")
            lines.append("")
        # OCR 文本内容
        if ocr_results:
            lines.append("---")
            lines.append("")
            lines.extend(ocr_results)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _build_json(self, ocr_results, json_path, total_pages, filename):
        """构建 output.json：包含文件名、总页数、每页识别文本"""
        import json
        pages = []
        current_page = 0
        current_lines = []
        for line in ocr_results:
            sep_match = re.match(r'^=== 第 (\d+) 页 ===$', line)
            if sep_match:
                if current_page > 0:
                    pages.append({
                        "page_num": current_page,
                        "text": "\n".join(current_lines)
                    })
                current_page = int(sep_match.group(1))
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            pages.append({
                "page_num": current_page,
                "text": "\n".join(current_lines)
            })
        output = {
            "filename": filename,
            "total_pages": total_pages,
            "pages": pages
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("JSON 输出已保存: %s", json_path)

    def _create_layout_pdf(self, images_dir, page_raw_results, pdf_path, total_pages):
        """创建带彩色标注框的 layout.pdf（MinerU 风格）
        
        颜色方案（参考 MinerU）:
            text    → 蓝   (255, 0, 0)
            title   → 红   (0, 0, 255)
            table   → 绿   (0, 255, 0)
            figure  → 橙   (0, 165, 255)
            formula → 紫   (128, 0, 128)
            default → 浅蓝 (200, 150, 50)
        """
        annotated_paths = []
        
        for page_idx in range(total_pages):
            jpg_path = os.path.join(images_dir, f"page_{page_idx + 1}.jpg")
            if not os.path.exists(jpg_path):
                continue
            
            img = cv2.imread(jpg_path)
            if img is None:
                continue
            
            # 获取当前页的检测结果
            if page_idx < len(page_raw_results):
                for res in page_raw_results[page_idx]:
                    self._draw_detection(img, res)
            
            # 保存标注图像
            annotated_path = os.path.join(images_dir, f"page_{page_idx + 1}_layout.jpg")
            cv2.imwrite(annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_paths.append(annotated_path)
        
        # 组装 PDF
        if annotated_paths:
            try:
                import img2pdf
                with open(pdf_path, 'wb') as f:
                    f.write(img2pdf.convert([str(p) for p in annotated_paths]))
                # 清理临时标注图
                for p in annotated_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            except ImportError:
                logger.error("缺少 img2pdf, layout.pdf 未生成。请执行: pip install img2pdf")
            except Exception as e:
                logger.error(f"生成 layout.pdf 失败: {e}")

    def _draw_detection(self, img, res):
        """在图像上绘制单个检测结果的标注框（含半透明位置覆盖层）"""
        boxes = []
        labels = []  # 每个box对应的文字
        colors = []  # 每个box对应的颜色
        color = (200, 150, 50)  # default: 浅蓝/棕
        label = ''

        # --- PP-StructureV3 OCRResult 格式 (paddlex OCRPipeline) ---
        if hasattr(res, 'json') and isinstance(res.json, dict) and 'res' in res.json:
            res_data = res.json['res']
            dt_polys = res_data.get('dt_polys', [])
            rec_texts = res_data.get('rec_texts', [])
            for i, poly in enumerate(dt_polys):
                if isinstance(poly, (list, tuple)) and len(poly) >= 4:
                    # poly = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    box_pts = [[int(p[0]), int(p[1])] for p in poly[:4]]
                    boxes.append(box_pts)
                    colors.append((255, 0, 0))  # 蓝 = text
                    labels.append(str(rec_texts[i]) if i < len(rec_texts) else '')

        # --- 结构模型格式 (PaddleOCR-VL / 旧PP-StructureV3) ---
        elif hasattr(res, 'res'):
            for item in res.res:
                if isinstance(item, dict) and 'bbox' in item:
                    bbox = item['bbox']  # [x1, y1, x2, y2]
                    # bbox 可能是 [x1,y1,x2,y2] 或 [[x1,y1],[x2,y2],...]
                    if (isinstance(bbox, (list, tuple)) and len(bbox) >= 4
                            and all(isinstance(v, (int, float)) for v in bbox[:4])):
                        # [x1, y1, x2, y2] 格式
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        box_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    elif (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                          and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in bbox)):
                        # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 格式
                        box_pts = [[int(p[0]), int(p[1])] for p in bbox[:4]]
                    else:
                        continue
                    item_type = item.get('type', 'text')
                    color = {
                        'text': (255, 0, 0),      # 蓝
                        'title': (0, 0, 255),     # 红
                        'table': (0, 255, 0),     # 绿
                        'figure': (0, 165, 255),  # 橙
                        'formula': (128, 0, 128), # 紫
                        'header': (255, 255, 0),  # 青
                        'footer': (255, 255, 0),  # 青
                    }.get(item_type, (200, 150, 50))
                    label = str(item.get('text', ''))
                    boxes.append(box_pts)
                    colors.append(color)
                    labels.append(label)

        # --- pp-ocrv5 格式: [box_coords, (text, conf)] ---
        elif isinstance(res, (list, tuple)) and len(res) >= 2:
            box_coords = res[0]
            text_info = res[1]
            if (isinstance(box_coords, (list, tuple)) and len(box_coords) == 4
                    and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box_coords)):
                boxes.append(box_coords)
                colors.append((255, 0, 0))  # 蓝 = text
                if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                    labels.append(str(text_info[0]))
                else:
                    labels.append('')

        # 绘制所有框（半透明填充 + 轮廓）
        overlay = None
        for box, box_color, box_label in zip(boxes, colors, labels):
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            # 半透明覆盖层（首次使用时创建）
            if overlay is None:
                overlay = np.zeros_like(img)
            cv2.fillPoly(overlay, [pts], box_color)
            # 轮廓线（直接画在 img 上，保持清晰）
            cv2.polylines(img, [pts], isClosed=True, color=box_color, thickness=2)

        # --- 半透明混合（alpha=0.10 浅色覆盖，避免遮挡原文） ---
        if overlay is not None:
            cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
            logger.debug("_draw_detection: 绘制了 %d 个半透明覆盖框", len(boxes))
        elif boxes:
            logger.warning("_draw_detection: 有 %d 个框但 overlay 未创建", len(boxes))
        else:
            logger.warning("_draw_detection: 未识别到任何检测框 (res 类型: %s, hasattr json: %s, hasattr res: %s)",
                          type(res).__name__, hasattr(res, 'json'), hasattr(res, 'res'))

        # --- 文字标签（支持 Unicode/中文，使用 PIL 避免乱码） ---
        self._draw_text_labels(img, boxes, colors, labels)

    _CJK_FONT_CACHE = None  # 类级缓存，只搜索一次字体

    @staticmethod
    def _get_cjk_font_path():
        """查找系统中可用的中文字体路径（结果缓存避免反复搜索）"""
        if PDFOCRHandler._CJK_FONT_CACHE is not None:
            return PDFOCRHandler._CJK_FONT_CACHE

        # 优先使用 PaddleX 自带的字体
        paddlex_font_dir = os.path.join(os.getcwd(), '.paddlex', 'fonts')
        candidates = [
            os.path.join(paddlex_font_dir, 'PingFang-SC-Regular.ttf'),
            os.path.join(paddlex_font_dir, 'simfang.ttf'),
            # Windows 系统字体
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/SIMHEI.TTF',
            'C:/Windows/Fonts/simsun.ttc',
            # Linux/macOS 常用字体
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
        for path in candidates:
            if os.path.isfile(path):
                logger.info("找到中文字体: %s", path)
                PDFOCRHandler._CJK_FONT_CACHE = path
                return path

        logger.warning("未找到中文字体，中文标签将显示为方框/乱码")
        PDFOCRHandler._CJK_FONT_CACHE = False
        return None

    def _draw_text_labels(self, img, boxes, colors, labels):
        """绘制文字标签，支持中文（Unicode）显示

        纯 ASCII 标签走 OpenCV 更快，含中文的走 PIL 避免乱码。
        """
        has_unicode = any(l and not l.isascii() for l in labels)
        if not has_unicode:
            # 快速路径：全部是 ASCII，直接使用 cv2
            for box, box_color, box_label in zip(boxes, colors, labels):
                if not box_label:
                    continue
                text_x, text_y = box[0]
                (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (text_x, text_y - th - 4),
                              (text_x + tw, text_y), box_color, -1)
                cv2.putText(img, box_label, (text_x, text_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return

        # 慢速路径：含 Unicode 文字，使用 PIL 渲染
        font_path = self._get_cjk_font_path()
        if not font_path:
            # 没有中文字体，退回到 cv2（虽然显示乱码，但有背景框标记位置）
            for box, box_color, box_label in zip(boxes, colors, labels):
                if not box_label:
                    continue
                text_x, text_y = box[0]
                (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (text_x, text_y - th - 4),
                              (text_x + tw, text_y), box_color, -1)
            return

        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, 14)

        # 批量转换：OpenCV BGR → PIL RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        for box, box_color, box_label in zip(boxes, colors, labels):
            if not box_label:
                continue
            text_x, text_y = int(box[0][0]), int(box[0][1])
            # 用 PIL 获取文本尺寸（支持中英文混排）
            bbox = draw.textbbox((0, 0), box_label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # 背景矩形（PIL 使用 RGB）
            pil_color = (int(box_color[2]), int(box_color[1]), int(box_color[0]))
            draw.rectangle([(text_x, text_y - th - 4),
                            (text_x + tw, text_y)], fill=pil_color)
            # 白色文字
            draw.text((text_x, text_y - th - 4), box_label, font=font, fill=(255, 255, 255))

        # 批量转换回 OpenCV BGR
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

class PDFFileHandler(FileSystemEventHandler):
    """监控目录中的新PDF文件（同步处理）"""
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
                 lang='ch', model_size='medium',
                 output_formats=None,
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
        self.output_dir = output_dir
        self.model = model
        self.device = device
        self.lang = lang
        self.model_size = model_size
        self.optimize_pdf_flag = optimize_pdf
        self.optimize_level = optimize_level
        self.grayscale = grayscale
        if output_formats is None:
            output_formats = ['markdown', 'json', 'img', 'pdf']
        self.output_formats = output_formats
        logger.info(f"初始化守护模式处理器，使用模型: {model}")
    
    def on_created(self, event):
        """当有新文件创建时触发"""
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            logger.info(f"检测到新的PDF文件: {event.src_path}")
            # 等待文件完全写入
            time.sleep(1)
            # 直接同步处理
            self.process_pdf_task(event.src_path)
    
    def process_pdf_task(self, pdf_path):
        """处理单个PDF文件的任务"""
        logger.info(f"开始处理文件: {os.path.basename(pdf_path)}")
        
        # 每个任务创建自己的OCR处理器
        ocr_handler = PDFOCRHandler(
            self.output_dir, 
            self.model,
            device=self.device,
            output_formats=self.output_formats,
            optimize_pdf=self.optimize_pdf_flag,
            optimize_level=self.optimize_level,
            grayscale=self.grayscale
        )
        try:
            result = ocr_handler.process_pdf(pdf_path)
            logger.info(f"完成处理文件: {os.path.basename(pdf_path)}, 结果: {'成功' if result else '失败'}")
            return result
        except Exception as e:
            logger.error(f"处理文件 {os.path.basename(pdf_path)} 时出错: {str(e)}")
            import traceback
            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
            return False
    
    def shutdown(self):
        """关闭处理器"""
        logger.info("正在关闭守护模式处理器...")
        logger.info("守护模式处理器已关闭")

def run_manual_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium',
                    output_formats=None,
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
    """手动模式：处理输入目录中已存在的所有PDF文件（同步处理）"""
    logger.info(f"手动模式启动，处理目录: {input_dir}")
    
    # 获取输入目录中的所有PDF文件
    pdf_files = [f for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) 
                and f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.info(f"目录中没有找到PDF文件: {input_dir}")
        return
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 初始化OCR处理器
    ocr_handler = PDFOCRHandler(
        output_dir,
        model,
        device=device,
        lang=lang,
        model_size=model_size,
        output_formats=output_formats,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )

    # 同步处理所有PDF文件（带进度条）
    success_count = 0
    failed_count = 0
    
    for pdf_file in tqdm(pdf_files, desc=f"批量处理 {os.path.basename(input_dir)}", unit="file"):
        pdf_path = os.path.join(input_dir, pdf_file)
        
        try:
            result = ocr_handler.process_pdf(pdf_path)
            if result:
                success_count += 1
                logger.info(f"完成处理文件: {pdf_file}, 结果: 成功")
            else:
                failed_count += 1
                logger.info(f"完成处理文件: {pdf_file}, 结果: 失败")
        except Exception as e:
            logger.error(f"处理文件 {pdf_file} 时出错: {str(e)}")
            import traceback
            logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
            failed_count += 1
    
    logger.info(f"手动模式处理完成，成功: {success_count} 个，失败: {failed_count} 个，总计: {len(pdf_files)} 个")

def run_daemon_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium',
                    output_formats=None,
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
    """守护模式：持续监控输入目录，处理新的PDF文件"""
    logger.info(f"守护模式启动，监控目录: {input_dir}")
    
    # 创建事件处理器
    event_handler = PDFFileHandler(
        output_dir,
        model,
        device=device,
        lang=lang,
        model_size=model_size,
        output_formats=output_formats,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
    
    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)
    
    # 启动观察者
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("守护模式停止")
    
    # 停止观察者
    observer.stop()
    observer.join()
    
    # 关闭处理器
    event_handler.shutdown()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF文字识别工具')
    parser.add_argument('-i', '--input', required=True, help='输入路径（支持单个PDF文件或目录）')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    parser.add_argument('-m', '--mode', choices=['manual', 'daemon'], default='manual', 
                       help='工作模式：manual（手动模式）或 daemon（守护模式）')
    parser.add_argument('-model', '--model',
                       choices=['paddleocr-vl', 'pp-ocrv6', 'pp-ocrv5', 'pp-structurev3', 'pp-chatocrv4'],
                       default='pp-ocrv6',
                       help='OCR模型选择: paddleocr-vl (多模态文档解析) / '
                            'pp-ocrv6 (PP-OCRv6 通用文字识别, 50 种语言, 默认) / '
                            'pp-ocrv5 (旧版) / '
                            'pp-structurev3 (复杂文档解析) / '
                            'pp-chatocrv4 (智能信息抽取, 需 API key)')
    parser.add_argument('--lang', default='ch',
                       choices=['ch', 'chinese_cht', 'en', 'japan', 'korean', 'latin'],
                       help='语言 (PP-OCRv6/v5): ch (简体中文, 默认) / '
                            'chinese_cht (繁体中文) / en (英语) / '
                            'japan (日语) / korean (韩语) / latin (拉丁语系)')
    parser.add_argument('--model-size', default='medium',
                       choices=['medium', 'small', 'tiny'],
                       help='PP-OCRv6 模型尺寸: medium (最高精度, 默认) / '
                            'small (平衡) / tiny (极小, 边缘设备)')
    parser.add_argument('-l', '--log-level', choices=LOG_LEVELS.keys(), default='info', 
                       help='日志输出级别：debug、info、warning、error、critical，默认：info')
    parser.add_argument('--optimize-pdf', action='store_true', help='是否优化PDF文件，默认：False')
    parser.add_argument('--optimize-level', choices=['low', 'medium', 'high'], default='medium', 
                       help='PDF优化级别，可选值：low、medium、high，默认：medium')
    parser.add_argument('--grayscale', action='store_true', help='是否使用灰度渲染，默认：False')
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto',
                       help='推理设备: auto(自动检测), gpu(强制GPU), cpu(强制CPU)，默认 auto')
    parser.add_argument('-of', '--output-formats',
                       type=str,
                       default='markdown,json,img,pdf',
                       help='输出格式 (逗号分隔). 可选值: markdown, json, img, pdf. '
                            '默认全部输出. 提示: pp-ocrv6 不支持 markdown, 将自动跳过. '
                            '示例: -of markdown,json')
    
    args = parser.parse_args()
    
    # 解析输出格式
    output_formats = [fmt.strip() for fmt in args.output_formats.split(',') if fmt.strip()]
    
    # 设置日志级别
    log_level = LOG_LEVELS[args.log_level]
    logging.getLogger().setLevel(log_level)
    # 同时设置paddleocr相关日志器的级别
    for logger_name in ['paddleocr', 'paddle', 'ppocr', 'paddlex']:
        logging.getLogger(logger_name).setLevel(log_level)
    logger.info(f"日志级别已设置为：{args.log_level}")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 输入是单个PDF文件
        if not args.input.lower().endswith('.pdf'):
            logger.error(f"输入文件不是PDF格式: {args.input}")
            return
        
        logger.info(f"手动模式启动，处理单个文件: {args.input}")
        ocr_handler = PDFOCRHandler(
            args.output,
            args.model,
            device=args.device,
            lang=args.lang,
            model_size=args.model_size,
            output_formats=output_formats,
            optimize_pdf=args.optimize_pdf,
            optimize_level=args.optimize_level,
            grayscale=args.grayscale
        )
        ocr_handler.process_pdf(args.input)
        logger.info("单个文件处理完成")
    elif os.path.isdir(args.input):
        # 输入是目录
        # 根据模式运行
        if args.mode == 'manual':
            run_manual_mode(
                args.input,
                args.output,
                args.model,
                device=args.device,
                lang=args.lang,
                model_size=args.model_size,
                output_formats=output_formats,
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
        else:
            run_daemon_mode(
                args.input,
                args.output,
                args.model,
                device=args.device,
                lang=args.lang,
                model_size=args.model_size,
                output_formats=output_formats,
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
    else:
        logger.error(f"输入路径不存在: {args.input}")
        return

if __name__ == '__main__':
    main()
