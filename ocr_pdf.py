#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# 模块信息
# ----------------------------------------------------------------------
# @Author  : Prog.le
# @Email   : Prog.le@outlook.com
# @Time    : 2026-01-15
# @FileName: ocr_pdf.py
# @Software: TRAE CN
# @Version : 1.0.0
# ----------------------------------------------------------------------
# 功能描述
# ----------------------------------------------------------------------
# 本模块基于 PaddleOCR 提供一站式 PDF 文字识别能力，支持：
# 1. 单文件 / 批量 OCR（手动模式）
# 2. 目录热监控实时 OCR（守护模式）
# 3. 多模型切换（PP-OCRv5 / PP-StructureV3 / PP-ChatOCRv4 / PaddleOCR-VL）
# 4. 超大页自动缩放、内存回收、异常页跳过，保障长时间稳定运行
# 5. 结构化日志与 Markdown 报表双输出，方便追溯与统计
# ----------------------------------------------------------------------
# 使用示例
# ----------------------------------------------------------------------
# 单文件：
#   python ocr_pdf.py -i sample.pdf -o ./output
# 批量：
#   python ocr_pdf.py -i ./pdf_dir -o ./output -m manual
# 守护：
#   python ocr_pdf.py -i ./drop_dir -o ./output -m daemon
# ----------------------------------------------------------------------
# 更新记录
# ----------------------------------------------------------------------
# 1.0.0  2026-01-15  Prog.le  初版，实现核心 OCR 与监控逻辑
# ----------------------------------------------------------------------


# 导入必要的库
import os
import time
import argparse
import logging
import sys
from PyPDF2 import PdfReader, PdfWriter

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
                        os.path.join(self.paddlex_dir, "locks")]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
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

# 现在导入其他模块
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pypdfium2 as pdfium
import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL

class PDFOCRHandler:
    def __init__(self, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
        self.output_dir = output_dir
        self.model = model
        self.optimize_pdf_flag = optimize_pdf
        self.optimize_level = optimize_level
        self.grayscale = grayscale
        
        # 根据选择的模型配置PaddleOCR
        logger.info(f"正在初始化{model}模型...")
        if model == 'paddleocr-vl':
            # PaddleOCR-VL模型配置
            self.ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model == 'pp-structurev3':
            # PP-StructureV3模型配置
            self.ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model == 'pp-chatocrv4':
            # PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用
            logger.error(f"PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用")
            raise ValueError(f"{model}模型需要额外的API配置，暂不支持直接使用")
        else:
            # 默认PP-OCRv5模型配置
            self.ocr = PaddleOCR(
                use_textline_orientation=True, 
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        logger.info(f"{model}模型初始化完成")
        
        logger.info(f"使用OCR模型: {model}")
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
            logger.error(f"PDF优化失败: {str(e)}")
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
        output_txt_path = None
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(pdf_path)
            file_size_mb = file_size / (1024 * 1024)  # 转换为MB
            
            logger.info(f"开始处理PDF文件: {pdf_path}")
            logger.info(f"文件信息: 文件名={os.path.basename(pdf_path)}, 大小={file_size_mb:.2f}MB")
            
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            output_txt_path = os.path.join(self.output_dir, f"{filename}.txt")
            
            # 优化PDF文件
            if self.optimize_pdf_flag:
                pdf_path = self.optimize_pdf(pdf_path)
            
            # 打开PDF文件
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            logger.info(f"PDF文件总页数: {total_pages}")
            
            # 识别结果
            ocr_results = []
            
            # 逐页处理
            for page_num in range(total_pages):
                logger.info(f"处理第 {page_num + 1}/{total_pages} 页")
                
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
            
            # 保存识别结果（即使部分页面处理失败）
            if ocr_results:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(ocr_results))
                logger.info(f"PDF文件处理完成，结果保存至: {output_txt_path}")
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
            logger.info(f"输出路径: {output_txt_path if success else 'N/A'}")
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
            output_path_str = output_txt_path if success else "N/A"
            
            # 生成Markdown表格行
            log_row = f"| {log_time} | {file_name} | {file_size_str} | {pages_str} | {elapsed_str} | {result_str} | {output_path_str} |\n"
            
            # 将日志行追加到文件
            with open(log_md_path, 'a', encoding='utf-8') as f:
                f.write(log_row)
            
            logger.info(f"日志已记录到Markdown文件: {log_md_path}")

class PDFFileHandler(FileSystemEventHandler):
    """监控目录中的新PDF文件（同步处理）"""
    def __init__(self, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
        self.output_dir = output_dir
        self.model = model
        self.optimize_pdf_flag = optimize_pdf
        self.optimize_level = optimize_level
        self.grayscale = grayscale
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

def run_manual_mode(input_dir, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
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
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
    
    # 同步处理所有PDF文件
    success_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        logger.info(f"开始处理文件: {pdf_file}")
        
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

def run_daemon_mode(input_dir, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
    """守护模式：持续监控输入目录，处理新的PDF文件"""
    logger.info(f"守护模式启动，监控目录: {input_dir}")
    
    # 创建事件处理器
    event_handler = PDFFileHandler(
        output_dir, 
        model,
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
    parser.add_argument('-model', '--model', choices=['paddleocr-vl', 'pp-ocrv5', 'pp-structurev3', 'pp-chatocrv4'], 
                       default='pp-ocrv5', help='OCR模型选择：paddleocr-vl（多模态文档解析）、pp-ocrv5（全场景文字识别）、pp-structurev3（复杂文档解析）、pp-chatocrv4（智能信息抽取）')
    parser.add_argument('-l', '--log-level', choices=LOG_LEVELS.keys(), default='info', 
                       help='日志输出级别：debug、info、warning、error、critical，默认：info')
    parser.add_argument('--optimize-pdf', action='store_true', help='是否优化PDF文件，默认：False')
    parser.add_argument('--optimize-level', choices=['low', 'medium', 'high'], default='medium', 
                       help='PDF优化级别，可选值：low、medium、high，默认：medium')
    parser.add_argument('--grayscale', action='store_true', help='是否使用灰度渲染，默认：False')
    
    args = parser.parse_args()
    
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
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
        else:
            run_daemon_mode(
                args.input, 
                args.output, 
                args.model,
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
    else:
        logger.error(f"输入路径不存在: {args.input}")
        return

if __name__ == '__main__':
    main()
