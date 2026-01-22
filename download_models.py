#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# 模块信息
# ----------------------------------------------------------------------
# @Author  : Prog.le
# @Email   : Prog.le@outlook.com
# @Time    : 2026-01-19
# @FileName: download_models.py
# @Software: TRAE CN
# @Version : 1.0.0
# ----------------------------------------------------------------------
# 功能描述
# ----------------------------------------------------------------------
# 本脚本用于下载PaddleOCR模型到指定文件夹
# 支持下载的模型：
#   - pp-ocrv5: PP-OCRv5模型
#   - pp-structurev3: PP-StructureV3模型
#   - paddleocr-vl: PaddleOCR-VL模型
# ----------------------------------------------------------------------
# 使用示例
# ----------------------------------------------------------------------
# 下载所有模型到默认目录(.paddlex)
#   python download_models.py
# 
# 下载指定模型到自定义目录
#   python download_models.py -m pp-ocrv5,paddleocr-vl -o ./models
# ----------------------------------------------------------------------

import os
import argparse
import logging
import sys

# 配置日志级别映射
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 初始默认值，后续会根据命令行参数更新
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 支持的模型列表
SUPPORTED_MODELS = ['pp-ocrv5', 'pp-structurev3', 'paddleocr-vl']

# 创建自定义的paddlex.utils.cache模块
class CustomCacheModule:
    def __init__(self, cache_dir=None):
        # 设置自定义的目录路径
        if cache_dir:
            self.paddlex_dir = os.path.abspath(cache_dir)
        else:
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

# 设置自定义缓存模块
def setup_custom_cache(cache_dir=None):
    # 创建模块实例
    custom_cache_module = CustomCacheModule(cache_dir)
    
    # 将自定义模块注入到sys.modules中
    sys.modules['paddlex.utils.cache'] = custom_cache_module
    logger.info(f"已设置自定义缓存目录: {custom_cache_module.CACHE_DIR}")


def download_model(model_name):
    """
    下载指定的PaddleOCR模型
    
    Args:
        model_name (str): 模型名称
    """
    logger.info(f"开始下载 {model_name} 模型...")
    
    try:
        # 动态导入模型类，确保在设置好缓存目录后再导入
        from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL
        
        if model_name == 'pp-ocrv5':
            # 下载PP-OCRv5模型
            ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model_name == 'pp-structurev3':
            # 下载PP-StructureV3模型
            # 检查pp-structurev3所需的依赖
            try:
                import paddlex
            except ImportError:
                logger.error(f"下载 {model_name} 模型需要安装额外依赖: 'pip install \"paddlex[ocr]\"'")
                return False
            
            ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model_name == 'paddleocr-vl':
            # 下载PaddleOCR-VL模型
            ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        else:
            logger.error(f"不支持的模型: {model_name}")
            return False
        
        logger.info(f"{model_name} 模型下载完成!")
        return True
    
    except Exception as e:
        logger.error(f"下载 {model_name} 模型时出错: {str(e)}")
        import traceback
        logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
        return False

def main():
    parser = argparse.ArgumentParser(description='下载PaddleOCR模型到指定文件夹')
    parser.add_argument(
        '-m', '--models',
        type=str,
        default='all',
        help=f"要下载的模型，多个模型用逗号分隔，可选值: {', '.join(SUPPORTED_MODELS)}, all (下载所有模型)，默认: all"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='模型保存目录，默认使用PaddleOCR默认缓存目录'
    )
    parser.add_argument(
        '-l', '--log-level',
        type=str,
        choices=LOG_LEVELS.keys(),
        default='info',
        help=f"日志输出级别，可选值: {', '.join(LOG_LEVELS.keys())}，默认: info"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = LOG_LEVELS[args.log_level]
    logging.getLogger().setLevel(log_level)
    # 同时设置paddleocr相关日志器的级别
    for logger_name in ['paddleocr', 'paddle', 'ppocr', 'paddlex']:
        logging.getLogger(logger_name).setLevel(log_level)
    logger.info(f"日志级别已设置为：{args.log_level}")
    
    # 确定要下载的模型列表
    if args.models == 'all':
        models_to_download = SUPPORTED_MODELS.copy()
    else:
        models_to_download = [m.strip() for m in args.models.split(',')]
        # 验证模型名称
        for model in models_to_download:
            if model not in SUPPORTED_MODELS:
                logger.error(f"不支持的模型: {model}")
                logger.info(f"支持的模型: {', '.join(SUPPORTED_MODELS)}")
                return
    
    # 确定模型保存目录
    cache_dir = args.output
    if cache_dir:
        logger.info(f"模型将保存到目录: {os.path.abspath(cache_dir)}")
    else:
        logger.info(f"使用默认缓存目录")
    
    # 设置自定义缓存目录
    setup_custom_cache(cache_dir)
    
    # 下载模型
    success_count = 0
    for model in models_to_download:
        if download_model(model):
            success_count += 1
    
    logger.info(f"\n下载完成: {success_count}/{len(models_to_download)} 个模型下载成功")
    
    if cache_dir:
        logger.info(f"所有模型已保存到: {os.path.abspath(cache_dir)}")
    else:
        logger.info(f"模型已保存到默认缓存目录")

if __name__ == '__main__':
    main()
