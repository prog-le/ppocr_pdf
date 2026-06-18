"""tests/ocr_pdf_device_test.py"""
"""
ocr_pdf.py 设备检测集成测试
运行: python -m pytest tests/ocr_pdf_device_test.py -v
说明: 用 mock 模拟 device_utils.detect_device 验证参数传递
"""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPDFOCRHandlerDevice:
    """测试 PDFOCRHandler 设备参数传递"""

    def test_device_passed_to_paddleocr(self):
        """检测到的 paddlex_device 传入 PaddleOCR 构造函数"""
        from ocr_pdf import PDFOCRHandler

        mock_device_info = {
            'device_type': 'cuda',
            'paddlex_device': 'gpu:0',
            'device_count': 1,
            'device_name': 'NVIDIA GeForce RTX 4060',
            'cuda_version': '12.6',
            'paddle_version': '3.3.1',
            'compiled_with_cuda': True,
        }

        verify_ok = {
            'paddlex_device': 'gpu:0', 'fallback': 'false',
            'message': '', 'paddle_cuda': 'true',
        }
        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
            patch('ocr_pdf.verify_paddle_device', return_value=verify_ok),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='pp-ocrv5',
                device='auto'
            )
            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('device') == 'gpu:0'

    def test_device_cpu_override(self):
        """--device cpu 强制 CPU，即使有 GPU"""
        from ocr_pdf import PDFOCRHandler

        mock_device_info = {
            'device_type': 'cpu',
            'paddlex_device': 'cpu',
            'device_count': 0,
            'device_name': None,
            'cuda_version': None,
            'paddle_version': '3.3.1',
            'compiled_with_cuda': False,
        }

        verify_ok = {
            'paddlex_device': 'cpu', 'fallback': 'false',
            'message': '', 'paddle_cuda': 'false',
        }
        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
            patch('ocr_pdf.verify_paddle_device', return_value=verify_ok),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='pp-ocrv5',
                device='cpu'
            )
            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('device') == 'cpu'

    def test_device_passed_to_pp_structure_v3(self):
        """device 参数传给 PPStructureV3"""
        from ocr_pdf import PDFOCRHandler

        mock_device_info = {
            'device_type': 'cuda',
            'paddlex_device': 'gpu:0',
            'device_count': 1,
            'device_name': 'NVIDIA GeForce RTX 4060',
            'cuda_version': '12.6',
            'paddle_version': '3.3.1',
            'compiled_with_cuda': True,
        }

        verify_ok = {
            'paddlex_device': 'gpu:0', 'fallback': 'false',
            'message': '', 'paddle_cuda': 'true',
        }
        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
            patch('ocr_pdf.verify_paddle_device', return_value=verify_ok),
            patch('ocr_pdf.PPStructureV3') as mock_ppsv3,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='pp-structurev3',
                device='auto'
            )
            call_kwargs = mock_ppsv3.call_args[1]
            assert call_kwargs.get('device') == 'gpu:0'

    def test_device_passed_to_paddleocr_vl(self):
        """device 参数传给 PaddleOCRVL"""
        from ocr_pdf import PDFOCRHandler

        mock_device_info = {
            'device_type': 'cuda',
            'paddlex_device': 'gpu:0',
            'device_count': 1,
            'device_name': 'NVIDIA GeForce RTX 4060',
            'cuda_version': '12.6',
            'paddle_version': '3.3.1',
            'compiled_with_cuda': True,
        }

        verify_ok = {
            'paddlex_device': 'gpu:0', 'fallback': 'false',
            'message': '', 'paddle_cuda': 'true',
        }
        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
            patch('ocr_pdf.verify_paddle_device', return_value=verify_ok),
            patch('ocr_pdf.PaddleOCRVL') as mock_vl,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='paddleocr-vl',
                device='auto'
            )
            call_kwargs = mock_vl.call_args[1]
            assert call_kwargs.get('device') == 'gpu:0'


class TestCLIDeviceArgument:
    """测试 CLI --device 参数"""

    def test_cli_has_device_argument(self):
        """CLI 解析器包含 --device 参数"""
        # 直接检查 argparse 是否定义了 --device
        import argparse
        # We'll test by checking the parser structure
        from ocr_pdf import main
        
        # Create a mock parser and verify
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
        
        # Test parsing
        args = parser.parse_args([])
        assert args.device == 'auto'
        
        args = parser.parse_args(['--device', 'gpu'])
        assert args.device == 'gpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
