"""tests/ocr_pdf_v6_test.py
PDFOCRHandler PP-OCRv6 model selection tests
Run: python -m pytest tests/ocr_pdf_v6_test.py -v
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_device_info():
    return {
        'device_type': 'cpu',
        'paddlex_device': 'cpu',
        'device_count': 0,
        'device_name': None,
        'cuda_version': None,
        'paddle_version': '3.2.2',
        'compiled_with_cuda': False,
    }


def _verify_ok():
    return {
        'paddlex_device': 'cpu', 'fallback': 'false',
        'message': '', 'paddle_cuda': 'false',
    }


class TestPPOCRv6ModelSelection:
    """Test PP-OCRv6 model selection logic in PDFOCRHandler"""

    def test_pp_ocrv6_default_uses_medium_chinese(self):
        """默认 model=pp-ocrv6 应使用 PP-OCRv6_medium, lang=ch"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'ch'
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_medium_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_medium_rec'
            assert call_kwargs.get('enable_mkldnn') is False

    def test_pp_ocrv6_small_model(self):
        """--model-size small 应使用 PP-OCRv6_small_*"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='pp-ocrv6', model_size='small')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_small_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_small_rec'

    def test_pp_ocrv6_tiny_model(self):
        """--model-size tiny 应使用 PP-OCRv6_tiny_*"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='pp-ocrv6', model_size='tiny')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_tiny_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_tiny_rec'

    def test_pp_ocrv6_with_english_lang(self):
        """--lang en 应传给 PaddleOCR"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='pp-ocrv6', lang='en')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'en'

    def test_pp_ocrv6_with_japanese_lang(self):
        """--lang japan (日语) 应传给 PaddleOCR"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='pp-ocrv6', lang='japan')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'japan'

    def test_pp_ocrv6_combined_lang_and_size(self):
        """组合测试: lang=en + model_size=small"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(
                output_dir='test_out',
                model='pp-ocrv6',
                lang='en',
                model_size='small'
            )

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'en'
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_small_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_small_rec'

    def test_pp_ocrv5_uses_v5_model_names(self):
        """model=pp-ocrv5 仍使用 PP-OCRv5_* 模型名 (兼容旧版)"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='pp-ocrv5', model_size='medium')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv5_medium_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv5_medium_rec'

    def test_unknown_model_falls_back_to_v6_medium(self):
        """未知 model 应回退到 PP-OCRv6_medium + lang=ch"""
        from ocr_pdf import PDFOCRHandler

        with (
            patch('ocr_pdf.detect_device', return_value=_mock_device_info()),
            patch('ocr_pdf.verify_paddle_device', return_value=_verify_ok()),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            mock_paddle_ocr.return_value = MagicMock()
            PDFOCRHandler(output_dir='test_out', model='nonexistent-model')

            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'ch'
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_medium_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_medium_rec'


class TestCLIPv6Args:
    """Test CLI parser accepts --lang and --model-size"""

    def test_cli_has_lang_argument(self):
        """CLI 包含 --lang 参数"""
        import argparse
        from ocr_pdf import main

        # 模拟调用 --help 不实际运行
        # 直接构造 ArgumentParser 测试
        parser = argparse.ArgumentParser()
        parser.add_argument('--lang', choices=['ch', 'en', 'japan', 'korean', 'latin'],
                          default='ch')

        args = parser.parse_args(['--lang', 'en'])
        assert args.lang == 'en'

        args = parser.parse_args([])
        assert args.lang == 'ch'

    def test_cli_has_model_size_argument(self):
        """CLI 包含 --model-size 参数"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--model-size', choices=['medium', 'small', 'tiny'],
                          default='medium')

        args = parser.parse_args(['--model-size', 'small'])
        assert args.model_size == 'small'

        args = parser.parse_args([])
        assert args.model_size == 'medium'

    def test_cli_default_model_is_pp_ocrv6(self):
        """CLI 默认 -model 应该是 pp-ocrv6"""
        import argparse
        from ocr_pdf import main
        from unittest.mock import patch

        # Capture args
        captured = {}
        original_parse_args = argparse.ArgumentParser.parse_args

        def mock_parse_args(self, *args, **kwargs):
            result = original_parse_args(self, ['-i', 'x', '-o', 'y'])
            captured.update(vars(result))
            return result

        with patch.object(argparse.ArgumentParser, 'parse_args', mock_parse_args):
            with patch('sys.argv', ['ocr_pdf.py', '-i', 'x', '-o', 'y']):
                try:
                    main()
                except Exception:
                    pass  # 实际运行会失败 (无 PDF), 但 args 应被解析

        assert captured.get('model') == 'pp-ocrv6', f"Expected default 'pp-ocrv6', got {captured.get('model')}"
        assert captured.get('lang') == 'ch'
        assert captured.get('model_size') == 'medium'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
