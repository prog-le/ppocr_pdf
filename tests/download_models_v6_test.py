"""tests/download_models_v6_test.py
download_models.py PP-OCRv6 + lang + model_size tests
Run: python -m pytest tests/download_models_v6_test.py -v
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDownloadModelsV6:
    """Test download_models.py with PP-OCRv6 + lang + model_size"""

    def test_pp_ocrv6_download(self):
        """pp-ocrv6 下载使用 PP-OCRv6_medium_det/rec + lang=ch"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_ocr.return_value = MagicMock()
            result = download_model('pp-ocrv6')

            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
            assert call_kwargs.get('lang') == 'ch'
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_medium_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_medium_rec'
            assert call_kwargs.get('enable_mkldnn') is False
            assert result is True

    def test_pp_ocrv6_with_small_size(self):
        """pp-ocrv6 + model_size=small"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_ocr.return_value = MagicMock()
            result = download_model('pp-ocrv6', model_size='small')

            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv6_small_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv6_small_rec'

    def test_pp_ocrv6_with_lang(self):
        """pp-ocrv6 + lang=english"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_ocr.return_value = MagicMock()
            result = download_model('pp-ocrv6', lang='en')

            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('lang') == 'en'

    def test_pp_ocrv5_uses_v5_model_names(self):
        """pp-ocrv5 仍使用 PP-OCRv5_* 模型名 (兼容旧版)"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_ocr.return_value = MagicMock()
            result = download_model('pp-ocrv5')

            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('text_detection_model_name') == 'PP-OCRv5_medium_det'
            assert call_kwargs.get('text_recognition_model_name') == 'PP-OCRv5_medium_rec'

    def test_supported_models_includes_v6(self):
        """SUPPORTED_MODELS 列表应包含 pp-ocrv6"""
        from download_models import SUPPORTED_MODELS

        assert 'pp-ocrv6' in SUPPORTED_MODELS
        assert 'pp-ocrv5' in SUPPORTED_MODELS
        assert 'pp-structurev3' in SUPPORTED_MODELS
        assert 'paddleocr-vl' in SUPPORTED_MODELS

    def test_cli_has_lang_and_model_size(self):
        """CLI 解析器包含 --lang 和 --model-size"""
        import argparse
        from download_models import main
        from unittest.mock import patch

        captured = {}
        original_parse_args = argparse.ArgumentParser.parse_args

        def mock_parse_args(self, *args, **kwargs):
            # 模拟: download_models.py -m pp-ocrv6 --lang en --model-size small
            result = original_parse_args(
                self,
                ['-m', 'pp-ocrv6', '--lang', 'en', '--model-size', 'small']
            )
            captured.update(vars(result))
            # 返回 args, 让 main 继续运行 (但会在 download_model 阶段失败)
            raise SystemExit(0)

        with patch.object(argparse.ArgumentParser, 'parse_args', mock_parse_args):
            with patch('sys.argv', ['download_models.py', '-m', 'pp-ocrv6', '--lang', 'en', '--model-size', 'small']):
                try:
                    main()
                except SystemExit:
                    pass

        assert captured.get('lang') == 'en'
        assert captured.get('model_size') == 'small'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
