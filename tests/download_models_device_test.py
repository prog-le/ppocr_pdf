"""tests/download_models_device_test.py"""
"""
download_models.py 设备检测测试
运行: python -m pytest tests/download_models_device_test.py -v
说明: 验证下载模型时固定使用 CPU，以及 CUDA 检测提示
"""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDownloadModelsDevice:
    """测试模型下载始终使用 CPU"""

    def test_download_cpu_device_passed_to_paddleocr(self):
        """PP-OCRv5 下载使用 device='cpu'"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_instance = MagicMock()
            mock_ocr.return_value = mock_instance

            result = download_model('pp-ocrv5')

            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
            assert result is True

    def test_download_cpu_device_passed_to_ppsv3(self):
        """PP-StructureV3 下载使用 device='cpu'"""
        from download_models import download_model

        mock_paddlex = MagicMock()

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PPStructureV3') as mock_ppsv3,
            patch('download_models.logger'),
            patch.dict('sys.modules', {'paddlex': mock_paddlex}),
        ):
            mock_instance = MagicMock()
            mock_ppsv3.return_value = mock_instance

            result = download_model('pp-structurev3')

            call_kwargs = mock_ppsv3.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
            assert result is True

    def test_download_cpu_device_passed_to_vl(self):
        """PaddleOCR-VL 下载使用 device='cpu'"""
        from download_models import download_model

        with (
            patch('download_models.setup_custom_cache'),
            patch('paddleocr.PaddleOCRVL') as mock_vl,
            patch('download_models.logger'),
        ):
            mock_instance = MagicMock()
            mock_vl.return_value = mock_instance

            result = download_model('paddleocr-vl')

            call_kwargs = mock_vl.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
            assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
