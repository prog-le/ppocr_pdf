"""tests/api_device_test.py"""
"""
api.py 设备参数测试
运行: python -m pytest tests/api_device_test.py -v
说明: 验证 API 接口传递 device 参数
"""
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _clear_handler_cache():
    """每个测试前清除 handler 缓存，防止 mock 实例跨测试污染"""
    from api import clear_handler_cache
    clear_handler_cache()


class TestAPIDevice:
    """测试 API device 参数传递"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from api import app
        return TestClient(app)

    def test_health_check_includes_device_info(self, client):
        """/health 返回包含 device_modes 信息"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "device_modes" in data
        assert "auto" in data["device_modes"]
        assert "gpu" in data["device_modes"]
        assert "cpu" in data["device_modes"]
        assert data["default_device"] == "auto"

    def _make_ocr_request(self, client, tmp_path, data=None):
        """辅助：发送 OCR 请求并返回 response"""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_text("%PDF-1.4 fake pdf content")

        with (
            patch('api.PDFOCRHandler') as mock_handler,
            patch('api.detect_device') as mock_detect,
        ):
            mock_handler_instance = MagicMock()
            mock_handler_instance.process_pdf.return_value = True
            mock_handler.return_value = mock_handler_instance

            mock_detect.return_value = {
                "device": "cpu", "device_type": "cpu", "paddlex_device": "cpu",
                "detail": "CPU (mock)", "cuda_version": "", "gpu_info": "",
                "source": "auto"
            }

            fake_output = tmp_path / "output" / "test.txt"
            fake_output.parent.mkdir(parents=True, exist_ok=True)
            fake_output.write_text("OCR result")

            with patch('api.os.path.exists', return_value=True):
                with patch('api.open', MagicMock()) as mock_open:
                    mock_file = MagicMock()
                    mock_file.__enter__.return_value.read.return_value = "OCR result"
                    mock_open.return_value = mock_file

                    req_data = data or {"model": "paddleocr-vl"}
                    return client.post(
                        "/ocr/pdf",
                        files={"file": ("test.pdf", fake_pdf.read_bytes(), "application/pdf")},
                        data=req_data
                    ), mock_handler

    def test_device_param_accepted(self, client, tmp_path):
        """device 表单参数被正确接收"""
        response, mock_handler = self._make_ocr_request(
            client, tmp_path, {"model": "paddleocr-vl", "device": "gpu"}
        )
        assert response.status_code == 200
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs.get('device') == 'gpu'

    def test_device_default_is_auto(self, client, tmp_path):
        """device 默认值为 auto"""
        response, mock_handler = self._make_ocr_request(
            client, tmp_path, {"model": "paddleocr-vl"}
        )
        assert response.status_code == 200
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs.get('device') == 'auto'

    def test_device_cpu_override(self, client, tmp_path):
        """device=cpu 强制 CPU"""
        response, mock_handler = self._make_ocr_request(
            client, tmp_path, {"model": "paddleocr-vl", "device": "cpu"}
        )
        assert response.status_code == 200
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs.get('device') == 'cpu'

    def test_device_invalid_rejected(self, client, tmp_path):
        """非法的 device 值应返回 400"""
        response, _ = self._make_ocr_request(
            client, tmp_path, {"model": "paddleocr-vl", "device": "mps"}
        )
        assert response.status_code == 400
        assert "设备参数错误" in response.json()["detail"]

    def test_response_includes_device_and_device_info(self, client, tmp_path):
        """OCR 响应中包含 device 和 device_info"""
        response, _ = self._make_ocr_request(
            client, tmp_path, {"model": "paddleocr-vl", "device": "auto"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert data["device"] == "auto"
        assert "device_info" in data
        assert "device" in data["device_info"]
        assert "detail" in data["device_info"]

    def test_device_endpoint(self, client):
        """/device 返回设备检测信息"""
        with patch('api.detect_device') as mock_detect:
            mock_detect.return_value = {
                "device": "cpu", "device_type": "cpu", "paddlex_device": "cpu",
                "cuda_version": "", "gpu_info": "", "detail": "CPU (mock)",
                "source": "auto", "pkg_suffix": ""
            }
            response = client.get("/device")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "device" in data
            assert "supported_devices" in data
            assert data["device"] == "cpu"

    def test_valid_models_excludes_chatocrv4_and_ppocrv5(self, client):
        """验证 api.valid_models 只含 3 模型，不含 pp-chatocrv4 和 pp-ocrv5"""
        from api import valid_models
        expected = ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]
        assert valid_models == expected, f"Expected {expected}, got {valid_models}"

    def test_chatocr_patch_file_removed(self):
        """验证 chatocr_patch.py 已被删除（不在仓库）"""
        assert not os.path.exists('chatocr_patch.py'), \
            "chatocr_patch.py 应该被删除"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
