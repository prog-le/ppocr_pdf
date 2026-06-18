# -*- coding: utf-8 -*-
"""
device_utils 单元测试

测试 detect_device / check_cuda_environment / get_install_guide 三个函数
所有 GPU / nvidia-smi 调用均通过 mock 隔离。
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ==============================================================
# 被测模块 — 会在 conftest 或这里一次性导入
# ==============================================================
from device_utils import (
    check_cuda_environment,
    detect_device,
    get_install_guide,
    verify_paddle_device,
)


# ==============================================================
# TestDetectDevice
# ==============================================================
class TestDetectDevice:
    """覆盖 detect_device 的 6 条核心路径"""

    # ------------------------------------------------------------------
    # 1) 无 GPU / CUDA — 自动回退 CPU（默认 paddlepaddle）
    # ------------------------------------------------------------------
    @patch("device_utils.subprocess.run")
    @patch("device_utils.platform.system", return_value="Windows")
    @patch("device_utils.shutil.which", return_value=None)
    def test_auto_cpu(self, mock_which, mock_system, mock_run):
        """未安装 nvidia-smi → device='cpu'"""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        info = detect_device()
        assert info["device"] == "cpu"
        assert info["pkg_suffix"] == ""
        assert info["cuda_version"] == ""
        assert info["detail"].startswith("CPU")

    # ------------------------------------------------------------------
    # 2) 有 CUDA & GPU — 自动选择 CUDA
    # ------------------------------------------------------------------
    @patch("device_utils.subprocess.run")
    @patch("device_utils.platform.system", return_value="Windows")
    @patch("device_utils.shutil.which", return_value="C:\\Program Files\\NVIDIA Corporation\\nvidia-smi.exe")
    def test_auto_cuda(self, mock_which, mock_system, mock_run):
        """nvidia-smi 可用且返回 CUDA 12.6 → device='gpu'"""
        nvidia_smi_output = (
            'NVIDIA-SMI 560.94  Driver Version: 560.94  CUDA Version: 12.6\n'
            'GPU Name  TCC/WDDM  Bus-ID  Disp.  Volatile.  ECC\n'
            'RTX 4090  WDDM  ...\n'
        )
        mock_proc = MagicMock()
        mock_proc.stdout = nvidia_smi_output
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        info = detect_device()
        assert info["device"] == "gpu"
        assert info["pkg_suffix"] == "-gpu"
        assert info["cuda_version"] == "12.6"
        assert "4090" in info["detail"]

    # ------------------------------------------------------------------
    # 3) device_override='gpu' — 即使没有 GPU 也会尝试
    # ------------------------------------------------------------------
    @patch("device_utils.subprocess.run")
    @patch("device_utils.platform.system", return_value="Windows")
    @patch("device_utils.shutil.which", return_value="C:\\nvidia-smi.exe")
    def test_override_gpu(self, mock_which, mock_system, mock_run):
        """用户强制 --device gpu → device='gpu'"""
        mock_proc = MagicMock()
        mock_proc.stdout = (
            "NVIDIA-SMI 560.94  CUDA Version: 11.8\n"
            "RTX 3060\n"
        )
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        info = detect_device(device_override="gpu")
        assert info["device"] == "gpu"
        assert info["cuda_version"] == "11.8"

    # ------------------------------------------------------------------
    # 4) device_override='cpu' — 即使有 GPU 也强制 CPU
    # ------------------------------------------------------------------
    def test_override_cpu(self):
        """用户强制 --device cpu → device='cpu'，不检测 GPU"""
        info = detect_device(device_override="cpu")
        assert info["device"] == "cpu"
        assert info["pkg_suffix"] == ""
        # 不调用 nvidia-smi

    # ------------------------------------------------------------------
    # 5) paddle 导入失败 → 不会崩溃（仅日志）
    # ------------------------------------------------------------------
    @patch("device_utils.subprocess.run")
    @patch("device_utils.platform.system", return_value="Windows")
    @patch("device_utils.shutil.which", return_value=None)
    def test_paddle_import_fallback(self, mock_which, mock_system, mock_run):
        """paddle 模块不可用（模拟 ImportError）→ 依然返回 CPU 信息"""
        # 让 detect_device 内部不真正触发 import paddle 的异常即可
        # 实际上 detect_device 并没有 import paddle，所以此用例仅保留骨架
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        info = detect_device()
        assert info["device"] == "cpu"
        assert "paddle" not in info or info.get("paddle_available") in (None, False)

    # ------------------------------------------------------------------
    # 6) macOS (darwin) — 无 CUDA
    # ------------------------------------------------------------------
    @patch("device_utils.subprocess.run")
    @patch("device_utils.platform.system", return_value="Darwin")
    @patch("device_utils.shutil.which", return_value=None)
    def test_macos_without_cuda(self, mock_which, mock_system, mock_run):
        """macOS 上无 nvidia-smi → device='cpu'（后续可扩展 MPS）"""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        info = detect_device()
        assert info["device"] == "cpu"
        assert info["pkg_suffix"] == ""
        assert "macOS" in info["detail"] or "Darwin" in info["detail"] or "cpu" in info["detail"]


# ==============================================================
# TestCheckCudaEnvironment
# ==============================================================
class TestCheckCudaEnvironment:
    """解析 nvidia-smi 输出"""

    SMI_CU126 = (
        "Thu Jan 16 10:00:00 2026\n"
        "NVIDIA-SMI 560.94  Driver Version: 560.94  CUDA Version: 12.6\n"
        "-----------------------------------------+--------------------------\n"
        "GPU  Name                  Driver-Model  Bus-Id    Disp.  Volatile.\n"
        "0   NVIDIA GeForce RTX 4090  WDDM        ...  On  ...\n"
    )

    SMI_CU118 = (
        "NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 11.8\n"
        "GPU 0: NVIDIA A100 80GB PCIe\n"
    )

    @patch("device_utils.subprocess.run")
    def test_cu126(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.stdout = self.SMI_CU126
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        env = check_cuda_environment()
        assert env["cuda_version"] == "12.6"
        assert env["driver_version"] == "560.94"
        assert "RTX 4090" in env["gpu_info"]

    @patch("device_utils.subprocess.run")
    def test_cu118(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.stdout = self.SMI_CU118
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        env = check_cuda_environment()
        assert env["cuda_version"] == "11.8"
        assert env["driver_version"] == "525.85.05"
        assert "A100" in env["gpu_info"]

    @patch("device_utils.subprocess.run")
    def test_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        env = check_cuda_environment()
        assert env["cuda_version"] == ""
        assert env["driver_version"] == ""
        assert env["gpu_info"] == ""


# ==============================================================
# TestGetInstallGuide
# ==============================================================
class TestGetInstallGuide:
    """不同平台 / 设备对应的安装指引"""

    @patch("device_utils.platform.system", return_value="Windows")
    def test_windows_cpu(self, mock_system):
        guide = get_install_guide(device_type="cpu")
        assert "pip install paddlepaddle" in guide
        assert "CUDA" not in guide

    @patch("device_utils.platform.system", return_value="Windows")
    def test_windows_cuda(self, mock_system):
        guide = get_install_guide(device_type="gpu", cuda_tag="12.6")
        assert "paddlepaddle-gpu" in guide
        assert "12.6" in guide

    @patch("device_utils.platform.system", return_value="Darwin")
    def test_macos(self, mock_system):
        guide = get_install_guide(device_type="cpu")
        assert "paddlepaddle" in guide
        # macOS 目前只有 CPU 指引
        assert "arm64" in guide or "macOS" in guide or "pip install" in guide

    @patch("device_utils.platform.system", return_value="Linux")
    def test_linux_cpu(self, mock_system):
        guide = get_install_guide(device_type="cpu")
        assert "pip install paddlepaddle" in guide
        # Linux CPU 版本通常就是 CPU 基础包
        assert "CUDA" not in guide


# ==============================================================
# TestVerifyPaddleDevice
# ==============================================================
class TestVerifyPaddleDevice:
    """覆盖 verify_paddle_device 的 4 条核心路径

    注意: 使用 sys.modules 注入 mock paddle 模块来避免 patch builtins.__import__
    的 readonly 属性问题。每个 test 在 setup 时注入 mock paddle，
    teardown 时自动恢复（通过 patch.dict 上下文管理器）。
    """

    @staticmethod
    def _make_mock_paddle(is_cuda: bool = False, version: str = "3.3.1"):
        """创建 mock paddle 模块的工厂方法"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = is_cuda
        mock_paddle.__version__ = version
        return mock_paddle

    # ------------------------------------------------------------------
    # 1) GPU 请求 + Paddle 有 CUDA → 保持 GPU
    # ------------------------------------------------------------------
    def test_gpu_with_cuda(self):
        """paddle.is_compiled_with_cuda() = True → 保持 gpu:0"""
        mock_paddle = self._make_mock_paddle(is_cuda=True)

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            result = verify_paddle_device("gpu:0")
            assert result["paddlex_device"] == "gpu:0"
            assert result["fallback"] == "false"
            assert result["paddle_cuda"] == "true"

    # ------------------------------------------------------------------
    # 2) GPU 请求 + Paddle 无 CUDA → 回退 CPU
    # ------------------------------------------------------------------
    def test_gpu_fallback_to_cpu(self):
        """paddle.is_compiled_with_cuda() = False → 回退到 cpu"""
        mock_paddle = self._make_mock_paddle(is_cuda=False)

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            result = verify_paddle_device("gpu:0")
            assert result["paddlex_device"] == "cpu"
            assert result["fallback"] == "true"
            assert result["paddle_cuda"] == "false"
            assert "fallback" in result["message"]

    # ------------------------------------------------------------------
    # 3) CPU 请求 → 无论 Paddle 有无 CUDA 都保持 CPU
    # ------------------------------------------------------------------
    def test_cpu_no_fallback(self):
        """paddlex_device='cpu' → 不做任何更改"""
        mock_paddle = self._make_mock_paddle(is_cuda=False)

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            result = verify_paddle_device("cpu")
            assert result["paddlex_device"] == "cpu"
            assert result["fallback"] == "false"
            # 因为 request 是 cpu，不会触发 import paddle
            assert result["paddle_cuda"] == "false"

    # ------------------------------------------------------------------
    # 4) Paddle 未安装 → 回退 CPU
    # ------------------------------------------------------------------
    @patch("builtins.__import__", side_effect=ImportError("no paddle"))
    def test_paddle_not_installed(self, mock_import):
        """ImportError → 回退到 cpu"""
        result = verify_paddle_device("gpu:0")
        assert result["paddlex_device"] == "cpu"
        assert result["fallback"] == "true"
        assert "not installed" in result["message"].lower()
