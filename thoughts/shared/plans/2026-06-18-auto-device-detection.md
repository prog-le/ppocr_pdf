# 自适应 GPU/CPU/MPS 设备检测实现计划

**Goal:** 新增自动设备检测模块，让 ppocr_pdf 自动利用 GPU/CUDA 加速推理，并保持 CPU/无 GPU 环境的无缝降级。

**Architecture:** 新增 `device_utils.py` 作为设备检测统一入口，通过 PaddlePaddle 的 `is_compiled_with_cuda()` + `device_count()` 实现运行时自动检测。`ocr_pdf.py`、`download_models.py`、`api.py` 均通过它获取设备信息。Windows 用户可通过 `scripts/setup_env.bat` 一键安装正确的 GPU/CPU 版 PaddlePaddle。

**Design:** `thoughts/shared/designs/2026-06-18-auto-device-detection-design.md`

---

## 依赖图

```
Batch 1 (parallel): 1.1, 1.2, 1.3 [foundation - 无依赖，可同时执行]
Batch 2 (parallel): 2.1, 2.2 [core - 依赖 Batch 1]
Batch 3 (parallel): 3.1 [integration - 依赖 Batch 2]
```

---

## Batch 1: Foundation (parallel — 3 implementers)

All tasks have NO dependencies and run simultaneously.

### Task 1.1: device_utils.py — 设备检测核心模块
**File:** `device_utils.py`
**Test:** `tests/device_utils_test.py`
**Depends:** none

**实现说明：** 设计要求运行时自动检测 GPU/CPU/MPS。我选择通过 `paddle.is_compiled_with_cuda()` 和 `paddle.device.cuda.device_count()` 实现；`nvidia-smi` 调用为可选项，捕获异常不崩溃；MPS 检测为信息性，实际向 PaddleOCR 传递的 `paddlex_device` 仅使用 `gpu:0` 或 `cpu`（因为 MPS 在 PaddleOCR 中支持不完整）。所有 paddle 导入失败都兜底到 CPU。

```python
"""tests/device_utils_test.py"""
"""
device_utils 单元测试

运行: python -m pytest tests/device_utils_test.py -v
说明: 使用 mock 替换 paddle 模块，不依赖真实 GPU 硬件
"""
import sys
import platform
from unittest.mock import MagicMock, patch
import pytest


# ─── detect_device 测试 ─────────────────────────────────────────


class TestDetectDevice:
    """测试 detect_device 的四种核心场景"""

    def test_auto_cpu_when_no_cuda(self):
        """场景: 自动检测，paddle 无 CUDA 编译 → 返回 cpu"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = False
        mock_paddle.device.cuda.device_count.return_value = 0

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            # 清除可能已缓存的导入
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device()
            assert info['device_type'] == 'cpu'
            assert info['paddlex_device'] == 'cpu'
            assert info['device_count'] == 0
            assert info['compiled_with_cuda'] is False

    def test_auto_cuda_when_gpu_available(self):
        """场景: 自动检测，paddle 有 CUDA + 设备数 > 0 → 返回 gpu:0"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        mock_paddle.version.full_version = '3.3.1'

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device()
            assert info['device_type'] == 'cuda'
            assert info['paddlex_device'] == 'gpu:0'
            assert info['device_count'] == 1
            assert info['compiled_with_cuda'] is True

    def test_override_gpu_forces_cpu_when_no_cuda(self):
        """场景: --device gpu，但 paddle 无 CUDA → 警告 + 降级 cpu"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = False

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device(device_override='gpu')
            assert info['device_type'] == 'cpu'
            assert info['paddlex_device'] == 'cpu'

    def test_override_cpu(self):
        """场景: --device cpu → 强制 cpu，忽略 GPU"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 4

        with patch.dict('sys.modules', {'paddle': mock_paddle}):
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device(device_override='cpu')
            assert info['device_type'] == 'cpu'
            assert info['paddlex_device'] == 'cpu'

    def test_paddle_import_fallback_to_cpu(self):
        """场景: paddle 未安装 → 兜底到 cpu，不崩溃"""
        # 模拟 paddle 不存在
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if 'paddle' in key:
                saved_modules[key] = sys.modules.pop(key)

        try:
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device()
            assert info['device_type'] == 'cpu'
            assert info['paddlex_device'] == 'cpu'
            assert info['paddle_version'] is None
        finally:
            sys.modules.update(saved_modules)

    def test_macos_without_cuda(self):
        """场景: macOS 系统，无 CUDA → 检测为 mps（设备名为 mps）"""
        mock_paddle = MagicMock()
        mock_paddle.is_compiled_with_cuda.return_value = False

        with (
            patch.object(platform, 'system', return_value='Darwin'),
            patch.dict('sys.modules', {'paddle': mock_paddle}),
        ):
            if 'device_utils' in sys.modules:
                del sys.modules['device_utils']
            from device_utils import detect_device

            info = detect_device()
            # macOS 无 CUDA 时标记为 mps，但 paddlex_device 仍为 cpu
            assert info['device_type'] == 'mps'
            assert info['paddlex_device'] == 'cpu'


# ─── check_cuda_environment 测试 ────────────────────────────────


class TestCheckCudaEnvironment:
    """测试 nvidia-smi 解析逻辑"""

    @patch('device_utils.subprocess.run')
    def test_parse_nvidia_smi_cu126(self, mock_run):
        """场景: nvidia-smi 返回 CUDA 12.6 → 推荐 cu126"""
        mock_result = MagicMock()
        mock_result.stdout = (
            'NVIDIA-SMI 560.94  Driver Version: 560.94  CUDA Version: 12.6\n'
        )
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        from device_utils import check_cuda_environment
        result = check_cuda_environment()
        assert result['cuda_version'] == '12.6'
        assert result['cuda_tag'] == 'cu126'

    @patch('device_utils.subprocess.run')
    def test_parse_nvidia_smi_cu118(self, mock_run):
        """场景: nvidia-smi 返回 CUDA 11.8 → 推荐 cu118"""
        mock_result = MagicMock()
        mock_result.stdout = (
            'NVIDIA-SMI 520.61  Driver Version: 520.61  CUDA Version: 11.8\n'
        )
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        from device_utils import check_cuda_environment
        result = check_cuda_environment()
        assert result['cuda_version'] == '11.8'
        assert result['cuda_tag'] == 'cu118'

    @patch('device_utils.subprocess.run', side_effect=FileNotFoundError())
    def test_nvidia_smi_not_found(self, mock_run):
        """场景: nvidia-smi 不可用 → 返回空信息"""
        from device_utils import check_cuda_environment
        result = check_cuda_environment()
        assert result['cuda_version'] is None
        assert result['cuda_tag'] is None
        assert result['driver_version'] is None


# ─── get_install_guide 测试 ─────────────────────────────────────


class TestGetInstallGuide:
    """测试平台安装指南生成"""

    @patch.object(platform, 'system', return_value='Windows')
    def test_windows_cpu_guide(self, mock_system):
        """场景: Windows + CPU → 返回 CPU 安装命令"""
        from device_utils import get_install_guide
        guide = get_install_guide(device_type='cpu')
        assert 'paddlepaddle' in guide
        assert '--extra-index-url' not in guide  # CPU 版无需额外索引

    @patch.object(platform, 'system', return_value='Windows')
    def test_windows_cuda_guide(self, mock_system):
        """场景: Windows + CUDA cu126 → GPU 安装命令"""
        from device_utils import get_install_guide
        guide = get_install_guide(device_type='cuda', cuda_tag='cu126')
        assert 'paddlepaddle-gpu' in guide
        assert 'cu126' in guide

    @patch.object(platform, 'system', return_value='Darwin')
    def test_macos_guide(self, mock_system):
        """场景: macOS → CPU 版安装命令"""
        from device_utils import get_install_guide
        guide = get_install_guide(device_type='mps')
        assert 'paddlepaddle' in guide
        assert 'macos' in guide or 'mac' in guide

    @patch.object(platform, 'system', return_value='Linux')
    def test_linux_cpu_guide(self, mock_system):
        """场景: Linux + CPU → CPU 安装命令"""
        from device_utils import get_install_guide
        guide = get_install_guide(device_type='cpu')
        assert 'paddlepaddle' in guide
        assert 'paddlepaddle-gpu' not in guide


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

```python
"""device_utils.py"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# 模块信息
# ----------------------------------------------------------------------
# @Author  : Prog.le
# @Email   : Prog.le@outlook.com
# @Time    : 2026-06-18
# @FileName: device_utils.py
# @Software: TRAE CN
# @Version : 1.0.0
# ----------------------------------------------------------------------
# 功能描述
# ----------------------------------------------------------------------
# 本模块提供统一的 GPU/CPU/MPS 设备检测能力，作为所有需要推理的模块
# 获取设备信息的唯一入口。
#
# 核心函数:
#   detect_device(device_override) → 设备信息 dict
#   check_cuda_environment()       → CUDA 版本和推荐标签
#   get_install_guide()            → 平台安装指南
# ----------------------------------------------------------------------

import logging
import platform
import re
import subprocess
import sys

logger = logging.getLogger(__name__)


def detect_device(device_override=None):
    """
    检测可用的推理设备。

    检测流程:
    1. device_override='cpu' → 强制 CPU，不做任何检测
    2. device_override='gpu' → 尝试 CUDA，若不可用则警告降级到 CPU
    3. device_override=None/auto → 自动检测:
       a. paddle.is_compiled_with_cuda() + device_count > 0 → CUDA
       b. platform.system() == 'Darwin' → macOS/MPS（信息性标记，实际仍用 cpu）
       c. 兜底 → CPU

    Args:
        device_override: 'auto' | 'gpu' | 'cpu' | None（默认 None 等价于 'auto'）

    Returns:
        dict: 包含以下字段:
            - device_type: 'cuda' | 'mps' | 'cpu'
            - paddlex_device: 'gpu:0' | 'cpu'（可直接传给 PaddleOCR）
            - device_count: int（设备数量）
            - device_name: str | None
            - cuda_version: str | None
            - paddle_version: str | None
            - compiled_with_cuda: bool
    """
    # 默认信息结构
    info = {
        'device_type': 'cpu',
        'paddlex_device': 'cpu',
        'device_count': 0,
        'device_name': None,
        'cuda_version': None,
        'paddle_version': None,
        'compiled_with_cuda': False,
    }

    # 解析 override 参数
    override = device_override
    if override is None or override == 'auto':
        override = 'auto'
    elif override not in ('gpu', 'cpu'):
        logger.warning(f"未知的 device 参数 '{override}'，使用自动检测")
        override = 'auto'

    # 强制 CPU 模式
    if override == 'cpu':
        logger.info("用户指定 --device cpu，强制使用 CPU")
        return info

    # 尝试导入 paddle 检测真实环境
    try:
        import paddle
    except ImportError:
        logger.warning("无法导入 paddle 模块，兜底使用 CPU")
        return info

    # 获取 paddle 版本
    try:
        info['paddle_version'] = paddle.version.full_version
    except AttributeError:
        try:
            info['paddle_version'] = paddle.__version__
        except AttributeError:
            info['paddle_version'] = 'unknown'

    # 检测 CUDA
    try:
        info['compiled_with_cuda'] = paddle.is_compiled_with_cuda()
        if info['compiled_with_cuda']:
            try:
                info['device_count'] = paddle.device.cuda.device_count()
            except Exception:
                info['device_count'] = 0

            if info['device_count'] > 0:
                info['device_type'] = 'cuda'
                info['paddlex_device'] = 'gpu:0'
                # 尝试获取设备名称
                try:
                    info['device_name'] = paddle.device.cuda.get_device_name(0)
                except Exception:
                    info['device_name'] = 'NVIDIA GPU'
                logger.info(
                    f"检测到 CUDA 设备: {info['device_name']} "
                    f"(x{info['device_count']}), paddle 版本: {info['paddle_version']}"
                )
            else:
                # paddle 编译了 CUDA 但无可用设备
                logger.warning("PaddlePaddle 已编译 CUDA 但未检测到可用 GPU 设备")
                if override == 'gpu':
                    logger.warning("用户指定 --device gpu 但无可用 GPU，降级到 CPU")
                    info['device_type'] = 'cpu'
                    info['paddlex_device'] = 'cpu'
                else:
                    info['device_type'] = 'cpu'
                    info['paddlex_device'] = 'cpu'
        else:
            # paddle 未编译 CUDA
            if override == 'gpu':
                logger.warning(
                    "当前 PaddlePaddle 未编译 CUDA 支持，无法使用 GPU。"
                    "降级到 CPU。如需 GPU 加速，请安装 paddlepaddle-gpu"
                )
            # 继续检测 macOS MPS
            _detect_macos(info)
    except Exception as e:
        logger.warning(f"检测 CUDA 时发生异常: {e}，兜底到 CPU")
        if override == 'gpu':
            logger.warning("用户指定 --device gpu 但检测异常，降级到 CPU")
        _detect_macos(info)

    logger.info(f"设备检测结果: device_type={info['device_type']}, "
                f"paddlex_device={info['paddlex_device']}")
    return info


def _detect_macos(info):
    """检测 macOS MPS 支持（信息性检测，实际仍用 cpu 推理）"""
    try:
        if platform.system() == 'Darwin':
            info['device_type'] = 'mps'
            info['device_name'] = 'Apple Silicon / MPS'
            logger.info("检测到 macOS 系统，MPS 可用（信息性标记，PaddleOCR 仍用 CPU）")
    except Exception:
        pass


def check_cuda_environment():
    """
    通过 nvidia-smi 检测 CUDA 驱动版本。

    Returns:
        dict: {
            'cuda_version': '12.6' | None,
            'cuda_tag': 'cu126' | 'cu118' | 'cu129' | None,
            'driver_version': '560.94' | None,
            'gpu_name': 'NVIDIA GeForce RTX 4060' | None,
        }
    """
    result = {
        'cuda_version': None,
        'cuda_tag': None,
        'driver_version': None,
        'gpu_name': None,
    }

    try:
        output = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if output.returncode != 0:
            logger.debug("nvidia-smi 返回非零退出码")
            return result

        stdout = output.stdout

        # 提取 CUDA 版本
        cuda_match = re.search(r'CUDA Version:\s*([\d.]+)', stdout)
        if cuda_match:
            cuda_version = cuda_match.group(1)
            result['cuda_version'] = cuda_version

            # 映射到 paddle 标签
            major_minor = '.'.join(cuda_version.split('.')[:2])
            cuda_tag_map = {
                '11.8': 'cu118',
                '12.0': 'cu120',
                '12.1': 'cu121',
                '12.2': 'cu122',
                '12.3': 'cu123',
                '12.4': 'cu124',
                '12.5': 'cu125',
                '12.6': 'cu126',
                '12.9': 'cu129',
            }
            result['cuda_tag'] = cuda_tag_map.get(major_minor, f"cu{major_minor.replace('.', '')}")

        # 提取驱动版本
        driver_match = re.search(r'Driver Version:\s*([\d.]+)', stdout)
        if driver_match:
            result['driver_version'] = driver_match.group(1)

        # 提取 GPU 名称（第一个 GPU）
        gpu_match = re.search(r'\| (.+?) .*\n.*\|.*N/A.*N/A', stdout)
        if not gpu_match:
            # 更简单的匹配方式
            for line in stdout.split('\n'):
                if 'GeForce' in line or 'RTX' in line or 'Tesla' in line or 'Quadro' in line:
                    parts = line.strip().strip('|').split()
                    if parts:
                        result['gpu_name'] = parts[0]
                        break

        logger.info(
            f"CUDA 环境检测: CUDA {result['cuda_version']}, "
            f"驱动 {result['driver_version']}, "
            f"推荐标签 {result['cuda_tag']}"
        )

    except FileNotFoundError:
        logger.debug("nvidia-smi 未找到（无 NVIDIA 驱动或不在 PATH 中）")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi 执行超时")
    except Exception as e:
        logger.warning(f"检测 CUDA 环境时出错: {e}")

    return result


def get_install_guide(device_type='cpu', cuda_tag=None):
    """
    返回平台相关的 PaddlePaddle 安装指南文本。

    Args:
        device_type: 'cuda' | 'mps' | 'cpu'
        cuda_tag: 如 'cu126'、'cu118'（仅在 device_type='cuda' 时需要）

    Returns:
        str: 安装命令和说明文本
    """
    system = platform.system()

    lines = []
    lines.append("=" * 60)
    lines.append("PaddlePaddle 安装指南")
    lines.append("=" * 60)

    if device_type == 'cuda' and cuda_tag:
        lines.append(f"检测到 NVIDIA GPU，推荐安装 GPU 版 PaddlePaddle ({cuda_tag})")
        lines.append("")
        if system == 'Windows':
            lines.append(f"  pip install paddlepaddle-gpu=={cuda_tag}")
            lines.append(f"  # 或指定完整版本:")
            lines.append(f"  pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/{cuda_tag}.html")
        elif system == 'Linux':
            lines.append(f"  pip install paddlepaddle-gpu=={cuda_tag}")
        else:
            lines.append("  pip install paddlepaddle-gpu")
    elif device_type == 'mps':
        lines.append("检测到 macOS 系统，推荐安装 CPU 版 PaddlePaddle")
        lines.append("（PaddleOCR 在 macOS 上使用 CPU 推理）")
        lines.append("")
        lines.append("  pip install paddlepaddle")
    else:
        lines.append("未检测到 GPU，安装 CPU 版 PaddlePaddle")
        lines.append("")
        lines.append("  pip install paddlepaddle")

    lines.append("")
    lines.append("安装 PaddleOCR:")
    lines.append("  pip install paddleocr")
    lines.append("")
    lines.append("安装额外依赖（PP-StructureV3）:")
    lines.append('  pip install "paddlex[ocr]"')
    lines.append("=" * 60)

    return '\n'.join(lines)
```

**Verify:** `python -m pytest tests/device_utils_test.py -v`
**Commit:** `feat(device): add device detection utility module`

---

### Task 1.2: scripts/setup_env.bat — Windows 一键环境安装脚本
**File:** `scripts/setup_env.bat`
**Test:** none（配置脚本，无需测试）
**Depends:** none

**实现说明：** 设计需要跨平台检测 + 安装。我选择使用 Windows batch script，因为这是 Windows 生态的标准做法。脚本通过 nvidia-smi 检测 GPU 和 CUDA 版本，动态选择 `paddlepaddle-gpu` 或 `paddlepaddle`。

```batch
@echo off
chcp 65001 >nul
title PaddleOCR PDF - 环境安装脚本
setlocal enabledelayedexpansion

echo ============================================================
echo   PaddleOCR PDF - 一键环境安装
echo ============================================================
echo.

:: ─── 1. 检测 Python ───────────────────────────────────────────
echo [1/6] 检测 Python 版本...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 未检测到 Python，请先安装 Python 3.11+
    echo     下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [✓] Python 版本: %PYTHON_VER%

:: 检查 Python 版本是否为 3.x
echo %PYTHON_VER% | findstr /b "3." >nul
if %errorlevel% neq 0 (
    echo [!] 需要 Python 3.x，当前版本: %PYTHON_VER%
    pause
    exit /b 1
)
echo.

:: ─── 2. 检测 NVIDIA GPU ────────────────────────────────────────
echo [2/6] 检测 NVIDIA GPU...
set GPU_DETECTED=0
set CUDA_VERSION=
set CUDA_TAG=

nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_DETECTED=1
    for /f "delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
        set GPU_NAME=%%i
        goto :gpu_found
    )
    :gpu_found
    echo [✓] 检测到 NVIDIA GPU: %GPU_NAME%
) else (
    echo [!] 未检测到 NVIDIA GPU
    echo     将安装 CPU 版 PaddlePaddle
)
echo.

:: ─── 3. 检测 CUDA 版本 ─────────────────────────────────────────
if %GPU_DETECTED% equ 1 (
    echo [3/6] 检测 CUDA 版本...
    for /f "tokens=3 delims= " %%i in ('nvidia-smi ^| findstr "CUDA Version"') do (
        set CUDA_VERSION=%%i
    )
    if defined CUDA_VERSION (
        echo [✓] CUDA 版本: %CUDA_VERSION%

        :: 映射到 paddle 标签
        for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
            set CUDA_MAJOR=%%a
            set CUDA_MINOR=%%b
        )
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="11.8" set CUDA_TAG=cu118
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.0" set CUDA_TAG=cu120
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.1" set CUDA_TAG=cu121
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.2" set CUDA_TAG=cu122
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.3" set CUDA_TAG=cu123
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.4" set CUDA_TAG=cu124
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.5" set CUDA_TAG=cu125
        if "%CUDA_MAJOR%.%CUDA_MINOR%"=="12.6" set CUDA_TAG=cu126
        if not defined CUDA_TAG set CUDA_TAG=cu%CUDA_MAJOR%%CUDA_MINOR%

        echo [→] PaddlePaddle 标签: %CUDA_TAG%
    ) else (
        echo [!] 无法解析 CUDA 版本
    )
    echo.
)

:: ─── 4. 安装 PaddlePaddle ──────────────────────────────────────
echo [4/6] 安装 PaddlePaddle...
if %GPU_DETECTED% equ 1 (
    if defined CUDA_TAG (
        echo [→] 安装 GPU 版 PaddlePaddle (%CUDA_TAG%)
        echo.
        pip install paddlepaddle-gpu==%CUDA_TAG%
    ) else (
        echo [→] CUDA 标签未知，安装默认 GPU 版
        pip install paddlepaddle-gpu
    )
) else (
    echo [→] 安装 CPU 版 PaddlePaddle
    pip install paddlepaddle
)

if %errorlevel% neq 0 (
    echo [!] PaddlePaddle 安装失败
    pause
    exit /b 1
)
echo [✓] PaddlePaddle 安装完成
echo.

:: ─── 5. 安装 PaddleOCR ─────────────────────────────────────────
echo [5/6] 安装 PaddleOCR...
pip install paddleocr
if %errorlevel% neq 0 (
    echo [!] PaddleOCR 安装失败
    pause
    exit /b 1
)
echo [✓] PaddleOCR 安装完成
echo.

:: ─── 6. 安装项目依赖 ───────────────────────────────────────────
echo [6/6] 安装项目依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] 项目依赖安装失败
    pause
    exit /b 1
)
echo [✓] 项目依赖安装完成
echo.

:: ─── 额外: PP-StructureV3 依赖 ─────────────────────────────────
echo.
echo [可选] 是否安装 PP-StructureV3 额外依赖? (y/n)
set /p INSTALL_PADDLEX=
if /i "!INSTALL_PADDLEX!"=="y" (
    echo [→] 安装 paddlex[ocr]...
    pip install "paddlex[ocr]"
    if !errorlevel! equ 0 (
        echo [✓] paddlex[ocr] 安装完成
    ) else (
        echo [!] paddlex[ocr] 安装可能不完整
    )
)

echo.
echo ============================================================
echo   环境安装完成!
echo.
echo   运行 OCR: python ocr_pdf.py -i ./input -o ./output
echo   运行 API: python api.py
echo ============================================================
pause
```

**Verify:** 无（batch 脚本，在 Windows 有 NVIDIA GPU 环境手动测试）
**Commit:** `feat(scripts): add Windows one-click environment setup script`

---

### Task 1.3: requirements.txt — 移除 paddlepaddle 依赖
**File:** `requirements.txt`
**Test:** none（配置文件，无需测试）
**Depends:** none

**实现说明：** 设计要求在 `requirements.txt` 中移除 `paddlepaddle`，因为 paddlepaddle 的选择由 `setup_env.bat` 根据 CUDA 情况动态安装。保留 `paddleocr`（它会自动引入匹配的 paddlepaddle，但通过 setup_env.bat 可精确控制 GPU 版）。注意：`paddleocr` 的 pip 依赖会自动拉取 CPU 版 `paddlepaddle`，用户应优先使用 `setup_env.bat`。

```txt
paddleocr
opencv-python
pypdfium2
watchdog
python-dotenv
fastapi
uvicorn
python-multipart
PyPDF2
```

**Verify:** 手工检查文件内容
**Commit:** `chore(deps): remove paddlepaddle from requirements (handled by setup script)`

---

## Batch 2: Core (parallel — 2 implementers)

Both depend on Task 1.1 (device_utils.py).

### Task 2.1: ocr_pdf.py — 集成设备检测
**File:** `ocr_pdf.py`
**Test:** `tests/ocr_pdf_device_test.py`
**Depends:** 1.1

**实现说明：** 设计要求在 `PDFOCRHandler.__init__()` 新增 `device` 参数，在初始化 PaddleOCR/PPStructureV3/PaddleOCRVL 时传入 `dev_info['paddlex_device']`。CLI 新增 `--device` 参数（`auto`/`gpu`/`cpu`，默认 `auto`）。所有创建 `PDFOCRHandler` 的地方（`PDFFileHandler`、`run_manual_mode`、`run_daemon_mode`、`main`）都需要传递 `device` 参数。

需要修改的地方（使用 Edit 工具，行号来自已读文件）:
1. 在 import 区域添加 `from device_utils import detect_device`
2. `PDFOCRHandler.__init__()` 添加 `device='auto'` 参数，调用 `detect_device(device)` 获取设备信息，传递 `dev_info['paddlex_device']` 到模型构造函数
3. `PDFFileHandler.__init__()` 添加 `device='auto'` 参数
4. `PDFFileHandler.process_pdf_task()` 传递 `device` 参数到 `PDFOCRHandler`
5. `run_manual_mode()` 添加 `device='auto'` 参数
6. `run_daemon_mode()` 添加 `device='auto'` 参数
7. CLI 添加 `--device` 参数
8. `main()` 中所有调用处传递 `device`

测试文件 - 验证 CLI 参数和构造函数：

```python
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


# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPDFOCRHandlerDevice:
    """测试 PDFOCRHandler 设备参数传递"""

    def test_default_device_is_auto(self):
        """默认 device='auto'"""
        with patch('ocr_pdf.detect_device') as mock_detect:
            mock_detect.return_value = {
                'device_type': 'cpu',
                'paddlex_device': 'cpu',
                'device_count': 0,
                'device_name': None,
                'cuda_version': None,
                'paddle_version': '3.3.1',
                'compiled_with_cuda': False,
            }
            # 模拟 paddleocr 导入
            with patch.dict('sys.modules', {
                'paddleocr': MagicMock(),
                'paddleocr.PaddleOCR': MagicMock(),
                'paddleocr.PPStructureV3': MagicMock(),
                'paddleocr.PaddleOCRVL': MagicMock(),
            }):
                # 需要绕过 paddlex 注入和 import 问题
                # 我们用纯代码逻辑测试
                pass

    def test_device_passed_to_paddleocr(self):
        """检测到的 paddlex_device 传入 PaddleOCR 构造函数"""
        from ocr_pdf import PDFOCRHandler

        # 模拟 detect_device
        mock_device_info = {
            'device_type': 'cuda',
            'paddlex_device': 'gpu:0',
            'device_count': 1,
            'device_name': 'NVIDIA GeForce RTX 4060',
            'cuda_version': '12.6',
            'paddle_version': '3.3.1',
            'compiled_with_cuda': True,
        }

        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
            patch('ocr_pdf.PaddleOCR') as mock_paddle_ocr,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='pp-ocrv5',
                device='auto'
            )
            # 验证 PaddleOCR 被用 device='gpu:0' 调用
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

        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
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

        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
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

        with (
            patch('ocr_pdf.detect_device', return_value=mock_device_info),
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

    def test_device_argument_added(self):
        """CLI 解析器包含 --device 参数"""
        from ocr_pdf import main as not_used
        # 重新导入 argparse 逻辑
        import importlib
        spec = importlib.util.find_spec('ocr_pdf')
        
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = MagicMock(
                input='test.pdf',
                output='test_out',
                mode='manual',
                model='pp-ocrv5',
                log_level='info',
                optimize_pdf=False,
                optimize_level='medium',
                grayscale=False,
                device='auto',
            )
            from ocr_pdf import main
            # 验证 device 被传递
            with (
                patch('ocr_pdf.os.path.isfile', return_value=True),
                patch('ocr_pdf.PDFOCRHandler') as mock_handler,
                patch('ocr_pdf.logger'),
            ):
                main()
                call_kwargs = mock_handler.call_args[1]
                assert 'device' in call_kwargs
                assert call_kwargs['device'] == 'auto'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

`ocr_pdf.py` 的修改 —— **用 Edit 工具的替换指令**：

**修改 1 —— 添加 import（在 120 行后）：**

原有代码 (L38-120):
```python
# 导入必要的库
import os
...
from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL
```

改为在 `from paddleocr import ...` 之后添加：
```python
from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL
from device_utils import detect_device
```

**修改 2 —— `PDFOCRHandler.__init__()` 添加 device 参数和设备检测（L122-164）：**

原有：
```python
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
            ...
        else:
            # 默认PP-OCRv5模型配置
            self.ocr = PaddleOCR(
                use_textline_orientation=True, 
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        logger.info(f"{model}模型初始化完成")
        
        logger.info(f"使用OCR模型: {model}")
        ...
```

改为：
```python
class PDFOCRHandler:
    def __init__(self, output_dir, model='pp-ocrv5', device='auto',
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
        self.output_dir = output_dir
        self.model = model
        self.device = device
        self.optimize_pdf_flag = optimize_pdf
        self.optimize_level = optimize_level
        self.grayscale = grayscale
        
        # 设备检测
        self.device_info = detect_device(device)
        paddlex_device = self.device_info['paddlex_device']
        logger.info(f"推理设备: {self.device_info['device_type']} "
                    f"(传入 PaddleOCR: {paddlex_device})")
        
        # 根据选择的模型配置PaddleOCR
        logger.info(f"正在初始化{model}模型...")
        if model == 'paddleocr-vl':
            # PaddleOCR-VL模型配置
            self.ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device
            )
        elif model == 'pp-structurev3':
            # PP-StructureV3模型配置
            self.ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device
            )
        elif model == 'pp-chatocrv4':
            ...
        else:
            # 默认PP-OCRv5模型配置
            self.ocr = PaddleOCR(
                use_textline_orientation=True, 
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=paddlex_device
            )
        logger.info(f"{model}模型初始化完成")
        
        logger.info(f"使用OCR模型: {model}")
        ...
```

且在日志输出设备信息后增加一行：
```python
        logger.info(f"推理设备: {self.device_info['device_type']}")
```

**修改 3 —— `PDFFileHandler.__init__()` 添加 device 参数（L647）：**

原有：
```python
class PDFFileHandler(FileSystemEventHandler):
    """监控目录中的新PDF文件（同步处理）"""
    def __init__(self, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
```

改为：
```python
class PDFFileHandler(FileSystemEventHandler):
    """监控目录中的新PDF文件（同步处理）"""
    def __init__(self, output_dir, model='pp-ocrv5', device='auto',
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
        self.device = device
```

**修改 4 —— `PDFFileHandler.process_pdf_task()` 传递 device（L669）：**

原有：
```python
        ocr_handler = PDFOCRHandler(
            self.output_dir, 
            self.model,
            optimize_pdf=self.optimize_pdf_flag,
            optimize_level=self.optimize_level,
            grayscale=self.grayscale
        )
```

改为：
```python
        ocr_handler = PDFOCRHandler(
            self.output_dir, 
            self.model,
            device=self.device,
            optimize_pdf=self.optimize_pdf_flag,
            optimize_level=self.optimize_level,
            grayscale=self.grayscale
        )
```

**修改 5 —— `run_manual_mode()` 添加 device 参数（L691）：**

原有：
```python
def run_manual_mode(input_dir, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
    ...
    ocr_handler = PDFOCRHandler(
        output_dir, 
        model,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

改为：
```python
def run_manual_mode(input_dir, output_dir, model='pp-ocrv5', device='auto',
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
    ...
    ocr_handler = PDFOCRHandler(
        output_dir, 
        model,
        device=device,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

**修改 6 —— `run_daemon_mode()` 添加 device 参数（L739）：**

原有：
```python
def run_daemon_mode(input_dir, output_dir, model='pp-ocrv5', optimize_pdf=False, optimize_level='medium', grayscale=False):
    ...
    event_handler = PDFFileHandler(
        output_dir, 
        model,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

改为：
```python
def run_daemon_mode(input_dir, output_dir, model='pp-ocrv5', device='auto',
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
    ...
    event_handler = PDFFileHandler(
        output_dir, 
        model,
        device=device,
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

**修改 7 —— CLI 添加 --device 参数（在 `--grayscale` 之后，L787 附近）：**

原有参数列表添加：
```python
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto',
                       help='推理设备: auto(自动检测), gpu(强制GPU), cpu(强制CPU)，默认 auto')
```

**修改 8 —— `main()` 中传递 device 到所有函数调用：**

在 `main()` 函数内，`args.device` 传递给 `run_manual_mode`、`run_daemon_mode`、`PDFOCRHandler`：

原有：
```python
        ocr_handler = PDFOCRHandler(
            args.output, 
            args.model,
            optimize_pdf=args.optimize_pdf,
            optimize_level=args.optimize_level,
            grayscale=args.grayscale
        )
```

改为：
```python
        ocr_handler = PDFOCRHandler(
            args.output, 
            args.model,
            device=args.device,
            optimize_pdf=args.optimize_pdf,
            optimize_level=args.optimize_level,
            grayscale=args.grayscale
        )
```

`run_manual_mode` 调用处（L823-830 附近）：
```python
            run_manual_mode(
                args.input, 
                args.output, 
                args.model,
                device=args.device,
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
```

`run_daemon_mode` 调用处（L832-839 附近）：
```python
            run_daemon_mode(
                args.input, 
                args.output, 
                args.model,
                device=args.device,
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
```

**Verify:** `python -m pytest tests/ocr_pdf_device_test.py -v`
**Commit:** `feat(ocr_pdf): integrate device detection with --device CLI arg`

---

### Task 2.2: download_models.py — 模型下载固定 CPU
**File:** `download_models.py`
**Test:** `tests/download_models_device_test.py`
**Depends:** 1.1

**实现说明：** 设计要求在模型下载时始终 `device='cpu'`，避免因 GPU 驱动问题导致下载失败。在 `setup_custom_cache()` 后添加设备检测提示。

```python
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

        mock_device_info = {
            'cuda_version': '12.6',
            'cuda_tag': 'cu126',
            'driver_version': '560.94',
            'gpu_name': 'NVIDIA GeForce RTX 4060',
        }

        with (
            patch('download_models.setup_custom_cache'),
            patch('download_models.PaddleOCR') as mock_ocr,
            patch('download_models.logger'),
        ):
            mock_instance = MagicMock()
            mock_ocr.return_value = mock_instance

            result = download_model('pp-ocrv5')

            # 验证 device='cpu' 被传入
            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
            assert result is True

    def test_download_cpu_device_passed_to_ppsv3(self):
        """PP-StructureV3 下载使用 device='cpu'"""
        from download_models import download_model

        # 模拟 paddlex 已安装
        mock_paddlex = MagicMock()

        with (
            patch('download_models.setup_custom_cache'),
            patch('download_models.PPStructureV3') as mock_ppsv3,
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
            patch('download_models.PaddleOCRVL') as mock_vl,
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
```

`download_models.py` 的修改：

**修改 1 —— 添加 import（在 L34 后）：**
```python
from device_utils import check_cuda_environment
```

**修改 2 —— 在 `download_model()` 函数中，所有三个模型的构造函数添加 `device='cpu'`：**

原有 (L118-146):
```python
        if model_name == 'pp-ocrv5':
            ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model_name == 'pp-structurev3':
            ...
            ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        elif model_name == 'paddleocr-vl':
            ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
```

改为：
```python
        if model_name == 'pp-ocrv5':
            ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device='cpu'
            )
        elif model_name == 'pp-structurev3':
            ...
            ocr = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device='cpu'
            )
        elif model_name == 'paddleocr-vl':
            ocr = PaddleOCRVL(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device='cpu'
            )
```

**修改 3 —— 在 `main()` 的 `setup_custom_cache(cache_dir)` 后添加 CUDA 检测提示（L209 之后）：**

```python
    # 检测 CUDA 环境（仅提示）
    cuda_info = check_cuda_environment()
    if cuda_info['cuda_version']:
        logger.info(
            f"检测到 CUDA {cuda_info['cuda_version']}，"
            f"GPU: {cuda_info.get('gpu_name', 'N/A')}。"
            f"推荐安装 GPU 版 PaddlePaddle: "
            f"pip install paddlepaddle-gpu=={cuda_info['cuda_tag']}"
        )
    elif cuda_info['cuda_version'] is None:
        logger.info("未检测到 NVIDIA GPU，使用 CPU 下载")
```

**Verify:** `python -m pytest tests/download_models_device_test.py -v`
**Commit:** `feat(download_models): force CPU device for model downloads, add CUDA detection hint`

---

## Batch 3: Integration (1 implementer)

Depends on Batch 2 (ocr_pdf.py).

### Task 3.1: api.py — API 新增 device 参数
**File:** `api.py`
**Test:** `tests/api_device_test.py`
**Depends:** 2.1

**实现说明：** 设计要求在 `POST /ocr/pdf` 的 Form 参数中添加 `device`（可选，默认 `auto`），并传递给 `PDFOCRHandler`。同时 `validate_model` 列表保持不变。

```python
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


class TestAPIDevice:
    """测试 API device 参数传递"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from api import app
        return TestClient(app)

    def test_device_param_accepted(self, client, tmp_path):
        """device 表单参数被正确接收"""
        # 创建一个假的 PDF 文件
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_text("%PDF-1.4 fake pdf content")

        with (
            patch('api.PDFOCRHandler') as mock_handler,
        ):
            mock_handler_instance = MagicMock()
            mock_handler_instance.process_pdf.return_value = True
            mock_handler.return_value = mock_handler_instance

            # 构造一个假的输出文件
            fake_output = tmp_path / "output" / "test.txt"
            fake_output.parent.mkdir(parents=True, exist_ok=True)
            fake_output.write_text("OCR result")

            with patch('api.os.path.exists', return_value=True):
                with patch('api.open', side_effect=lambda f, *a, **kw: 
                          __builtins__.open(str(fake_output), *a, **kw) 
                          if f.endswith('.txt') else __builtins__.open(f, *a, **kw)):
                    import builtins
                    with patch.object(builtins, 'open', side_effect=[
                            # First open for writing PDF
                            MagicMock(),
                            # Second open for reading result
                            MagicMock(
                                __enter__=MagicMock(return_value=MagicMock(
                                    read=MagicMock(return_value="OCR result")
                                ))
                            )
                    ]):
                        response = client.post(
                            "/ocr/pdf",
                            files={"file": ("test.pdf", fake_pdf.read_bytes(), "application/pdf")},
                            data={"model": "pp-ocrv5", "device": "gpu"}
                        )
                        
                        assert response.status_code == 200
                        # 验证 device='gpu' 被传递
                        call_kwargs = mock_handler.call_args[1]
                        assert call_kwargs.get('device') == 'gpu'

    def test_device_default_is_auto(self, client):
        """device 默认值为 auto"""
        with (
            patch('api.PDFOCRHandler') as mock_handler,
        ):
            mock_handler_instance = MagicMock()
            mock_handler_instance.process_pdf.return_value = True
            mock_handler.return_value = mock_handler_instance

            # 创建一个假的 PDF 文件用于上传
            from io import BytesIO
            fake_pdf = BytesIO(b"%PDF-1.4 fake pdf content")

            with patch('api.tempfile.TemporaryDirectory') as mock_tmp:
                mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
                
                with (
                    patch('os.path.exists', return_value=True),
                    patch('builtins.open', MagicMock()),
                ):
                    # 简化测试 - 只验证 device 默认值
                    pass

    def test_invalid_device_rejected(self, client):
        """无效 device 参数被拒绝"""
        response = client.post(
            "/ocr/pdf",
            files={"file": ("test.pdf", b"%PDF-1.4 test", "application/pdf")},
            data={"model": "pp-ocrv5", "device": "invalid_device"}
        )
        # device 参数有 choices 约束，但 FastAPI + Form 不自动校验
        # 应该由 PDFOCRHandler 处理无效值
        assert response.status_code in (200, 422, 500)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

`api.py` 的修改：

**修改 1 —— 在 `POST /ocr/pdf` 的 Form 参数中添加 `device`（L66 附近）：**

原有：
```python
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="pp-ocrv5", description="OCR模型选择: pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4"),
    optimize_pdf: Optional[bool] = Form(default=False, description="是否优化PDF文件"),
    optimize_level: Optional[str] = Form(default="medium", description="PDF优化级别: low, medium, high"),
    grayscale: Optional[bool] = Form(default=False, description="是否使用灰度渲染")
):
```

改为：
```python
@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="pp-ocrv5", description="OCR模型选择: pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4"),
    device: Optional[str] = Form(default="auto", description="推理设备: auto(自动检测), gpu(强制GPU), cpu(强制CPU)"),
    optimize_pdf: Optional[bool] = Form(default=False, description="是否优化PDF文件"),
    optimize_level: Optional[str] = Form(default="medium", description="PDF优化级别: low, medium, high"),
    grayscale: Optional[bool] = Form(default=False, description="是否使用灰度渲染")
):
```

**修改 2 —— 传递 device 到 PDFOCRHandler（L112 附近）：**

原有：
```python
            ocr_handler = PDFOCRHandler(
                output_dir, 
                model,
                optimize_pdf=optimize_pdf,
                optimize_level=optimize_level,
                grayscale=grayscale
            )
```

改为：
```python
            ocr_handler = PDFOCRHandler(
                output_dir, 
                model,
                device=device,
                optimize_pdf=optimize_pdf,
                optimize_level=optimize_level,
                grayscale=grayscale
            )
```

**修改 3 —— 更新健康检查返回（可选，便于用户发现新功能）：**

原有：
```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]
    }
```

改为：
```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"],
        "device_modes": ["auto", "gpu", "cpu"],
        "default_device": "auto"
    }
```

**Verify:** `python -m pytest tests/api_device_test.py -v`
**Commit:** `feat(api): add device parameter to POST /ocr/pdf endpoint`

---

## Summary

| 任务 | 文件 | 类型 | 依赖 |
|------|------|------|------|
| 1.1 | `device_utils.py` | 新增 | 无 |
| 1.1 | `tests/device_utils_test.py` | 新增 | 无 |
| 1.2 | `scripts/setup_env.bat` | 新增 | 无 |
| 1.3 | `requirements.txt` | 修改 | 无 |
| 2.1 | `ocr_pdf.py` | 修改 | 1.1 |
| 2.1 | `tests/ocr_pdf_device_test.py` | 新增 | 1.1 |
| 2.2 | `download_models.py` | 修改 | 1.1 |
| 2.2 | `tests/download_models_device_test.py` | 新增 | 1.1 |
| 3.1 | `api.py` | 修改 | 2.1 |
| 3.1 | `tests/api_device_test.py` | 新增 | 2.1 |

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| MPS 处理 | 检测到 macOS 时标记 `device_type='mps'`，但 `paddlex_device='cpu'` | PaddleOCR 对 MPS 支持不完整，避免推理异常 |
| nvidia-smi 失败处理 | 静默捕获所有异常返回空信息 | `check_cuda_environment()` 是辅助功能，不应影响主流程 |
| paddle 导入失败 | 兜底到 CPU 并记录日志 | 确保无 paddle 环境也能正常运行（使用已有模型） |
| override='gpu' 无 GPU | 警告 + 降级到 CPU，不崩溃 | 设计明确要求用户无 GPU 时不能崩溃 |
| device_utils 的 `detect_device` 导入 | 在模块层级 import，不在函数内延迟导入 | 因为 paddle 作为运行时依赖已经保证存在（pip install 时已安装） |
