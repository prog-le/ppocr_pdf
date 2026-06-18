# -*- coding: utf-8 -*-
"""
device_utils — GPU / CPU / MPS 设备检测模块

本模块提供三个核心函数：
  1. detect_device(device_override=None)  → dict
  2. check_cuda_environment()             → dict
  3. get_install_guide(device_type, cuda_tag) → str

设计原则：
  - 所有外部调用（subprocess / import）都集中在顶层，方便 mock 测试。
  - 不自动安装任何包，只做检测和建议。
  - 返回结构化的 dict，调用方可以直接序列化为 JSON 或用于日志。
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ==============================================================
# 环境变量 — 在 PaddlePaddle 任何初始化之前生效
# ==============================================================
# PaddlePaddle 3.x oneDNN PIR bug (issue #63659)
#   ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
# 禁用 PIR 执行器以回退到旧版执行器，避开此 bug。
os.environ.setdefault("FLAGS_enable_pir_api", "0")


# ==============================================================
# 常量
# ==============================================================

NVIDIA_SMI_NAMES = ["nvidia-smi", "nvidia-smi.exe"]

# 官方推荐的 CUDA → paddlepaddle-gpu 对照表
# 参考 https://www.paddlepaddle.org.cn/install/quick
CUDA_TAG_MAP: dict[str, str] = {
    "12.6": "cu126",
    "12.5": "cu125",
    "12.4": "cu124",
    "12.3": "cu123",
    "12.2": "cu122",
    "12.1": "cu121",
    "12.0": "cu120",
    "11.8": "cu118",
    "11.7": "cu117",
    "11.6": "cu116",
    "11.5": "cu115",
    "11.4": "cu114",
    "11.3": "cu113",
    "11.2": "cu112",
}

INSTALL_BASE_URL = "https://www.paddlepaddle.org.cn/install/quick"


# ==============================================================
# 数据结构
# ==============================================================


def _to_paddlex_device(device: str, gpu_info: str = "") -> str:
    """
    将 device_utils 的设备名转换为 PaddleX / PaddleOCR 可接受的 device 参数。
    
    - "gpu" → "gpu:0" （如果有 GPU，默认用第 0 块）
    - "cpu"  → "cpu"
    - "mps"  → "mps" （macOS Metal，Paddle 尚未支持，保留占位）
    """
    device_lower = device.lower().strip()
    if device_lower == "gpu":
        return "gpu:0"
    # cpu / mps 等原样返回
    return device_lower


@dataclass
class DeviceInfo:
    """设备检测结果"""

    device: str = "cpu"  # "cpu" / "gpu" / "mps" / "npu"
    device_type: str = "cpu"  # 更精确的设备类型: "cuda" / "cpu" / "mps"
    paddlex_device: str = "cpu"  # 传入 PaddleOCR 的 device 参数，如 "gpu:0"
    pkg_suffix: str = ""  # paddlepaddle 包后缀: "" / "-gpu"
    cuda_version: str = ""  # 检测到的 CUDA 版本号，如 "12.6"
    cuda_tag: str = ""  # 映射后的标签，如 "cu126"
    driver_version: str = ""  # NVIDIA 驱动版本
    gpu_info: str = ""  # GPU 型号 / 数量
    detail: str = ""  # 人类可读的描述
    source: str = "auto"  # "auto" / "override"


@dataclass
class CudaEnvironment:
    """check_cuda_environment 返回值"""

    cuda_version: str = ""
    driver_version: str = ""
    gpu_info: str = ""
    nvidia_smi_path: str = ""
    detail: str = ""


# ==============================================================
# 底层工具函数（便于 mock）
# ==============================================================


def _find_nvidia_smi() -> Optional[str]:
    """查找 nvidia-smi 可执行文件路径。"""
    for name in NVIDIA_SMI_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _run_nvidia_smi(smi_path: str) -> str:
    """执行 nvidia-smi 并返回 stdout。"""
    result = subprocess.run(
        [smi_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _parse_cuda_version(smi_output: str) -> str:
    """从 nvidia-smi 输出中提取 CUDA 版本。"""
    match = re.search(r"CUDA Version:\s*([\d.]+)", smi_output)
    return match.group(1).strip() if match else ""


def _parse_driver_version(smi_output: str) -> str:
    """从 nvidia-smi 输出中提取驱动版本。"""
    match = re.search(r"Driver Version:\s*([\d.]+)", smi_output)
    return match.group(1).strip() if match else ""


def _parse_gpu_info(smi_output: str) -> str:
    """从 nvidia-smi 输出中提取 GPU 型号信息。"""
    lines = smi_output.splitlines()
    # 1) 标准格式: "0   NVIDIA GeForce RTX 4090  WDDM ..."
    for line in lines:
        gpu_match = re.match(r"\s*\d+\s+(NVIDIA\s+.+?)\s{2,}", line)
        if gpu_match:
            return gpu_match.group(1).strip()
    # 2) "GPU 0: NVIDIA A100 80GB PCIe" 格式
    for line in lines:
        gpu_match = re.match(r".*?GPU\s+\d+:\s+(NVIDIA\s+.+)", line)
        if gpu_match:
            return gpu_match.group(1).strip()
    # 3) 自由文本中提取 NVIDIA 行
    for line in lines:
        gpu_match = re.search(r"(NVIDIA\s+\S+(?:\s+\S+){0,4})", line)
        if gpu_match:
            return gpu_match.group(1).strip()
    # 4) 裸型号行 (如 "RTX 4090  WDDM ...")
    for line in lines:
        gpu_match = re.match(r"\s*(RTX\s+\d+|GTX\s+\d+|A\d{2,3}|Quadro\s+\S+|Tesla\s+\S+)\s", line)
        if gpu_match:
            return gpu_match.group(1).strip()
    return ""


def _map_cuda_tag(cuda_version: str) -> str:
    """将 CUDA 版本号映射为 paddle 标签。"""
    # 尝试精确匹配
    if cuda_version in CUDA_TAG_MAP:
        return CUDA_TAG_MAP[cuda_version]
    # 尝试前缀匹配（如 "12.6" 匹配 "cu126"）
    for ver, tag in CUDA_TAG_MAP.items():
        if cuda_version.startswith(ver):
            return tag
    return ""


# ==============================================================
# 公开函数
# ==============================================================


def verify_paddle_device(paddlex_device: str) -> Dict[str, str]:
    """
    验证当前安装的 PaddlePaddle 是否能实际使用指定的设备。

    当 detect_device() 返回了 ``gpu:0`` 但 PaddlePaddle 是 CPU 版本时，
    此函数会强制回退到 CPU，并打印详细的安装指引。

    Args:
        paddlex_device: detect_device() 返回的 paddlex_device 值，
                        如 ``"gpu:0"`` 或 ``"cpu"``。

    Returns:
        dict: 包含以下键:
            - paddlex_device (str): 修正后的设备名（回退后为 ``"cpu"``）
            - fallback (str): 是否发生了回退, ``"true"`` 或 ``"false"``
            - message (str): 人类可读的说明，无回退时为空字符串
            - paddle_cuda (str): PaddlePaddle 是否编译了 CUDA 支持,
                                 ``"true"`` 或 ``"false"``
    """
    try:
        import paddle  # type: ignore[import-untyped]
    except ImportError:
        logger.error("PaddlePaddle 未安装，无法进行设备验证")
        return {
            "paddlex_device": "cpu",
            "fallback": "true",
            "message": "PaddlePaddle is not installed",
            "paddle_cuda": "false",
        }

    needs_gpu = paddlex_device.startswith("gpu:")
    paddle_has_cuda = paddle.is_compiled_with_cuda()

    if needs_gpu and not paddle_has_cuda:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("   GPU 检测与 PaddlePaddle 不匹配")
        logger.warning("=" * 60)
        logger.warning("  nvidia-smi 检测到 NVIDIA GPU，但当前安装的")
        logger.warning("  PaddlePaddle 是 CPU 版本（未编译 CUDA 支持）。")
        logger.warning("")
        logger.warning("  自动切换到 CPU 推理以保持可用性。")
        logger.warning("")
        logger.warning("  如需 GPU 加速，请安装 GPU 版 PaddlePaddle:")
        logger.warning("    pip uninstall paddlepaddle -y")
        logger.warning("    pip install paddlepaddle-gpu")
        logger.warning("")
        logger.warning("  当前 PaddlePaddle 版本: %s", paddle.__version__)
        logger.warning("=" * 60)
        logger.warning("")
        return {
            "paddlex_device": "cpu",
            "fallback": "true",
            "message": (
                "GPU detected via nvidia-smi but PaddlePaddle is CPU-only; "
                f"fallback to CPU (paddle v{paddle.__version__})"
            ),
            "paddle_cuda": "false",
        }

    return {
        "paddlex_device": paddlex_device,
        "fallback": "false",
        "message": "",
        "paddle_cuda": "true" if paddle_has_cuda else "false",
    }


def check_cuda_environment() -> Dict[str, str]:
    """
    探测 CUDA 环境信息。

    Returns:
        dict: 包含 cuda_version, driver_version, gpu_info, nvidia_smi_path, detail
    """
    env = CudaEnvironment()

    smi_path = _find_nvidia_smi()
    if not smi_path:
        env.detail = "nvidia-smi not found on PATH"
        logger.debug(env.detail)
        return asdict(env)

    env.nvidia_smi_path = smi_path

    try:
        stdout = _run_nvidia_smi(smi_path)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        env.detail = f"nvidia-smi execution failed: {exc}"
        logger.warning(env.detail)
        return asdict(env)

    env.cuda_version = _parse_cuda_version(stdout)
    env.driver_version = _parse_driver_version(stdout)
    env.gpu_info = _parse_gpu_info(stdout)

    parts = []
    if env.cuda_version:
        parts.append(f"CUDA {env.cuda_version}")
    if env.driver_version:
        parts.append(f"Driver {env.driver_version}")
    if env.gpu_info:
        parts.append(env.gpu_info)
    env.detail = " | ".join(parts) if parts else "CUDA environment detected"

    return asdict(env)


def detect_device(device_override: Optional[str] = None) -> Dict[str, str]:
    """
    自动检测可用计算设备，返回结构化的设备信息。

    检测优先级:
      1. device_override — 如果提供且不是 "auto"，直接使用
      2. nvidia-smi + CUDA — 可用则返回 GPU
      3. macOS + Apple Silicon — 返回 mps (预告)
      4. 以上均不可用 → CPU

    Args:
        device_override: 可选 "auto" / "cpu" / "gpu" / "mps"

    Returns:
        dict: DeviceInfo 的所有字段，包括:
          - device: "cpu" / "gpu" / "mps"
          - device_type: 更精确的类型 "cpu" / "cuda" / "mps"
          - paddlex_device: 传入 PaddleOCR 的 device 参数 (如 "gpu:0" / "cpu")
          - pkg_suffix, cuda_version, cuda_tag, driver_version, gpu_info, detail, source
    """
    info = DeviceInfo()

    # ------ 1. 用户手动指定 (排除 "auto") ------
    if device_override is not None and device_override.lower() != "auto":
        device_lower = device_override.lower()
        info.source = "override"
        info.detail = f"User override: {device_lower}"

        if device_lower == "gpu":
            info.device = "gpu"
            info.device_type = "cuda"
            info.pkg_suffix = "-gpu"
            # 仍然尝试检测 CUDA 版本用于日志
            cuda_env = check_cuda_environment()
            if cuda_env["cuda_version"]:
                info.cuda_version = cuda_env["cuda_version"]
                info.cuda_tag = _map_cuda_tag(info.cuda_version)
                info.driver_version = cuda_env["driver_version"]
                info.gpu_info = cuda_env["gpu_info"]
                info.detail = (
                    f"GPU (override, {info.cuda_version}, {info.gpu_info})"
                )
        else:
            # cpu / mps 等
            info.device = device_lower
            info.detail = f"{device_lower.upper()} (override)"

        info.paddlex_device = _to_paddlex_device(info.device, info.gpu_info)
        return asdict(info)

    # ------ 2. 自动检测 (device_override is None or "auto") ------
    smi_path = _find_nvidia_smi()

    if not smi_path:
        # 无 NVIDIA 驱动
        system = platform.system()
        if system == "Darwin":
            # macOS — 预告 MPS 支持
            info.device = "cpu"
            info.detail = f"CPU (macOS {platform.machine()}, no CUDA)"
            logger.info("macOS detected: MPS support is planned for a future version")
        else:
            info.device = "cpu"
            info.detail = f"CPU ({system}, no NVIDIA driver detected)"
        info.paddlex_device = _to_paddlex_device(info.device)
        return asdict(info)

    # ------ 3. 有 nvidia-smi → 检测 CUDA ------
    try:
        stdout = _run_nvidia_smi(smi_path)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        info.device = "cpu"
        info.detail = f"CPU (nvidia-smi error: {exc})"
        info.paddlex_device = "cpu"
        return asdict(info)

    cuda_version = _parse_cuda_version(stdout)
    driver_version = _parse_driver_version(stdout)
    gpu_info = _parse_gpu_info(stdout)

    if cuda_version:
        info.device = "gpu"
        info.device_type = "cuda"
        info.pkg_suffix = "-gpu"
        info.cuda_version = cuda_version
        info.cuda_tag = _map_cuda_tag(cuda_version)
        info.driver_version = driver_version
        info.gpu_info = gpu_info
        info.detail = (
            f"GPU ({cuda_version}, {driver_version}, {gpu_info or 'unknown GPU'})"
        )
        info.paddlex_device = _to_paddlex_device("gpu", gpu_info)
    else:
        # nvidia-smi 存在但无法正常解析 CUDA 版本
        info.device = "cpu"
        info.device_type = "cpu"
        info.detail = "CPU (nvidia-smi found but no CUDA version parsed)"
        info.paddlex_device = "cpu"

    return asdict(info)


def get_install_guide(
    device_type: str = "cpu",
    cuda_tag: Optional[str] = None,
) -> str:
    """
    根据设备和平台返回 PaddlePaddle 安装指引文本。

    Args:
        device_type: "cpu" | "gpu" | "mps"
        cuda_tag:  CUDA 标签，如 "cu126"；为 None 时从 CUDA_TAG_MAP 推断

    Returns:
        包含安装命令和建议的多行文本
    """
    system = platform.system()  # Windows / Linux / Darwin

    lines: list[str] = []

    if device_type == "gpu":
        tag = cuda_tag or "cu126"  # 默认使用最新的稳定 CUDA 标签
        lines.append(f"# PaddlePaddle GPU ({tag}) 安装指南 — {system}")
        lines.append("")
        lines.append(f"推荐命令：")
        lines.append(f"  pip install paddlepaddle-gpu=={tag}")
        lines.append("")
        if system == "Windows":
            lines.append("Windows 提示：")
            lines.append("  1. 确保已安装 NVIDIA GPU 驱动 (建议 525+)")
            lines.append("  2. CUDA 运行时库会随 pip 包自动安装，无需单独安装 CUDA Toolkit")
            lines.append("  3. 验证安装: python -c \"import paddle; print(paddle.__version__)\"")
            lines.append("")
            lines.append(f"更多信息: {INSTALL_BASE_URL}")
        elif system == "Linux":
            lines.append("Linux 提示：")
            lines.append("  1. 推荐使用 NVIDIA 官方提供的 CUDA 11.8+ Docker 镜像")
            lines.append("  2. 或安装 nvidia-driver-535+ 和 nvidia-cuda-toolkit")
            lines.append("  3. 验证安装: python -c \"import paddle; print(paddle.__version__)\"")
            lines.append("")
            lines.append(f"更多信息: {INSTALL_BASE_URL}")
        else:
            # Darwin / 其他
            lines.append(f"{system} 上 GPU 支持有限，建议使用 CPU 版本。")
            lines.append("  pip install paddlepaddle")
    else:
        # CPU / MPS
        lines.append(f"# PaddlePaddle CPU 安装指南 — {system}")
        lines.append("")
        lines.append("推荐命令：")
        lines.append("  pip install paddlepaddle")
        lines.append("")
        if system == "Windows":
            lines.append("Windows 提示：")
            lines.append("  1. Python 3.9 ~ 3.12 均受支持")
            lines.append("  2. 建议使用虚拟环境 (conda 或 venv)")
            lines.append("  3. 验证安装: python -c \"import paddle; print(paddle.__version__)\"")
        elif system == "Darwin":
            lines.append("macOS 提示：")
            lines.append("  1. Apple Silicon (M 系列) 建议使用 arm64 Python")
            lines.append("  2. Intel Mac 使用 x86_64 Python")
            lines.append("  3. 验证安装: python -c \"import paddle; print(paddle.__version__)\"")
        else:
            lines.append("Linux 提示：")
            lines.append("  1. 建议使用虚拟环境 (conda 或 venv)")
            lines.append("  2. 验证安装: python -c \"import paddle; print(paddle.__version__)\"")
        lines.append("")
        lines.append(f"更多信息: {INSTALL_BASE_URL}")

    return "\n".join(lines)


# ==============================================================
# CLI 入口（快捷检测）
# ==============================================================


def main():
    """命令行快速设备检测。"""
    import argparse

    parser = argparse.ArgumentParser(description="PaddlePaddle 设备检测工具")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "mps"],
        default="auto",
        help="设备类型 (默认 auto 自动检测)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出",
    )
    args = parser.parse_args()

    device_override = None if args.device == "auto" else args.device
    info = detect_device(device_override)

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print(f"Device         : {info['device']}")
        print(f"Device Type    : {info['device_type']}")
        print(f"PaddleX Device : {info['paddlex_device']}")
        print(f"CUDA Version   : {info['cuda_version'] or '(none)'}")
        print(f"Driver         : {info['driver_version'] or '(none)'}")
        print(f"GPU            : {info['gpu_info'] or '(none)'}")
        print(f"Detail         : {info['detail']}")
        print()
        print(get_install_guide(info["device"], info["cuda_tag"] or None))


if __name__ == "__main__":
    main()
