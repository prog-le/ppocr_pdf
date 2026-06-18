---
date: 2026-06-18
topic: "自动设备检测与GPU/CPU/MPS自适应"
status: validated
---

# 自适应 GPU/CPU/MPS 设备检测设计

## 问题陈述

当前 `ppocr_pdf` 项目在 conda 环境中只安装了 CPU 版的 `paddlepaddle`，即使宿主机器有 NVIDIA RTX 4060 GPU 也无法利用 GPU 加速推理。且在不同平台上（Windows CUDA / macOS MPS / Linux CPU）需要手动选择不同的 paddlepaddle 安装包，用户体验差。

## 约束

- 必须兼容现有 CLI 参数（不能破坏现有调用方式）
- PaddleOCR 3.x 使用 `device` 参数控制推理设备
- PaddlePaddle 在不同平台有不同安装包（`paddlepaddle` CPU / `paddlepaddle-gpu` CUDA）
- 设备检测必须在运行时完成，不能依赖用户手动配置环境变量
- macOS MPS 目前只在 PaddleOCR-VL 中完整支持（Apple Silicon）

## 方案

### 核心架构

新增 `device_utils.py` 模块作为设备检测的统一入口，所有需要推理的模块都通过它获取设备信息。

```
device_utils.py         ← 新增
    detect_device()     → 返回设备信息 dict
    
ocr_pdf.py              ← 修改
    PDFOCRHandler        → 初始化时传入 device 参数
    CLI                  → 新增 --device 参数
    
download_models.py      ← 修改
    下载时固定 CPU       → 仅检测并提示 GPU 可用

api.py                  ← 修改
    POST /ocr/pdf        → 新增 device 表单参数

setup_env.bat           ← 新增
    一键环境检测安装脚本
```

### `device_utils.py` 详细设计

**`detect_device(device_override=None)` 函数：**

```python
返回格式:
{
    'device_type': 'cuda' | 'mps' | 'cpu',       # 逻辑设备类型
    'paddlex_device': 'gpu:0' | 'cpu',            # 传给 PaddleOCR 的 device 参数
    'device_count': 0 | 1 | 2 | ...,              # 设备数量
    'device_name': 'NVIDIA GeForce RTX 4060' | None,  # 设备名称
    'cuda_version': '12.6' | None,                # CUDA 版本（仅 CUDA）
    'paddle_version': '3.3.1' | None,             # PaddlePaddle 版本
    'compiled_with_cuda': True | False,            # 当前 paddle 是否 CUDA 编译
}
```

检测流程：

```
device_override?
  ├─ "gpu" → 强制 gpu:0（如果 paddle 不支持 CUDA 则警告降级到 cpu）
  ├─ "cpu" → 强制 cpu
  └─ None → 自动检测
       1. import paddle
       2. paddle.is_compiled_with_cuda() and device_count > 0 → cuda
       3. platform.system() == 'Darwin' and check MPS → mps
       4. 兜底 → cpu
```

**`check_cuda_environment()` 函数：**
- 调用 `nvidia-smi` 检测 CUDA 驱动版本
- 返回推荐的 CUDA 版本号（`cu118` / `cu126` / `cu129`）

**`get_install_guide()` 函数：**
- 根据当前系统和检测结果，输出正确的 pip install 命令

### `ocr_pdf.py` 修改

**`PDFOCRHandler.__init__()` 接收新参数：**
```python
def __init__(self, output_dir, model='pp-ocrv5', device='auto', ...):
```

初始化时：
1. 调用 `detect_device(device)` 获取设备信息
2. 传入 `device=dev_info['paddlex_device']` 到 `PaddleOCR()` / `PPStructureV3()` / `PaddleOCRVL()`
3. 日志输出检测到的设备

**CLI 新增参数：**
```python
parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto',
                    help='推理设备: auto(自动检测), gpu, cpu，默认 auto')
```

**设备选择优先级：** CLI 参数 > 自动检测 > CPU 兜底

### `download_models.py` 修改

- `setup_custom_cache()` 后添加设备检测
- 下载过程始终 `device="cpu"`（模型下载不依赖 GPU，避免因 GPU 驱动问题导致下载失败）
- 检测到 CUDA 环境时，额外打印提示："检测到 NVIDIA GPU，建议安装 paddlepaddle-gpu"

### `api.py` 修改

- `POST /ocr/pdf` 新增 `device` 表单参数（可选，默认 `auto`）
- 传递给 `PDFOCRHandler`

### 安装脚本 `scripts/setup_env.bat`

分步执行：
1. 检测 Python 版本
2. 检测 NVIDIA GPU（通过 `nvidia-smi`）
3. 检测 CUDA 版本（`cu118` / `cu126` / `cu129`）
4. 安装对应 `paddlepaddle-gpu` 或 `paddlepaddle`
5. 安装 `paddleocr[all]`
6. 安装项目 requirements.txt

### `requirements.txt` 更新

移除 `paddlepaddle` 依赖（因为需要根据平台动态选择安装 CPU 或 GPU 版），改为在安装脚本中处理。

## 数据流

```
CLI/API 调用
    │
    ▼
PDFOCRHandler.__init__(device='auto')
    │
    ▼
device_utils.detect_device('auto')
    │  ├─ paddle.is_compiled_with_cuda() + device_count > 0 → 'gpu:0'
    │  ├─ macOS + MPS available → 'mps' → 实际 PaddleOCR 仍用 'cpu'
    │  └─ 兜底 → 'cpu'
    │
    ▼
PaddleOCR(device=detected_device)  # 传入 device 参数
    │
    ▼
OCR 推理使用指定设备
```

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户指定 `--device gpu` 但无 GPU | 警告 + 降级到 CPU，不崩溃 |
| `paddle` 导入失败 | 日志警告，兜底 CPU |
| 设备检测异常 | 捕获异常，兜底 CPU |
| 多 GPU 环境 | 默认使用 `gpu:0`，用户可 `--device gpu:1` |

## 测试策略

- 在 CPU 环境验证自动检测到 `cpu`
- 在 CUDA 环境验证自动检测到 `gpu:0`
- 验证 `--device gpu` 在无 GPU 环境降级到 CPU
- 验证 `--device cpu` 强制走 CPU
- 验证 `download_models.py` 在 GPU 环境下下载时使用 CPU
- 验证 API 接口的 device 参数传递

## 开放问题

- macOS MPS 的实际 PaddleOCR device 参数需要确认是否直接支持 `mps` 还是需要特殊处理
