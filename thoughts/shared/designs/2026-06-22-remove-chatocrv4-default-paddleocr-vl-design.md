---
date: 2026-06-22
topic: "禁用 PP-ChatOCRv4 + 移除 pp-ocrv5 + 默认 PaddleOCR-VL"
status: validated

## 基线确认 (2026-06-22 00:10)

用户预先测试 `python ocr_pdf.py -i .\test.pdf -o .\ --device gpu`（默认 `pp-ocrv6`），运行成功：
- GPU 0 (CC 8.9) 激活，PP-OCRv6_medium_det + PP-OCRv6_medium_rec 从 ModelScope 下载
- 单 PDF 处理完成，无错误
- 验证 `--device gpu` 仍工作（device_utils.py 未改）
- 验证 `PaddleOCR(...)` 走默认 v6 medium 兜底分支仍正常

**基线 = 重构后必须保持的最小行为。** 默认 model 切到 `paddleocr-vl` 后的新基线验收需走相同命令（但 `-model paddleocr-vl`）。
---

# 禁用 PP-ChatOCRv4 + 收窄到 3 模型 + 默认 PaddleOCR-VL 设计

## 问题陈述

`ppocr_pdf` 项目当前在 CLI / API / 下载器三处暴露 **5 个模型名**：

- `paddleocr-vl`（多模态 VLM）
- `pp-ocrv6`（PaddleOCR 3.7+ 默认通用 OCR）
- `pp-ocrv5`（旧版通用 OCR）
- `pp-structurev3`（结构化解析）
- `pp-chatocrv4`（信息抽取，需千帆 API key）

实际使用中有 3 个具体障碍：

1. **PP-ChatOCRv4 在代码里是死代码**——`ocr_pdf.py` 第 220-223 行 `raise ValueError("需要额外的API配置，暂不支持直接使用")`，README 表格也标注「需 API 密钥，暂不直接支持」。但它的存在污染了 CLI choices、API valid_models、import 链路（`chatocr_patch.py` 412 行 monkey-patch + 两个文件顶部的 `try: import chatocr_patch` 块）。
2. **PP-OCRv5 是过期版本**——v6 已是 PaddleOCR 3.7+ 默认，v5 没有保留价值。
3. **PaddleOCR-VL 应为默认**——PaddleOCR-VL-1.6（2026-05-28 发布）96.3% 准确率，SOTA，最适合 PDF 文档场景。当前默认 `pp-ocrv6` 是「过时的稳妥」。

**用户目标：** 收窄到 3 模型（`paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`），默认 `paddleocr-vl`，**代码层硬删除** pp-chatocrv4 分支和 chatocr_patch 工具，**同步更新**所有引用（测试、文档、下载器、依赖清单）。

## 约束

- 3 个模型硬编码：仅 `paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`
- 默认模型 = `paddleocr-vl`
- `pp-chatocrv4` 在 valid_models / argparse choices / dispatch 分支 **全部硬删除**（不是配置层 hide）
- `pp-ocrv5` **同上硬删除**（用户明确要求）
- `chatocr_patch.py` **整文件删除**（user: "chatocr不需要了"）
- `chatocr_patch` 在 `ocr_pdf.py` 和 `api.py` 顶部的 import 块**删除**
- 既有 49/49 测试必须全部通过
- 设备检测逻辑（`device_utils.py`）保持不变
- 现有 PaddleOCR/PPStructureV3/PaddleOCRVL 三个 **专用类** 架构保持不变（最小风险）
- 不改 PaddleOCR 调用参数（`use_queues=False`、`enable_mkldnn=False` 等生产环境调优）

## 方案

### 核心架构（最小改动原则）

**保持现有 per-pipeline 专用类架构**，仅在 dispatch 层做减法：

| 模型 | 当前调用 | 改动 |
|------|----------|------|
| `paddleocr-vl` | `PaddleOCRVL(...)` | **不变**（已有专用分支） |
| `pp-ocrv6` | `PaddleOCR(text_detection_model_name=PP-OCRv6_..._det, ...)` | **不变**（去掉 pp-ocrv5 的内部分支） |
| `pp-structurev3` | `PPStructureV3(...)` | **不变**（已有专用分支） |
| `pp-chatocrv4` | `raise ValueError` | **删除整分支**（死代码） |
| `pp-ocrv5` | 共享 v6 分支，内部 if 设置 `PP-OCRv5_..._det/rec` | **删除 v5 fork**（`det_name`/`rec_name` 不再需要 if） |
| 兜底（未知 model） | 警告 + 走 PaddleOCR 默认 v6 medium | **保留**（防御性兜底） |

**为什么不用统一 `PaddleOCR(pipeline="...")` 高层 API？**

PaddleOCR 3.x 文档确实支持 `PaddleOCR(pipeline="PaddleOCR-VL")` 这种统一写法，但：
- 当前代码 3 个 pipeline 已用专用类（`PaddleOCRVL`、`PPStructureV3`、`PaddleOCR`）跑通生产，**改写调用层会引入新风险**
- 专用类能精准控制 PaddleX 底层参数（`enable_mkldnn`、`use_queues` 等）
- 用户没有要求重写架构，只要求「禁用 + 收窄」

**推荐 = A 方案（per-pipeline 保留，删 chatocrv4 死分支）。** 已选 B 候选方案，鉴于现状选 A。

### 改动清单

#### 1. `ocr_pdf.py`（核心）

| 行号 | 现状 | 改动 |
|------|------|------|
| **46-55** | `try: import chatocr_patch  # noqa: F401 ...` 含 4 行注释 + 6 行 import 块 | **删除整块**（H-3 import 现在无用） |
| **137** | `def __init__(self, output_dir, model='pp-ocrv6', ...)` | 默认改 `model='paddleocr-vl'` |
| **142-150** | docstring `model` 说明含 5 个值 | 删 `'pp-ocrv5'` 和 `'pp-chatocrv4'` 两行说明 |
| **200-220** | `if model == 'paddleocr-vl':` / `elif model == 'pp-structurev3':` | **不变** |
| **212-215** | `elif model == 'pp-chatocrv4': raise ValueError(...)` | **删除整 elif** |
| **216** | `elif model in ('pp-ocrv6', 'pp-ocrv5'):` | 改 `elif model == 'pp-ocrv6':`（v5 移除） |
| **230-238** | 内部 if 区分 v6/v5 设置 det_name/rec_name | **去掉内嵌 if**，直接 `det_name = f"PP-OCRv6_{model_size}_det"`，`rec_name = f"PP-OCRv6_{model_size}_rec"` |
| **247-254** | 兜底分支（未知 model） | **不变**（防御性保留） |
| **746** | `class PDFFileHandler(...): def __init__(self, ... model='pp-ocrv6' ...):` | 默认改 `'paddleocr-vl'` |
| **796** | `def run_manual_mode(... model='pp-ocrv6' ...):` | 默认改 `'paddleocr-vl'` |
| **847** | `def run_daemon_mode(... model='pp-ocrv6' ...):` | 默认改 `'paddleocr-vl'` |
| **892** | argparse `--model` `choices=['paddleocr-vl', 'pp-ocrv6', 'pp-ocrv5', 'pp-structurev3', 'pp-chatocrv4']` | choices 改 3 个：`['paddleocr-vl', 'pp-ocrv6', 'pp-structurev3']` |
| **893** | argparse `--model` `default='pp-ocrv6'` | default 改 `'paddleocr-vl'` |
| **894-898** | `--model` help 文本 | 删 v5/chatocrv4 描述，保留 3 模型说明 |

**保留不变：** `PaddleOCRVL`、`PPStructureV3`、`PaddleOCR` 三个 import 和所有参数（`use_queues=False`、`enable_mkldnn=False` 等）。

#### 2. `api.py`（同步）

| 行号 | 现状 | 改动 |
|------|------|------|
| **68-74** | `try: import chatocr_patch  # noqa: F401 ...` | **删除整块** |
| **203** | `models: ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]` | 改 `["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]` |
| **229** | `model: Optional[str] = Form(default="pp-ocrv6", ...)` | default 改 `"paddleocr-vl"`，description 改 3 模型 |
| **242** | docstring `model` 说明 | 改 3 模型 |
| **257** | `valid_models = ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]` | 改 3 模型 |

#### 3. `download_models.py`（同步）

| 行号 | 现状 | 改动 |
|------|------|------|
| **61** | `SUPPORTED_MODELS = ['pp-ocrv6', 'pp-ocrv5', 'pp-structurev3', 'paddleocr-vl']` | 删 `'pp-ocrv5'` |
| **117** | docstring `model_name` 说明 | 改 3 模型 |
| **122-123** | log 行里 `if model_name in ('pp-ocrv6', 'pp-ocrv5')` | 改 `if model_name == 'pp-ocrv6':` |
| **129** | `if model_name in ('pp-ocrv6', 'pp-ocrv5'):` | 改 `if model_name == 'pp-ocrv6':` |
| **132** | `prefix = 'PP-OCRv6' if model_name == 'pp-ocrv6' else 'PP-OCRv5'` | 简化 `prefix = 'PP-OCRv6'` |
| **133-134** | det_name / rec_name 拼接 | 不变（用 prefix 即可） |

#### 4. `tests/ocr_pdf_v6_test.py`（同步测试，10+ 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **145** | `def test_pp_ocrv5_uses_v5_model_names` docstring | **删除整个测试函数**（v5 不再支持） |
| **155** | `PDFOCRHandler(output_dir='test_out', model='pp-ocrv5', model_size='medium')` | **删除**（随函数删除） |
| **214-236** | `def test_cli_default_model_is_pp_ocrv6` 断言默认 `'pp-ocrv6'` | 改名为 `test_cli_default_model_is_paddleocr_vl`，断言默认 `'paddleocr-vl'` |
| 其他 pp-ocrv6 用例 | `text_detection_model_name=PP-OCRv6_*_det` 类 | **不变** |

**注：** `tests/ocr_pdf_v6_test.py` 中 `import argparse` 等 CLI mock 段（行 181-230）和 `lang`/`model-size` 测试（行 56-95）**与 model 字段无关，不动**。

#### 5. `tests/api_device_test.py`（同步测试，6 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **73** | `req_data = data or {"model": "pp-ocrv5"}` | 改 `{"model": "paddleocr-vl"}`（新默认） |
| **83** | `{"model": "pp-ocrv5", "device": "gpu"}` | 改 `{"model": "paddleocr-vl", ...}` |
| **92** | `{"model": "pp-ocrv5"}` | 改 `{"model": "paddleocr-vl"}` |
| **101** | `{"model": "pp-ocrv5", "device": "cpu"}` | 改 `{"model": "paddleocr-vl", ...}` |
| **110** | `{"model": "pp-ocrv5", "device": "mps"}` | 改 `{"model": "paddleocr-vl", ...}` |
| **118** | `{"model": "pp-ocrv5", "device": "auto"}` | 改 `{"model": "paddleocr-vl", ...}` |

#### 6. `tests/ocr_pdf_device_test.py`（同步测试，2 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **45** | `model='pp-ocrv5'` | 改 `model='paddleocr-vl'` |
| **77** | `model='pp-ocrv5'` | 改 `model='paddleocr-vl'` |

#### 7. `tests/download_models_v6_test.py`（同步测试，3 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **69-78** | `def test_pp_ocrv5_uses_v5_model_names` | **删除整个测试函数** |
| **85-91** | `def test_supported_models_contains_*` 含 4 个 assert `'pp-ocrv6' in SUPPORTED_MODELS` / `'pp-ocrv5' in SUPPORTED_MODELS` 等 | 改：移除 `'pp-ocrv5'`，添加 `'paddleocr-vl' in SUPPORTED_MODELS` 断言（已存在则保留） |

#### 8. `tests/download_models_device_test.py`（同步测试，1 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **31** | `result = download_model('pp-ocrv5')` | 改 `result = download_model('paddleocr-vl')` |

#### 9. `README.md`（用户文档同步，17 处引用）

grep 列出 17 处 `pp-ocrv5` / `pp-chatocrv4` 引用（行 162, 174, 177, 182, 193, 201, 215, 217, 221, 257, 265, 360, 411, 423, 455, 478, 486），全部要改：

| 区域 | 改动 |
|------|------|
| 「模型下载脚本」段（行 162-182）| 改文档/示例为 3 模型：删 `pp-ocrv5`，示例用 `paddleocr-vl` / `pp-ocrv6` |
| 「命令行参数」段（行 193, 201）| 改 `-model` choices 列表、help 说明、默认值说明 |
| 「模型选择说明」表格（行 215, 217）| **删 pp-ocrv5 行**（旧版 OCR），**删 pp-chatocrv4 行**（需 API key），**改 pp-ocrv5 行为 pp-ocrv6** |
| 「示例用法」段（行 257, 265, 360）| 改默认模型说明：`pp-ocrv5` → `paddleocr-vl`（新默认） |
| 「常见问题」第 2-3 条（行 478, 486）| 改模型速度/精度建议（删 `pp-ocrv5`、`pp-chatocrv4` 提及） |
| 「项目结构」段 | 删 `chatocr_patch.py` 行（已 git rm） |
| 「更新日志」段末 | 加 v1.0.3 条目：禁用 PP-ChatOCRv4、移除 PP-OCRv5、默认改 PaddleOCR-VL |

#### 10. `Docker.md`（用户文档同步，3 处引用）

| 行号 | 现状 | 改动 |
|------|------|------|
| **128** | `-F "model=pp-ocrv5"` curl 示例 | 改 `-F "model=paddleocr-vl"` |
| **134** | `-F "model=pp-ocrv5" \` | 改 `-F "model=paddleocr-vl" \` |
| **238** | `python download_models.py -m pp-ocrv5,paddleocr-vl` | 改 `python download_models.py -m paddleocr-vl,pp-ocrv6,pp-structurev3`（或简化为 `-m all`） |

#### 11. 删除 `chatocr_patch.py`

- `git rm chatocr_patch.py`（412 行 monkey-patch 工具，仅 chatocrv4 用到）
- 同步核查 `.dockerignore` / `.gitignore` / `Dockerfile` 是否含 `chatocr_patch.py` 引用（核查结果：**无**，`Dockerfile` 第 26 行 `COPY` 只列 `requirements.txt ocr_pdf.py api.py download_models.py`，未引用 chatocr_patch.py）

#### 12. **不需要改动的文件**（确认核查）

| 文件 | 原因 |
|------|------|
| `requirements.txt` | 9 行均为通用依赖（paddleocr/opencv/pypdfium2/watchdog/python-dotenv/fastapi/uvicorn/python-multipart/PyPDF2），**无 chatocr 专用依赖**（chatocr_patch 是 optional import） |
| `Dockerfile` | 第 26 行 `COPY` 不含 `chatocr_patch.py`；第 29 行 `pip install -r requirements.txt` 也不含 chatocr 依赖 |
| `device_utils.py` | 纯设备检测（GPU/CPU/MPS），无 model 名称引用 |

#### 13. **不修改的历史文件**（`thoughts/` 内）

- `thoughts/shared/audits/2026-06-21-ppocr-pdf-audit.md`：引用 chatocr 是历史审计记录，**不动**（文档化当时发现的问题）
- `thoughts/ledgers/CONTINUITY_ses_*.md`：连续性 ledger，引用 chatocr 是 session 当时的实际状态，**不动**（保留作为时间线）

### 数据流（变化后）

```
CLI 调用
  -model paddleocr-vl      ← 默认
  -model pp-ocrv6
  -model pp-structurev3
        │
        ▼
PDFOCRHandler.__init__(model='paddleocr-vl')
        │
        ▼
device_utils.detect_device('auto')
        │
        ▼
if model == 'paddleocr-vl':  PaddleOCRVL(...)
elif model == 'pp-ocrv6':   PaddleOCR(..., text_detection_model_name='PP-OCRv6_medium_det', ...)
elif model == 'pp-structurev3': PPStructureV3(...)
else: 警告 + PaddleOCR 默认 v6 medium   ← 防御性兜底
        │
        ▼
self.ocr.predict(img_cv)  （VL/Structure 用结构化输出，其他走通用 OCR）

API 调用
POST /ocr/pdf  model=paddleocr-vl
        │
        ▼
valid_models = ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]  ← 校验
        │
        ▼
get_handler(...)  → PDFOCRHandler 缓存命中/创建
        │
        ▼
同 CLI 路径
```

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户传 `model=pp-chatocrv4` 或 `model=pp-ocrv5` | argparse 拒绝（choices 不含） / API 返回 400（valid_models 不含） |
| 未知 model 走兜底分支 | logger.warning + 走 PP-OCRv6 medium 默认配置（保留防御性） |
| `paddleocr-vl` 缺少 GPU/CPU 包 | 错误冒泡到调用方（沿用现有 PaddleOCR 异常） |
| `chatocr_patch` 缺失 | 已被删除的 import 块不会再触发 ImportError（彻底解决） |

## 测试策略

**回归（必须通过）：**
- 49 现有测试保持绿
- 删除 2 个 `test_pp_ocrv5_uses_v5_model_names` 函数（`tests/ocr_pdf_v6_test.py:145` 和 `tests/download_models_v6_test.py:69`）
- 改写 1 个函数名 + 断言（`test_cli_default_model_is_pp_ocrv6` → `test_cli_default_model_is_paddleocr_vl`）
- 改 9 处硬编码 model 字段（`tests/api_device_test.py:73/83/92/101/110/118` + `tests/ocr_pdf_device_test.py:45/77` + `tests/download_models_device_test.py:31`）
- 改 `tests/download_models_v6_test.py:85-91` 的 SUPPORTED_MODELS 断言

**新增（确定 3 个回归测试，必加）：**
1. `test_valid_models_excludes_chatocrv4_and_ppocrv5`：断言 `valid_models == ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]`
2. `test_cli_default_model_is_paddleocr_vl`：替换原 `test_cli_default_model_is_pp_ocrv6`，断言 `args.model == 'paddleocr-vl'`
3. `test_chatocr_patch_file_removed`：检查 `chatocr_patch.py` 不在仓库（`not os.path.exists('chatocr_patch.py')` 或 `git ls-files` 不含）

**预期测试数：** 49 - 2 删除 + 3 新增 = 50/50 通过

**验证流程：**
1. `git rm chatocr_patch.py`
2. 改 `ocr_pdf.py` / `api.py` / `download_models.py`
3. 改 `tests/` 下 5 个测试文件（`ocr_pdf_v6_test.py`、`api_device_test.py`、`ocr_pdf_device_test.py`、`download_models_v6_test.py`、`download_models_device_test.py`）
4. 改 `README.md` / `Docker.md`
5. `pytest tests/ -v` → 期望 50/50 通过
6. `python -c "from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL; print('imports OK')"` → 期望成功
7. `python ocr_pdf.py --help` → 检查 `--model` choices 列表
8. `python api.py &` + `curl http://localhost:8000/health` → 检查 `models` 字段
9. `git diff --stat` 复核改动范围
10. `git grep -n "pp-ocrv5\|pp-chatocrv4\|chatocr_patch" -- ':!thoughts/' ':!*.md'` → 应返回空（除 `thoughts/` 历史记录和 README/Docker.md 中保留的更新日志）

## 开放问题（已全部解决）

| # | 问题 | 决策 |
|---|------|------|
| Q1 | 是否保留「未知 model 走 v6 medium」兜底分支？ | **保留**（防御性，避免 NoneType 崩溃） |
| Q2 | `requirements.txt` 是否补 `paddlex[ocr]`？ | **不动**（保持现状，运行时 import 提示） |
| Q3 | `Dockerfile` 是否含 `chatocr_patch.py` 引用？ | **否**（第 26 行 `COPY` 只列 4 个文件） |
| Q4 | `device_utils.py` 是否含 model 引用？ | **否**（纯设备检测） |
| Q5 | `tests/device_utils_test.py` 等 3 文件是否含 pp-ocrv5 引用？ | **是**（已列出：ocr_pdf_device_test.py:45/77, download_models_device_test.py:31, **device_utils_test.py:0 处**） |
| Q6 | `logs/chatocr_patch.log` 是否清理？ | **不处理**（gitignore 控制，不影响功能） |

## 风险评估

| 风险 | 等级 | 缓解 |
|------|------|------|
| 改动面广（10+ 文件），漏改导致 CI 失败 | 中 | planner 阶段全量 grep `pp-ocrv5` / `pp-chatocrv4` / `chatocr_patch` 引用清单 |
| 删除 chatocr_patch.py 后是否有 hidden import | 低 | 顶部 import 块也删，双保险；grep 全仓确认 |
| 默认 model 改 `paddleocr-vl` 性能/资源变化 | 中 | PaddleOCR-VL 1.6 是 SOTA 但需要更多显存；用户已知此风险，CPU 也能跑（耗时较长） |
| 测试覆盖率下降 | 低 | 49+N 测试保持，新增 3 个针对 3 模型的回归测试 |
| 现有用户调用脚本传 `pp-ocrv5` 突然失败 | 中 | README 明确标注为「破坏性变更」并在更新日志说明；用户主动要求，无 backward-compat 需求 |
