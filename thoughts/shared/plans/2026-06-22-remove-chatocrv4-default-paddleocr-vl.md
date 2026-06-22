# 禁用 PP-ChatOCRv4 + 移除 pp-ocrv5 + 默认 PaddleOCR-VL 实现计划

**Goal:** 将 ppocr_pdf 暴露的模型从 5 个收窄到 3 个（`paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`），默认模型从 `pp-ocrv6` 改为 `paddleocr-vl`，硬删除 pp-chatocrv4/pp-ocrv5 所有分支与依赖。

**Architecture:** 保持现有 per-pipeline 专用类架构（`PaddleOCRVL`、`PPStructureV3`、`PaddleOCR`），仅在 dispatch 层做减法：
- 删 `elif model == 'pp-chatocrv4': raise ValueError(...)` 分支
- 删 `elif model in ('pp-ocrv6', 'pp-ocrv5'):` 内部 v5 fork，改为 `elif model == 'pp-ocrv6':` + hardcode `PP-OCRv6_*` 前缀
- 保留 else 兜底分支（防御性，未知 model 走 PaddleOCR 默认 v6 medium）
- 删顶部 `try: import chatocr_patch` 块
- 删 `chatocr_patch.py` 整文件

**Design:** `thoughts/shared/designs/2026-06-22-remove-chatocrv4-default-paddleocr-vl-design.md`

---

## 全局约束

1. **3 模型硬编码：** 仅 `paddleocr-vl`、`pp-ocrv6`、`pp-structurev3` 出现在 choices/valid_models/SUPPORTED_MODELS
2. **默认模型 = `paddleocr-vl`**（不是配置项，是代码层 default）
3. **硬删除（不是配置 hide）：** pp-chatocrv4 和 pp-ocrv5 从所有 argparse choices、API valid_models、dispatch 分支、SUPPORTED_MODELS 移除
4. **`chatocr_patch.py` 整文件删除**（当前是 untracked 文件，`??` 状态）
5. **顶部 `try: import chatocr_patch` 块**从 `ocr_pdf.py` 和 `api.py` 两处删除
6. **测试计数：** 49 - 2 删除 + 3 新增 = **50/50**
7. **设备检测逻辑（`device_utils.py`）保持不变**
8. **不修改 PaddleOCR 调用参数**（`use_queues=False`、`enable_mkldnn=False` 等生产参数不变）
9. **不修改历史文件**（`thoughts/` 内 audit/ledger 不动）
10. **不修改 `requirements.txt`、`Dockerfile`、`device_utils.py`**（设计确认无引用）

---

## 执行顺序

```
Batch 1 (parallel — 5 implementers):  1.1  1.2  1.3  1.4  1.5   (测试文件修改)
Batch 2 (parallel — 3 implementers):  2.1  2.2  2.3              (源代码修改)
Batch 3 (parallel — 3 implementers):  3.1  3.2  3.3              (删除文件 + 文档)
Batch 4 (1 implementer):              4.1                         (最终验证)
```

**总 commit 数：** 4（1 test + 1 feat + 1 chore + 1 docs）

---

## Batch 1: 测试文件修改（parallel — 5 implementers）

所有任务修改不同的测试文件，可并行执行。**先改测试再改代码**（TDD：先验证新预期）。

### 提交标签：`test: update tests for 3-model only (drop pp-ocrv5/pp-chatocrv4)`

---

### Task 1.1: `tests/ocr_pdf_v6_test.py` — 删除 v5 测试 + 重命名默认模型测试

**File:** `tests/ocr_pdf_v6_test.py`
**Depends:** none

#### 步骤

##### Step 1: 删除 `test_pp_ocrv5_uses_v5_model_names` 函数（L144-159）

删除以下整段：

```python
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
```

**操作：** 删除 L144-159（共 16 行）。使用 Edit 工具，`oldString` 为完整函数体。

##### Step 2: 将 `test_cli_default_model_is_pp_ocrv6` 重命名为 `test_cli_default_model_is_paddleocr_vl` 并改断言

修改 L214-236。将：

```python
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
```

改为：

```python
    def test_cli_default_model_is_paddleocr_vl(self):
        """CLI 默认 -model 应该是 paddleocr-vl"""
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

        assert captured.get('model') == 'paddleocr-vl', f"Expected default 'paddleocr-vl', got {captured.get('model')}"
        assert captured.get('lang') == 'ch'
        assert captured.get('model_size') == 'medium'
```

**操作：** 用 Edit 替换 L214-236 整块。

##### Step 3: 验证

```powershell
conda activate ppocr
python -m pytest tests/ocr_pdf_v6_test.py -v
```

预期输出（注意 tests 总数为调整后值）：

```
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_default_uses_medium_chinese PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_small_model PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_tiny_model PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_with_english_lang PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_with_japanese_lang PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_pp_ocrv6_combined_lang_and_size PASSED
tests/ocr_pdf_v6_test.py::TestPPOCRv6ModelSelection::test_unknown_model_falls_back_to_v6_medium PASSED
tests/ocr_pdf_v6_test.py::TestCLIPv6Args::test_cli_has_lang_argument PASSED
tests/ocr_pdf_v6_test.py::TestCLIPv6Args::test_cli_has_model_size_argument PASSED
tests/ocr_pdf_v6_test.py::TestCLIPv6Args::test_cli_default_model_is_paddleocr_vl PASSED
```

注意：`test_paddleocr_vl_default` 在这一步可能会 FAIL（因为 `ocr_pdf.py` 还没改，还是 `default='pp-ocrv6'`）。**这符合 TDD：先写红测试，再改代码让它变绿。** 但 `test_pp_ocrv5_uses_v5_model_names` 已被删除所以不再有 FAIL。

---

### Task 1.2: `tests/api_device_test.py` — 改 6 处 model 字符串 + 加 2 个新回归测试

**File:** `tests/api_device_test.py`
**Depends:** none

#### 步骤

##### Step 1: 改默认 model 参数（L73）

将：

```python
                    req_data = data or {"model": "pp-ocrv5"}
```

改为：

```python
                    req_data = data or {"model": "paddleocr-vl"}
```

##### Step 2: 改 `test_device_param_accepted`（L83）

将：

```python
            client, tmp_path, {"model": "pp-ocrv5", "device": "gpu"}
```

改为：

```python
            client, tmp_path, {"model": "paddleocr-vl", "device": "gpu"}
```

##### Step 3: 改 `test_device_default_is_auto`（L92）

将：

```python
            client, tmp_path, {"model": "pp-ocrv5"}
```

改为：

```python
            client, tmp_path, {"model": "paddleocr-vl"}
```

##### Step 4: 改 `test_device_cpu_override`（L101）

将：

```python
            client, tmp_path, {"model": "pp-ocrv5", "device": "cpu"}
```

改为：

```python
            client, tmp_path, {"model": "paddleocr-vl", "device": "cpu"}
```

##### Step 5: 改 `test_device_invalid_rejected`（L110）

将：

```python
            client, tmp_path, {"model": "pp-ocrv5", "device": "mps"}
```

改为：

```python
            client, tmp_path, {"model": "paddleocr-vl", "device": "mps"}
```

##### Step 6: 改 `test_response_includes_device_and_device_info`（L118）

将：

```python
            client, tmp_path, {"model": "pp-ocrv5", "device": "auto"}
```

改为：

```python
            client, tmp_path, {"model": "paddleocr-vl", "device": "auto"}
```

##### Step 7: 添加新回归测试 `test_valid_models_excludes_chatocrv4_and_ppocrv5`

在 `test_device_endpoint`（L128-143）后面，`if __name__ == '__main__':` 之前添加：

```python
    def test_valid_models_excludes_chatocrv4_and_ppocrv5(self, client):
        """验证 api.valid_models 只含 3 模型，不含 pp-chatocrv4 和 pp-ocrv5"""
        from api import valid_models
        expected = ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]
        assert valid_models == expected, f"Expected {expected}, got {valid_models}"
```

注意：这个测试会 FAIL 直到 Task 2.2（api.py）修改完成。**TDD 红/绿周期。**

##### Step 8: 添加新回归测试 `test_chatocr_patch_file_removed`

```python
    def test_chatocr_patch_file_removed(self):
        """验证 chatocr_patch.py 已被删除（不在仓库）"""
        assert not os.path.exists('chatocr_patch.py'), \
            "chatocr_patch.py 应该被删除"
```

添加到同一个文件中，放在 `test_valid_models_excludes_chatocrv4_and_ppocrv5` 之后。

**注意：** 需要确保文件顶部已有 `import os`（L7 已有 `import os`）。这个测试会 FAIL 直到 Batch 3 删除了 `chatocr_patch.py`。

##### Step 9: 验证

```powershell
conda activate ppocr
python -m pytest tests/api_device_test.py -v
```

预期：除 `test_valid_models_*`（因 api.py 未改）和 `test_chatocr_patch_*`（因文件未删）外，其他 6 个 PASS。

---

### Task 1.3: `tests/ocr_pdf_device_test.py` — 改 2 处 model 字符串

**File:** `tests/ocr_pdf_device_test.py`
**Depends:** none

#### 步骤

##### Step 1: 改 L45

将：

```python
                model='pp-ocrv5',
```

改为：

```python
                model='paddleocr-vl',
```

##### Step 2: 改 L77

将：

```python
                model='pp-ocrv5',
```

改为：

```python
                model='paddleocr-vl',
```

##### Step 3: 验证

```powershell
conda activate ppocr
python -m pytest tests/ocr_pdf_device_test.py -v
```

预期：4 tests PASS（注意：这些 test 只是 mock 测试 model 参数传递到 PaddleOCR 构造函数，现在传 `paddleocr-vl` 会走到 PaddleOCRVL 分支而非 PaddleOCR 分支，因此 mock 会捕获 PaddleOCRVL 调用而非 PaddleOCR 调用。但测试本身 assert `call_kwargs.get('device')` — 两个分支都传 device 参数，所以应继续 PASS）。

等一下——实际上 `test_device_passed_to_paddleocr` 和 `test_device_cpu_override` 这两个测试 mock 的是 `PaddleOCR`，但传 `paddleocr-vl` 会走 `PaddleOCRVL` 分支（不调用 `PaddleOCR`）。所以 `mock_paddle_ocr.call_args[1]` 会获取不到参数，测试会 FAIL。

**修正方案：** 这两个测试目前 mock 的是 `ocr_pdf.PaddleOCR` 并 assert `device` 参数。但当我们传 `paddleocr-vl` 后，代码会走 `PaddleOCRVL` 分支而不是 `PaddleOCR` 分支。所以需要把这两个测试的 mock 目标从 `ocr_pdf.PaddleOCR` 改为 `ocr_pdf.PaddleOCRVL`。

**修改 L37-49（`test_device_passed_to_paddleocr`）：**

```python
    def test_device_passed_to_paddleocr_vl(self):
        """检测到的 paddlex_device 传入 PaddleOCRVL 构造函数"""
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
```

注意：这个测试现在和 L115-145 的 `test_device_passed_to_paddleocr_vl` 重复（都在测试 `model=paddleocr-vl` → `PaddleOCRVL` 分支）。由于旧 `test_device_passed_to_paddleocr` 测试了 `PaddleOCR` 分支（当 model=pp-ocrv6 时），而 `test_device_passed_to_paddleocr_vl` 测试了 `PaddleOCRVL` 分支，现在我们调整后两者都测试 PaddleOCRVL 分支。

**更好的方案：** 把 `test_device_passed_to_paddleocr` 改为测试 `pp-ocrv6` 分支（mock `PaddleOCR`），把 `test_device_cpu_override` 改为测试 `paddleocr-vl` 分支（mock `PaddleOCRVL`）。

**实际修改（L19-49）：**

将整个 `test_device_passed_to_paddleocr` 改为：

```python
    def test_device_passed_to_paddleocr_v6(self):
        """pp-ocrv6 将 paddlex_device 传入 PaddleOCR 构造函数"""
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
                model='pp-ocrv6',
                device='auto'
            )
            call_kwargs = mock_paddle_ocr.call_args[1]
            assert call_kwargs.get('device') == 'gpu:0'
```

**修改 `test_device_cpu_override`（L51-81）：**

将 mock 从 `ocr_pdf.PaddleOCR` 改为 `ocr_pdf.PaddleOCRVL`：

```python
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
            patch('ocr_pdf.PaddleOCRVL') as mock_vl,
            patch('ocr_pdf.logger'),
        ):
            handler = PDFOCRHandler(
                output_dir='test_out',
                model='paddleocr-vl',
                device='cpu'
            )
            call_kwargs = mock_vl.call_args[1]
            assert call_kwargs.get('device') == 'cpu'
```

##### Step 4: 验证

```powershell
conda activate ppocr
python -m pytest tests/ocr_pdf_device_test.py -v
```

预期：5 tests PASS（原有 5 个 test 都应通过——结构调整后各分支 mock 正确）。

---

### Task 1.4: `tests/download_models_v6_test.py` — 删除 v5 测试 + 更新 SUPPORTED_MODELS 断言

**File:** `tests/download_models_v6_test.py`
**Depends:** none

#### 步骤

##### Step 1: 删除 `test_pp_ocrv5_uses_v5_model_names`（L68-82）

删除整段：

```python
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
```

##### Step 2: 更新 `test_supported_models_includes_v6`（L84-91）

将：

```python
    def test_supported_models_includes_v6(self):
        """SUPPORTED_MODELS 列表应包含 pp-ocrv6"""
        from download_models import SUPPORTED_MODELS

        assert 'pp-ocrv6' in SUPPORTED_MODELS
        assert 'pp-ocrv5' in SUPPORTED_MODELS
        assert 'pp-structurev3' in SUPPORTED_MODELS
        assert 'paddleocr-vl' in SUPPORTED_MODELS
```

改为：

```python
    def test_supported_models_includes_correct_models(self):
        """SUPPORTED_MODELS 列表应包含 3 模型，不含 pp-ocrv5"""
        from download_models import SUPPORTED_MODELS

        assert SUPPORTED_MODELS == ['pp-ocrv6', 'pp-structurev3', 'paddleocr-vl'], \
            f"Got {SUPPORTED_MODELS}"
```

##### Step 3: 验证

```powershell
conda activate ppocr
python -m pytest tests/download_models_v6_test.py -v
```

预期：5 tests collected，其中 `test_supported_models_includes_correct_models` FAIL（因为 `download_models.py` 还没改，`SUPPORTED_MODELS` 仍含 `pp-ocrv5`）。**TDD 红/绿周期。**

---

### Task 1.5: `tests/download_models_device_test.py` — 改 1 处 model 字符串

**File:** `tests/download_models_device_test.py`
**Depends:** none

#### 步骤

##### Step 1: 改 L31

将：

```python
            result = download_model('pp-ocrv5')
```

改为：

```python
            result = download_model('paddleocr-vl')
```

同时更新测试 docstring（L20）：

将：

```python
        """PP-OCRv5 下载使用 device='cpu'"""
```

改为：

```python
        """PaddleOCR-VL 下载使用 device='cpu'"""
```

##### Step 2: 验证

```powershell
conda activate ppocr
python -m pytest tests/download_models_device_test.py -v
```

预期：3 tests PASS（`test_download_cpu_device_passed_to_paddleocr` 现在 mock `PaddleOCRVL`。注意──这个测试 mock 的依然是 `paddleocr.PaddleOCR`，但 `download_model('paddleocr-vl')` 会走 `PaddleOCRVL` 分支。所以需要改 mock 对象。）

**修正：** 改 L24-31 mock 从 `paddleocr.PaddleOCR` 改为 `paddleocr.PaddleOCRVL`：

```python
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
```

验证：

```powershell
conda activate ppocr
python -m pytest tests/download_models_device_test.py -v
```

预期：3 tests PASS。

---

## Batch 2: 源代码修改（parallel — 3 implementers）

所有任务修改不同的源文件，可并行执行。

### 提交标签：`feat: remove pp-chatocrv4 and pp-ocrv5, default to paddleocr-vl`

---

### Task 2.1: `ocr_pdf.py` — 核心改动（import 块 + 默认值 + dispatch + argparse）

**File:** `ocr_pdf.py`
**Depends:** Batch 1（测试已写）

#### 步骤

##### Step 1: 删除 `chatocr_patch` import 块（L50-57）

删除以下 8 行：

```python
# PP-ChatOCRv4 运行时补丁：修复 LLM JSON 数组/裸字符串解析 + 注入 few-shot 示例
# 无论当前是否使用 pp-chatocrv4 模型，提前 import 确保补丁就位
try:
    import chatocr_patch  # noqa: F401  (patches apply on import)

    chatocr_patch  # silence linter
except ImportError:
    pass
```

**操作：** 用 Edit 工具，`oldString` 为上述完整块。

##### Step 2: 更新模块 docstring（L16-21，去掉已移除模型引用）

将：

```python
# 3. 多模型切换（PP-OCRv5 / PP-StructureV3 / PP-ChatOCRv4 / PaddleOCR-VL）
```

改为：

```python
# 3. 多模型切换（PaddleOCR-VL / PP-OCRv6 / PP-StructureV3）
```

##### Step 3: 改 `PDFOCRHandler.__init__` 默认值（L137）

将：

```python
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
```

改为：

```python
    def __init__(self, output_dir, model='paddleocr-vl', device='auto',
```

##### Step 4: 更新 docstring `model` 说明（L146-150）

将：

```python
                - 'paddleocr-vl': 0.9B 视觉语言多模态模型
                - 'pp-ocrv6': PP-OCRv6 通用文字识别 (默认，50 种语言)
                - 'pp-ocrv5': PP-OCRv5 旧版通用文字识别
                - 'pp-structurev3': 复杂文档结构化解析
                - 'pp-chatocrv4': 信息抽取 (需 API key)
```

改为：

```python
                - 'paddleocr-vl': 0.9B 视觉语言多模态模型 (默认)
                - 'pp-ocrv6': PP-OCRv6 通用文字识别 (50 种语言)
                - 'pp-structurev3': 复杂文档结构化解析
```

##### Step 5: 删除 `elif model == 'pp-chatocrv4'` 分支（L212-215）

删除以下 4 行：

```python
        elif model == 'pp-chatocrv4':
            # PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用
            logger.error(f"PP-ChatOCRv4模型需要额外的API配置，暂不支持直接使用")
            raise ValueError(f"{model}模型需要额外的API配置，暂不支持直接使用")
```

##### Step 6: 改 `elif model in ('pp-ocrv6', 'pp-ocrv5'):`（L216）

将：

```python
        elif model in ('pp-ocrv6', 'pp-ocrv5'):
```

改为：

```python
        elif model == 'pp-ocrv6':
```

##### Step 7: 简化内嵌 if/else，去掉 v5 fork（L221-227）

将：

```python
            if model == 'pp-ocrv6':
                det_name = f"PP-OCRv6_{model_size}_det"
                rec_name = f"PP-OCRv6_{model_size}_rec"
            else:
                # 旧版 v5 仍保留兼容入口
                det_name = f"PP-OCRv5_{model_size}_det"
                rec_name = f"PP-OCRv5_{model_size}_rec"
```

改为：

```python
            det_name = f"PP-OCRv6_{model_size}_det"
            rec_name = f"PP-OCRv6_{model_size}_rec"
```

##### Step 8: 改 `PDFFileHandler` 默认值（L746）

将：

```python
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
```

改为：

```python
    def __init__(self, output_dir, model='paddleocr-vl', device='auto',
```

##### Step 9: 改 `run_manual_mode` 默认值（L796）

将：

```python
def run_manual_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium', optimize_pdf=False, optimize_level='medium', grayscale=False):
```

改为：

```python
def run_manual_mode(input_dir, output_dir, model='paddleocr-vl', device='auto', lang='ch', model_size='medium', optimize_pdf=False, optimize_level='medium', grayscale=False):
```

##### Step 10: 改 `run_daemon_mode` 默认值（L847）

将：

```python
def run_daemon_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium', optimize_pdf=False, optimize_level='medium', grayscale=False):
```

改为：

```python
def run_daemon_mode(input_dir, output_dir, model='paddleocr-vl', device='auto', lang='ch', model_size='medium', optimize_pdf=False, optimize_level='medium', grayscale=False):
```

##### Step 11: 改 argparse `--model` choices（L892）

将：

```python
                       choices=['paddleocr-vl', 'pp-ocrv6', 'pp-ocrv5', 'pp-structurev3', 'pp-chatocrv4'],
```

改为：

```python
                       choices=['paddleocr-vl', 'pp-ocrv6', 'pp-structurev3'],
```

##### Step 12: 改 argparse `--model` default（L893）

将：

```python
                       default='pp-ocrv6',
```

改为：

```python
                       default='paddleocr-vl',
```

##### Step 13: 改 `--model` help 文本（L894-898）

将：

```python
                       help='OCR模型选择: paddleocr-vl (多模态文档解析) / '
                            'pp-ocrv6 (PP-OCRv6 通用文字识别, 50 种语言, 默认) / '
                            'pp-ocrv5 (旧版) / '
                            'pp-structurev3 (复杂文档解析) / '
                            'pp-chatocrv4 (智能信息抽取, 需 API key)')
```

改为：

```python
                       help='OCR模型选择: paddleocr-vl (多模态文档解析, 默认) / '
                            'pp-ocrv6 (PP-OCRv6 通用文字识别, 50 种语言) / '
                            'pp-structurev3 (复杂文档解析)')
```

##### Step 14: [可选] 更新结果解析区注释（L581）

将：

```python
                            # 处理PP-OCRv5模型的输出格式
```

改为：

```python
                            # 处理PP-OCRv6模型的输出格式（通用PaddleOCR返回）
```

此步是 cosmetic，不影响功能。

##### Step 15: 验证

```powershell
conda activate ppocr
python -m pytest tests/ocr_pdf_v6_test.py -v
```

预期：10 tests all PASS（含重命名后的 `test_cli_default_model_is_paddleocr_vl`）。

```powershell
python -m pytest tests/ocr_pdf_device_test.py -v
```

预期：5 tests PASS。

---

### Task 2.2: `api.py` — import 块 + 模型列表 + 默认值 + docstring

**File:** `api.py`
**Depends:** Batch 1（测试已写）

#### 步骤

##### Step 1: 删除 `chatocr_patch` import 块（L68-74）

删除以下 7 行：

```python
# PP-ChatOCRv4 运行时补丁：修复 LLM JSON 数组/裸字符串解析 + 注入 few-shot 示例
try:
    import chatocr_patch  # noqa: F401  (patches apply on import)

    chatocr_patch
except ImportError:
    pass
```

##### Step 2: 改 `/health` 响应 models 列表（L203）

将：

```python
        "models": ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"],
```

改为：

```python
        "models": ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"],
```

##### Step 3: 改 Form 默认值 + description（L229）

将：

```python
    model: Optional[str] = Form(default="pp-ocrv6", description="OCR模型选择: pp-ocrv6 (默认), pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4"),
```

改为：

```python
    model: Optional[str] = Form(default="paddleocr-vl", description="OCR模型选择: paddleocr-vl (默认), pp-ocrv6, pp-structurev3"),
```

##### Step 4: 改 docstring model 说明（L242）

将：

```python
        model: OCR模型选择，可选值: pp-ocrv6, pp-ocrv5, pp-structurev3, paddleocr-vl, pp-chatocrv4
```

改为：

```python
        model: OCR模型选择，可选值: paddleocr-vl, pp-ocrv6, pp-structurev3
```

##### Step 5: 改 `valid_models` 列表（L257）

将：

```python
    valid_models = ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"]
```

改为：

```python
    valid_models = ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"]
```

##### Step 6: 验证

```powershell
conda activate ppocr
python -m pytest tests/api_device_test.py -v
```

预期：原先 FAIL 的 `test_valid_models_excludes_chatocrv4_and_ppocrv5` 现在 PASS。共 9 tests（7 原有 + 2 新增）全部 PASS。

---

### Task 2.3: `download_models.py` — SUPPORTED_MODELS + dispatch 简化

**File:** `download_models.py`
**Depends:** Batch 1（测试已写）

#### 步骤

##### Step 1: 改模块 docstring（L18-21）

将：

```python
#   - pp-ocrv6: PP-OCRv6 通用文字识别 (PaddleOCR 3.7+ 默认, 50 种语言)
#   - pp-ocrv5: PP-OCRv5 旧版通用文字识别
#   - pp-structurev3: PP-StructureV3 复杂文档结构化解析
#   - paddleocr-vl: PaddleOCR-VL 多模态文档解析
```

改为：

```python
#   - pp-ocrv6: PP-OCRv6 通用文字识别 (PaddleOCR 3.7+ 默认, 50 种语言)
#   - pp-structurev3: PP-StructureV3 复杂文档结构化解析
#   - paddleocr-vl: PaddleOCR-VL 多模态文档解析
```

##### Step 2: 改 `SUPPORTED_MODELS`（L61）

将：

```python
SUPPORTED_MODELS = ['pp-ocrv6', 'pp-ocrv5', 'pp-structurev3', 'paddleocr-vl']
```

改为：

```python
SUPPORTED_MODELS = ['pp-ocrv6', 'pp-structurev3', 'paddleocr-vl']
```

##### Step 3: 改 `download_model()` docstring（L117）

将：

```python
        model_name (str): 模型名称 (pp-ocrv6 / pp-ocrv5 / pp-structurev3 / paddleocr-vl)
```

改为：

```python
        model_name (str): 模型名称 (pp-ocrv6 / pp-structurev3 / paddleocr-vl)
```

##### Step 4: 改 log 行 `model_name in ('pp-ocrv6', 'pp-ocrv5')`（L122-123）

将：

```python
    logger.info(f"开始下载 {model_name} 模型"
                f"{f' ({model_size})' if model_name in ('pp-ocrv6', 'pp-ocrv5') else ''}"
                f"{f' lang={lang}' if model_name in ('pp-ocrv6', 'pp-ocrv5') else ''}...")
```

改为：

```python
    logger.info(f"开始下载 {model_name} 模型"
                f"{f' ({model_size})' if model_name == 'pp-ocrv6' else ''}"
                f"{f' lang={lang}' if model_name == 'pp-ocrv6' else ''}...")
```

##### Step 5: 改条件 `model_name in ('pp-ocrv6', 'pp-ocrv5')`（L129）

将：

```python
        if model_name in ('pp-ocrv6', 'pp-ocrv5'):
```

改为：

```python
        if model_name == 'pp-ocrv6':
```

##### Step 6: 简化 `prefix` 赋值（L132）

将：

```python
            prefix = 'PP-OCRv6' if model_name == 'pp-ocrv6' else 'PP-OCRv5'
```

改为：

```python
            prefix = 'PP-OCRv6'
```

##### Step 7: 验证

```powershell
conda activate ppocr
python -m pytest tests/download_models_v6_test.py -v
python -m pytest tests/download_models_device_test.py -v
```

预期：`test_supported_models_includes_correct_models` 现在 PASS。两台共 8 tests PASS。

---

## Batch 3: 删除文件 + 文档同步（parallel — 3 implementers）

### 提交标签：`chore: remove chatocr_patch.py` + `docs: sync README and Docker.md for 3-model only`

---

### Task 3.1: 删除 `chatocr_patch.py`

**File:** (删除) `chatocr_patch.py`
**Depends:** Batch 2（import 块已删除，无遗留引用）

#### 步骤

##### Step 1: 删除文件

```powershell
Remove-Item chatocr_patch.py
```

##### Step 2: 验证删除

```powershell
Test-Path chatocr_patch.py
# Expected: False
```

##### Step 3: 验证测试

```powershell
conda activate ppocr
python -m pytest tests/api_device_test.py::TestAPIDevice::test_chatocr_patch_file_removed -v
```

预期：PASS（现在 `os.path.exists('chatocr_patch.py')` 返回 False）。

---

### Task 3.2: `README.md` — 同步 17+ 处引用

**File:** `README.md`
**Depends:** none（纯文档）

**设计确认的 17 处引用（以实际 grep 为准）：** 以下列出实际文件中出现的精确位置。

#### 步骤

##### Step 1: 改「模型下载脚本」段
- 删除 `pp-ocrv5` 从所有列举处
- 更新示例命令

**位置：** "支持下载三种PaddleOCR模型：`pp-ocrv5`、`pp-structurev3`、`paddleocr-vl`"

将：

```
- 支持下载三种PaddleOCR模型：`pp-ocrv5`、`pp-structurev3`、`paddleocr-vl`
```

改为：

```
- 支持下载三种PaddleOCR模型：`paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`
```

**位置：** 示例命令

将：

```bash
# 下载指定模型到自定义目录
python download_models.py -m pp-ocrv5,paddleocr-vl -o ./models

# 下载单个模型
python download_models.py -m pp-ocrv5
```

改为：

```bash
# 下载指定模型到自定义目录
python download_models.py -m paddleocr-vl,pp-ocrv6 -o ./models

# 下载单个模型
python download_models.py -m paddleocr-vl
```

**位置：** `-m, --models` 参数说明

将：

```
可选值: `pp-ocrv5, pp-structurev3, paddleocr-vl, all`
```

改为：

```
可选值: `paddleocr-vl, pp-ocrv6, pp-structurev3, all`
```

##### Step 2: 改「命令行参数」段
**位置：** `-model` choices 和默认值

将：

```
python ocr_pdf.py [-h] -i INPUT -o OUTPUT [-m {manual,daemon}] [-model {paddleocr-vl,pp-ocrv5,pp-structurev3,pp-chatocrv4}] [-l {debug,info,warning,error,critical}] [--optimize-pdf] [--optimize-level {low,medium,high}] [--grayscale]
```

改为：

```
python ocr_pdf.py [-h] -i INPUT -o OUTPUT [-m {manual,daemon}] [-model {paddleocr-vl,pp-ocrv6,pp-structurev3}] [-l {debug,info,warning,error,critical}] [--optimize-pdf] [--optimize-level {low,medium,high}] [--grayscale]
```

将参数说明中：

```
- `-model, --model`: OCR模型选择，可选值：paddleocr-vl、pp-ocrv5、pp-structurev3、pp-chatocrv4，默认：pp-ocrv5
```

改为：

```
- `-model, --model`: OCR模型选择，可选值：paddleocr-vl、pp-ocrv6、pp-structurev3，默认：paddleocr-vl
```

##### Step 3: 改「模型选择说明」表格
**位置：** 模型表格（当前含 4 行）

将：

```
| paddleocr-vl   | 多模态模型   | ... | ... | ✅ 支持                      |
| pp-ocrv5       | 全场景识别   | ... | ... | ✅ 支持                      |
| pp-structurev3 | 复杂文档解析 | ... | ... | ✅ 支持（需额外依赖）        |
| pp-chatocrv4   | 智能信息抽取 | ... | ... | ⚠️ 需API密钥，暂不直接支持 |
```

改为（删 pp-ocrv5 行 + pp-chatocrv4 行，加 pp-ocrv6 行）：

```
| 模型名称       | 模型类型     | 特点                                                                      | 适用场景                                     | 支持状态              |
| -------------- | ------------ | ------------------------------------------------------------------------- | -------------------------------------------- | --------------------- |
| paddleocr-vl   | 多模态模型   | 通过0.9B超紧凑视觉语言模型增强，支持109种语言，在复杂元素识别方面表现出色 | 多语言混合文档、包含表格/公式/图表的复杂文档 | ✅ 支持               |
| pp-ocrv6       | 通用文字识别 | PP-OCRv6 通用文字识别，支持50种语言，检测+识别一体化，识别精度高          | 普通文档识别、日常使用                       | ✅ 支持               |
| pp-structurev3 | 复杂文档解析 | 将复杂PDF转换为保留原始结构的Markdown和JSON文件，保持文档版式和层次结构   | 结构化文档处理、需要保留格式的文档           | ✅ 支持（需额外依赖） |
```

更新表格下方的注意：

将：

```
**注意：**
- pp-structurev3模型需要安装额外依赖：`pip install "paddlex[ocr]"`
- pp-chatocrv4模型需要配置百度千帆API密钥，目前暂不直接支持在本程序中使用
```

改为：

```
**注意：**
- pp-structurev3模型需要安装额外依赖：`pip install "paddlex[ocr]"`
```

##### Step 4: 改「示例用法」段
**位置：** "使用默认模型(pp-ocrv5)监控目录"

将：

```
# 使用默认模型(pp-ocrv5)监控目录
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon
```

改为：

```
# 使用默认模型(paddleocr-vl)监控目录
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon
```

**位置：** 其他引用 `pp-ocrv5` 的注释和说明

将：

```
# pp-chatocrv4模型需要配置API密钥，目前暂不直接支持
```

改为：

```
# pp-ocrv6是通用OCR模型，适用于日常文档识别
```

（或者删除该行——因为它引用了已删除的模型。最安全的做法是删掉该注释行。）

##### Step 5: 改「API服务」段
**位置：** POST /ocr/pdf 参数说明

将：

```
- `model`：OCR模型选择（可选，默认：pp-ocrv5），可选值：pp-ocrv5, pp-structurev3, paddleocr-vl
```

改为：

```
- `model`：OCR模型选择（可选，默认：paddleocr-vl），可选值：paddleocr-vl, pp-ocrv6, pp-structurev3
```

##### Step 6: 改「常见问题」段
**位置：** FAQ 第 2 条「识别速度慢」

将：

```
- 选择资源消耗较低的模型（如pp-ocrv5）
```

改为：

```
- 选择资源消耗较低的模型（如pp-ocrv6）
```

**位置：** FAQ 第 3 条「识别准确率低」

将：

```
- 选择适合的模型（如paddleocr-vl适合复杂文档，pp-chatocrv4适合信息抽取）
```

改为：

```
- 选择适合的模型（如paddleocr-vl适合复杂文档，pp-structurev3适合结构化文档）
```

##### Step 7: 改「配置方法」段
**位置：** `def __init__(self, output_dir, model='pp-ocrv5'):`

将：

```python
    def __init__(self, output_dir, model='pp-ocrv5'):
```

改为：

```python
    def __init__(self, output_dir, model='paddleocr-vl'):
```

将配置代码示例中引用 `pp-ocrv5` 的部分去掉（配置段中是示例代码，含 v5 相关代码）。

**注意：** 配置段中的示例代码是教学性代码，与运行代码无关。更新默认值即可。

##### Step 8: 改「项目结构」段
**位置：** 项目结构图不列 chatocr_patch.py（当前已不列，不需要改）。但如果设计有要求才改。

##### Step 9: 加「更新日志」条目
在 `v1.0.2` 前面加：

```
### v1.0.3 (2026-06-22)

- **破坏性变更：** 禁用 PP-ChatOCRv4（硬删除），移除 PP-OCRv5
- 默认模型从 `pp-ocrv6` 改为 `paddleocr-vl`
- 支持的模型收窄到 3 个：`paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`
- 删除 `chatocr_patch.py` 和相关 import 块
- CLI `--model` 和 API `model` 参数仅接受上述 3 个值
```

---

### Task 3.3: `Docker.md` — 同步 3 处引用

**File:** `Docker.md`
**Depends:** none（纯文档）

#### 步骤

##### Step 1: 改 curl 示例（L128）

将：

```
  -F "model=pp-ocrv5"
```

改为：

```
  -F "model=paddleocr-vl"
```

##### Step 2: 改 curl 示例（L134）

将：

```
  -F "model=pp-ocrv5" \
```

改为：

```
  -F "model=paddleocr-vl" \
```

##### Step 3: 改模型下载示例（L238）

将：

```
  python download_models.py -m pp-ocrv5,paddleocr-vl
```

改为：

```
  python download_models.py -m paddleocr-vl,pp-ocrv6,pp-structurev3
```

---

## Batch 4: 最终验证

### Task 4.1: 全量验证

**Depends:** Batch 1 + Batch 2 + Batch 3

#### 步骤

##### Step 1: 运行完整测试套件

```powershell
conda activate ppocr
python -m pytest tests/ -v
```

**预期输出：** 50/50 tests passed（49 - 2 删除 + 3 新增 = 50）

##### Step 2: 验证 import

```powershell
conda activate ppocr
python -c "from paddleocr import PaddleOCR, PPStructureV3, PaddleOCRVL; print('imports OK')"
```

**预期：** `imports OK`

##### Step 3: 验证 CLI --help

```powershell
conda activate ppocr
python ocr_pdf.py --help
```

**预期输出片段：**

```
  -model, --model      OCR模型选择: paddleocr-vl (多模态文档解析, 默认) / pp-ocrv6 (PP-OCRv6 通用文字识别, 50 种语言) / pp-structurev3 (复杂文档解析)
```

确认 choices 只含 `paddleocr-vl`、`pp-ocrv6`、`pp-structurev3`。

##### Step 4: 验证 API health endpoint

```powershell
conda activate ppocr
python -c "
from api import app
from fastapi.testclient import TestClient
client = TestClient(app)
resp = client.get('/health')
data = resp.json()
assert data['models'] == ['paddleocr-vl', 'pp-ocrv6', 'pp-structurev3'], f'Got {data[\"models\"]}'
print('health check OK:', data['models'])
"
```

**预期：** `health check OK: ['paddleocr-vl', 'pp-ocrv6', 'pp-structurev3']`

##### Step 5: 全仓 grep 确认无残留

```powershell
git grep -n "pp-ocrv5\|pp-chatocrv4\|chatocr_patch" -- ':!thoughts/' ':!*.md'
```

**预期：** 空（无输出）。`thoughts/` 和 `*.md` 被排除（历史记录和更新日志可保留引用）。

---

## 分步提交清单

| # | 提交信息 | 包含任务 | 文件数 |
|---|---------|---------|--------|
| 1 | `test: update tests for 3-model only (drop pp-ocrv5/pp-chatocrv4)` | Batch 1 (1.1-1.5) | 5 test files |
| 2 | `feat: remove pp-chatocrv4 and pp-ocrv5, default to paddleocr-vl` | Batch 2 (2.1-2.3) | 3 source files |
| 3 | `chore: remove chatocr_patch.py` | Task 3.1 | 1 file deleted |
| 4 | `docs: sync README and Docker.md for 3-model only` | Tasks 3.2, 3.3 | 2 doc files |

**注意：** 如果你偏好 squash，可以合并为 1 个 commit。按设计验证流程，推荐 4 个独立 commit 以便回滚时精确定位。

---

## 自检清单

- [ ] **Spec coverage:** 所有设计清单的改动项都已映射到具体 Task
- [ ] **无占位符：** 每个代码示例都是完整可执行的
- [ ] **行号准确：** 所有行号基于实际文件验证（2026-06-22 当前版本）
- [ ] **类型一致：** 函数签名、变量名跨 Task 一致
- [ ] **测试计数：** 49 - 2 + 3 = 50
- [ ] **依赖正确：** Batch 1（改测试）→ Batch 2（改代码）→ Batch 3（删文件+文档）→ Batch 4（验证）
- [ ] **git rm 确认：** `chatocr_patch.py` 是 untracked 文件，使用 `Remove-Item` 而非 `git rm`
- [ ] **排除文件确认：** `requirements.txt`、`Dockerfile`、`device_utils.py`、`thoughts/` 下的历史文件——设计确认不改
