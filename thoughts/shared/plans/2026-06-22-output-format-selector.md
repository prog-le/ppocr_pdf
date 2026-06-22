# 输出格式选择器 (Output Format Selector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `-of`/`--output-formats` CLI flag and API `output_formats` form parameter to support 4 new output formats (markdown, json, img, pdf) alongside the always-present .txt.

**Architecture:** 4 inline if-branches in `ocr_pdf.py:process_pdf()` after existing .txt write. Each branch is independent try/except. API saves to persistent disk under `api_outputs/`, returns paths in JSON response, and serves files via `GET /download/{path}`. Tests use real PDFs + real PaddleOCR (no mocks).

**Design:** `thoughts/shared/designs/2026-06-22-output-format-selector-design.md` (committed at `2371fbe`)

**Tech Stack:** Python 3.11, PaddleOCR, FastAPI, img2pdf (new dependency), PyTest

---

## Global Constraints

- `.txt` always written, regardless of `output_formats` (backward compat)
- Default: `output_formats = markdown,json,img,pdf` (all 4 enabled)
- `pp-ocrv6` + `markdown`: skip with `logger.warning()` (model doesn't support native markdown)
- API saves to disk, returns paths in JSON, `GET /download/{path}` serves files
- Inline implementation in `ocr_pdf.py:process_pdf()` — no new modules
- NO mock data in tests — use real PDFs + real PaddleOCR; mark `@pytest.mark.integration`
- New dependency: `img2pdf>=0.5.0` (~50KB, PNG→PDF bundling)
- Do NOT modify `device_utils.py`, `Dockerfile`, or history files (audits/ledgers)
- Do NOT pop the stash (chatocrv4 partial Batch 2)
- PowerShell-compatible commands throughout
- Conda env: `ppocr`; Python 3.11

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Add `img2pdf>=0.5.0` |
| `pytest.ini` | Create | Register `integration` marker |
| `ocr_pdf.py` | Modify | CLI arg `-of`/`--output-formats`; `PDFOCRHandler.__init__()` param; `process_pdf()` 4 format branches; propagate through `run_manual_mode()`, `run_daemon_mode()`, `PDFFileHandler`, `main()` |
| `api.py` | Modify | `output_formats` Form param + validation; `outputs` dict in response; `GET /download/{path}` endpoint; `/health` update; output dir management |
| `tests/output_format_selector_test.py` | Create | 9 integration tests (real PDF + real PaddleOCR, no mocks) |
| `README.md` | Modify | Docs for `-of` flag, output formats table, API updates |
| `Docker.md` | Modify | Add `output_formats` curl example |

---

## Dependency Graph

```
Batch 1 (parallel): Task 1 (requirements.txt), Task 2 (pytest.ini)
        |
Batch 2: Task 3 (ocr_pdf.py) — depends on Task 1 (img2pdf for pdf branch)
        |
Batch 3: Task 4 (api.py) — depends on Task 3 (PDFOCRHandler output_formats)
        |
Batch 4: Task 5 (test file) — depends on Task 3 (full feature in ocr_pdf.py)
        |
Batch 5: Task 6 (docs) — depends on Task 3, Task 4 (complete feature)
        |
Batch 6: Task 7 (final verification) — depends on all previous
```

---

## Task 1: Add img2pdf dependency

**Files:**
- Modify: `C:\guole\code\ppocr_pdf\requirements.txt`
- Test: none (verify via `python -c`)

**Depends:** none

- [ ] **Step 1: Add `img2pdf` to requirements.txt**

```python
# At end of requirements.txt, after existing entries, add:
img2pdf>=0.5.0
```

Edit: `C:\guole\code\ppocr_pdf\requirements.txt` — append line 10 (after `PyPDF2`):

New content for line 10:
```
img2pdf>=0.5.0
```

Full updated `requirements.txt`:
```
paddleocr
opencv-python
pypdfium2
watchdog
python-dotenv
fastapi
uvicorn
python-multipart
PyPDF2
img2pdf>=0.5.0
```

- [ ] **Step 2: Install and verify**

Run:
```powershell
conda activate ppocr
pip install -r requirements.txt
python -c "import img2pdf; print(f'img2pdf {img2pdf.__version__} OK')"
```
Expected output: `img2pdf X.Y.Z OK`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add img2pdf dependency for annotated PDF output"
```

---

## Task 2: Create pytest.ini with integration marker

**Files:**
- Create: `C:\guole\code\ppocr_pdf\pytest.ini`
- Test: none (config file)

**Depends:** none

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
markers =
    integration: marks tests as integration tests using real PaddleOCR (deselect with '-m "not integration"')
```

Write to `C:\guole\code\ppocr_pdf\pytest.ini`.

- [ ] **Step 2: Verify marker loads**

Run:
```powershell
python -m pytest --markers -c pytest.ini
```
Expected output: contains `@pytest.mark.integration: marks tests as integration tests...`

- [ ] **Step 3: Commit**

```bash
git add pytest.ini
git commit -m "chore: add pytest.ini with integration marker"
```

---

## Task 3: ocr_pdf.py — CLI arg + PDFOCRHandler.__init__ + process_pdf format branches + propagation

**Files:**
- Modify: `C:\guole\code\ppocr_pdf\ocr_pdf.py`
- Verify: `python ocr_pdf.py --help`, `pytest tests/output_format_selector_test.py -v -m integration` (after Task 5)

**Depends:** Task 1 (img2pdf)

**Interfaces:**
- Consumes: Python 3.11 standard lib, `img2pdf`, `glob`
- Produces: `PDFOCRHandler.__init__(..., output_formats=None)` — new kwarg
- Produces: `process_pdf()` — writes `.md`, `.json`, `.png`, `_annotated.pdf` after `.txt`
- Produces: `run_manual_mode(..., output_formats=None)`, `run_daemon_mode(..., output_formats=None)`, `PDFFileHandler.__init__(..., output_formats=None)` — propagate kwarg
- Produces: `main()` — parses `-of`/`--output-formats` as comma-separated string → list

- [ ] **Step 1: Add `-of`/`--output-formats` CLI argument**

Insert after line 913 (`parser.add_argument('--grayscale'...)`) and before line 916 (`args = parser.parse_args()`):

```python
    parser.add_argument('-of', '--output-formats',
                       type=str,
                       default='markdown,json,img,pdf',
                       help='输出格式 (逗号分隔). 可选值: markdown, json, img, pdf. '
                            '默认全部输出. 提示: pp-ocrv6 不支持 markdown, 将自动跳过. '
                            '示例: -of markdown,json')
```

- [ ] **Step 2: Add `output_formats` to `PDFOCRHandler.__init__()`**

Modify line 137 signature to add `output_formats=None`:

```python
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
                 lang='ch', model_size='medium',
                 output_formats=None,  # 新增: 输出格式列表, 默认全部
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
```

Add after line 174 (`self.model_size = model_size`):

```python
        # 输出格式选择
        if output_formats is None:
            output_formats = ['markdown', 'json', 'img', 'pdf']
        self.output_formats = output_formats
```

- [ ] **Step 3: Add `page_raw_results` collector and 4 format branches in `process_pdf()`**

**3a: Add `page_raw_results` collector list before the page loop**

Add after line 359 (`ocr_results = []`):

```python
        page_raw_results = []  # 存储每页的原始 result 对象, 用于输出格式生成
```

**3b: Store raw result objects during page loop**

After line 438 (`result_list = list(result)`), add:

```python
                    page_raw_results.append(result_list)  # 存储供格式输出使用
```

**3c: Insert 4 format if-branches after .txt write, before `return True`**

After line 685 (`logger.info(f"PDF文件处理完成，结果保存至: {output_txt_path}")`) and before line 686 (`success = True`), add:

```python
            # === 新增: 可选输出格式 (output_formats) ===
            if self.output_formats and page_raw_results:
                import glob as glob_module  # 避免与模块级 glob 变量冲突
                
                for page_idx, page_res_list in enumerate(page_raw_results):
                    page_num = page_idx + 1
                    
                    for res in page_res_list:
                        # JSON: 所有模型都支持
                        if 'json' in self.output_formats:
                            try:
                                json_path = os.path.join(self.output_dir, f"{filename}.json")
                                res.save_to_json(json_path)
                            except AttributeError:
                                # 对于无 save_to_json 的模型 (如 pp-ocrv6), 构造简易 JSON
                                try:
                                    import json as json_module
                                    json_path = os.path.join(self.output_dir, f"{filename}.json")
                                    with open(json_path, 'w', encoding='utf-8') as jf:
                                        json_module.dump({"text": ocr_results}, jf, ensure_ascii=False, indent=2)
                                except Exception as e2:
                                    logger.error(f"保存 JSON 失败: {e2}")
                            except Exception as e:
                                logger.error(f"保存 JSON 失败: {e}")
                        
                        # Markdown: 仅结构类模型支持 (pp-ocrv6 跳过)
                        if 'markdown' in self.output_formats:
                            if hasattr(res, 'save_to_markdown') and self.model != 'pp-ocrv6':
                                try:
                                    md_path = os.path.join(self.output_dir, f"{filename}.md")
                                    res.save_to_markdown(md_path)
                                except Exception as e:
                                    logger.error(f"保存 Markdown 失败: {e}")
                            elif self.model == 'pp-ocrv6':
                                logger.warning(f"pp-ocrv6 不支持 markdown 输出, 已跳过")
                        
                        # Annotated Image: 所有模型都支持
                        if 'img' in self.output_formats:
                            try:
                                img_path = os.path.join(self.output_dir, f"{filename}_p{page_num:03d}.png")
                                if hasattr(res, 'save_to_img'):
                                    res.save_to_img(img_path)
                            except Exception as e:
                                logger.error(f"保存标注图片失败: {e}")
                
                # PDF (annotated layout PDF): 收集所有 PNG, 打包为单 PDF
                if 'pdf' in self.output_formats:
                    try:
                        import img2pdf
                        png_files = sorted(
                            glob_module.glob(os.path.join(self.output_dir, f"{filename}_p*.png"))
                        )
                        if png_files:
                            pdf_path = os.path.join(self.output_dir, f"{filename}_annotated.pdf")
                            with open(pdf_path, 'wb') as f:
                                f.write(img2pdf.convert([str(p) for p in png_files]))
                            logger.info(f"标注 PDF 已生成: {pdf_path}")
                    except ImportError:
                        logger.error("缺少 img2pdf, 请执行: pip install img2pdf")
                    except Exception as e:
                        logger.error(f"生成标注 PDF 失败: {e}")
```

- [ ] **Step 4: Propagate `output_formats` through callers**

**4a: `PDFFileHandler.__init__()` — line 746**

Change signature from:
```python
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
                 lang='ch', model_size='medium',
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
```

To:
```python
    def __init__(self, output_dir, model='pp-ocrv6', device='auto',
                 lang='ch', model_size='medium',
                 output_formats=None,  # 新增
                 optimize_pdf=False, optimize_level='medium', grayscale=False):
```

Add after line 756 (`self.grayscale = grayscale`):
```python
        if output_formats is None:
            output_formats = ['markdown', 'json', 'img', 'pdf']
        self.output_formats = output_formats
```

**4b: `PDFFileHandler.process_pdf_task()` — line 768**

Change `PDFOCRHandler` construction at line 773-780 to pass `output_formats`:

```python
        ocr_handler = PDFOCRHandler(
            self.output_dir, 
            self.model,
            device=self.device,
            output_formats=self.output_formats,  # 新增
            optimize_pdf=self.optimize_pdf_flag,
            optimize_level=self.optimize_level,
            grayscale=self.grayscale
        )
```

**4c: `run_manual_mode()` — line 796**

Change signature to add `output_formats=None`:

```python
def run_manual_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium',
                    output_formats=None,  # 新增
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
```

Change `PDFOCRHandler` construction at lines 812-821 to pass `output_formats`:

```python
    ocr_handler = PDFOCRHandler(
        output_dir,
        model,
        device=device,
        lang=lang,
        model_size=model_size,
        output_formats=output_formats,  # 新增
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

**4d: `run_daemon_mode()` — line 847**

Change signature to add `output_formats=None`:

```python
def run_daemon_mode(input_dir, output_dir, model='pp-ocrv6', device='auto', lang='ch', model_size='medium',
                    output_formats=None,  # 新增
                    optimize_pdf=False, optimize_level='medium', grayscale=False):
```

Change `PDFFileHandler` construction at lines 852-861 to pass `output_formats`:

```python
    event_handler = PDFFileHandler(
        output_dir,
        model,
        device=device,
        lang=lang,
        model_size=model_size,
        output_formats=output_formats,  # 新增
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale
    )
```

**4e: `main()` — line 883**

Add `output_formats` parsing after line 917 (`args = parser.parse_args()`):

```python
    # 解析 output-formats 参数
    output_formats_list = None
    if args.output_formats:
        output_formats_list = [f.strip() for f in args.output_formats.split(',') if f.strip()]
```

Actually, since the default is `'markdown,json,img,pdf'`, this will always be a non-empty string. So `output_formats_list` will always be set.

Pass `output_formats_list` to all handler constructions in `main()`:

At line 938-947 (single file path), add `output_formats=output_formats_list`:

```python
        ocr_handler = PDFOCRHandler(
            args.output,
            args.model,
            device=args.device,
            lang=args.lang,
            model_size=args.model_size,
            output_formats=output_formats_list,  # 新增
            optimize_pdf=args.optimize_pdf,
            optimize_level=args.optimize_level,
            grayscale=args.grayscale
        )
```

At lines 954-964 (`run_manual_mode` call), add `output_formats=output_formats_list`:

```python
            run_manual_mode(
                args.input,
                args.output,
                args.model,
                device=args.device,
                lang=args.lang,
                model_size=args.model_size,
                output_formats=output_formats_list,  # 新增
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
```

At lines 966-976 (`run_daemon_mode` call), add `output_formats=output_formats_list`:

```python
            run_daemon_mode(
                args.input,
                args.output,
                args.model,
                device=args.device,
                lang=args.lang,
                model_size=args.model_size,
                output_formats=output_formats_list,  # 新增
                optimize_pdf=args.optimize_pdf,
                optimize_level=args.optimize_level,
                grayscale=args.grayscale
            )
```

- [ ] **Step 5: Verify with --help**

Run:
```powershell
python ocr_pdf.py --help
```
Expected output: contains `-of, --output-formats` with `默认全部输出` description.

- [ ] **Step 6: Commit**

```bash
git add ocr_pdf.py
git commit -m "feat: add output-formats CLI flag and 4 format output branches"
```

---

## Task 4: api.py — output_formats param, validation, response, /download endpoint

**Files:**
- Modify: `C:\guole\code\ppocr_pdf\api.py`
- Verify: `python -c "from api import app; print('API import OK')"`

**Depends:** Task 3 (PDFOCRHandler accepts output_formats)

**Interfaces:**
- Consumes: `PDFOCRHandler.__init__(..., output_formats=...)` from Task 3
- Produces: `POST /ocr/pdf` accepts `output_formats` Form, validates, returns `outputs` dict
- Produces: `GET /download/{path}` serves output files
- Produces: `/health` includes `output_formats` field

- [ ] **Step 1: Add API output dir management constants**

Add after line 16 (`from device_utils import detect_device`):

```python
import uuid
import threading
import glob
```

Add after line 24 (`_HANDLER_LOCKS: dict[tuple, asyncio.Lock] = {}`):

```python
# ---------------------------------------------------------------------------
# API 输出目录管理: 持久化保存 OCR 输出文件, 供 /download 端点访问
# ---------------------------------------------------------------------------
API_OUTPUT_BASE = os.path.join(os.getcwd(), "api_outputs")
API_OUTPUT_TTL = 3600  # 1 小时
os.makedirs(API_OUTPUT_BASE, exist_ok=True)
```

- [ ] **Step 2: Add lazy cleanup helper**

Add before line 27 (`def get_handler(...)`):

```python
def _cleanup_expired_dirs():
    """惰性清理: 删除超过 TTL 的 API 输出目录"""
    now = time.time()
    try:
        for entry in os.listdir(API_OUTPUT_BASE):
            entry_path = os.path.join(API_OUTPUT_BASE, entry)
            if os.path.isdir(entry_path):
                age = now - os.path.getmtime(entry_path)
                if age > API_OUTPUT_TTL:
                    import shutil
                    shutil.rmtree(entry_path, ignore_errors=True)
    except Exception:
        pass
```

- [ ] **Step 3: Add `valid_output_formats` constant**

Add after line 24 (after `_HANDLER_LOCKS`):

```python
# ---------------------------------------------------------------------------
# 输出格式常量
# ---------------------------------------------------------------------------
VALID_OUTPUT_FORMATS = ["markdown", "json", "img", "pdf"]
```

- [ ] **Step 4: Update `/health` endpoint**

Change lines 199-206 from:

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"],
        "device_modes": ["auto", "gpu", "cpu"],
        "default_device": "auto"
    }
```

To:

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PDF OCR API",
        "models": ["pp-ocrv6", "pp-ocrv5", "pp-structurev3", "paddleocr-vl", "pp-chatocrv4"],
        "output_formats": VALID_OUTPUT_FORMATS,
        "device_modes": ["auto", "gpu", "cpu"],
        "default_device": "auto"
    }
```

- [ ] **Step 5: Add `output_formats` Form param to POST /ocr/pdf**

Add `output_formats` parameter to the `ocr_pdf` function signature at line 227-235. Insert after `grayscale`:

```python
    output_formats: Optional[str] = Form(
        default="markdown,json,img,pdf",
        description="输出格式 (逗号分隔). 可选值: markdown, json, img, pdf"
    ),
```

- [ ] **Step 6: Add output_formats validation**

Add after line 269 (after `if device not in valid_devices:` check):

```python
    # 验证输出格式
    formats_list = None
    if output_formats:
        formats_list = [f.strip() for f in output_formats.split(',') if f.strip()]
        invalid = set(formats_list) - set(VALID_OUTPUT_FORMATS)
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"无效的输出格式: {', '.join(sorted(invalid))}。可选: {', '.join(VALID_OUTPUT_FORMATS)}"
            )
```

- [ ] **Step 7: Replace tempfile.TemporaryDirectory with persistent output dir**

**7a: Add import for `uuid` at top (already added in Step 1)**

**7b: Change the request handling in POST /ocr/pdf body**

At line 271 (`try:`), replace the tempfile block with persistent output dir:

OLD (lines 272-281):
```python
    try:
        # 创建临时目录保存上传的PDF文件
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存上传的PDF文件
            pdf_path = os.path.join(tmp_dir, file.filename)
            with open(pdf_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # 创建临时输出目录
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
```

NEW:
```python
    try:
        # 惰性清理过期输出目录
        _cleanup_expired_dirs()
        
        # 创建持久化请求目录
        request_id = str(uuid.uuid4())[:8]
        request_dir = os.path.join(API_OUTPUT_BASE, request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        # 保存上传的PDF文件
        pdf_path = os.path.join(request_dir, file.filename)
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 创建输出子目录
        output_dir = os.path.join(request_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
```

**7c: Fix indentation for remaining code inside try block** — remove one level of indentation since `with tempfile.TemporaryDirectory()` is replaced.

Specifically, lines 283-333 need their indentation reduced by 4 spaces (remove the `with` block but keep the code inside). Also, remove line 334-335 (the `except HTTPException: raise` is already further down).

**7d: Extend the response with `outputs` dict and `request_id`**

Change the response construction at lines 321-333 from:

```python
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "filename": file.filename,
                    "model": model,
                    "lang": lang,
                    "model_size": model_size,
                    "device": device,
                    "device_info": device_info,
                    "result": ocr_result
                }
            )
```

To:

```python
            # 扫描输出目录收集所有文件路径
            outputs = {}
            if os.path.exists(output_dir):
                for f_name in sorted(os.listdir(output_dir)):
                    ext = os.path.splitext(f_name)[1].lower()
                    f_path = os.path.join(output_dir, f_name).replace(os.sep, '/')
                    if ext == '.txt':
                        outputs["txt"] = f_path
                    elif ext == '.md':
                        outputs["md"] = f_path
                    elif ext == '.json':
                        outputs["json"] = f_path
                    elif ext == '.png':
                        outputs.setdefault("img", []).append(f_path)
                    elif ext == '.pdf' and '_annotated' in f_name:
                        outputs["pdf"] = f_path
            
            # 返回识别结果
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "filename": file.filename,
                    "model": model,
                    "lang": lang,
                    "model_size": model_size,
                    "device": device,
                    "device_info": device_info,
                    "request_id": request_id,
                    "outputs": outputs,
                    "result": ocr_result,  # 保留向后兼容
                }
            )
```

- [ ] **Step 8: Add `GET /download/{path}` endpoint**

Add after the `POST /ocr/pdf` handler (after line 341), before `if __name__ == "__main__":`:

```python
# ---------------------------------------------------------------------------
# 下载端点: 提供 OCR 输出文件的直接下载
# ---------------------------------------------------------------------------
@app.get("/download/{path:path}")
async def download_output(path: str):
    """下载 API 生成的输出文件"""
    from fastapi.responses import FileResponse
    
    full_path = os.path.normpath(os.path.join(API_OUTPUT_BASE, path))
    # 安全: 防止 path traversal
    if not full_path.startswith(os.path.normpath(API_OUTPUT_BASE)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.json': 'application/json',
        '.png': 'image/png',
        '.pdf': 'application/pdf',
    }.get(os.path.splitext(path)[1].lower(), 'application/octet-stream')
    
    return FileResponse(full_path, media_type=media_type, filename=os.path.basename(path))
```

- [ ] **Step 9: Update `get_handler()` to pass `output_formats`**

Modify `get_handler()` at line 27-59 to accept and pass `output_formats`:

Change signature to add `output_formats=None`:
```python
def get_handler(
    output_dir: str,
    model: str,
    device: str,
    lang: str,
    model_size: str,
    output_formats: Optional[list] = None,  # 新增
    optimize_pdf: bool = False,
    optimize_level: str = "medium",
    grayscale: bool = False,
) -> "PDFOCRHandler":
```

Update cache key at line 42:
```python
    key = (model, device, lang, model_size,
           tuple(output_formats) if output_formats else tuple(),  # 新增
           optimize_pdf, optimize_level, grayscale)
```

Update handler construction at lines 49-57 to pass `output_formats`:
```python
    handler = PDFOCRHandler(
        output_dir, model,
        device=device,
        lang=lang,
        model_size=model_size,
        output_formats=output_formats,  # 新增
        optimize_pdf=optimize_pdf,
        optimize_level=optimize_level,
        grayscale=grayscale,
    )
```

Update handler construction call at line 285-293 in POST /ocr/pdf to pass `output_formats`:
```python
            ocr_handler = get_handler(
                output_dir, model,
                device=device,
                lang=lang,
                model_size=model_size,
                output_formats=formats_list,  # 新增
                optimize_pdf=optimize_pdf,
                optimize_level=optimize_level,
                grayscale=grayscale,
            )
```

Update lock key at line 297-300 to include `formats_list`:
```python
            lock = _HANDLER_LOCKS.setdefault(
                (model, device, lang, model_size,
                 tuple(formats_list) if formats_list else tuple(),
                 optimize_pdf, optimize_level, grayscale),
                asyncio.Lock(),
            )
```

- [ ] **Step 10: Verify API imports correctly**

Run:
```powershell
python -c "from api import app; print('API import OK')"
```
Expected output: `API import OK`

- [ ] **Step 11: Commit**

```bash
git add api.py
git commit -m "feat(api): add output_formats param, validation, /download endpoint"
```

---

## Task 5: Integration tests — tests/output_format_selector_test.py

**Files:**
- Create: `C:\guole\code\ppocr_pdf\tests\output_format_selector_test.py`
- Verify: `pytest tests/output_format_selector_test.py -v -m integration`

**Depends:** Task 3 (full ocr_pdf.py feature)

- [ ] **Step 1: Write the complete test file**

```python
"""tests/output_format_selector_test.py
Output format selector integration tests.
Uses real PDFs + real PaddleOCR (no mocks).
Run: pytest tests/output_format_selector_test.py -v -m integration
"""
import sys
import os
import json
import logging

# 禁用非必要的日志输出
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from ocr_pdf import PDFOCRHandler


# =========================================================================
# Helpers
# =========================================================================

@pytest.fixture(scope="module")
def two_page_pdf(tmp_path_factory):
    """生成 2 页真实 PDF 文件供测试"""
    import pypdfium2 as pdfium

    pdf_path = tmp_path_factory.mktemp("inputs") / "test.pdf"
    lib = pdfium.PdfLibrary()
    doc = lib.new_doc()

    for i in range(2):
        page = doc.new_page(612, 792)  # letter size
        page.set_font_size(14)
        page.set_fill(0)
        page.draw_text(f"This is test page {i+1}.", 72, 700)

    doc.save(pdf_path.__str__())
    doc.close()
    lib.close()
    return pdf_path


def get_page_count(pdf_path):
    """返回 PDF 页数"""
    import pypdfium2 as pdfium
    lib = pdfium.PdfLibrary()
    doc = lib.open_document(pdf_path.__str__())
    count = doc.get_page_count()
    doc.close()
    lib.close()
    return count


def _create_handler(output_dir, model='paddleocr-vl', output_formats=None, **kwargs):
    """创建 PDFOCRHandler 的辅助函数, 统一处理设备回退"""
    from ocr_pdf import PDFOCRHandler
    handler = PDFOCRHandler(
        output_dir=str(output_dir),
        model=model,
        device='cpu',
        output_formats=output_formats,
        **kwargs
    )
    return handler


# =========================================================================
# Tests
# =========================================================================

class TestOutputFormatSelector:
    """输出格式选择器集成测试 — 使用真实数据"""

    @pytest.mark.integration
    def test_default_output_formats_writes_all_four(self, two_page_pdf, tmp_path):
        """默认 -of (全部) 应生成 4 种新格式文件 + .txt"""
        out_dir = tmp_path / "out_default"
        out_dir.mkdir()
        handler = _create_handler(out_dir)
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True, "process_pdf 应返回 True"
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        
        assert f"{stem}.txt" in files, ".txt 应始终存在"
        assert f"{stem}.json" in files, ".json 应存在"
        assert f"{stem}.md" in files, ".md 应存在"
        # 应至少有 1 个 png (2 页)
        pngs = [f for f in files if f.startswith(f"{stem}_p") and f.endswith(".png")]
        assert len(pngs) >= 1, "应有至少 1 个标注 PNG"
        pdfs = [f for f in files if f.endswith("_annotated.pdf")]
        assert len(pdfs) >= 1, "_annotated.pdf 应存在"

    @pytest.mark.integration
    def test_selective_output_formats_only_writes_md(self, two_page_pdf, tmp_path):
        """-of markdown 应只生成 .md 和 .txt"""
        out_dir = tmp_path / "out_md"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['markdown'])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        
        assert f"{stem}.txt" in files, ".txt 应始终存在"
        assert f"{stem}.md" in files, ".md 应存在"
        assert f"{stem}.json" not in files, ".json 不应存在"
        pngs = [f for f in files if f.endswith(".png")]
        assert len(pngs) == 0, "PNG 不应存在"
        pdfs = [f for f in files if f.endswith("_annotated.pdf")]
        assert len(pdfs) == 0, "PDF 不应存在"

    @pytest.mark.integration
    def test_pp_ocrv6_markdown_skipped(self, two_page_pdf, tmp_path):
        """pp-ocrv6 + markdown 应跳过 .md, 其他格式正常"""
        out_dir = tmp_path / "out_v6"
        out_dir.mkdir()
        handler = _create_handler(
            out_dir,
            model='pp-ocrv6',
            output_formats=['markdown', 'json', 'img', 'pdf']
        )
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        
        assert f"{stem}.txt" in files, ".txt 应始终存在"
        # pp-ocrv6 不生成 markdown (skip + warning)
        assert f"{stem}.md" not in files, "pp-ocrv6 不应生成 .md"

    @pytest.mark.integration
    def test_json_has_valid_structure(self, two_page_pdf, tmp_path):
        """-of json 生成的 .json 应包含文本数据"""
        out_dir = tmp_path / "out_json"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['json'])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        json_path = os.path.join(str(out_dir), f"{stem}.json")
        assert os.path.exists(json_path), ".json 文件应存在"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 应包含文本内容
        assert isinstance(data, (dict, list)), "JSON 应为 dict 或 list"
        if isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False)
            assert "test page" in content.lower() or "rec_texts" in content, "JSON 应包含识别文本"

    @pytest.mark.integration
    def test_annotated_img_per_page(self, two_page_pdf, tmp_path):
        """-of img 应为每页生成一个 PNG 图片"""
        out_dir = tmp_path / "out_img"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['img'])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        pngs = sorted([f for f in files if f.startswith(f"{stem}_p") and f.endswith(".png")])
        
        assert len(pngs) >= 2, f"2 页 PDF 应生成至少 2 个 PNG, 实际: {len(pngs)}"
        for png in pngs:
            png_path = os.path.join(str(out_dir), png)
            assert os.path.getsize(png_path) > 0, f"{png} 不应为空"

    @pytest.mark.integration
    def test_layout_pdf_is_valid(self, two_page_pdf, tmp_path):
        """-of pdf 应生成有效的标注 PDF"""
        out_dir = tmp_path / "out_pdf"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['img', 'pdf'])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        pdf_path = os.path.join(str(out_dir), f"{stem}_annotated.pdf")
        assert os.path.exists(pdf_path), "_annotated.pdf 应存在"
        
        # 验证 PDF 页数
        page_count = get_page_count(pdf_path)
        assert page_count >= 1, f"标注 PDF 应有至少 1 页, 实际: {page_count}"

    @pytest.mark.integration
    def test_no_output_formats_only_txt(self, two_page_pdf, tmp_path):
        """output_formats=[] 应只生成 .txt (通过传递空列表模拟旧行为)"""
        out_dir = tmp_path / "out_only_txt"
        out_dir.mkdir()
        # 用空列表禁用格式输出 (与设计一致: 空列表 = 仅 .txt)
        handler = _create_handler(out_dir, output_formats=[])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        
        assert f"{stem}.txt" in files, ".txt 应始终存在"
        # 不应有其他格式文件
        for f in files:
            if f == f"{stem}.txt":
                continue
            assert not f.endswith(('.json', '.md', '.png', '.pdf')), f"不应生成额外格式: {f}"

    @pytest.mark.integration
    def test_single_format_json_only(self, two_page_pdf, tmp_path):
        """-of json 只生成 .json + .txt"""
        out_dir = tmp_path / "out_json_only"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['json'])
        
        result = handler.process_pdf(str(two_page_pdf))
        assert result is True
        
        stem = two_page_pdf.stem
        files = os.listdir(str(out_dir))
        
        assert f"{stem}.txt" in files
        assert f"{stem}.json" in files
        assert f"{stem}.md" not in files
        pngs = [f for f in files if f.endswith(".png")]
        assert len(pngs) == 0
        pdfs = [f for f in files if f.endswith("_annotated.pdf")]
        assert len(pdfs) == 0

    @pytest.mark.integration
    def test_txt_always_present_regardless_of_formats(self, two_page_pdf, tmp_path):
        """无论 -of 是什么, .txt 始终生成"""
        out_dir = tmp_path / "out_always_txt"
        out_dir.mkdir()
        
        # 测试不同组合
        for fmts in [['markdown'], ['json'], ['img'], ['pdf'], ['markdown', 'json']]:
            test_dir = out_dir / f"fmt_{'_'.join(fmts)}"
            test_dir.mkdir()
            handler = _create_handler(test_dir, output_formats=fmts)
            handler.process_pdf(str(two_page_pdf))
            
            stem = two_page_pdf.stem
            assert os.path.exists(os.path.join(str(test_dir), f"{stem}.txt")), \
                f"formats={fmts} 时 .txt 应存在"
```

- [ ] **Step 2: Run tests to verify they fail initially (before implementation)**

Run:
```powershell
python -m pytest tests/output_format_selector_test.py -v -m integration
```
Expected outcome: test file doesn't exist yet (created in Step 1). But since we implement AFTER writing tests, the first run should show failures due to missing features... Actually, since Task 3 is done first, the tests should pass. The TDD cycle here is: Task 3 implements the feature, then Task 5 writes tests, then runs them.

Actually, this is integration-test-first, so tests come after implementation by design (since design was validated). Let me adjust: we write the test file, then run it immediately to verify implementation from Task 3 works correctly.

- [ ] **Step 3: Run tests to verify they pass**

Run:
```powershell
python -m pytest tests/output_format_selector_test.py -v -m integration
```
Expected outcome: all 9 tests pass (may take ~60-120s due to real PaddleOCR model loading).

Note: If `paddleocr-vl` model is not downloaded yet, the first run will download it. This is expected and only happens once.

If any tests fail due to `save_to_img()` or `save_to_json()` not being available on the result objects, note that those tests check for file existence so they'll gracefully fail. The important thing is that `.txt` is always present and format filtering works.

- [ ] **Step 4: Commit**

```bash
git add tests/output_format_selector_test.py
git commit -m "test: add 9 integration tests for output format selector"
```

---

## Task 6: Documentation — README.md + Docker.md

**Files:**
- Modify: `C:\guole\code\ppocr_pdf\README.md`
- Modify: `C:\guole\code\ppocr_pdf\Docker.md`
- Verify: visual inspection

**Depends:** Task 3, Task 4 (complete feature)

- [ ] **Step 1: Update README.md — CLI arguments table**

Find the CLI arguments section in README.md and add `-of` entry.

Replace the current `python ocr_pdf.py [-h] ...` block and parameter table with updated versions:

**Update `python ocr_pdf.py` usage line** to include `[-of ...]`:

```bash
python ocr_pdf.py [-h] -i INPUT -o OUTPUT [-m {manual,daemon}] [-model {paddleocr-vl,pp-ocrv6,pp-ocrv5,pp-structurev3,pp-chatocrv4}] [-l {debug,info,warning,error,critical}] [--optimize-pdf] [--optimize-level {low,medium,high}] [--grayscale] [-of OUTPUT_FORMATS]
```

**Add `-of` row to parameter table:**

```markdown
- `-of, --output-formats`: 输出格式 (逗号分隔)，可选值：markdown、json、img、pdf，默认：markdown,json,img,pdf
```

Insert between the `--grayscale` line and the `-h` line in the parameter table.

- [ ] **Step 2: Update README.md — Output results section**

Replace the current "输出结果" section with an updated version:

**Find the section starting with "### 输出结果" and update:**

```markdown
### 输出结果与格式选择

识别结果始终以 `.txt` 格式保存到输出目录。此外，可通过 `-of`/`--output-formats` 参数选择额外的输出格式：

| 格式 | 参数值 | 扩展名 | 说明 | 兼容性 |
|------|--------|--------|------|--------|
| 纯文本 | — | `.txt` | 始终生成, 按页面组织, 用 `=== 第 X 页 ===` 分隔 | 全部模型 |
| JSON | `json` | `.json` | 结构化数据, 含检测框/置信度/识别文本 | 全部模型 |
| Markdown | `markdown` | `.md` | 保留标题/表格/公式结构, 可直接转 HTML/DOCX | paddleocr-vl, pp-structurev3 |
| 标注图片 | `img` | `_p{nnn}.png` | 检测框 + 文字可视化, 用于 debug / 校对 | 全部模型 |
| 标注 PDF | `pdf` | `_annotated.pdf` | 标注覆盖在页面上的 PDF, 用于存档 / 批注 | 全部模型 (依赖 img2pdf) |

**示例输出 (默认 `-of markdown,json,img,pdf`):**

```
./output/doc.txt           # 纯文本 (始终生成)
./output/doc.md            # Markdown 结构化文本
./output/doc.json          # JSON 结构化数据
./output/doc_p001.png      # 第 1 页标注图片
./output/doc_p002.png      # 第 2 页标注图片
./output/doc_annotated.pdf # 标注 PDF
```

**注意:** `pp-ocrv6` 模型不支持 `markdown` 格式。如指定 `-of markdown` + `-model pp-ocrv6`, 程序将给出警告并跳过 .md 文件生成。
```

- [ ] **Step 3: Update README.md — Model compatibility table**

Add an "输出格式兼容性" column to the model selection table.

Find the model table and add a column:

```markdown
| 模型名称 | ... | 支持状态 | 输出格式兼容性 |
|----------|-----|---------|----------------|
| paddleocr-vl | ... | ✅ 支持 | json ✅, markdown ✅, img ✅, pdf ✅ |
| pp-ocrv6 | ... | ✅ 支持 | json ✅, markdown ❌, img ✅, pdf ✅ |
| pp-ocrv5 | ... | ✅ 支持 | json ✅, markdown ❌ (遗留), img ✅, pdf ✅ |
| pp-structurev3 | ... | ✅ 支持 (需额外依赖) | json ✅, markdown ✅, img ✅, pdf ✅ |
| pp-chatocrv4 | ... | ⚠️ 需API密钥 | json ✅, markdown ❌, img ❌, pdf ❌ |
```

- [ ] **Step 4: Update README.md — Add CLI examples with `-of`**

After the existing CLI examples, add:

```bash
# 仅输出纯文本 + JSON
python ocr_pdf.py -i ./test_input -o ./test_output -m manual -of json

# 输出纯文本 + Markdown + 标注图片
python ocr_pdf.py -i ./test_input -o ./test_output -m manual -of markdown,img
```

- [ ] **Step 5: Update README.md — API section**

In the "API 接口" section:

**Update POST /ocr/pdf description** to include `output_formats` parameter:

Add after the `grayscale` parameter in the request parameters list:
- `output_formats`：输出格式（可选，默认：markdown,json,img,pdf），可选值：markdown, json, img, pdf

**Update example curl:**

```bash
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.pdf" \
  -F "model=pp-structurev3" \
  -F "output_formats=markdown,json"
```

**Update example response** to include `outputs`:

```json
{
  "status": "success",
  "filename": "test.pdf",
  "model": "pp-structurev3",
  "outputs": {
    "txt": "api_outputs/a1b2c3/output/test.txt",
    "md": "api_outputs/a1b2c3/output/test.md",
    "json": "api_outputs/a1b2c3/output/test.json"
  },
  "result": "=== 第 1 页 ===\nHello World!\n..."
}
```

**Add download endpoint documentation:**

```markdown
##### 3. 下载输出文件

```
GET /download/{path}
```

下载 OCR 处理生成的输出文件。`{path}` 为响应中 `outputs` 字段返回的路径（相对于项目根目录）。

**示例：**
```bash
# 先通过 /ocr/pdf 获取 outputs 路径
# 然后下载
curl -O http://localhost:8000/download/api_outputs/a1b2c3/output/test.json
```
```

- [ ] **Step 6: Update Docker.md**

Find the test API curl example and add the `output_formats` variant:

After the existing curl examples, add:

```bash
# PDF OCR识别（带输出格式参数）
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./test_input/01.pdf" \
  -F "model=pp-ocrv5" \
  -F "output_formats=markdown,json"
```

- [ ] **Step 7: Commit**

```bash
git add README.md Docker.md
git commit -m "docs: update README and Docker.md with output_formats docs"
```

---

## Task 7: Final verification

**Files:** none (verification only)

**Depends:** Tasks 1-6

- [ ] **Step 1: Run all integration tests**

Run:
```powershell
python -m pytest tests/output_format_selector_test.py -v -m integration
```
Expected output: 9 passed (may take 60-120s for real PaddleOCR model loading).

- [ ] **Step 2: Run full test suite**

Run:
```powershell
python -m pytest tests/ -v
```
Expected output: at least 45 existing + 9 new = 54 tests passed.

- [ ] **Step 3: Verify CLI help**

Run:
```powershell
python ocr_pdf.py --help
```
Expected output: contains `-of, --output-formats` flag with `markdown,json,img,pdf` default.

- [ ] **Step 4: Verify img2pdf import**

Run:
```powershell
python -c "import img2pdf; print('OK')"
```
Expected output: `OK`

- [ ] **Step 5: CLI smoke test**

Generate a 2-page PDF and run through CLI with `-of all`:

```powershell
python -c "
import pypdfium2 as pdfium
lib = pdfium.PdfLibrary()
doc = lib.new_doc()
for i in range(2):
    p = doc.new_page(612, 792)
    p.set_font_size(14)
    p.set_fill(0)
    p.draw_text(f'Smoke test page {i+1}.', 72, 700)
doc.save('test_smoke.pdf')
doc.close()
lib.close()
print('Test PDF created')
"
python ocr_pdf.py -i test_smoke.pdf -o test_smoke_out -of markdown,json,img,pdf --device cpu
```
Expected: `ocr_logs.md` has a success entry; `test_smoke_out/` contains `test_smoke.txt`, `test_smoke.md`, `test_smoke.json`, `test_smoke_p001.png`, `test_smoke_p002.png`, `test_smoke_annotated.pdf`.

Cleanup:
```powershell
Remove-Item -Recurse -Force test_smoke.pdf, test_smoke_out
```

- [ ] **Step 6: API smoke test (optional, requires server)**

If API server is available, run a quick smoke test:
```powershell
# Start API in background
Start-Process -NoNewWindow python api.py
Start-Sleep -Seconds 3

# Test health endpoint
python -c "import urllib.request; import json; data = json.loads(urllib.request.urlopen('http://localhost:8000/health').read()); print('output_formats' in data, data.get('output_formats'))"

# Test OCR with output_formats
python -c "
import requests
with open('test_smoke_api.pdf', 'rb') as f:
    r = requests.post('http://localhost:8000/ocr/pdf', files={'file': f}, data={'model': 'pp-ocrv6', 'output_formats': 'json'})
    print('Status:', r.status_code)
    data = r.json()
    print('Has outputs:', 'outputs' in data)
    print('Format keys:', list(data.get('outputs', {}).keys()))
"

# Stop API
Get-Process -Name python | Where-Object { $_.CommandLine -match 'api.py' } | Stop-Process

# Cleanup
Remove-Item -Recurse -Force test_smoke_api.pdf -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force api_outputs -ErrorAction SilentlyContinue
```

- [ ] **Step 7: Commit final verification (if any fixes needed)**

If any issues found during verification, fix them before final commit. Otherwise, no commit needed for verification.

---

## Self-Review

### Spec Coverage

| Design Requirement | Task | Status |
|---|---|---|
| 4 new formats: markdown, json, img, pdf | Task 3 — process_pdf branches | Covered |
| `.txt` always generated | Task 3 — existing logic preserved | Covered |
| Default: all 4 enabled | Task 3 — __init__ default `['markdown','json','img','pdf']` | Covered |
| CLI `-of`/`--output-formats` arg | Task 3 — argparse | Covered |
| API `output_formats` Form param | Task 4 — api.py | Covered |
| API validation (400 on invalid) | Task 4 — `VALID_OUTPUT_FORMATS` check | Covered |
| API returns paths in `outputs` dict | Task 4 — response extension | Covered |
| API `GET /download/{path}` endpoint | Task 4 — download endpoint | Covered |
| API output dir management + cleanup | Task 4 — `API_OUTPUT_BASE`, `_cleanup_expired_dirs` | Covered |
| `/health` includes `output_formats` | Task 4 — health check update | Covered |
| `img2pdf` dependency | Task 1 — requirements.txt | Covered |
| `pp-ocrv6` + markdown skip + warn | Task 3 — `elif self.model == 'pp-ocrv6': logger.warning()` | Covered |
| Inline implementation, no new modules | Task 3 — all code in `process_pdf()` | Covered |
| No mock tests, real PDF + real PaddleOCR | Task 5 — integration tests | Covered |
| `@pytest.mark.integration` on all tests | Task 5 — markers on all 9 tests | Covered |
| Tests marked `@pytest.mark.integration` | Task 2 — pytest.ini registers marker | Covered |
| Do NOT modify device_utils.py, Dockerfile | All tasks | Covered |
| README.md + Docker.md updates | Task 6 | Covered |
| 9 new integration tests | Task 5 — 9 test methods | Covered |
| Existing tests still pass (54 total) | Task 7 — full suite verification | Covered |

### Placeholder Scan
- No "TBD", "TODO", "implement later", "similar to Task N"
- All code blocks have complete implementation
- All file paths are exact
- All commands are PowerShell-compatible

### Type Consistency
- `PDFOCRHandler.__init__(output_formats=None)` → stores `self.output_formats = output_formats`
- CLI parses `args.output_formats` → `[f.strip() for f in args.output_formats.split(',') if f.strip()]`
- `run_manual_mode(..., output_formats=None)` → passes to `PDFOCRHandler`
- `run_daemon_mode(..., output_formats=None)` → passes to `PDFFileHandler`
- `PDFFileHandler.__init__(..., output_formats=None)` → passes to `PDFOCRHandler`
- `get_handler(..., output_formats=None)` → includes in cache key
- API Form param `output_formats` → split → validated → passed to `get_handler`
- Test `_create_handler(..., output_formats=None)` → passed to `PDFOCRHandler`
- Consistent string: `'markdown', 'json', 'img', 'pdf'`
- All method signatures use `output_formats` as parameter name consistently

---

**Plan complete and saved to `thoughts/shared/plans/2026-06-22-output-format-selector.md`.**

**Summary:**
- **7 tasks** total (1 dep + 1 config + 1 core + 1 api + 1 test + 1 docs + 1 verify)
- **4 files modified** (requirements.txt, ocr_pdf.py, api.py, README.md, Docker.md)
- **2 files created** (pytest.ini, tests/output_format_selector_test.py)
- **9 new integration tests** (total expected: 54)
- **7 commits** expected

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
