---
date: 2026-06-22
topic: "输出格式选择器 (markdown/json/img/layout-pdf)"
status: validated
---

# 输出格式选择器 (Output Format Selector) 设计

## 问题陈述

`ppocr_pdf` 当前只输出 `{stem}.txt` 一种格式，且所有模型输出被扁平化为纯文本。实际 PDF OCR 场景需要多样化输出：

1. **Markdown** — 保留标题/表格/公式结构，可直接转 HTML / DOCX
2. **JSON** — 含位置/置信度/检测框的结构化数据，供下游 pipeline 消费
3. **Annotated Images** — 检测框 + 文字可视化，用于 debug / 校对
4. **Layout-PDF** — 标注框覆盖在页面上，用于存档 / 批注校对

## 约束 (已确认)

- **4 个新格式**: `markdown`, `json`, `img`, `pdf` (layout-pdf 内部命名简化为 `pdf`)
- **`.txt` 常驻**: 始终生成，不受 `output_formats` 影响 (向后兼容)
- **默认值**: `output_formats = markdown,json,img,pdf` (全开)
- **`pp-ocrv6` + `markdown`**: 跳过 + logger.warning (该模型不支持原生 markdown)
- **API 写盘 + 返回路径**: API 保存输出文件到临时目录，JSON 返回路径列表
- **内联实现**: 不新建模块，在 `ocr_pdf.py:process_pdf()` 加 4 个 if 分支
- **测试用真实数据**: 不用 mock 数据，用真实 PDF + 真实 PaddleOCR，标记 `@pytest.mark.integration`
- **依赖**: 新增 `img2pdf` (~50KB)，将 PNG 序列无损打包成 PDF
- **不修改** `device_utils.py`、`Dockerfile`、历史文件 (audits/ledgers)

## 架构 — 内联 Format Branches

`PDFOCRHandler.process_pdf()` 现有 `.txt` 写入**之后**，串行追加 4 个 if 分支：

```
process_pdf():
  现有逻辑: render → ocr.predict() → extract text → write {stem}.txt
  
  新增:
  if 'json' in output_formats:     res.save_to_json("{dir}/{stem}.json")
  if 'markdown' in output_formats and model != 'pp-ocrv6':
    res.save_to_markdown("{dir}/{stem}.md")
  if 'img' in output_formats:      per page → res.save_to_img("{dir}/{stem}_p{nnn}.png")
  if 'pdf' in output_formats:      img2pdf.convert(pngs → "{dir}/{stem}_annotated.pdf")
```

每个 if 分支独立 try/except，一个格式失败不影响其他格式。

### Model-Format 兼容矩阵

| 模型 \ 格式 | `json` | `markdown` | `img` | `pdf` |
|-------------|:------:|:----------:|:-----:|:-----:|
| `paddleocr-vl` | ✅ | ✅ native | ✅ | ✅ img wrapper |
| `pp-ocrv6` | ✅ | ❌ skip+warn | ✅ | ✅ img wrapper |
| `pp-structurev3` | ✅ | ✅ native | ✅ | ✅ img wrapper |

### 为什么不建新模块

Q5 确认：内联。原因：
- 4 个 if 分支逻辑简单 (每个分支 ≈5 行)
- 所有写盘已经在一个地方 (`process_pdf()` 末尾)
- 不引入新文件，降低认知负担
- 测试用真实数据直接测 `process_pdf()` 即可

## 改动清单

### 1. `ocr_pdf.py` (核心改动 ~80 行)

#### CLI 参数 (新增)

```python
parser.add_argument(
    '-of', '--output-formats',
    type=str,
    default='markdown,json,img,pdf',
    help='输出格式 (逗号分隔). 可选值: markdown, json, img, pdf. '
         '默认全部输出. 提示: pp-ocrv6 不支持 markdown, 将自动跳过. '
         '示例: -of markdown,json'
)
```

#### PDFOCRHandler.__init__() (L137 ~ L183)

```python
class PDFOCRHandler:
    def __init__(self, output_dir, model='paddleocr-vl',
                 model_size='medium', lang='ch', device='auto',
                 output_formats=None,  # 新增
                 optimize_pdf=False, optimize_level='medium',
                 grayscale=False):
        if output_formats is None:
            output_formats = ['markdown', 'json', 'img', 'pdf']
        self.output_formats = output_formats
        # ... 现有初始化 ...
```

#### process_pdf() — 在 .txt 写入后追加 (L663 之后, finally 之前)

```python
# === 现有 .txt 写入 (L655-663) ===
ocr_output = '\n'.join(ocr_results)
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write(ocr_output)

# === 新增: 可选输出格式 (output_formats) ===
if self.output_formats:
    page_results = list(result)  # 将 generator 转为 list (如果还没转)
    
    for page_idx, res in enumerate(page_results):
        page_num = page_idx + 1
        
        # JSON: 所有模型都支持
        if 'json' in self.output_formats:
            try:
                json_path = os.path.join(self.output_dir, f"{stem}.json")
                res.save_to_json(json_path)
            except Exception as e:
                logger.error(f"保存 JSON 失败: {e}")
        
        # Markdown: 仅结构类模型支持 (pp-ocrv6 跳过)
        if 'markdown' in self.output_formats:
            if hasattr(res, 'save_to_markdown') and self.model != 'pp-ocrv6':
                try:
                    md_path = os.path.join(self.output_dir, f"{stem}.md")
                    res.save_to_markdown(md_path)
                except Exception as e:
                    logger.error(f"保存 Markdown 失败: {e}")
            elif self.model == 'pp-ocrv6':
                logger.warning(f"pp-ocrv6 不支持 markdown 输出, 已跳过")
        
        # Annotated Image: 所有模型都支持
        if 'img' in self.output_formats:
            try:
                img_path = os.path.join(self.output_dir, f"{stem}_p{page_num:03d}.png")
                res.save_to_img(img_path)
            except Exception as e:
                logger.error(f"保存标注图片失败: {e}")
    
    # PDF (annotated layout PDF): 收集所有 PNG, 打包为单 PDF
    if 'pdf' in self.output_formats:
        try:
            import img2pdf
            png_files = sorted(
                glob.glob(os.path.join(self.output_dir, f"{stem}_p*.png"))
            )
            if png_files:
                pdf_path = os.path.join(self.output_dir, f"{stem}_annotated.pdf")
                with open(pdf_path, 'wb') as f:
                    f.write(img2pdf.convert([str(p) for p in png_files]))
                logger.info(f"标注 PDF 已生成: {pdf_path}")
        except ImportError:
            logger.error("缺少 img2pdf, 请执行: pip install img2pdf")
        except Exception as e:
            logger.error(f"生成标注 PDF 失败: {e}")
```

**注:** `result` 的 `list()` 转换时机取决于当前的代码结构。当前代码 (L405-411) 在 v6 路径下做了 `result_list = list(result)`。上面方案的 `page_results = list(result)` 可能需要统一迁移，具体在实现阶段精调。

#### API 相关方法透传

- `run_manual_mode()`: 新增 `output_formats` 参数，传入 `PDFOCRHandler` 构造函数
- `run_daemon_mode()`: 同上
- `PDFFileHandler`: 同上
- `main()`: 解析 `args.output_formats` 为 list (按逗号 split + strip)，传给上述方法

### 2. `api.py` (API 同步 ~40 行)

#### 新增常量

```python
valid_output_formats = ["markdown", "json", "img", "pdf"]  # L70 附近
```

#### POST /ocr/pdf Form (L222-231)

```python
output_formats: Optional[str] = Form(
    default="markdown,json,img,pdf",
    description="输出格式 (逗号分隔). 可选值: markdown, json, img, pdf"
)
```

#### 校验 (新增)

```python
if output_formats:
    formats_list = [f.strip() for f in output_formats.split(',') if f.strip()]
    invalid = set(formats_list) - set(valid_output_formats)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"无效的输出格式: {', '.join(invalid)}。可选: {', '.join(valid_output_formats)}"
        )
else:
    formats_list = []  # 仅 .txt
```

#### 响应结构 (L318-328 → 改)

```python
output_dir = os.path.join(temp_dir, "output")
json_response = {
    "status": "success",
    "filename": file.filename,
    "model": model,
    "lang": getattr(handler, 'lang', lang),
    "model_size": getattr(handler, 'model_size', model_size),
    "device": device,
    "device_info": device_info,
    "outputs": {},
    "result": ocr_result,  # 保留向后兼容, 后续标记 deprecated
}

# 扫描输出目录收集所有文件路径
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext == '.txt':
            json_response["outputs"]["txt"] = os.path.join(output_dir, f)
        elif ext == '.md':
            json_response["outputs"]["md"] = os.path.join(output_dir, f)
        elif ext == '.json':
            json_response["outputs"]["json"] = os.path.join(output_dir, f)
        elif ext == '.png':
            json_response["outputs"].setdefault("img", []).append(os.path.join(output_dir, f))
        elif ext == '.pdf' and '_annotated' in f:
            json_response["outputs"]["pdf"] = os.path.join(output_dir, f)
```

#### GET /download/{path:path} (新增下载端点)

```python
@app.get("/download/{path:path}")
async def download_output(path: str):
    """下载 API 生成的输出文件"""
    full_path = os.path.normpath(os.path.join(OUTPUT_DIR, path))
    # 安全: 防止 path traversal
    if not full_path.startswith(os.path.normpath(OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = {
        '.txt': 'text/plain', '.md': 'text/markdown',
        '.json': 'application/json',
        '.png': 'image/png', '.pdf': 'application/pdf',
    }.get(os.path.splitext(path)[1].lower(), 'application/octet-stream')
    
    return FileResponse(full_path, media_type=media_type, filename=os.path.basename(path))
```

#### API 输出目录管理

```python
# 新增模块级变量
API_OUTPUT_BASE = os.path.join(os.getcwd(), "api_outputs")
API_OUTPUT_TTL = 3600  # 1 小时
```

启动时创建目录 + 定时清理线程 (或惰性清理: 上传时检查 + 清理过期 dirs)。

#### /health 更新 (L193-201)

```json
{
    "status": "healthy",
    "service": "PDF OCR API",
    "models": ["paddleocr-vl", "pp-ocrv6", "pp-structurev3"],
    "output_formats": ["markdown", "json", "img", "pdf"],
    "device_modes": ["auto", "gpu", "cpu"],
    "default_device": "auto"
}
```

### 3. `requirements.txt` (新增依赖)

```
# 输出格式: 将 PNG 序列无损打包为 PDF
img2pdf>=0.5.0
```

### 4. `README.md` 更新

- 「命令行参数」表: 新增 `-of, --output-formats` 行
-「输出结果」段: 重写为「输出结果与格式选择」, 列出 5 种格式 (txt 常驻 + 4 可选)
-「模型选择说明」表: 增加「输出格式兼容性」列
-「示例用法」: 加 `-of markdown,img` 和 `-of json` 示例
-「API 接口」: `/ocr/pdf` 新增 `output_formats` 参数说明 + `/download/` 端点说明

### 5. `Docker.md` 更新

- curl 示例加 `-F "output_formats=markdown,json"`

### 6. 测试: `tests/output_format_selector_test.py` (新建)

**约束：不用 mock 数据。使用真实 PDF + 真实 PaddleOCR。**

测试框架：
- 生成 2 页简单 PDF (用 `pypdfium2` 或 `PyPDF2` 写入)
- 所有测试标记 `@pytest.mark.integration`
- 使用 `model=tiny` 或最小模型加速
- 使用 `device=cpu` 避免 GPU 时序问题

| # | 测试方法 | 输入 | 验证 |
|---|----------|------|------|
| 1 | `test_default_output_formats_writes_all_four` | 默认 `-of` = all | 4 个新文件 + .txt 都存在 |
| 2 | `test_selective_output_formats_only_writes_md` | `-of markdown` | 只生成 .md 和 .txt, 不生成 .json/.png/.pdf |
| 3 | `test_pp_ocrv6_markdown_skipped` | `model=pp-ocrv6 -of markdown` | .md 不生成, logger.warning 输出 |
| 4 | `test_json_has_valid_structure` | `-of json`, 任意模型 | .json 文件存在, 含 `rec_texts` 或 `parsing_res_list` |
| 5 | `test_annotated_img_per_page` | `-of img`, 2 页 PDF | `_p001.png`、`_p002.png` 存在 |
| 6 | `test_layout_pdf_is_valid` | `-of pdf`, 2 页 PDF | `_annotated.pdf` 存在, pypdfium2 打开验证页数 |
| 7 | `test_api_returns_output_paths` | POST 带 `output_formats=markdown` | 响应含 `outputs.md`, 路径指向真实文件 |
| 8 | `test_api_invalid_format_rejected` | POST `output_formats=foo` | 400 `detail` 含提示 |
| 9 | `test_api_download_endpoint` | 先生成, 再 GET /download/... | 文件内容与本地一致 |

**预期测试队列总数:** 现有 45 + 新增 9 = **54 tests total**

### 7. 数据流 (CLI)

```
python ocr_pdf.py -i doc.pdf -o ./out -of markdown,img --device gpu
  ↓
args.output_formats = "markdown,img"
  ↓ 解析 (split + strip)
output_formats_list = ["markdown", "img"]
  ↓
PDFOCRHandler(output_dir="./out", output_formats=["markdown", "img"])
  ↓ process_pdf()
PDF 渲染 → ocr.predict() → 提取 text
  ↓
write ./out/doc.txt  (always)
write ./out/doc.md   (if 'markdown' in formats and model支持)
for page in pages: write ./out/doc_p001.png, doc_p002.png  (if 'img')
skip ./out/doc.json  (not in formats)
skip ./out/doc_annotated.pdf  (not in formats)
  ↓
append ocr_logs.md
```

### 8. 数据流 (API)

```
POST /ocr/pdf (file=doc.pdf, output_formats="json,pdf")
  ↓
output_formats_list = ["json", "pdf"]
  ↓ valid_models check passed
  ↓ temp_dir = api_outputs/{request_id}/
  ↓
PDFOCRHandler(output_dir=temp_dir/output, output_formats=["json", "pdf"])
  ↓ process_pdf() → 写 2 个文件 (+ 1 .txt)
  ↓
返回 JSON:
{
  "status": "success",
  "outputs": {
    "txt": "api_outputs/{req_id}/output/doc.txt",
    "json": "api_outputs/{req_id}/output/doc.json",
    "pdf": "api_outputs/{req_id}/output/doc_annotated.pdf"
  }
}
  ↓
用户 GET /download/api_outputs/{req_id}/output/doc.json 下载
```

## 错误处理

| 场景 | CLI 行为 | API 行为 |
|------|----------|----------|
| `-of foo` | argparse error: invalid choice | 400: detail 含可选列表 |
| `-of markdown` + `model=pp-ocrv6` | warning + 跳过 .md | 同 CLI + 响应不含 `outputs.md` |
| `save_to_json()` 抛异常 | logger.error + 继续下一个格式 | 同 CLI + `outputs.json` 不存在 |
| `save_to_img()` 抛异常 | logger.error + 跳过 img+pdf | 同 CLI |
| 磁盘写满 | IOError → exit 1 | 500 + error detail |
| `img2pdf` 未安装 | ImportError → 提示 `pip install img2pdf` | 500 + error detail |
| 空输出目录 | (正常) .txt 不存在 → process_pdf returns False | 500 |

## 测试策略

### 原则
1. **无 mock 数据** — 使用真实 PDF + 真实 PaddleOCR
2. **标记 `@pytest.mark.integration`** — 不混入单元测试
3. **使用小模型 + CPU 设备** — `model_size=tiny` + `device=cpu` 加速

### 测试 PDF 生成

测试用 2 页 PDF 动态创建 (fixture 或 helper 函数):

```python
@pytest.fixture(scope="module")
def two_page_pdf(tmp_path_factory):
    """生成 2 页真实 PDF 文件供测试"""
    from pypdfium2 import PdfDocument, PdfLibrary
    
    pdf_path = tmp_path_factory.mktemp("inputs") / "test.pdf"
    lib = PdfLibrary()
    doc = lib.new_doc()
    
    for i in range(2):
        page = doc.new_page(612, 792)  # letter size
        page.set_font_size(14)
        page.set_fill(0)
        page.draw_text(f"This is test page {i+1}.", 72, 700)
    
    doc.save(pdf_path.__str__())
    doc.close()
    return pdf_path
```

### 验证工具函数

```python
def get_page_count(pdf_path):
    """返回 PDF 页数"""
    import pypdfium2 as pdfium
    lib = pdfium.PdfLibrary()
    doc = lib.open_document(pdf_path.__str__())
    count = doc.get_page_count()
    doc.close()
    lib.close()
    return count
```

## 风险评估

| 风险 | 等级 | 缓解 |
|------|------|------|
| 测试慢 (9 集成测试 × ~15s = ~135s) | 中 | `@pytest.mark.integration` 默认跳过; CI 指定 `pytest -m integration` |
| `res.save_to_img()` 行为差异 (不同模型返回不同) | 中 | 3 模型各生成 1 页, 仅在 `test_img_per_page` 验证; 不混入格式兼容测试 |
| `img2pdf` 编码 1-bit PNG 问题 | 低 | 测试先验证; 如出问题改用 `Pillow` (`Images.save(fp, 'PDF')`) |
| API 输出目录膨胀 (未清理) | 中 | 惰性清理: 每次上传时扫描, 删除 >1h 的 dir; counts limit 1000 个 |
| `.txt` 仍包含 pp-ocrv5 引用 (遗留问题) | 低 | 另案处理 (chatocrv4 refactor plan) |
| `result` 字段 deprecated 导致下游断裂 | 中 | 保留 `result` 字段但文档标记 deprecated; 充分沟通 |

## 实施计划前瞻

建议实现顺序:
1. Task 1: `requirements.txt` + `img2pdf` 依赖
2. Task 2: `ocr_pdf.py` — CLI 参数 + `PDFOCRHandler.__init__()` + `process_pdf()` 格式分支
3. Task 3: `api.py` — Form 参数 + 校验 + 响应结构 + `/download/` 端点
4. Task 4: `tests/output_format_selector_test.py` — 9 个集成测试
5. Task 5: `README.md` + `Docker.md` 文档同步
6. Task 6: 最终验证 — 全部集成测试 + CLI 端到端 + API 端到端

---

## 批准记录

- 2026-06-22: 设计审批通过 (用户确认 Q1=A, Q2=c, Q3=a, Q4=a, Q5=B + 无 mock 测试)
