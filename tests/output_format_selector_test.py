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
