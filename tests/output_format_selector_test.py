"""tests/output_format_selector_test.py
MinerU-style output integration tests.
Uses real PDFs + real PaddleOCR (no mocks).
Run: pytest tests/output_format_selector_test.py -v -m integration
"""
import sys
import os
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
    """生成 2 页真实 PDF 文件供测试 (使用 pikepdf 创建)"""
    import pikepdf

    pdf_path = tmp_path_factory.mktemp("inputs") / "test.pdf"
    pdf = pikepdf.Pdf.new()

    for i in range(2):
        pdf.add_blank_page(page_size=(612, 792))  # letter size

    for i, page in enumerate(pdf.pages):
        content = (
            f"BT "
            f"/F1 14 Tf "
            f"72 700 Td "
            f"(This is test page {i+1}.) Tj "
            f"ET"
        )
        page.Contents = pdf.make_stream(content.encode())

    pdf.save(pdf_path)
    pdf.close()
    return pdf_path


def get_page_count(pdf_path):
    """返回 PDF 页数"""
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(pdf_path) as doc:
        return len(doc)


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

class TestMinerUOutput:
    """MinerU 风格输出集成测试 — 使用真实数据"""

    @pytest.mark.integration
    def test_mineru_directory_structure(self, two_page_pdf, tmp_path):
        """处理后应生成 {stem}/images/page_N.jpg + output.md + layout.pdf"""
        out_dir = tmp_path / "out_mineru"
        out_dir.mkdir()
        handler = _create_handler(out_dir)

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True, "process_pdf 应返回 True"

        stem = two_page_pdf.stem
        pdf_dir = out_dir / stem
        images_dir = pdf_dir / "images"

        assert pdf_dir.exists(), f"{stem}/ 目录应存在"
        assert images_dir.exists(), f"{stem}/images/ 目录应存在"
        assert (pdf_dir / "output.md").exists(), "output.md 应存在"
        assert (pdf_dir / "layout.pdf").exists(), "layout.pdf 应存在"

        # 检查图片: 应为 JPG 格式
        jpgs = sorted(images_dir.iterdir())
        assert len(jpgs) >= 1, "应有至少 1 个 JPG 图片"
        for jpg in jpgs:
            assert jpg.suffix == '.jpg', f"图片应为 .jpg 格式: {jpg.name}"
            assert jpg.stat().st_size > 0, f"{jpg.name} 不应为空"

    @pytest.mark.integration
    def test_images_per_page(self, two_page_pdf, tmp_path):
        """JPG 数量应与 PDF 页数一致"""
        out_dir = tmp_path / "out_img_count"
        out_dir.mkdir()
        handler = _create_handler(out_dir)

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True

        jpgs = list((out_dir / two_page_pdf.stem / "images").glob("*.jpg"))
        assert len(jpgs) == 2, f"2 页 PDF 应生成 2 张 JPG, 实际: {len(jpgs)}"

    @pytest.mark.integration
    def test_markdown_has_content(self, two_page_pdf, tmp_path):
        """output.md 应包含页面引用和文本内容"""
        out_dir = tmp_path / "out_md_content"
        out_dir.mkdir()
        handler = _create_handler(out_dir)

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True

        md_path = out_dir / two_page_pdf.stem / "output.md"
        assert md_path.exists()

        content = md_path.read_text(encoding='utf-8')
        assert "test page" in content or "This is test page" in content, \
            "output.md 应包含 PDF 中的文本内容"
        assert "page_" in content or "images/" in content, \
            "output.md 应包含图片引用"

    @pytest.mark.integration
    def test_layout_pdf_is_valid(self, two_page_pdf, tmp_path):
        """layout.pdf 应为有效的 PDF 文件"""
        out_dir = tmp_path / "out_layout"
        out_dir.mkdir()
        handler = _create_handler(out_dir)

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True

        layout_path = out_dir / two_page_pdf.stem / "layout.pdf"
        assert layout_path.exists()
        assert layout_path.stat().st_size > 0, "layout.pdf 不应为空"

        page_count = get_page_count(str(layout_path))
        assert page_count >= 1, f"layout.pdf 应有至少 1 页, 实际: {page_count}"

    @pytest.mark.integration
    def test_old_output_formats_param_still_accepted(self, two_page_pdf, tmp_path):
        """旧的 output_formats 参数（已废弃）传入不报错, 输出结构不变"""
        out_dir = tmp_path / "out_compat"
        out_dir.mkdir()
        handler = _create_handler(out_dir, output_formats=['markdown', 'json'])

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True

        # 输出始终是 MinerU 结构，不受 output_formats 影响
        stem = two_page_pdf.stem
        assert (out_dir / stem / "output.md").exists(), "output.md 应始终生成"
        assert (out_dir / stem / "layout.pdf").exists(), "layout.pdf 应始终生成"

    @pytest.mark.integration
    def test_pp_ocrv6_model_works(self, two_page_pdf, tmp_path):
        """pp-ocrv6 模型应能正常处理并生成 MinerU 输出"""
        out_dir = tmp_path / "out_v6"
        out_dir.mkdir()
        handler = _create_handler(out_dir, model='pp-ocrv6')

        result = handler.process_pdf(str(two_page_pdf))
        assert result is True

        stem = two_page_pdf.stem
        assert (out_dir / stem / "output.md").exists(), "output.md 应存在"
        assert (out_dir / stem / "layout.pdf").exists(), "layout.pdf 应存在"
