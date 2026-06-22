"""
routers/output_format.py — 输出格式常量与校验逻辑

提供:
  - VALID_OUTPUT_FORMATS: 支持的输出格式列表
  - validate_output_formats(): 将逗号分隔字符串解析并校验为列表

此模块不依赖 api.py 中的 handler 缓存或 FastAPI 应用实例，
可安全地被 api.py 和测试模块导入。
"""

from typing import Optional

# ---------------------------------------------------------------------------
# 支持的输出格式
# ---------------------------------------------------------------------------
VALID_OUTPUT_FORMATS = ["markdown", "json", "img", "pdf"]


def validate_output_formats(raw: Optional[str]) -> Optional[list[str]]:
    """将逗号分隔的输出格式字符串解析为列表并校验。

    Args:
        raw: 逗号分隔的格式字符串，如 "markdown,json"，None 表示不限制。

    Returns:
        合法的格式名列表，或 None（当 raw 为 None/空时）。

    Raises:
        ValueError: 当包含不受支持的格式名时抛出，消息中列出非法值。
    """
    if not raw:
        return None

    formats = [f.strip() for f in raw.split(",") if f.strip()]
    invalid = set(formats) - set(VALID_OUTPUT_FORMATS)
    if invalid:
        raise ValueError(
            f"无效的输出格式: {', '.join(sorted(invalid))}。"
            f" 可选: {', '.join(VALID_OUTPUT_FORMATS)}"
        )
    return formats if formats else None
