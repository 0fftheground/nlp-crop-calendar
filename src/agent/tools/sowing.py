from __future__ import annotations

from ...application.services.sowing_suitability_service import (
    lookup_sowing_suitability,
)
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "sowing_suitability_lookup",
    description=(
        "查询水稻播期推荐。根据品种名、稻作类型、播种方式以及区域或默认农场，"
        "调用播期推荐接口返回推荐播期和不推荐原因。"
    ),
)
def sowing_suitability_lookup(prompt: str) -> ToolInvocation:
    """查询播期推荐；缺少关键字段时返回追问态。"""
    return lookup_sowing_suitability(prompt or "")
