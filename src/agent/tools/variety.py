from __future__ import annotations

from ...application.services.variety_service import lookup_variety
from ...infra.config import get_config
from ...infra.tool_provider import normalize_provider
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool, _get_cached_tool_result, _store_tool_result


@auto_register_tool(
    "variety_lookup",
    description=(
        "查询水稻品种基础信息。仅用于用户明确询问品种特性/抗性/生育期等单点信息；"
        "不用于完整种植方案。"
    ),
)
def variety_lookup(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.variety_provider)
    cached = _get_cached_tool_result("variety_lookup", provider, prompt)
    if cached:
        return cached
    result = lookup_variety(prompt)
    _store_tool_result("variety_lookup", provider, prompt, result)
    return result
