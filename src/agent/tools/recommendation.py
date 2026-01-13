from __future__ import annotations

from ...application.services.recommendation_service import recommend_farming
from ...infra.config import get_config
from ...infra.tool_provider import normalize_provider
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool, _get_cached_tool_result, _store_tool_result


@auto_register_tool(
    "farming_recommendation",
    description=(
        "生成简短农事建议（单段建议）。仅在用户已提供作物/地区/时间或生育期时使用；"
        "若用户要完整种植计划/全流程安排，请走 workflow。"
    ),
)
def farming_recommendation(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.recommendation_provider)
    cached = _get_cached_tool_result("farming_recommendation", provider, prompt)
    if cached:
        return cached
    result = recommend_farming(prompt)
    _store_tool_result("farming_recommendation", provider, prompt, result)
    return result
