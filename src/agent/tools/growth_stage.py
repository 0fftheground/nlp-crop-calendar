from __future__ import annotations

from ...infra.cache_keys import parse_planting_cache_key
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool, _get_cached_tool_result


@auto_register_tool(
    "growth_stage_prediction",
    description=(
        "仅用于读取已缓存的生育期预测结果。"
        "缺少缓存时需走生育期预测 workflow。"
    ),
)
def growth_stage_prediction(prompt: str) -> ToolInvocation:
    provider = "workflow"
    cache_key = parse_planting_cache_key(prompt)
    if cache_key:
        cached = _get_cached_tool_result(
            "growth_stage_prediction", provider, cache_key
        )
        if cached:
            return cached
    return ToolInvocation(
        name="growth_stage_prediction",
        message="生育期预测必须走工作流，当前仅支持返回历史缓存结果。",
        data={"cache_hit": False},
    )
