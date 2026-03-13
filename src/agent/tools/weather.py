from __future__ import annotations

import json

from ...application.services.weather_service import (
    apply_weather_operation_view,
    lookup_weather,
    normalize_weather_prompt,
    parse_weather_prompt_operations,
)
from ...infra.tool_provider import normalize_provider
from .registry import (
    TOOL_CACHEABLE,
    _get_cached_tool_result,
    _store_tool_result,
    auto_register_tool,
)
from ...schemas.models import ToolInvocation


WEATHER_PROVIDER = normalize_provider("agri_weather_api")
TOOL_CACHEABLE.add("weather_lookup")


@auto_register_tool(
    "weather_lookup",
    description="查询气象与农事适宜度数据。支持默认农场查询，或根据用户提供的区域名称匹配 region_id 后调用接口；农事适宜度当前仅支持展示施肥、炼苗、移栽、翻地、打药、收割、整地，超出该范围需明确告知用户暂无法显示；未提供区域时默认使用农场，不要求作物信息。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    """规范化天气查询入参后，统一走 weather service 完成查询。"""
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    requested_operations, _, unsupported_note = parse_weather_prompt_operations(prompt_text)
    if not requested_operations:
        try:
            payload = json.loads(prompt_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            requested_operations = [
                str(item).strip()
                for item in list(payload.get("requested_operations") or [])
                if str(item).strip()
            ]
    if query is None:
        return lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
    cached = _get_cached_tool_result("weather_lookup", WEATHER_PROVIDER, cache_prompt)
    if cached:
        return apply_weather_operation_view(
            cached,
            requested_operations=requested_operations,
            unsupported_note=unsupported_note,
        )
    base_result = lookup_weather(cache_prompt, cache_prompt=cache_prompt, query=query)
    _store_tool_result("weather_lookup", WEATHER_PROVIDER, cache_prompt, base_result)
    return apply_weather_operation_view(
        base_result,
        requested_operations=requested_operations,
        unsupported_note=unsupported_note,
    )
