from __future__ import annotations

from ...application.services.weather_service import (
    lookup_weather,
    normalize_weather_prompt,
)
from ...infra.config import get_config
from ...infra.tool_provider import normalize_provider
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool, _get_cached_tool_result, _store_tool_result


@auto_register_tool(
    "weather_lookup",
    description="查询指定地区气象数据。仅用于获取天气数据本身；不生成农事建议或计划。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.weather_provider)
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    cached = _get_cached_tool_result("weather_lookup", provider, cache_prompt)
    if cached:
        return cached
    result = lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
    _store_tool_result("weather_lookup", provider, cache_prompt, result)
    return result
