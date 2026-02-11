from __future__ import annotations

from ...application.services.weather_service import (
    lookup_weather,
    normalize_weather_prompt,
)
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "weather_lookup",
    description="查询指定地区气象数据。仅用于获取天气数据本身；不生成农事建议或计划。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    return lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
