from __future__ import annotations

from ...application.services.weather_service import (
    lookup_weather,
    normalize_weather_prompt,
)
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "weather_lookup",
    description="查询默认农场气象数据（仅使用默认农场ID）。需要提供起止日期（最多30天）。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    return lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
