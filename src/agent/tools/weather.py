from __future__ import annotations

from ...application.services.weather_service import (
    lookup_weather,
    normalize_weather_prompt,
)
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "weather_lookup",
    description="查询气象数据。支持默认农场天气，或根据用户提供的区域名称匹配区域表后按 region_id 查询。需要提供起止日期（最多30天）。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    """规范化天气查询入参后，统一走 weather service 完成查询。"""
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    return lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
