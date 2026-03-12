from __future__ import annotations

from ...application.services.weather_service import (
    lookup_weather,
    normalize_weather_prompt,
)
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "weather_lookup",
    description="查询气象与农事适宜度数据。支持默认农场查询，或根据用户提供的区域名称匹配 region_id 后调用接口；农事适宜度当前仅支持展示施肥、炼苗、移栽、翻地、打药、收割、整地，超出该范围需明确告知用户暂无法显示；未提供区域时默认使用农场，不要求作物信息。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    """规范化天气查询入参后，统一走 weather service 完成查询。"""
    prompt_text = prompt or ""
    cache_prompt, query = normalize_weather_prompt(prompt_text)
    return lookup_weather(prompt_text, cache_prompt=cache_prompt, query=query)
