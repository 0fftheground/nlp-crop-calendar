from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional, Tuple, List

from ...infra.config import get_config
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...schemas.models import (
    ToolInvocation,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)


def _parse_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, date):
        return value.year
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) == 4:
            return int(text)
        try:
            return date.fromisoformat(text).year
        except ValueError:
            return None
    return None


def _build_weather_query_from_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    region = payload.get("region")
    if not region:
        return None
    year = _parse_year(payload.get("year"))
    if year is None:
        year = _parse_year(payload.get("start_date")) or _parse_year(payload.get("end_date"))
    data: Dict[str, object] = {"region": region}
    if year is not None:
        data["year"] = year
    granularity = payload.get("granularity")
    if granularity in {"hourly", "daily"}:
        data["granularity"] = granularity
    include_advice = payload.get("include_advice")
    if isinstance(include_advice, bool):
        data["include_advice"] = include_advice
    try:
        return WeatherQueryInput(**data)
    except Exception:
        return None


def normalize_weather_prompt(
    prompt: str,
) -> Tuple[str, Optional[WeatherQueryInput]]:
    if not prompt:
        return "", None
    text = prompt.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            query = WeatherQueryInput(region=text, year=date.today().year)
        except Exception:
            return text, None
        canonical = json.dumps(
            query.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        return canonical, query
    if not isinstance(payload, dict):
        return text, None
    query = _build_weather_query_from_payload(payload)
    if query is None:
        return text, None
    canonical = json.dumps(
        query.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
    return canonical, query


def lookup_weather(
    prompt: str,
    *,
    cache_prompt: Optional[str] = None,
    query: Optional[WeatherQueryInput] = None,
) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.weather_provider)
    text = prompt or ""
    if cache_prompt is None or query is None:
        cache_prompt, query = normalize_weather_prompt(text)
    intranet_prompt = cache_prompt if query else text
    intranet = maybe_intranet_tool(
        "weather_lookup",
        intranet_prompt,
        provider,
        cfg.weather_api_url,
        cfg.weather_api_key,
    )
    if intranet:
        return intranet
    if query:
        start = date(query.year, 1, 1)
        end = date(query.year, 12, 31)
        granularity = query.granularity or "daily"
        region = query.region
        total_days = max(1, (end - start).days + 1)
    else:
        start = date.today()
        end = start + timedelta(days=2)
        granularity = "daily"
        region = text or "unknown"
        total_days = 3
    points: List[WeatherDataPoint] = []
    for offset in range(total_days):
        current = start + timedelta(days=offset)
        points.append(
            WeatherDataPoint(
                timestamp=datetime.combine(current, time.min),
                temperature=20 + offset,
                temperature_max=26 + offset,
                temperature_min=16 + offset,
                humidity=60 + offset * 2,
                precipitation=0.5 * offset,
                wind_speed=2.0 + 0.3 * offset,
                condition="sunny" if offset < 2 else "cloudy",
            )
        )
    series = WeatherSeries(
        region=region,
        granularity=granularity,
        start_date=start,
        end_date=end,
        points=points,
        source="mock",
    )
    return ToolInvocation(
        name="weather_lookup",
        message="已返回模拟气象数据。",
        data=series.model_dump(mode="json"),
    )
