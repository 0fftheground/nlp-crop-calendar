from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional, Tuple, List

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from ...infra.config import get_config
from ...infra.export_store import resolve_export_path, write_export
from ...infra.llm import get_chat_model
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.weather_cache import (
    get_weather_series,
    make_weather_grid_cache_key,
    store_weather_series,
)
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


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_lat_lon_from_text(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    lat_match = re.search(r"(?:lat|纬度)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
    lon_match = re.search(r"(?:lon|lng|经度)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if lat_match and lon_match:
        lat = _parse_float(lat_match.group(1))
        lon = _parse_float(lon_match.group(1))
        if lat is not None and lon is not None:
            return lat, lon
    if any(token in text for token in ("坐标", "经纬", "lat", "lon", "纬度", "经度")):
        pair_match = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", text)
        if pair_match:
            lat = _parse_float(pair_match.group(1))
            lon = _parse_float(pair_match.group(2))
            if lat is not None and lon is not None:
                return lat, lon
    return None


def _build_weather_query_from_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    region = payload.get("region")
    lat = _parse_float(payload.get("lat") or payload.get("latitude") or payload.get("纬度"))
    lon = _parse_float(payload.get("lon") or payload.get("longitude") or payload.get("lng") or payload.get("经度"))
    if not region:
        if lat is not None and lon is not None:
            region = f"{lat},{lon}"
        else:
            return None
    year = _parse_year(payload.get("year"))
    if year is None:
        year = _parse_year(payload.get("start_date")) or _parse_year(payload.get("end_date"))
    data: Dict[str, object] = {"region": region}
    if lat is not None:
        data["lat"] = lat
    if lon is not None:
        data["lon"] = lon
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
        lat_lon = _extract_lat_lon_from_text(text)
        if lat_lon:
            lat, lon = lat_lon
            region = f"{lat},{lon}"
            try:
                query = WeatherQueryInput(
                    region=region,
                    lat=lat,
                    lon=lon,
                    year=date.today().year,
                )
            except Exception:
                return text, None
        else:
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


def _parse_forecast_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _coerce_forecast_points(payload: object) -> List[WeatherDataPoint]:
    if not isinstance(payload, list):
        return []
    points: List[WeatherDataPoint] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        day = _parse_forecast_date(
            item.get("date")
            or item.get("fxDate")
            or item.get("day")
            or item.get("forecast_date")
            or item.get("datatime")
        )
        if day is None:
            continue
        temp_max = _parse_float(
            item.get("tmax")
            or item.get("tMax")
            or item.get("temp_max")
            or item.get("tempMax")
            or item.get("t_max")
            or item.get("max")
            or item.get("high")
        )
        temp_min = _parse_float(
            item.get("tmin")
            or item.get("tMin")
            or item.get("temp_min")
            or item.get("tempMin")
            or item.get("t_min")
            or item.get("min")
            or item.get("low")
        )
        temp = _parse_float(
            item.get("tavg")
            or item.get("temp")
            or item.get("temperature")
            or item.get("t_avg")
            or item.get("avg")
        )
        humidity = _parse_float(item.get("rh") or item.get("humidity"))
        precipitation = _parse_float(
            item.get("precip")
            or item.get("precipitation")
            or item.get("rain")
            or item.get("pre")
        )
        wind_speed = _parse_float(
            item.get("windSpeed")
            or item.get("wind_speed")
            or item.get("wind")
            or item.get("wins")
        )
        condition = (
            item.get("wp_pm")
            or item.get("wp_am")
            or item.get("condition")
            or item.get("text")
            or item.get("weather")
            or item.get("wind_describe")
        )
        points.append(
            WeatherDataPoint(
                timestamp=datetime.combine(day, time.min),
                temperature=temp,
                temperature_max=temp_max,
                temperature_min=temp_min,
                humidity=humidity,
                precipitation=precipitation,
                wind_speed=wind_speed,
                condition=str(condition) if condition is not None else None,
            )
        )
    return points


def _build_91weather_series(
    payload: object, query: WeatherQueryInput
) -> Optional[WeatherSeries]:
    if not isinstance(payload, dict):
        return None
    data = (
        payload.get("data")
        or payload.get("forecast")
        or payload.get("result")
        or payload.get("list")
    )
    if isinstance(data, dict) and "data" in data:
        data = data.get("data")
    points = _coerce_forecast_points(data)
    if not points:
        return None
    start_date = points[0].timestamp.date()
    end_date = points[-1].timestamp.date()
    return WeatherSeries(
        region=query.region,
        granularity=query.granularity or "daily",
        start_date=start_date,
        end_date=end_date,
        points=points,
        source="91weather",
    )


def _build_lat_lon_followup(query: Optional[WeatherQueryInput]) -> ToolInvocation:
    draft: Dict[str, object] = {}
    if query:
        draft = query.model_dump(mode="json")
    return ToolInvocation(
        name="weather_lookup",
        message="需要经纬度才能调用外部气象接口，请补充纬度(lat)与经度(lon)。",
        data={
            "missing_fields": ["lat", "lon"],
            "draft": draft,
            "followup_count": 0,
        },
    )


def _build_weather_csv(series: WeatherSeries) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "date",
            "temperature_avg",
            "temperature_max",
            "temperature_min",
            "humidity",
            "precipitation",
            "wind_speed",
            "condition",
        ]
    )
    for point in series.points:
        writer.writerow(
            [
                point.timestamp.date().isoformat(),
                point.temperature if point.temperature is not None else "",
                point.temperature_max if point.temperature_max is not None else "",
                point.temperature_min if point.temperature_min is not None else "",
                point.humidity if point.humidity is not None else "",
                point.precipitation if point.precipitation is not None else "",
                point.wind_speed if point.wind_speed is not None else "",
                point.condition or "",
            ]
        )
    return output.getvalue()


def _ensure_weather_export(
    series: WeatherSeries,
) -> Tuple[str, str, WeatherSeries, bool]:
    updated = False
    file_id = series.export_file_id
    export_path = series.export_path
    path = None
    if file_id:
        try:
            path = resolve_export_path(file_id)
        except ValueError:
            path = None
    if path and path.exists():
        current_path = str(path)
        if export_path != current_path:
            series = series.model_copy(update={"export_path": current_path})
            updated = True
        return file_id, current_path, series, updated
    csv_content = _build_weather_csv(series)
    file_id = write_export(csv_content, suffix="csv")
    path = resolve_export_path(file_id)
    export_path = str(path)
    series = series.model_copy(
        update={"export_file_id": file_id, "export_path": export_path}
    )
    updated = True
    return file_id, export_path, series, updated


def _summarize_weather_series(series: WeatherSeries) -> str:
    temps: List[float] = []
    tmax: List[float] = []
    tmin: List[float] = []
    humidity: List[float] = []
    precipitation: List[float] = []
    wind: List[float] = []
    condition_counts: Dict[str, int] = {}

    for point in series.points:
        if point.temperature is not None:
            temps.append(point.temperature)
        elif point.temperature_max is not None and point.temperature_min is not None:
            temps.append((point.temperature_max + point.temperature_min) / 2)
        if point.temperature_max is not None:
            tmax.append(point.temperature_max)
        if point.temperature_min is not None:
            tmin.append(point.temperature_min)
        if point.humidity is not None:
            humidity.append(point.humidity)
        if point.precipitation is not None:
            precipitation.append(point.precipitation)
        if point.wind_speed is not None:
            wind.append(point.wind_speed)
        if point.condition:
            condition_counts[point.condition] = (
                condition_counts.get(point.condition, 0) + 1
            )

    stats = {
        "region": series.region,
        "start_date": series.start_date.isoformat() if series.start_date else None,
        "end_date": series.end_date.isoformat() if series.end_date else None,
        "days": len(series.points),
        "temp_avg": round(sum(temps) / len(temps), 2) if temps else None,
        "temp_max": round(max(tmax), 2) if tmax else None,
        "temp_min": round(min(tmin), 2) if tmin else None,
        "precip_total": round(sum(precipitation), 2) if precipitation else None,
        "humidity_avg": round(sum(humidity) / len(humidity), 2) if humidity else None,
        "wind_avg": round(sum(wind) / len(wind), 2) if wind else None,
        "conditions": sorted(
            condition_counts.items(), key=lambda item: item[1], reverse=True
        )[:3],
    }

    fallback_parts = [f"{stats['region']} 未来{stats['days']}天"]
    if stats["temp_min"] is not None and stats["temp_max"] is not None:
        fallback_parts.append(
            f"气温 {stats['temp_min']}~{stats['temp_max']}°C"
        )
    if stats["precip_total"] is not None:
        fallback_parts.append(f"累计降水 {stats['precip_total']}mm")
    if stats["conditions"]:
        fallback_parts.append(
            "主要天气: " + "、".join([item[0] for item in stats["conditions"]])
        )
    fallback = "；".join(fallback_parts)

    try:
        model = get_chat_model()
        system_prompt = (
            "你是气象助理，请基于统计信息输出简洁摘要。"
            "要求：2-3 句中文，包含温度范围、降水概况和主要天气。"
        )
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(stats, ensure_ascii=True)),
            ]
        )
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        return fallback
    return fallback


def _lookup_91weather(
    query: Optional[WeatherQueryInput],
    *,
    api_url: Optional[str],
) -> ToolInvocation:
    if not query or query.lat is None or query.lon is None:
        return _build_lat_lon_followup(query)
    cache_key = make_weather_grid_cache_key(
        query.lat,
        query.lon,
        day=date.today(),
    )
    cached_series = get_weather_series(cache_key)
    if cached_series:
        series = cached_series
        updated = False
        if query.region and cached_series.region != query.region:
            series = cached_series.model_copy(
                update={"region": query.region, "summary": None}
            )
            updated = True
        summary = series.summary
        if not summary:
            summary = _summarize_weather_series(series)
            series = series.model_copy(update={"summary": summary})
            updated = True
        file_id, _, series, export_updated = _ensure_weather_export(series)
        if export_updated:
            updated = True
        if updated:
            store_weather_series(series, cache_key=cache_key)
        download_url = f"/api/v1/download/{file_id}"
        data_payload = series.model_dump(mode="json")
        data_payload["summary"] = summary
        data_payload["download_url"] = download_url
        message = f"{summary}\n下载链接: {download_url}"
        return ToolInvocation(
            name="weather_lookup",
            message=message,
            data=data_payload,
        )
    url = api_url or "https://data-api.91weather.com/Zoomlion/higf_day_plus"
    params = {"lat": query.lat, "lon": query.lon}
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return ToolInvocation(
            name="weather_lookup",
            message=f"外部气象接口请求失败: {exc}",
            data={},
        )
    series = _build_91weather_series(payload, query)
    if series:
        summary = _summarize_weather_series(series)
        series = series.model_copy(update={"summary": summary})
        file_id, _, series, _ = _ensure_weather_export(series)
        store_weather_series(series, cache_key=cache_key)
        download_url = f"/api/v1/download/{file_id}"
        data_payload = series.model_dump(mode="json")
        data_payload["summary"] = summary
        data_payload["download_url"] = download_url
        message = (
            f"{summary}\n下载链接: {download_url}"
        )
        return ToolInvocation(
            name="weather_lookup",
            message=message,
            data=data_payload,
        )
    return ToolInvocation(
        name="weather_lookup",
        message="外部气象接口返回格式未识别。",
        data={"payload": payload},
    )


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
    if provider in {"91weather", "external"}:
        return _lookup_91weather(query, api_url=cfg.weather_api_url)
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
