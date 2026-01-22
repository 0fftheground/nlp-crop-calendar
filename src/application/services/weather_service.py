from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from ...infra.config import get_config
from ...infra.export_store import resolve_export_path, write_export
from ...infra.geocode_cache import get_geocode_cached, set_geocode_cached
from ...infra.llm import get_chat_model
from ...infra.tool_provider import normalize_provider
from ...infra.weather_archive_store import (
    build_weather_archive_path,
    get_weather_archive_store,
)
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


def _parse_payload_date(value: object) -> Optional[date]:
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


def _parse_geocode_location(location: object) -> Optional[Tuple[float, float]]:
    if location is None:
        return None
    parts = str(location).split(",")
    if len(parts) != 2:
        return None
    lon = _parse_float(parts[0])
    lat = _parse_float(parts[1])
    if lat is None or lon is None:
        return None
    return lat, lon


def _normalize_geocode_city(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _geocode_with_amap(
    region: str,
    *,
    api_key: Optional[str],
    geocode_url: Optional[str],
) -> Optional[dict]:
    if not region:
        return None
    address = region.strip()
    cached = get_geocode_cached(region)
    if cached:
        return cached
    if not api_key:
        return None
    url = geocode_url or "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": api_key, "address": address}
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status")) != "1":
        return None
    geocodes = payload.get("geocodes")
    if not isinstance(geocodes, list) or not geocodes:
        return None
    primary = geocodes[0] if isinstance(geocodes[0], dict) else None
    if not primary:
        return None
    location = _parse_geocode_location(primary.get("location"))
    if location is None:
        return None
    lat, lon = location
    result = {
        "address": region,
        "formatted_address": primary.get("formatted_address"),
        "lat": lat,
        "lon": lon,
        "province": primary.get("province"),
        "city": _normalize_geocode_city(primary.get("city")),
        "district": primary.get("district"),
        "level": primary.get("level"),
        "adcode": primary.get("adcode"),
    }
    set_geocode_cached(region, None, result)
    return result


def _build_weather_query_from_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    region = payload.get("region")
    lat = _parse_float(payload.get("lat") or payload.get("latitude") or payload.get("纬度"))
    lon = _parse_float(payload.get("lon") or payload.get("longitude") or payload.get("lng") or payload.get("经度"))
    start_date = _parse_payload_date(
        payload.get("start_date") or payload.get("start")
    )
    end_date = _parse_payload_date(payload.get("end_date") or payload.get("end"))
    if not region:
        if lat is not None and lon is not None:
            region = f"{lat},{lon}"
        else:
            return None
    year = _parse_year(payload.get("year"))
    if year is None:
        year = _parse_year(start_date) or _parse_year(end_date)
    if start_date is not None:
        year = start_date.year
    elif end_date is not None:
        year = end_date.year
    data: Dict[str, object] = {"region": region}
    if lat is not None:
        data["lat"] = lat
    if lon is not None:
        data["lon"] = lon
    if start_date is not None:
        data["start_date"] = start_date
    if end_date is not None:
        data["end_date"] = end_date
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
            ensure_ascii=False,
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
        ensure_ascii=False,
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
            or item.get("datetime")
            or item.get("ymd")
            or item.get("date_time")
        )
        if day is None:
            continue
        temp_max = _parse_float(
            item.get("tmax")
            or item.get("tMax")
            or item.get("temp_max")
            or item.get("tempMax")
            or item.get("t_max")
            or item.get("tem_max")
            or item.get("max")
            or item.get("high")
        )
        temp_min = _parse_float(
            item.get("tmin")
            or item.get("tMin")
            or item.get("temp_min")
            or item.get("tempMin")
            or item.get("t_min")
            or item.get("tem_min")
            or item.get("min")
            or item.get("low")
        )
        temp = _parse_float(
            item.get("tavg")
            or item.get("temp")
            or item.get("temperature")
            or item.get("t_avg")
            or item.get("tem")
            or item.get("tmp")
            or item.get("tem_avg")
            or item.get("avg")
        )
        humidity = _parse_float(
            item.get("rh")
            or item.get("humidity")
            or item.get("rhu_avg")
            or item.get("rhu")
        )
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
            or item.get("win_s_2mi_avg")
            or item.get("win_s_max")
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


def _resolve_existing_export(
    series: WeatherSeries,
) -> Tuple[Optional[str], Optional[Path]]:
    if series.export_file_id:
        try:
            path = resolve_export_path(series.export_file_id)
        except ValueError:
            path = None
        if path and path.exists():
            return series.export_file_id, path
    if series.export_path:
        path = Path(series.export_path)
        if path.exists():
            file_id = series.export_file_id or path.stem
            return file_id, path
    return None, None


def _load_weather_points_from_csv(path: Path) -> List[WeatherDataPoint]:
    points: List[WeatherDataPoint] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                day = _parse_forecast_date(row.get("date"))
                if day is None:
                    continue
                points.append(
                    WeatherDataPoint(
                        timestamp=datetime.combine(day, time.min),
                        temperature=_parse_float(row.get("temperature_avg")),
                        temperature_max=_parse_float(row.get("temperature_max")),
                        temperature_min=_parse_float(row.get("temperature_min")),
                        humidity=_parse_float(row.get("humidity")),
                        precipitation=_parse_float(row.get("precipitation")),
                        wind_speed=_parse_float(row.get("wind_speed")),
                        condition=row.get("condition") or None,
                    )
                )
    except Exception:
        return []
    return points


def _hydrate_series_from_export(
    series: WeatherSeries, path: Path
) -> Optional[WeatherSeries]:
    points = _load_weather_points_from_csv(path)
    if not points:
        return None
    start_date = points[0].timestamp.date()
    end_date = points[-1].timestamp.date()
    return series.model_copy(
        update={"points": points, "start_date": start_date, "end_date": end_date}
    )


def _trim_weather_series_for_cache(series: WeatherSeries) -> WeatherSeries:
    if series.points:
        return series.model_copy(update={"points": []})
    return series


def _ensure_weather_export(
    series: WeatherSeries,
) -> Tuple[str, str, WeatherSeries, bool]:
    updated = False
    file_id = series.export_file_id
    export_path = series.export_path
    file_id, path = _resolve_existing_export(series)
    if path:
        current_path = str(path)
        if file_id and series.export_file_id != file_id:
            series = series.model_copy(update={"export_file_id": file_id})
            updated = True
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


def _build_download_url(file_id: str, *, base_url: Optional[str]) -> str:
    if not base_url:
        return f"/api/v1/download/{file_id}"
    base = base_url.rstrip("/")
    return f"{base}/api/v1/download/{file_id}"


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

    summary_parts = [f"{stats['region']} 未来{stats['days']}天"]
    if stats["temp_min"] is not None and stats["temp_max"] is not None:
        summary_parts.append(f"气温 {stats['temp_min']}~{stats['temp_max']}°C")
    if stats["precip_total"] is not None:
        summary_parts.append(f"累计降水 {stats['precip_total']}mm")
    if stats["conditions"]:
        summary_parts.append(
            "主要天气: " + "、".join([item[0] for item in stats["conditions"]])
        )
    template_summary = "；".join(summary_parts)
    cfg = get_config()
    mode = (cfg.weather_summary_mode or "template").lower()
    if mode != "llm":
        return template_summary

    try:
        model = get_chat_model()
        system_prompt = (
            "你是气象助理，请基于统计信息输出简洁摘要。"
            "要求：2-3 句中文，包含温度范围、降水概况和主要天气。"
        )
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(stats, ensure_ascii=False)),
            ]
        )
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        return template_summary
    return template_summary


def _resolve_query_range(query: WeatherQueryInput) -> Tuple[date, date]:
    base_year = query.year
    if query.start_date:
        base_year = query.start_date.year
    elif query.end_date:
        base_year = query.end_date.year
    start = query.start_date or date(base_year, 1, 1)
    end = query.end_date or date(base_year, 12, 31)
    return start, end


def _build_archive_series(
    query: WeatherQueryInput, path: Path
) -> Optional[WeatherSeries]:
    points = _load_weather_points_from_csv(path)
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


def _persist_weather_archive(
    series: WeatherSeries,
    *,
    region: str,
    lat: float,
    lon: float,
    year: int,
) -> Optional[str]:
    try:
        path = build_weather_archive_path(region, lat, lon, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_build_weather_csv(series), encoding="utf-8")
    except Exception:
        return None
    return str(path)


def _lookup_91weather(
    query: Optional[WeatherQueryInput],
    *,
    api_url: Optional[str],
) -> ToolInvocation:
    cfg = get_config()
    if not query or query.lat is None or query.lon is None:
        if query and query.region:
            geocode = _geocode_with_amap(
                query.region,
                api_key=cfg.amap_api_key,
                geocode_url=cfg.amap_geocode_url,
            )
            if geocode:
                query = query.model_copy(
                    update={"lat": geocode["lat"], "lon": geocode["lon"]}
                )
                formatted = geocode.get("formatted_address")
                if formatted and formatted != query.region:
                    query = query.model_copy(update={"region": formatted})
        if not query or query.lat is None or query.lon is None:
            return _build_lat_lon_followup(query)
    cache_key = make_weather_grid_cache_key(
        query.lat,
        query.lon,
        day=date.today(),
    )
    cached_series = get_weather_series(cache_key)
    if cached_series:
        existing_file_id, existing_path = _resolve_existing_export(cached_series)
        if not existing_path:
            cached_series = None
        else:
            updated = False
            response_series = cached_series
            hydrated = _hydrate_series_from_export(
                cached_series, existing_path
            )
            if hydrated:
                response_series = hydrated
            if query.region and cached_series.region != query.region:
                cached_series = cached_series.model_copy(
                    update={"region": query.region, "summary": None}
                )
                response_series = response_series.model_copy(
                    update={"region": query.region, "summary": None}
                )
                updated = True
            summary = cached_series.summary
            if not summary:
                summary = _summarize_weather_series(response_series)
                cached_series = cached_series.model_copy(
                    update={"summary": summary}
                )
                response_series = response_series.model_copy(
                    update={"summary": summary}
                )
                updated = True
            if cached_series.export_file_id and not response_series.export_file_id:
                response_series = response_series.model_copy(
                    update={
                        "export_file_id": cached_series.export_file_id,
                        "export_path": cached_series.export_path,
                    }
                )
            file_id, _, export_series, export_updated = _ensure_weather_export(
                response_series
            )
            response_series = export_series
            if export_updated:
                updated = True
            if updated:
                cache_series = _trim_weather_series_for_cache(export_series)
                store_weather_series(cache_series, cache_key=cache_key)
            download_url = _build_download_url(
                file_id, base_url=cfg.public_base_url
            )
            data_payload = response_series.model_dump(mode="json")
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
        cache_series = _trim_weather_series_for_cache(series)
        store_weather_series(cache_series, cache_key=cache_key)
        download_url = _build_download_url(file_id, base_url=cfg.public_base_url)
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


def lookup_goso_weather(
    query: Optional[WeatherQueryInput],
    *,
    api_url: Optional[str] = None,
) -> ToolInvocation:
    cfg = get_config()
    if not query or not query.region:
        return ToolInvocation(
            name="growth_weather_lookup",
            message="缺少地区信息，无法查询历史气象。",
            data={},
        )
    if query.lat is None or query.lon is None:
        geocode = _geocode_with_amap(
            query.region,
            api_key=cfg.amap_api_key,
            geocode_url=cfg.amap_geocode_url,
        )
        if geocode:
            query = query.model_copy(
                update={"lat": geocode["lat"], "lon": geocode["lon"]}
            )
            formatted = geocode.get("formatted_address")
            if formatted and formatted != query.region:
                query = query.model_copy(update={"region": formatted})
    if query.lat is None or query.lon is None:
        return ToolInvocation(
            name="growth_weather_lookup",
            message="需要经纬度才能查询历史气象数据。",
            data={},
        )

    year = query.year
    archive_store = get_weather_archive_store()
    archive_path = archive_store.get(
        region=query.region, lat=query.lat, lon=query.lon, year=year
    )
    if archive_path:
        path = Path(archive_path)
        if path.exists():
            series = _build_archive_series(query, path)
            if series:
                message = f"已获取{year}年历史气象数据（本地缓存）。"
                return ToolInvocation(
                    name="growth_weather_lookup",
                    message=message,
                    data=series.model_dump(mode="json"),
                )

    url = api_url or "https://data-api.91weather.com/Zoomlion/goso_day"
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    params = {
        "lat": query.lat,
        "lon": query.lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return ToolInvocation(
            name="growth_weather_lookup",
            message=f"历史气象接口请求失败: {exc}",
            data={},
        )
    series = _build_91weather_series(payload, query)
    if not series:
        return ToolInvocation(
            name="growth_weather_lookup",
            message="历史气象接口返回格式未识别。",
            data={"payload": payload},
        )
    archive_path = _persist_weather_archive(
        series,
        region=query.region,
        lat=query.lat,
        lon=query.lon,
        year=year,
    )
    if archive_path:
        archive_store.set(
            region=query.region,
            lat=query.lat,
            lon=query.lon,
            year=year,
            data_path=archive_path,
        )
    message = f"已获取{year}年历史气象数据。"
    return ToolInvocation(
        name="growth_weather_lookup",
        message=message,
        data=series.model_dump(mode="json"),
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
    if query:
        start, end = _resolve_query_range(query)
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
