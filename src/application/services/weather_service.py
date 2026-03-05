from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional, Tuple, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...infra.db_catalog import TABLE_KEY_WEATHER, resolve_db_table
from ...infra.llm import get_chat_model
from ...observability.llm_usage import (
    apply_span_attributes,
    build_llm_input_token_attrs,
    build_llm_output_token_attrs,
)
from ...observability.otel import record_exception, start_span
from ...schemas.models import (
    ToolInvocation,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER


def configure_weather_ports(
    *,
    config_port: Optional[ConfigPort] = None,
    sql_port: Optional[SqlPort] = None,
    http_port: Optional[HttpPort] = None,
) -> None:
    global _CONFIG_PORT, _SQL_PORT, _HTTP_PORT
    if config_port is not None:
        _CONFIG_PORT = config_port
    if sql_port is not None:
        _SQL_PORT = sql_port
    if http_port is not None:
        _HTTP_PORT = http_port


def _cfg():
    return _CONFIG_PORT.get()


def _fetch_all(url: str, sql: str, params: tuple[object, ...] = ()) -> list[dict]:
    return _SQL_PORT.fetch_all(url, sql, params)


def _qid(name: str) -> str:
    return _SQL_PORT.quote_identifier(name)


def _get_http(
    url: str,
    *,
    params: Optional[dict[str, object]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
):
    return _HTTP_PORT.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
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


def _extract_dates_from_text(text: str) -> List[date]:
    if not text:
        return []
    dates: List[date] = []
    for match in re.finditer(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", text):
        try:
            dates.append(
                date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
        except ValueError:
            continue
    for match in re.finditer(r"(20\d{2})(\d{2})(\d{2})", text):
        try:
            dates.append(
                date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
        except ValueError:
            continue
    return dates


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
    if not api_key:
        return None
    url = geocode_url or "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": api_key, "address": address}
    try:
        response = _get_http(url, params=params, timeout=10.0)
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
    if not region and lat is not None and lon is not None:
        region = f"{lat},{lon}"
    year = _parse_year(payload.get("year"))
    if year is None:
        year = _parse_year(start_date) or _parse_year(end_date)
    if start_date is not None:
        year = start_date.year
    elif end_date is not None:
        year = end_date.year
    data: Dict[str, object] = {}
    if region:
        data["region"] = region
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
        dates = _extract_dates_from_text(text)
        start_date = dates[0] if len(dates) >= 1 else None
        end_date = dates[1] if len(dates) >= 2 else None
        lat_lon = _extract_lat_lon_from_text(text)
        if start_date and end_date:
            lat = lon = None
            if lat_lon:
                lat, lon = lat_lon
            try:
                query = WeatherQueryInput(
                    start_date=start_date,
                    end_date=end_date,
                    lat=lat,
                    lon=lon,
                    year=start_date.year,
                )
            except Exception:
                return text, None
        else:
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
    cfg = _cfg()
    mode = (cfg.weather_summary_mode or "template").lower()
    if mode != "llm":
        return template_summary

    try:
        model = get_chat_model()
        system_prompt = (
            "你是气象助理，请基于统计信息输出简洁摘要。"
            "要求：2-3 句中文，包含温度范围、降水概况和主要天气。"
        )
        user_payload = json.dumps(stats, ensure_ascii=False)
        span_attrs = build_llm_input_token_attrs(
            model, system_prompt=system_prompt, user_prompt=user_payload
        )
        with start_span("llm.weather_summary", attributes=span_attrs) as span:
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_payload),
                    ]
                )
            except Exception as exc:
                record_exception(span, exc)
                raise
            apply_span_attributes(
                span, build_llm_output_token_attrs(response)
            )
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        return template_summary
    return template_summary


def _resolve_query_range(query: WeatherQueryInput) -> Tuple[date, date]:
    return query.start_date, query.end_date


def _build_date_followup(
    *,
    prompt: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> ToolInvocation:
    draft: Dict[str, object] = {}
    if start_date:
        draft["start_date"] = start_date.isoformat()
    if end_date:
        draft["end_date"] = end_date.isoformat()
    return ToolInvocation(
        name="weather_lookup",
        message="需要提供起始日期和结束日期（YYYY-MM-DD），且最多 30 天。",
        data={
            "missing_fields": ["start_date", "end_date"],
            "draft": draft,
            "followup_count": 0,
        },
    )


def _lookup_91weather(
    query: Optional[WeatherQueryInput],
    *,
    api_url: Optional[str],
) -> ToolInvocation:
    cfg = _cfg()
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
    url = api_url or "https://data-api.91weather.com/Zoomlion/higf_day_plus"
    params = {"lat": query.lat, "lon": query.lon}
    try:
        response = _get_http(url, params=params, timeout=10.0)
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
        data_payload = series.model_dump(mode="json")
        data_payload["summary"] = summary
        message = summary
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
    cfg = _cfg()
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
        response = _get_http(url, params=params, timeout=10.0)
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
    message = f"已获取{year}年历史气象数据。"
    return ToolInvocation(
        name="growth_weather_lookup",
        message=message,
        data=series.model_dump(mode="json"),
    )


def _require_db_url() -> str:
    cfg = _cfg()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取气象数据。")
    return cfg.agri_db_url


def _get_weather_table() -> str:
    return resolve_db_table(_cfg(), TABLE_KEY_WEATHER)


def _normalize_weather_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _query_farm_weather_rows(
    farm_id: str, start_date: date, end_date: date
) -> List[Dict[str, object]]:
    url = _require_db_url()
    table = _qid(_get_weather_table())
    col_date = _qid("date")
    col_farm = _qid("farm_id")
    sql = (
        f"SELECT {col_date} AS date, "
        "tmax, tmin, tavg, wins, pre, rh "
        f"FROM {table} "
        f"WHERE {col_farm} = %s AND {col_date} >= %s AND {col_date} <= %s "
        f"ORDER BY {col_date} ASC"
    )
    return _fetch_all(url, sql, (farm_id, start_date, end_date))


def _build_series_from_rows(
    farm_id: str,
    rows: List[Dict[str, object]],
    *,
    region: Optional[str] = None,
) -> Optional[WeatherSeries]:
    if not rows:
        return None
    points: List[WeatherDataPoint] = []
    for row in rows:
        day = _normalize_weather_date(row.get("date"))
        if day is None:
            continue
        points.append(
            WeatherDataPoint(
                timestamp=datetime.combine(day, time.min),
                temperature=_parse_float(row.get("tavg")),
                temperature_max=_parse_float(row.get("tmax")),
                temperature_min=_parse_float(row.get("tmin")),
                humidity=_parse_float(row.get("rh")),
                precipitation=_parse_float(row.get("pre")),
                wind_speed=_parse_float(row.get("wins")),
                condition=None,
            )
        )
    if not points:
        return None
    start_date = points[0].timestamp.date()
    end_date = points[-1].timestamp.date()
    return WeatherSeries(
        region=f"farm:{farm_id}",
        granularity="daily",
        start_date=start_date,
        end_date=end_date,
        points=points,
        source="db",
    )


def lookup_farm_weather_by_user(
    *,
    user_id: Optional[str],
    start_date: date,
    end_date: date,
    region: Optional[str] = None,
) -> ToolInvocation:
    farm_id = "1"
    rows = _query_farm_weather_rows(farm_id, start_date, end_date)
    series = _build_series_from_rows(farm_id, rows, region=region)
    if not series:
        return ToolInvocation(
            name="growth_weather_lookup",
            message="未找到对应农场的气象数据。",
            data={},
        )
    return ToolInvocation(
        name="growth_weather_lookup",
        message="已获取农场气象数据。",
        data=series.model_dump(mode="json"),
    )


def lookup_weather(
    prompt: str,
    *,
    cache_prompt: Optional[str] = None,
    query: Optional[WeatherQueryInput] = None,
) -> ToolInvocation:
    text = prompt or ""
    if cache_prompt is None or query is None:
        cache_prompt, query = normalize_weather_prompt(text)
    if query is None:
        dates = _extract_dates_from_text(text)
        start_date = dates[0] if len(dates) >= 1 else None
        end_date = dates[1] if len(dates) >= 2 else None
        return _build_date_followup(
            prompt=text, start_date=start_date, end_date=end_date
        )
    start, end = _resolve_query_range(query)
    return lookup_farm_weather_by_user(
        user_id=None,
        start_date=start,
        end_date=end,
        region=None,
    )
