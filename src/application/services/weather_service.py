from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional, List

from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...agent.followup import build_tool_followup_invocation
from ...infra.db_catalog import resolve_region_lookup_sources
from ...schemas.models import (
    ToolInvocation,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER
_WEATHER_REGION_RE = re.compile(
    r"(?P<region>[\u4e00-\u9fa5]{2,20})(?:的)?"
    r"(?:天气|气温|气象|降雨|降水|湿度|风速|预报)"
)
_REGION_SUFFIX_RE = re.compile(r"(特别行政区|自治区|自治州|省|市|州|盟|地区|区|县)$")


def configure_weather_ports(
    *,
    config_port: Optional[ConfigPort] = None,
    http_port: Optional[HttpPort] = None,
    sql_port: Optional[SqlPort] = None,
) -> None:
    global _CONFIG_PORT, _HTTP_PORT, _SQL_PORT
    if config_port is not None:
        _CONFIG_PORT = config_port
    if http_port is not None:
        _HTTP_PORT = http_port
    if sql_port is not None:
        _SQL_PORT = sql_port


def _cfg():
    return _CONFIG_PORT.get()


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


def _fetch_all(url: str, sql: str, params: tuple[object, ...] = ()) -> list[dict]:
    return _SQL_PORT.fetch_all(url, sql, params)


def _qid(name: str) -> str:
    return _SQL_PORT.quote_identifier(name)


def _build_api_headers(*, api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-KEY"] = token
    return headers


def _require_db_url() -> str:
    cfg = _cfg()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法查询区域表。")
    return cfg.agri_db_url


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


def _extract_region_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = _WEATHER_REGION_RE.search(text)
    if not match:
        return None
    region = str(match.group("region") or "").strip()
    return region or None


def _build_weather_query_from_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    region = payload.get("region")
    start_date = _parse_payload_date(
        payload.get("start_date") or payload.get("start")
    )
    end_date = _parse_payload_date(payload.get("end_date") or payload.get("end"))
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


def _merge_followup_weather_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    followup = payload.get("followup")
    if not isinstance(followup, dict):
        return None
    draft = followup.get("draft")
    merged: Dict[str, object] = {}
    if isinstance(draft, dict):
        merged.update(draft)
    prompt = followup.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        dates = _extract_dates_from_text(prompt)
        if dates:
            merged["start_date"] = dates[0].isoformat()
            if len(dates) >= 2:
                merged["end_date"] = dates[1].isoformat()
        region = _extract_region_from_text(prompt)
        if region:
            merged["region"] = region
    for key in ("region", "start_date", "end_date", "year", "granularity"):
        value = payload.get(key)
        if value is not None and value != "":
            merged[key] = value
    if not merged:
        return None
    return _build_weather_query_from_payload(merged)


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
        region = _extract_region_from_text(text)
        if start_date and end_date:
            try:
                query = WeatherQueryInput(
                    region=region,
                    start_date=start_date,
                    end_date=end_date,
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
        query = _merge_followup_weather_payload(payload)
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


def _resolve_query_range(query: WeatherQueryInput) -> Tuple[date, date]:
    return query.start_date, query.end_date


def _build_date_followup(
    *,
    prompt: str,
    region: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> ToolInvocation:
    draft: Dict[str, object] = {}
    if region:
        draft["region"] = region
    if start_date:
        draft["start_date"] = start_date.isoformat()
    if end_date:
        draft["end_date"] = end_date.isoformat()
    return build_tool_followup_invocation(
        name="weather_lookup",
        message="需要提供起始日期和结束日期（YYYY-MM-DD），且最多 30 天。",
        missing_fields=["start_date", "end_date"],
        draft=draft,
        query=prompt,
    )


def _normalize_region_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[，。；、,.!！?？\s]+", "", text)


def _region_text_variants(value: object) -> List[str]:
    normalized = _normalize_region_token(value)
    if not normalized:
        return []
    variants = [normalized]
    trimmed = _REGION_SUFFIX_RE.sub("", normalized)
    if trimmed and trimmed not in variants:
        variants.append(trimmed)
    return variants


def _coerce_region_id_value(value: object) -> Optional[object]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def _query_region_id_by_table(
    table: str, id_column: str, name_column: str, region_text: str
) -> Optional[object]:
    url = _require_db_url()
    variants = _region_text_variants(region_text)
    if not variants:
        return None
    table_name = _qid(table)
    id_col = _qid(id_column)
    name_col = _qid(name_column)
    for variant in variants:
        try:
            sql = (
                f"SELECT {id_col} AS region_id, {name_col} AS region_name "
                f"FROM {table_name} "
                f"WHERE CAST({name_col} AS TEXT) ILIKE %s "
                "LIMIT 20"
            )
            rows = _fetch_all(url, sql, (f"%{variant}%",))
        except Exception:
            continue
        best_score = -1
        best_id: Optional[object] = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            region_name = row.get("region_name")
            region_id = _coerce_region_id_value(row.get("region_id"))
            if region_id is None:
                continue
            name_norm = _normalize_region_token(region_name)
            if not name_norm:
                continue
            score = 0
            if name_norm == variant:
                score = 100 + len(name_norm)
            elif variant in name_norm:
                score = 70 + len(variant)
            elif name_norm in variant:
                score = 60 + len(name_norm)
            if score > best_score:
                best_score = score
                best_id = region_id
        if best_id is not None:
            return best_id
    return None


def _resolve_region_id(region_text: object) -> Optional[object]:
    variants = _region_text_variants(region_text)
    if not variants:
        return None
    for source in resolve_region_lookup_sources(_cfg()):
        region_id = _query_region_id_by_table(
            source.table,
            source.id_column,
            source.name_column,
            variants[0],
        )
        if region_id is not None:
            return region_id
    return None


def _build_region_followup(
    *,
    prompt: str,
    region: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> ToolInvocation:
    draft: Dict[str, object] = {}
    if region:
        draft["region"] = region
    if start_date:
        draft["start_date"] = start_date.isoformat()
    if end_date:
        draft["end_date"] = end_date.isoformat()
    return build_tool_followup_invocation(
        name="weather_lookup",
        message="未匹配到对应区域，请补充更准确的区域名称。",
        missing_fields=["region"],
        draft=draft,
        query=prompt,
    )


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
        day = _parse_forecast_date(row.get("date"))
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


def _get_farm_weather_api_url() -> Optional[str]:
    cfg = _cfg()
    raw = getattr(cfg, "farm_weather_api_url", None)
    if raw:
        return str(raw).strip()
    base = str(getattr(cfg, "business_api_base_url", None) or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/weather/farm"


def _lookup_farm_weather_by_api(
    *,
    farm_id: str,
    start_date: date,
    end_date: date,
    region: Optional[str] = None,
) -> Optional[WeatherSeries]:
    cfg = _cfg()
    url = _get_farm_weather_api_url()
    if not url:
        raise RuntimeError("缺少农场天气接口地址。")
    params: dict[str, object] = {
        "farm_id": farm_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    if region:
        text = str(region).strip()
        if text:
            params["region_id"] = text
    response = _get_http(
        url,
        params=params,
        headers=_build_api_headers(
            api_key=getattr(cfg, "business_api_key", None)
        ),
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("农场天气接口返回格式未识别。")
    code = str(payload.get("code", "")).strip()
    if code and code != "0":
        raise RuntimeError(str(payload.get("msg") or "农场天气接口返回失败。"))
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    api_farm_id = data.get("farm_id")
    points = data.get("points")
    if not isinstance(points, list):
        return None
    rows = []
    for item in points:
        if not isinstance(item, dict):
            continue
        rows.append(dict(item))
    farm_label = str(api_farm_id if api_farm_id is not None else farm_id)
    series = _build_series_from_rows(farm_label, rows, region=region)
    if series is None:
        return None
    return series.model_copy(update={"source": "business_api"})


def lookup_farm_weather_by_user(
    *,
    user_id: Optional[str],
    start_date: date,
    end_date: date,
    region_id: Optional[object] = None,
) -> ToolInvocation:
    del user_id
    cfg = _cfg()
    farm_id = str(getattr(cfg, "default_farm_id", None) or "1").strip() or "1"
    try:
        series = _lookup_farm_weather_by_api(
            farm_id=farm_id,
            start_date=start_date,
            end_date=end_date,
            region=str(region_id).strip() if region_id is not None else None,
        )
    except Exception as exc:
        return ToolInvocation(
            name="weather_lookup",
            message=f"查询农场气象数据失败: {exc}",
            data={},
        )
    if not series:
        return ToolInvocation(
            name="weather_lookup",
            message="未找到对应农场的气象数据。",
            data={},
        )
    return ToolInvocation(
        name="weather_lookup",
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
        region = _extract_region_from_text(text)
        return _build_date_followup(
            prompt=text,
            region=region,
            start_date=start_date,
            end_date=end_date,
        )
    start, end = _resolve_query_range(query)
    region_id = None
    if query.region:
        region_id = _resolve_region_id(query.region)
        if region_id is None:
            return _build_region_followup(
                prompt=text,
                region=query.region,
                start_date=start,
                end_date=end,
            )
    return lookup_farm_weather_by_user(
        user_id=None,
        start_date=start,
        end_date=end,
        region_id=region_id,
    )
