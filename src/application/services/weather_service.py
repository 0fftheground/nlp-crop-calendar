from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from typing import Dict, Optional, List, Tuple

from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...agent.followup import build_tool_followup_invocation
from ...domain.date_parser import extract_date_range, extract_explicit_dates
from ...domain.region_text import build_region_text_variants, normalize_region_token
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
_SUPPORTED_WEATHER_OPERATIONS = (
    "施肥",
    "炼苗",
    "移栽",
    "翻地",
    "打药",
    "收割",
    "整地",
)
_SUPPORTED_WEATHER_OPERATION_ALIASES = {
    "施肥": ("施肥", "追肥"),
    "炼苗": ("炼苗",),
    "移栽": ("移栽", "插秧"),
    "翻地": ("翻地",),
    "打药": ("打药", "喷药"),
    "收割": ("收割", "收获"),
    "整地": ("整地",),
}
_WEATHER_OPERATION_FIELD_PREFIXES = {
    "施肥": "sf",
    "炼苗": "lm",
    "移栽": "yz",
    "翻地": "fd",
    "打药": "dy",
    "收割": "sg",
    "整地": "zd",
}
_UNSUPPORTED_WEATHER_OPERATION_ALIASES = {
    "浇水": ("浇水", "灌溉"),
    "除草": ("除草",),
    "育秧": ("育秧",),
    "病虫害防治": ("病虫害防治", "防病", "治虫"),
}
_WEATHER_SUITABILITY_CUES = (
    "农事适宜度",
    "适合",
    "适宜",
    "是否适合",
    "能否",
    "能不能",
    "可否",
    "合适吗",
    "宜不宜",
)


def extract_weather_operations(
    text: str, *, require_suitability_cues: bool = True
) -> Tuple[List[str], List[str]]:
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return [], []
    if require_suitability_cues and not any(
        token in normalized for token in _WEATHER_SUITABILITY_CUES
    ):
        return [], []
    supported: List[str] = []
    unsupported: List[str] = []
    for label, aliases in _SUPPORTED_WEATHER_OPERATION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            supported.append(label)
    for label, aliases in _UNSUPPORTED_WEATHER_OPERATION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            unsupported.append(label)
    return _dedupe_preserve_order(supported), _dedupe_preserve_order(unsupported)


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

def _post_json(
    url: str,
    *,
    payload: dict[str, object],
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
):
    return _HTTP_PORT.post(
        url,
        json_payload=payload,
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
    return extract_explicit_dates(text)


def _extract_region_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = _WEATHER_REGION_RE.search(text)
    if not match:
        return None
    region = str(match.group("region") or "").strip()
    return region or None


def _extract_weather_source_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    source_text = str(payload.get("query") or payload.get("prompt") or "").strip()
    return source_text or text


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _detect_weather_operation_support(text: str) -> Tuple[List[str], List[str]]:
    return extract_weather_operations(text, require_suitability_cues=True)


def _build_unsupported_weather_operation_message(operations: List[str]) -> str:
    supported = "、".join(_SUPPORTED_WEATHER_OPERATIONS)
    unsupported = "、".join(_dedupe_preserve_order(operations))
    return f"{unsupported}当前暂不支持气象适宜度展示；目前支持：{supported}。"


def _prepend_message(note: str, message: str) -> str:
    note_text = str(note or "").strip()
    message_text = str(message or "").strip()
    if not note_text:
        return message_text
    if not message_text:
        return note_text
    return f"{note_text}\n{message_text}"


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
    requested_operations = payload.get("requested_operations")
    if isinstance(requested_operations, list):
        data["requested_operations"] = [
            str(item).strip()
            for item in requested_operations
            if str(item).strip()
        ]
    try:
        return WeatherQueryInput(**data)
    except Exception:
        return None


def parse_weather_prompt_operations(
    prompt: str,
) -> Tuple[List[str], List[str], str]:
    source_text = _extract_weather_source_prompt(prompt or "")
    supported_ops, unsupported_ops = _detect_weather_operation_support(source_text)
    if not supported_ops and not unsupported_ops:
        supported_ops, unsupported_ops = extract_weather_operations(
            source_text, require_suitability_cues=False
        )
    unsupported_note = ""
    if unsupported_ops:
        unsupported_note = _build_unsupported_weather_operation_message(unsupported_ops)
    return supported_ops, unsupported_ops, unsupported_note


def apply_weather_operation_view(
    result: ToolInvocation,
    *,
    requested_operations: Optional[List[str]] = None,
    unsupported_note: str = "",
) -> ToolInvocation:
    data = dict(result.data or {})
    ops = _dedupe_preserve_order(list(requested_operations or []))
    if ops:
        points = data.get("points")
        allowed_prefixes = {
            _WEATHER_OPERATION_FIELD_PREFIXES[label]
            for label in ops
            if label in _WEATHER_OPERATION_FIELD_PREFIXES
        }
        if isinstance(points, list):
            trimmed_points: List[dict[str, object]] = []
            for item in points:
                if not isinstance(item, dict):
                    trimmed_points.append(item)
                    continue
                trimmed: dict[str, object] = {}
                for key, value in item.items():
                    if not re.match(r"^[a-z]{2}_(ws|reason)$", str(key)):
                        trimmed[key] = value
                        continue
                    prefix = str(key).split("_", 1)[0]
                    if prefix in allowed_prefixes:
                        trimmed[key] = value
                trimmed_points.append(trimmed)
            data["points"] = trimmed_points
    if ops:
        data["requested_operations"] = ops
    elif "requested_operations" in data:
        data.pop("requested_operations", None)
    message = result.message
    if unsupported_note:
        message = _prepend_message(unsupported_note, message)
    return result.model_copy(update={"message": message, "data": data})


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
        parsed_range = extract_date_range(text, today=date.today())
        if parsed_range and not (start_date and end_date):
            start_date, end_date = parsed_range
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
    source_prompt = str(payload.get("query") or payload.get("prompt") or "").strip()
    parsed_range = extract_date_range(source_prompt, today=date.today())
    if parsed_range:
        region = payload.get("region")
        requested_operations = payload.get("requested_operations")
        try:
            query = WeatherQueryInput(
                region=str(region).strip() if region not in (None, "") else None,
                start_date=parsed_range[0],
                end_date=parsed_range[1],
                year=parsed_range[0].year,
                granularity=payload.get("granularity")
                if payload.get("granularity") in {"hourly", "daily"}
                else "daily",
                include_advice=bool(payload.get("include_advice", False)),
                requested_operations=requested_operations
                if isinstance(requested_operations, list)
                else [],
            )
            canonical = json.dumps(
                query.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            return canonical, query
        except Exception:
            pass
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
    return normalize_region_token(value)


def _region_text_variants(value: object) -> List[str]:
    return build_region_text_variants(value)


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
    summary: Optional[str] = None,
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
                sf_ws=_parse_float(row.get("sf_ws")),
                sf_reason=str(row.get("sf_reason")).strip()
                if row.get("sf_reason") is not None
                else None,
                lm_ws=_parse_float(row.get("lm_ws")),
                lm_reason=str(row.get("lm_reason")).strip()
                if row.get("lm_reason") is not None
                else None,
                yz_ws=_parse_float(row.get("yz_ws")),
                yz_reason=str(row.get("yz_reason")).strip()
                if row.get("yz_reason") is not None
                else None,
                fd_ws=_parse_float(row.get("fd_ws")),
                fd_reason=str(row.get("fd_reason")).strip()
                if row.get("fd_reason") is not None
                else None,
                dy_ws=_parse_float(row.get("dy_ws")),
                dy_reason=str(row.get("dy_reason")).strip()
                if row.get("dy_reason") is not None
                else None,
                sg_ws=_parse_float(row.get("sg_ws")),
                sg_reason=str(row.get("sg_reason")).strip()
                if row.get("sg_reason") is not None
                else None,
                zd_ws=_parse_float(row.get("zd_ws")),
                zd_reason=str(row.get("zd_reason")).strip()
                if row.get("zd_reason") is not None
                else None,
            )
        )
    if not points:
        return None
    start_date = points[0].timestamp.date()
    end_date = points[-1].timestamp.date()
    region_label = str(region).strip() if region else f"farm:{farm_id}"
    return WeatherSeries(
        region=region_label,
        granularity="daily",
        start_date=start_date,
        end_date=end_date,
        points=points,
        source="db",
        summary=summary,
    )


def _get_farm_weather_api_url() -> Optional[str]:
    cfg = _cfg()
    raw = getattr(cfg, "farm_weather_api_url", None)
    if raw:
        return str(raw).strip()
    base = str(getattr(cfg, "business_api_base_url", None) or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/suit_rili"


def _lookup_farm_weather_by_api(
    *,
    farm_id: str,
    start_date: date,
    end_date: date,
    region_id: Optional[object] = None,
    region_label: Optional[str] = None,
) -> Optional[WeatherSeries]:
    cfg = _cfg()
    url = _get_farm_weather_api_url()
    if not url:
        raise RuntimeError("缺少农场天气接口地址。")
    payload: dict[str, object] = {
        "start_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
    }
    if region_id is not None:
        text = str(region_id).strip()
        if text:
            payload["region_id"] = text
    else:
        payload["farm_id"] = farm_id
    response = _post_json(
        url,
        payload=payload,
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
    message = str(payload.get("message") or payload.get("msg") or "").strip()
    if code and code != "200":
        raise RuntimeError(message or "农场天气接口返回失败。")
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append(dict(item))
    series = _build_series_from_rows(
        str(farm_id),
        rows,
        region=region_label,
        summary=message or None,
    )
    if series is None:
        return None
    return series.model_copy(update={"source": "agri_weather_api"})


def lookup_farm_weather_by_user(
    *,
    user_id: Optional[str],
    start_date: date,
    end_date: date,
    region_id: Optional[object] = None,
    region: Optional[str] = None,
) -> ToolInvocation:
    del user_id
    cfg = _cfg()
    farm_id = str(getattr(cfg, "default_farm_id", None) or "1").strip() or "1"
    try:
        series = _lookup_farm_weather_by_api(
            farm_id=farm_id,
            start_date=start_date,
            end_date=end_date,
            region_id=region_id,
            region_label=region,
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
        message=series.summary or "已获取农场气象与适宜度数据。",
        data=series.model_dump(mode="json"),
    )


def lookup_weather(
    prompt: str,
    *,
    cache_prompt: Optional[str] = None,
    query: Optional[WeatherQueryInput] = None,
) -> ToolInvocation:
    text = prompt or ""
    supported_ops, _, unsupported_note = parse_weather_prompt_operations(text)
    if not supported_ops and query is not None:
        supported_ops = _dedupe_preserve_order(list(query.requested_operations or []))
    if unsupported_note and not supported_ops:
        return ToolInvocation(
            name="weather_lookup",
            message=unsupported_note,
            data={},
        )
    if cache_prompt is None or query is None:
        cache_prompt, query = normalize_weather_prompt(text)
    if not supported_ops and query is not None:
        supported_ops = _dedupe_preserve_order(list(query.requested_operations or []))
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
    result = lookup_farm_weather_by_user(
        user_id=None,
        start_date=start,
        end_date=end,
        region_id=region_id,
        region=query.region,
    )
    return apply_weather_operation_view(
        result,
        requested_operations=supported_ops,
        unsupported_note=unsupported_note,
    )
