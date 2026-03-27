from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from ..adapters import DEFAULT_CONFIG_ADAPTER, DEFAULT_HTTP_ADAPTER, DEFAULT_SQL_ADAPTER
from ..ports import ConfigPort, HttpPort, SqlPort
from ...infra.db_catalog import (
    TABLE_KEY_VARIETY,
    resolve_db_table,
)
from ...observability.logging_utils import log_event, summarize_text
from ...schemas import (
    GrowthStageResult,
    PlantingDetails,
    PlantingDetailsDraft,
    PredictGrowthStageInput,
)


GROWTH_STAGE_COLUMNS = [
    "三叶一心",
    "返青",
    "分蘖期",
    "有效分蘖终止期",
    "拔节期",
    "幼穗分化1期",
    "幼穗分化2期",
    "幼穗分化4期",
    "孕穗期",
    "破口期",
    "始穗期",
    "抽穗期",
    "齐穗期",
    "成熟期",
]
_PLAN_NAME_COLUMNS = (
    "name",
    "plan_name",
    "plan_title",
    "planTitle",
    "计划名称",
    "种植计划名称",
)
_PLAN_VARIETY_ID_COLUMNS = ("variety_id", "varietyId", "variety_id", "品种ID")
_PLAN_VARIETY_COLUMNS = ("variety", "variety_name", "品种", "品种名称")
_PLAN_CROP_COLUMNS = ("crop", "crop_name", "作物", "作物名称")
_PLAN_METHOD_COLUMNS = (
    "planting_method",
    "plant_method",
    "method",
    "sowing_method",
    "种植方式",
    "播种方式",
)
_PLAN_CULTI_TYPE_COLUMNS = ("culti_type", "cultiType", "稻作类型")
_PLAN_SOWING_DATE_COLUMNS = (
    "sowing_date",
    "sow_date",
    "seed_date",
    "播种日期",
    "播种时间",
)
_PLAN_TRANSPLANT_DATE_COLUMNS = (
    "transplant_date",
    "transplanting_date",
    "transp_date",
    "移栽日期",
    "移栽时间",
    "插秧日期",
    "插秧时间",
)

_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER


def configure_growth_stage_ports(
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


def _post_json(
    url: str,
    *,
    payload: dict,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
):
    return _HTTP_PORT.post(
        url,
        json_payload=payload,
        headers=headers,
        timeout=timeout,
    )


def _get_variety_table() -> str:
    return resolve_db_table(_cfg(), TABLE_KEY_VARIETY)


def _build_api_headers(*, api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-KEY"] = token
    return headers


def _join_api_url(base_url: Optional[str], suffix: str) -> Optional[str]:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/{suffix.lstrip('/')}"


def _get_growth_stage_api_key() -> Optional[str]:
    cfg = _cfg()
    return getattr(cfg, "growth_stage_api_key", None) or getattr(
        cfg, "business_api_key", None
    )


def _get_planting_plan_search_api_url() -> Optional[str]:
    cfg = _cfg()
    raw = getattr(cfg, "planting_plan_search_api_url", None)
    if raw:
        return str(raw).strip()
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        "/planting-plan/search",
    )


def _get_planting_plan_active_api_url() -> Optional[str]:
    cfg = _cfg()
    raw = getattr(cfg, "planting_plan_active_api_url", None)
    if raw:
        return str(raw).strip()
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        "/planting-plan/active",
    )


def _get_planting_plan_detail_api_url(plan_id: object) -> Optional[str]:
    cfg = _cfg()
    raw = str(getattr(cfg, "planting_plan_detail_api_url", "") or "").strip()
    if raw:
        if "{plan_id}" in raw:
            return raw.format(plan_id=plan_id)
        return f"{raw.rstrip('/')}/{plan_id}"
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        f"/planting-plan/{plan_id}",
    )


def _get_growth_stage_by_plan_api_url(plan_id: object) -> Optional[str]:
    cfg = _cfg()
    raw = str(getattr(cfg, "growth_stage_api_url", "") or "").strip()
    if raw:
        if "{plan_id}" in raw:
            return raw.format(plan_id=plan_id)
        return f"{raw.rstrip('/')}/{plan_id}"
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        f"/growth-stage/by-plan/{plan_id}",
    )


def _unwrap_api_data(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("业务接口返回格式未识别。")
    code = str(payload.get("code", "")).strip()
    if code == "204":
        return {}
    if code and code not in {"0", "200"}:
        raise RuntimeError(str(payload.get("msg") or "业务接口返回失败。"))
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return {}


def _log_growth_stage_api_request(
    operation: str,
    *,
    url: str,
    params: Optional[dict[str, object]] = None,
    payload: Optional[dict[str, object]] = None,
) -> None:
    log_event(
        "growth_stage_api_request",
        operation=operation,
        url=url,
        params=params,
        payload=payload,
    )


def _log_growth_stage_api_http_error(
    operation: str,
    *,
    url: str,
    exc: Exception,
    params: Optional[dict[str, object]] = None,
    payload: Optional[dict[str, object]] = None,
) -> None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        log_event(
            "growth_stage_api_http_error",
            operation=operation,
            url=url,
            params=params,
            payload=payload,
            status_code=getattr(resp, "status_code", None),
            response_text=summarize_text(getattr(resp, "text", str(exc)), limit=1200),
        )
    else:
        log_event(
            "growth_stage_api_request_error",
            operation=operation,
            url=url,
            params=params,
            payload=payload,
            error=str(exc),
        )


def _parse_growth_stage_api_data(
    operation: str,
    *,
    url: str,
    response,
    params: Optional[dict[str, object]] = None,
    payload: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    try:
        raw = response.json()
    except Exception as exc:
        log_event(
            "growth_stage_api_parse_error",
            operation=operation,
            url=url,
            params=params,
            payload=payload,
            status_code=response.status_code,
            response_text=summarize_text(response.text or "", limit=1200),
        )
        raise RuntimeError("业务接口返回格式未识别。") from exc
    log_event(
        "growth_stage_api_response",
        operation=operation,
        url=url,
        params=params,
        payload=payload,
        status_code=response.status_code,
        response_summary=summarize_text(
            json.dumps(raw, ensure_ascii=False, default=str), limit=1200
        ),
    )
    try:
        return _unwrap_api_data(raw)
    except RuntimeError as exc:
        if isinstance(raw, dict):
            log_event(
                "growth_stage_api_business_error",
                operation=operation,
                url=url,
                params=params,
                payload=payload,
                code=str(raw.get("code", "")).strip() or None,
                msg=str(raw.get("msg") or raw.get("message") or exc),
            )
        else:
            log_event(
                "growth_stage_api_parse_error",
                operation=operation,
                url=url,
                params=params,
                payload=payload,
                status_code=response.status_code,
            )
        raise


def _normalize_plan_api_row(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    row = dict(raw)
    if "id" not in row and row.get("plan_id") is not None:
        row["id"] = row.get("plan_id")
    return row


def _build_plan_api_payload(filters: Dict[str, object], *, limit: Optional[int]) -> dict:
    payload: dict[str, object] = {}
    for key in (
        "plan_name",
        "variety",
        "crop",
        "culti_type",
        "planting_method",
        "sowing_date",
        "transplant_date",
    ):
        value = filters.get(key)
        if value is None:
            continue
        if isinstance(value, date):
            payload[key] = value.isoformat()
            continue
        text = str(value).strip()
        if text:
            payload[key] = text
    if limit is not None:
        payload["limit"] = max(1, int(limit))
    return payload


_CODE_MAP_CACHE: Dict[str, Dict[int, str]] = {}


def _fetch_code_map(category: str) -> Dict[int, str]:
    url = _require_db_url()
    try:
        sql = (
            "SELECT code, code_name FROM agri_code_dict "
            "WHERE category = %s AND is_active = true"
        )
        rows = _fetch_all(url, sql, (category,))
    except Exception:
        return {}
    mapping: Dict[int, str] = {}
    for row in rows:
        code = row.get("code") if isinstance(row, dict) else None
        name = row.get("code_name") if isinstance(row, dict) else None
        if code is None or name is None:
            continue
        try:
            mapping[int(code)] = str(name).strip()
        except Exception:
            continue
    return mapping


def _get_code_map(category: str) -> Dict[int, str]:
    if category not in _CODE_MAP_CACHE:
        _CODE_MAP_CACHE[category] = _fetch_code_map(category)
    return _CODE_MAP_CACHE[category]


def _code_to_name(category: str, value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        code = int(value)
    except Exception:
        return None
    mapping = _get_code_map(category)
    name = mapping.get(code)
    if name:
        return name
    return str(value)


def _normalize_method_for_planting(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        label = _code_to_name("sowingmtd", text)
    else:
        label = text
    if not label:
        return None
    if any(key in label for key in ("直播", "撒播")):
        return "direct_seeding"
    if any(key in label for key in ("插秧", "移栽", "抛秧")):
        return "transplanting"
    return text


def _fetch_variety_name(variety_id: object) -> Optional[str]:
    if variety_id is None:
        return None
    url = _require_db_url()
    table = _get_variety_table()
    try:
        sql = (
            f"SELECT name FROM {_qid(table)} WHERE id = %s LIMIT 1"
        )
        rows = _fetch_all(url, sql, (variety_id,))
    except Exception:
        return None
    if not rows:
        return None
    value = rows[0].get("name") if isinstance(rows[0], dict) else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fetch_variety_id_by_name(variety_name: str) -> Optional[object]:
    if not variety_name:
        return None
    url = _require_db_url()
    table = _get_variety_table()
    try:
        sql = (
            f"SELECT id FROM {_qid(table)} WHERE name = %s LIMIT 1"
        )
        rows = _fetch_all(url, sql, (variety_name,))
    except Exception:
        return None
    if not rows:
        return None
    return rows[0].get("id") if isinstance(rows[0], dict) else None


def _parse_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        text = str(int(value))
    else:
        text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _build_plan_filters(planting: PlantingDetails) -> Dict[str, object]:
    method = planting.planting_method
    method_value = method.value if hasattr(method, "value") else str(method)
    return {
        "crop": planting.crop,
        "variety": planting.variety,
        "culti_type": getattr(planting, "culti_type", None),
        "planting_method": method_value,
        "sowing_date": planting.sowing_date,
        "transplant_date": planting.transplant_date,
    }


def _search_planting_plans_api(
    filters: Dict[str, object], *, limit: int = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    url = _get_planting_plan_search_api_url()
    if not url:
        raise RuntimeError("缺少种植计划搜索接口地址。")
    payload = _build_plan_api_payload(filters, limit=limit)
    _log_growth_stage_api_request("planting_plan_search", url=url, payload=payload)
    try:
        response = _post_json(
            url,
            payload=payload,
            headers=_build_api_headers(api_key=_get_growth_stage_api_key()),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        _log_growth_stage_api_http_error(
            "planting_plan_search", url=url, exc=exc, payload=payload
        )
        raise
    data = _parse_growth_stage_api_data(
        "planting_plan_search", url=url, response=response, payload=payload
    )
    plans = data.get("plans")
    if not isinstance(plans, list):
        plans = []
    rows = [_normalize_plan_api_row(item) for item in plans]
    columns = list(rows[0].keys()) if rows else ["plan_id", "id"]
    return rows, "id", columns


def _list_active_planting_plans_api(
    *, limit: Optional[int] = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    url = _get_planting_plan_active_api_url()
    if not url:
        raise RuntimeError("缺少启用计划接口地址。")
    params: dict[str, object] = {}
    if limit is not None:
        params["limit"] = max(1, int(limit))
    _log_growth_stage_api_request("planting_plan_active", url=url, params=params)
    try:
        response = _get_http(
            url,
            params=params,
            headers=_build_api_headers(api_key=_get_growth_stage_api_key()),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        _log_growth_stage_api_http_error(
            "planting_plan_active", url=url, exc=exc, params=params
        )
        raise
    data = _parse_growth_stage_api_data(
        "planting_plan_active", url=url, response=response, params=params
    )
    plans = data.get("plans")
    if not isinstance(plans, list):
        plans = []
    rows = [_normalize_plan_api_row(item) for item in plans]
    columns = list(rows[0].keys()) if rows else ["plan_id", "id"]
    return rows, "id", columns


def _fetch_planting_plan_row_by_id_api(
    plan_id: object,
) -> Tuple[Dict[str, object], str, List[str]]:
    url = _get_planting_plan_detail_api_url(plan_id)
    if not url:
        raise RuntimeError("缺少计划详情接口地址。")
    _log_growth_stage_api_request("planting_plan_detail", url=url)
    try:
        response = _get_http(
            url,
            headers=_build_api_headers(api_key=_get_growth_stage_api_key()),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        _log_growth_stage_api_http_error(
            "planting_plan_detail", url=url, exc=exc
        )
        raise
    data = _parse_growth_stage_api_data(
        "planting_plan_detail", url=url, response=response
    )
    plan = _normalize_plan_api_row(data.get("plan"))
    columns = list(plan.keys()) if plan else ["plan_id", "id"]
    return plan, "id", columns


def _query_growth_stage_from_plan_id_api(plan_id: object) -> GrowthStageResult:
    url = _get_growth_stage_by_plan_api_url(plan_id)
    if not url:
        raise RuntimeError("缺少生育期查询接口地址。")
    _log_growth_stage_api_request("growth_stage_by_plan", url=url)
    try:
        response = _get_http(
            url,
            headers=_build_api_headers(api_key=_get_growth_stage_api_key()),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        _log_growth_stage_api_http_error("growth_stage_by_plan", url=url, exc=exc)
        raise
    data = _parse_growth_stage_api_data(
        "growth_stage_by_plan", url=url, response=response
    )
    stages = data.get("stages")
    if isinstance(stages, dict):
        normalized: Dict[str, str] = {}
        for key, value in stages.items():
            if key == "stage_dates" and isinstance(value, dict):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            elif value is not None:
                normalized[str(key)] = str(value)
        return GrowthStageResult(stages=normalized)
    return GrowthStageResult(stages={})


def search_planting_plans(
    filters: Dict[str, object], *, limit: int = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    return _search_planting_plans_api(filters, limit=limit)


def list_active_planting_plans(
    *, limit: Optional[int] = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    return _list_active_planting_plans_api(limit=limit)


def _fetch_planting_plan_row_by_id(
    plan_id: object,
) -> Tuple[Dict[str, object], str, List[str]]:
    return _fetch_planting_plan_row_by_id_api(plan_id)


def _build_planting_from_plan_row(
    row: Dict[str, object],
    columns: List[str],
    fallback: Optional[PlantingDetails] = None,
) -> Optional[PlantingDetails]:
    if not row:
        return fallback
    data: Dict[str, object] = {}

    crop_col = next((c for c in _PLAN_CROP_COLUMNS if c in columns), None)
    if crop_col:
        value = row.get(crop_col)
        if value is not None and str(value).strip():
            data["crop"] = str(value).strip()

    variety_id_col = next((c for c in _PLAN_VARIETY_ID_COLUMNS if c in columns), None)
    if variety_id_col:
        value = row.get(variety_id_col)
        variety_name = _fetch_variety_name(value)
        if variety_name:
            data["variety"] = variety_name
    variety_col = next((c for c in _PLAN_VARIETY_COLUMNS if c in columns), None)
    if variety_col and "variety" not in data:
        value = row.get(variety_col)
        if value is not None and str(value).strip():
            data["variety"] = str(value).strip()

    culti_col = next((c for c in _PLAN_CULTI_TYPE_COLUMNS if c in columns), None)
    if culti_col:
        value = row.get(culti_col)
        culti_name = _code_to_name("culti_type", value)
        if culti_name:
            data["culti_type"] = culti_name

    method_col = next((c for c in _PLAN_METHOD_COLUMNS if c in columns), None)
    if method_col:
        value = row.get(method_col)
        if value is not None and str(value).strip():
            data["planting_method"] = _normalize_method_for_planting(value)

    sow_col = next((c for c in _PLAN_SOWING_DATE_COLUMNS if c in columns), None)
    if sow_col:
        sow = _parse_date(row.get(sow_col))
        if sow:
            data["sowing_date"] = sow

    trans_col = next((c for c in _PLAN_TRANSPLANT_DATE_COLUMNS if c in columns), None)
    if trans_col:
        trans = _parse_date(row.get(trans_col))
        if trans:
            data["transplant_date"] = trans

    if fallback:
        for field in (
            "crop",
            "variety",
            "culti_type",
            "planting_method",
            "sowing_date",
            "transplant_date",
        ):
            if field not in data or data[field] in ("", None):
                value = getattr(fallback, field, None)
                if value is not None:
                    data[field] = value
    if "crop" not in data or not data.get("crop"):
        data["crop"] = "水稻"

    if not data:
        return fallback
    try:
        draft = PlantingDetailsDraft(**data)
        return draft.to_canonical()
    except Exception:
        return fallback


def build_planting_from_plan_row(
    row: Dict[str, object],
    columns: List[str],
    *,
    fallback: Optional[PlantingDetails] = None,
) -> Optional[PlantingDetails]:
    return _build_planting_from_plan_row(row, columns, fallback=fallback)


def extract_plan_name_from_row(
    row: Dict[str, object], columns: List[str]
) -> Optional[str]:
    col = next((c for c in _PLAN_NAME_COLUMNS if c in columns), None)
    if not col:
        return None
    value = row.get(col)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_planting_plan(
    filters: Dict[str, object]
) -> Tuple[object, Dict[str, object], List[str]]:
    rows, id_col, columns = search_planting_plans(filters, limit=1)
    row = rows[0] if rows else {}
    if not row:
        raise ValueError("未找到对应的种植计划。")
    plan_id = row.get(id_col)
    if plan_id is None:
        raise ValueError("种植计划记录缺少计划 id。")
    return plan_id, row, columns


def resolve_planting_from_plan(
    filters: Dict[str, object],
    *,
    fallback: Optional[PlantingDetails] = None,
) -> Optional[PlantingDetails]:
    plan_id, row, columns = resolve_planting_plan(filters)
    return _build_planting_from_plan_row(row, columns, fallback=fallback)


def resolve_planting_from_plan_id(
    plan_id: object,
    *,
    fallback: Optional[PlantingDetails] = None,
) -> Optional[PlantingDetails]:
    row, _, columns = _fetch_planting_plan_row_by_id(plan_id)
    if not row:
        return fallback
    return _build_planting_from_plan_row(row, columns, fallback=fallback)


def query_growth_stage_from_plan_id(plan_id: object) -> GrowthStageResult:
    return _query_growth_stage_from_plan_id_api(plan_id)


def query_growth_stage_from_planting(
    input: PredictGrowthStageInput,
) -> GrowthStageResult:
    filters = _build_plan_filters(input.planting)
    plan_id, _, _ = resolve_planting_plan(filters)
    return query_growth_stage_from_plan_id(plan_id)
