from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from ...infra.config import get_config
from ...infra.postgres import fetch_all, quote_identifier
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
_PLAN_ID_COLUMNS = ("id", "plan_id", "planting_plan_id", "planting_id")
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

_FORECAST_PLAN_ID_COLUMNS = ("planting_plan_id", "plan_id", "planting_id")
_FORECAST_STAGE_NAME_COLUMNS = (
    "stage",
    "stage_name",
    "growth_stage",
    "阶段",
    "生育期",
)
_FORECAST_STAGE_DATE_COLUMNS = (
    "stage_date",
    "date",
    "forecast_date",
    "预测日期",
    "stage_time",
)


def _require_db_url() -> str:
    cfg = get_config()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取生育期预测结果数据。")
    return cfg.agri_db_url


def _get_growth_stage_table() -> str:
    cfg = get_config()
    return cfg.growth_stage_db_table or "agri_growth_stage_forecast"


def _get_planting_plan_table() -> str:
    cfg = get_config()
    return cfg.planting_plan_db_table or "agri_plant_plan"


def _get_variety_table() -> str:
    cfg = get_config()
    return cfg.variety_db_table or "agri_rice_variety"


def _split_schema_table(table: str) -> Tuple[str, str]:
    if "." in table:
        schema, name = table.split(".", 1)
        return schema, name
    return "public", table


def _resolve_column(
    columns: List[str], candidates: Tuple[str, ...]
) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _get_table_columns(url: str, table: str) -> List[str]:
    schema, name = _split_schema_table(table)
    try:
        rows = fetch_all(
            url,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, name),
        )
    except Exception:
        return []
    columns = []
    for row in rows:
        col = row.get("column_name") if isinstance(row, dict) else None
        if col:
            columns.append(str(col))
    return columns


def _format_value(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_json(value: object) -> Optional[Dict[str, object]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


_CODE_MAP_CACHE: Dict[str, Dict[int, str]] = {}
_CODE_REVERSE_CACHE: Dict[str, Dict[str, int]] = {}


def _fetch_code_map(category: str) -> Dict[int, str]:
    url = _require_db_url()
    try:
        sql = (
            "SELECT code, code_name FROM agri_code_dict "
            "WHERE category = %s AND is_active = true"
        )
        rows = fetch_all(url, sql, (category,))
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


def _get_code_reverse_map(category: str) -> Dict[str, int]:
    if category not in _CODE_REVERSE_CACHE:
        mapping = _get_code_map(category)
        _CODE_REVERSE_CACHE[category] = {
            name: code for code, name in mapping.items() if name
        }
    return _CODE_REVERSE_CACHE[category]


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


def _name_to_code(category: str, value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    reverse = _get_code_reverse_map(category)
    return reverse.get(text)


def _normalize_plan_method_for_filter(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    mapping = {
        "direct_seeding": "直播",
        "直播": "直播",
        "撒播": "直播",
        "1": "直播",
        "transplanting": "插秧",
        "移栽": "插秧",
        "插秧": "插秧",
        "抛秧": "插秧",
        "3": "插秧",
    }
    label = mapping.get(text, text)
    code = _name_to_code("sowingmtd", label)
    return str(code) if code is not None else None


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
            f"SELECT name FROM {quote_identifier(table)} WHERE id = %s LIMIT 1"
        )
        rows = fetch_all(url, sql, (variety_id,))
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
            f"SELECT id FROM {quote_identifier(table)} WHERE name = %s LIMIT 1"
        )
        rows = fetch_all(url, sql, (variety_name,))
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


def _looks_like_stages(payload: Dict[str, object]) -> bool:
    keys = set(payload.keys())
    if keys.intersection(_meta_stage_keys()):
        return True
    for stage in GROWTH_STAGE_COLUMNS:
        if stage in keys:
            return True
    return False


def _meta_stage_keys() -> set:
    return {
        "predicted_stage",
        "estimated_next_stage",
        "gdd_accumulated",
        "gdd_required_maturity",
        "base_temperature",
        "stage_dates",
        "start_date",
        "match_rule",
    }


def _payload_is_stage_dates(payload: Dict[str, object]) -> bool:
    keys = set(payload.keys())
    if keys.intersection(_meta_stage_keys()):
        return False
    return any(stage in keys for stage in GROWTH_STAGE_COLUMNS)


def _extract_stage_dates(row: Dict[str, object]) -> Optional[str]:
    raw = (
        row.get("stage_dates")
        or row.get("stage_date")
        or row.get("stages")
        or row.get("生育期阶段")
        or row.get("阶段日期")
    )
    payload = _parse_json(raw)
    if payload and _looks_like_stages(payload):
        return None
    if payload and isinstance(payload, dict):
        stage_dates = {
            key: _format_value(value)
            for key, value in payload.items()
            if _format_value(value)
        }
        if stage_dates:
            return json.dumps(stage_dates, ensure_ascii=False)
    stage_dates: Dict[str, str] = {}
    for stage in GROWTH_STAGE_COLUMNS:
        if stage in row:
            value = _format_value(row.get(stage))
            if value:
                stage_dates[stage] = value
    if stage_dates:
        return json.dumps(stage_dates, ensure_ascii=False)
    return None


def _coerce_stages_from_row(row: Dict[str, object]) -> Dict[str, str]:
    for key in ("stages", "growth_stage", "payload", "result", "data"):
        payload = _parse_json(row.get(key))
        if not payload:
            continue
        if "stages" in payload and isinstance(payload["stages"], dict):
            payload = payload["stages"]
        if _payload_is_stage_dates(payload):
            stage_dates = {
                str(k): _format_value(v) or ""
                for k, v in payload.items()
                if _format_value(v) is not None
            }
            if stage_dates:
                return {
                    "stage_dates": json.dumps(stage_dates, ensure_ascii=False)
                }
        if _looks_like_stages(payload):
            stages = {
                str(k): _format_value(v) or ""
                for k, v in payload.items()
                if _format_value(v) is not None
            }
            stage_dates = {
                str(stage): stages.pop(stage)
                for stage in list(stages.keys())
                if stage in GROWTH_STAGE_COLUMNS
            }
            if stage_dates and "stage_dates" not in stages:
                stages["stage_dates"] = json.dumps(
                    stage_dates, ensure_ascii=False
                )
            return stages

    stages: Dict[str, str] = {}
    for field in (
        "predicted_stage",
        "estimated_next_stage",
        "gdd_accumulated",
        "gdd_required_maturity",
        "base_temperature",
        "start_date",
        "match_rule",
    ):
        if field in row:
            value = _format_value(row.get(field))
            if value:
                stages[field] = value
    stage_dates = _extract_stage_dates(row)
    if stage_dates:
        stages["stage_dates"] = stage_dates
    if not stages:
        raise ValueError("数据库记录缺少可解析的生育期字段。")
    return stages


def _coerce_stages_from_rows(
    rows: List[Dict[str, object]], columns: List[str]
) -> Dict[str, str]:
    if not rows:
        raise ValueError("未找到对应的生育期预测结果。")
    stage_name_col = _resolve_column(columns, _FORECAST_STAGE_NAME_COLUMNS)
    stage_date_col = _resolve_column(columns, _FORECAST_STAGE_DATE_COLUMNS)
    if stage_name_col and stage_date_col:
        stage_dates: Dict[str, str] = {}
        for row in rows:
            name = row.get(stage_name_col)
            value = row.get(stage_date_col)
            name_text = ""
            if name is not None:
                if isinstance(name, (int, float)) or str(name).isdigit():
                    mapped = _code_to_name("growth_stage", name)
                    name_text = mapped or str(name).strip()
                else:
                    name_text = str(name).strip()
            date_text = _format_value(value)
            if name_text and date_text:
                stage_dates[name_text] = date_text
        if stage_dates:
            return {
                "stage_dates": json.dumps(stage_dates, ensure_ascii=False)
            }
    return _coerce_stages_from_row(rows[0])


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


def _fetch_planting_plan_row(
    filters: Dict[str, object]
) -> Tuple[Dict[str, object], str, List[str]]:
    url = _require_db_url()
    table = _get_planting_plan_table()
    columns = _get_table_columns(url, table)
    if not columns:
        raise RuntimeError("无法读取种植计划表结构。")
    id_col = _resolve_column(columns, _PLAN_ID_COLUMNS)
    if not id_col:
        raise RuntimeError("种植计划表缺少 id/plan_id 字段。")

    conditions, params = _build_plan_conditions(filters, columns)

    order_column = None
    for candidate in ("updated_at", "created_at", "created_on", "sowing_date", id_col):
        if candidate in columns:
            order_column = candidate
            break

    sql = f"SELECT * FROM {quote_identifier(table)} WHERE " + " AND ".join(
        conditions
    )
    if order_column:
        sql += f" ORDER BY {quote_identifier(order_column)} DESC"
    sql += " LIMIT 1"
    try:
        rows = fetch_all(url, sql, tuple(params))
    except Exception as exc:
        raise RuntimeError(f"种植计划查询失败: {exc}") from exc
    return (rows[0] if rows else {}), id_col, columns


def _build_plan_conditions(
    filters: Dict[str, object], columns: List[str]
) -> Tuple[List[str], List[object]]:
    conditions: List[str] = []
    params: List[object] = []

    value = filters.get("plan_name") or filters.get("name")
    if value:
        col = _resolve_column(columns, _PLAN_NAME_COLUMNS)
        if col:
            conditions.append(f"{quote_identifier(col)} ILIKE %s")
            params.append(f"%{value}%")

    value = filters.get("variety")
    if value:
        variety_id_col = _resolve_column(columns, _PLAN_VARIETY_ID_COLUMNS)
        if variety_id_col:
            variety_id = (
                value
                if isinstance(value, (int, float))
                else _fetch_variety_id_by_name(str(value))
            )
            if variety_id is not None:
                conditions.append(f"{quote_identifier(variety_id_col)} = %s")
                params.append(variety_id)
        else:
            col = _resolve_column(columns, _PLAN_VARIETY_COLUMNS)
            if col:
                conditions.append(f"{quote_identifier(col)} = %s")
                params.append(value)

    value = filters.get("crop")
    if value:
        col = _resolve_column(columns, _PLAN_CROP_COLUMNS)
        if col:
            conditions.append(f"{quote_identifier(col)} = %s")
            params.append(value)

    value = filters.get("culti_type")
    if value:
        col = _resolve_column(columns, _PLAN_CULTI_TYPE_COLUMNS)
        if col:
            code = _name_to_code("culti_type", value)
            if code is not None:
                conditions.append(f"{quote_identifier(col)} = %s")
                params.append(code)
            elif isinstance(value, (int, float)):
                conditions.append(f"{quote_identifier(col)} = %s")
                params.append(int(value))

    value = filters.get("planting_method")
    if value:
        col = _resolve_column(columns, _PLAN_METHOD_COLUMNS)
        if col:
            normalized = _normalize_plan_method_for_filter(value)
            if normalized:
                conditions.append(f"{quote_identifier(col)} = %s")
                params.append(normalized)

    value = filters.get("sowing_date")
    if value:
        col = _resolve_column(columns, _PLAN_SOWING_DATE_COLUMNS)
        if col:
            conditions.append(f"{quote_identifier(col)} = %s")
            params.append(value)

    value = filters.get("transplant_date")
    if value:
        col = _resolve_column(columns, _PLAN_TRANSPLANT_DATE_COLUMNS)
        if col:
            conditions.append(f"{quote_identifier(col)} = %s")
            params.append(value)

    if not conditions:
        raise RuntimeError("未找到可用的种植计划查询字段。")

    return conditions, params


def search_planting_plans(
    filters: Dict[str, object], *, limit: int = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    url = _require_db_url()
    table = _get_planting_plan_table()
    columns = _get_table_columns(url, table)
    if not columns:
        raise RuntimeError("无法读取种植计划表结构。")
    id_col = _resolve_column(columns, _PLAN_ID_COLUMNS)
    if not id_col:
        raise RuntimeError("种植计划表缺少 id/plan_id 字段。")

    conditions, params = _build_plan_conditions(filters, columns)
    order_column = None
    for candidate in ("updated_at", "created_at", "created_on", "sowing_date", id_col):
        if candidate in columns:
            order_column = candidate
            break

    sql = f"SELECT * FROM {quote_identifier(table)} WHERE " + " AND ".join(
        conditions
    )
    if order_column:
        sql += f" ORDER BY {quote_identifier(order_column)} DESC"
    sql += " LIMIT %s"
    params.append(max(1, int(limit)))
    try:
        rows = fetch_all(url, sql, tuple(params))
    except Exception as exc:
        raise RuntimeError(f"种植计划查询失败: {exc}") from exc
    return rows or [], id_col, columns


def list_active_planting_plans(
    *, limit: Optional[int] = 5
) -> Tuple[List[Dict[str, object]], str, List[str]]:
    url = _require_db_url()
    table = _get_planting_plan_table()
    columns = _get_table_columns(url, table)
    if not columns:
        raise RuntimeError("无法读取种植计划表结构。")
    id_col = _resolve_column(columns, _PLAN_ID_COLUMNS)
    if not id_col:
        raise RuntimeError("种植计划表缺少 id/plan_id 字段。")
    if "is_active" not in columns:
        return [], id_col, columns

    order_column = None
    for candidate in ("updated_at", "created_at", "created_on", "sowing_date", id_col):
        if candidate in columns:
            order_column = candidate
            break

    sql = (
        f"SELECT * FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier('is_active')} IS TRUE"
    )
    if order_column:
        sql += f" ORDER BY {quote_identifier(order_column)} DESC"
    params: Tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (max(1, int(limit)),)
    try:
        rows = fetch_all(url, sql, params)
    except Exception as exc:
        raise RuntimeError(f"种植计划查询失败: {exc}") from exc
    return rows or [], id_col, columns


def _fetch_planting_plan_row_by_id(
    plan_id: object,
) -> Tuple[Dict[str, object], str, List[str]]:
    url = _require_db_url()
    table = _get_planting_plan_table()
    columns = _get_table_columns(url, table)
    if not columns:
        raise RuntimeError("无法读取种植计划表结构。")
    id_col = _resolve_column(columns, _PLAN_ID_COLUMNS)
    if not id_col:
        raise RuntimeError("种植计划表缺少 id/plan_id 字段。")
    sql = (
        f"SELECT * FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(id_col)} = %s "
        "LIMIT 1"
    )
    try:
        rows = fetch_all(url, sql, (plan_id,))
    except Exception as exc:
        raise RuntimeError(f"种植计划查询失败: {exc}") from exc
    return (rows[0] if rows else {}), id_col, columns


def _fetch_growth_stage_rows_by_plan_id(
    plan_id: object,
) -> Tuple[List[Dict[str, object]], List[str]]:
    url = _require_db_url()
    table = _get_growth_stage_table()
    columns = _get_table_columns(url, table)
    if not columns:
        raise RuntimeError("无法读取生育期预测结果表结构。")
    plan_id_col = _resolve_column(columns, _FORECAST_PLAN_ID_COLUMNS)
    if not plan_id_col:
        raise RuntimeError("生育期预测结果表缺少 planting_plan_id/plan_id 字段。")

    order_column = None
    for candidate in (
        "updated_at",
        "update_date",
        "created_at",
        "created_on",
        "stage_date",
    ):
        if candidate in columns:
            order_column = candidate
            break

    sql = (
        f"SELECT * FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(plan_id_col)} = %s"
    )
    if order_column:
        sql += f" ORDER BY {quote_identifier(order_column)} DESC"
    try:
        rows = fetch_all(url, sql, (plan_id,))
    except Exception as exc:
        raise RuntimeError(f"生育期预测结果查询失败: {exc}") from exc
    return rows, columns


def _build_planting_from_plan_row(
    row: Dict[str, object],
    columns: List[str],
    fallback: Optional[PlantingDetails] = None,
) -> Optional[PlantingDetails]:
    if not row:
        return fallback
    data: Dict[str, object] = {}

    crop_col = _resolve_column(columns, _PLAN_CROP_COLUMNS)
    if crop_col:
        value = row.get(crop_col)
        if value is not None and str(value).strip():
            data["crop"] = str(value).strip()

    variety_id_col = _resolve_column(columns, _PLAN_VARIETY_ID_COLUMNS)
    if variety_id_col:
        value = row.get(variety_id_col)
        variety_name = _fetch_variety_name(value)
        if variety_name:
            data["variety"] = variety_name
    variety_col = _resolve_column(columns, _PLAN_VARIETY_COLUMNS)
    if variety_col and "variety" not in data:
        value = row.get(variety_col)
        if value is not None and str(value).strip():
            data["variety"] = str(value).strip()

    culti_col = _resolve_column(columns, _PLAN_CULTI_TYPE_COLUMNS)
    if culti_col:
        value = row.get(culti_col)
        culti_name = _code_to_name("culti_type", value)
        if culti_name:
            data["culti_type"] = culti_name

    method_col = _resolve_column(columns, _PLAN_METHOD_COLUMNS)
    if method_col:
        value = row.get(method_col)
        if value is not None and str(value).strip():
            data["planting_method"] = _normalize_method_for_planting(value)

    sow_col = _resolve_column(columns, _PLAN_SOWING_DATE_COLUMNS)
    if sow_col:
        sow = _parse_date(row.get(sow_col))
        if sow:
            data["sowing_date"] = sow

    trans_col = _resolve_column(columns, _PLAN_TRANSPLANT_DATE_COLUMNS)
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
    col = _resolve_column(columns, _PLAN_NAME_COLUMNS)
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
    row, id_col, columns = _fetch_planting_plan_row(filters)
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


def resolve_plan_and_planting(
    filters: Dict[str, object],
    *,
    fallback: Optional[PlantingDetails] = None,
) -> Tuple[object, Optional[PlantingDetails]]:
    plan_id, row, columns = resolve_planting_plan(filters)
    planting = _build_planting_from_plan_row(row, columns, fallback=fallback)
    return plan_id, planting


def query_growth_stage_from_plan_id(plan_id: object) -> GrowthStageResult:
    rows, columns = _fetch_growth_stage_rows_by_plan_id(plan_id)
    stages = _coerce_stages_from_rows(rows, columns)
    return GrowthStageResult(stages=stages)


def query_growth_stage_from_db(
    input: PredictGrowthStageInput,
) -> GrowthStageResult:
    filters = _build_plan_filters(input.planting)
    plan_id, _, _ = resolve_planting_plan(filters)
    return query_growth_stage_from_plan_id(plan_id)


def query_growth_stage_from_db_by_filters(
    filters: Dict[str, object],
) -> GrowthStageResult:
    plan_id, _, _ = resolve_planting_plan(filters)
    return query_growth_stage_from_plan_id(plan_id)
