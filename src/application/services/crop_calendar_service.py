from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, List, Optional, TypedDict

from .growth_stage_service import query_growth_stage_from_planting
from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...infra.db_catalog import (
    TABLE_KEY_VARIETY,
    resolve_db_table,
    resolve_region_lookup_sources,
)
from ...infra.tool_provider import normalize_provider
from ...observability.logging_utils import log_event, summarize_text
from ...schemas import (
    GrowthStageResult,
    OperationPlanResult,
    OperationItem,
    PlantingDetails,
    PlantingDetailsDraft,
    PredictGrowthStageInput,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)
from ...domain.planting import merge_planting_answers
from .planting_service import (
    extract_planting_details,
    normalize_and_validate_planting,
)


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER


def configure_crop_calendar_ports(
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
    _CODE_MAP_CACHE.clear()
    _CODE_REVERSE_CACHE.clear()


def _cfg():
    return _CONFIG_PORT.get()


def _fetch_all(url: str, sql: str, params: tuple[object, ...] = ()) -> list[dict]:
    return _SQL_PORT.fetch_all(url, sql, params)


def _qid(name: str) -> str:
    return _SQL_PORT.quote_identifier(name)


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


class CropCalendarArtifacts(TypedDict):
    planting: PlantingDetails
    weather_series: WeatherSeries
    growth_stage: GrowthStageResult
    operation_plan: OperationPlanResult
    assumptions: List[str]


class CropCalendarPlanResult(TypedDict):
    operation_plan: OperationPlanResult
    growth_stage: GrowthStageResult
    plant_season_id: Optional[object]
    resolved_region_id: Optional[object]
    raw: Dict[str, object]


def derive_weather_range(
    planting: PlantingDetails,
    *,
    duration_days: int = 160,
    default_region: Optional[str] = None,
) -> WeatherQueryInput:
    """
    Infer the weather query year based on sowing/transplanting dates.
    """
    cfg = _cfg()
    region = default_region or cfg.default_region
    if not region:
        raise ValueError("查询气象必须提供地区信息。")

    start_date = planting.sowing_date
    end_date = planting.sowing_date + timedelta(days=max(1, int(duration_days)) - 1)
    return WeatherQueryInput(
        region=region,
        start_date=start_date,
        end_date=end_date,
        year=start_date.year,
        granularity="daily",
        include_advice=True,
    )


def fetch_weather(query: WeatherQueryInput) -> WeatherSeries:
    """
    Invoke the weather tool/service. This demo returns synthetic data.
    """
    return get_farm_weather(query)


def assemble_weather_series(
    raw: WeatherSeries, query: Optional[WeatherQueryInput] = None
) -> WeatherSeries:
    """
    Ensure the weather payload conforms to WeatherSeries for downstream tasks.
    """
    payload = raw.model_dump()
    if query:
        base_year = query.year
        if query.start_date:
            base_year = query.start_date.year
        elif query.end_date:
            base_year = query.end_date.year
        payload.setdefault(
            "start_date", query.start_date or date(base_year, 1, 1)
        )
        payload.setdefault(
            "end_date", query.end_date or date(base_year, 12, 31)
        )
    allowed = {"region", "granularity", "start_date", "end_date", "points", "source"}
    payload = {k: v for k, v in payload.items() if k in allowed}
    return WeatherSeries(**payload)


def query_growth_stage(
    planting: PlantingDetails, weather_series: Optional[WeatherSeries] = None
) -> GrowthStageResult:
    """
    Helper wrapper that prepares PredictGrowthStageInput for the query service.
    """
    if weather_series is None:
        cfg = _cfg()
        weather_series = WeatherSeries(
            region=cfg.default_region or "unknown",
            granularity="daily",
            start_date=planting.sowing_date,
            end_date=None,
            points=[],
            source="synthetic",
        )
    request = PredictGrowthStageInput(planting=planting, weatherSeries=weather_series)
    return query_growth_stage_from_planting(request)


def build_operation_plan(
    planting: PlantingDetails,
    weather_series: Optional[WeatherSeries] = None,
    *,
    user_prompt: str = "",
) -> OperationPlanResult:
    """
    Produce field operation recommendations using the normalized planting details.
    """
    del weather_series, user_prompt
    return request_operation_plan(planting)


def generate_crop_calendar(
    user_prompt: str,
    *,
    draft_override: Optional[PlantingDetailsDraft] = None,
    llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
    answers: Optional[Dict[str, object]] = None,
    unknown_fields: Optional[List[str]] = None,
    fallback_planting: Optional[PlantingDetails] = None,
    weather_duration_days: int = 120,
    default_region: Optional[str] = None,
) -> CropCalendarArtifacts:
    """
    Full pipeline: free-form sentence -> crop calendar assets.

    若抽取后仍缺字段，请使用 list_missing_required_fields() 获得一次性补问清单；
    再用 merge_planting_answers() 把用户补充合并进 draft，或在此函数中传入 answers/unknown_fields。
    """
    draft = draft_override or extract_planting_details(
        user_prompt,
        llm_extract=llm_extract,
    )

    if answers or unknown_fields:
        draft = merge_planting_answers(
            draft,
            answers=answers,
            unknown_fields=unknown_fields,
            fallback=fallback_planting,
        )

    planting = normalize_and_validate_planting(draft)
    weather_query = derive_weather_range(
        planting,
        duration_days=weather_duration_days,
        default_region=default_region,
    )
    weather_result = fetch_weather(weather_query)
    weather_series = assemble_weather_series(weather_result, weather_query)
    growth_stage = query_growth_stage(planting, weather_series)
    operation_plan = build_operation_plan(
        planting,
        weather_series,
        user_prompt=user_prompt,
    )
    assumptions = list(draft.assumptions)
    return CropCalendarArtifacts(
        planting=planting,
        weather_series=weather_series,
        growth_stage=growth_stage,
        operation_plan=operation_plan,
        assumptions=assumptions,
    )


def get_farm_weather(input: WeatherQueryInput) -> WeatherSeries:
    cfg = _cfg()
    base_year = input.year
    if input.start_date:
        base_year = input.start_date.year
    elif input.end_date:
        base_year = input.end_date.year
    start = input.start_date or date(base_year, 1, 1)
    end = input.end_date or date(base_year, 12, 31)
    total_days = (end - start).days + 1
    points: List[WeatherDataPoint] = []

    for offset in range(total_days):
        current_date = start + timedelta(days=offset)
        temp = 20 + 5 * math.sin(offset / 14)
        precipitation = max(0.0, 5 * math.cos(offset / 21))
        point = WeatherDataPoint(
            timestamp=datetime.combine(current_date, time.min),
            temperature=round(temp, 1),
            temperature_max=round(temp + 3, 1),
            temperature_min=round(temp - 4, 1),
            humidity=60 + 20 * math.sin(offset / 10),
            precipitation=round(precipitation, 1),
            wind_speed=2 + math.sin(offset / 5),
            condition="sunny" if precipitation < 1 else "rain",
        )
        points.append(point)

    return WeatherSeries(
        region=input.region or cfg.default_region or "unknown",
        granularity=input.granularity,
        start_date=start,
        end_date=end,
        points=points,
        source="synthetic",
    )


def _mock_operation_plan(planting: PlantingDetails) -> OperationPlanResult:
    ops = [
        OperationItem(
            stage="field_preparation",
            title="清沟排水",
            description="播种前疏通田间沟系，确保雨后排水顺畅。",
            window="播种前 7 天",
            priority="medium",
        ),
        OperationItem(
            stage="seedling",
            title="查苗补苗",
            description="播后 10-15 天查苗，缺苗处适量补播。",
            window="出苗后 10 天",
            priority="high",
        ),
        OperationItem(
            stage="fertilization",
            title="分蘖肥",
            description="根据苗情施用分蘖肥，氮肥控制在 5-8 公斤/亩。",
            window="出苗后 20-30 天",
            priority="medium",
        ),
    ]
    summary = f"{planting.crop} 农事建议（演示数据）。"
    return OperationPlanResult(
        crop=planting.crop, summary=summary, operations=ops
    )


def _mock_crop_calendar_plan(
    planting: PlantingDetails,
) -> CropCalendarPlanResult:
    plan = _mock_operation_plan(planting)
    return {
        "operation_plan": plan,
        "growth_stage": GrowthStageResult(stages={}),
        "plant_season_id": None,
        "raw": {},
    }


_CODE_MAP_CACHE: Dict[str, Dict[int, str]] = {}
_CODE_REVERSE_CACHE: Dict[str, Dict[str, int]] = {}
_REGION_SUFFIX_RE = re.compile(r"(特别行政区|自治区|自治州|省|市|州|盟|地区|区|县)$")
_REGION_CODE_CATEGORIES = (
    "region",
    "region_id",
    "area",
    "district",
    "city",
    "county",
    "plant_region",
    "planting_region",
)


def _require_db_url() -> str:
    cfg = _cfg()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取品种或码表数据。")
    return cfg.agri_db_url


def _fetch_code_map(category: str) -> Dict[int, str]:
    url = _require_db_url()
    try:
        rows = _fetch_all(
            url,
            "SELECT code, code_name FROM agri_code_dict "
            "WHERE category = %s AND is_active = true",
            (category,),
        )
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


def _resolve_code(category: str, value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        code = int(value)
        mapping = _get_code_map(category)
        return code if mapping and code in mapping else None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        code = int(text)
        mapping = _get_code_map(category)
        return code if mapping and code in mapping else None
    reverse = _get_code_reverse_map(category)
    return reverse.get(text)


def _normalize_region_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[，。；、,.!！?？\s]+", "", text)
    return text


def _region_text_variants(value: object) -> List[str]:
    normalized = _normalize_region_token(value)
    if not normalized:
        return []
    variants = [normalized]
    trimmed = _REGION_SUFFIX_RE.sub("", normalized)
    if trimmed and trimmed not in variants:
        variants.append(trimmed)
    return variants


def _pick_region_code_from_mapping(
    mapping: Dict[int, str], region_text: object
) -> Optional[int]:
    variants = _region_text_variants(region_text)
    if not variants:
        return None
    best_code: Optional[int] = None
    best_score = -1
    for code, name in mapping.items():
        name_norm = _normalize_region_token(name)
        if not name_norm:
            continue
        name_trim = _REGION_SUFFIX_RE.sub("", name_norm)
        for candidate in (name_norm, name_trim):
            if not candidate:
                continue
            score = -1
            if any(v == candidate for v in variants):
                score = 100 + len(candidate)
            elif any(v in candidate for v in variants):
                score = 60 + min(len(v) for v in variants if v in candidate)
            elif any(candidate in v for v in variants):
                score = 50 + len(candidate)
            if score > best_score:
                best_score = score
                best_code = code
    return best_code


def _resolve_region_id_from_code_dict(region_text: object) -> Optional[int]:
    variants = _region_text_variants(region_text)
    if not variants:
        return None
    for category in _REGION_CODE_CATEGORIES:
        mapping = _get_code_map(category)
        if not mapping:
            continue
        code = _pick_region_code_from_mapping(mapping, variants[0])
        if code is not None:
            return code
    return None


def _coerce_region_id_value(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, bool):
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
    table_name = _qid(table)
    id_col = _qid(id_column)
    name_col = _qid(name_column)
    variants = _region_text_variants(region_text)
    if not variants:
        return None
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
        if not rows:
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


def _resolve_region_id_from_region_table(region_text: object) -> Optional[object]:
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


def _resolve_region_id_for_payload(value: object) -> Optional[object]:
    direct = _coerce_region_id_value(value)
    if isinstance(direct, int):
        return direct
    if not direct:
        return None
    resolved = _resolve_region_id_from_code_dict(direct)
    if resolved is not None:
        return resolved
    return _resolve_region_id_from_region_table(direct)


def _fetch_variety_id_by_name(variety_name: str) -> Optional[object]:
    if not variety_name:
        return None
    url = _require_db_url()
    table = resolve_db_table(_cfg(), TABLE_KEY_VARIETY)
    try:
        sql = f"SELECT id FROM {_qid(table)} WHERE name = %s LIMIT 1"
        rows = _fetch_all(url, sql, (variety_name,))
    except Exception:
        return None
    if not rows:
        return None
    return rows[0].get("id") if isinstance(rows[0], dict) else None


def _normalize_sowing_method_code(value: object) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            value = value.value
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return None
    mapping = {
        "direct_seeding": "直播",
        "直播": "直播",
        "撒播": "直播",
        "transplanting": "插秧",
        "移栽": "插秧",
        "插秧": "插秧",
        "抛秧": "插秧",
    }
    label = mapping.get(text, text)
    return _resolve_code("sowingmtd", label)


def _planting_method_value(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        try:
            return str(value.value).strip()
        except Exception:
            pass
    return str(value).strip()


def _normalize_culti_type_code(value: object) -> Optional[int]:
    if value is None:
        return None
    code = _resolve_code("culti_type", value)
    if code is not None:
        return code
    text = str(value).strip()
    aliases = {
        "一季稻": ("一季晚稻", "中稻"),
        "单季稻": ("一季晚稻", "中稻"),
        "晚稻": ("一季晚稻",),
    }
    for candidate in aliases.get(text, ()):
        code = _resolve_code("culti_type", candidate)
        if code is not None:
            return code
    return None


def _normalize_culti_type_label(value: object) -> Optional[str]:
    code = _normalize_culti_type_code(value)
    if code is None:
        return None
    return _get_code_map("culti_type").get(code)


def resolve_sowing_method_code(value: object) -> Optional[int]:
    return _normalize_sowing_method_code(value)


def resolve_culti_type_code(value: object) -> Optional[int]:
    return _normalize_culti_type_code(value)


def resolve_culti_type_label(value: object) -> Optional[str]:
    return _normalize_culti_type_label(value)


def list_code_names(category: str, *, limit: int = 8) -> List[str]:
    mapping = _get_code_map(category)
    if not mapping:
        return []
    names = [name for _, name in sorted(mapping.items()) if name]
    return names[: max(1, int(limit))]


def _build_crop_calendar_payload(
    planting: PlantingDetails,
) -> tuple[Dict[str, object], Optional[object]]:
    cfg = _cfg()
    explicit_farm_id = str(planting.farm_id or "").strip()
    region_raw = planting.region_id
    farm_id_raw = explicit_farm_id or (
        None if region_raw else getattr(cfg, "default_farm_id", None)
    )
    farm_id: Optional[int] = None
    resolved_region_id: Optional[object] = None
    if farm_id_raw:
        try:
            farm_id = int(str(farm_id_raw).strip())
        except Exception as exc:
            raise RuntimeError(f"farm_id 非法: {farm_id_raw}") from exc
    else:
        resolved_region_id = _resolve_region_id_for_payload(region_raw)
        if region_raw and resolved_region_id is None:
            raise RuntimeError(f"未匹配到区域 region_id: {region_raw}")
    if not planting.variety:
        raise RuntimeError("缺少品种信息，无法生成种植计划。")
    variety_id = _fetch_variety_id_by_name(planting.variety)
    if variety_id is None:
        raise RuntimeError(f"未找到品种ID: {planting.variety}")
    planting_method_value = _planting_method_value(planting.planting_method)
    sowing_method = _normalize_sowing_method_code(planting_method_value)
    if sowing_method is None:
        raise RuntimeError("无法解析 sowing_method 代码。")
    if planting.transplant_date is None and planting_method_value != "direct_seeding":
        raise RuntimeError("非直播模式必须提供 transp_date。")
    if not planting.culti_type:
        raise RuntimeError("缺少稻作类型信息，无法生成种植计划。")
    culti_type_code = _normalize_culti_type_code(planting.culti_type)
    if culti_type_code is None:
        raise RuntimeError("无法解析 culti_type 代码。")
    payload: Dict[str, object] = {
        "sowing_date": planting.sowing_date.isoformat(),
        "variety_id": variety_id,
        "sowing_method": sowing_method,
        "transp_date": (
            planting.transplant_date.isoformat()
            if planting.transplant_date
            else ""
        ),
        "culti_type": culti_type_code,
    }
    if resolved_region_id is not None:
        payload["region_id"] = _coerce_region_id_value(resolved_region_id)
    elif farm_id is not None:
        payload["farm_id"] = farm_id
    return payload, resolved_region_id


def _build_growth_stage_result(
    stages_payload: object,
) -> GrowthStageResult:
    if not isinstance(stages_payload, dict) or not stages_payload:
        return GrowthStageResult(stages={})
    stage_dates = {
        str(name): str(value)
        for name, value in stages_payload.items()
        if value is not None and str(value).strip()
    }
    if not stage_dates:
        return GrowthStageResult(stages={})
    return GrowthStageResult(
        stages={
            "stage_dates": json.dumps(stage_dates, ensure_ascii=False),
        }
    )


def _build_operation_plan_from_farmworks(
    farmworks: object,
    *,
    crop: str,
    plant_season_id: Optional[object] = None,
) -> OperationPlanResult:
    ops_with_sort_keys: List[tuple[Optional[date], int, OperationItem]] = []
    if isinstance(farmworks, dict):
        for idx, (title, date_text) in enumerate(farmworks.items()):
            if not title:
                continue
            if isinstance(date_text, (list, tuple, set)):
                date_values = [
                    str(item).strip()
                    for item in date_text
                    if item is not None and str(item).strip()
                ]
                date_str = "、".join(date_values)
                sort_key = _extract_operation_sort_date(date_values[0]) if date_values else None
            else:
                date_str = str(date_text).strip() if date_text is not None else ""
                sort_key = _extract_operation_sort_date(date_str)
                date_values = [date_str] if date_str else []
            op = OperationItem(
                stage=str(title),
                title=str(title),
                description=date_str,
                dates=date_values,
                window=None,
                priority="medium",
            )
            ops_with_sort_keys.append((sort_key, idx, op))
    ops = [
        item
        for _, _, item in sorted(
            ops_with_sort_keys,
            key=lambda row: (
                row[0] is None,
                row[0] or date.max,
                row[1],
            ),
        )
    ]
    summary = f"{crop} 农事方案"
    metadata: Dict[str, object] = {"source": "external"}
    if plant_season_id is not None:
        metadata["plant_season_id"] = plant_season_id
    return OperationPlanResult(
        crop=crop,
        summary=summary,
        operations=ops,
        metadata=metadata,
    )


def _extract_operation_sort_date(text: str) -> Optional[date]:
    if not text:
        return None
    value = str(text).strip()
    if not value:
        return None
    # Accept common formats from external APIs, e.g. 2026-03-01 / 2026/3/1 / 2026年3月1日
    match = re.search(
        r"(20\d{2})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})",
        value,
    )
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def request_crop_calendar_plan(
    planting: PlantingDetails,
) -> CropCalendarPlanResult:
    cfg = _cfg()
    provider = normalize_provider(cfg.crop_calendar_provider)
    if provider not in {"external", "api", "http"}:
        return _mock_crop_calendar_plan(planting)
    if not cfg.crop_calendar_api_url:
        raise RuntimeError("缺少 CROP_CALENDAR_API_URL，无法调用外部计算接口。")
    payload, resolved_region_id = _build_crop_calendar_payload(planting)
    log_event(
        "crop_calendar_api_request",
        url=cfg.crop_calendar_api_url,
        payload=payload,
    )
    headers = {"Content-Type": "application/json"}
    if cfg.crop_calendar_api_key:
        headers["Authorization"] = f"Bearer {cfg.crop_calendar_api_key}"
        headers["X-API-KEY"] = cfg.crop_calendar_api_key
    try:
        response = _post_json(
            cfg.crop_calendar_api_url,
            payload=payload,
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)
            response_text = summarize_text(
                getattr(resp, "text", str(exc)), limit=1200
            )
            log_event(
                "crop_calendar_api_http_error",
                url=cfg.crop_calendar_api_url,
                status_code=status_code,
                payload=payload,
                response_text=response_text,
            )
            raise RuntimeError(
                f"外部计算接口请求失败(status={status_code}): {response_text}"
            ) from exc
        log_event(
            "crop_calendar_api_request_error",
            url=cfg.crop_calendar_api_url,
            payload=payload,
            error=str(exc),
        )
        raise RuntimeError(f"外部计算接口请求失败: {exc}") from exc
    try:
        raw = response.json()
    except Exception as exc:
        raw_text = summarize_text(response.text or "", limit=1200)
        log_event(
            "crop_calendar_api_parse_error",
            url=cfg.crop_calendar_api_url,
            status_code=response.status_code,
            payload=payload,
            response_text=raw_text,
        )
        raise RuntimeError("外部计算接口返回格式未识别。") from exc
    log_event(
        "crop_calendar_api_response",
        url=cfg.crop_calendar_api_url,
        status_code=response.status_code,
        response_summary=summarize_text(
            json.dumps(raw, ensure_ascii=False, default=str), limit=1200
        ),
    )
    if not isinstance(raw, dict):
        raise RuntimeError("外部计算接口返回格式未识别。")
    code = str(raw.get("code", "")).strip()
    if code and code != "0":
        msg = raw.get("msg") or "计算接口返回失败。"
        log_event(
            "crop_calendar_api_business_error",
            url=cfg.crop_calendar_api_url,
            payload=payload,
            code=code,
            msg=str(msg),
        )
        raise RuntimeError(str(msg))
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    plant_season_id = data.get("plant_season_id")
    farmworks = data.get("farmworks") or {}
    growth_stages = data.get("growth_stages") or {}
    operation_plan = _build_operation_plan_from_farmworks(
        farmworks, crop=planting.crop, plant_season_id=plant_season_id
    )
    growth_stage = _build_growth_stage_result(growth_stages)
    return {
        "operation_plan": operation_plan,
        "growth_stage": growth_stage,
        "plant_season_id": plant_season_id,
        "resolved_region_id": resolved_region_id,
        "raw": raw,
    }


def set_crop_calendar_active(
    plant_season_id: object, *, is_active: bool = True
) -> Dict[str, object]:
    cfg = _cfg()
    if not cfg.crop_calendar_save_api_url:
        raise RuntimeError(
            "缺少 CROP_CALENDAR_SAVE_API_URL，无法保存种植计划。"
        )
    payload = {
        "plant_season_id": str(plant_season_id),
        "is_active": bool(is_active),
    }
    headers = {"Content-Type": "application/json"}
    if cfg.crop_calendar_api_key:
        headers["Authorization"] = f"Bearer {cfg.crop_calendar_api_key}"
        headers["X-API-KEY"] = cfg.crop_calendar_api_key
    try:
        response = _post_json(
            cfg.crop_calendar_save_api_url,
            payload=payload,
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as exc:
        raise RuntimeError(f"保存接口请求失败: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("保存接口返回格式未识别。")
    code = str(raw.get("code", "")).strip()
    if code and code != "0":
        msg = raw.get("msg") or "保存失败。"
        raise RuntimeError(str(msg))
    return raw


def _derive_crop_calendar_delete_url(cfg) -> Optional[str]:
    if cfg.crop_calendar_delete_api_url:
        return cfg.crop_calendar_delete_api_url
    candidates = [cfg.crop_calendar_api_url, cfg.crop_calendar_save_api_url]
    for url in candidates:
        if not url:
            continue
        if url.endswith("/cropCalender/previewCalender"):
            return url.replace("/cropCalender/previewCalender", "/cropCalender/plantPlan/delete")
        if url.endswith("/cropCalender/plantPlan/activate"):
            return url.replace("/cropCalender/plantPlan/activate", "/cropCalender/plantPlan/delete")
        if url.endswith("/cropCalender/plantPlan/setActive"):
            return url.replace("/cropCalender/plantPlan/setActive", "/cropCalender/plantPlan/delete")
        if url.endswith("/cropCalender/plantPlan/add"):
            return url.replace("/cropCalender/plantPlan/add", "/cropCalender/plantPlan/delete")
    return None


def delete_crop_calendar_plan(
    plant_season_id: object,
) -> Dict[str, object]:
    cfg = _cfg()
    delete_url = _derive_crop_calendar_delete_url(cfg)
    if not delete_url:
        raise RuntimeError(
            "缺少 CROP_CALENDAR_DELETE_API_URL，无法删除种植计划。"
        )
    payload = {"plant_season_id": str(plant_season_id)}
    headers = {"Content-Type": "application/json"}
    if cfg.crop_calendar_api_key:
        headers["Authorization"] = f"Bearer {cfg.crop_calendar_api_key}"
        headers["X-API-KEY"] = cfg.crop_calendar_api_key
    try:
        response = _post_json(
            delete_url,
            payload=payload,
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as exc:
        raise RuntimeError(f"删除接口请求失败: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("删除接口返回格式未识别。")
    code = str(raw.get("code", "")).strip()
    if code and code != "0":
        msg = raw.get("msg") or "删除失败。"
        raise RuntimeError(str(msg))
    return raw


def request_operation_plan(planting: PlantingDetails) -> OperationPlanResult:
    result = request_crop_calendar_plan(planting)
    return result["operation_plan"]
