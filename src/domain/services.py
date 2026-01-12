from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, List, Optional, TypedDict

from ..infra.config import get_config
from ..infra.tool_provider import maybe_intranet_tool, normalize_provider
from ..infra.variety_store import retrieve_variety_candidates
from ..infra.weather_cache import (
    get_weather_series_by_query,
    store_weather_series_by_query,
)
from ..observability.logging_utils import log_event
from ..schemas import (
    FarmWorkRecommendInput,
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


class CropCalendarArtifacts(TypedDict):
    planting: PlantingDetails
    weather_series: WeatherSeries
    growth_stage: GrowthStageResult
    operation_plan: OperationPlanResult
    assumptions: List[str]


DEFAULT_CROP = "水稻"
CROP_REQUIRED_FIELDS = ["crop", "variety", "planting_method", "sowing_date"]


CROP_KEYWORDS = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
METHOD_KEYWORDS = {
    "直播": "direct_seeding",
    "撒播": "direct_seeding",
    "条播": "direct_seeding",
    "移栽": "transplanting",
    "插秧": "transplanting",
    "机插": "transplanting",
}
DATE_PATTERN = re.compile(r"(20\\d{2})[-/年](\\d{1,2})[-/月](\\d{1,2})")
REGION_PATTERN = re.compile(r"(\\w+[省市区县])")


class MissingPlantingInfoError(ValueError):
    """Raised when required planting fields are missing."""

    def __init__(self, missing_fields: List[str]):
        self.missing_fields = missing_fields
        message = (
            "缺少关键种植信息，请向用户追问："
            + "、".join(missing_fields)
            + "。若用户明确表示不知道，可允许使用默认值。"
        )
        super().__init__(message)


def list_missing_required_fields(draft: PlantingDetailsDraft) -> List[str]:
    payload = draft.model_dump(exclude_none=True)
    return [field for field in CROP_REQUIRED_FIELDS if payload.get(field) is None]


def merge_planting_answers(
    draft: PlantingDetailsDraft,
    *,
    answers: Optional[Dict[str, object]] = None,
    unknown_fields: Optional[List[str]] = None,
    fallback: Optional[PlantingDetails] = None,
) -> PlantingDetailsDraft:
    """
    Merge structured user answers back into the draft and apply defaults when necessary.
    """
    data = draft.model_dump()
    assumptions = list(data.get("assumptions") or [])

    if answers:
        for key, value in answers.items():
            if value is None:
                continue
            if key in PlantingDetailsDraft.model_fields:
                data[key] = value

    if unknown_fields:
        if fallback is None:
            raise MissingPlantingInfoError(unknown_fields)
        for field in unknown_fields:
            default_value = getattr(fallback, field, None)
            if default_value is None:
                raise MissingPlantingInfoError([field])
            data[field] = default_value
            assumptions.append(f"{field}: 用户不知道，使用默认值 {default_value}")

    data["assumptions"] = assumptions
    return PlantingDetailsDraft(**data)


def _infer_variety_from_prompt(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    candidates = retrieve_variety_candidates(prompt, limit=1)
    if candidates:
        return candidates[0]
    return None


def _is_known_variety(candidate: str) -> bool:
    if not candidate:
        return False
    candidates = retrieve_variety_candidates(candidate, limit=1)
    return bool(candidates) and candidates[0] == candidate


def _apply_heuristics(data: Dict[str, object], prompt: str) -> None:
    if "crop" not in data:
        for crop in CROP_KEYWORDS:
            if crop in prompt:
                data["crop"] = crop
                break

    if "planting_method" not in data:
        for alias, canonical in METHOD_KEYWORDS.items():
            if alias in prompt:
                data["planting_method"] = canonical
                break

    if "sowing_date" not in data:
        date_match = DATE_PATTERN.search(prompt)
        if date_match:
            year, month, day = date_match.groups()
            month = month.zfill(2)
            day = day.zfill(2)
            iso_string = f"{year}-{month}-{day}"
            data["sowing_date"] = date.fromisoformat(iso_string)

    if "region" not in data:
        region_match = REGION_PATTERN.search(prompt)
        if region_match:
            data["region"] = region_match.group(1)

    if "variety" not in data:
        variety = _infer_variety_from_prompt(prompt)
        if variety:
            data["variety"] = variety


def _sanitize_crop_field(data: Dict[str, object], prompt: str) -> None:
    crop = data.get("crop")
    if not isinstance(crop, str):
        return
    candidate = crop.strip()
    if not candidate:
        data.pop("crop", None)
        return
    if candidate in CROP_KEYWORDS:
        return
    variety = data.get("variety")
    if isinstance(variety, str) and variety.strip() == candidate:
        data.pop("crop", None)
        return
    if _is_known_variety(candidate):
        if not data.get("variety"):
            data["variety"] = candidate
        data.pop("crop", None)
        return
    if candidate not in prompt:
        data.pop("crop", None)


def _apply_rice_default(data: Dict[str, object]) -> None:
    if data.get("crop"):
        return
    data["crop"] = DEFAULT_CROP


def extract_planting_details(
    prompt: str,
    *,
    llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
) -> PlantingDetailsDraft:
    """
    调用真实 LLM 抽取种植信息；若没有提供 llm_extract，则退回到启发式解析。
    """
    data: Dict[str, object] = {"source_text": prompt}

    if llm_extract is not None:
        raw_payload = llm_extract(prompt)
        if not isinstance(raw_payload, dict):
            raise TypeError("llm_extract 必须返回包含字段的 dict。")
        data.update(raw_payload)
        _apply_heuristics(data, prompt)
        _sanitize_crop_field(data, prompt)
        _apply_rice_default(data)
        data.setdefault("confidence", 0.9)
        return PlantingDetailsDraft(**data)

    _apply_heuristics(data, prompt)
    _sanitize_crop_field(data, prompt)
    _apply_rice_default(data)
    data["confidence"] = 0.4 if len(data) == 1 else 0.75
    return PlantingDetailsDraft(**data)


def normalize_and_validate_planting(draft: PlantingDetailsDraft) -> PlantingDetails:
    """
    将完成的 Draft 转换为严格的 PlantingDetails。
    """
    missing = list_missing_required_fields(draft)
    if missing:
        raise MissingPlantingInfoError(missing)

    try:
        planting = draft.to_canonical()
        log_event(
            "normalized_planting",
            planting=planting.model_dump(mode="json"),
        )
        return planting
    except ValueError as exc:
        raise ValueError(f"种植信息校验失败: {exc}") from exc


def derive_weather_range(
    planting: PlantingDetails,
    *,
    duration_days: int = 160,
    default_region: Optional[str] = None,
) -> WeatherQueryInput:
    """
    Infer the weather query year based on sowing/transplanting dates.
    """
    region = planting.region or default_region
    if not region:
        raise ValueError("查询气象必须提供地区信息。")

    return WeatherQueryInput(
        region=region,
        year=planting.sowing_date.year,
        granularity="daily",
        include_advice=True,
    )


def fetch_weather(query: WeatherQueryInput) -> WeatherSeries:
    """
    Invoke the weather tool/service. This demo returns synthetic data.
    """
    cached = get_weather_series_by_query(query)
    if cached is not None:
        return cached
    cfg = get_config()
    provider = normalize_provider(cfg.weather_provider)
    prompt = json.dumps(query.model_dump(mode="json"), ensure_ascii=True, default=str)
    tool_payload = maybe_intranet_tool(
        "weather_lookup",
        prompt,
        provider,
        cfg.weather_api_url,
        cfg.weather_api_key,
    )
    weather_series = None
    if tool_payload:
        weather_series = _coerce_tool_weather_series(tool_payload.data, query)
    if weather_series is None:
        weather_series = get_farm_weather(query)
    store_weather_series_by_query(query, weather_series)
    return weather_series


def assemble_weather_series(
    raw: WeatherSeries, query: Optional[WeatherQueryInput] = None
) -> WeatherSeries:
    """
    Ensure the weather payload conforms to WeatherSeries for downstream tasks.
    """
    payload = raw.model_dump()
    if query:
        payload.setdefault("start_date", date(query.year, 1, 1))
        payload.setdefault("end_date", date(query.year, 12, 31))
    allowed = {"region", "granularity", "start_date", "end_date", "points", "source"}
    payload = {k: v for k, v in payload.items() if k in allowed}
    return WeatherSeries(**payload)


def _coerce_tool_weather_series(
    data: Dict[str, object], query: WeatherQueryInput
) -> Optional[WeatherSeries]:
    if not data:
        return None
    try:
        series = WeatherSeries.model_validate(data)
    except Exception:
        return None
    return assemble_weather_series(series, query)


def _default_weather_series(planting: PlantingDetails) -> WeatherSeries:
    return WeatherSeries(
        region=planting.region or "unknown",
        granularity="daily",
        start_date=planting.sowing_date,
        end_date=None,
        points=[],
        source="synthetic",
    )


def _coerce_operation_plan(data: Dict[str, object]) -> Optional[OperationPlanResult]:
    if not data:
        return None
    try:
        return OperationPlanResult.model_validate(data)
    except Exception:
        return None


def _coerce_growth_stage_result(
    data: Dict[str, object],
) -> Optional[GrowthStageResult]:
    if not data:
        return None
    try:
        return GrowthStageResult.model_validate(data)
    except Exception:
        pass
    nested = data.get("growth_stage")
    if isinstance(nested, dict):
        try:
            return GrowthStageResult.model_validate(nested)
        except Exception:
            return None
    return None


def predict_growth_stage(
    planting: PlantingDetails, weather_series: Optional[WeatherSeries] = None
) -> GrowthStageResult:
    """
    Helper wrapper that prepares PredictGrowthStageInput for the prediction service.
    """
    if weather_series is None:
        weather_series = WeatherSeries(
            region=planting.region or "unknown",
            granularity="daily",
            start_date=planting.sowing_date,
            end_date=None,
            points=[],
            source="synthetic",
        )
    request = PredictGrowthStageInput(planting=planting, weatherSeries=weather_series)
    cfg = get_config()
    provider = normalize_provider(cfg.growth_stage_provider)
    prompt = json.dumps(request.model_dump(mode="json"), ensure_ascii=True, default=str)
    tool_payload = maybe_intranet_tool(
        "growth_stage_prediction",
        prompt,
        provider,
        cfg.growth_stage_api_url,
        cfg.growth_stage_api_key,
    )
    if not tool_payload:
        raise ValueError(
            "生育期预测需要内网接口，请配置 GROWTH_STAGE_PROVIDER=intranet。"
        )
    result = _coerce_growth_stage_result(tool_payload.data)
    if result is None:
        raise ValueError(f"生育期接口返回异常: {tool_payload.message}")
    return result


def build_operation_plan(
    planting: PlantingDetails,
    weather_series: Optional[WeatherSeries] = None,
    *,
    user_prompt: str = "",
) -> OperationPlanResult:
    """
    Produce field operation recommendations using the normalized planting details.
    """
    cfg = get_config()
    provider = normalize_provider(cfg.recommendation_provider)
    context = {
        "user_prompt": user_prompt,
        "planting": planting.model_dump(mode="json"),
    }
    if weather_series:
        context["weather"] = weather_series.model_dump(mode="json")
    prompt = json.dumps(context, ensure_ascii=True, default=str)
    tool_payload = maybe_intranet_tool(
        "farming_recommendation",
        prompt,
        provider,
        cfg.recommendation_api_url,
        cfg.recommendation_api_key,
    )
    if tool_payload:
        plan = _coerce_operation_plan(tool_payload.data)
        if plan is not None:
            return plan
    if weather_series is None:
        weather_series = _default_weather_series(planting)
    request = FarmWorkRecommendInput(
        weatherSeries=weather_series,
        planting=planting,
    )
    return recommend_ops(request)


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
    growth_stage = predict_growth_stage(planting, weather_series)
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


def predict_growth_stage_gdd(input: PredictGrowthStageInput) -> GrowthStageResult:
    raise NotImplementedError(
        "生育期预测已迁移至内网接口，请改用 GROWTH_STAGE_PROVIDER=intranet。"
    )


def get_farm_weather(input: WeatherQueryInput) -> WeatherSeries:
    start = date(input.year, 1, 1)
    end = date(input.year, 12, 31)
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
        region=input.region,
        granularity=input.granularity,
        start_date=start,
        end_date=end,
        points=points,
        source="synthetic",
    )


def recommend_ops(input: FarmWorkRecommendInput) -> OperationPlanResult:
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
    summary = f"{input.crop} 农事建议（演示数据）。"
    return OperationPlanResult(crop=input.crop, summary=summary, operations=ops)
