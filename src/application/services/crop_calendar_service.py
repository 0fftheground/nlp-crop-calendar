from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, List, Optional, TypedDict

from ...infra.config import get_config
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.weather_cache import (
    get_weather_series_by_query,
    store_weather_series_by_query,
)
from ...schemas import (
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
from ...domain.planting import merge_planting_answers, normalize_and_validate_planting
from .planting_service import extract_planting_details


class CropCalendarArtifacts(TypedDict):
    planting: PlantingDetails
    weather_series: WeatherSeries
    growth_stage: GrowthStageResult
    operation_plan: OperationPlanResult
    assumptions: List[str]


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
