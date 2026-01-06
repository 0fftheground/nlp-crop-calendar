"""
LangGraph workflows for crop calendar planning and growth stage prediction.

Crop calendar workflow extracts planting details, asks follow-up questions if
required, and generates a recommendation plan once required fields are complete.
Growth stage workflow follows a similar flow, ending with stage prediction.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from ...domain.services import (
    MissingPlantingInfoError,
    extract_planting_details,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ...infra.config import get_config
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.weather_cache import get_weather_series, store_weather_series
from ...observability.logging_utils import get_trace_id, reset_trace_id, set_trace_id
from ...schemas import (
    GrowthStageResult,
    OperationPlanResult,
    PlantingDetails,
    PlantingDetailsDraft,
    PredictGrowthStageInput,
    Recommendation,
    WeatherSeries,
)
from ...tools.registry import execute_tool
from .common import (
    build_fallback_planting,
    format_missing_question,
    infer_unknown_fields,
    llm_extract_planting,
)
from .state import GraphState, add_trace


CROP_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
}
GROWTH_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "region": "地区",
}
PLANTING_METHOD_LABELS = {
    "direct_seeding": "直播",
    "transplanting": "移栽",
}


def _format_plan_message(
    planting: PlantingDetails,
    recommendations: List[Recommendation],
    assumptions: List[str],
    weather_note: Optional[str] = None,
    variety_note: Optional[str] = None,
    recommendation_note: Optional[str] = None,
) -> str:
    info_parts = [f"作物: {planting.crop}"]
    if planting.variety:
        info_parts.append(f"品种: {planting.variety}")
    if planting.region:
        info_parts.append(f"区域: {planting.region}")
    method_key = (
        planting.planting_method.value
        if hasattr(planting.planting_method, "value")
        else str(planting.planting_method)
    )
    method_label = PLANTING_METHOD_LABELS.get(method_key, method_key)
    info_parts.append(f"方式: {method_label}")
    info_parts.append(f"播种日期: {planting.sowing_date.isoformat()}")

    lines = ["已生成农事推荐。", "种植信息: " + "，".join(info_parts)]
    if weather_note:
        lines.append(f"气象信息: {weather_note}")
    if variety_note:
        lines.append(f"品种信息: {variety_note}")
    if recommendation_note:
        lines.append(f"推荐摘要: {recommendation_note}")
    if recommendations:
        lines.append("推荐操作:")
        for idx, rec in enumerate(recommendations, start=1):
            line = f"{idx}. {rec.title} - {rec.description}"
            lines.append(line)

    if assumptions:
        lines.append("默认/假设: " + "；".join(assumptions))

    return "\n".join(lines)


def _coerce_operation_plan(
    data: Dict[str, object],
) -> Optional[OperationPlanResult]:
    if not data:
        return None
    try:
        return OperationPlanResult.model_validate(data)
    except Exception:
        return None


def _build_recommendations_from_plan(
    plan: OperationPlanResult, planting: PlantingDetails
) -> List[Recommendation]:
    crop = plan.crop or planting.crop
    regions = [planting.region] if planting.region else []
    recommendations: List[Recommendation] = []
    for item in plan.operations:
        reasoning_parts = []
        if item.window:
            reasoning_parts.append(f"时间窗: {item.window}")
        if item.priority:
            reasoning_parts.append(f"优先级: {item.priority}")
        reasoning = "；".join(reasoning_parts)
        recommendations.append(
            Recommendation(
                crop=crop,
                stage=item.stage,
                title=item.title,
                description=item.description,
                reasoning=reasoning,
                months=[],
                regions=list(regions),
            )
        )
    return recommendations


def _coerce_planting_draft(value: object) -> Optional[PlantingDetailsDraft]:
    if value is None:
        return None
    if isinstance(value, PlantingDetailsDraft):
        return value
    if isinstance(value, dict):
        try:
            return PlantingDetailsDraft.model_validate(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            try:
                return PlantingDetailsDraft.model_validate(payload)
            except Exception:
                return None
    return None


def _extract_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    prior_draft = _coerce_planting_draft(state.get("planting_draft"))
    prior_missing = state.get("missing_fields") or []
    followup_count = state.get("followup_count", 0)
    try:
        fresh_draft = extract_planting_details(
            prompt, llm_extract=llm_extract_planting
        )
    except Exception as exc:
        state = add_trace(state, f"llm_extract_failed={exc}")
        fresh_draft = extract_planting_details(prompt)

    if prior_draft and prior_missing:
        answers = fresh_draft.model_dump(exclude_none=True)
        draft = merge_planting_answers(prior_draft, answers=answers)
        followup_count += 1
    else:
        draft = fresh_draft
    missing_fields = list_missing_required_fields(draft)
    unknown_fields = infer_unknown_fields(prompt, missing_fields, CROP_FIELD_LABELS)
    if unknown_fields:
        fallback = build_fallback_planting(draft)
        draft = merge_planting_answers(
            draft,
            unknown_fields=unknown_fields,
            fallback=fallback,
        )
        missing_fields = list_missing_required_fields(draft)
    elif missing_fields and followup_count >= 2:
        fallback = build_fallback_planting(draft)
        draft = merge_planting_answers(
            draft,
            unknown_fields=missing_fields,
            fallback=fallback,
        )
        missing_fields = list_missing_required_fields(draft)

    state = add_trace(
        state, f"extract missing={missing_fields} followup_count={followup_count}"
    )
    state.update(
        {
            "planting_draft": draft,
            "missing_fields": missing_fields,
            "followup_count": followup_count,
            "assumptions": list(draft.assumptions),
        }
    )
    return state


def _ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    message = format_missing_question(
        missing_fields,
        CROP_FIELD_LABELS,
        "为了给出农事推荐，还需要补充：",
    )
    state = add_trace(state, f"ask missing={missing_fields}")
    state.update({"message": message})
    return state


def _fetch_variety_info(planting: PlantingDetails, prompt: str) -> Dict[str, object]:
    query = planting.variety or planting.crop or prompt
    result = execute_tool("variety_lookup", query)
    if not result:
        return {"name": "variety_lookup", "message": "tool not found", "data": {}}
    return result.model_dump(mode="json")


def _fetch_weather_info(planting: PlantingDetails, prompt: str) -> Dict[str, object]:
    query = planting.region or prompt
    result = execute_tool("weather_lookup", query)
    if not result:
        weather_series = _growth_coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_series_ref = store_weather_series(weather_series)
        return {
            "name": "weather_lookup",
            "message": "tool not found",
            "data": {
                "weather_series_ref": weather_series_ref,
                "summary": _summarize_weather_series(weather_series),
            },
        }
    weather_series = _growth_coerce_weather_series(
        result.data, region=planting.region or "unknown"
    )
    weather_series_ref = store_weather_series(weather_series)
    return {
        "name": result.name,
        "message": result.message,
        "data": {
            "weather_series_ref": weather_series_ref,
            "summary": _summarize_weather_series(weather_series),
        },
    }


def _context_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    draft = state.get("planting_draft")
    if draft is None:
        state = add_trace(state, "context missing draft")
        state.update(
            {
                "message": format_missing_question(
                    list(CROP_FIELD_LABELS.keys()),
                    CROP_FIELD_LABELS,
                    "为了给出农事推荐，还需要补充：",
                )
            }
        )
        return state

    try:
        planting = normalize_and_validate_planting(draft)
    except MissingPlantingInfoError as exc:
        missing = exc.missing_fields
        state = add_trace(state, f"context missing={missing}")
        state.update(
            {
                "missing_fields": missing,
                "message": format_missing_question(
                    missing,
                    CROP_FIELD_LABELS,
                    "为了给出农事推荐，还需要补充：",
                ),
            }
        )
        return state

    trace_id = get_trace_id()

    def _wrap_with_trace(func):
        def _inner(*args, **kwargs):
            token = set_trace_id(trace_id)
            try:
                return func(*args, **kwargs)
            finally:
                reset_trace_id(token)

        return _inner

    with ThreadPoolExecutor(max_workers=2) as executor:
        weather_future = executor.submit(
            _wrap_with_trace(_fetch_weather_info), planting, prompt
        )
        variety_future = executor.submit(
            _wrap_with_trace(_fetch_variety_info), planting, prompt
        )
        weather_info = weather_future.result()
        variety_info = variety_future.result()

    state = add_trace(state, "context ready")
    state.update(
        {
            "planting": planting,
            "assumptions": list(draft.assumptions),
            "weather_info": weather_info,
            "variety_info": variety_info,
        }
    )
    return state


def _recommend_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    planting = state.get("planting")
    weather_info = state.get("weather_info") or {}
    variety_info = state.get("variety_info") or {}
    assumptions = state.get("assumptions", [])
    if planting is None:
        state = add_trace(state, "recommend missing planting")
        state.update(
            {
                "message": format_missing_question(
                    list(CROP_FIELD_LABELS.keys()),
                    CROP_FIELD_LABELS,
                    "为了给出农事推荐，还需要补充：",
                )
            }
        )
        return state

    recommendation_context = {
        "user_prompt": prompt,
        "planting": planting.model_dump(mode="json"),
        "weather": weather_info,
        "variety": variety_info,
    }
    recommendation_prompt = json.dumps(
        recommendation_context, ensure_ascii=True, default=str
    )
    recommendation_payload = execute_tool(
        "farming_recommendation",
        recommendation_prompt,
    )
    recommendation_info = (
        recommendation_payload.model_dump(mode="json")
        if recommendation_payload
        else {"name": "farming_recommendation", "message": "tool not found", "data": {}}
    )
    weather_note = weather_info.get("message") or ""
    variety_note = variety_info.get("message") or ""
    recommendation_note = recommendation_info.get("message") or ""
    plan = _coerce_operation_plan(recommendation_info.get("data", {}))
    recommendations = (
        _build_recommendations_from_plan(plan, planting) if plan else []
    )
    message = _format_plan_message(
        planting,
        recommendations,
        assumptions,
        weather_note=weather_note,
        variety_note=variety_note,
        recommendation_note=recommendation_note,
    )

    state = add_trace(state, "recommend complete")
    state.update(
        {
            "recommendations": recommendations,
            "message": message,
            "weather_info": weather_info,
            "variety_info": variety_info,
            "recommendation_info": recommendation_info,
        }
    )
    return state


def _route_after_extract(state: GraphState) -> str:
    missing = state.get("missing_fields") or []
    return "ask" if missing else "context"


def build_graph():
    """
    Construct and return the crop calendar LangGraph workflow.
    """
    graph = StateGraph(GraphState)
    graph.add_node("extract", _extract_node)
    graph.add_node("ask", _ask_node)
    graph.add_node("context", _context_node)
    graph.add_node("recommend", _recommend_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _route_after_extract)
    graph.add_edge("ask", END)
    graph.add_edge("context", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def _growth_format_missing_question(missing_fields: List[str]) -> str:
    return format_missing_question(
        missing_fields,
        GROWTH_FIELD_LABELS,
        "生育期预测还需要补充：",
    )


def _growth_coerce_weather_series(data: Dict[str, object], *, region: str) -> WeatherSeries:
    if data:
        try:
            return WeatherSeries.model_validate(data)
        except Exception:
            pass
    return WeatherSeries(
        region=region or "unknown",
        granularity="daily",
        start_date=None,
        end_date=None,
        points=[],
        source="workflow",
    )


def _summarize_weather_series(weather_series: WeatherSeries) -> Dict[str, object]:
    return {
        "region": weather_series.region,
        "start_date": (
            weather_series.start_date.isoformat()
            if weather_series.start_date
            else None
        ),
        "end_date": (
            weather_series.end_date.isoformat() if weather_series.end_date else None
        ),
        "points": len(weather_series.points),
        "source": weather_series.source,
    }


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


def _growth_format_growth_stage_message(
    planting: PlantingDetails,
    stages: Dict[str, str],
    weather_note: str = "",
    variety_note: str = "",
) -> str:
    lines = [
        "已完成生育期预测。",
        (
            f"作物: {planting.crop}，品种: {planting.variety or '未知'}，"
            f"地区: {planting.region or '未知'}"
        ),
        f"播种日期: {planting.sowing_date.isoformat()}",
    ]
    predicted = stages.get("predicted_stage")
    next_stage = stages.get("estimated_next_stage")
    if predicted or next_stage:
        lines.append(f"当前阶段: {predicted}，预计下一阶段: {next_stage}")
    gdd = stages.get("gdd_accumulated")
    gdd_required = stages.get("gdd_required_maturity")
    base_temp = stages.get("base_temperature")
    if gdd and gdd_required and base_temp:
        lines.append(f"积温: {gdd}/{gdd_required} (基温 {base_temp})")
    if weather_note:
        lines.append(f"气象信息: {weather_note}")
    if variety_note:
        lines.append(f"品种信息: {variety_note}")
    return "\n".join(lines)


def _growth_extract_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    prior_draft = _coerce_planting_draft(state.get("planting_draft"))
    prior_missing = state.get("missing_fields") or []
    followup_count = state.get("followup_count", 0)
    try:
        fresh_draft = extract_planting_details(
            prompt, llm_extract=llm_extract_planting
        )
    except Exception as exc:
        state = add_trace(state, f"llm_extract_failed={exc}")
        fresh_draft = extract_planting_details(prompt)

    if prior_draft and prior_missing:
        answers = fresh_draft.model_dump(exclude_none=True)
        draft = merge_planting_answers(prior_draft, answers=answers)
        followup_count += 1
    else:
        draft = fresh_draft

    missing_fields = list_missing_required_fields(draft)
    if not draft.region:
        missing_fields.append("region")
    missing_fields = list(dict.fromkeys(missing_fields))
    unknown_fields = infer_unknown_fields(prompt, missing_fields, GROWTH_FIELD_LABELS)
    if unknown_fields:
        fallback = build_fallback_planting(draft)
        draft = merge_planting_answers(
            draft,
            unknown_fields=unknown_fields,
            fallback=fallback,
        )
        missing_fields = list_missing_required_fields(draft)
        if not draft.region:
            missing_fields.append("region")
        missing_fields = list(dict.fromkeys(missing_fields))
    elif missing_fields and followup_count >= 2:
        fallback = build_fallback_planting(draft)
        draft = merge_planting_answers(
            draft,
            unknown_fields=missing_fields,
            fallback=fallback,
        )
        missing_fields = list_missing_required_fields(draft)
        if not draft.region:
            missing_fields.append("region")
        missing_fields = list(dict.fromkeys(missing_fields))

    state = add_trace(
        state, f"extract missing={missing_fields} followup_count={followup_count}"
    )
    state.update(
        {
            "planting_draft": draft,
            "missing_fields": missing_fields,
            "followup_count": followup_count,
            "assumptions": list(draft.assumptions),
        }
    )
    return state


def _growth_ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    message = _growth_format_missing_question(missing_fields)
    state = add_trace(state, f"ask missing={missing_fields}")
    state.update({"message": message})
    return state


def _growth_weather_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    draft = state.get("planting_draft")
    if draft is None:
        state = add_trace(state, "weather missing draft")
        state.update(
            {
                "message": _growth_format_missing_question(
                    list(GROWTH_FIELD_LABELS.keys())
                )
            }
        )
        return state

    try:
        planting = normalize_and_validate_planting(draft)
    except MissingPlantingInfoError as exc:
        missing = exc.missing_fields
        if not draft.region and "region" not in missing:
            missing = list(missing) + ["region"]
        missing = list(dict.fromkeys(missing))
        state = add_trace(state, f"weather missing={missing}")
        state.update(
            {"missing_fields": missing, "message": _growth_format_missing_question(missing)}
        )
        return state
    except ValueError as exc:
        state = add_trace(state, f"weather invalid={exc}")
        state.update({"message": f"种植信息校验失败: {exc}"})
        return state

    query = planting.region or prompt
    result = execute_tool("weather_lookup", query)
    if not result:
        weather_series = _growth_coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_info = {
            "name": "weather_lookup",
            "message": "tool not found",
            "data": {},
        }
    else:
        weather_series = _growth_coerce_weather_series(
            result.data, region=planting.region or "unknown"
        )
        weather_info = {
            "name": result.name,
            "message": result.message,
            "data": {},
        }

    weather_series_ref = store_weather_series(weather_series)
    weather_info["data"] = {
        "weather_series_ref": weather_series_ref,
        "summary": _summarize_weather_series(weather_series),
    }

    state = add_trace(state, "weather ready")
    state.update(
        {
            "planting": planting,
            "weather_series_ref": weather_series_ref,
            "weather_info": weather_info,
            "assumptions": list(draft.assumptions),
        }
    )
    return state


def _growth_predict_node(state: GraphState) -> GraphState:
    planting = state.get("planting")
    weather_series_ref = state.get("weather_series_ref")
    weather_info = state.get("weather_info") or {}
    if planting is None:
        state = add_trace(state, "predict missing planting")
        state.update(
            {
                "message": _growth_format_missing_question(
                    list(GROWTH_FIELD_LABELS.keys())
                )
            }
        )
        return state
    weather_series = get_weather_series(weather_series_ref)
    if weather_series is None:
        weather_series = _growth_coerce_weather_series(
            {}, region=planting.region or "unknown"
        )

    crop = planting.crop
    variety = planting.variety or f"{crop}示例品种"
    variety_prompt = f"{crop} {variety}".strip()
    variety_result = execute_tool("variety_lookup", variety_prompt)
    if variety_result:
        variety_info = variety_result.model_dump(mode="json")
        variety_data = variety_result.data
        state = add_trace(state, "variety_lookup ok")
    else:
        variety_info = {"name": "variety_lookup", "message": "tool not found", "data": {}}
        variety_data = {}
        state = add_trace(state, "variety_lookup missing")

    request = PredictGrowthStageInput(planting=planting, weatherSeries=weather_series)
    request_payload = json.dumps(
        request.model_dump(mode="json"), ensure_ascii=True, default=str
    )
    cfg = get_config()
    provider = normalize_provider(cfg.growth_stage_provider)
    tool_payload = maybe_intranet_tool(
        "growth_stage_prediction",
        request_payload,
        provider,
        cfg.growth_stage_api_url,
        cfg.growth_stage_api_key,
    )

    weather_note = weather_info.get("message") or ""
    variety_note = variety_info.get("message") or ""
    weather_summary = _summarize_weather_series(weather_series)
    workflow_payload = {
        "inputs": {
            "planting": planting.model_dump(exclude_none=True, mode="json"),
        },
        "variety": variety_data,
        "weather_summary": weather_summary,
    }

    if not tool_payload:
        state = add_trace(state, "growth_stage_intranet missing")
        trace = list(state.get("trace") or [])
        workflow_payload["trace"] = trace
        state.update(
            {
                "message": "生育期预测需要内网接口，请配置 GROWTH_STAGE_PROVIDER=intranet。",
                "data": {"workflow": workflow_payload},
                "weather_info": weather_info,
                "variety_info": variety_info,
            }
        )
        return state

    state = add_trace(state, "growth_stage_intranet ok")
    result = _coerce_growth_stage_result(tool_payload.data or {})
    if result:
        message = _growth_format_growth_stage_message(
            planting,
            result.stages,
            weather_note=weather_note,
            variety_note=variety_note,
        )
    else:
        message = tool_payload.message or "已完成生育期预测。"
    state = add_trace(state, "predict complete")
    trace = list(state.get("trace") or [])
    workflow_payload.update(
        {
            "trace": trace,
            "provider_message": tool_payload.message,
            "provider_response": tool_payload.data,
        }
    )
    data_payload = {"workflow": workflow_payload}
    if result:
        data_payload["growth_stage"] = result.model_dump(mode="json")
    state.update(
        {
            "growth_stage": result,
            "message": message,
            "data": data_payload,
            "weather_info": weather_info,
            "variety_info": variety_info,
        }
    )
    return state


def _growth_route_after_extract(state: GraphState) -> str:
    missing = state.get("missing_fields") or []
    return "ask" if missing else "weather"


def build_growth_stage_graph():
    """
    Construct and return the growth stage prediction LangGraph workflow.
    """
    graph = StateGraph(GraphState)
    graph.add_node("extract", _growth_extract_node)
    graph.add_node("ask", _growth_ask_node)
    graph.add_node("weather", _growth_weather_node)
    graph.add_node("predict", _growth_predict_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _growth_route_after_extract)
    graph.add_edge("ask", END)
    graph.add_edge("weather", "predict")
    graph.add_edge("predict", END)
    return graph.compile()
