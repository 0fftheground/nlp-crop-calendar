"""
LangGraph workflow for crop calendar planning.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from ...application.services.planting_service import extract_planting_details
from ...domain.planting import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ...infra.cache_keys import build_planting_cache_key
from ...infra.tool_cache import get_tool_result_cache
from ...infra.weather_cache import store_weather_series
from ...observability.logging_utils import get_trace_id, reset_trace_id, set_trace_id
from ...schemas import (
    OperationPlanResult,
    PlantingDetails,
    Recommendation,
    WorkflowResponse,
)
from ...prompts.workflow_messages import (
    build_crop_calendar_missing_question,
    format_crop_calendar_plan_message,
    format_memory_confirmation,
)
from ...prompts.tool_messages import TOOL_NOT_FOUND_MESSAGE
from ..tools.registry import execute_tool
from .common import (
    apply_memory_to_draft,
    coerce_planting_draft,
    coerce_weather_series,
    build_fallback_planting,
    classify_memory_confirmation,
    infer_unknown_fields,
    llm_extract_planting,
    summarize_weather_series,
)
from .state import GraphState, add_trace


CROP_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
}
CROP_CACHE_NAME = "crop_calendar_workflow"
CROP_CACHE_PROVIDER = "workflow"


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


def _get_cached_calendar_response(
    cache_key: str,
) -> Optional[WorkflowResponse]:
    cache = get_tool_result_cache()
    payload = cache.get(CROP_CACHE_NAME, CROP_CACHE_PROVIDER, cache_key)
    if not payload:
        return None
    try:
        return WorkflowResponse.model_validate(payload)
    except Exception:
        return None


def _store_calendar_response(
    cache_key: str, response: WorkflowResponse
) -> None:
    cache = get_tool_result_cache()
    cache.set(
        CROP_CACHE_NAME,
        CROP_CACHE_PROVIDER,
        cache_key,
        response.model_dump(mode="json"),
    )


def _extract_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    prior_draft = coerce_planting_draft(state.get("planting_draft"))
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
    memory_planting = state.get("memory_planting")
    memory_prompted = bool(state.get("memory_prompted"))
    memory_decision = state.get("memory_decision")
    memory_choice = classify_memory_confirmation(prompt, prompted=memory_prompted)
    if memory_choice is not None:
        memory_decision = memory_choice
    if memory_decision and memory_planting:
        draft = apply_memory_to_draft(draft, memory_planting)
        missing_fields = list_missing_required_fields(draft)
        state = add_trace(state, "memory_applied")
    if memory_choice is not None:
        state = add_trace(state, f"memory_decision={memory_decision}")
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
            "memory_decision": memory_decision,
        }
    )
    return state


def _ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    message = build_crop_calendar_missing_question(
        missing_fields,
        CROP_FIELD_LABELS,
    )
    memory_planting = state.get("memory_planting")
    memory_decision = state.get("memory_decision")
    memory_prompted = bool(state.get("memory_prompted"))
    if memory_planting and memory_decision is None and not memory_prompted:
        memory_note = format_memory_confirmation(memory_planting)
        message = f"{memory_note}\n{message}"
        state["memory_prompted"] = True
        state = add_trace(state, "memory_prompted")
    state = add_trace(state, f"ask missing={missing_fields}")
    state.update({"message": message})
    return state


def _fetch_variety_info(planting: PlantingDetails, prompt: str) -> Dict[str, object]:
    query = planting.variety or planting.crop or prompt
    result = execute_tool("variety_lookup", query)
    if not result:
        return {
            "name": "variety_lookup",
            "message": TOOL_NOT_FOUND_MESSAGE,
            "data": {},
        }
    return result.model_dump(mode="json")


def _fetch_weather_info(planting: PlantingDetails, prompt: str) -> Dict[str, object]:
    query = planting.region or prompt
    result = execute_tool("weather_lookup", query)
    if not result:
        weather_series = coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_series_ref = store_weather_series(weather_series)
        return {
            "name": "weather_lookup",
            "message": TOOL_NOT_FOUND_MESSAGE,
            "data": {
                "weather_series_ref": weather_series_ref,
                "summary": summarize_weather_series(weather_series),
            },
        }
    weather_series = coerce_weather_series(
        result.data, region=planting.region or "unknown"
    )
    weather_series_ref = store_weather_series(weather_series)
    return {
        "name": result.name,
        "message": result.message,
        "data": {
            "weather_series_ref": weather_series_ref,
            "summary": summarize_weather_series(weather_series),
        },
    }


def _context_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    draft = state.get("planting_draft")
    if draft is None:
        state = add_trace(state, "context missing draft")
        state.update(
            {
                "message": build_crop_calendar_missing_question(
                    list(CROP_FIELD_LABELS.keys()),
                    CROP_FIELD_LABELS,
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
                "message": build_crop_calendar_missing_question(
                    missing,
                    CROP_FIELD_LABELS,
                ),
            }
        )
        return state

    cache_key = build_planting_cache_key(planting)
    cached = _get_cached_calendar_response(cache_key)
    if cached:
        state = add_trace(state, "calendar_cache_hit")
        state.update(
            {
                "planting": planting,
                "assumptions": list(draft.assumptions),
                "recommendations": cached.recommendations,
                "message": cached.message,
                "data": cached.data,
                "cache_hit": True,
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
                "message": build_crop_calendar_missing_question(
                    list(CROP_FIELD_LABELS.keys()),
                    CROP_FIELD_LABELS,
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
        else {
            "name": "farming_recommendation",
            "message": TOOL_NOT_FOUND_MESSAGE,
            "data": {},
        }
    )
    weather_note = weather_info.get("message") or ""
    variety_note = variety_info.get("message") or ""
    recommendation_note = recommendation_info.get("message") or ""
    plan = _coerce_operation_plan(recommendation_info.get("data", {}))
    recommendations = (
        _build_recommendations_from_plan(plan, planting) if plan else []
    )
    message = format_crop_calendar_plan_message(
        planting,
        recommendations,
        assumptions,
        weather_note=weather_note,
        variety_note=variety_note,
        recommendation_note=recommendation_note,
    )

    state = add_trace(state, "recommend complete")
    cache_key = build_planting_cache_key(planting)
    _store_calendar_response(
        cache_key,
        WorkflowResponse(
            message=message,
            recommendations=recommendations,
        ),
    )
    state = add_trace(state, "calendar_cached")
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


def _route_after_context(state: GraphState) -> str:
    if state.get("cache_hit"):
        return END
    return "recommend"


def build_crop_calendar_graph():
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
    graph.add_conditional_edges("context", _route_after_context)
    graph.add_edge("recommend", END)
    return graph.compile()
