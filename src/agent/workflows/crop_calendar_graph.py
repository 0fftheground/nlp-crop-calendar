"""
LangGraph workflow for crop calendar planning.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from ...application.services.planting_service import extract_planting_details
from ...application.services.weather_service import (
    lookup_goso_weather,
    normalize_weather_prompt,
)
from ...domain.planting import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ...infra.cache_keys import build_planting_cache_key
from ...infra.planting_choice_store import (
    build_choice_key,
    get_planting_choice_store,
)
from ...infra.tool_cache import get_tool_result_cache
from ...infra.variety_store import (
    find_exact_variety_in_text,
    load_variety_names,
    retrieve_variety_candidates,
)
from ...infra.weather_cache import store_weather_series
from ...observability.logging_utils import get_trace_id, reset_trace_id, set_trace_id
from ...observability.otel import (
    build_span_attributes,
    record_exception,
    start_span,
    summarize_state,
    wrap_with_otel_context,
)
from ...schemas import (
    OperationPlanResult,
    PlantingDetails,
    Recommendation,
    WeatherQueryInput,
    WorkflowResponse,
)
from ...prompts.workflow_messages import (
    build_crop_calendar_missing_question,
    build_future_weather_warning,
    format_crop_calendar_plan_message,
)
from ...prompts.tool_messages import TOOL_NOT_FOUND_MESSAGE
from ..tools.registry import execute_tool
from .common import (
    apply_experience_choice_to_draft,
    build_experience_notice,
    clear_experience_fields,
    coerce_planting_draft,
    coerce_weather_series,
    build_fallback_planting,
    detect_experience_change_fields,
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
_FOLLOWUP_INDEX_RE = re.compile(r"^第?\s*(\d+)\s*(?:个|条|项)?$")


@lru_cache(maxsize=1)
def _get_variety_name_set() -> set[str]:
    return set(load_variety_names())


def _is_known_variety(name: Optional[str]) -> bool:
    if not name:
        return False
    return name in _get_variety_name_set()


def _resolve_followup_candidate(
    answer: str, candidates: list[str]
) -> Optional[str]:
    if not answer or not candidates:
        return None
    text = answer.strip()
    match = _FOLLOWUP_INDEX_RE.match(text)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
    for candidate in candidates:
        if candidate == text:
            return candidate
    for candidate in candidates:
        if text in candidate or candidate in text:
            return candidate
    return None


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
    pending_options = list(state.get("pending_options") or [])
    followup_count = state.get("followup_count", 0)
    prior_future_warning = bool(state.get("future_sowing_date_warning"))
    try:
        fresh_draft = extract_planting_details(
            prompt, llm_extract=llm_extract_planting
        )
    except Exception as exc:
        state = add_trace(state, f"llm_extract_failed={exc}")
        fresh_draft = extract_planting_details(prompt)

    # Follow-up: merge newly extracted answers into the prior draft.
    if prior_draft and prior_missing:
        answers = fresh_draft.model_dump(exclude_none=True)
        draft = merge_planting_answers(prior_draft, answers=answers)
        followup_count += 1
    else:
        draft = fresh_draft
    if draft.variety is not None:
        draft = draft.model_copy(update={"variety": None})
    if prior_future_warning and (
        draft.sowing_date is None or draft.sowing_date.year >= 2026
    ):
        state = add_trace(state, "extract future_sowing_date_rejected")
        state.update(
            {
                "planting_draft": draft,
                "missing_fields": ["sowing_date"],
                "followup_count": followup_count,
                "pending_message": "播种日期仍为2026年及以后，请直接回复新的播种日期（2025年及以前）。",
                "future_sowing_date_warning": True,
                "pending_options": [],
            }
        )
        return state
    if draft.sowing_date and draft.sowing_date.year >= 2026:
        state = add_trace(state, "extract future_sowing_date")
        state.update(
            {
                "planting_draft": draft,
                "missing_fields": ["sowing_date"],
                "followup_count": followup_count,
                "pending_message": "暂不支持未来气象数据，请提供新的播种日期（2025年及以前）。",
                "future_sowing_date_warning": True,
                "pending_options": [],
            }
        )
        return state
    missing_fields = list_missing_required_fields(draft)
    is_followup = bool(prior_draft and prior_missing)
    experience_applied = list(state.get("experience_applied") or [])
    experience_skip_fields = set(state.get("experience_skip_fields") or [])
    experience_key = state.get("experience_key")
    experience_notice = state.get("experience_notice")
    change_fields = detect_experience_change_fields(prompt)
    # User explicitly wants to change fields -> clear any carried experience values.
    if change_fields:
        experience_skip_fields.update(change_fields)
        to_clear = [
            field for field in experience_applied if field in experience_skip_fields
        ]
        if to_clear:
            draft = clear_experience_fields(draft, to_clear)
            experience_applied = [
                field for field in experience_applied if field not in to_clear
            ]
        experience_notice = None
        state = add_trace(state, f"experience_change_fields={change_fields}")
        missing_fields = list_missing_required_fields(draft)
    if experience_skip_fields:
        experience_skip_fields = {
            field
            for field in experience_skip_fields
            if getattr(draft, field, None) is None
        }
    user_id = state.get("user_id")
    current_key = build_choice_key(draft.crop, draft.region) if user_id else None
    if current_key and experience_key and current_key != experience_key:
        experience_applied = []
        experience_notice = None
    # Apply stored planting defaults for this user+crop+region unless skipped.
    if user_id and current_key:
        choice = get_planting_choice_store().get(
            user_id, draft.crop, draft.region
        )
        if choice:
            draft, applied = apply_experience_choice_to_draft(
                draft,
                choice.planting,
                skip_fields=experience_skip_fields,
            )
            if applied:
                experience_applied = list(
                    dict.fromkeys(experience_applied + applied)
                )
                experience_notice = (
                    build_experience_notice(choice.planting, applied)
                    or experience_notice
                )
                state = add_trace(state, f"experience_applied={applied}")
            missing_fields = list_missing_required_fields(draft)
    # Resolve variety selection from the previous candidate list.
    resolved_from_followup = False
    if prior_missing and "variety" in prior_missing and pending_options:
        resolved = _resolve_followup_candidate(prompt, pending_options)
        if resolved:
            draft = draft.model_copy(update={"variety": resolved})
            missing_fields = list_missing_required_fields(draft)
            resolved_from_followup = True
    variety_candidates: List[str] = []
    prompt_candidates: List[str] = []
    exact_variety = None
    should_check_prompt = (not is_followup) or (
        prior_missing and "variety" in prior_missing
    )
    if resolved_from_followup:
        should_check_prompt = False
    # Prefer DB-driven exact match from the raw prompt; else propose candidates.
    if should_check_prompt:
        exact_variety = find_exact_variety_in_text(prompt)
        if exact_variety:
            if draft.variety != exact_variety:
                draft = draft.model_copy(update={"variety": exact_variety})
            missing_fields = list_missing_required_fields(draft)
        else:
            prompt_candidates = retrieve_variety_candidates(prompt, limit=5)
            if prompt_candidates:
                if draft.variety:
                    draft = draft.model_copy(update={"variety": None})
                if "variety" not in missing_fields:
                    missing_fields.append("variety")
                variety_candidates = prompt_candidates
            elif draft.variety and "variety" not in experience_applied:
                # LLM-only variety without DB evidence -> clear and re-ask.
                draft = draft.model_copy(update={"variety": None})
                missing_fields = list_missing_required_fields(draft)
    if draft.variety:
        if not _is_known_variety(draft.variety):
            if not variety_candidates:
                variety_candidates = retrieve_variety_candidates(
                    draft.variety, limit=5
                )
            if "variety" not in missing_fields:
                missing_fields.append("variety")
    elif "variety" in missing_fields and not variety_candidates:
        variety_candidates = retrieve_variety_candidates(prompt, limit=5)
    # If user says "不确定/不知道", allow fallback defaults to avoid dead-ends.
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
            "experience_key": current_key,
            "experience_applied": experience_applied,
            "experience_skip_fields": sorted(experience_skip_fields),
            "experience_notice": experience_notice,
            "variety_candidates": variety_candidates,
            "future_sowing_date_warning": False,
            "pending_message": None,
        }
    )
    return state


def _ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    candidates = state.get("variety_candidates") or []
    pending_message = state.get("pending_message")
    if pending_message:
        message = pending_message
    elif "variety" in missing_fields and candidates:
        options = "\n".join(
            f"{idx + 1}. {name}" for idx, name in enumerate(candidates)
        )
        message = (
            "未找到完全匹配的品种。你是不是想查询以下品种：\n"
            f"{options}\n"
            "请回复序号或品种名称。"
        )
    else:
        message = build_crop_calendar_missing_question(
            missing_fields,
            CROP_FIELD_LABELS,
        )
        draft = state.get("planting_draft")
        warning = (
            build_future_weather_warning(draft.sowing_date) if draft else None
        )
        if warning:
            message = f"{warning}\n{message}"
    experience_notice = state.get("experience_notice")
    if experience_notice:
        message = f"{experience_notice}\n{message}"
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
    weather_query = None
    if planting.region:
        try:
            weather_query = WeatherQueryInput(
                region=planting.region,
                year=planting.sowing_date.year,
                granularity="daily",
            )
        except Exception:
            weather_query = None
    if weather_query is None and prompt:
        _, parsed = normalize_weather_prompt(prompt)
        if parsed:
            try:
                weather_query = parsed.model_copy(
                    update={
                        "year": planting.sowing_date.year,
                        "granularity": "daily",
                    }
                )
            except Exception:
                weather_query = None
    result = lookup_goso_weather(weather_query)
    if not result:
        weather_series = coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_series_ref = store_weather_series(weather_series)
        return {
            "name": "growth_weather_lookup",
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

    user_id = state.get("user_id")
    if user_id and planting.crop and planting.region:
        get_planting_choice_store().set(
            user_id, planting.crop, planting.region, planting
        )
        state = add_trace(state, "experience_stored")

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

        return wrap_with_otel_context(_inner)

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
        recommendation_context, ensure_ascii=False, default=str
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
    experience_notice = state.get("experience_notice")
    if experience_notice and experience_notice not in message:
        message = f"{experience_notice}\n{message}"

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
    if state.get("halt"):
        return END
    return "ask" if missing else "context"


def _route_after_context(state: GraphState) -> str:
    if state.get("cache_hit"):
        return END
    return "recommend"


def build_crop_calendar_graph():
    """
    Construct and return the crop calendar LangGraph workflow.
    """
    def _trace_node(node_name: str, func):
        def _inner(state: GraphState) -> GraphState:
            attrs = {"workflow.name": CROP_CACHE_NAME, "node.name": node_name}
            attrs.update(build_span_attributes("node.input", summarize_state(state)))
            with start_span(
                f"workflow.{CROP_CACHE_NAME}.{node_name}", attributes=attrs
            ) as span:
                try:
                    result = func(state)
                except Exception as exc:
                    record_exception(span, exc)
                    raise
                output_attrs = build_span_attributes(
                    "node.output", summarize_state(result)
                )
                if span:
                    for key, value in output_attrs.items():
                        try:
                            span.set_attribute(key, value)
                        except Exception:
                            pass
                return result

        return _inner

    graph = StateGraph(GraphState)
    graph.add_node("extract", _trace_node("extract", _extract_node))
    graph.add_node("ask", _trace_node("ask", _ask_node))
    graph.add_node("context", _trace_node("context", _context_node))
    graph.add_node("recommend", _trace_node("recommend", _recommend_node))

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _route_after_extract)
    graph.add_edge("ask", END)
    graph.add_conditional_edges("context", _route_after_context)
    graph.add_edge("recommend", END)
    return graph.compile()
