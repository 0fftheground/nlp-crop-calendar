"""
LangGraph workflow for growth stage prediction.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from langgraph.graph import END, StateGraph

from ...application.services.planting_service import extract_planting_details
from ...application.services.growth_stage_service import predict_growth_stage_local
from ...application.services.weather_service import lookup_goso_weather
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
from ...infra.weather_cache import get_weather_series, store_weather_series
from ...observability.otel import (
    build_span_attributes,
    record_exception,
    start_span,
    summarize_state,
)
from ...prompts.workflow_messages import (
    build_growth_stage_missing_question,
    build_future_weather_warning,
    format_growth_stage_message,
    format_planting_validation_error,
)
from ...prompts.tool_messages import TOOL_NOT_FOUND_MESSAGE
from ...schemas import (
    GrowthStageResult,
    PredictGrowthStageInput,
    ToolInvocation,
    WeatherQueryInput,
)
from ..tools.registry import cache_tool_result, execute_tool, get_cached_tool_result
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


GROWTH_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "region": "地区",
}
GROWTH_CACHE_PROVIDER = "workflow"
GROWTH_WORKFLOW_NAME = "growth_stage_workflow"


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


def _growth_extract_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    prior_draft = coerce_planting_draft(state.get("planting_draft"))
    prior_missing = state.get("missing_fields") or []
    followup_count = state.get("followup_count", 0)
    prior_future_warning = bool(state.get("future_sowing_date_warning"))

    def _drop_variety(fields):
        return [field for field in fields if field != "variety"]

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
                "variety_tool_query": None,
                "variety_tool_draft": None,
                "variety_tool_missing_fields": [],
                "variety_tool_followup_count": 0,
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
                "variety_tool_query": None,
                "variety_tool_draft": None,
                "variety_tool_missing_fields": [],
                "variety_tool_followup_count": 0,
            }
        )
        return state
    missing_fields = _drop_variety(list_missing_required_fields(draft))
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
        missing_fields = _drop_variety(list_missing_required_fields(draft))
    if experience_skip_fields:
        experience_skip_fields = {
            field
            for field in experience_skip_fields
            if getattr(draft, field, None) is None
        }
    experience_skip_fields.add("variety")
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
            missing_fields = _drop_variety(list_missing_required_fields(draft))
    if draft.variety is not None:
        draft = draft.model_copy(update={"variety": None})
    if not draft.region:
        missing_fields.append("region")
    missing_fields = list(dict.fromkeys(missing_fields))
    # If user says "不确定/不知道", allow fallback defaults to avoid dead-ends.
    unknown_fields = infer_unknown_fields(prompt, missing_fields, GROWTH_FIELD_LABELS)
    if unknown_fields:
        fallback = build_fallback_planting(draft)
        draft = merge_planting_answers(
            draft,
            unknown_fields=unknown_fields,
            fallback=fallback,
        )
        missing_fields = _drop_variety(list_missing_required_fields(draft))
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
        missing_fields = _drop_variety(list_missing_required_fields(draft))
    if not draft.region:
        missing_fields.append("region")
    missing_fields = list(dict.fromkeys(missing_fields))

    resolved_future_warning = (
        prior_future_warning
        and draft.sowing_date is not None
        and draft.sowing_date.year < 2026
    )
    variety_tool_query = state.get("variety_tool_query")
    variety_tool_draft = state.get("variety_tool_draft")
    variety_tool_missing_fields = list(state.get("variety_tool_missing_fields") or [])
    variety_tool_followup_count = state.get("variety_tool_followup_count", 0)
    pending_message = state.get("pending_message")
    pending_options = list(state.get("pending_options") or [])
    if resolved_future_warning:
        pending_message = None
        pending_options = []
    reset_tool_followup = False
    if change_fields and {"variety", "region", "crop"} & set(change_fields):
        reset_tool_followup = True
    if any(field in missing_fields for field in GROWTH_FIELD_LABELS):
        reset_tool_followup = True
    if reset_tool_followup:
        variety_tool_query = None
        variety_tool_draft = None
        variety_tool_missing_fields = []
        variety_tool_followup_count = 0
        pending_message = None
        pending_options = []

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
            "variety_tool_query": variety_tool_query,
            "variety_tool_draft": variety_tool_draft,
            "variety_tool_missing_fields": variety_tool_missing_fields,
            "variety_tool_followup_count": variety_tool_followup_count,
            "pending_message": pending_message,
            "pending_options": pending_options,
            "future_sowing_date_warning": False,
        }
    )
    return state


def _growth_ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    pending_message = state.get("pending_message")
    if pending_message:
        message = pending_message
    else:
        message = build_growth_stage_missing_question(
            missing_fields,
            GROWTH_FIELD_LABELS,
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


def _build_variety_tool_prompt(planting, prompt: str) -> str:
    base = f"{planting.crop} {planting.variety or ''}".strip()
    if planting.region:
        base = f"{base} 在{planting.region}种植".strip()
    cleaned = prompt.strip() if prompt else ""
    if cleaned and not cleaned.isdigit() and cleaned not in base:
        base = f"{cleaned} {base}".strip()
    return base


def _growth_variety_node(state: GraphState) -> GraphState:
    prompt = state.get("user_prompt", "")
    draft = state.get("planting_draft")
    if draft is None:
        state = add_trace(state, "variety missing draft")
        state.update(
            {
                "message": build_growth_stage_missing_question(
                    list(GROWTH_FIELD_LABELS.keys()),
                    GROWTH_FIELD_LABELS,
                )
            }
        )
        return state

    missing = [
        field
        for field in list_missing_required_fields(draft)
        if field != "variety"
    ]
    if not draft.region and "region" not in missing:
        missing = list(missing) + ["region"]
    missing = list(dict.fromkeys(missing))
    if missing:
        state = add_trace(state, f"variety missing={missing}")
        state.update(
            {
                "missing_fields": missing,
                "message": build_growth_stage_missing_question(
                    missing,
                    GROWTH_FIELD_LABELS,
                ),
            }
        )
        return state
    try:
        planting = draft.to_canonical()
    except ValueError as exc:
        state = add_trace(state, f"variety invalid={exc}")
        state.update({"message": format_planting_validation_error(exc)})
        return state

    user_id = state.get("user_id")
    variety_tool_query = state.get("variety_tool_query")
    variety_tool_draft = state.get("variety_tool_draft")
    variety_tool_missing_fields = list(state.get("variety_tool_missing_fields") or [])
    variety_tool_followup_count = state.get("variety_tool_followup_count", 0)
    planting_payload = planting.model_dump(exclude_none=True, mode="json")
    if variety_tool_draft and variety_tool_query:
        followup_payload = {
            "user_id": user_id,
            "query": variety_tool_query,
            "planting": planting_payload,
            "followup": {
                "prompt": prompt,
                "draft": variety_tool_draft,
                "missing_fields": variety_tool_missing_fields,
                "followup_count": variety_tool_followup_count,
            },
        }
        tool_input = json.dumps(
            followup_payload, ensure_ascii=False, default=str
        )
    else:
        variety_prompt = _build_variety_tool_prompt(planting, prompt)
        tool_input = json.dumps(
            {
                "prompt": variety_prompt,
                "user_id": user_id,
                "planting": planting_payload,
            },
            ensure_ascii=False,
            default=str,
        )
    variety_result = execute_tool("variety_lookup", tool_input)
    if variety_result:
        variety_info = variety_result.model_dump(mode="json")
        variety_data = variety_result.data
        state = add_trace(state, "variety_lookup ok")
    else:
        variety_info = {
            "name": "variety_lookup",
            "message": TOOL_NOT_FOUND_MESSAGE,
            "data": {},
        }
        variety_data = {}
        state = add_trace(state, "variety_lookup missing")

    missing_fields = list(variety_data.get("missing_fields") or []) if variety_data else []
    if missing_fields:
        pending_options = (
            variety_data.get("region_candidates")
            or variety_data.get("record_candidates")
            or variety_data.get("candidates")
            or variety_data.get("options")
            or []
        )
        state = add_trace(state, "variety_lookup needs_followup")
        state.update(
            {
                "missing_fields": missing_fields,
                "pending_options": pending_options,
                "pending_message": variety_info.get("message") or "",
                "message": variety_info.get("message") or "",
                "variety_info": variety_info,
                "variety_record": None,
                "variety_tool_query": variety_data.get("query") or variety_tool_query,
                "variety_tool_draft": variety_data.get("draft") or {},
                "variety_tool_missing_fields": missing_fields,
                "variety_tool_followup_count": variety_data.get(
                    "followup_count", variety_tool_followup_count
                ),
            }
        )
        return state

    selected_variety = variety_data.get("variety") if variety_data else None
    if selected_variety and draft.variety != selected_variety:
        draft = draft.model_copy(update={"variety": selected_variety})
    variety_record = variety_data.get("raw_selected") if variety_data else None
    state.update(
        {
            "planting_draft": draft,
            "variety_info": variety_info,
            "variety_record": variety_record,
            "pending_message": None,
            "pending_options": [],
            "variety_tool_query": None,
            "variety_tool_draft": None,
            "variety_tool_missing_fields": [],
            "variety_tool_followup_count": 0,
        }
    )
    return state


def _growth_weather_node(state: GraphState) -> GraphState:
    draft = state.get("planting_draft")
    if draft is None:
        state = add_trace(state, "weather missing draft")
        state.update(
            {
                "message": build_growth_stage_missing_question(
                    list(GROWTH_FIELD_LABELS.keys()),
                    GROWTH_FIELD_LABELS,
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
            {
                "missing_fields": missing,
                "message": build_growth_stage_missing_question(
                    missing,
                    GROWTH_FIELD_LABELS,
                ),
            }
        )
        return state
    except ValueError as exc:
        state = add_trace(state, f"weather invalid={exc}")
        state.update({"message": format_planting_validation_error(exc)})
        return state

    user_id = state.get("user_id")
    if user_id and planting.crop and planting.region:
        get_planting_choice_store().set(
            user_id, planting.crop, planting.region, planting
        )
        state = add_trace(state, "experience_stored")

    cache_key = build_planting_cache_key(planting)
    cached = get_cached_tool_result(
        "growth_stage_prediction",
        GROWTH_CACHE_PROVIDER,
        cache_key,
    )
    if cached:
        state = add_trace(state, "growth_stage_cache_hit")
        cached_growth = _coerce_growth_stage_result(
            cached.data.get("growth_stage", {}) if cached.data else {}
        )
        state.update(
            {
                "planting": planting,
                "message": cached.message,
                "data": cached.data,
                "growth_stage": cached_growth,
                "cache_hit": True,
            }
        )
        return state

    try:
        weather_query = WeatherQueryInput(
            region=planting.region or "unknown",
            year=planting.sowing_date.year,
            granularity="daily",
        )
    except Exception:
        weather_query = None
    result = lookup_goso_weather(weather_query)
    if not result:
        weather_series = coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_info = {
            "name": "growth_weather_lookup",
            "message": TOOL_NOT_FOUND_MESSAGE,
            "data": {},
        }
    else:
        weather_series = coerce_weather_series(
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
        "summary": summarize_weather_series(weather_series),
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
                "message": build_growth_stage_missing_question(
                    list(GROWTH_FIELD_LABELS.keys()),
                    GROWTH_FIELD_LABELS,
                )
            }
        )
        return state
    weather_series = get_weather_series(weather_series_ref)
    if weather_series is None:
        weather_series = coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
    variety_info = state.get("variety_info") or {}
    if not isinstance(variety_info, dict):
        variety_info = {}
    variety_data = variety_info.get("data") or {}
    variety_record = state.get("variety_record") or variety_data.get("raw_selected")
    request = PredictGrowthStageInput(
        planting=planting,
        weatherSeries=weather_series,
        variety_record=variety_record,
    )
    weather_note = weather_info.get("message") or ""
    variety_note = ""
    if variety_data and not variety_data.get("missing_fields"):
        variety_note = variety_info.get("message") or ""
    weather_summary = summarize_weather_series(weather_series)
    workflow_payload = {
        "inputs": {
            "planting": planting.model_dump(exclude_none=True, mode="json"),
        },
        "variety": variety_data,
        "weather_summary": weather_summary,
    }

    result = None
    provider_message = ""
    provider_response: Dict[str, object] = {}
    try:
        result = predict_growth_stage_local(request)
        provider_message = "local growth stage prediction"
        provider_response = result.model_dump(mode="json")
        state = add_trace(state, "growth_stage_local ok")
    except Exception as exc:
        state = add_trace(state, f"growth_stage_local failed={exc}")
        trace = list(state.get("trace") or [])
        workflow_payload["trace"] = trace
        state.update(
            {
                "message": f"生育期预测失败: {exc}",
                "data": {"workflow": workflow_payload},
                "weather_info": weather_info,
                "variety_info": variety_info,
            }
        )
        return state

    if result:
        message = format_growth_stage_message(
            planting,
            result.stages,
            weather_note=weather_note,
            variety_note=variety_note,
        )
    else:
        message = provider_message or "已完成生育期预测。"
    experience_notice = state.get("experience_notice")
    if experience_notice and experience_notice not in message:
        message = f"{experience_notice}\n{message}"
    state = add_trace(state, "predict complete")
    trace = list(state.get("trace") or [])
    workflow_payload.update(
        {
            "trace": trace,
            "provider_message": provider_message,
            "provider_response": provider_response,
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
    if result:
        cache_key = build_planting_cache_key(planting)
        cache_tool_result(
            "growth_stage_prediction",
            GROWTH_CACHE_PROVIDER,
            cache_key,
            ToolInvocation(
                name="growth_stage_prediction",
                message=message,
                data=data_payload,
            ),
        )
        state = add_trace(state, "growth_stage_cached")
    return state


def _growth_route_after_extract(state: GraphState) -> str:
    missing = state.get("missing_fields") or []
    if state.get("halt"):
        return END
    return "ask" if missing else "variety"


def _growth_route_after_variety(state: GraphState) -> str:
    missing = state.get("missing_fields") or []
    if state.get("halt"):
        return END
    return "ask" if missing else "weather"


def _growth_route_after_weather(state: GraphState) -> str:
    if state.get("cache_hit"):
        return END
    return "predict"


def build_growth_stage_graph():
    """
    Construct and return the growth stage prediction LangGraph workflow.
    """
    def _trace_node(node_name: str, func):
        def _inner(state: GraphState) -> GraphState:
            attrs = {"workflow.name": GROWTH_WORKFLOW_NAME, "node.name": node_name}
            attrs.update(build_span_attributes("node.input", summarize_state(state)))
            with start_span(
                f"workflow.{GROWTH_WORKFLOW_NAME}.{node_name}", attributes=attrs
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
    graph.add_node("extract", _trace_node("extract", _growth_extract_node))
    graph.add_node("ask", _trace_node("ask", _growth_ask_node))
    graph.add_node("variety", _trace_node("variety", _growth_variety_node))
    graph.add_node("weather", _trace_node("weather", _growth_weather_node))
    graph.add_node("predict", _trace_node("predict", _growth_predict_node))

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _growth_route_after_extract)
    graph.add_conditional_edges("variety", _growth_route_after_variety)
    graph.add_edge("ask", END)
    graph.add_conditional_edges("weather", _growth_route_after_weather)
    graph.add_edge("predict", END)
    return graph.compile()
