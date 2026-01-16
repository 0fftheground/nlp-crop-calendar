"""
LangGraph workflow for growth stage prediction.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from langgraph.graph import END, StateGraph

from ...application.services.planting_service import extract_planting_details
from ...domain.planting import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ...infra.config import get_config
from ...infra.cache_keys import build_planting_cache_key
from ...infra.planting_choice_store import (
    build_choice_key,
    get_planting_choice_store,
)
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.weather_cache import get_weather_series, store_weather_series
from ...prompts.workflow_messages import (
    GROWTH_STAGE_INTRANET_REQUIRED_MESSAGE,
    build_growth_stage_missing_question,
    format_growth_stage_message,
    format_planting_validation_error,
)
from ...prompts.tool_messages import TOOL_NOT_FOUND_MESSAGE
from ...schemas import (
    GrowthStageResult,
    PredictGrowthStageInput,
    ToolInvocation,
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
    experience_applied = list(state.get("experience_applied") or [])
    experience_skip_fields = set(state.get("experience_skip_fields") or [])
    experience_key = state.get("experience_key")
    experience_notice = state.get("experience_notice")
    change_fields = detect_experience_change_fields(prompt)
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
            "experience_key": current_key,
            "experience_applied": experience_applied,
            "experience_skip_fields": sorted(experience_skip_fields),
            "experience_notice": experience_notice,
        }
    )
    return state


def _growth_ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    message = build_growth_stage_missing_question(
        missing_fields,
        GROWTH_FIELD_LABELS,
    )
    experience_notice = state.get("experience_notice")
    if experience_notice:
        message = f"{experience_notice}\n{message}"
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

    query = planting.region or prompt
    result = execute_tool("weather_lookup", query)
    if not result:
        weather_series = coerce_weather_series(
            {}, region=planting.region or "unknown"
        )
        weather_info = {
            "name": "weather_lookup",
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

    crop = planting.crop
    variety = planting.variety or f"{crop}示例品种"
    variety_prompt = f"{crop} {variety}".strip()
    variety_result = execute_tool("variety_lookup", variety_prompt)
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
    weather_summary = summarize_weather_series(weather_series)
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
                "message": GROWTH_STAGE_INTRANET_REQUIRED_MESSAGE,
                "data": {"workflow": workflow_payload},
                "weather_info": weather_info,
                "variety_info": variety_info,
            }
        )
        return state

    state = add_trace(state, "growth_stage_intranet ok")
    result = _coerce_growth_stage_result(tool_payload.data or {})
    if result:
        message = format_growth_stage_message(
            planting,
            result.stages,
            weather_note=weather_note,
            variety_note=variety_note,
        )
    else:
        message = tool_payload.message or "已完成生育期预测。"
    experience_notice = state.get("experience_notice")
    if experience_notice and experience_notice not in message:
        message = f"{experience_notice}\n{message}"
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
    return "ask" if missing else "weather"


def _growth_route_after_weather(state: GraphState) -> str:
    if state.get("cache_hit"):
        return END
    return "predict"


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
    graph.add_conditional_edges("weather", _growth_route_after_weather)
    graph.add_edge("predict", END)
    return graph.compile()
