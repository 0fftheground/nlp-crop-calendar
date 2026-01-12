"""
LangGraph workflow for growth stage prediction.
"""

from __future__ import annotations

import json
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
from ...infra.cache_keys import build_planting_cache_key
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.weather_cache import get_weather_series, store_weather_series
from ...schemas import (
    GrowthStageResult,
    PlantingDetails,
    PredictGrowthStageInput,
    ToolInvocation,
)
from ..tools.registry import cache_tool_result, execute_tool, get_cached_tool_result
from .common import (
    apply_memory_to_draft,
    coerce_planting_draft,
    coerce_weather_series,
    build_fallback_planting,
    classify_memory_confirmation,
    format_missing_question,
    format_memory_confirmation,
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


def _growth_format_missing_question(missing_fields: List[str]) -> str:
    return format_missing_question(
        missing_fields,
        GROWTH_FIELD_LABELS,
        "生育期预测还需要补充：",
    )


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
    if not draft.region:
        missing_fields.append("region")
    missing_fields = list(dict.fromkeys(missing_fields))
    if memory_choice is not None:
        state = add_trace(state, f"memory_decision={memory_decision}")
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
            "memory_decision": memory_decision,
        }
    )
    return state


def _growth_ask_node(state: GraphState) -> GraphState:
    missing_fields = state.get("missing_fields", [])
    message = _growth_format_missing_question(missing_fields)
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
            "message": "tool not found",
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
                "message": _growth_format_missing_question(
                    list(GROWTH_FIELD_LABELS.keys())
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
