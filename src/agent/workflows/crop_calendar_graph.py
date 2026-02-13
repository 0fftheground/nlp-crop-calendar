"""
LangGraph workflow for crop calendar planning.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from ...application.services.crop_calendar_service import (
    list_code_names,
    request_crop_calendar_plan,
    resolve_culti_type_code,
    resolve_sowing_method_code,
    set_crop_calendar_active,
)
from ...application.services.planting_service import extract_planting_details
from ...domain.planting import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ...infra.cache_keys import build_planting_cache_key
from ...infra.tool_cache import get_tool_result_cache
from ...infra.variety_store import (
    find_exact_variety_in_text,
    load_variety_names,
    retrieve_variety_candidates,
)
from ...observability.otel import (
    build_span_attributes,
    record_exception,
    start_span,
    summarize_state,
)
from ...schemas import (
    OperationPlanResult,
    PlantingDetails,
    Recommendation,
    WorkflowResponse,
)
from ...prompts.workflow_messages import (
    GROWTH_STAGE_ORDER,
    build_crop_calendar_missing_question,
    format_crop_calendar_plan_message,
)
from .common import (
    coerce_planting_draft,
    build_fallback_planting,
    infer_unknown_fields,
    llm_extract_planting,
)
from .state import GraphState, add_trace


CROP_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "transplant_date": "移栽日期",
    "culti_type": "稻作类型",
}
CROP_CACHE_NAME = "crop_calendar_workflow"
CROP_CACHE_PROVIDER = "workflow"
_FOLLOWUP_INDEX_RE = re.compile(r"^第?\s*(\d+)\s*(?:个|条|项)?$")
_YES_WORDS = {"是", "保存", "需要", "要", "确认", "好的", "好", "ok", "yes", "y"}
_NO_WORDS = {
    "否",
    "不",
    "不用",
    "不需要",
    "不保存",
    "取消",
    "算了",
    "no",
    "n",
}


def _optional_fields_for_prompt(state: GraphState) -> list[str]:
    draft = coerce_planting_draft(state.get("planting_draft"))
    prompt = state.get("user_prompt", "") or ""
    optional: list[str] = []
    if not (draft and getattr(draft, "culti_type", None)):
        optional.append("culti_type")
    if _requires_transplant_date(draft, prompt) and not (
        draft and draft.transplant_date
    ):
        optional.append("transplant_date")
    return optional


def _requires_transplant_date(draft: Optional[PlantingDetailsDraft], prompt: str) -> bool:
    if draft:
        method = getattr(draft, "planting_method", None)
        if method:
            value = method.value if hasattr(method, "value") else str(method)
            if value in {"transplanting", "插秧", "移栽", "机插", "抛秧"}:
                return True
    text = prompt or ""
    return any(token in text for token in ("插秧", "移栽", "机插", "抛秧"))


def _build_missing_question(
    state: GraphState, missing_fields: list[str]
) -> str:
    return build_crop_calendar_missing_question(
        missing_fields,
        CROP_FIELD_LABELS,
        optional_fields=_optional_fields_for_prompt(state),
    )


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


def _build_recommendations_from_plan(
    plan: OperationPlanResult, planting: PlantingDetails
) -> List[Recommendation]:
    crop = plan.crop or planting.crop
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
                regions=[],
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
    plant_season_id = state.get("plant_season_id")
    pending_options = list(state.get("pending_options") or [])
    followup_count = state.get("followup_count", 0)
    if "save_confirmation" in prior_missing:
        decision = _parse_save_confirmation(prompt)
        if decision is None:
            state = add_trace(state, "save confirm missing")
            state.update(
                {
                    "planting_draft": prior_draft,
                    "missing_fields": ["save_confirmation"],
                    "followup_count": followup_count + 1,
                    "pending_message": "是否保存该方案？请回复“是/否”。",
                }
            )
            return state
        if not decision:
            state = add_trace(state, "save cancelled")
            state.update(
                {
                    "missing_fields": [],
                    "message": "已取消保存。",
                    "halt": True,
                }
            )
            return state
        if plant_season_id is None:
            state = add_trace(state, "save missing plan_id")
            state.update(
                {
                    "missing_fields": [],
                    "message": "缺少 plant_season_id，无法保存。",
                    "halt": True,
                }
            )
            return state
        try:
            result = set_crop_calendar_active(plant_season_id, is_active=True)
            state = add_trace(state, "save ok")
            state.update(
                {
                    "missing_fields": [],
                    "message": "已保存种植计划。",
                    "data": {"save_response": result},
                    "halt": True,
                }
            )
            return state
        except Exception as exc:
            state = add_trace(state, f"save failed={exc}")
            state.update(
                {
                    "missing_fields": [],
                    "message": f"保存失败: {exc}",
                    "halt": True,
                }
            )
            return state
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
    missing_fields = list_missing_required_fields(draft)
    is_followup = bool(prior_draft and prior_missing)
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
            elif draft.variety:
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
    invalid_messages: List[str] = []
    if draft.planting_method:
        method_code = resolve_sowing_method_code(draft.planting_method)
        if method_code is None:
            options = list_code_names("sowingmtd", limit=6)
            hint = "、".join(options) if options else "直播/插秧"
            invalid_messages.append(
                f"播种方式需匹配字典表，请确认（{hint}）。"
            )
            draft = draft.model_copy(update={"planting_method": None})
            if "planting_method" not in missing_fields:
                missing_fields.append("planting_method")
    if getattr(draft, "culti_type", None):
        culti_code = resolve_culti_type_code(draft.culti_type)
        if culti_code is None:
            options = list_code_names("culti_type", limit=6)
            hint = "、".join(options) if options else "早稻/中稻/晚稻/双季晚稻"
            invalid_messages.append(
                f"稻作类型需匹配字典表，请确认（{hint}）。"
            )
            draft = draft.model_copy(update={"culti_type": None})
            if "culti_type" not in missing_fields:
                missing_fields.append("culti_type")
    if invalid_messages:
        state = add_trace(state, "extract code_dict_mismatch")
        state.update(
            {
                "planting_draft": draft,
                "missing_fields": missing_fields,
                "followup_count": followup_count,
                "pending_message": "\n".join(invalid_messages),
                "pending_options": [],
            }
        )
        return state
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
            "variety_candidates": variety_candidates,
            "pending_message": None,
        }
    )
    return state


def _parse_save_confirmation(prompt: str) -> Optional[bool]:
    text = (prompt or "").strip().lower()
    if not text:
        return None
    for token in _NO_WORDS:
        if token in text:
            return False
    for token in _YES_WORDS:
        if token in text:
            return True
    return None


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
        message = _build_missing_question(state, missing_fields)
    state = add_trace(state, f"ask missing={missing_fields}")
    state.update({"message": message})
    return state


def _context_node(state: GraphState) -> GraphState:
    raw_draft = state.get("planting_draft")
    draft = coerce_planting_draft(raw_draft)
    if draft is None:
        state = add_trace(state, "context missing draft")
        state.update(
            {
                "message": _build_missing_question(
                    state, list(CROP_FIELD_LABELS.keys())
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
                "message": _build_missing_question(state, missing),
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

    state = add_trace(state, "context ready")
    state.update(
        {
            "planting": planting,
            "assumptions": list(draft.assumptions),
            "weather_info": {},
            "variety_info": {},
        }
    )
    return state


def _recommend_node(state: GraphState) -> GraphState:
    planting = state.get("planting")
    weather_info = state.get("weather_info") or {}
    variety_info = state.get("variety_info") or {}
    assumptions = state.get("assumptions", [])
    if planting is None:
        state = add_trace(state, "recommend missing planting")
        state.update(
            {
                "message": _build_missing_question(
                    state, list(CROP_FIELD_LABELS.keys())
                )
            }
        )
        return state

    try:
        plan_result = request_crop_calendar_plan(planting)
    except Exception as exc:
        state = add_trace(state, f"recommend failed={exc}")
        state.update(
            {
                "message": f"农事方案生成失败: {exc}",
            }
        )
        return state
    plan = plan_result["operation_plan"]
    growth_stage = plan_result["growth_stage"]
    plant_season_id = plan_result.get("plant_season_id")
    raw_payload = plan_result.get("raw") or {}
    raw_data = raw_payload.get("data") if isinstance(raw_payload, dict) else {}
    farmworks_payload = (
        raw_data.get("farmworks") if isinstance(raw_data, dict) else {}
    )
    growth_stages_payload = (
        raw_data.get("growth_stages") if isinstance(raw_data, dict) else {}
    )
    recommendation_info = {
        "name": "crop_calendar_plan",
        "message": plan.summary or "已生成农事方案。",
        "data": plan.model_dump(mode="json"),
    }
    weather_note = weather_info.get("message") or ""
    variety_note = variety_info.get("message") or ""
    recommendation_note = recommendation_info.get("message") or ""
    recommendations = _build_recommendations_from_plan(plan, planting)
    message = format_crop_calendar_plan_message(
        planting,
        recommendations,
        assumptions,
        weather_note=weather_note,
        variety_note=variety_note,
        recommendation_note=recommendation_note,
    )
    growth_lines = _format_growth_stage_lines(growth_stage.stages)
    if growth_lines:
        message = f"{message}\n{growth_lines}"
    if plant_season_id is not None:
        message = f"{message}\n是否保存该方案？请回复“是/否”。"
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
            "growth_stage": growth_stage,
            "plant_season_id": plant_season_id,
            "message": message,
            "data": {
                "plant_season_id": plant_season_id,
                "farmworks": farmworks_payload,
                "growth_stages": growth_stages_payload,
            },
            "weather_info": weather_info,
            "variety_info": variety_info,
            "recommendation_info": recommendation_info,
            "missing_fields": (
                ["save_confirmation"] if plant_season_id is not None else []
            ),
            "pending_message": (
                "是否保存该方案？请回复“是/否”。"
                if plant_season_id is not None
                else None
            ),
        }
    )
    return state


def _format_growth_stage_lines(stages: Dict[str, str]) -> str:
    if not stages:
        return ""
    stage_dates = stages.get("stage_dates")
    if not stage_dates:
        return ""
    try:
        payload = json.loads(stage_dates)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict) or not payload:
        return ""
    ordered: List[tuple[str, str]] = []
    seen = set()
    for name in GROWTH_STAGE_ORDER:
        value = payload.get(name)
        if isinstance(value, str) and value:
            ordered.append((name, value))
            seen.add(name)
    for name, value in payload.items():
        if name in seen:
            continue
        if isinstance(value, str) and value:
            ordered.append((name, value))
    if not ordered:
        return ""
    lines = ["", "【生育期预测结果】"]
    for name, value in ordered:
        lines.append(f"{name}：{value}")
    return "\n".join(lines)


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
