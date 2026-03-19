"""
LangGraph workflow for growth stage query.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from langgraph.graph import END, StateGraph

from ...application.services.planting_service import extract_planting_details
from ...application.services.growth_stage_service import (
    build_planting_from_plan_row,
    extract_plan_name_from_row,
    list_active_planting_plans,
    query_growth_stage_from_plan_id,
    resolve_planting_from_plan_id,
    search_planting_plans,
)
from ...domain.planting import merge_planting_answers
from ...infra.variety_store import find_exact_variety_in_text
from ...observability.otel import (
    build_span_attributes,
    record_exception,
    start_span,
    summarize_state,
)
from ...messages.workflow_messages import (
    GROWTH_STAGE_MISSING_PREFIX,
    GROWTH_STAGE_ORDER,
    format_growth_stage_message,
)
from ..followup import (
    build_workflow_followup_update,
    clear_workflow_followup_update,
    get_followup_draft,
    get_followup_message,
    get_followup_missing_fields,
    get_followup_options,
    render_followup_message,
    resolve_followup_choice,
)
from .common import (
    coerce_planting_draft,
    llm_extract_planting,
)
from .state import GraphState, add_trace


GROWTH_WORKFLOW_NAME = "growth_stage_query_workflow"
_PLAN_ID_RE = re.compile(
    r"(?:plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']+)[\"“”']")


def _parse_prompt_payload(prompt: str) -> dict | None:
    if not prompt:
        return None
    text = prompt.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_plan_id(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)
    return None


def _extract_plan_id_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    if text.strip().isdigit():
        return text.strip()
    match = _PLAN_ID_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_plan_id_from_payload(payload: dict | None) -> Optional[str]:
    if not payload:
        return None
    for key in ("plan_id", "planting_plan_id", "id", "planId"):
        value = payload.get(key)
        if value is None:
            continue
        normalized = _normalize_plan_id(value)
        if normalized:
            return normalized
    return None


def _extract_plan_name_from_payload(payload: dict | None) -> str:
    if not payload:
        return ""
    for key in ("plan_name", "planName", "plan", "name", "种植计划名称", "计划名称"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_plan_name_from_text(text: str) -> str:
    if not text:
        return ""
    for quoted in _QUOTED_RE.findall(text):
        if "计划" in text:
            return quoted.strip()
    match = re.search(r"(?:种植)?计划(?:名称|名)?[:：]?\s*([^\s，。]+)", text)
    if match:
        return match.group(1).strip()
    if "计划" in text and len(text.strip()) <= 20:
        cleaned = re.sub(r"^(?:查询|查|看|获取)?", "", text)
        return cleaned.strip()
    return ""


def _extract_variety_from_payload(payload: dict | None) -> str:
    if not payload:
        return ""
    for key in ("variety", "variety_name", "品种", "品种名称"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_plan_choice(
    prompt: str, options: list[str]
) -> Optional[str]:
    return resolve_followup_choice(prompt, options)


def _parse_plan_id_from_option(option: str) -> Optional[str]:
    if not option:
        return None
    match = _PLAN_ID_RE.search(option)
    if match:
        return match.group(1).strip()
    if option.strip().isdigit():
        return option.strip()
    return None


def _format_plan_option_text(
    row: dict, columns: list[str], id_col: str
) -> str:
    plan_id = row.get(id_col)
    plan_name = extract_plan_name_from_row(row, columns)
    planting = build_planting_from_plan_row(row, columns)
    parts = []
    if plan_id is not None:
        parts.append(f"id={plan_id}")
    if planting:
        if planting.variety:
            parts.append(f"品种={planting.variety}")
        if planting.sowing_date:
            parts.append(f"播种={planting.sowing_date.isoformat()}")
        method = planting.planting_method
        method_value = (
            method.value if hasattr(method, "value") else str(method)
        )
        if method_value:
            label = "直播" if method_value == "direct_seeding" else "插秧" if method_value == "transplanting" else method_value
            parts.append(f"方式={label}")
        culti_type = getattr(planting, "culti_type", None)
        if culti_type:
            parts.append(f"类型={culti_type}")
    title = plan_name or "种植计划"
    if parts:
        return f"{title}（{'，'.join(parts)}）"
    if plan_id is not None:
        return f"{title}（id={plan_id}）"
    return title


def _format_stage_only_message(stages: Dict[str, str]) -> str:
    lines = ["已获取生育期预测结果。"]
    stage_dates = stages.get("stage_dates")
    if stage_dates:
        try:
            payload = json.loads(stage_dates)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload:
            ordered = []
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
            if ordered:
                lines.append("生育期阶段日期:")
                for name, value in ordered:
                    lines.append(f"{name}: {value}")
    return "\n".join(lines)


def _growth_extract_node(state: GraphState) -> GraphState:
    """解析用户输入，优先锁定 plan_id；否则按条件检索种植计划并决定是否追问。"""
    prompt = state.get("user_prompt", "")
    prior_draft = coerce_planting_draft(get_followup_draft(state))
    prior_missing = get_followup_missing_fields(state)
    followup_count = state.get("followup_count", 0)
    options = get_followup_options(state)

    # 处理多计划追问场景：把用户回复的序号/名称映射回具体 plan_id。
    if "plan_choice" in prior_missing and options:
        choice = _resolve_plan_choice(prompt, options)
        plan_id = _parse_plan_id_from_option(choice or "") if choice else None
        if not plan_id:
            plan_id = _extract_plan_id_from_text(prompt)
        if plan_id:
            state = add_trace(state, f"plan_choice resolved={plan_id}")
            state.update(
                clear_workflow_followup_update(extra={"plan_id": plan_id})
            )
            return state
        followup_count += 1
        state = add_trace(state, "plan_choice unresolved")
        state.update(
            build_workflow_followup_update(
                draft=prior_draft,
                missing_fields=["plan_choice"],
                followup_count=followup_count,
                pending_message="未识别到有效的序号/计划，请回复序号或计划名称。",
                options=options,
            )
        )
        return state

    # 第一优先级是直接从 prompt/payload 中拿到 plan_id，命中后直接进入查询节点。
    payload = _parse_prompt_payload(prompt)
    plan_id = _extract_plan_id_from_payload(payload) or _extract_plan_id_from_text(
        prompt
    )
    if plan_id:
        state = add_trace(state, f"plan_id from prompt={plan_id}")
        state.update(clear_workflow_followup_update(extra={"plan_id": plan_id}))
        return state

    # 没有 plan_id 时退回到种植信息抽取，用品种/日期/方式等条件查计划。
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

    plan_name = _extract_plan_name_from_payload(payload) or _extract_plan_name_from_text(
        prompt
    )
    variety = _extract_variety_from_payload(payload)
    if not variety:
        try:
            variety = find_exact_variety_in_text(prompt) or ""
        except Exception:
            variety = ""
    filters: Dict[str, object] = {}
    if plan_name:
        filters["plan_name"] = plan_name
    if variety:
        filters["variety"] = variety
    if draft and draft.sowing_date:
        filters["sowing_date"] = draft.sowing_date
    if draft and draft.transplant_date:
        filters["transplant_date"] = draft.transplant_date
    if draft and draft.planting_method:
        filters["planting_method"] = draft.planting_method
    if draft and getattr(draft, "culti_type", None):
        filters["culti_type"] = getattr(draft, "culti_type")

    if not filters:
        state = add_trace(state, "missing plan query")
        state.update(
            build_workflow_followup_update(
                draft=draft,
                missing_fields=["plan_query"],
                followup_count=followup_count,
                pending_message="请提供品种名称或种植计划名称，以便查询种植计划。",
                options=[],
            )
        )
        return state

    # 根据抽取出的过滤条件查计划：0 条则追问，1 条直达，多条让用户选。
    try:
        rows, id_col, columns = search_planting_plans(filters, limit=5)
    except Exception as exc:
        state = add_trace(state, f"plan_search failed={exc}")
        state.update(
            build_workflow_followup_update(
                draft=draft,
                missing_fields=["plan_query"],
                followup_count=followup_count,
                pending_message=f"查询种植计划失败: {exc}。请提供更具体的品种或计划名称。",
                options=[],
            )
        )
        return state

    if not rows:
        if "plan_name" in filters:
            try:
                active_rows, active_id_col, active_columns = (
                    list_active_planting_plans(limit=5)
                )
            except Exception:
                active_rows, active_id_col, active_columns = ([], "", [])
            if active_rows:
                options = [
                    _format_plan_option_text(
                        row, active_columns, active_id_col
                    )
                    for row in active_rows
                ]
                message_lines = [
                    "未找到与计划名称匹配的记录，以下是当前启用的种植计划（仅展示前5条），请回复序号："
                ]
                for idx, option in enumerate(options, start=1):
                    message_lines.append(f"{idx}. {option}")
                state = add_trace(state, "plan_search fallback active")
                state.update(
                    build_workflow_followup_update(
                        draft=draft,
                        missing_fields=["plan_choice"],
                        followup_count=followup_count + 1,
                        pending_message="\n".join(message_lines),
                        options=options,
                        extra={"plan_filters": filters},
                    )
                )
                return state
        state = add_trace(state, "plan_search empty")
        state.update(
            build_workflow_followup_update(
                draft=draft,
                missing_fields=["plan_query"],
                followup_count=followup_count,
                pending_message="未找到符合条件的种植计划，请提供更具体的品种或计划名称。",
                options=[],
                extra={"plan_filters": filters},
            )
        )
        return state

    if len(rows) == 1:
        plan_id = rows[0].get(id_col)
        state = add_trace(state, f"plan_search single={plan_id}")
        state.update(
            clear_workflow_followup_update(
                extra={"plan_id": plan_id, "plan_filters": filters}
            )
        )
        return state

    options = [
        _format_plan_option_text(row, columns, id_col) for row in rows
    ]
    message_lines = ["找到多个种植计划，请回复序号："]
    for idx, option in enumerate(options, start=1):
        message_lines.append(f"{idx}. {option}")
    state = add_trace(state, f"plan_search multi={len(options)}")
    state.update(
        build_workflow_followup_update(
            draft=draft,
            missing_fields=["plan_choice"],
            followup_count=followup_count + 1,
            pending_message="\n".join(message_lines),
            options=options,
            extra={"plan_filters": filters},
        )
    )
    return state


def _growth_ask_node(state: GraphState) -> GraphState:
    """根据缺失信息生成下一轮提问，主要覆盖计划选择或查询条件补充。"""
    missing_fields = get_followup_missing_fields(state)
    options = get_followup_options(state)
    pending_message = get_followup_message(state)
    if (
        "plan_choice" in missing_fields
        and not pending_message
        and not options
    ):
        message = "请回复序号选择对应的种植计划。"
    else:
        message = render_followup_message(
            pending_message=pending_message,
            missing_fields=missing_fields,
            field_labels={
                "plan_choice": "种植计划",
                "plan_query": "品种名称或种植计划名称",
                "plan_id": "计划 ID",
            },
            default_prefix=GROWTH_STAGE_MISSING_PREFIX,
            options=options if "plan_choice" in missing_fields else [],
            options_intro="找到多个种植计划，请回复序号：",
            reply_hint=(
                "请回复序号或计划名称。"
                if "plan_choice" in missing_fields
                else None
            ),
        )
    state = add_trace(state, f"ask missing={missing_fields}")
    state.update({"message": message})
    return state


def _growth_predict_node(state: GraphState) -> GraphState:
    """使用 plan_id 查询生育期结果，并拼装最终展示消息与调试数据。"""
    plan_id = state.get("plan_id")
    workflow_payload = {
        "plan_id": plan_id,
        "plan_filters": state.get("plan_filters") or {},
    }
    if isinstance(plan_id, str) and plan_id and not plan_id.isdigit():
        state = add_trace(state, f"predict invalid plan_id={plan_id}")
        state.update(
            {
                "message": "计划 ID 需要是数字，请提供有效的计划 ID。",
                "missing_fields": ["plan_id"],
                "data": {"workflow": workflow_payload},
            }
        )
        return state
    if not plan_id:
        state = add_trace(state, "predict missing plan_id")
        state.update(
            {
                "message": "请先提供品种名称或种植计划名称，以便查询种植计划。",
                "data": {"workflow": workflow_payload},
            }
        )
        return state

    result = None
    provider_response: Dict[str, object] = {}
    # 先查生育期结果；成功后再补查计划详情，仅用于生成更友好的回复文案。
    try:
        result = query_growth_stage_from_plan_id(plan_id)
        provider_response = result.model_dump(mode="json")
        state = add_trace(state, "growth_stage_api ok")
    except Exception as exc:
        state = add_trace(state, f"growth_stage_api failed={exc}")
        trace = list(state.get("trace") or [])
        workflow_payload["trace"] = trace
        state.update(
            {
                "message": f"生育期结果查询失败: {exc}",
                "data": {"workflow": workflow_payload},
            }
        )
        return state

    try:
        planting_for_message = resolve_planting_from_plan_id(plan_id)
    except Exception as exc:
        state = add_trace(state, f"resolve_plan_failed={exc}")
        planting_for_message = None
    if result and planting_for_message:
        message = format_growth_stage_message(
            planting_for_message,
            result.stages,
        )
    else:
        message = _format_stage_only_message(result.stages if result else {})
    state = add_trace(state, "predict complete")
    trace = list(state.get("trace") or [])
    workflow_payload.update(
        {
            "trace": trace,
            "provider_response": provider_response,
        }
    )
    data_payload = {"workflow": workflow_payload}
    if result:
        data_payload["growth_stage"] = result.model_dump(mode="json")
        if planting_for_message:
            data_payload["planting"] = planting_for_message.model_dump(
                mode="json"
            )
    state.update(
        {
            "growth_stage": result,
            "message": message,
            "data": data_payload,
        }
    )
    return state


def _growth_route_after_extract(state: GraphState) -> str:
    """extract 后的路由：有缺口去 ask，信息齐全去 predict。"""
    missing = get_followup_missing_fields(state)
    if state.get("halt"):
        return END
    return "ask" if missing else "predict"


def build_growth_stage_graph():
    """
    Construct and return the growth stage query LangGraph workflow.

    Flow: extract -> ask/predict -> END.
    """
    def _trace_node(node_name: str, func):
        """为每个节点统一包一层 tracing，便于排查节点输入输出。"""
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
    graph.add_node("predict", _trace_node("predict", _growth_predict_node))

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", _growth_route_after_extract)
    graph.add_edge("ask", END)
    graph.add_edge("predict", END)
    return graph.compile()
