from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ...application.services.crop_calendar_service import (
    delete_crop_calendar_plan,
)
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
from ...messages.workflow_messages import GROWTH_STAGE_ORDER, format_growth_stage_message
from ..followup import build_tool_followup_invocation
from ..followup import get_followup_count, get_followup_draft, get_followup_missing_fields
from ..followup import get_followup_options, resolve_followup_choice
from ..workflows.common import llm_extract_planting
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


_PLAN_ID_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']+)[\"“”']") 


def _parse_payload(prompt: str) -> Optional[Dict[str, Any]]:
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


def _extract_plan_id(prompt: str) -> Optional[str]:
    """优先从 JSON payload 取计划 ID，失败后再回退到文本正则提取。"""
    if not prompt:
        return None
    payload = _parse_payload(prompt)
    if payload:
        for key in ("plant_season_id", "plan_id", "id", "planId"):
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    text = prompt.strip()
    if text.isdigit():
        return text
    match = _PLAN_ID_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


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


def _extract_query_text(prompt: str) -> str:
    payload = _parse_payload(prompt)
    if not payload:
        return str(prompt or "").strip()
    for key in ("query", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(prompt or "").strip()


def _extract_followup_payload(prompt: str) -> Optional[Dict[str, Any]]:
    payload = _parse_payload(prompt)
    followup = payload.get("followup") if isinstance(payload, dict) else None
    return dict(followup) if isinstance(followup, dict) else None


def _extract_followup_prompt(prompt: str) -> str:
    followup = _extract_followup_payload(prompt)
    if isinstance(followup, dict):
        value = followup.get("prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_query_text(prompt)


def _extract_plan_name(prompt: str) -> str:
    payload = _parse_payload(prompt)
    if isinstance(payload, dict):
        for key in ("plan_name", "planName", "plan", "name", "种植计划名称", "计划名称"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = _extract_query_text(prompt)
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


def _format_plan_option_text(row: dict, columns: list[str], id_col: str) -> str:
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
        method_value = method.value if hasattr(method, "value") else str(method)
        if method_value:
            label = (
                "直播"
                if method_value == "direct_seeding"
                else "插秧" if method_value == "transplanting" else method_value
            )
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


def _parse_plan_id_from_option(option: str) -> Optional[str]:
    if not option:
        return None
    match = _PLAN_ID_RE.search(option)
    if match:
        return match.group(1).strip()
    if option.strip().isdigit():
        return option.strip()
    return None


def _format_stage_only_message(stages: Dict[str, str]) -> str:
    lines = ["已获取生育期预测结果。"]
    stage_dates = stages.get("stage_dates")
    if not stage_dates:
        return "\n".join(lines)
    try:
        payload = json.loads(stage_dates)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict) or not payload:
        return "\n".join(lines)
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


def _resolve_growth_stage_query(prompt: str) -> ToolInvocation:
    query_text = _extract_query_text(prompt)
    current_prompt = _extract_followup_prompt(prompt)
    followup = _extract_followup_payload(prompt) or {}
    prior_draft = get_followup_draft(followup)
    prior_missing = get_followup_missing_fields(followup)
    followup_count = get_followup_count(followup)
    options = get_followup_options(followup)

    if "plan_choice" in prior_missing and options:
        choice = resolve_followup_choice(current_prompt, options)
        plan_id = _parse_plan_id_from_option(choice or "") if choice else None
        if not plan_id:
            plan_id = _extract_plan_id(current_prompt)
        if not plan_id:
            return build_tool_followup_invocation(
                name="growth_stage_lookup",
                message="未识别到有效的序号/计划，请回复序号或计划名称。",
                missing_fields=["plan_choice"],
                draft=prior_draft if isinstance(prior_draft, dict) else {},
                query=query_text,
                followup_count=followup_count + 1,
                options=options,
                choice_hint=True,
                extra={"plan_filters": followup.get("plan_filters") or {}},
            )
        return _build_growth_stage_result(
            plan_id,
            plan_filters=followup.get("plan_filters") or {},
        )

    plan_id = _extract_plan_id(prompt) or _extract_plan_id(current_prompt)
    if plan_id:
        return _build_growth_stage_result(
            plan_id,
            plan_filters=followup.get("plan_filters") or {},
        )

    try:
        fresh_draft = extract_planting_details(current_prompt, llm_extract=llm_extract_planting)
    except Exception:
        fresh_draft = extract_planting_details(current_prompt)
    if prior_draft and prior_missing:
        answers = fresh_draft.model_dump(exclude_none=True)
        draft = merge_planting_answers(prior_draft, answers=answers)
        followup_count += 1
    else:
        draft = fresh_draft

    filters: Dict[str, object] = {}
    plan_name = _extract_plan_name(prompt)
    if plan_name:
        filters["plan_name"] = plan_name
    if draft and draft.variety:
        filters["variety"] = draft.variety
    if draft and draft.sowing_date:
        filters["sowing_date"] = draft.sowing_date
    if draft and draft.transplant_date:
        filters["transplant_date"] = draft.transplant_date
    if draft and draft.planting_method:
        filters["planting_method"] = draft.planting_method
    if draft and getattr(draft, "culti_type", None):
        filters["culti_type"] = getattr(draft, "culti_type")

    draft_payload = draft.model_dump(mode="json", exclude_none=True) if draft else {}
    if not filters:
        return build_tool_followup_invocation(
            name="growth_stage_lookup",
            message="请提供品种名称或种植计划名称，以便查询种植计划。",
            missing_fields=["plan_query"],
            draft=draft_payload,
            query=query_text,
            followup_count=followup_count,
            extra={"plan_filters": {}},
        )

    try:
        rows, id_col, columns = search_planting_plans(filters, limit=5)
    except Exception as exc:
        return build_tool_followup_invocation(
            name="growth_stage_lookup",
            message=f"查询种植计划失败: {exc}。请提供更具体的品种或计划名称。",
            missing_fields=["plan_query"],
            draft=draft_payload,
            query=query_text,
            followup_count=followup_count,
            extra={"plan_filters": filters},
        )

    if not rows:
        if "plan_name" in filters:
            try:
                active_rows, active_id_col, active_columns = list_active_planting_plans(limit=5)
            except Exception:
                active_rows, active_id_col, active_columns = ([], "", [])
            if active_rows:
                choice_options = [
                    _format_plan_option_text(row, active_columns, active_id_col)
                    for row in active_rows
                ]
                message_lines = [
                    "未找到与计划名称匹配的记录，以下是当前启用的种植计划（仅展示前5条），请回复序号："
                ]
                for idx, option in enumerate(choice_options, start=1):
                    message_lines.append(f"{idx}. {option}")
                return build_tool_followup_invocation(
                    name="growth_stage_lookup",
                    message="\n".join(message_lines),
                    missing_fields=["plan_choice"],
                    draft=draft_payload,
                    query=query_text,
                    followup_count=followup_count + 1,
                    options=choice_options,
                    choice_hint=True,
                    extra={"plan_filters": filters},
                )
        return build_tool_followup_invocation(
            name="growth_stage_lookup",
            message="未找到符合条件的种植计划，请提供更具体的品种或计划名称。",
            missing_fields=["plan_query"],
            draft=draft_payload,
            query=query_text,
            followup_count=followup_count,
            extra={"plan_filters": filters},
        )

    if len(rows) == 1:
        return _build_growth_stage_result(rows[0].get(id_col), plan_filters=filters)

    choice_options = [_format_plan_option_text(row, columns, id_col) for row in rows]
    message_lines = ["找到多个种植计划，请回复序号："]
    for idx, option in enumerate(choice_options, start=1):
        message_lines.append(f"{idx}. {option}")
    return build_tool_followup_invocation(
        name="growth_stage_lookup",
        message="\n".join(message_lines),
        missing_fields=["plan_choice"],
        draft=draft_payload,
        query=query_text,
        followup_count=followup_count + 1,
        options=choice_options,
        choice_hint=True,
        extra={"plan_filters": filters},
    )


def _build_growth_stage_result(
    plan_id: object, *, plan_filters: Optional[Dict[str, object]] = None
) -> ToolInvocation:
    normalized_plan_id = _normalize_plan_id(plan_id)
    if not normalized_plan_id:
        return build_tool_followup_invocation(
            name="growth_stage_lookup",
            message="请提供有效的计划 ID、计划名称或品种名称。",
            missing_fields=["plan_id"],
            draft={},
            extra={"plan_filters": plan_filters or {}},
        )
    try:
        result = query_growth_stage_from_plan_id(normalized_plan_id)
        provider_response = result.model_dump(mode="json")
    except Exception as exc:
        return ToolInvocation(
            name="growth_stage_lookup",
            message=f"生育期结果查询失败: {exc}",
            data={"plan_id": normalized_plan_id, "plan_filters": plan_filters or {}},
        )

    try:
        planting = resolve_planting_from_plan_id(normalized_plan_id)
    except Exception:
        planting = None
    if result and planting:
        message = format_growth_stage_message(planting, result.stages)
    else:
        message = _format_stage_only_message(result.stages if result else {})
    data: Dict[str, object] = {
        "plan_id": normalized_plan_id,
        "plan_filters": plan_filters or {},
        "provider_response": provider_response,
        "growth_stage": provider_response,
    }
    if planting:
        data["planting"] = planting.model_dump(mode="json")
    return ToolInvocation(name="growth_stage_lookup", message=message, data=data)


@auto_register_tool(
    "plant_plan_list_active",
    description="查询全部启用的种植计划（is_active=true）。",
)
def plant_plan_list_active(prompt: str) -> ToolInvocation:
    """列出当前启用计划，并附带前 5 条可读预览供人工核查。"""
    _ = prompt
    rows, id_col, columns = list_active_planting_plans(limit=None)
    plans: List[Dict[str, object]] = []
    for row in rows:
        plan_id = row.get(id_col) if isinstance(row, dict) else None
        plan_name = extract_plan_name_from_row(row, columns) or ""
        planting = build_planting_from_plan_row(row, columns)
        planting_payload = (
            planting.model_dump(mode="json") if planting else {}
        )
        plans.append(
            {
                "plan_id": plan_id,
                "plan_name": plan_name,
                "planting": planting_payload,
            }
        )
    count = len(plans)
    if not plans:
        return ToolInvocation(
            name="plant_plan_list_active",
            message="未找到启用的种植计划。",
            data={"count": 0, "plans": []},
        )

    preview = plans[:5]
    preview_lines = []
    for idx, item in enumerate(preview, start=1):
        label = item.get("plan_name") or "种植计划"
        plan_id = item.get("plan_id")
        planting = item.get("planting") or {}
        details = []
        if planting.get("variety"):
            details.append(f"品种={planting.get('variety')}")
        if planting.get("sowing_date"):
            details.append(f"播种={planting.get('sowing_date')}")
        if planting.get("planting_method"):
            method = planting.get("planting_method")
            label_method = (
                "直播"
                if method == "direct_seeding"
                else "插秧"
                if method == "transplanting"
                else str(method)
            )
            details.append(f"方式={label_method}")
        if planting.get("culti_type"):
            details.append(f"类型={planting.get('culti_type')}")
        suffix = f"（{'，'.join(details)}）" if details else ""
        if plan_id is not None:
            preview_lines.append(f"{idx}. {label}（id={plan_id}）{suffix}")
        else:
            preview_lines.append(f"{idx}. {label}{suffix}")
    message = (
        f"已获取启用的种植计划 {count} 条。"
        + "\n示例（前5条）:\n"
        + "\n".join(preview_lines)
    )
    return ToolInvocation(
        name="plant_plan_list_active",
        message=message,
        data={"count": count, "plans": plans},
    )


@auto_register_tool(
    "plant_plan_delete",
    description="删除种植计划。需要提供 plant_season_id。",
)
def plant_plan_delete(prompt: str) -> ToolInvocation:
    """删除指定计划；缺少 plan_id 时返回追问态而不是直接报错。"""
    plan_id = _extract_plan_id(prompt)
    if not plan_id:
        return build_tool_followup_invocation(
            name="plant_plan_delete",
            message="请提供要删除的 plant_season_id。",
            missing_fields=["plant_season_id"],
            draft={"plant_season_id": None},
            query=prompt,
        )
    try:
        response = delete_crop_calendar_plan(plan_id)
    except Exception as exc:
        return ToolInvocation(
            name="plant_plan_delete",
            message=f"删除失败: {exc}",
            data={"plant_season_id": plan_id},
        )
    return ToolInvocation(
        name="plant_plan_delete",
        message="已删除种植计划。",
        data={"plant_season_id": plan_id, "response": response},
    )


@auto_register_tool(
    "growth_stage_lookup",
    description="根据已有种植计划查询生育期结果。可直接提供 plan_id，也可提供计划名称、品种等线索后由系统匹配种植计划。",
)
def growth_stage_lookup(prompt: str) -> ToolInvocation:
    """根据计划 ID 或计划线索查询生育期结果；多计划时返回追问态。"""
    return _resolve_growth_stage_query(prompt)
