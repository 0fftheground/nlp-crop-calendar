from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ...application.services.crop_calendar_service import (
    delete_crop_calendar_plan,
)
from ...application.services.growth_stage_service import (
    build_planting_from_plan_row,
    extract_plan_name_from_row,
    list_active_planting_plans,
)
from ..followup import build_tool_followup_invocation
from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


_PLAN_ID_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)


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
