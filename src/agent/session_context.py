from __future__ import annotations

import re
from datetime import date, datetime
from typing import Mapping, Optional

from ..application.services.sowing_suitability_service import (
    build_contextual_sowing_query,
)
from ..application.services.weather_service import normalize_weather_prompt
from ..domain.planting import extract_planting_details
from ..infra.variety_store import find_exact_variety_in_text
from ..schemas.models import ToolInvocation, WorkflowResponse
from .followup import get_followup_missing_fields
from .planner import ActionPlan

_TOOL_CONTEXT_KEY = "tool_contexts"
_WORKFLOW_CONTEXT_KEY = "workflow_contexts"
_LAST_CONTEXT_KEY = "last_context"
_REGION_PATTERNS = (
    r"^在([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)?$",
    r"^到([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)?$",
    r"^([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)$",
)
_YEAR_RE = re.compile(r"(20\d{2})年")
_VARIETY_FOLLOWUP_TOKENS = (
    "审定",
    "品种",
    "适种",
    "适宜",
    "抗",
    "产量",
    "生育期",
    "湖南",
    "湖北",
    "安徽",
    "江苏",
    "区域",
    "地区",
)
_METHOD_LABELS = {
    "direct_seeding": "直播",
    "transplanting": "移栽",
}
_INVALID_REGION_TOKENS = (
    "改成",
    "换成",
    "直播",
    "移栽",
    "插秧",
    "播种",
    "生育期",
    "方案",
)


def build_contextual_plan(
    prompt: str, session_payload: Optional[Mapping[str, object]]
) -> Optional[ActionPlan]:
    if not isinstance(session_payload, Mapping):
        return None
    text = str(prompt or "").strip()
    if not text:
        return None
    for kind, name, context in _iter_context_candidates(session_payload):
        if kind == "tool" and name == "weather_lookup":
            payload = _build_contextual_weather_query(text, context)
            if payload:
                return ActionPlan(
                    action="tool",
                    name=name,
                    input=payload,
                    reason=f"session_context:{name}",
                )
        if kind == "tool" and name == "variety_lookup":
            query = _build_contextual_variety_query(text, context)
            if query:
                return ActionPlan(
                    action="tool",
                    name=name,
                    input={"query": query},
                    reason=f"session_context:{name}",
                )
        if kind == "tool" and name == "sowing_suitability_lookup":
            payload = build_contextual_sowing_query(text, context)
            if payload:
                return ActionPlan(
                    action="tool",
                    name=name,
                    input=payload,
                    reason=f"session_context:{name}",
                )
        if kind == "workflow" and name == "growth_stage_query_workflow":
            query = _build_contextual_growth_prompt(text, context)
            if query:
                return ActionPlan(
                    action="workflow",
                    name=name,
                    input={"prompt": query},
                    reason=f"session_context:{name}",
                )
        if kind == "workflow" and name == "crop_calendar_workflow":
            query = _build_contextual_crop_calendar_prompt(text, context)
            if query:
                return ActionPlan(
                    action="workflow",
                    name=name,
                    input={"prompt": query},
                    reason=f"session_context:{name}",
                )
    return None


def extract_session_context_from_tool(
    tool: Optional[ToolInvocation],
) -> Optional[tuple[str, dict[str, object]]]:
    if tool is None:
        return None
    data = tool.data or {}
    if get_followup_missing_fields(data):
        return None
    if tool.name == "weather_lookup":
        context: dict[str, object] = {}
        region = str(data.get("region") or "").strip()
        if region and not region.startswith("farm:"):
            context["region"] = region
        for key in ("start_date", "end_date", "granularity"):
            value = data.get(key)
            if value not in (None, ""):
                context[key] = value
        include_advice = data.get("include_advice")
        if isinstance(include_advice, bool):
            context["include_advice"] = include_advice
        if context.get("start_date") and context.get("end_date"):
            return tool.name, context
        return None
    if tool.name == "variety_lookup":
        variety = str(data.get("variety") or "").strip()
        selected = data.get("selected")
        if not variety and isinstance(selected, Mapping):
            variety = str(selected.get("品种名称") or "").strip()
        if not variety:
            return None
        context = {"variety": variety}
        crop = str(data.get("crop") or "").strip()
        if crop:
            context["crop"] = crop
        region_choice = str(data.get("region_choice") or "").strip()
        if region_choice and region_choice != "__all__":
            context["region_choice"] = region_choice
        if isinstance(selected, Mapping):
            context["selected"] = dict(selected)
        return tool.name, context
    if tool.name == "sowing_suitability_lookup":
        resolved = data.get("resolved")
        if isinstance(resolved, Mapping):
            return tool.name, dict(resolved)
    return None


def extract_session_context_from_workflow(
    workflow_name: str, plan: Optional[WorkflowResponse]
) -> Optional[tuple[str, dict[str, object]]]:
    if plan is None:
        return None
    data = plan.data or {}
    if workflow_name == "growth_stage_query_workflow":
        context: dict[str, object] = {}
        workflow = data.get("workflow")
        if isinstance(workflow, Mapping):
            plan_id = workflow.get("plan_id")
            if plan_id not in (None, ""):
                context["plan_id"] = plan_id
            plan_filters = workflow.get("plan_filters")
            if isinstance(plan_filters, Mapping) and plan_filters:
                context["plan_filters"] = dict(plan_filters)
        planting = _reduce_planting_context(data.get("planting"))
        if planting:
            context["planting"] = planting
        return (workflow_name, context) if context else None
    if workflow_name == "crop_calendar_workflow":
        planting = _reduce_planting_context(data.get("planting"))
        if not planting:
            return None
        context = {"planting": planting}
        for key in ("plant_season_id", "resolved_region_id"):
            value = data.get(key)
            if value not in (None, ""):
                context[key] = value
        return workflow_name, context
    return None


def _iter_context_candidates(session_payload: Mapping[str, object]):
    last = session_payload.get(_LAST_CONTEXT_KEY)
    if isinstance(last, Mapping):
        kind = str(last.get("kind") or "").strip()
        name = str(last.get("name") or "").strip()
        context = _get_context(session_payload, kind, name)
        if context:
            yield kind, name, context


def _get_context(
    session_payload: Mapping[str, object], kind: str, name: str
) -> Optional[dict[str, object]]:
    key = _TOOL_CONTEXT_KEY if kind == "tool" else _WORKFLOW_CONTEXT_KEY
    bucket = session_payload.get(key)
    if not isinstance(bucket, Mapping):
        return None
    context = bucket.get(name)
    return dict(context) if isinstance(context, Mapping) else None


def _build_contextual_weather_query(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[dict[str, object]]:
    if not isinstance(context, Mapping):
        return None
    base = {
        key: value
        for key, value in dict(context).items()
        if key in {"region", "start_date", "end_date", "granularity", "include_advice"}
        and value not in (None, "")
    }
    if not base:
        return None
    overrides: dict[str, object] = {}
    _, query = normalize_weather_prompt(prompt)
    if query is not None:
        payload = query.model_dump(mode="json")
        for key in ("region", "start_date", "end_date", "granularity", "include_advice"):
            value = payload.get(key)
            if value not in (None, ""):
                overrides[key] = value
    else:
        region = _extract_region_hint(prompt)
        if region:
            overrides["region"] = region
        dates = _extract_dates(prompt)
        if len(dates) >= 2:
            overrides["start_date"] = dates[0]
            overrides["end_date"] = dates[1]
        year = _extract_year(prompt)
        if year is not None and "start_date" in base and "end_date" in base:
            start = _parse_iso_date(base.get("start_date"))
            end = _parse_iso_date(base.get("end_date"))
            if start and end:
                overrides["start_date"] = start.replace(year=year).isoformat()
                overrides["end_date"] = end.replace(year=year).isoformat()
    if not overrides:
        return None
    merged = dict(base)
    merged.update(overrides)
    if not (merged.get("start_date") and merged.get("end_date")):
        return None
    return merged


def _build_contextual_variety_query(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not isinstance(context, Mapping):
        return None
    variety = str(context.get("variety") or "").strip()
    if not variety:
        return None
    region = _extract_region_hint(prompt) or str(context.get("region_choice") or "").strip()
    explicit_variety = (
        _find_exact_variety(prompt) if _should_try_variety_match(prompt) else None
    ) or variety
    if not _should_resume_variety(prompt, region != str(context.get("region_choice") or "").strip()):
        return None
    region_part = f"在{region}" if region else ""
    return f"查询品种{explicit_variety}{region_part}的审定信息。用户补充：{prompt}"


def _build_contextual_growth_prompt(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    planting = _merge_planting_context(prompt, context)
    if not planting:
        return None
    return f"查询{_describe_planting(planting)}的生育期。"


def _build_contextual_crop_calendar_prompt(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    planting = _merge_planting_context(prompt, context)
    if not planting:
        return None
    return f"请基于以下条件生成农事方案：{_describe_planting(planting)}。"


def _merge_planting_context(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[dict[str, object]]:
    if not isinstance(context, Mapping):
        return None
    base = _reduce_planting_context(context.get("planting"))
    if not base:
        plan_filters = context.get("plan_filters")
        if isinstance(plan_filters, Mapping):
            base = _reduce_planting_context(plan_filters)
    if not base:
        return None
    overrides = _extract_planting_overrides(prompt)
    if not overrides:
        return None
    merged = dict(base)
    merged.update(overrides)
    return merged


def _extract_planting_overrides(prompt: str) -> dict[str, object]:
    text = str(prompt or "").strip()
    if not text:
        return {}
    overrides: dict[str, object] = {}
    region = _extract_region_hint(text)
    if region:
        overrides["region_id"] = region
    try:
        draft = extract_planting_details(text, variety_resolver=lambda _value: [])
    except Exception:
        draft = None
    if draft is not None:
        for key in ("culti_type", "planting_method", "sowing_date", "transplant_date"):
            value = getattr(draft, key, None)
            if value not in (None, ""):
                overrides[key] = value.isoformat() if hasattr(value, "isoformat") else value
    variety = _find_exact_variety(text) if _should_try_variety_match(text) else None
    if variety:
        overrides["variety"] = variety
    return overrides


def _reduce_planting_context(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    context: dict[str, object] = {}
    for key in (
        "region_id",
        "crop",
        "variety",
        "culti_type",
        "planting_method",
        "sowing_date",
        "transplant_date",
    ):
        item = value.get(key)
        if item in (None, ""):
            continue
        context[key] = item
    return context


def _describe_planting(planting: Mapping[str, object]) -> str:
    parts: list[str] = []
    region = str(planting.get("region_id") or "").strip()
    if region:
        parts.append(f"地区{region}")
    crop = str(planting.get("crop") or "").strip()
    if crop:
        parts.append(f"作物{crop}")
    variety = str(planting.get("variety") or "").strip()
    if variety:
        parts.append(f"品种{variety}")
    culti_type = str(planting.get("culti_type") or "").strip()
    if culti_type:
        parts.append(f"稻作类型{culti_type}")
    method = str(planting.get("planting_method") or "").strip()
    if method:
        parts.append(f"种植方式{_METHOD_LABELS.get(method, method)}")
    sowing_date = _stringify_date(planting.get("sowing_date"))
    if sowing_date:
        parts.append(f"播种日期{sowing_date}")
    transplant_date = _stringify_date(planting.get("transplant_date"))
    if transplant_date:
        parts.append(f"移栽日期{transplant_date}")
    return "，".join(parts)


def _should_resume_variety(prompt: str, region_changed: bool) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if region_changed:
        return True
    if any(token in text for token in _VARIETY_FOLLOWUP_TOKENS):
        return True
    if len(text) <= 12 and re.search(r"(呢|吗|如何|怎么样)$", text):
        return True
    return False


def _extract_region_hint(text: str) -> Optional[str]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    for pattern in _REGION_PATTERNS:
        match = re.search(pattern, prompt)
        if not match:
            continue
        region = str(match.group(1) or "").strip()
        region = re.sub(r"(呢|吗|呀|啊)$", "", region).strip()
        if region and not any(token in region for token in _INVALID_REGION_TOKENS):
            return region
    return None


def _extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", text):
        try:
            values.append(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                ).isoformat()
            )
        except ValueError:
            continue
    for match in re.finditer(r"(20\d{2})(\d{2})(\d{2})", text):
        try:
            values.append(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                ).isoformat()
            )
        except ValueError:
            continue
    return values


def _extract_year(text: str) -> Optional[int]:
    match = _YEAR_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_iso_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _stringify_date(value: object) -> str:
    parsed = _parse_iso_date(value)
    return parsed.isoformat() if parsed else ""


def _find_exact_variety(text: str) -> Optional[str]:
    try:
        return find_exact_variety_in_text(text) or None
    except Exception:
        return None


def _should_try_variety_match(text: str) -> bool:
    prompt = str(text or "").strip()
    if len(prompt) < 4:
        return False
    if "品种" in prompt:
        return True
    if "号" in prompt:
        return True
    return bool(re.search(r"[A-Za-z0-9]{2,}", prompt))
