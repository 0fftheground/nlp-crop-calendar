from __future__ import annotations

from dataclasses import dataclass, replace
import re
from datetime import date, datetime
from typing import Callable, Mapping, Optional

from ..application.services.sowing_suitability_service import (
    build_contextual_sowing_query,
)
from .field_updates import (
    extract_planting_field_overrides,
    extract_region_followup_hint,
)
from ..application.services.weather_service import (
    extract_weather_operations,
    normalize_weather_prompt,
)
from ..domain.date_parser import (
    extract_date_range,
    extract_explicit_dates,
    extract_relative_date_range,
)
from ..infra.variety_store import find_exact_variety_in_text
from ..schemas.models import ToolInvocation, WorkflowResponse
from .followup import get_followup_missing_fields, parse_followup_index
from .intent_boundaries import (
    looks_like_crop_calendar_query,
    looks_like_non_agri_life_query,
    looks_like_sowing_query,
)
from .planner import ActionPlan

_TOOL_CONTEXT_KEY = "tool_contexts"
_WORKFLOW_CONTEXT_KEY = "workflow_contexts"
_LAST_CONTEXT_KEY = "last_context"
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
    "下周",
    "上周",
    "本周",
    "这周",
    "下星期",
    "上星期",
    "这星期",
    "本星期",
    "直播",
    "移栽",
    "插秧",
    "播种",
    "生育期",
    "方案",
    "施肥",
    "炼苗",
    "移栽",
    "翻地",
    "打药",
    "喷药",
    "收割",
    "收获",
    "整地",
)
_BRIEF_FOLLOWUP_SUFFIX_RE = re.compile(r"(呢|吗|呀|啊|怎么样|如何|行吗|可以吗)$")
_WEATHER_QUERY_TOKENS = ("天气", "气象", "气温", "降雨", "降水", "湿度", "风速", "预报")
_WEATHER_OPERATION_ONLY_PATTERNS = (
    re.compile(r"施[\u4e00-\u9fffA-Za-z0-9]{0,8}肥"),
    re.compile(r"打[\u4e00-\u9fffA-Za-z0-9]{0,8}药"),
    re.compile(r"喷[\u4e00-\u9fffA-Za-z0-9]{0,8}药"),
)
_WEATHER_OPERATION_ONLY_ALIASES = (
    "施肥",
    "追肥",
    "炼苗",
    "移栽",
    "插秧",
    "翻地",
    "打药",
    "喷药",
    "收割",
    "收获",
    "整地",
    "浇水",
    "灌溉",
    "除草",
    "育秧",
    "病虫害防治",
    "防病",
    "治虫",
)
_PLAN_ID_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_PLAN_DELETE_TOKENS = ("删除", "删掉", "删了", "移除")
_GROWTH_STAGE_QUERY_TOKENS = (
    "生育期",
    "生长阶段",
    "成熟期",
    "抽穗",
    "返青",
    "分蘖",
    "拔节",
    "孕穗",
)
_PLAN_SELF_REFERENCE_TOKENS = (
    "这个计划",
    "该计划",
    "这个种植计划",
    "该种植计划",
    "这个",
    "该",
)
_EXPLICIT_THREAD_SWITCH_TOKENS = (
    "换个问题",
    "换一个问题",
    "另一个问题",
    "新问题",
    "新任务",
    "重新问",
    "重新开始",
    "不相关",
    "无关",
    "先不说这个",
    "换一个",
)
_CHINESE_INDEX_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass(frozen=True)
class ContextualPlanCandidate:
    plan: ActionPlan
    confidence: float
    kind: str
    name: str
    evidence: tuple[str, ...] = ()
    adapter_task_type: str = ""
    updatable_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionContextAdapter:
    kind: str
    name: str
    task_type: str
    updatable_fields: tuple[str, ...] = ()
    extract_context: Optional[Callable[[object], Optional[dict[str, object]]]] = None
    build_candidate: Optional[
        Callable[[str, Mapping[str, object]], Optional[ContextualPlanCandidate]]
    ] = None


def build_contextual_plan(
    prompt: str, session_payload: Optional[Mapping[str, object]]
) -> Optional[ActionPlan]:
    candidate = build_contextual_candidate(prompt, session_payload)
    return candidate.plan if candidate else None


def build_contextual_candidate(
    prompt: str, session_payload: Optional[Mapping[str, object]]
) -> Optional[ContextualPlanCandidate]:
    if not isinstance(session_payload, Mapping):
        return None
    text = str(prompt or "").strip()
    if not text:
        return None
    for adapter, context in _iter_context_candidates(session_payload):
        if adapter.build_candidate is None:
            continue
        candidate = adapter.build_candidate(text, context)
        if candidate is not None:
            return replace(
                candidate,
                adapter_task_type=adapter.task_type,
                updatable_fields=adapter.updatable_fields,
            )
    return None


def is_explicit_thread_switch_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if looks_like_non_agri_life_query(text):
        return True
    if any(token in text for token in _EXPLICIT_THREAD_SWITCH_TOKENS):
        return True
    return False


def build_thread_ownership_clarification(
    prompt: str,
    contextual_candidate: Optional[ContextualPlanCandidate],
    standalone_plan: Optional[ActionPlan],
) -> Optional[str]:
    text = str(prompt or "").strip()
    if not text or contextual_candidate is None or standalone_plan is None:
        return None
    if is_explicit_thread_switch_prompt(text):
        return None
    if (
        standalone_plan.action == contextual_candidate.plan.action
        and standalone_plan.name == contextual_candidate.plan.name
    ):
        return None
    if len(text) > 16 or parse_followup_index(text) is not None:
        return None
    standalone_label = _describe_thread_target(standalone_plan)
    contextual_label = _describe_thread_target(contextual_candidate.plan)
    if not standalone_label or not contextual_label:
        return None
    return (
        f"我不确定你是想继续当前的{contextual_label}，还是想改成新的{standalone_label}。"
        "请回复“继续当前任务”或“开启新任务”。"
    )


def should_short_circuit_contextual_candidate(
    candidate: Optional[ContextualPlanCandidate],
) -> bool:
    if candidate is None:
        return False
    return candidate.confidence >= 0.85


def resolve_session_plan(
    standalone_plan: Optional[ActionPlan],
    contextual_candidate: Optional[ContextualPlanCandidate],
) -> Optional[ActionPlan]:
    if contextual_candidate is None:
        return standalone_plan
    if standalone_plan is None:
        return contextual_candidate.plan
    if standalone_plan.action == "none":
        return contextual_candidate.plan
    if (
        standalone_plan.action == contextual_candidate.plan.action
        and standalone_plan.name == contextual_candidate.plan.name
    ):
        return contextual_candidate.plan
    return standalone_plan


def list_session_context_adapters() -> tuple[SessionContextAdapter, ...]:
    return _SESSION_CONTEXT_ADAPTERS


def get_session_context_adapter(
    kind: str, name: str
) -> Optional[SessionContextAdapter]:
    if kind == "tool":
        return _TOOL_CONTEXT_ADAPTERS.get(name)
    if kind == "workflow":
        return _WORKFLOW_CONTEXT_ADAPTERS.get(name)
    return None


def _build_weather_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    payload = _build_contextual_weather_query(prompt, context)
    if not payload:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="tool",
            name="weather_lookup",
            input=payload,
            reason="session_context:weather_lookup",
        ),
        confidence=_score_weather_contextual_candidate(prompt, payload),
        kind="tool",
        name="weather_lookup",
        evidence=_collect_weather_candidate_evidence(prompt, payload),
    )


def _build_variety_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    query = _build_contextual_variety_query(prompt, context)
    if not query:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="tool",
            name="variety_lookup",
            input={"query": query},
            reason="session_context:variety_lookup",
        ),
        confidence=_score_variety_contextual_candidate(prompt),
        kind="tool",
        name="variety_lookup",
        evidence=_collect_variety_candidate_evidence(prompt),
    )


def _build_sowing_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    if _looks_like_weather_query(prompt):
        return None
    payload = build_contextual_sowing_query(prompt, context)
    if not payload:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="tool",
            name="sowing_suitability_lookup",
            input=payload,
            reason="session_context:sowing_suitability_lookup",
        ),
        confidence=_score_sowing_contextual_candidate(prompt, payload),
        kind="tool",
        name="sowing_suitability_lookup",
        evidence=_collect_sowing_candidate_evidence(prompt, payload),
    )


def _build_plan_list_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    payload = _build_contextual_plan_delete_input(prompt, context)
    if payload:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="plant_plan_delete",
                input=payload,
                reason="session_context:plant_plan_list_active->plant_plan_delete",
            ),
            confidence=_score_plan_action_contextual_candidate(prompt),
            kind="tool",
            name="plant_plan_list_active",
            evidence=_collect_plan_delete_candidate_evidence(prompt, payload),
        )
    query = _build_contextual_growth_prompt_from_plan_context(prompt, context)
    if not query:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="tool",
            name="growth_stage_lookup",
            input={"query": query},
            reason="session_context:plant_plan_list_active->growth_stage_lookup",
        ),
        confidence=_score_plan_action_contextual_candidate(prompt),
        kind="tool",
        name="plant_plan_list_active",
        evidence=_collect_growth_stage_candidate_evidence(prompt, query),
    )


def _build_growth_stage_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    query = _build_contextual_growth_prompt(prompt, context)
    if not query:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="tool",
            name="growth_stage_lookup",
            input={"query": query},
            reason="session_context:growth_stage_lookup",
        ),
        confidence=_score_workflow_contextual_candidate(prompt),
        kind="tool",
        name="growth_stage_lookup",
        evidence=_collect_workflow_candidate_evidence(prompt),
    )


def _build_crop_calendar_contextual_candidate(
    prompt: str, context: Mapping[str, object]
) -> Optional[ContextualPlanCandidate]:
    payload = _build_contextual_plan_delete_input(prompt, context)
    if payload:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="plant_plan_delete",
                input=payload,
                reason="session_context:crop_calendar_workflow->plant_plan_delete",
            ),
            confidence=_score_plan_action_contextual_candidate(prompt),
            kind="workflow",
            name="crop_calendar_workflow",
            evidence=_collect_plan_delete_candidate_evidence(prompt, payload),
        )
    growth_query = _build_contextual_growth_prompt_from_plan_context(prompt, context)
    if growth_query:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="growth_stage_lookup",
                input={"query": growth_query},
                reason="session_context:crop_calendar_workflow->growth_stage_lookup",
            ),
            confidence=_score_plan_action_contextual_candidate(prompt),
            kind="workflow",
            name="crop_calendar_workflow",
            evidence=_collect_growth_stage_candidate_evidence(prompt, growth_query),
        )
    query = _build_contextual_crop_calendar_prompt(prompt, context)
    if not query:
        return None
    return ContextualPlanCandidate(
        plan=ActionPlan(
            action="workflow",
            name="crop_calendar_workflow",
            input={"prompt": query},
            reason="session_context:crop_calendar_workflow",
        ),
        confidence=_score_workflow_contextual_candidate(prompt),
        kind="workflow",
        name="crop_calendar_workflow",
        evidence=_collect_workflow_candidate_evidence(prompt),
    )


def _looks_like_weather_query(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text or looks_like_sowing_query(text) or looks_like_crop_calendar_query(text):
        return False
    if any(token in text for token in _WEATHER_QUERY_TOKENS):
        return True
    supported, unsupported = extract_weather_operations(
        text, require_suitability_cues=True
    )
    return bool(supported or unsupported)


def _score_weather_contextual_candidate(
    prompt: str, payload: Mapping[str, object]
) -> float:
    score = 0.55
    if _is_brief_weather_followup(prompt):
        score += 0.35
    if payload.get("region") not in (None, ""):
        score += 0.1
    if payload.get("requested_operations"):
        score += 0.1
    if _prompt_has_temporal_signal(prompt):
        score += 0.1
    return min(score, 0.98)


def _collect_weather_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["weather_context"]
    if _is_brief_weather_followup(prompt):
        evidence.append("brief_followup")
    if payload.get("region") not in (None, ""):
        evidence.append("region")
    if payload.get("requested_operations"):
        evidence.append("requested_operations")
    if payload.get("start_date") and payload.get("end_date"):
        evidence.append("date_range")
    return tuple(evidence)


def _score_variety_contextual_candidate(prompt: str) -> float:
    score = 0.65
    if len(str(prompt or "").strip()) <= 12:
        score += 0.2
    if _extract_region_hint(prompt):
        score += 0.1
    return min(score, 0.95)


def _collect_variety_candidate_evidence(prompt: str) -> tuple[str, ...]:
    evidence: list[str] = ["variety_context"]
    if len(str(prompt or "").strip()) <= 12:
        evidence.append("brief_followup")
    if _extract_region_hint(prompt):
        evidence.append("region")
    return tuple(evidence)


def _score_sowing_contextual_candidate(
    prompt: str, payload: Mapping[str, object]
) -> float:
    score = 0.65
    text = str(prompt or "").strip()
    if len(text) <= 16 or _extract_region_hint(text):
        score += 0.2
    if any(payload.get(key) not in (None, "") for key in ("region_id", "farm_id")):
        score += 0.1
    if payload.get("variety") not in (None, ""):
        score += 0.05
    return min(score, 0.95)


def _collect_sowing_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["sowing_context"]
    if len(str(prompt or "").strip()) <= 16:
        evidence.append("brief_followup")
    if any(payload.get(key) not in (None, "") for key in ("region_id", "farm_id")):
        evidence.append("region")
    if payload.get("variety") not in (None, ""):
        evidence.append("variety")
    return tuple(evidence)


def _score_plan_action_contextual_candidate(prompt: str) -> float:
    score = 0.75
    text = str(prompt or "").strip()
    if _extract_plan_reference(text) is not None or _has_plan_self_reference(text):
        score += 0.15
    if len(text) <= 16:
        score += 0.05
    return min(score, 0.96)


def _collect_plan_delete_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["plan_context", "delete_action"]
    if payload.get("plant_season_id") not in (None, ""):
        evidence.append("plan_id")
    if _extract_plan_reference(prompt) is not None or _has_plan_self_reference(prompt):
        evidence.append("plan_reference")
    return tuple(evidence)


def _collect_growth_stage_candidate_evidence(
    prompt: str, query: str
) -> tuple[str, ...]:
    evidence: list[str] = ["plan_context", "growth_stage_action"]
    if _extract_plan_reference(prompt) is not None or _has_plan_self_reference(prompt):
        evidence.append("plan_reference")
    if "id=" in str(query):
        evidence.append("plan_id")
    return tuple(evidence)


def _score_workflow_contextual_candidate(prompt: str) -> float:
    score = 0.7
    if len(str(prompt or "").strip()) <= 16:
        score += 0.15
    return min(score, 0.95)


def _collect_workflow_candidate_evidence(prompt: str) -> tuple[str, ...]:
    evidence: list[str] = ["workflow_context"]
    if len(str(prompt or "").strip()) <= 16:
        evidence.append("brief_followup")
    return tuple(evidence)


def _extract_weather_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    data = tool.data or {}
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
    requested_operations = data.get("requested_operations")
    if isinstance(requested_operations, list):
        operations = [str(item).strip() for item in requested_operations if str(item).strip()]
        if operations:
            context["requested_operations"] = operations
    if context.get("start_date") and context.get("end_date"):
        return context
    return None


def _extract_variety_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    data = tool.data or {}
    variety = str(data.get("variety") or "").strip()
    selected = data.get("selected")
    if not variety and isinstance(selected, Mapping):
        variety = str(selected.get("品种名称") or "").strip()
    if not variety:
        return None
    context: dict[str, object] = {"variety": variety}
    crop = str(data.get("crop") or "").strip()
    if crop:
        context["crop"] = crop
    region_choice = str(data.get("region_choice") or "").strip()
    if region_choice and region_choice != "__all__":
        context["region_choice"] = region_choice
    if isinstance(selected, Mapping):
        context["selected"] = dict(selected)
    return context


def _extract_sowing_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    data = tool.data or {}
    resolved = data.get("resolved")
    if not isinstance(resolved, Mapping):
        return None
    context = dict(resolved)
    farm_id = context.get("farm_id")
    if farm_id not in (None, ""):
        context["farm_id"] = str(farm_id).strip()
    return context


def _extract_plan_list_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    raw_plans = (tool.data or {}).get("plans")
    if not isinstance(raw_plans, list):
        return None
    plans: list[dict[str, object]] = []
    for item in raw_plans:
        if not isinstance(item, Mapping):
            continue
        plan_id = item.get("plan_id")
        if plan_id in (None, ""):
            continue
        plan_payload: dict[str, object] = {"plan_id": str(plan_id).strip()}
        plan_name = str(item.get("plan_name") or "").strip()
        if plan_name:
            plan_payload["plan_name"] = plan_name
        planting = item.get("planting")
        if isinstance(planting, Mapping) and planting:
            plan_payload["planting"] = dict(planting)
        plans.append(plan_payload)
    return {"plans": plans} if plans else None


def _extract_plan_delete_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    plan_id = (tool.data or {}).get("plant_season_id")
    if plan_id in (None, ""):
        return None
    return {"plant_season_id": str(plan_id).strip()}


def _extract_growth_stage_context(value: object) -> Optional[dict[str, object]]:
    data: Mapping[str, object]
    if isinstance(value, ToolInvocation):
        data = value.data or {}
    elif isinstance(value, WorkflowResponse):
        data = value.data or {}
        workflow = data.get("workflow")
        context: dict[str, object] = {}
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
        return context or None
    else:
        return None
    context = {}
    plan_id = data.get("plan_id")
    if plan_id not in (None, ""):
        context["plan_id"] = str(plan_id).strip()
    plan_filters = data.get("plan_filters")
    if isinstance(plan_filters, Mapping) and plan_filters:
        context["plan_filters"] = dict(plan_filters)
    planting = _reduce_planting_context(data.get("planting"))
    if planting:
        context["planting"] = planting
    return context or None


def _extract_crop_calendar_context(value: object) -> Optional[dict[str, object]]:
    if not isinstance(value, WorkflowResponse):
        return None
    data = value.data or {}
    if "save_response" in data:
        return None
    planting = _reduce_planting_context(data.get("planting"))
    if not planting:
        return None
    context: dict[str, object] = {"planting": planting}
    for key in ("plant_season_id", "resolved_region_id"):
        field_value = data.get(key)
        if field_value not in (None, ""):
            context[key] = field_value
    return context


def extract_session_context_from_tool(
    tool: Optional[ToolInvocation],
) -> Optional[tuple[str, dict[str, object]]]:
    if tool is None:
        return None
    data = tool.data or {}
    if get_followup_missing_fields(data):
        return None
    adapter = get_session_context_adapter("tool", tool.name)
    if adapter is None or adapter.extract_context is None:
        return None
    context = adapter.extract_context(tool)
    if not context:
        return None
    return tool.name, context
    return None


def extract_session_context_from_workflow(
    workflow_name: str, plan: Optional[WorkflowResponse]
) -> Optional[tuple[str, dict[str, object]]]:
    if plan is None:
        return None
    adapter = get_session_context_adapter("workflow", workflow_name)
    if adapter is None or adapter.extract_context is None:
        return None
    context = adapter.extract_context(plan)
    if not context:
        return None
    return workflow_name, context


def _iter_context_candidates(session_payload: Mapping[str, object]):
    last = session_payload.get(_LAST_CONTEXT_KEY)
    if isinstance(last, Mapping):
        kind = str(last.get("kind") or "").strip()
        name = str(last.get("name") or "").strip()
        adapter = get_session_context_adapter(kind, name)
        if adapter is None:
            return
        context = _get_context(session_payload, kind, name)
        if context:
            yield adapter, context


def _get_context(
    session_payload: Mapping[str, object], kind: str, name: str
) -> Optional[dict[str, object]]:
    key = _TOOL_CONTEXT_KEY if kind == "tool" else _WORKFLOW_CONTEXT_KEY
    bucket = session_payload.get(key)
    if not isinstance(bucket, Mapping):
        return None
    context = bucket.get(name)
    return dict(context) if isinstance(context, Mapping) else None


_SESSION_CONTEXT_ADAPTERS: tuple[SessionContextAdapter, ...] = (
    SessionContextAdapter(
        kind="tool",
        name="weather_lookup",
        task_type="weather",
        updatable_fields=("region", "start_date", "end_date", "requested_operations"),
        extract_context=_extract_weather_context,
        build_candidate=_build_weather_contextual_candidate,
    ),
    SessionContextAdapter(
        kind="tool",
        name="variety_lookup",
        task_type="variety",
        updatable_fields=("variety", "crop", "region_choice", "selected"),
        extract_context=_extract_variety_context,
        build_candidate=_build_variety_contextual_candidate,
    ),
    SessionContextAdapter(
        kind="tool",
        name="sowing_suitability_lookup",
        task_type="sowing",
        updatable_fields=("variety", "culti_type", "planting_method", "region_id", "farm_id"),
        extract_context=_extract_sowing_context,
        build_candidate=_build_sowing_contextual_candidate,
    ),
    SessionContextAdapter(
        kind="tool",
        name="plant_plan_list_active",
        task_type="plan_list",
        updatable_fields=("plant_season_id", "plan_name"),
        extract_context=_extract_plan_list_context,
        build_candidate=_build_plan_list_contextual_candidate,
    ),
    SessionContextAdapter(
        kind="tool",
        name="plant_plan_delete",
        task_type="plan_delete",
        updatable_fields=("plant_season_id",),
        extract_context=_extract_plan_delete_context,
    ),
    SessionContextAdapter(
        kind="tool",
        name="growth_stage_lookup",
        task_type="growth_stage",
        updatable_fields=("plan_id", "plan_filters", "planting"),
        extract_context=_extract_growth_stage_context,
        build_candidate=_build_growth_stage_contextual_candidate,
    ),
    SessionContextAdapter(
        kind="workflow",
        name="crop_calendar_workflow",
        task_type="crop_calendar",
        updatable_fields=(
            "region_id",
            "crop",
            "variety",
            "culti_type",
            "planting_method",
            "sowing_date",
            "transplant_date",
            "plant_season_id",
        ),
        extract_context=_extract_crop_calendar_context,
        build_candidate=_build_crop_calendar_contextual_candidate,
    ),
)

_TOOL_CONTEXT_ADAPTERS = {
    adapter.name: adapter for adapter in _SESSION_CONTEXT_ADAPTERS if adapter.kind == "tool"
}
_WORKFLOW_CONTEXT_ADAPTERS = {
    adapter.name: adapter
    for adapter in _SESSION_CONTEXT_ADAPTERS
    if adapter.kind == "workflow"
}


def _build_contextual_weather_query(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[dict[str, object]]:
    if not isinstance(context, Mapping):
        return None
    if looks_like_sowing_query(prompt) or looks_like_crop_calendar_query(prompt):
        return None
    base = {
        key: value
        for key, value in dict(context).items()
        if key
        in {
            "region",
            "start_date",
            "end_date",
            "granularity",
            "include_advice",
            "requested_operations",
        }
        and value not in (None, "")
    }
    if not base:
        return None
    supported_ops, unsupported_ops = extract_weather_operations(
        prompt, require_suitability_cues=False
    )
    overrides: dict[str, object] = {}
    _, query = normalize_weather_prompt(prompt)
    if query is not None:
        payload = query.model_dump(mode="json")
        for key in ("region", "start_date", "end_date", "granularity", "include_advice"):
            value = payload.get(key)
            if value not in (None, ""):
                overrides[key] = value
    else:
        region = None
        if not _looks_like_weather_operation_only_followup(
            prompt, supported_ops, unsupported_ops
        ):
            region = _extract_region_hint(prompt)
        if region:
            overrides["region"] = region
        parsed_range = extract_date_range(prompt, today=date.today())
        if parsed_range:
            overrides["start_date"] = parsed_range[0].isoformat()
            overrides["end_date"] = parsed_range[1].isoformat()
        year = _extract_year(prompt)
        if year is not None and "start_date" in base and "end_date" in base:
            start = _parse_iso_date(base.get("start_date"))
            end = _parse_iso_date(base.get("end_date"))
            if start and end:
                overrides["start_date"] = start.replace(year=year).isoformat()
                overrides["end_date"] = end.replace(year=year).isoformat()
    if supported_ops:
        overrides["requested_operations"] = supported_ops
    if not overrides and (supported_ops or unsupported_ops):
        if unsupported_ops and not supported_ops and not _is_brief_weather_followup(prompt):
            return None
        return dict(base)
    if not overrides:
        return None
    merged = dict(base)
    merged.update(overrides)
    if not (merged.get("start_date") and merged.get("end_date")):
        return None
    return merged


def _prompt_has_temporal_signal(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if extract_explicit_dates(text):
        return True
    if extract_relative_date_range(text, today=date.today()) is not None:
        return True
    if extract_date_range(text, today=date.today()) is not None:
        return True
    return False


def _is_brief_weather_followup(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if len(text) <= 12 and _BRIEF_FOLLOWUP_SUFFIX_RE.search(text):
        return True
    if any(text.startswith(prefix) for prefix in ("那", "那就", "那改成", "改成", "换成")):
        return True
    return False


def _looks_like_weather_operation_only_followup(
    prompt: str, supported_ops: list[str], unsupported_ops: list[str]
) -> bool:
    if not (supported_ops or unsupported_ops):
        return False
    text = str(prompt or "").strip()
    if not text:
        return False
    text = re.sub(r"^(?:那|那就|那改成|改成|换成)", "", text).strip()
    text = _BRIEF_FOLLOWUP_SUFFIX_RE.sub("", text).strip()
    if not text:
        return False
    normalized = re.sub(r"\s+", "", text)
    if normalized in _WEATHER_OPERATION_ONLY_ALIASES:
        return True
    if any(pattern.fullmatch(normalized) for pattern in _WEATHER_OPERATION_ONLY_PATTERNS):
        return True
    return False


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


def _build_contextual_growth_prompt_from_plan_context(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not isinstance(context, Mapping):
        return None
    if not _looks_like_growth_stage_query(prompt):
        return None
    plan_id = _resolve_plan_id_from_context(prompt, context)
    if not plan_id:
        return None
    return f"查询id={plan_id}的种植计划的生育期。"


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


def _build_contextual_plan_delete_input(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[dict[str, object]]:
    if not isinstance(context, Mapping):
        return None
    if not _looks_like_plan_delete_query(prompt):
        return None
    plan_id = _resolve_plan_id_from_context(prompt, context)
    if not plan_id:
        return None
    return {"plant_season_id": plan_id}


def _extract_planting_overrides(prompt: str) -> dict[str, object]:
    text = str(prompt or "").strip()
    if not text:
        return {}
    include_variety = _should_try_variety_match(text)
    return extract_planting_field_overrides(
        text,
        include_variety=include_variety,
        include_dates=True,
        include_crop=False,
        variety_matcher=_find_exact_variety if include_variety else None,
    )


def _describe_thread_target(plan: ActionPlan) -> str:
    if plan.action == "workflow" and plan.name == "crop_calendar_workflow":
        return "农事方案生成"
    if plan.action == "tool" and plan.name == "weather_lookup":
        return "天气/农事适宜度查询"
    if plan.action == "tool" and plan.name == "sowing_suitability_lookup":
        return "播期推荐"
    if plan.action == "tool" and plan.name == "variety_lookup":
        return "品种信息查询"
    if plan.action == "tool" and plan.name == "growth_stage_lookup":
        return "生育期查询"
    if plan.action == "tool" and plan.name == "plant_plan_list_active":
        return "种植计划列表查询"
    if plan.action == "tool" and plan.name == "plant_plan_delete":
        return "种植计划删除"
    return str(plan.name or plan.action or "").strip()


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


def _looks_like_plan_delete_query(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    return any(token in text for token in _PLAN_DELETE_TOKENS)


def _looks_like_growth_stage_query(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    return any(token in text for token in _GROWTH_STAGE_QUERY_TOKENS)


def _extract_plan_reference(text: str) -> Optional[int]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    explicit = parse_followup_index(prompt)
    if explicit is not None and explicit >= 1:
        return explicit
    match = re.search(r"第\s*([一二两三四五六七八九十\d]+)\s*(?:个|条|项)?(?:计划)?", prompt)
    if not match:
        return None
    token = str(match.group(1) or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _CHINESE_INDEX_MAP.get(token)


def _extract_plan_id_from_text(text: str) -> Optional[str]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    match = _PLAN_ID_RE.search(prompt)
    if match:
        return str(match.group(1) or "").strip() or None
    if prompt.isdigit():
        return prompt
    return None


def _has_plan_self_reference(text: str) -> bool:
    prompt = str(text or "").strip()
    if not prompt:
        return False
    return any(token in prompt for token in _PLAN_SELF_REFERENCE_TOKENS)


def _resolve_plan_id_from_context(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not isinstance(context, Mapping):
        return None
    explicit_plan_id = _extract_plan_id_from_text(prompt)
    if explicit_plan_id:
        return explicit_plan_id
    raw_plans = context.get("plans")
    if isinstance(raw_plans, list) and raw_plans:
        index = _extract_plan_reference(prompt)
        if index is not None and 1 <= index <= len(raw_plans):
            selected = raw_plans[index - 1]
            if isinstance(selected, Mapping):
                value = selected.get("plan_id")
                if value not in (None, ""):
                    return str(value).strip()
        text = str(prompt or "").strip()
        for item in raw_plans:
            if not isinstance(item, Mapping):
                continue
            plan_name = str(item.get("plan_name") or "").strip()
            if plan_name and plan_name in text:
                value = item.get("plan_id")
                if value not in (None, ""):
                    return str(value).strip()
    if _has_plan_self_reference(prompt):
        value = context.get("plant_season_id") or context.get("plan_id")
        if value not in (None, ""):
            return str(value).strip()
    return None


def _extract_region_hint(text: str) -> Optional[str]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    if extract_relative_date_range(prompt, today=date.today()):
        return None
    return extract_region_followup_hint(
        prompt, invalid_tokens=_INVALID_REGION_TOKENS
    )


def _extract_dates(text: str) -> list[str]:
    return [item.isoformat() for item in extract_explicit_dates(text)]


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
