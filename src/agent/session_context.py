from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import re
import time
from datetime import date, datetime
from typing import Callable, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..application.services.sowing_suitability_service import (
    build_contextual_sowing_query,
)
from .field_updates import (
    extract_field_overrides,
    extract_region_followup_hint,
)
from ..application.services.weather_service import (
    extract_weather_operations,
)
from ..domain.date_parser import (
    extract_date_range,
    extract_explicit_dates,
    extract_relative_date_range,
)
from ..infra.llm import get_extractor_model
from ..infra.variety_store import find_exact_variety_in_text
from ..observability.llm_usage import log_llm_error, log_llm_request, log_llm_response
from ..schemas.models import ToolInvocation, WorkflowResponse
from .followup import get_followup_missing_fields, parse_followup_index
from .planner import ActionPlan

_TOOL_CONTEXT_KEY = "tool_contexts"
_WORKFLOW_CONTEXT_KEY = "workflow_contexts"
_LAST_CONTEXT_KEY = "last_context"
_YEAR_RE = re.compile(r"(20\d{2})年")
_METHOD_LABELS = {
    "direct_seeding": "直播",
    "transplanting": "移栽",
}
_VARIETY_SUPPORTED_ATTRIBUTE_ALIASES = {
    "审定区域": ("审定区域", "审定地区", "哪里审定", "哪些地区审定", "哪些地方审定"),
    "审定年份": ("审定年份", "哪年审定", "什么时候审定", "何时审定"),
    "适种地区": ("适种地区", "适种区域", "适宜地区", "适宜区域", "适合哪里种"),
    "稻作类型": ("稻作类型",),
    "亚种类型": ("亚种", "亚种类型"),
    "熟期/熟制": ("熟期", "熟制"),
    "生育期(天)": ("生育期", "生育期天数", "多少天"),
}
_VARIETY_UNSUPPORTED_ATTRIBUTE_ALIASES = (
    "成熟期",
    "抗病",
    "抗性",
    "抗倒",
    "产量",
    "品质",
    "米质",
    "口感",
    "株高",
    "穗长",
    "千粒重",
    "蛋白",
    "直链淀粉",
    "香味",
    "香型",
)
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
_PLAN_ID_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_PLAN_SELF_REFERENCE_TOKENS = (
    "这个计划",
    "该计划",
    "这个种植计划",
    "该种植计划",
    "这个",
    "该",
)
_THREAD_SWITCH_TOKENS = (
    "开启新任务",
    "新任务",
    "换个问题",
    "换个任务",
    "重新开始",
    "先不说这个",
    "先不聊这个",
)
_PLAN_DELETE_TOKENS = ("删除", "删掉", "移除")
_PLAN_TASK_TOKENS = ("记录", "录", "新增", "添加", "补记")
_GROWTH_STAGE_TOKENS = ("生育期", "生长阶段", "生长周期")
_AMBIGUOUS_THREAD_TOKENS = ("这个", "那个", "这个呢", "那个呢", "那这个", "接着", "然后")
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


class _SessionActionDecision(BaseModel):
    thread_switch: bool = False
    action_type: str = "none"
    confidence: float = 0.0


_SESSION_ACTION_PROMPT = (
    "你是会话线程动作判定器。"
    "只判断这条用户输入是否在表达以下动作之一："
    "delete_plan、record_task、query_growth_stage、thread_switch。"
    "如果都不是，则 action_type=none。"
    "只有在表达非常明确时才返回对应动作；不要因为局部关键词或主题词误判。"
    "thread_switch=true 只用于用户明确说要换问题、换任务、重新开始当前以外的话题。"
    "输出严格 JSON："
    '{"thread_switch":true|false,"action_type":"none|delete_plan|record_task|query_growth_stage","confidence":0-1}'
)

_session_action_llm = None
_session_action_llm_initialized = False


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
    best: Optional[ContextualPlanCandidate] = None
    for adapter, context in _iter_context_candidates(session_payload):
        if adapter.build_candidate is None:
            continue
        candidate = adapter.build_candidate(text, context)
        if candidate is not None:
            candidate = replace(
                candidate,
                adapter_task_type=adapter.task_type,
                updatable_fields=adapter.updatable_fields,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
    return best


def _get_session_action_llm():
    global _session_action_llm, _session_action_llm_initialized
    if _session_action_llm_initialized:
        return _session_action_llm
    _session_action_llm_initialized = True
    try:
        _session_action_llm = get_extractor_model()
    except Exception:
        _session_action_llm = None
    return _session_action_llm


@lru_cache(maxsize=256)
def _extract_session_action_decision(prompt: str) -> _SessionActionDecision:
    text = str(prompt or "").strip()
    if not text:
        return _SessionActionDecision()
    llm = _get_session_action_llm()
    if llm is None:
        return _heuristic_session_action_decision(text)
    try:
        extractor = llm.with_structured_output(_SessionActionDecision)
        log_llm_request(
            "session_context_action",
            model=llm,
            system_prompt=_SESSION_ACTION_PROMPT,
            user_prompt=text,
        )
        started_at = time.perf_counter()
        raw = extractor.invoke(
            [
                SystemMessage(content=_SESSION_ACTION_PROMPT),
                HumanMessage(content=text),
            ]
        )
        if isinstance(raw, _SessionActionDecision):
            result = raw
        else:
            result = _SessionActionDecision.model_validate(raw)
        log_llm_response(
            "session_context_action",
            model=llm,
            result=result,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            response_text=result.model_dump(mode="json"),
        )
        if result.action_type == "none" and not result.thread_switch:
            heuristic = _heuristic_session_action_decision(text)
            if heuristic.action_type != "none" or heuristic.thread_switch:
                return heuristic
        return result
    except Exception as exc:
        log_llm_error(
            "session_context_action",
            error=exc,
            model=llm,
            system_prompt=_SESSION_ACTION_PROMPT,
            user_prompt=text,
            latency_ms=None,
        )
        return _heuristic_session_action_decision(text)


def is_explicit_thread_switch_prompt(prompt: str) -> bool:
    return _extract_session_action_decision(prompt).thread_switch


def _heuristic_session_action_decision(prompt: str) -> _SessionActionDecision:
    text = str(prompt or "").strip()
    if not text:
        return _SessionActionDecision()
    if any(token in text for token in _THREAD_SWITCH_TOKENS):
        return _SessionActionDecision(
            thread_switch=True,
            action_type="none",
            confidence=0.95,
        )
    if any(token in text for token in _PLAN_DELETE_TOKENS) and (
        _extract_plan_reference(text) is not None
        or _has_plan_self_reference(text)
        or _extract_plan_id_from_text(text) is not None
    ):
        return _SessionActionDecision(
            action_type="delete_plan",
            confidence=0.9,
        )
    if any(token in text for token in _GROWTH_STAGE_TOKENS) and (
        _extract_plan_reference(text) is not None
        or _has_plan_self_reference(text)
        or _extract_plan_id_from_text(text) is not None
    ):
        return _SessionActionDecision(
            action_type="query_growth_stage",
            confidence=0.88,
        )
    if any(token in text for token in _PLAN_TASK_TOKENS):
        task_overrides = extract_field_overrides(
            text,
            ("name", "task_type", "date", "operator", "work_desc"),
        )
        if task_overrides or _extract_plan_reference(text) is not None or _has_plan_self_reference(text):
            return _SessionActionDecision(
                action_type="record_task",
                confidence=0.86,
            )
    return _SessionActionDecision()


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
    if not _is_ambiguous_thread_ownership_prompt(text):
        return None
    standalone_label = _describe_thread_target(standalone_plan)
    contextual_label = _describe_thread_target(contextual_candidate.plan)
    if not standalone_label or not contextual_label:
        return None
    return (
        f"我不确定你是想继续当前的{contextual_label}，还是想改成新的{standalone_label}。"
        "请回复“继续当前任务”或“开启新任务”。"
    )


def _is_ambiguous_thread_ownership_prompt(text: str) -> bool:
    prompt = str(text or "").strip()
    if not prompt:
        return False
    if parse_followup_index(prompt) is not None:
        return False
    if len(prompt) <= 24:
        return True
    return len(prompt) <= 40 and any(token in prompt for token in _AMBIGUOUS_THREAD_TOKENS)


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
    payload = _build_contextual_plan_task_input(
        prompt, context, allow_implicit_current_plan=False
    )
    if payload:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="plant_task_create",
                input=payload,
                reason="session_context:plant_plan_list_active->plant_task_create",
            ),
            confidence=_score_plan_task_contextual_candidate(prompt, payload),
            kind="tool",
            name="plant_plan_list_active",
            evidence=_collect_plan_task_candidate_evidence(prompt, payload),
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
    payload = _build_contextual_plan_task_input(
        prompt, context, allow_implicit_current_plan=True
    )
    if payload:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="plant_task_create",
                input=payload,
                reason="session_context:growth_stage_lookup->plant_task_create",
            ),
            confidence=_score_plan_task_contextual_candidate(prompt, payload),
            kind="tool",
            name="growth_stage_lookup",
            evidence=_collect_plan_task_candidate_evidence(prompt, payload),
        )
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
    payload = _build_contextual_plan_task_input(
        prompt, context, allow_implicit_current_plan=True
    )
    if payload:
        return ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="plant_task_create",
                input=payload,
                reason="session_context:crop_calendar_workflow->plant_task_create",
            ),
            confidence=_score_plan_task_contextual_candidate(prompt, payload),
            kind="workflow",
            name="crop_calendar_workflow",
            evidence=_collect_plan_task_candidate_evidence(prompt, payload),
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


def _score_weather_contextual_candidate(
    prompt: str, payload: Mapping[str, object]
) -> float:
    score = 0.55
    if payload.get("region") not in (None, ""):
        score += 0.15
    if payload.get("requested_operations"):
        score += 0.15
    if _prompt_has_temporal_signal(prompt):
        score += 0.15
    return min(score, 0.98)


def _collect_weather_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["weather_context"]
    if payload.get("region") not in (None, ""):
        evidence.append("region")
    if payload.get("requested_operations"):
        evidence.append("requested_operations")
    if payload.get("start_date") and payload.get("end_date"):
        evidence.append("date_range")
    return tuple(evidence)


def _score_variety_contextual_candidate(prompt: str) -> float:
    score = 0.55
    if _extract_region_hint(prompt):
        score += 0.2
    if _find_exact_variety(prompt):
        score += 0.15
    if _has_explicit_variety_followup(prompt):
        score += 0.1
    return min(score, 0.95)


def _collect_variety_candidate_evidence(prompt: str) -> tuple[str, ...]:
    evidence: list[str] = ["variety_context"]
    if _extract_region_hint(prompt):
        evidence.append("region")
    if _find_exact_variety(prompt):
        evidence.append("variety")
    if _has_explicit_variety_followup(prompt):
        evidence.append("attribute_request")
    return tuple(evidence)


def _score_sowing_contextual_candidate(
    prompt: str, payload: Mapping[str, object]
) -> float:
    score = 0.55
    if any(payload.get(key) not in (None, "") for key in ("region_id", "farm_id")):
        score += 0.15
    if payload.get("variety") not in (None, ""):
        score += 0.15
    if payload.get("culti_type") not in (None, ""):
        score += 0.1
    if payload.get("planting_method") not in (None, ""):
        score += 0.1
    return min(score, 0.95)


def _collect_sowing_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["sowing_context"]
    if any(payload.get(key) not in (None, "") for key in ("region_id", "farm_id")):
        evidence.append("region")
    if payload.get("variety") not in (None, ""):
        evidence.append("variety")
    if payload.get("culti_type") not in (None, ""):
        evidence.append("culti_type")
    if payload.get("planting_method") not in (None, ""):
        evidence.append("planting_method")
    return tuple(evidence)


def _score_plan_task_contextual_candidate(
    prompt: str, payload: Mapping[str, object]
) -> float:
    score = 0.7
    text = str(prompt or "").strip()
    if _has_explicit_plan_task_action(text):
        score += 0.12
    if _extract_plan_reference(text) is not None or _has_plan_self_reference(text):
        score += 0.08
    followup = payload.get("followup")
    if isinstance(followup, Mapping):
        draft = followup.get("draft")
        if isinstance(draft, Mapping) and draft.get("plan_id") not in (None, ""):
            score += 0.05
        if isinstance(draft, Mapping) and any(
            draft.get(key) not in (None, "", [])
            for key in ("name", "date", "task_type", "operator", "work_desc")
        ):
            score += 0.08
    return min(score, 0.97)


def _collect_plan_task_candidate_evidence(
    prompt: str, payload: Mapping[str, object]
) -> tuple[str, ...]:
    evidence: list[str] = ["plan_context", "task_record_action"]
    if _has_explicit_plan_task_action(prompt):
        evidence.append("explicit_action")
    if _extract_plan_reference(prompt) is not None or _has_plan_self_reference(prompt):
        evidence.append("plan_reference")
    followup = payload.get("followup")
    if isinstance(followup, Mapping):
        draft = followup.get("draft")
        if isinstance(draft, Mapping) and draft.get("plan_id") not in (None, ""):
            evidence.append("plan_id")
        if isinstance(draft, Mapping):
            for key in ("name", "date", "task_type", "operator", "work_desc"):
                if draft.get(key) not in (None, "", []):
                    evidence.append(key)
    return tuple(evidence)


def _score_plan_action_contextual_candidate(prompt: str) -> float:
    score = 0.65
    text = str(prompt or "").strip()
    if _extract_plan_reference(text) is not None or _has_plan_self_reference(text):
        score += 0.15
    if _has_explicit_plan_delete_action(text) or _has_explicit_growth_stage_action(text):
        score += 0.1
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
    score = 0.62
    overrides = _extract_planting_overrides(prompt)
    if overrides:
        score += min(0.28, 0.08 * len(overrides))
    return min(score, 0.95)


def _collect_workflow_candidate_evidence(prompt: str) -> tuple[str, ...]:
    evidence: list[str] = ["workflow_context"]
    overrides = _extract_planting_overrides(prompt)
    evidence.extend(sorted(str(key) for key in overrides.keys()))
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


def _extract_plan_task_context(tool: object) -> Optional[dict[str, object]]:
    if not isinstance(tool, ToolInvocation):
        return None
    data = tool.data or {}
    plan_id = data.get("plant_season_id")
    if plan_id in (None, ""):
        return None
    context: dict[str, object] = {"plan_id": str(plan_id).strip()}
    request = data.get("request")
    if isinstance(request, Mapping):
        for key in ("name", "task_type", "date", "is_completed"):
            value = request.get(key)
            if value in (None, ""):
                continue
            context[key] = value
    return context


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
    yielded: set[tuple[str, str]] = set()
    last = session_payload.get(_LAST_CONTEXT_KEY)
    if isinstance(last, Mapping):
        kind = str(last.get("kind") or "").strip()
        name = str(last.get("name") or "").strip()
        adapter = get_session_context_adapter(kind, name)
        context = _get_context(session_payload, kind, name)
        if adapter is not None and context:
            yielded.add((kind, name))
            yield adapter, context
    for adapter in _SESSION_CONTEXT_ADAPTERS:
        key = (adapter.kind, adapter.name)
        if key in yielded:
            continue
        context = _get_context(session_payload, adapter.kind, adapter.name)
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
        name="plant_task_create",
        task_type="plan_task",
        updatable_fields=(
            "plant_season_id",
            "name",
            "date",
            "is_completed",
            "task_type",
            "work_desc",
        ),
        extract_context=_extract_plan_task_context,
        build_candidate=lambda prompt, context: (
            ContextualPlanCandidate(
                plan=ActionPlan(
                    action="tool",
                    name="plant_task_create",
                    input=payload,
                    reason="session_context:plant_task_create",
                ),
                confidence=_score_plan_task_contextual_candidate(prompt, payload),
                kind="tool",
                name="plant_task_create",
                evidence=_collect_plan_task_candidate_evidence(prompt, payload),
            )
            if (
                payload := _build_contextual_plan_task_input(
                    prompt, context, allow_implicit_current_plan=True
                )
            )
            else None
        ),
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
    overrides = extract_field_overrides(
        prompt,
        ("region", "start_date", "end_date", "granularity", "include_advice", "requested_operations"),
    )
    text = str(prompt or "").strip()
    if _should_block_weather_contextual_candidate(text, overrides, supported_ops):
        return None
    has_date_override = bool(
        overrides.get("start_date") not in (None, "")
        and overrides.get("end_date") not in (None, "")
    )
    has_supported_operation = bool(overrides.get("requested_operations"))
    has_region_override = overrides.get("region") not in (None, "")
    year = _extract_year(prompt)
    if (
        year is not None
        and not has_date_override
        and "start_date" in base
        and "end_date" in base
    ):
        start = _parse_iso_date(base.get("start_date"))
        end = _parse_iso_date(base.get("end_date"))
        if start and end:
            overrides["start_date"] = start.replace(year=year).isoformat()
            overrides["end_date"] = end.replace(year=year).isoformat()
            has_date_override = True
    if unsupported_ops and not supported_ops and not has_date_override:
        return None
    has_structured_weather_evidence = (
        has_date_override
        or has_supported_operation
        or (
            has_region_override
            and (
                (
                    base.get("start_date") not in (None, "")
                    and base.get("end_date") not in (None, "")
                )
                or bool(base.get("requested_operations"))
            )
        )
    )
    if not overrides and has_structured_weather_evidence:
        return dict(base)
    if not overrides and (supported_ops or unsupported_ops):
        if unsupported_ops and not supported_ops:
            return None
        return dict(base)
    if not overrides or not has_structured_weather_evidence:
        return None
    merged = dict(base)
    merged.update(overrides)
    if not (merged.get("start_date") and merged.get("end_date")):
        return None
    return merged


def _should_block_weather_contextual_candidate(
    prompt: str,
    overrides: Mapping[str, object],
    supported_ops: list[str],
) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if any(
        token in text
        for token in ("播种", "播期", "适播", "建立", "生成", "方案")
    ):
        weather_keywords = ("天气", "气象", "降雨", "降水", "温度", "预报")
        if not any(keyword in text for keyword in weather_keywords):
            return True
    if not supported_ops and not overrides.get("requested_operations"):
        if "适合" in text and "施肥" not in text and "移栽" not in text and "打药" not in text:
            return True
    return False


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

def _build_contextual_variety_query(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not isinstance(context, Mapping):
        return None
    variety = str(context.get("variety") or "").strip()
    if not variety:
        return None
    overrides = extract_field_overrides(
        prompt,
        ("variety", "region_choice"),
        variety_matcher=_find_exact_variety if _should_try_variety_match(prompt) else None,
    )
    region = str(
        overrides.get("region_choice")
        or overrides.get("region_id")
        or context.get("region_choice")
        or ""
    ).strip()
    explicit_variety = str(overrides.get("variety") or variety).strip() or variety
    if not _should_resume_variety(
        prompt,
        region_changed=region != str(context.get("region_choice") or "").strip(),
        explicit_variety=explicit_variety != variety,
    ):
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
    if not _has_explicit_growth_stage_action(prompt):
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
    if not _has_explicit_plan_delete_action(prompt):
        return None
    plan_id = _resolve_plan_id_from_context(prompt, context)
    if not plan_id:
        return None
    return {"plant_season_id": plan_id}


def _build_contextual_plan_task_input(
    prompt: str,
    context: Optional[Mapping[str, object]],
    *,
    allow_implicit_current_plan: bool,
) -> Optional[dict[str, object]]:
    text = str(prompt or "").strip()
    if not isinstance(context, Mapping):
        return None
    draft = _extract_plan_task_followup_draft(text)
    if not draft:
        return None
    plan_id = _resolve_plan_id_from_context(text, context)
    if not plan_id and allow_implicit_current_plan:
        current_plan_id = context.get("plant_season_id") or context.get("plan_id")
        if current_plan_id not in (None, ""):
            plan_id = str(current_plan_id).strip()
    if not plan_id:
        return None
    draft["plan_id"] = plan_id
    plan_name = str(context.get("plan_name") or "").strip()
    if plan_name:
        draft["plan_name"] = plan_name
    return {
        "query": text,
        "followup": {
            "draft": draft,
            "missing_fields": [],
            "followup_count": 0,
        },
    }


def _extract_planting_overrides(prompt: str) -> dict[str, object]:
    text = str(prompt or "").strip()
    if not text:
        return {}
    include_variety = _should_try_variety_match(text)
    return extract_field_overrides(
        text,
        ("region_id", "culti_type", "planting_method", "sowing_date", "transplant_date", "variety"),
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
    if plan.action == "tool" and plan.name == "plant_task_create":
        return "农事新增/记录"
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


def _should_resume_variety(
    prompt: str, *, region_changed: bool, explicit_variety: bool
) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if region_changed:
        return True
    if explicit_variety:
        return True
    if _has_explicit_variety_followup(text):
        return True
    return False


def _has_explicit_variety_followup(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if _find_exact_variety(text):
        return True
    if _has_variety_attribute_followup(text):
        return True
    return False


def _has_variety_attribute_followup(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    for aliases in _VARIETY_SUPPORTED_ATTRIBUTE_ALIASES.values():
        if any(alias in text for alias in aliases):
            return True
    return any(alias in text for alias in _VARIETY_UNSUPPORTED_ATTRIBUTE_ALIASES)


def _has_explicit_plan_delete_action(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    decision = _extract_session_action_decision(text)
    if decision.action_type != "delete_plan":
        return False
    return (
        _extract_plan_reference(text) is not None
        or _has_plan_self_reference(text)
        or _extract_plan_id_from_text(text) is not None
    )


def _has_explicit_plan_task_action(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    decision = _extract_session_action_decision(text)
    if decision.action_type != "record_task":
        return False
    task_overrides = extract_field_overrides(
        text,
        ("name", "task_type", "date", "operator", "work_desc"),
    )
    if task_overrides:
        return True
    if (
        _extract_plan_reference(text) is not None
        or _has_plan_self_reference(text)
        or extract_field_overrides(text, ("plan_id",)).get("plan_id") not in (None, "")
    ):
        return True
    return False


def _has_explicit_growth_stage_action(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    decision = _extract_session_action_decision(text)
    if decision.action_type != "query_growth_stage":
        return False
    return (
        _extract_plan_reference(text) is not None
        or _has_plan_self_reference(text)
        or _extract_plan_id_from_text(text) is not None
    )


def _extract_plan_task_followup_draft(prompt: str) -> dict[str, object]:
    text = str(prompt or "").strip()
    if not _has_explicit_plan_task_action(text):
        return {}
    draft = extract_field_overrides(
        text,
        ("name", "task_type", "date", "operator", "work_desc"),
    )
    return draft or {"_explicit_action": True}


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
