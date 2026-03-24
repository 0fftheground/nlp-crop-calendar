from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Callable, Dict

from ..agent.pending_manager import PendingManager
from ..agent.planner import PlannerRunner
from ..agent.followup import summarize_pending
from ..agent.workflows.common import coerce_planting_draft
from ..agent.workflows.crop_calendar_graph import _extract_node as crop_calendar_extract_node
from ..agent.session_context import build_contextual_candidate
from ..agent.tools.registry import list_tool_specs
from ..agent.workflows.registry import list_workflow_specs
from ..application.services.variety_service import _llm_choose_variety_record
from ..domain.planting_models import PlantingDetailsDraft
from ..infra.llm import get_chat_model, get_extractor_model
from ..infra.llm_extract import llm_structured_extract
from ..prompts.planting_extract import build_planting_extract_prompt
from ..prompts.variety_match import VARIETY_MATCH_SYSTEM_PROMPT
from .common import (
    estimate_message_tokens,
    estimate_tokens,
    get_model_name,
    pick_token_estimate,
    to_jsonable,
)


_ACTUAL_KEY = "__eval_actual__"
_METRICS_KEY = "__eval_metrics__"


def _finalize_metrics(
    *,
    latency_ms: float,
    model_name: str | None = None,
    estimated_input_tokens: int | None = None,
    estimated_output_tokens: int | None = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "latency_ms": round(float(latency_ms), 2),
    }
    if model_name:
        metrics["model"] = model_name
    if estimated_input_tokens is not None:
        metrics["estimated_input_tokens"] = int(estimated_input_tokens)
    if estimated_output_tokens is not None:
        metrics["estimated_output_tokens"] = int(estimated_output_tokens)
    if estimated_input_tokens is not None or estimated_output_tokens is not None:
        metrics["estimated_total_tokens"] = int(estimated_input_tokens or 0) + int(
            estimated_output_tokens or 0
        )
    return metrics


def _pack_eval_result(actual: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        _ACTUAL_KEY: to_jsonable(actual),
        _METRICS_KEY: to_jsonable(metrics),
    }


def run_planner_case(case: Dict[str, Any]) -> Dict[str, Any]:
    planner = PlannerRunner(list_tool_specs(), list_workflow_specs())
    pending = case.get("input", {}).get("pending")
    prompt = str(case.get("input", {}).get("prompt") or "").strip()
    user_payload = json.dumps(
        {
            "prompt": prompt,
            "pending": summarize_pending(pending),
        },
        ensure_ascii=False,
        default=str,
    )
    estimated_input_tokens = pick_token_estimate(
        estimate_message_tokens(
            planner._llm,
            system_prompt=planner._system_prompt,
            user_prompt=user_payload,
        ),
        (estimate_tokens(planner._llm, planner._system_prompt) or 0)
        + (estimate_tokens(planner._llm, user_payload) or 0),
    )
    started_at = perf_counter()
    result = planner.plan(prompt, pending=pending)
    elapsed_ms = (perf_counter() - started_at) * 1000
    if result is None:
        actual = {"action": "none"}
    else:
        actual = to_jsonable(result.model_dump(mode="json", exclude_none=True))
    metrics = _finalize_metrics(
        latency_ms=elapsed_ms,
        model_name=get_model_name(planner._llm),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimate_tokens(planner._llm, actual),
    )
    return _pack_eval_result(actual, metrics)


def run_extractor_case(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("input", {})
    prompt = str(payload.get("prompt") or "").strip()
    hint = str(payload.get("hint") or "")
    system_prompt = build_planting_extract_prompt(hint)
    try:
        model = get_extractor_model()
    except Exception:
        model = None
    estimated_input_tokens = pick_token_estimate(
        estimate_message_tokens(
            model,
            system_prompt=system_prompt,
            user_prompt=prompt,
        ),
        (estimate_tokens(model, system_prompt) or 0) + (estimate_tokens(model, prompt) or 0),
    )
    started_at = perf_counter()
    result = llm_structured_extract(
        prompt,
        schema=PlantingDetailsDraft,
        system_prompt=system_prompt,
    )
    elapsed_ms = (perf_counter() - started_at) * 1000
    actual = to_jsonable(result)
    metrics = _finalize_metrics(
        latency_ms=elapsed_ms,
        model_name=get_model_name(model),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimate_tokens(model, actual),
    )
    return _pack_eval_result(actual, metrics)


def run_variety_match_case(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("input", {})
    prompt = str(payload.get("prompt") or "").strip()
    candidates = list(payload.get("candidates") or [])
    region_tokens = list(payload.get("region_tokens") or [])
    user_payload = json.dumps(
        {
            "prompt": prompt,
            "region_tokens": region_tokens,
            "candidates": candidates,
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        model = get_chat_model()
    except Exception:
        model = None
    estimated_input_tokens = pick_token_estimate(
        estimate_message_tokens(
            model,
            system_prompt=VARIETY_MATCH_SYSTEM_PROMPT,
            user_prompt=user_payload,
        ),
        (estimate_tokens(model, VARIETY_MATCH_SYSTEM_PROMPT) or 0)
        + (estimate_tokens(model, user_payload) or 0),
    )
    started_at = perf_counter()
    decision = _llm_choose_variety_record(prompt, candidates, region_tokens)
    elapsed_ms = (perf_counter() - started_at) * 1000
    if decision is None:
        actual = {"index": -1}
    else:
        actual = to_jsonable(decision.model_dump(mode="json", exclude_none=True))
    metrics = _finalize_metrics(
        latency_ms=elapsed_ms,
        model_name=get_model_name(model),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimate_tokens(model, actual),
    )
    return _pack_eval_result(actual, metrics)


def run_workflow_extract_case(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("input", {})
    prompt = str(payload.get("prompt") or "").strip()
    prior_draft = payload.get("draft")
    missing_fields = list(payload.get("missing_fields") or [])
    options = list(payload.get("options") or payload.get("pending_options") or [])
    followup_count = int(payload.get("followup_count") or 0)
    pending_message = payload.get("pending_message")
    plant_season_id = payload.get("plant_season_id")
    system_prompt = build_planting_extract_prompt()
    try:
        model = get_extractor_model()
    except Exception:
        model = None
    estimated_input_tokens = pick_token_estimate(
        estimate_message_tokens(
            model,
            system_prompt=system_prompt,
            user_prompt=prompt,
        ),
        (estimate_tokens(model, system_prompt) or 0) + (estimate_tokens(model, prompt) or 0),
    )
    started_at = perf_counter()
    result_state = crop_calendar_extract_node(
        {
            "user_prompt": prompt,
            "draft": prior_draft,
            "missing_fields": missing_fields,
            "followup_count": followup_count,
            "pending_message": pending_message,
            "options": options,
            "plant_season_id": plant_season_id,
            "trace": [],
        }
    )
    elapsed_ms = (perf_counter() - started_at) * 1000
    actual: Dict[str, Any] = {
        "missing_fields": list(result_state.get("missing_fields") or []),
    }
    draft = coerce_planting_draft(result_state.get("draft") or result_state.get("planting_draft"))
    if draft is not None:
        actual["draft"] = to_jsonable(draft)
    if result_state.get("pending_message") not in (None, ""):
        actual["pending_message"] = result_state.get("pending_message")
    if result_state.get("options"):
        actual["options"] = list(result_state.get("options") or [])
    if result_state.get("variety_candidates"):
        actual["variety_candidates"] = list(result_state.get("variety_candidates") or [])
    if result_state.get("assumptions"):
        actual["assumptions"] = list(result_state.get("assumptions") or [])
    if "halt" in result_state:
        actual["halt"] = bool(result_state.get("halt"))
    if result_state.get("message"):
        actual["message"] = str(result_state.get("message") or "")
    metrics = _finalize_metrics(
        latency_ms=elapsed_ms,
        model_name=get_model_name(model),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimate_tokens(model, actual),
    )
    return _pack_eval_result(actual, metrics)


class _NoopRuleEngine:
    def match(self, _prompt: str):
        return None


class _MemoryStore:
    def __init__(self, payload: Dict[str, Any] | None = None) -> None:
        self._payload = dict(payload or {})

    def get(self, _session_id: str) -> Dict[str, Any] | None:
        return dict(self._payload) if self._payload else None

    def set(self, _session_id: str, payload: dict) -> None:
        self._payload = dict(payload)

    def delete(self, _session_id: str) -> None:
        self._payload = {}


def run_session_context_case(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("input", {})
    prompt = str(payload.get("prompt") or "").strip()
    session_context = payload.get("session_context")
    started_at = perf_counter()
    candidate = build_contextual_candidate(prompt, session_context)
    elapsed_ms = (perf_counter() - started_at) * 1000
    if candidate is None:
        actual = {"matched": False}
    else:
        actual = {
            "matched": True,
            "action": candidate.plan.action,
            "name": candidate.plan.name,
            "input": to_jsonable(candidate.plan.input),
            "reason": candidate.plan.reason,
            "confidence": round(float(candidate.confidence), 4),
            "evidence": list(candidate.evidence),
        }
    return _pack_eval_result(actual, _finalize_metrics(latency_ms=elapsed_ms))


def run_followup_resume_case(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("input", {})
    prompt = str(payload.get("prompt") or "").strip()
    session_id = str(payload.get("session_id") or "eval-session")
    pending = dict(payload.get("pending") or {})
    manager = PendingManager(_MemoryStore(pending), _NoopRuleEngine())
    started_at = perf_counter()
    should_resume = manager.should_resume_pending(prompt, pending)
    result: Dict[str, Any] = {
        "should_resume": should_resume,
        "mode": pending.get("mode"),
    }
    if not should_resume:
        elapsed_ms = (perf_counter() - started_at) * 1000
        return _pack_eval_result(result, _finalize_metrics(latency_ms=elapsed_ms))
    if pending.get("mode") == "tool":
        followup_prompt = manager.build_tool_followup_prompt(
            prompt=prompt,
            pending=pending,
            memory_id=str(payload.get("memory_id") or "eval-user"),
        )
        result["tool_name"] = pending.get("tool_name")
        result["followup_payload"] = to_jsonable(json.loads(followup_prompt))
        elapsed_ms = (perf_counter() - started_at) * 1000
        return _pack_eval_result(result, _finalize_metrics(latency_ms=elapsed_ms))
    if pending.get("mode") == "workflow":
        workflow_name = str(pending.get("workflow_name") or "")
        result["workflow_name"] = workflow_name
        result["resume_state"] = to_jsonable(
            manager.build_workflow_resume_state(pending, workflow_name)
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        return _pack_eval_result(result, _finalize_metrics(latency_ms=elapsed_ms))
    elapsed_ms = (perf_counter() - started_at) * 1000
    return _pack_eval_result(result, _finalize_metrics(latency_ms=elapsed_ms))


TASK_RUNNERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "planner": run_planner_case,
    "extractor": run_extractor_case,
    "variety_match": run_variety_match_case,
    "workflow_extract": run_workflow_extract_case,
    "session_context": run_session_context_case,
    "followup_resume": run_followup_resume_case,
}


def dataset_to_pretty_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
