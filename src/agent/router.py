import contextvars
from pathlib import Path
from typing import Optional

from .followup import (
    get_followup_missing_fields,
    get_followup_kind,
    get_followup_options,
    resolve_followup_choice,
)
from .session_context import (
    build_thread_ownership_clarification,
    build_contextual_candidate,
    extract_session_context_from_tool,
    extract_session_context_from_workflow,
    is_explicit_thread_switch_prompt,
    resolve_session_plan,
    should_short_circuit_contextual_candidate,
)
from ..infra.config import get_config
from ..infra.pending_store import build_pending_followup_store
from ..infra.session_context_store import build_session_context_store
from ..observability.logging_utils import log_event
from ..observability.interaction_context import get_interaction_context, update_interaction_context
from ..observability.otel import annotate_current_span, build_span_attributes
from ..schemas.models import HandleResponse, UserRequest, WorkflowResponse
from .fast_intent import FastIntentRouter
from .intent_router import IntentRouter
from .intent_rules import IntentRuleEngine
from .plan_executor import PlanExecutor
from .planner import ActionPlan, PlannerRunner
from .pending_manager import PendingManager
from .tools.registry import execute_tool, list_tool_specs
from .workflows.registry import get_workflow_spec, list_workflow_specs


DEFAULT_NONE_MESSAGE = "未识别到与农事相关的需求。"
_YES_WORDS = {"是", "好", "好的", "确认", "需要", "要", "ok", "yes", "y"}
_NO_WORDS = {"否", "不", "不用", "不需要", "取消", "算了", "no", "n"}


class RequestRouter:
    """Route requests with a Planner+Executor workflow."""

    _session_id_ctx = contextvars.ContextVar("session_id", default="default")
    _memory_id_ctx = contextvars.ContextVar("memory_id", default="default")

    def __init__(self):
        self._workflow_specs = list_workflow_specs()
        self._workflow_names = {spec.name for spec in self._workflow_specs}
        tool_specs = list_tool_specs()
        self._tool_names = {spec["name"] for spec in tool_specs}
        self._planner = PlannerRunner(tool_specs, self._workflow_specs)
        self._pending_store = build_pending_followup_store()
        self._session_context_store = build_session_context_store()
        cfg = get_config()

        mode = (cfg.intent_routing_mode or "hybrid").strip().lower()
        if mode not in {"llm_only", "hybrid"}:
            log_event("intent_mode_invalid", value=mode)
            mode = "hybrid"
        self._intent_mode = mode

        if cfg.intent_rules_path:
            rules_path = Path(cfg.intent_rules_path)
        else:
            rules_path = (
                Path(__file__).resolve().parents[2] / "resources" / "intent_rules.json"
            )
        self._rule_engine = IntentRuleEngine(
            rules_path, reload_seconds=cfg.intent_rules_reload_seconds
        )
        try:
            self._fast_router: Optional[FastIntentRouter] = FastIntentRouter(
                tool_specs, self._workflow_specs
            )
        except Exception as exc:
            log_event("fast_intent_init_error", error=str(exc))
            self._fast_router = None

        self._pending_manager = PendingManager(self._pending_store, self._rule_engine)
        self._intent_router = IntentRouter(
            tool_names=self._tool_names,
            workflow_names=self._workflow_names,
            planner=self._planner,
            rule_engine=self._rule_engine,
            fast_router=self._fast_router,
            cache_max_items=cfg.tool_cache_max_items,
            cache_ttl_seconds=cfg.tool_cache_ttl_seconds,
        )
        self._plan_executor = PlanExecutor(
            tool_names=self._tool_names,
            workflow_names=self._workflow_names,
            pending_manager=self._pending_manager,
            get_workflow_spec_fn=get_workflow_spec,
        )

    def _get_session_context_payload(self, session_id: str) -> dict:
        payload = self._session_context_store.get(session_id)
        return dict(payload) if isinstance(payload, dict) else {}

    def _set_session_named_context(
        self, session_id: str, *, kind: str, name: str, context: dict
    ) -> None:
        payload = self._get_session_context_payload(session_id)
        tool_contexts = dict(payload.get("tool_contexts") or {})
        workflow_contexts = dict(payload.get("workflow_contexts") or {})
        if kind == "tool":
            tool_contexts[name] = dict(context)
        elif kind == "workflow":
            workflow_contexts[name] = dict(context)
        payload.update(
            {
                "tool_contexts": tool_contexts,
                "workflow_contexts": workflow_contexts,
                "last_context": {"kind": kind, "name": name},
            }
        )
        self._session_context_store.set(session_id, payload)

    def _clear_session_tool_contexts(self, session_id: str) -> None:
        self._session_context_store.delete(session_id)

    def _get_contextual_candidate(self, prompt: str, session_id: str):
        payload = self._get_session_context_payload(session_id)
        return build_contextual_candidate(prompt, payload)

    def handle(self, request: UserRequest) -> HandleResponse:
        session_id = request.session_id or request.user_id or "default"
        memory_id = request.user_id or session_id
        session_token = self._session_id_ctx.set(session_id)
        memory_token = self._memory_id_ctx.set(memory_id)
        prompt = (request.prompt or "").strip()
        try:
            if not prompt:
                self._mark_interaction_lineage(
                    continuity_type="standalone",
                    continuity_source="none",
                    continue_thread=False,
                    dialogue_act="empty_input",
                    task_type="none",
                )
                plan = WorkflowResponse(message=DEFAULT_NONE_MESSAGE)
                return HandleResponse(mode="none", plan=plan)
            pending = self._pending_manager.get(session_id)
            if self._pending_manager.should_resume_pending(prompt, pending):
                if pending and pending.get("mode") == "clarification":
                    self._mark_interaction_lineage(
                        continuity_type="pending_resume",
                        continuity_source="pending",
                        continue_thread=False,
                        dialogue_act="select_option",
                        task_type="clarification",
                    )
                    return self._resume_clarification_pending(
                        prompt, pending, session_id
                    )
                self._mark_interaction_lineage(
                    continuity_type="pending_resume",
                    continuity_source="pending",
                    continue_thread=True,
                    dialogue_act=self._infer_pending_dialogue_act(prompt, pending),
                    task_type=self._task_type_from_pending(pending),
                )
                log_event(
                    "pending_resume",
                    mode=pending.get("mode") if pending else None,
                    reason="auto",
                )
                return self._resume_pending(prompt, pending, session_id)
            if pending:
                # New question: clear stale pending to avoid misrouting follow-ups.
                self._pending_manager.delete(session_id)
                pending = None
            contextual_candidate = self._get_contextual_candidate(prompt, session_id)
            if contextual_candidate:
                contextual_summary = {
                    "action": contextual_candidate.plan.action,
                    "name": contextual_candidate.plan.name,
                    "reason": contextual_candidate.plan.reason,
                    "confidence": contextual_candidate.confidence,
                    "evidence": list(contextual_candidate.evidence),
                    "adapter_task_type": contextual_candidate.adapter_task_type,
                    "updatable_fields": list(contextual_candidate.updatable_fields),
                }
                log_event(
                    "session_candidate_built",
                    **contextual_summary,
                )
                annotate_current_span(
                    build_span_attributes(
                        "session.contextual_candidate", contextual_summary
                    )
                )
            if contextual_candidate and not is_explicit_thread_switch_prompt(prompt) and should_short_circuit_contextual_candidate(contextual_candidate):
                self._mark_interaction_lineage(
                    continuity_type="session_context_resume",
                    continuity_source="session_context",
                    continue_thread=True,
                    dialogue_act="update_fields",
                    task_type=self._task_type_from_plan(contextual_candidate.plan),
                )
                selection_summary = {
                    "selected": "contextual",
                    "strategy": "short_circuit",
                    "action": contextual_candidate.plan.action,
                    "name": contextual_candidate.plan.name,
                }
                log_event(
                    "session_candidate_selected",
                    **selection_summary,
                )
                annotate_current_span(
                    build_span_attributes("session.resolution", selection_summary)
                )
                return self._execute_with_validation(
                    contextual_candidate.plan, prompt, pending, session_id
                )
            standalone_plan = self._intent_router.plan(
                prompt,
                pending=pending,
                intent_mode=self._intent_mode,
            )
            if standalone_plan:
                standalone_summary = {
                    "action": standalone_plan.action,
                    "name": standalone_plan.name,
                    "reason": standalone_plan.reason,
                }
                annotate_current_span(
                    build_span_attributes("session.standalone_plan", standalone_summary)
                )
            clarification = build_thread_ownership_clarification(
                prompt,
                contextual_candidate,
                standalone_plan,
            )
            if clarification:
                self._pending_manager.set(
                    session_id,
                    {
                        "mode": "clarification",
                        "pending_kind": "clarification",
                        "options": ["继续当前任务", "开启新任务"],
                        "pending_message": clarification,
                        "contextual_plan": contextual_candidate.plan.model_dump(
                            mode="json"
                        )
                        if contextual_candidate
                        else None,
                        "standalone_plan": standalone_plan.model_dump(mode="json")
                        if standalone_plan
                        else None,
                    },
                )
                self._mark_interaction_lineage(
                    continuity_type="standalone",
                    continuity_source="none",
                    continue_thread=False,
                    dialogue_act="continue_task",
                    task_type="clarification",
                )
                return HandleResponse(
                    mode="none",
                    plan=WorkflowResponse(message=clarification),
                )
            plan = resolve_session_plan(standalone_plan, contextual_candidate)
            if plan:
                selected = "contextual" if contextual_candidate and plan is contextual_candidate.plan else "standalone"
                resolution_summary = {
                    "selected": selected,
                    "strategy": "resolved",
                    "action": plan.action,
                    "name": plan.name,
                    "reason": plan.reason,
                }
                annotate_current_span(
                    build_span_attributes("session.resolution", resolution_summary)
                )
            if contextual_candidate and plan is contextual_candidate.plan:
                self._mark_interaction_lineage(
                    continuity_type="session_context_resume",
                    continuity_source="session_context",
                    continue_thread=True,
                    dialogue_act="update_fields",
                    task_type=self._task_type_from_plan(plan),
                )
                log_event(
                    "session_candidate_selected",
                    selected="contextual",
                    strategy="resolved",
                    action=plan.action,
                    name=plan.name,
                )
            elif standalone_plan:
                self._mark_interaction_lineage(
                    continuity_type="standalone",
                    continuity_source="none",
                    continue_thread=False,
                    dialogue_act="start_new_task",
                    task_type=self._task_type_from_plan(standalone_plan),
                )
                log_event(
                    "session_candidate_selected",
                    selected="standalone",
                    strategy="resolved",
                    action=standalone_plan.action,
                    name=standalone_plan.name,
                )
            if not plan:
                self._mark_interaction_lineage(
                    continuity_type="standalone",
                    continuity_source="none",
                    continue_thread=False,
                    dialogue_act="start_new_task",
                    task_type="none",
                )
                return self._fallback_from_planner(prompt, pending, session_id)
            return self._execute_with_validation(plan, prompt, pending, session_id)
        finally:
            self._memory_id_ctx.reset(memory_token)
            self._session_id_ctx.reset(session_token)

    def _mark_interaction_lineage(
        self,
        *,
        continuity_type: str,
        continuity_source: str,
        continue_thread: bool,
        dialogue_act: str,
        task_type: str,
    ) -> None:
        ctx = get_interaction_context()
        request_id = str(ctx.get("request_id") or "").strip()
        previous_interaction_id = ctx.get("previous_interaction_id")
        previous_thread_id = str(ctx.get("previous_thread_id") or "").strip()
        thread_id = request_id
        parent_interaction_id = None
        if continue_thread and previous_interaction_id:
            parent_interaction_id = previous_interaction_id
            thread_id = previous_thread_id or request_id
        update_interaction_context(
            thread_id=thread_id,
            parent_interaction_id=parent_interaction_id,
            continuity_type=continuity_type,
            continuity_source=continuity_source,
            dialogue_act=dialogue_act,
            task_type=task_type,
        )

    def _task_type_from_plan(self, plan: Optional[ActionPlan]) -> str:
        if not plan:
            return "none"
        if plan.action == "none":
            return "none"
        if plan.name:
            return str(plan.name)
        return str(plan.action)

    def _task_type_from_pending(self, pending: Optional[dict]) -> str:
        if not isinstance(pending, dict):
            return "none"
        name = (
            pending.get("tool_name")
            or pending.get("workflow_name")
            or pending.get("name")
            or pending.get("mode")
        )
        return str(name or "none")

    def _infer_pending_dialogue_act(self, prompt: str, pending: Optional[dict]) -> str:
        text = (prompt or "").strip()
        if not isinstance(pending, dict):
            return "continue_task"
        missing_fields = get_followup_missing_fields(pending)
        if get_followup_kind(pending) == "confirmation":
            if text in _YES_WORDS:
                return "confirm"
            if text in _NO_WORDS:
                return "cancel"
            return "confirm"
        options = get_followup_options(pending)
        if options and resolve_followup_choice(text, options):
            return "select_option"
        if missing_fields:
            return "update_fields"
        return "continue_task"

    def _execute_with_validation(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        plan, exec_pending, response = self._plan_executor.apply_input_validation(
            plan, pending, session_id
        )
        if response:
            return response
        return self._execute_plan(plan, prompt, exec_pending, session_id)

    def _execute_plan(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        if plan.action == "tool":
            return self._execute_tool_plan(plan, prompt, pending, session_id)
        if plan.action == "workflow":
            return self._execute_workflow_plan(plan, prompt, pending, session_id)
        if pending and not plan.response:
            log_event(
                "planner_fallback",
                reason="none_action_with_pending",
                pending_mode=pending.get("mode"),
            )
            return self._resume_pending(prompt, pending, session_id)
        return self._respond_none(plan, pending, session_id)

    def _execute_tool_plan(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        response = self._plan_executor.execute_tool_plan(
            plan,
            prompt=prompt,
            pending=pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            execute_tool_fn=execute_tool,
        )
        self._update_session_context_from_response(
            response, session_id=session_id, pending=pending, plan=plan
        )
        return response

    def _execute_workflow_plan(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        response = self._plan_executor.execute_workflow_plan(
            plan,
            prompt=prompt,
            pending=pending,
            session_id=session_id,
            run_named_workflow=self._run_named_workflow,
        )
        self._update_session_context_from_response(
            response, session_id=session_id, pending=pending, plan=plan
        )
        return response

    def _respond_none(
        self,
        plan: ActionPlan,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        return self._plan_executor.respond_none(
            plan,
            pending=pending,
            session_id=session_id,
        )

    def _fallback_from_planner(
        self, prompt: str, pending: Optional[dict], session_id: str
    ) -> HandleResponse:
        response = self._plan_executor.fallback_from_planner(
            prompt,
            pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            run_named_workflow=self._run_named_workflow,
            execute_tool_fn=execute_tool,
        )
        self._update_session_context_from_response(
            response, session_id=session_id, pending=pending
        )
        return response

    def _resume_pending(
        self, prompt: str, pending: dict, session_id: str
    ) -> HandleResponse:
        response = self._plan_executor.resume_pending(
            prompt,
            pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            run_named_workflow=self._run_named_workflow,
            execute_tool_fn=execute_tool,
        )
        self._update_session_context_from_response(
            response, session_id=session_id, pending=pending
        )
        return response

    def _resume_clarification_pending(
        self, prompt: str, pending: dict, session_id: str
    ) -> HandleResponse:
        choice = self._pending_manager.resolve_pending_clarification_choice(
            prompt, pending
        )
        if choice == "contextual":
            raw_plan = pending.get("contextual_plan") or {}
        else:
            raw_plan = pending.get("standalone_plan") or {}
        self._pending_manager.delete(session_id)
        plan = ActionPlan.model_validate(raw_plan)
        return self._execute_with_validation(plan, prompt, None, session_id)

    def _update_session_context_from_response(
        self,
        response: HandleResponse,
        *,
        session_id: str,
        pending: Optional[dict] = None,
        plan: Optional[ActionPlan] = None,
    ) -> None:
        if response.mode == "tool":
            tool = response.tool
            if tool and tool.name == "memory_clear":
                self._clear_session_tool_contexts(session_id)
            session_context = extract_session_context_from_tool(tool)
            if session_context:
                name, context = session_context
                self._set_session_named_context(
                    session_id, kind="tool", name=name, context=context
                )
            return None
        if response.mode != "workflow":
            return None
        workflow_name = plan.name if plan else None
        if not workflow_name and pending and pending.get("mode") == "workflow":
            workflow_name = pending.get("workflow_name")
        if not workflow_name:
            return None
        session_context = extract_session_context_from_workflow(
            workflow_name, response.plan
        )
        if session_context:
            name, context = session_context
            self._set_session_named_context(
                session_id, kind="workflow", name=name, context=context
            )
        return None

    def _run_named_workflow(
        self, prompt: str, workflow_name: Optional[str]
    ) -> WorkflowResponse:
        return self._plan_executor.run_named_workflow(
            prompt,
            workflow_name,
            session_id=self._session_id_ctx.get(),
            memory_id=self._memory_id_ctx.get(),
        )
