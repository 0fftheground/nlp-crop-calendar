import contextvars
from pathlib import Path
from typing import Optional

from ..infra.config import get_config
from ..infra.pending_store import build_pending_followup_store
from ..observability.logging_utils import log_event
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

    def handle(self, request: UserRequest) -> HandleResponse:
        session_id = request.session_id or request.user_id or "default"
        memory_id = request.user_id or session_id
        session_token = self._session_id_ctx.set(session_id)
        memory_token = self._memory_id_ctx.set(memory_id)
        prompt = (request.prompt or "").strip()
        try:
            if not prompt:
                plan = WorkflowResponse(message=DEFAULT_NONE_MESSAGE)
                return HandleResponse(mode="none", plan=plan)
            pending = self._pending_manager.get(session_id)
            if self._pending_manager.should_resume_pending(prompt, pending):
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
            plan = self._intent_router.plan(
                prompt,
                pending=pending,
                intent_mode=self._intent_mode,
            )
            if not plan:
                return self._fallback_from_planner(prompt, pending, session_id)
            return self._execute_with_validation(plan, prompt, pending, session_id)
        finally:
            self._memory_id_ctx.reset(memory_token)
            self._session_id_ctx.reset(session_token)

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
        return self._plan_executor.execute_tool_plan(
            plan,
            prompt=prompt,
            pending=pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            execute_tool_fn=execute_tool,
        )

    def _execute_workflow_plan(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        return self._plan_executor.execute_workflow_plan(
            plan,
            prompt=prompt,
            pending=pending,
            session_id=session_id,
            run_named_workflow=self._run_named_workflow,
        )

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
        return self._plan_executor.fallback_from_planner(
            prompt,
            pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            run_named_workflow=self._run_named_workflow,
            execute_tool_fn=execute_tool,
        )

    def _resume_pending(
        self, prompt: str, pending: dict, session_id: str
    ) -> HandleResponse:
        return self._plan_executor.resume_pending(
            prompt,
            pending,
            session_id=session_id,
            memory_id=self._memory_id_ctx.get(),
            run_named_workflow=self._run_named_workflow,
            execute_tool_fn=execute_tool,
        )

    def _run_named_workflow(
        self, prompt: str, workflow_name: Optional[str]
    ) -> WorkflowResponse:
        return self._plan_executor.run_named_workflow(
            prompt,
            workflow_name,
            session_id=self._session_id_ctx.get(),
            memory_id=self._memory_id_ctx.get(),
        )
