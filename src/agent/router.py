import contextvars
import json
from typing import Optional

from ..infra.pending_store import build_pending_followup_store
from ..infra.preference_store import build_preference_store
from ..observability.logging_utils import log_event
from ..schemas.models import (
    HandleResponse,
    PlantingDetails,
    PlantingDetailsDraft,
    ToolInvocation,
    UserRequest,
    WorkflowResponse,
)
from .intent_rules import (
    is_cancel_only,
    is_cancel_request,
    is_memory_clear_request,
    strip_cancel_request,
    strip_memory_clear,
)
from .planner import ActionPlan, PlannerRunner
from .tools.registry import execute_tool, list_tool_specs
from .workflows.registry import get_workflow_spec, list_workflow_specs


DEFAULT_NONE_MESSAGE = "未识别到与农事相关的需求。"
CANCEL_MESSAGE = "已取消追问，如需继续请描述新的问题。"
MEMORY_CLEAR_MESSAGE = "已清除记忆。"


class RequestRouter:
    """Route requests with a Planner+Executor workflow."""

    _session_id_ctx = contextvars.ContextVar("session_id", default="default")

    def __init__(self):
        self._workflow_specs = list_workflow_specs()
        self._workflow_names = {spec.name for spec in self._workflow_specs}
        self._workflow_graphs: dict[str, object] = {}
        tool_specs = list_tool_specs()
        self._tool_names = {spec["name"] for spec in tool_specs}
        self._planner = PlannerRunner(tool_specs, self._workflow_specs)
        self._pending_store = build_pending_followup_store()
        self._preference_store = build_preference_store()

    def handle(self, request: UserRequest) -> HandleResponse:
        session_id = request.session_id or "default"
        token = self._session_id_ctx.set(session_id)
        prompt = (request.prompt or "").strip()
        try:
            if not prompt:
                plan = WorkflowResponse(message=DEFAULT_NONE_MESSAGE)
                return HandleResponse(mode="none", plan=plan)
            if is_memory_clear_request(prompt):
                self._preference_store.delete(session_id)
                pending = self._pending_store.get(session_id)
                if pending and pending.get("mode") == "workflow":
                    pending["memory_decision"] = None
                    pending["memory_prompted"] = False
                    self._pending_store.set(session_id, pending)
                prompt = strip_memory_clear(prompt)
                if not prompt:
                    plan = WorkflowResponse(message=MEMORY_CLEAR_MESSAGE)
                    return HandleResponse(mode="none", plan=plan)
            pending = self._pending_store.get(session_id)
            if is_cancel_request(prompt):
                if pending:
                    self._pending_store.delete(session_id)
                    pending = None
                if is_cancel_only(prompt):
                    plan = WorkflowResponse(message=CANCEL_MESSAGE)
                    return HandleResponse(mode="none", plan=plan)
                prompt = strip_cancel_request(prompt)
                if not prompt:
                    plan = WorkflowResponse(message=CANCEL_MESSAGE)
                    return HandleResponse(mode="none", plan=plan)
            pending = self._pending_store.get(session_id)
            plan = self._planner.plan(prompt, pending=pending)
            if not plan:
                return self._fallback_from_planner(prompt, pending, session_id)
            return self._execute_plan(plan, prompt, pending, session_id)
        finally:
            self._session_id_ctx.reset(token)

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
        tool_name = self._resolve_tool_name(plan, pending)
        if not tool_name:
            message = plan.response or "未指定可用工具。"
            return HandleResponse(mode="none", plan=WorkflowResponse(message=message))
        if pending and pending.get("mode") != "tool":
            self._pending_store.delete(session_id)
            pending = None
        if pending and pending.get("mode") == "tool":
            if pending.get("tool_name") == tool_name:
                tool_payload = self._run_tool_followup(prompt, pending, session_id)
                return HandleResponse(mode="tool", tool=tool_payload)
            self._pending_store.delete(session_id)
        tool_input = self._coerce_plan_input(plan.input, prompt)
        tool_payload = execute_tool(tool_name, tool_input)
        if not tool_payload:
            tool_payload = ToolInvocation(
                name=tool_name,
                message="tool not found",
                data={},
            )
        self._update_tool_followup_state(session_id, tool_payload)
        return HandleResponse(mode="tool", tool=tool_payload)

    def _execute_workflow_plan(
        self,
        plan: ActionPlan,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        workflow_name = self._resolve_workflow_name(plan, pending)
        if not workflow_name:
            message = plan.response or "workflow_name 缺失，无法执行。"
            return HandleResponse(mode="none", plan=WorkflowResponse(message=message))
        if pending and pending.get("mode") != "workflow":
            self._pending_store.delete(session_id)
        if pending and pending.get("mode") == "workflow":
            if pending.get("workflow_name") != workflow_name:
                self._pending_store.delete(session_id)
        workflow_prompt = prompt
        if isinstance(plan.input, str) and plan.input.strip():
            workflow_prompt = plan.input
        plan_payload = self._run_named_workflow(workflow_prompt, workflow_name)
        return HandleResponse(mode="workflow", plan=plan_payload)

    def _respond_none(
        self,
        plan: ActionPlan,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        if pending:
            self._pending_store.delete(session_id)
        message = plan.response or DEFAULT_NONE_MESSAGE
        return HandleResponse(mode="none", plan=WorkflowResponse(message=message))

    def _fallback_from_planner(
        self, prompt: str, pending: Optional[dict], session_id: str
    ) -> HandleResponse:
        log_event(
            "planner_fallback",
            reason="planner_error",
            pending_mode=pending.get("mode") if pending else None,
        )
        if pending:
            return self._resume_pending(prompt, pending, session_id)
        plan = WorkflowResponse(message=DEFAULT_NONE_MESSAGE)
        return HandleResponse(mode="none", plan=plan)

    def _resume_pending(
        self, prompt: str, pending: dict, session_id: str
    ) -> HandleResponse:
        if pending.get("mode") == "tool":
            tool_payload = self._run_tool_followup(prompt, pending, session_id)
            return HandleResponse(mode="tool", tool=tool_payload)
        plan = self._run_named_workflow(prompt, pending.get("workflow_name"))
        return HandleResponse(mode="workflow", plan=plan)

    def _resolve_tool_name(
        self, plan: ActionPlan, pending: Optional[dict]
    ) -> Optional[str]:
        name = plan.name
        if name and name not in self._tool_names:
            log_event("planner_invalid_tool", name=name)
            name = None
        if not name and pending and pending.get("mode") == "tool":
            name = pending.get("tool_name")
        return name

    def _resolve_workflow_name(
        self, plan: ActionPlan, pending: Optional[dict]
    ) -> Optional[str]:
        name = plan.name
        if name and name not in self._workflow_names:
            log_event("planner_invalid_workflow", name=name)
            name = None
        if not name and pending and pending.get("mode") == "workflow":
            name = pending.get("workflow_name")
        return name

    @staticmethod
    def _coerce_plan_input(value: object, fallback: str) -> str:
        if value is None:
            return fallback
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True, default=str)
        except TypeError:
            return str(value)

    def _run_named_workflow(
        self, prompt: str, workflow_name: Optional[str]
    ) -> WorkflowResponse:
        if not workflow_name:
            return WorkflowResponse(message="workflow_name 缺失，无法执行。")
        graph = self._workflow_graphs.get(workflow_name)
        if graph is None:
            spec = get_workflow_spec(workflow_name)
            if spec is None:
                return WorkflowResponse(message=f"workflow 未注册: {workflow_name}")
            graph = spec.builder()
            self._workflow_graphs[workflow_name] = graph
        return self._run_graph(prompt, graph, workflow_name)

    def _run_graph(self, prompt: str, graph, workflow_name: str) -> WorkflowResponse:
        session_id = self._session_id_ctx.get()
        initial_state = {"user_prompt": prompt, "trace": []}
        memory = self._preference_store.get(session_id)
        if memory:
            initial_state["memory_planting"] = memory.planting
        pending = self._pending_store.get(session_id)
        if pending and pending.get("workflow_name") == workflow_name:
            initial_state.update(
                {
                    "planting_draft": pending.get("planting_draft"),
                    "missing_fields": pending.get("missing_fields"),
                    "followup_count": pending.get("followup_count", 0),
                    "memory_decision": pending.get("memory_decision"),
                    "memory_prompted": pending.get("memory_prompted", False),
                }
            )
        state = graph.invoke(initial_state)
        planting = state.get("planting")
        if isinstance(planting, PlantingDetails):
            self._preference_store.set(session_id, planting)
        self._update_workflow_followup_state(session_id, state, workflow_name)
        return WorkflowResponse(
            query=state.get("query"),
            recommendations=state.get("recommendations", []),
            message=state.get("message", ""),
            trace=state.get("trace", []),
            data=state.get("data", {}),
        )

    def _run_tool_followup(
        self, prompt: str, pending: dict, session_id: str
    ) -> ToolInvocation:
        tool_name = pending.get("tool_name")
        if not tool_name:
            self._pending_store.delete(session_id)
            return ToolInvocation(
                name="unknown_tool",
                message="tool followup missing tool_name",
                data={},
            )
        followup_payload = {
            "followup": {
                "prompt": prompt,
                "draft": pending.get("draft") or {},
                "missing_fields": pending.get("missing_fields") or [],
                "followup_count": pending.get("followup_count", 0),
            }
        }
        followup_prompt = json.dumps(
            followup_payload, ensure_ascii=True, default=str
        )
        result = execute_tool(tool_name, followup_prompt)
        if not result:
            self._pending_store.delete(session_id)
            return ToolInvocation(
                name=tool_name,
                message="tool not found",
                data={},
            )
        self._update_tool_followup_state(session_id, result)
        return result

    def _update_workflow_followup_state(
        self, session_id: str, state: dict, workflow_name: str
    ) -> None:
        missing = state.get("missing_fields") or []
        draft = state.get("planting_draft")
        draft_payload = None
        if isinstance(draft, PlantingDetailsDraft):
            draft_payload = draft.model_dump(mode="json")
        elif isinstance(draft, dict):
            try:
                draft_payload = PlantingDetailsDraft.model_validate(draft).model_dump(
                    mode="json"
                )
            except Exception:
                draft_payload = draft
        if missing and isinstance(draft_payload, dict):
            payload = {
                "mode": "workflow",
                "workflow_name": workflow_name,
                "planting_draft": draft_payload,
                "missing_fields": missing,
                "followup_count": state.get("followup_count", 0),
                "memory_decision": state.get("memory_decision"),
                "memory_prompted": state.get("memory_prompted", False),
            }
            self._pending_store.set(session_id, payload)
        else:
            pending = self._pending_store.get(session_id)
            if (
                pending
                and pending.get("mode") == "workflow"
                and pending.get("workflow_name") == workflow_name
            ):
                self._pending_store.delete(session_id)

    def _update_tool_followup_state(
        self, session_id: str, tool_payload: ToolInvocation
    ) -> None:
        data = tool_payload.data or {}
        missing = data.get("missing_fields") or []
        draft = data.get("draft")
        if missing and isinstance(draft, dict):
            followup_count = data.get("followup_count", 0)
            self._pending_store.set(
                session_id,
                {
                    "mode": "tool",
                    "tool_name": tool_payload.name,
                    "draft": draft,
                    "missing_fields": missing,
                    "followup_count": followup_count,
                },
            )
        else:
            pending = self._pending_store.get(session_id)
            if pending and pending.get("mode") == "tool":
                self._pending_store.delete(session_id)
