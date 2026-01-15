import contextvars
import json
from typing import Optional

from pydantic import ValidationError

from ..infra.pending_store import build_pending_followup_store
from ..infra.memory_store import build_memory_store
from ..observability.logging_utils import log_event
from ..prompts.input_validation import (
    INPUT_SCHEMA_FALLBACK_MESSAGE,
    format_input_validation_message,
)
from ..prompts.tool_messages import (
    TOOL_FOLLOWUP_MISSING_NAME_MESSAGE,
    TOOL_NOT_FOUND_MESSAGE,
)
from ..schemas.models import (
    HandleResponse,
    PlantingDetails,
    PlantingDetailsDraft,
    ToolInvocation,
    UserRequest,
    WorkflowResponse,
)
from .input_specs import get_input_spec
from .planner import ActionPlan, PlannerRunner
from .tools.registry import execute_tool, list_tool_specs
from .workflows.registry import get_workflow_spec, list_workflow_specs


DEFAULT_NONE_MESSAGE = "未识别到与农事相关的需求。"
INPUT_VALIDATION_MODE = "input_validation"
INPUT_VALIDATION_MAX_ATTEMPTS = 1


class RequestRouter:
    """Route requests with a Planner+Executor workflow."""

    _session_id_ctx = contextvars.ContextVar("session_id", default="default")
    _memory_id_ctx = contextvars.ContextVar("memory_id", default="default")

    def __init__(self):
        self._workflow_specs = list_workflow_specs()
        self._workflow_names = {spec.name for spec in self._workflow_specs}
        self._workflow_graphs: dict[str, object] = {}
        tool_specs = list_tool_specs()
        self._tool_names = {spec["name"] for spec in tool_specs}
        self._planner = PlannerRunner(tool_specs, self._workflow_specs)
        self._pending_store = build_pending_followup_store()
        self._memory_store = build_memory_store()

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
            pending = self._pending_store.get(session_id)
            plan = self._planner.plan(prompt, pending=pending)
            if not plan:
                return self._fallback_from_planner(prompt, pending, session_id)
            plan, exec_pending, response = self._apply_input_validation(
                plan, pending, session_id
            )
            if response:
                return response
            return self._execute_plan(plan, prompt, exec_pending, session_id)
        finally:
            self._memory_id_ctx.reset(memory_token)
            self._session_id_ctx.reset(session_token)

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

    def _apply_input_validation(
        self,
        plan: ActionPlan,
        pending: Optional[dict],
        session_id: str,
    ) -> tuple[ActionPlan, Optional[dict], Optional[HandleResponse]]:
        if plan.action not in {"tool", "workflow"}:
            return plan, pending, None
        if pending and pending.get("mode") in {"tool", "workflow"}:
            return plan, pending, None
        input_attempts = 0
        target_action = plan.action
        target_name = plan.name
        if pending and pending.get("mode") == INPUT_VALIDATION_MODE:
            target_action = pending.get("action") or target_action
            target_name = pending.get("name") or target_name
            input_attempts = int(pending.get("input_attempts") or 0)
            plan = plan.model_copy(
                update={"action": target_action, "name": target_name}
            )
        if not target_name:
            return plan, None, None
        spec = get_input_spec(target_action, target_name)
        if not spec:
            return plan, None, None
        payload = self._coerce_input_payload(plan.input)
        if not spec.required_fields and (payload is None or payload == ""):
            payload = {}
        try:
            validated = spec.model.model_validate(payload)
        except ValidationError as exc:
            if not spec.required_fields:
                try:
                    validated = spec.model.model_validate({})
                    plan = plan.model_copy(update={"input": spec.to_prompt(validated)})
                    return plan, None, None
                except ValidationError:
                    pass
            missing_fields = (
                self._extract_missing_fields(exc)
                or list(spec.required_fields)
            )
            if input_attempts >= INPUT_VALIDATION_MAX_ATTEMPTS:
                if pending and pending.get("mode") == INPUT_VALIDATION_MODE:
                    self._pending_store.delete(session_id)
                response = HandleResponse(
                    mode="none",
                    plan=WorkflowResponse(message=INPUT_SCHEMA_FALLBACK_MESSAGE),
                )
                return plan, None, response
            self._pending_store.set(
                session_id,
                {
                    "mode": INPUT_VALIDATION_MODE,
                    "action": target_action,
                    "name": target_name,
                    "missing_fields": missing_fields,
                    "input_attempts": input_attempts + 1,
                },
            )
            message = format_input_validation_message(
                target_name, missing_fields, spec.field_labels
            )
            response = HandleResponse(
                mode="none", plan=WorkflowResponse(message=message)
            )
            return plan, None, response
        if pending and pending.get("mode") == INPUT_VALIDATION_MODE:
            self._pending_store.delete(session_id)
        plan = plan.model_copy(update={"input": spec.to_prompt(validated)})
        return plan, None, None

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
        if tool_name == "memory_clear":
            tool_payload = self._clear_session_memory(session_id, pending)
            return HandleResponse(mode="tool", tool=tool_payload)
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
                message=TOOL_NOT_FOUND_MESSAGE,
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
        if pending and pending.get("mode") in {"tool", "workflow"}:
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

    @staticmethod
    def _coerce_input_payload(value: object) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _extract_missing_fields(exc: ValidationError) -> list[str]:
        fields: list[str] = []
        for error in exc.errors():
            loc = error.get("loc") or []
            if not loc:
                continue
            field = str(loc[0])
            if field not in fields:
                fields.append(field)
        return fields

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
        memory_id = self._memory_id_ctx.get()
        initial_state = {"user_prompt": prompt, "trace": []}
        memory = self._memory_store.get(memory_id)
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
            self._memory_store.set(memory_id, planting)
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
                message=TOOL_FOLLOWUP_MISSING_NAME_MESSAGE,
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
                message=TOOL_NOT_FOUND_MESSAGE,
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

    def _clear_session_memory(
        self, session_id: str, pending: Optional[dict]
    ) -> ToolInvocation:
        memory_id = self._memory_id_ctx.get()
        self._memory_store.delete(memory_id)
        if pending and pending.get("mode") == "workflow":
            pending = dict(pending)
            pending["memory_decision"] = None
            pending["memory_prompted"] = False
            self._pending_store.set(session_id, pending)
        return ToolInvocation(
            name="memory_clear",
            message="已清除记忆。",
            data={},
        )
