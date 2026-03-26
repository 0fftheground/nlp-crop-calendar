from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from ..application.services.weather_service import parse_weather_prompt_operations
from ..observability.logging_utils import log_event
from ..observability.otel import (
    build_span_attributes,
    record_exception,
    start_span,
    summarize_state,
)
from ..messages.input_validation import (
    INPUT_SCHEMA_FALLBACK_MESSAGE,
    format_input_validation_message,
)
from ..messages.tool_messages import (
    TOOL_FOLLOWUP_MISSING_NAME_MESSAGE,
    TOOL_NOT_FOUND_MESSAGE,
)
from ..schemas.models import (
    HandleResponse,
    ToolInvocation,
    WorkflowResponse,
)
from .input_specs import get_input_spec


DEFAULT_NONE_MESSAGE = "未识别到与农事相关的需求。"
INPUT_VALIDATION_MODE = "input_validation"
INPUT_VALIDATION_MAX_ATTEMPTS = 1


class PlanExecutor:
    def __init__(
        self,
        *,
        tool_names: set[str],
        workflow_names: set[str],
        pending_manager,
        get_workflow_spec_fn,
    ) -> None:
        self._tool_names = tool_names
        self._workflow_names = workflow_names
        self._pending_manager = pending_manager
        self._get_workflow_spec = get_workflow_spec_fn
        self._workflow_graphs: dict[str, object] = {}

    @staticmethod
    def _annotate_workflow_response(
        plan_payload: WorkflowResponse, workflow_name: str
    ) -> WorkflowResponse:
        data = dict(plan_payload.data or {})
        if data.get("workflow_name") == workflow_name:
            return plan_payload
        data["workflow_name"] = workflow_name
        return plan_payload.model_copy(update={"data": data})

    @staticmethod
    def _build_workflow_state_snapshot(state: dict) -> dict:
        snapshot: dict[str, object] = {}
        for key in ("draft", "missing_fields", "followup_count", "pending_message", "options"):
            value = state.get(key)
            if value in (None, "", []):
                continue
            snapshot[key] = value
        return snapshot

    def apply_input_validation(
        self,
        plan,
        pending: Optional[dict],
        session_id: str,
    ) -> tuple[object, Optional[dict], Optional[HandleResponse]]:
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
            plan = plan.model_copy(update={"action": target_action, "name": target_name})
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
            missing_fields, invalid_fields = self._classify_validation_errors(exc)
            if not missing_fields and not invalid_fields:
                missing_fields = list(spec.required_fields)
            if input_attempts >= INPUT_VALIDATION_MAX_ATTEMPTS:
                if pending and pending.get("mode") == INPUT_VALIDATION_MODE:
                    self._pending_manager.delete(session_id)
                response = HandleResponse(
                    mode="none",
                    plan=WorkflowResponse(message=INPUT_SCHEMA_FALLBACK_MESSAGE),
                )
                return plan, None, response
            self._pending_manager.set(
                session_id,
                {
                    "mode": INPUT_VALIDATION_MODE,
                    "action": target_action,
                    "name": target_name,
                    "missing_fields": missing_fields,
                    "invalid_fields": invalid_fields,
                    "input_attempts": input_attempts + 1,
                },
            )
            message = format_input_validation_message(
                target_name,
                missing_fields,
                spec.field_labels,
                invalid_fields=invalid_fields,
            )
            response = HandleResponse(mode="none", plan=WorkflowResponse(message=message))
            return plan, None, response
        if pending and pending.get("mode") == INPUT_VALIDATION_MODE:
            self._pending_manager.delete(session_id)
        plan = plan.model_copy(update={"input": spec.to_prompt(validated)})
        return plan, None, None

    def execute_plan(
        self,
        plan,
        *,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
        memory_id: str,
        run_named_workflow,
        execute_tool_fn,
    ) -> HandleResponse:
        if plan.action == "tool":
            return self.execute_tool_plan(
                plan,
                prompt=prompt,
                pending=pending,
                session_id=session_id,
                memory_id=memory_id,
                execute_tool_fn=execute_tool_fn,
            )
        if plan.action == "workflow":
            return self.execute_workflow_plan(
                plan,
                prompt=prompt,
                pending=pending,
                session_id=session_id,
                run_named_workflow=run_named_workflow,
            )
        if pending and not plan.response:
            log_event(
                "planner_fallback",
                reason="none_action_with_pending",
                pending_mode=pending.get("mode"),
            )
            return self.resume_pending(
                prompt,
                pending,
                session_id=session_id,
                memory_id=memory_id,
                run_named_workflow=run_named_workflow,
                execute_tool_fn=execute_tool_fn,
            )
        return self.respond_none(plan, pending=pending, session_id=session_id)

    def execute_tool_plan(
        self,
        plan,
        *,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
        memory_id: str,
        execute_tool_fn,
    ) -> HandleResponse:
        tool_name = self._resolve_tool_name(plan, pending)
        if not tool_name:
            message = plan.response or "未指定可用工具。"
            return HandleResponse(mode="none", plan=WorkflowResponse(message=message))
        if tool_name == "memory_clear":
            tool_payload = self.clear_session_memory(session_id, pending)
            return HandleResponse(mode="tool", tool=tool_payload)
        if pending and pending.get("mode") != "tool":
            self._pending_manager.delete(session_id)
            pending = None
        if pending and pending.get("mode") == "tool":
            if pending.get("tool_name") == tool_name:
                tool_payload = self.run_tool_followup(
                    prompt,
                    pending,
                    session_id=session_id,
                    memory_id=memory_id,
                    execute_tool_fn=execute_tool_fn,
                )
                return HandleResponse(mode="tool", tool=tool_payload)
            self._pending_manager.delete(session_id)
        tool_input = self._coerce_plan_input(plan.input, prompt)
        if tool_name == "variety_lookup" and not pending:
            tool_input = self._build_variety_tool_input(
                tool_input, prompt=prompt, memory_id=memory_id
            )
        if tool_name == "weather_lookup" and not pending:
            tool_input = self._build_weather_tool_input(tool_input, prompt=prompt)
        tool_payload = execute_tool_fn(tool_name, tool_input)
        if not tool_payload:
            tool_payload = ToolInvocation(
                name=tool_name,
                message=TOOL_NOT_FOUND_MESSAGE,
                data={},
            )
        self._pending_manager.update_tool_followup_state(session_id, tool_payload)
        return HandleResponse(mode="tool", tool=tool_payload)

    def execute_workflow_plan(
        self,
        plan,
        *,
        prompt: str,
        pending: Optional[dict],
        session_id: str,
        run_named_workflow,
    ) -> HandleResponse:
        workflow_name = self._resolve_workflow_name(plan, pending)
        if not workflow_name:
            message = plan.response or "workflow_name 缺失，无法执行。"
            return HandleResponse(mode="none", plan=WorkflowResponse(message=message))
        if pending and pending.get("mode") != "workflow":
            self._pending_manager.delete(session_id)
        if pending and pending.get("mode") == "workflow":
            if pending.get("workflow_name") != workflow_name:
                self._pending_manager.delete(session_id)
        workflow_prompt = prompt
        if isinstance(plan.input, str) and plan.input.strip():
            workflow_prompt = plan.input
        plan_payload = run_named_workflow(workflow_prompt, workflow_name)
        plan_payload = self._annotate_workflow_response(plan_payload, workflow_name)
        return HandleResponse(mode="workflow", plan=plan_payload)

    def respond_none(
        self,
        plan,
        *,
        pending: Optional[dict],
        session_id: str,
    ) -> HandleResponse:
        if pending:
            self._pending_manager.delete(session_id)
        message = plan.response or DEFAULT_NONE_MESSAGE
        return HandleResponse(mode="none", plan=WorkflowResponse(message=message))

    def fallback_from_planner(
        self,
        prompt: str,
        pending: Optional[dict],
        *,
        session_id: str,
        memory_id: str,
        run_named_workflow,
        execute_tool_fn,
    ) -> HandleResponse:
        log_event(
            "planner_fallback",
            reason="planner_error",
            pending_mode=pending.get("mode") if pending else None,
        )
        if pending and pending.get("mode") in {"tool", "workflow"}:
            return self.resume_pending(
                prompt,
                pending,
                session_id=session_id,
                memory_id=memory_id,
                run_named_workflow=run_named_workflow,
                execute_tool_fn=execute_tool_fn,
            )
        plan = WorkflowResponse(message=DEFAULT_NONE_MESSAGE)
        return HandleResponse(mode="none", plan=plan)

    def resume_pending(
        self,
        prompt: str,
        pending: dict,
        *,
        session_id: str,
        memory_id: str,
        run_named_workflow,
        execute_tool_fn,
    ) -> HandleResponse:
        if pending.get("mode") == "tool":
            tool_payload = self.run_tool_followup(
                prompt,
                pending,
                session_id=session_id,
                memory_id=memory_id,
                execute_tool_fn=execute_tool_fn,
            )
            return HandleResponse(mode="tool", tool=tool_payload)
        workflow_name = str(pending.get("workflow_name") or "").strip()
        plan = run_named_workflow(prompt, workflow_name)
        if workflow_name:
            plan = self._annotate_workflow_response(plan, workflow_name)
        return HandleResponse(mode="workflow", plan=plan)

    def run_named_workflow(
        self,
        prompt: str,
        workflow_name: Optional[str],
        *,
        session_id: str,
        memory_id: str,
    ) -> WorkflowResponse:
        if not workflow_name:
            return WorkflowResponse(message="workflow_name 缺失，无法执行。")
        graph = self._workflow_graphs.get(workflow_name)
        if graph is None:
            spec = self._get_workflow_spec(workflow_name)
            if spec is None:
                return WorkflowResponse(message=f"workflow 未注册: {workflow_name}")
            graph = spec.builder()
            self._workflow_graphs[workflow_name] = graph
        return self.run_graph(
            prompt, graph, workflow_name, session_id=session_id, memory_id=memory_id
        )

    def run_graph(
        self,
        prompt: str,
        graph,
        workflow_name: str,
        *,
        session_id: str,
        memory_id: str,
    ) -> WorkflowResponse:
        initial_state = {"user_prompt": prompt, "trace": [], "user_id": memory_id}
        pending = self._pending_manager.get(session_id)
        initial_state.update(
            self._pending_manager.build_workflow_resume_state(
                pending, workflow_name
            )
        )
        span_attrs = {"workflow.name": workflow_name}
        span_attrs.update(
            build_span_attributes(
                "workflow.input",
                {"prompt": prompt, "workflow": workflow_name},
            )
        )
        with start_span(f"workflow.{workflow_name}", attributes=span_attrs) as span:
            try:
                state = graph.invoke(initial_state)
            except Exception as exc:
                record_exception(span, exc)
                raise
            self._pending_manager.update_workflow_followup_state(
                session_id, state, workflow_name
            )
            output_summary = summarize_state(state)
            output_attrs = build_span_attributes("workflow.output", output_summary)
            if span:
                for key, value in output_attrs.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass
            data = dict(state.get("data", {}) or {})
            workflow_state = self._build_workflow_state_snapshot(state)
            if workflow_state:
                data["workflow_state"] = workflow_state
            return WorkflowResponse(
                recommendations=state.get("recommendations", []),
                growth_stage=state.get("growth_stage"),
                message=state.get("message", ""),
                trace=state.get("trace", []),
                data=data,
            )

    def run_tool_followup(
        self,
        prompt: str,
        pending: dict,
        *,
        session_id: str,
        memory_id: str,
        execute_tool_fn,
    ) -> ToolInvocation:
        tool_name = pending.get("tool_name")
        if not tool_name:
            self._pending_manager.delete(session_id)
            return ToolInvocation(
                name="unknown_tool",
                message=TOOL_FOLLOWUP_MISSING_NAME_MESSAGE,
                data={},
            )
        followup_prompt = self._pending_manager.build_tool_followup_prompt(
            prompt=prompt,
            pending=pending,
            memory_id=memory_id,
        )
        result = execute_tool_fn(tool_name, followup_prompt)
        if not result:
            self._pending_manager.delete(session_id)
            return ToolInvocation(
                name=tool_name,
                message=TOOL_NOT_FOUND_MESSAGE,
                data={},
            )
        self._pending_manager.update_tool_followup_state(session_id, result)
        return result

    def clear_session_memory(
        self, session_id: str, pending: Optional[dict]
    ) -> ToolInvocation:
        if pending and pending.get("mode") == "workflow":
            pending = dict(pending)
            self._pending_manager.set(session_id, pending)
        return ToolInvocation(
            name="memory_clear",
            message="已清除历史经验记录。",
            data={},
        )

    def _resolve_tool_name(self, plan, pending: Optional[dict]) -> Optional[str]:
        name = plan.name
        if name and name not in self._tool_names:
            log_event("planner_invalid_tool", name=name)
            name = None
        if not name and pending and pending.get("mode") == "tool":
            name = pending.get("tool_name")
        return name

    def _resolve_workflow_name(self, plan, pending: Optional[dict]) -> Optional[str]:
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
            return json.dumps(value, ensure_ascii=False, default=str)
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
    def _build_variety_tool_input(
        tool_input: str, *, prompt: str, memory_id: str
    ) -> str:
        payload = {"prompt": prompt, "query": prompt, "user_id": memory_id}
        try:
            parsed = json.loads(tool_input)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if value not in (None, ""):
                    payload[key] = value
            effective_prompt = str(
                parsed.get("prompt")
                or parsed.get("query")
                or payload.get("prompt")
                or payload.get("query")
                or prompt
            ).strip()
            payload["prompt"] = effective_prompt or prompt
            payload.setdefault("query", payload["prompt"])
        else:
            text = str(tool_input or "").strip()
            if text:
                payload["prompt"] = text
                payload["query"] = text
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _build_weather_tool_input(tool_input: str, *, prompt: str) -> str:
        payload = {"prompt": prompt, "query": prompt}
        try:
            parsed = json.loads(tool_input)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if value not in (None, ""):
                    payload[key] = value
            payload["prompt"] = str(payload.get("prompt") or prompt).strip() or prompt
            payload["query"] = str(payload.get("query") or prompt).strip() or prompt
        else:
            text = str(tool_input or "").strip()
            if text:
                payload["prompt"] = text
                payload["query"] = text
        requested_operations = payload.get("requested_operations")
        if not isinstance(requested_operations, list) or not any(
            str(item).strip() for item in requested_operations
        ):
            supported_ops, _, _ = parse_weather_prompt_operations(payload["query"])
            if supported_ops:
                payload["requested_operations"] = supported_ops
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _classify_validation_errors(
        exc: ValidationError,
    ) -> tuple[list[str], list[str]]:
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        for error in exc.errors():
            loc = error.get("loc") or []
            if not loc:
                continue
            field = str(loc[0])
            if not field:
                continue
            error_type = str(error.get("type") or "")
            target = missing_fields if error_type == "missing" else invalid_fields
            if field not in target:
                target.append(field)
        return missing_fields, invalid_fields
