from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..infra.llm import get_chat_model
from .input_specs import format_input_schema, get_input_spec
from ..observability.logging_utils import log_event, summarize_text
from ..prompts.planner import build_planner_prompt
from .workflows.registry import WorkflowSpec


class ActionPlan(BaseModel):
    action: Literal["tool", "workflow", "none"]
    name: Optional[str] = None
    input: Optional[Any] = None
    response: Optional[str] = None
    reason: Optional[str] = None


class PlannerRunner:
    def __init__(
        self,
        tool_specs: Iterable[Dict[str, str]],
        workflow_specs: Iterable[WorkflowSpec],
    ) -> None:
        self._tool_specs = list(tool_specs)
        self._workflow_specs = list(workflow_specs)
        self._llm = get_chat_model()
        self._system_prompt = self._build_prompt()

    def plan(self, prompt: str, *, pending: Optional[dict] = None) -> Optional[ActionPlan]:
        payload = {
            "prompt": prompt,
            "pending": self._summarize_pending(pending),
        }
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=True, default=str)
            ),
        ]
        log_event(
            "planner_call",
            prompt_summary=summarize_text(prompt),
            pending=payload["pending"],
        )
        try:
            planner = self._llm.with_structured_output(ActionPlan)
            result = planner.invoke(messages)
        except Exception as exc:
            log_event("planner_error", error=str(exc))
            return None
        try:
            plan = (
                result
                if isinstance(result, ActionPlan)
                else ActionPlan.model_validate(result)
            )
        except Exception as exc:
            log_event("planner_error", error=f"invalid_plan:{exc}")
            return None
        log_event(
            "planner_response",
            action=plan.action,
            name=plan.name,
            reason=summarize_text(plan.reason or ""),
            response_summary=summarize_text(plan.response or ""),
        )
        return plan

    def _build_prompt(self) -> str:
        tool_lines = []
        for tool in self._tool_specs:
            name = tool["name"]
            desc = tool["description"]
            spec = get_input_spec("tool", name)
            if spec:
                tool_lines.append(
                    f"- {name}: {desc} 输入: {format_input_schema(spec)}"
                )
            else:
                tool_lines.append(f"- {name}: {desc}")
        tools_text = "\n".join(tool_lines) or "(none)"
        workflow_lines = []
        for spec in self._workflow_specs:
            input_spec = get_input_spec("workflow", spec.name)
            if input_spec:
                workflow_lines.append(
                    f"- {spec.name}: {spec.description} 输入: {format_input_schema(input_spec)}"
                )
            else:
                workflow_lines.append(f"- {spec.name}: {spec.description}")
        workflows_text = "\n".join(workflow_lines) or "(none)"
        return build_planner_prompt(tools_text, workflows_text)

    @staticmethod
    def _summarize_pending(pending: Optional[dict]) -> Optional[dict]:
        if not isinstance(pending, dict):
            return None
        name = pending.get("tool_name") or pending.get("workflow_name") or pending.get(
            "name"
        )
        summary: Dict[str, Any] = {
            "mode": pending.get("mode"),
            "name": name,
            "missing_fields": pending.get("missing_fields"),
            "followup_count": pending.get("followup_count"),
        }
        if "action" in pending:
            summary["action"] = pending.get("action")
        if "input_attempts" in pending:
            summary["input_attempts"] = pending.get("input_attempts")
        return summary
