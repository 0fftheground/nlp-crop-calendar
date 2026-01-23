from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, Optional, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..infra.llm import get_chat_model
from .input_specs import format_input_schema, get_input_spec
from ..observability.logging_utils import log_event, summarize_text
from ..prompts.planner import build_planner_prompt
from .workflows.registry import WorkflowSpec


VALID_ACTIONS = {"tool", "workflow", "none"}


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
        self._tool_names = {spec["name"] for spec in self._tool_specs}
        self._workflow_names = {spec.name for spec in self._workflow_specs}
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
                content=json.dumps(payload, ensure_ascii=False, default=str)
            ),
        ]
        log_event(
            "planner_call",
            prompt_summary=summarize_text(prompt),
            pending=payload["pending"],
        )
        try:
            raw_result = self._llm.invoke(messages)
        except Exception as exc:
            log_event("planner_error", error=str(exc))
            return None
        raw_text = self._extract_llm_text(raw_result)
        log_event(
            "planner_raw",
            raw=raw_text,
            raw_summary=summarize_text(raw_text),
        )
        payload_data = self._load_json_payload(raw_text)
        plan = self._normalize_plan(payload_data)
        if plan is None:
            plan = self._recover_plan(raw_text, payload_data)
        if plan is None:
            plan = ActionPlan(action="none")
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

    @staticmethod
    def _extract_llm_text(result: object) -> str:
        content = getattr(result, "content", result)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        if content is None:
            return ""
        return str(content).strip()

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        return cleaned

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    @classmethod
    def _load_json_payload(cls, text: str) -> Optional[dict]:
        if not text:
            return None
        cleaned = cls._strip_code_fence(text)
        for candidate in (cleaned, cls._extract_json_block(cleaned)):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(candidate)
                except Exception:
                    continue
            if isinstance(data, dict):
                return data
        return None

    def _normalize_plan(self, payload: Optional[dict]) -> Optional[ActionPlan]:
        if not isinstance(payload, dict):
            return None
        try:
            plan = ActionPlan.model_validate(payload)
        except Exception as exc:
            log_event("planner_error", error=f"invalid_plan:{exc}")
            return None
        if plan.action not in VALID_ACTIONS:
            return None
        if plan.action == "tool":
            if not plan.name or plan.name not in self._tool_names:
                log_event("planner_invalid_tool", name=plan.name)
                return None
        if plan.action == "workflow":
            if not plan.name or plan.name not in self._workflow_names:
                log_event("planner_invalid_workflow", name=plan.name)
                return None
        return plan

    def _recover_plan(
        self, raw_text: str, payload: Optional[dict]
    ) -> Optional[ActionPlan]:
        if not raw_text and not payload:
            return None
        payload_text = ""
        if isinstance(payload, dict):
            payload_text = json.dumps(payload, ensure_ascii=False, default=str)
        haystack = f"{raw_text}\n{payload_text}".lower()
        workflow_name = self._find_name(haystack, self._workflow_names)
        if workflow_name:
            log_event(
                "planner_recover",
                recovered_action="workflow",
                name=workflow_name,
                reason="name_match",
            )
            return ActionPlan(
                action="workflow",
                name=workflow_name,
                input=self._payload_value(payload, "input"),
                response=self._payload_value(payload, "response"),
                reason=self._payload_value(payload, "reason"),
            )
        tool_name = self._find_name(haystack, self._tool_names)
        if tool_name:
            log_event(
                "planner_recover",
                recovered_action="tool",
                name=tool_name,
                reason="name_match",
            )
            return ActionPlan(
                action="tool",
                name=tool_name,
                input=self._payload_value(payload, "input"),
                response=self._payload_value(payload, "response"),
                reason=self._payload_value(payload, "reason"),
            )
        action_hint = None
        if "workflow" in haystack:
            action_hint = "workflow"
        elif "tool" in haystack:
            action_hint = "tool"
        if action_hint:
            log_event(
                "planner_recover",
                recovered_action=action_hint,
                reason="keyword_only",
            )
        return ActionPlan(
            action="none",
            response=self._payload_value(payload, "response"),
            reason=self._payload_value(payload, "reason"),
        )

    @staticmethod
    def _find_name(text: str, names: Iterable[str]) -> Optional[str]:
        for name in names:
            if name.lower() in text:
                return name
        return None

    @staticmethod
    def _payload_value(payload: Optional[dict], key: str) -> Optional[object]:
        if not isinstance(payload, dict):
            return None
        return payload.get(key)
