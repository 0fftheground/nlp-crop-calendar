from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..infra.llm import get_extractor_model
from ..observability.logging_utils import log_event, summarize_text
from .workflows.registry import WorkflowSpec


VALID_ACTIONS = {"tool", "workflow", "none"}


class FastIntentResult(BaseModel):
    action: Literal["tool", "workflow", "none"]
    name: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    input: Optional[Any] = None


class FastIntentRouter:
    def __init__(
        self,
        tool_specs: Iterable[Dict[str, str]],
        workflow_specs: Iterable[WorkflowSpec],
    ) -> None:
        self._tool_specs = list(tool_specs)
        self._workflow_specs = list(workflow_specs)
        self._tool_names = {spec["name"] for spec in self._tool_specs}
        self._workflow_names = {spec.name for spec in self._workflow_specs}
        self._llm = get_extractor_model()
        self._system_prompt = self._build_prompt()

    def classify(
        self, prompt: str, *, pending: Optional[dict] = None
    ) -> Optional[FastIntentResult]:
        payload = {
            "prompt": prompt,
            "pending": self._summarize_pending(pending),
        }
        user_payload = json.dumps(payload, ensure_ascii=False, default=str)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user_payload),
        ]
        log_event(
            "fast_intent_call",
            prompt_summary=summarize_text(prompt),
            pending=payload["pending"],
        )
        try:
            raw_result = self._llm.invoke(messages)
        except Exception as exc:
            log_event("fast_intent_error", error=str(exc))
            return None
        raw_text = self._extract_llm_text(raw_result)
        log_event(
            "fast_intent_raw",
            raw=raw_text,
            raw_summary=summarize_text(raw_text),
        )
        payload_data = self._load_json_payload(raw_text)
        result = self._normalize_result(payload_data)
        if result is None:
            return None
        log_event(
            "fast_intent_result",
            action=result.action,
            name=result.name,
            confidence=result.confidence,
            reason=summarize_text(result.reason or ""),
        )
        return result

    def _build_prompt(self) -> str:
        tool_lines = []
        for tool in self._tool_specs:
            name = tool["name"]
            desc = tool["description"]
            tool_lines.append(f"- {name}: {desc}")
        tools_text = "\n".join(tool_lines) or "(none)"
        workflow_lines = []
        for spec in self._workflow_specs:
            workflow_lines.append(f"- {spec.name}: {spec.description}")
        workflows_text = "\n".join(workflow_lines) or "(none)"
        return (
            "你是意图路由器，任务是从给定列表中选择 action 与 name。\n"
            "仅允许以下 action：tool / workflow / none。\n"
            "仅允许以下 name（必须与列表完全一致）：\n"
            f"tools:\n{tools_text}\n"
            f"workflows:\n{workflows_text}\n"
            "如果无法确定或不相关，选择 action=none，并将 confidence <= 0.6。\n"
            "输出严格的 JSON："
            '{"action":"tool|workflow|none","name":"...","confidence":0-1,'
            '"reason":"...", "input":{...}}\n'
            "name 仅在 action=tool/workflow 时填写。"
        )

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

    def _normalize_result(self, payload: Optional[dict]) -> Optional[FastIntentResult]:
        if not isinstance(payload, dict):
            return None
        try:
            result = FastIntentResult.model_validate(payload)
        except Exception as exc:
            log_event("fast_intent_error", error=f"invalid_result:{exc}")
            return None
        if result.action not in VALID_ACTIONS:
            return None
        if result.action == "tool" and result.name not in self._tool_names:
            return None
        if result.action == "workflow" and result.name not in self._workflow_names:
            return None
        result.confidence = max(0.0, min(1.0, float(result.confidence or 0.0)))
        return result
