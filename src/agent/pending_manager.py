from __future__ import annotations

import re
from typing import Optional

from ..infra.variety_store import find_exact_variety_in_text, retrieve_variety_candidates
from ..observability.logging_utils import log_event
from ..schemas.models import PlantingDetailsDraft, ToolInvocation


_FOLLOWUP_INDEX_RE = re.compile(r"^第?\s*(\d+)\s*(?:个|条|项)?$")
_FOLLOWUP_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']+)[\"“”']")
_NEW_TOPIC_TOKENS = {
    "另一个",
    "另外",
    "再问",
    "新问题",
    "换个",
    "改问",
    "顺便",
    "不相关",
    "无关",
    "取消",
    "不用了",
    "停止",
    "结束",
    "退出",
    "算了",
    "先不",
}
_QUESTION_HINTS = {
    "请问",
    "怎么",
    "如何",
    "为什么",
    "多少",
    "哪里",
    "哪个",
    "是否",
    "能否",
    "可以吗",
    "有无",
    "有没有",
    "帮我",
    "查询",
    "查一下",
    "帮忙",
}
_PENDING_INTERRUPT_RULE_NAMES = {
    "plant_plan_list_active",
    "plant_plan_delete",
    "memory_clear",
}


class PendingManager:
    def __init__(self, pending_store, rule_engine) -> None:
        self._pending_store = pending_store
        self._rule_engine = rule_engine

    def get(self, session_id: str) -> Optional[dict]:
        return self._pending_store.get(session_id)

    def set(self, session_id: str, payload: dict) -> None:
        self._pending_store.set(session_id, payload)

    def delete(self, session_id: str) -> None:
        self._pending_store.delete(session_id)

    def update_workflow_followup_state(
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
                "pending_message": state.get("message"),
                "future_sowing_date_warning": state.get(
                    "future_sowing_date_warning", False
                ),
                "plant_season_id": state.get("plant_season_id"),
                "variety_tool_query": state.get("variety_tool_query"),
                "variety_tool_draft": state.get("variety_tool_draft"),
                "variety_tool_missing_fields": state.get(
                    "variety_tool_missing_fields", []
                ),
                "variety_tool_followup_count": state.get(
                    "variety_tool_followup_count", 0
                ),
            }
            options = self._build_pending_options(
                payload.get("pending_message"), draft_payload
            )
            if options:
                payload["options"] = options
            self.set(session_id, payload)
            return None
        pending = self.get(session_id)
        if (
            pending
            and pending.get("mode") == "workflow"
            and pending.get("workflow_name") == workflow_name
        ):
            self.delete(session_id)
        return None

    def update_tool_followup_state(
        self, session_id: str, tool_payload: ToolInvocation
    ) -> None:
        data = tool_payload.data or {}
        missing = data.get("missing_fields") or []
        draft = data.get("draft")
        choice_hint = bool(data.get("choice_hint"))
        options = data.get("options")
        if (missing and isinstance(draft, dict)) or (
            choice_hint and isinstance(options, list)
        ):
            followup_count = data.get("followup_count", 0)
            payload = {
                "mode": "tool",
                "tool_name": tool_payload.name,
                "draft": draft if isinstance(draft, dict) else {},
                "missing_fields": missing,
                "followup_count": followup_count,
                "pending_message": tool_payload.message,
            }
            query = data.get("query") or data.get("prompt")
            if isinstance(query, str) and query.strip():
                payload["query"] = query.strip()
            if choice_hint and isinstance(options, list):
                payload["choice_hint"] = True
                payload["strict_options_only"] = True
                payload["options"] = [
                    str(item).strip() for item in options if str(item).strip()
                ]
            else:
                built = self._build_pending_options(
                    payload.get("pending_message"), payload.get("draft")
                )
                if built:
                    payload["options"] = built
            self.set(session_id, payload)
            return None
        pending = self.get(session_id)
        if pending and pending.get("mode") == "tool":
            self.delete(session_id)
        return None

    def should_resume_pending(self, prompt: str, pending: Optional[dict]) -> bool:
        if not isinstance(pending, dict):
            return False
        if pending.get("mode") not in {"tool", "workflow"}:
            return False
        rule = self._rule_engine.match(prompt or "")
        if (
            rule
            and rule.action in {"tool", "workflow"}
            and rule.name in _PENDING_INTERRUPT_RULE_NAMES
        ):
            log_event(
                "pending_interrupt",
                reason="rule_override",
                rule_id=rule.id,
                rule_name=rule.name,
            )
            return False
        if pending.get("missing_fields") and "variety" in pending.get(
            "missing_fields", []
        ):
            if find_exact_variety_in_text(prompt):
                return True
            if retrieve_variety_candidates(prompt, limit=3):
                return True
        if pending.get("strict_options_only"):
            return self._matches_pending_choice(prompt, pending)
        if self._matches_pending_choice(prompt, pending):
            return True
        if self._looks_like_new_question(prompt):
            return False
        return True

    @staticmethod
    def _parse_followup_index(text: str) -> Optional[int]:
        if not text:
            return None
        match = _FOLLOWUP_INDEX_RE.match(text.strip())
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_pending_candidates(pending: Optional[dict]) -> list[str]:
        if not isinstance(pending, dict):
            return []
        draft = pending.get("draft")
        if not isinstance(draft, dict):
            return []
        for key in ("candidates", "variety_candidates", "region_candidates"):
            value = draft.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _extract_pending_options(pending: Optional[dict]) -> list[str]:
        if not isinstance(pending, dict):
            return []
        options = pending.get("options")
        if isinstance(options, list):
            return [str(item).strip() for item in options if str(item).strip()]
        return []

    @staticmethod
    def _extract_pending_message(pending: Optional[dict]) -> str:
        if not isinstance(pending, dict):
            return ""
        message = pending.get("pending_message") or pending.get("message")
        return message.strip() if isinstance(message, str) else ""

    @staticmethod
    def _extract_message_options(message: str) -> list[str]:
        if not message:
            return []
        options: list[str] = []
        for line in message.splitlines():
            text = line.strip()
            if not text:
                continue
            if "回复" in text:
                continue
            if text.endswith("：") or "请选择" in text:
                continue
            match = re.match(r"^(\d+)[\.\、]\s*(.+)$", text)
            if match:
                text = match.group(2).strip()
            if text:
                options.append(text)
        if not options:
            for token in _FOLLOWUP_QUOTED_RE.findall(message):
                for piece in re.split(r"[、/或]", token):
                    piece = piece.strip()
                    if piece:
                        options.append(piece)
        return options

    def _build_pending_options(
        self, message: Optional[str], draft: Optional[dict]
    ) -> list[str]:
        options: list[str] = []
        if isinstance(draft, dict):
            for key in ("options", "candidates", "variety_candidates", "region_candidates"):
                value = draft.get(key)
                if isinstance(value, list):
                    for item in value:
                        item = str(item).strip()
                        if item and item not in options:
                            options.append(item)
        if isinstance(message, str) and message.strip():
            for item in self._extract_message_options(message):
                if item not in options:
                    options.append(item)
        return options

    def _matches_pending_choice(self, prompt: str, pending: Optional[dict]) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        options = self._extract_pending_options(pending)
        candidates = options or self._extract_pending_candidates(pending)
        if candidates:
            index = self._parse_followup_index(text)
            if index is not None and 1 <= index <= len(candidates):
                return True
            for candidate in candidates:
                if candidate == text:
                    return True
                if text in candidate or candidate in text:
                    return True
        message = self._extract_pending_message(pending)
        if message and len(text) <= 10 and text in message:
            return True
        if self._parse_followup_index(text) is not None and message and "序号" in message:
            return True
        return False

    @staticmethod
    def _looks_like_new_question(prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        for token in _NEW_TOPIC_TOKENS:
            if token in text:
                return True
        if "?" in text or "？" in text:
            return True
        if len(text) >= 12:
            for token in _QUESTION_HINTS:
                if token in text:
                    return True
        return False
