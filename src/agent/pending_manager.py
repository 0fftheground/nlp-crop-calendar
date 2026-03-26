from __future__ import annotations

import json
import re
from typing import Optional

from .followup import (
    build_followup_options,
    get_followup_count,
    get_followup_draft,
    get_followup_kind,
    get_followup_message,
    get_followup_missing_fields,
    get_followup_options,
    extract_draft_options,
    parse_followup_index,
    resolve_followup_choice,
)
from ..infra.variety_store import find_exact_variety_in_text, retrieve_variety_candidates
from ..observability.logging_utils import log_event
from ..schemas import PlantingDetailsDraft, ToolInvocation


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
_UNKNOWN_REPLY_TOKENS = {
    "不知道",
    "不确定",
    "不清楚",
    "不晓得",
    "不太清楚",
}
_FOLLOWUP_REPLY_SPLIT_RE = re.compile(r"[，,、/\s]+")
_YES_REPLY_TOKENS = {"是", "好", "好的", "确认", "要", "保存", "继续", "yes", "y", "ok"}
_NO_REPLY_TOKENS = {"否", "不", "取消", "不用了", "不保存", "算了", "no", "n"}
_CLARIFICATION_CONTINUE_TOKENS = {
    "继续当前",
    "继续当前的",
    "当前",
    "前一个",
    "上一个",
    "第一个",
    "1",
}
_CLARIFICATION_NEW_TOKENS = {
    "新的",
    "新任务",
    "新问题",
    "新的那个",
    "后一个",
    "下一个",
    "第二个",
    "2",
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

    def build_workflow_resume_state(
        self, pending: Optional[dict], workflow_name: str
    ) -> dict:
        if not isinstance(pending, dict):
            return {}
        if pending.get("workflow_name") != workflow_name:
            return {}
        draft = get_followup_draft(pending)
        options = get_followup_options(pending)
        return {
            "draft": draft,
            "missing_fields": get_followup_missing_fields(pending),
            "followup_count": get_followup_count(pending),
            "options": options,
            "pending_message": get_followup_message(pending),
            "pending_kind": get_followup_kind(pending),
            "future_sowing_date_warning": pending.get(
                "future_sowing_date_warning", False
            ),
            "plant_season_id": pending.get("plant_season_id"),
            "variety_tool_query": pending.get("variety_tool_query"),
            "variety_tool_draft": pending.get("variety_tool_draft"),
            "variety_tool_missing_fields": pending.get(
                "variety_tool_missing_fields"
            ),
            "variety_tool_followup_count": pending.get(
                "variety_tool_followup_count", 0
            ),
        }

    def build_tool_followup_prompt(
        self,
        *,
        prompt: str,
        pending: Optional[dict],
        memory_id: str,
    ) -> str:
        followup_payload = {
            "user_id": memory_id,
            "query": pending.get("query") if isinstance(pending, dict) else None,
            "followup": {
                "prompt": prompt,
                "draft": get_followup_draft(pending) or {},
                "missing_fields": get_followup_missing_fields(pending),
                "followup_count": get_followup_count(pending),
            },
        }
        return json.dumps(followup_payload, ensure_ascii=False, default=str)

    def update_workflow_followup_state(
        self, session_id: str, state: dict, workflow_name: str
    ) -> None:
        missing = get_followup_missing_fields(state)
        draft = get_followup_draft(state)
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
                "draft": draft_payload,
                "missing_fields": missing,
                "followup_count": get_followup_count(state),
                "pending_message": get_followup_message(state),
                "pending_kind": get_followup_kind(state),
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
            options = get_followup_options(state) or build_followup_options(
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
        missing = get_followup_missing_fields(data)
        draft = get_followup_draft(data)
        choice_hint = bool(data.get("choice_hint"))
        options = get_followup_options(data)
        if (missing and isinstance(draft, dict)) or (
            choice_hint and isinstance(options, list)
        ):
            followup_count = get_followup_count(data)
            payload = {
                "mode": "tool",
                "tool_name": tool_payload.name,
                "draft": draft if isinstance(draft, dict) else {},
                "missing_fields": missing,
                "followup_count": followup_count,
                "pending_message": tool_payload.message,
                "pending_kind": get_followup_kind(data),
            }
            query = data.get("query") or data.get("prompt")
            if isinstance(query, str) and query.strip():
                payload["query"] = query.strip()
            if choice_hint and isinstance(options, list):
                payload["choice_hint"] = True
                payload["strict_options_only"] = True
                payload["options"] = build_followup_options(
                    payload.get("pending_message"),
                    payload.get("draft"),
                    extra_options=options,
                )
            else:
                built = build_followup_options(
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
        if pending.get("mode") not in {"tool", "workflow", "clarification"}:
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
        missing_fields = get_followup_missing_fields(pending)
        if missing_fields and "variety" in missing_fields:
            if find_exact_variety_in_text(prompt):
                return True
            if retrieve_variety_candidates(prompt, limit=3):
                return True
        pending_kind = get_followup_kind(pending)
        if pending_kind == "clarification":
            return self._matches_pending_clarification(prompt, pending)
        if pending_kind == "confirmation":
            return self._matches_pending_confirmation(prompt)
        if pending.get("strict_options_only"):
            return self._matches_pending_choice(prompt, pending)
        if self._matches_pending_choice(prompt, pending):
            return True
        if self._looks_like_new_question(prompt):
            return False
        return self._looks_like_pending_field_reply(prompt, pending)

    @staticmethod
    def _extract_pending_candidates(pending: Optional[dict]) -> list[str]:
        if not isinstance(pending, dict):
            return []
        draft = get_followup_draft(pending)
        return extract_draft_options(draft)

    @staticmethod
    def _extract_pending_options(pending: Optional[dict]) -> list[str]:
        return get_followup_options(pending)

    @staticmethod
    def _extract_pending_message(pending: Optional[dict]) -> str:
        return get_followup_message(pending)

    def _matches_pending_choice(self, prompt: str, pending: Optional[dict]) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        options = self._extract_pending_options(pending)
        candidates = options or self._extract_pending_candidates(pending)
        if candidates:
            if resolve_followup_choice(text, candidates):
                return True
        message = self._extract_pending_message(pending)
        if message and len(text) <= 10 and text in message:
            return True
        if parse_followup_index(text) is not None and message and "序号" in message:
            return True
        return False

    @staticmethod
    def _matches_pending_confirmation(prompt: str) -> bool:
        text = (prompt or "").strip().lower()
        if not text:
            return False
        return text in _YES_REPLY_TOKENS or text in _NO_REPLY_TOKENS

    def resolve_pending_clarification_choice(
        self, prompt: str, pending: Optional[dict]
    ) -> Optional[str]:
        text = (prompt or "").strip().lower()
        if not text:
            return None
        if text in _CLARIFICATION_CONTINUE_TOKENS:
            return "contextual"
        if text in _CLARIFICATION_NEW_TOKENS:
            return "standalone"
        choice = resolve_followup_choice(text, self._extract_pending_options(pending))
        if not choice:
            return None
        if "继续当前" in choice:
            return "contextual"
        if "开启新" in choice or "新任务" in choice:
            return "standalone"
        return None

    def _matches_pending_clarification(
        self, prompt: str, pending: Optional[dict]
    ) -> bool:
        return self.resolve_pending_clarification_choice(prompt, pending) is not None

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
        if text.endswith(("吗", "么")):
            return True
        for token in _QUESTION_HINTS:
            if token in text:
                return True
        return False

    @staticmethod
    def _looks_like_pending_field_reply(prompt: str, pending: Optional[dict]) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        missing_fields = get_followup_missing_fields(pending)
        if not missing_fields:
            return False
        if text in _UNKNOWN_REPLY_TOKENS:
            return True
        pieces = [
            piece.strip()
            for piece in _FOLLOWUP_REPLY_SPLIT_RE.split(text)
            if piece.strip()
        ]
        if len(missing_fields) == 1:
            return len(text) <= 16
        if pieces and len(pieces) <= len(missing_fields) + 1 and len(text) <= 32:
            return True
        return len(text) <= 20
