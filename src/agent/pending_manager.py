from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from ..domain.date_parser import extract_explicit_dates
from .followup_extract import classify_pending_thread, extract_followup_overrides
from .followup import (
    build_followup_options,
    extract_draft_options,
    get_followup_count,
    get_followup_draft,
    get_followup_kind,
    get_followup_message,
    get_followup_missing_fields,
    get_followup_options,
    parse_followup_index,
    resolve_followup_choice,
    summarize_pending,
)
from ..infra.variety_store import find_exact_variety_in_text, retrieve_variety_candidates
from ..observability.logging_utils import log_event
from ..schemas import PlantingDetailsDraft, ToolInvocation


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
_ID_LIKE_REPLY_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_YES_REPLY_TOKENS = {"是", "好", "好的", "确认", "要", "保存", "继续", "yes", "y", "ok"}
_NO_REPLY_TOKENS = {"否", "不", "取消", "不用了", "不保存", "算了", "no", "n"}
_CLARIFICATION_CONTINUE_TOKENS = {
    "继续当前任务",
    "第一个",
    "1",
}
_CLARIFICATION_NEW_TOKENS = {
    "新任务",
    "开启新任务",
    "第二个",
    "2",
}
_FIELD_REPLY_TYPES = {
    "plant_season_id": "id_like",
    "plan_id": "id_like",
    "region": "region_like",
    "region_id": "region_like",
    "variety": "variety_like",
    "planting_method": "enum_like",
    "culti_type": "enum_like",
    "sowing_date": "date_like",
    "transplant_date": "date_like",
    "date": "date_like",
    "task_type": "enum_like",
    "name": "task_name_like",
    "operator": "operator_like",
    "work_desc": "work_desc_like",
}
_PLANTING_METHOD_REPLY_TOKENS = {
    "直播",
    "移栽",
    "插秧",
    "机插",
    "抛秧",
    "direct_seeding",
    "transplanting",
}
_CULTI_TYPE_REPLY_TOKENS = {
    "早稻",
    "中稻",
    "晚稻",
    "一季稻",
    "一季晚稻",
    "双季早稻",
    "双季晚稻",
}
_TASK_TYPE_REPLY_TOKENS = {"施肥", "打药", "其他"}
_EXPLICIT_NEW_TASK_TOKENS = (
    "开启新任务",
    "新任务",
    "换个问题",
    "换个任务",
    "重新开始",
    "先不说这个",
    "先不聊这个",
    "别管这个了",
    "另外一个问题",
)
_QUESTION_MARK_TOKENS = ("吗", "么", "？", "?")
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
        draft = get_followup_draft(pending)
        missing_fields = get_followup_missing_fields(pending)
        extraction = extract_followup_overrides(
            prompt,
            missing_fields,
            draft=draft if isinstance(draft, dict) else None,
        )
        merged_draft = dict(draft) if isinstance(draft, dict) else {}
        merged_draft.update(extraction.overrides)
        followup_payload = {
            "user_id": memory_id,
            "query": pending.get("query") if isinstance(pending, dict) else None,
            "followup": {
                "prompt": prompt,
                "draft": merged_draft,
                "missing_fields": missing_fields,
                "followup_count": get_followup_count(pending),
                "field_overrides": extraction.overrides,
                "source": extraction.source,
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

    def get_pending_disposition(self, prompt: str, pending: Optional[dict]) -> str:
        if not isinstance(pending, dict):
            return "drop"
        mode = str(pending.get("mode") or "").strip()
        if mode == "input_validation":
            return self._classify_input_validation_pending(prompt, pending)
        if mode not in {"tool", "workflow", "clarification"}:
            return "drop"
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
            return "drop"
        missing_fields = get_followup_missing_fields(pending)
        if missing_fields and "variety" in missing_fields:
            if find_exact_variety_in_text(prompt):
                return "resume_direct"
            if retrieve_variety_candidates(prompt, limit=3):
                return "resume_direct"
        pending_kind = get_followup_kind(pending)
        if pending_kind == "clarification":
            return (
                "resume_direct"
                if self._matches_pending_clarification(prompt, pending)
                else "drop"
            )
        if pending_kind == "confirmation":
            return (
                "resume_direct"
                if self._matches_pending_confirmation(prompt)
                else self._classify_pending_thread(prompt, pending, can_carry=False)
            )
        if pending.get("strict_options_only"):
            return (
                "resume_direct"
                if self._matches_pending_choice(prompt, pending)
                else self._classify_pending_thread(prompt, pending, can_carry=False)
            )
        if self._matches_pending_choice(prompt, pending):
            return "resume_direct"
        typed_match = self._matches_typed_pending_field_reply(prompt, pending)
        if typed_match is not None:
            return (
                "resume_direct"
                if typed_match
                else self._classify_pending_thread(prompt, pending)
            )
        if self._matches_structured_pending_reply(prompt, pending):
            return "resume_direct"
        return self._classify_pending_thread(prompt, pending)

    def should_resume_pending(self, prompt: str, pending: Optional[dict]) -> bool:
        return self.get_pending_disposition(prompt, pending) == "resume_direct"

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
        return PendingManager.resolve_pending_confirmation_choice(prompt) is not None

    @staticmethod
    def resolve_pending_confirmation_choice(prompt: str) -> Optional[str]:
        text = (prompt or "").strip().lower()
        if not text:
            return None
        if text in _YES_REPLY_TOKENS:
            return "yes"
        if text in _NO_REPLY_TOKENS:
            return "no"
        return None

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
        options = [str(item).strip().lower() for item in self._extract_pending_options(pending)]
        if text in options:
            if text == "继续当前任务":
                return "contextual"
            if text == "开启新任务":
                return "standalone"
        return None

    def _matches_pending_clarification(
        self, prompt: str, pending: Optional[dict]
    ) -> bool:
        return self.resolve_pending_clarification_choice(prompt, pending) is not None

    def _matches_typed_pending_field_reply(
        self, prompt: str, pending: Optional[dict]
    ) -> Optional[bool]:
        missing_fields = get_followup_missing_fields(pending)
        if len(missing_fields) != 1:
            return None
        field = missing_fields[0]
        reply_type = _FIELD_REPLY_TYPES.get(field)
        if not reply_type:
            return None
        if reply_type == "id_like":
            return self._matches_id_like_reply(prompt, pending)
        if reply_type == "enum_like":
            return self._matches_enum_like_reply(prompt, field)
        if reply_type == "date_like":
            return self._matches_date_like_reply(prompt)
        if reply_type == "region_like":
            return self._matches_region_like_reply(prompt)
        if reply_type == "variety_like":
            return self._matches_variety_like_reply(prompt)
        if reply_type == "task_name_like":
            return self._matches_task_name_like_reply(prompt)
        if reply_type == "operator_like":
            return self._matches_operator_like_reply(prompt)
        if reply_type == "work_desc_like":
            return self._matches_work_desc_like_reply(prompt)
        return None

    def _matches_id_like_reply(self, prompt: str, pending: Optional[dict]) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        if self._matches_pending_choice(text, pending):
            return True
        if text.isdigit():
            return True
        return _ID_LIKE_REPLY_RE.search(text) is not None

    @staticmethod
    def _matches_enum_like_reply(prompt: str, field: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        if text in _UNKNOWN_REPLY_TOKENS:
            return True
        if field == "planting_method":
            return any(token in text for token in _PLANTING_METHOD_REPLY_TOKENS)
        if field == "culti_type":
            return any(token in text for token in _CULTI_TYPE_REPLY_TOKENS)
        if field == "task_type":
            return any(token in text for token in _TASK_TYPE_REPLY_TOKENS)
        return False

    @staticmethod
    def _matches_date_like_reply(prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        if text in _UNKNOWN_REPLY_TOKENS:
            return True
        return bool(extract_explicit_dates(text, today=date.today()))

    @staticmethod
    def _matches_region_like_reply(prompt: str) -> bool:
        extraction = extract_followup_overrides(prompt, ("region_id",))
        return bool(extraction.overrides.get("region_id"))

    @staticmethod
    def _matches_variety_like_reply(prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        if text in _UNKNOWN_REPLY_TOKENS:
            return True
        if extract_followup_overrides(text, ("variety",)).overrides.get("variety"):
            return True
        if find_exact_variety_in_text(text):
            return True
        return bool(retrieve_variety_candidates(text, limit=3))

    @staticmethod
    def _matches_task_name_like_reply(prompt: str) -> bool:
        text = str(prompt or "").strip()
        if not text:
            return False
        if extract_followup_overrides(text, ("name",)).overrides.get("name"):
            return True
        return True

    @staticmethod
    def _matches_operator_like_reply(prompt: str) -> bool:
        return bool(
            extract_followup_overrides(prompt, ("operator",)).overrides.get("operator")
        )

    @staticmethod
    def _matches_work_desc_like_reply(prompt: str) -> bool:
        return bool(
            extract_followup_overrides(prompt, ("work_desc",)).overrides.get("work_desc")
        )

    def _classify_input_validation_pending(
        self, prompt: str, pending: Optional[dict]
    ) -> str:
        if not isinstance(pending, dict):
            return "drop"
        if self._looks_like_explicit_new_task(prompt):
            return "drop"
        draft = get_followup_draft(pending)
        fields = list(
            dict.fromkeys(
                get_followup_missing_fields(pending)
                + [str(item).strip() for item in list(pending.get("invalid_fields") or [])]
            )
        )
        extraction = extract_followup_overrides(
            prompt,
            fields,
            draft=draft if isinstance(draft, dict) else None,
        )
        if extraction.overrides:
            return "carry_pending"
        return self._classify_pending_thread(prompt, pending)

    def _classify_pending_thread(
        self,
        prompt: str,
        pending: Optional[dict],
        *,
        can_carry: bool = True,
    ) -> str:
        text = str(prompt or "").strip()
        if not text:
            return "drop"
        if self._looks_like_explicit_new_task(text):
            return "drop"
        if self._looks_like_standalone_question(text):
            return "drop"
        decision = classify_pending_thread(
            text,
            pending_summary=summarize_pending(pending) or {},
        )
        if decision is not None and float(decision.confidence or 0.0) >= 0.75:
            if decision.decision == "continue":
                return "carry_pending" if can_carry else "drop"
            if decision.decision == "new":
                return "drop"
        return "carry_pending" if can_carry else "drop"

    @staticmethod
    def _looks_like_explicit_new_task(prompt: str) -> bool:
        text = str(prompt or "").strip()
        if not text:
            return False
        return any(token in text for token in _EXPLICIT_NEW_TASK_TOKENS)

    @staticmethod
    def _looks_like_standalone_question(prompt: str) -> bool:
        text = str(prompt or "").strip()
        if not text:
            return False
        if any(token in text for token in _QUESTION_MARK_TOKENS):
            return True
        return any(
            token in text
            for token in ("天气", "播种", "播期", "方案", "计划", "删除", "生育期", "适合")
        )

    def _matches_structured_pending_reply(
        self, prompt: str, pending: Optional[dict]
    ) -> bool:
        text = (prompt or "").strip()
        if not text or not isinstance(pending, dict):
            return False
        missing_fields = get_followup_missing_fields(pending)
        if not missing_fields:
            return False
        if text in _UNKNOWN_REPLY_TOKENS:
            return True
        if any(field in {"plant_season_id", "plan_id"} for field in missing_fields):
            if self._matches_id_like_reply(text, pending):
                return True
        planting_overrides = extract_followup_overrides(
            text,
            tuple(
                field
                for field in missing_fields
                if field
                in {
                    "region_id",
                    "region",
                    "variety",
                    "culti_type",
                    "planting_method",
                    "sowing_date",
                    "transplant_date",
                    "crop",
                    "date",
                }
            ),
        ).overrides
        if "date" in missing_fields and (
            planting_overrides.get("sowing_date") or planting_overrides.get("transplant_date")
        ):
            return True
        if any(field in planting_overrides for field in missing_fields):
            return True
        for field in missing_fields:
            reply_type = _FIELD_REPLY_TYPES.get(field)
            if reply_type == "enum_like" and self._matches_enum_like_reply(text, field):
                return True
            if reply_type == "task_name_like" and self._matches_task_name_like_reply(text):
                return True
            if reply_type == "operator_like" and self._matches_operator_like_reply(text):
                return True
            if reply_type == "work_desc_like" and self._matches_work_desc_like_reply(text):
                return True
        return False
