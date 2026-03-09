from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from ..schemas.models import ToolInvocation


_FOLLOWUP_INDEX_RE = re.compile(r"^第?\s*(\d+)\s*(?:个|条|项)?$")
_FOLLOWUP_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']+)[\"“”']")


def get_followup_draft(payload: Optional[Mapping[str, object]]) -> object:
    if not isinstance(payload, Mapping):
        return None
    return payload.get("draft")


def get_followup_options(payload: Optional[Mapping[str, object]]) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    options = payload.get("options")
    if not isinstance(options, list):
        return []
    return _dedupe_options(str(item) for item in options)


def get_followup_message(payload: Optional[Mapping[str, object]]) -> str:
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get("pending_message") or payload.get("message")
    return value.strip() if isinstance(value, str) else ""


def get_followup_missing_fields(payload: Optional[Mapping[str, object]]) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    missing = payload.get("missing_fields")
    if not isinstance(missing, list):
        return []
    return [str(item).strip() for item in missing if str(item).strip()]


def get_followup_count(payload: Optional[Mapping[str, object]]) -> int:
    if not isinstance(payload, Mapping):
        return 0
    try:
        return int(payload.get("followup_count") or 0)
    except (TypeError, ValueError):
        return 0


def parse_followup_index(text: str) -> Optional[int]:
    if not text:
        return None
    match = _FOLLOWUP_INDEX_RE.match(text.strip())
    if not match:
        return None
    return int(match.group(1))


def extract_message_options(message: str) -> list[str]:
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
    if options:
        return _dedupe_options(options)
    for token in _FOLLOWUP_QUOTED_RE.findall(message):
        for piece in re.split(r"[、/或]", token):
            piece = piece.strip()
            if piece:
                options.append(piece)
    return _dedupe_options(options)


def extract_draft_options(draft: Optional[Mapping[str, object]]) -> list[str]:
    if not isinstance(draft, Mapping):
        return []
    options: list[str] = []
    for key in ("options", "candidates", "variety_candidates", "region_candidates"):
        value = draft.get(key)
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    options.append(text)
    return _dedupe_options(options)


def build_followup_options(
    message: Optional[str],
    draft: Optional[Mapping[str, object]] = None,
    extra_options: Optional[Iterable[object]] = None,
) -> list[str]:
    options: list[str] = []
    options.extend(extract_draft_options(draft))
    if extra_options is not None:
        for item in extra_options:
            text = str(item).strip()
            if text:
                options.append(text)
    if isinstance(message, str) and message.strip():
        options.extend(extract_message_options(message))
    return _dedupe_options(options)


def resolve_followup_choice(answer: str, options: list[str]) -> Optional[str]:
    text = (answer or "").strip()
    if not text or not options:
        return None
    index = parse_followup_index(text)
    if index is not None and 1 <= index <= len(options):
        return options[index - 1]
    for option in options:
        if option == text:
            return option
    for option in options:
        if text in option or option in text:
            return option
    return None


def render_followup_message(
    *,
    pending_message: Optional[str],
    missing_fields: list[str],
    field_labels: Mapping[str, str],
    default_prefix: str,
    allow_unknown: bool = False,
    optional_fields: Optional[list[str]] = None,
    options: Optional[list[str]] = None,
    options_intro: Optional[str] = None,
    reply_hint: Optional[str] = None,
) -> str:
    if isinstance(pending_message, str) and pending_message.strip():
        return pending_message
    cleaned_options = _dedupe_options(options or [])
    if cleaned_options:
        lines = [options_intro or default_prefix]
        for idx, option in enumerate(cleaned_options, start=1):
            lines.append(f"{idx}. {option}")
        if reply_hint:
            lines.append(reply_hint)
        return "\n".join(lines)
    labels = [field_labels.get(field, field) for field in missing_fields]
    if optional_fields:
        for field in optional_fields:
            if field in missing_fields:
                continue
            labels.append(f"{field_labels.get(field, field)}(可选)")
    joined = "、".join(labels) if labels else default_prefix
    message = f"{default_prefix}{joined}。"
    if allow_unknown:
        message = (
            f"{message}如果不清楚，可以直接回复“不知道/不确定”，我会使用默认值继续。"
        )
    return message


def summarize_pending(pending: Optional[dict]) -> Optional[dict]:
    if not isinstance(pending, dict):
        return None
    name = pending.get("tool_name") or pending.get("workflow_name") or pending.get(
        "name"
    )
    summary: dict[str, Any] = {
        "mode": pending.get("mode"),
        "name": name,
        "missing_fields": get_followup_missing_fields(pending),
        "followup_count": get_followup_count(pending),
    }
    if "action" in pending:
        summary["action"] = pending.get("action")
    if "input_attempts" in pending:
        summary["input_attempts"] = pending.get("input_attempts")
    return summary


def is_followup_payload(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    missing = get_followup_missing_fields(data)
    if missing:
        return True
    if data.get("choice_hint") and get_followup_options(data):
        return True
    if data.get("source") == "candidate":
        return True
    draft = get_followup_draft(data)
    if isinstance(draft, dict) and build_followup_options(None, draft):
        return True
    return False


def build_tool_followup_data(
    *,
    missing_fields: list[str],
    draft: Optional[Mapping[str, object]] = None,
    query: Optional[str] = None,
    followup_count: int = 0,
    options: Optional[Iterable[object]] = None,
    choice_hint: bool = False,
    strict_options_only: bool = False,
    source: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(extra or {})
    draft_payload = dict(draft) if isinstance(draft, Mapping) else {}
    if isinstance(query, str) and query.strip():
        payload["query"] = query.strip()
    if source:
        payload["source"] = source
    payload["missing_fields"] = list(missing_fields or [])
    payload["draft"] = draft_payload
    payload["followup_count"] = followup_count
    option_values = build_followup_options(
        None,
        draft_payload,
        extra_options=options or get_followup_options(payload),
    )
    if option_values:
        payload["options"] = option_values
    else:
        payload.pop("options", None)
    if choice_hint:
        payload["choice_hint"] = True
    elif "choice_hint" in payload:
        payload["choice_hint"] = bool(payload["choice_hint"])
    if strict_options_only:
        payload["strict_options_only"] = True
    elif "strict_options_only" in payload:
        payload["strict_options_only"] = bool(payload["strict_options_only"])
    return payload


def build_tool_followup_invocation(
    *,
    name: str,
    message: str,
    missing_fields: list[str],
    draft: Optional[Mapping[str, object]] = None,
    query: Optional[str] = None,
    followup_count: int = 0,
    options: Optional[Iterable[object]] = None,
    choice_hint: bool = False,
    strict_options_only: bool = False,
    source: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> ToolInvocation:
    return ToolInvocation(
        name=name,
        message=message,
        data=build_tool_followup_data(
            missing_fields=missing_fields,
            draft=draft,
            query=query,
            followup_count=followup_count,
            options=options,
            choice_hint=choice_hint,
            strict_options_only=strict_options_only,
            source=source,
            extra=extra,
        ),
    )


def build_workflow_followup_update(
    *,
    draft: object = None,
    missing_fields: Optional[list[str]] = None,
    followup_count: Optional[int] = None,
    pending_message: Optional[str] = None,
    options: Optional[Iterable[object]] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(extra or {})
    if draft is not None:
        payload["draft"] = draft
    if missing_fields is not None:
        payload["missing_fields"] = list(missing_fields)
    if followup_count is not None:
        payload["followup_count"] = followup_count
    payload["pending_message"] = pending_message
    if options is not None:
        option_values = _dedupe_options(str(item) for item in options)
        payload["options"] = option_values
    else:
        draft_mapping = _coerce_mapping(draft)
        built_options = build_followup_options(pending_message, draft_mapping)
        if built_options:
            payload["options"] = built_options
    return payload


def clear_workflow_followup_update(
    *, extra: Optional[Mapping[str, object]] = None
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(extra or {})
    payload.update(
        {
            "missing_fields": [],
            "options": [],
            "pending_message": None,
        }
    )
    return payload


def _coerce_mapping(value: object) -> Optional[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _dedupe_options(options: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for item in options:
        text = str(item).strip()
        if text and text not in unique:
            unique.append(text)
    return unique
