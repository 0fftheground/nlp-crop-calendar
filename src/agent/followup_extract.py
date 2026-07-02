from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel

from ..domain.date_parser import extract_explicit_dates
from ..infra.llm_extract import llm_structured_extract
from .field_updates import extract_field_overrides, extract_region_followup_hint

_WEAK_FIELDS = {
    "region",
    "region_id",
    "region_choice",
    "variety",
    "name",
    "operator",
    "work_desc",
    "plan_name",
    "crop",
    "query",
    "prompt",
    "reason",
}
_REGION_TEXT_RE = re.compile(r"^[\u4e00-\u9fff]{2,20}$")
_REGION_INVALID_TOKENS = (
    "取消",
    "追问",
    "问题",
    "任务",
    "继续",
    "开启",
    "删除",
    "播种",
    "播期",
    "方案",
    "计划",
    "天气",
    "适合",
    "生育期",
)


class _FollowupWeakFieldExtraction(BaseModel):
    region: Optional[str] = None
    region_id: Optional[str] = None
    region_choice: Optional[str] = None
    variety: Optional[str] = None
    name: Optional[str] = None
    operator: Optional[str] = None
    work_desc: Optional[str] = None
    plan_name: Optional[str] = None
    crop: Optional[str] = None
    query: Optional[str] = None
    prompt: Optional[str] = None
    reason: Optional[str] = None


class _PendingThreadDecision(BaseModel):
    decision: str = "unknown"
    confidence: float = 0.0
    reason: Optional[str] = None


@dataclass(frozen=True)
class FollowupExtractionResult:
    overrides: dict[str, object]
    source: str


_FOLLOWUP_WEAK_FIELD_PROMPT = (
    "你是追问补字段抽取器。"
    "给定当前用户回复、允许补充的字段、以及已有 draft，只抽取当前回复里明确提供的缺失字段。"
    "不要改写已有字段，不要猜测，不要补造。"
    "如果用户是在问新问题，而不是补当前字段，应尽量返回空字段。"
    "输出严格 JSON。"
)

_PENDING_THREAD_PROMPT = (
    "你是多轮追问线程判定器。"
    "判断当前用户输入是在继续补充当前 pending，还是已经开启了新的问题。"
    "只输出 decision=continue|new|unknown。"
    "continue 只用于明显在补 missing_fields、做选项选择、或继续当前追问。"
    "new 只用于明显开启了与当前 pending 无关的新任务。"
    "不确定时输出 unknown。"
    "输出严格 JSON："
    '{"decision":"continue|new|unknown","confidence":0-1,"reason":"..."}'
)


def extract_followup_overrides(
    prompt: str,
    allowed_fields: Iterable[str],
    *,
    draft: Optional[Mapping[str, object]] = None,
) -> FollowupExtractionResult:
    text = str(prompt or "").strip()
    fields = tuple(
        field for field in (str(item).strip() for item in allowed_fields) if field
    )
    if not text or not fields:
        return FollowupExtractionResult(overrides={}, source="none")
    rule_overrides = extract_field_overrides(text, fields)
    if "date" in fields and "date" not in rule_overrides:
        explicit_dates = extract_explicit_dates(text)
        if explicit_dates:
            rule_overrides["date"] = explicit_dates[0].isoformat()
    unresolved_weak_fields = [
        field for field in fields if field in _WEAK_FIELDS and field not in rule_overrides
    ]
    if _REGION_TEXT_RE.fullmatch(text) and not any(
        token in text for token in _REGION_INVALID_TOKENS
    ):
        for field in ("region", "region_id", "region_choice"):
            if field in unresolved_weak_fields and field not in rule_overrides:
                rule_overrides[field] = text
    for field in ("query", "prompt", "reason"):
        if field in unresolved_weak_fields and text:
            rule_overrides[field] = text
    unresolved_weak_fields = [
        field for field in unresolved_weak_fields if field not in rule_overrides
    ]
    if not unresolved_weak_fields:
        return FollowupExtractionResult(overrides=rule_overrides, source="rule")
    llm_overrides = _extract_weak_followup_fields(
        text, unresolved_weak_fields, draft=draft
    )
    merged = dict(rule_overrides)
    for key, value in llm_overrides.items():
        if key not in merged and value not in (None, "", []):
            merged[key] = value
    if rule_overrides and llm_overrides:
        source = "mixed"
    elif llm_overrides:
        source = "llm"
    else:
        source = "rule"
    return FollowupExtractionResult(overrides=merged, source=source)


def classify_pending_thread(
    prompt: str,
    *,
    pending_summary: Mapping[str, object],
) -> Optional[_PendingThreadDecision]:
    text = str(prompt or "").strip()
    if not text:
        return None
    payload = {
        "prompt": text,
        "pending": dict(pending_summary or {}),
    }
    extracted = llm_structured_extract(
        json.dumps(payload, ensure_ascii=False, default=str),
        schema=_PendingThreadDecision,
        system_prompt=_PENDING_THREAD_PROMPT,
    )
    if not extracted:
        return None
    try:
        return _PendingThreadDecision.model_validate(extracted)
    except Exception:
        return None


def _extract_weak_followup_fields(
    prompt: str,
    allowed_fields: list[str],
    *,
    draft: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    payload = {
        "prompt": prompt,
        "allowed_fields": allowed_fields,
        "draft": dict(draft or {}),
    }
    extracted = llm_structured_extract(
        json.dumps(payload, ensure_ascii=False, default=str),
        schema=_FollowupWeakFieldExtraction,
        system_prompt=_FOLLOWUP_WEAK_FIELD_PROMPT,
    )
    if not extracted:
        return {}
    return _validate_weak_followup_fields(extracted, allowed_fields)


def _validate_weak_followup_fields(
    extracted: Mapping[str, object], allowed_fields: Iterable[str]
) -> dict[str, object]:
    allowed = {str(field).strip() for field in allowed_fields if str(field).strip()}
    validated: dict[str, object] = {}
    for key in allowed:
        value = extracted.get(key)
        if value in (None, "", []):
            continue
        text = str(value).strip()
        if not text:
            continue
        if key in {"region", "region_id", "region_choice"}:
            region = extract_region_followup_hint(text)
            if not region and _REGION_TEXT_RE.fullmatch(text):
                region = text
            if region:
                validated[key] = region
            continue
        if key in {"query", "prompt", "reason"}:
            validated[key] = text
            continue
        validated[key] = text
    return validated
