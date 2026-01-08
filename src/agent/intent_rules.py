from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel

from ..domain.services import (
    CROP_KEYWORDS,
    DATE_PATTERN,
    METHOD_KEYWORDS,
    REGION_PATTERN,
    VARIETY_FALLBACK_PATTERN,
    VARIETY_PATTERN,
)
from ..infra.llm import get_extractor_model
from ..observability.logging_utils import log_event, summarize_text
from .workflows.common import UNKNOWN_MARKERS


WEATHER_KEYWORDS = [
    "天气",
    "气温",
    "温度",
    "降雨",
    "降水",
    "预报",
    "气象",
]
VARIETY_KEYWORDS = ["品种", "品系", "抗性", "特性"]
PLAN_KEYWORDS = ["全流程", "流程", "完整", "方案", "计划", "整季", "全季", "详细"]
RECOMMEND_KEYWORDS = ["建议", "注意", "该做什么", "需要做什么", "怎么做"]
GROWTH_STAGE_CUES = ["预测", "当前", "判断", "现在"]
PLANTING_KEYWORDS = ["播种", "插秧", "直播", "移栽", "播期", "播种日期"]
CONTROL_CANCEL_KEYWORDS = [
    "取消追问",
    "结束追问",
    "开始新问题",
    "换个问题",
    "重新开始",
    "不继续",
    "取消",
]
CONTROL_NEW_QUESTION_KEYWORDS = ["开始新问题", "换个问题", "重新开始"]
CONTROL_RELATIVE_DATE_MARKERS = [
    "今天",
    "明天",
    "后天",
    "本周",
    "下周",
    "本月",
    "下月",
    "今年",
    "明年",
    "近期",
    "最近",
]
CONTROL_MONTH_DAY_PATTERN = re.compile(r"(?:^|\\D)(\\d{1,2})月(\\d{1,2})[日号]?")
CONTROL_MONTH_PATTERN = re.compile(r"(?:^|\\D)(\\d{1,2})月(?:上旬|中旬|下旬|初|底)?")
CONTROL_SOWING_TOKENS = ("播种", "播期", "种", "栽")


class ControlIntentResult(BaseModel):
    intent: str
    reason: Optional[str] = None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(word in text for word in keywords)


def _is_growth_stage_query(prompt: str) -> bool:
    if "生育期" in prompt:
        if _contains_any(prompt, GROWTH_STAGE_CUES):
            return True
        if _contains_any(prompt, PLANTING_KEYWORDS):
            return True
    return _contains_any(prompt, ["生长阶段", "生育阶段", "当前阶段"])


def _is_recommendation_query(prompt: str) -> bool:
    if _contains_any(prompt, RECOMMEND_KEYWORDS):
        return True
    return False


def classify_intent(prompt: str) -> Tuple[str, Optional[str]]:
    text = (prompt or "").strip()
    if not text:
        return "none", None

    if _contains_any(text, WEATHER_KEYWORDS):
        return "tool", "weather_lookup"
    if _is_growth_stage_query(text):
        return "workflow", None
    if _contains_any(text, VARIETY_KEYWORDS):
        return "tool", "variety_lookup"
    if _contains_any(text, PLAN_KEYWORDS):
        return "workflow", None
    if _is_recommendation_query(text):
        return "tool", "farming_recommendation"
    return "none", None


def _is_cancel_only(prompt: str) -> bool:
    text = (prompt or "").strip()
    for keyword in CONTROL_CANCEL_KEYWORDS:
        text = text.replace(keyword, "")
    text = text.strip(" \t,.;:!?，。！？；：")
    return not text


def _has_weather_intent(text: str) -> bool:
    return _contains_any(text, WEATHER_KEYWORDS)


def _has_date_signal(text: str, *, weather_intent: bool) -> bool:
    sowing_context = any(token in text for token in CONTROL_SOWING_TOKENS)
    if weather_intent and not sowing_context:
        return False
    if DATE_PATTERN.search(text):
        return True
    if CONTROL_MONTH_DAY_PATTERN.search(text) or CONTROL_MONTH_PATTERN.search(text):
        return True
    if _contains_any(text, CONTROL_RELATIVE_DATE_MARKERS):
        if len(text) <= 6 or sowing_context:
            return True
    return False


def _has_region_signal(text: str, *, weather_intent: bool) -> bool:
    if weather_intent:
        return False
    if REGION_PATTERN.search(text):
        return True
    return _contains_any(text, ["地区", "区域", "位置"])


def _has_followup_signal(text: str, missing_fields: List[str]) -> bool:
    weather_intent = _has_weather_intent(text)
    if "crop" in missing_fields and _contains_any(text, CROP_KEYWORDS):
        return True
    if "variety" in missing_fields and (
        VARIETY_PATTERN.search(text) or VARIETY_FALLBACK_PATTERN.search(text)
    ):
        return True
    if "planting_method" in missing_fields and _contains_any(
        text, list(METHOD_KEYWORDS)
    ):
        return True
    if "sowing_date" in missing_fields and _has_date_signal(
        text, weather_intent=weather_intent
    ):
        return True
    if "region" in missing_fields and _has_region_signal(
        text, weather_intent=weather_intent
    ):
        return True
    return False


def _llm_classify_control_intent(
    prompt: str, missing_fields: List[str]
) -> Optional[str]:
    try:
        llm = get_extractor_model()
    except Exception:
        return None
    system_prompt = (
        "你是对话控制分类器，负责判断用户是否在继续补充种植信息。"
        "输出 intent 只能是 cancel / continue / new_question。"
        "cancel: 用户明确表示取消/停止追问。"
        "continue: 用户在补充缺失字段或表示不知道/不确定。"
        "new_question: 用户提出与缺失字段无关的新问题或切换话题。"
    )
    payload = json.dumps(
        {"missing_fields": missing_fields, "user_prompt": prompt},
        ensure_ascii=True,
        default=str,
    )
    try:
        classifier = llm.with_structured_output(ControlIntentResult)
        result = classifier.invoke([("system", system_prompt), ("human", payload)])
        intent = result.intent if isinstance(result, ControlIntentResult) else None
        log_event(
            "control_intent_llm_response",
            intent=intent,
            reason=summarize_text(getattr(result, "reason", "")),
        )
        return intent
    except Exception:
        return None


class ControlIntentRouter:
    def __init__(self, *, enable_llm: bool = True) -> None:
        self._enable_llm = enable_llm

    def route(self, prompt: str, *, pending: Optional[dict] = None) -> str:
        text = (prompt or "").strip()
        if not text:
            return "new_question"
        if _contains_any(text, CONTROL_CANCEL_KEYWORDS):
            return "cancel_only" if _is_cancel_only(text) else "cancel"
        if _contains_any(text, CONTROL_NEW_QUESTION_KEYWORDS):
            return "new_question"
        if _contains_any(text, UNKNOWN_MARKERS):
            return "continue"

        intent_mode, _ = classify_intent(text)
        if intent_mode == "tool":
            return "new_question"

        missing_fields = []
        if isinstance(pending, dict):
            missing_fields = pending.get("missing_fields") or []
        if missing_fields and _has_followup_signal(text, missing_fields):
            return "continue"
        if self._enable_llm and missing_fields:
            intent = _llm_classify_control_intent(text, missing_fields)
            if intent in {"cancel", "continue", "new_question"}:
                return intent
        return "continue" if missing_fields else "new_question"


class IntentRouter:
    def __init__(self) -> None:
        self._pending_workflows: set[str] = set()

    def route(self, prompt: str, session_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        if session_id and session_id in self._pending_workflows:
            return "workflow", None

        mode, tool = classify_intent(prompt)
        if session_id and mode == "workflow":
            self._pending_workflows.add(session_id)
        return mode, tool
