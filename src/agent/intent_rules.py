from __future__ import annotations

from typing import Optional, Tuple


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
CONTROL_MEMORY_CLEAR_KEYWORDS = [
    "清除记忆",
    "清空记忆",
    "删除记忆",
    "清除偏好",
    "清空偏好",
    "删除偏好",
    "忘记之前",
    "忘掉之前",
    "清掉记忆",
]


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


def is_memory_clear_request(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    return _contains_any(text, CONTROL_MEMORY_CLEAR_KEYWORDS)


def strip_memory_clear(prompt: str) -> str:
    text = prompt or ""
    for keyword in CONTROL_MEMORY_CLEAR_KEYWORDS:
        text = text.replace(keyword, "")
    return text.strip(" \t,.;:!?，。！？；：")


def is_cancel_request(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    return _contains_any(text, CONTROL_CANCEL_KEYWORDS)


def is_cancel_only(prompt: str) -> bool:
    return _is_cancel_only(prompt)


def strip_cancel_request(prompt: str) -> str:
    text = prompt or ""
    for keyword in CONTROL_CANCEL_KEYWORDS:
        text = text.replace(keyword, "")
    return text.strip(" \t,.;:!?，。！？；：")


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
