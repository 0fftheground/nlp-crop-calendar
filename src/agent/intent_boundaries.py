from __future__ import annotations

import re

_SOWING_QUERY_CUES = (
    "播种",
    "播期",
    "适播",
)
_SOWING_INTENT_CUES = (
    "适合",
    "适宜",
    "什么时候",
    "何时",
    "窗口",
    "推荐",
    "怎么播",
    "播吗",
    "播嘛",
    "播呢",
)
_PLAN_QUERY_CUES = ("计划", "方案", "生成", "制定", "新增", "创建")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def looks_like_sowing_query(text: str) -> bool:
    prompt = normalize_prompt(text)
    if not prompt:
        return False
    if any(token in prompt for token in _PLAN_QUERY_CUES):
        return False
    if not any(token in prompt for token in _SOWING_QUERY_CUES):
        return False
    if any(token in prompt for token in _SOWING_INTENT_CUES):
        return True
    return len(prompt) <= 12
