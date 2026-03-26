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
_CROP_CALENDAR_PLAN_CUES = (
    "计划",
    "方案",
    "建立",
    "生成",
    "制定",
    "创建",
    "新建",
    "做一个",
    "做一份",
)
_PLANTING_CONTEXT_CUES = (
    "种植",
    "品种",
    "移栽",
    "直播",
    "播种",
    "播期",
    "稻",
)
_NON_AGRI_LIFE_DOMAIN_CUES = (
    "旅游",
    "旅行",
    "出游",
    "游玩",
    "景点",
    "酒店",
    "住宿",
    "航班",
    "机票",
    "高铁",
    "火车票",
    "出差",
    "通勤",
    "穿衣",
    "紫外线",
    "洗车",
)


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


def looks_like_non_agri_life_query(text: str) -> bool:
    prompt = normalize_prompt(text)
    if not prompt:
        return False
    return any(token in prompt for token in _NON_AGRI_LIFE_DOMAIN_CUES)


def looks_like_crop_calendar_query(text: str) -> bool:
    prompt = normalize_prompt(text)
    if not prompt:
        return False
    if not any(token in prompt for token in _CROP_CALENDAR_PLAN_CUES):
        return False
    return any(token in prompt for token in _PLANTING_CONTEXT_CUES)
