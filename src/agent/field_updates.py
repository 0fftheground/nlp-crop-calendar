from __future__ import annotations

import re
from typing import Callable, Optional

from ..domain.planting import extract_planting_details
from ..infra.variety_store import find_exact_variety_in_text

_DEFAULT_INVALID_REGION_TOKENS = (
    "改成",
    "换成",
    "下周",
    "上周",
    "本周",
    "这周",
    "下星期",
    "上星期",
    "这星期",
    "本星期",
    "直播",
    "移栽",
    "插秧",
    "播种",
    "生育期",
    "方案",
    "施肥",
    "炼苗",
    "翻地",
    "打药",
    "喷药",
    "收割",
    "收获",
    "整地",
)
_REGION_PATTERNS = (
    r"^在([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)?$",
    r"^到([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)?$",
    r"^([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)$",
)


def extract_region_followup_hint(
    text: str,
    *,
    invalid_tokens: tuple[str, ...] = _DEFAULT_INVALID_REGION_TOKENS,
) -> Optional[str]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    for pattern in _REGION_PATTERNS:
        match = re.search(pattern, prompt)
        if not match:
            continue
        region = str(match.group(1) or "").strip()
        region = re.sub(r"(呢|吗|呀|啊)$", "", region).strip()
        if region and not any(token in region for token in invalid_tokens):
            return region
    return None


def extract_planting_field_overrides(
    text: str,
    *,
    include_variety: bool = False,
    include_dates: bool = True,
    include_crop: bool = True,
    variety_matcher: Optional[Callable[[str], Optional[str]]] = None,
) -> dict[str, object]:
    prompt = str(text or "").strip()
    if not prompt:
        return {}
    overrides: dict[str, object] = {}
    region = extract_region_followup_hint(prompt)
    if region:
        overrides["region_id"] = region
    try:
        draft = extract_planting_details(prompt, variety_resolver=lambda _value: [])
    except Exception:
        draft = None
    if draft is not None:
        keys = ["culti_type", "planting_method"]
        if include_dates:
            keys.extend(["sowing_date", "transplant_date"])
        if include_crop:
            keys.append("crop")
        for key in keys:
            value = getattr(draft, key, None)
            if value in (None, ""):
                continue
            overrides[key] = value.isoformat() if hasattr(value, "isoformat") else value
    if include_variety:
        matcher = variety_matcher or find_exact_variety_in_text
        variety = matcher(prompt)
        if variety:
            overrides["variety"] = variety
    return overrides
