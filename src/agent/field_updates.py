from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

from ..domain.planting import extract_planting_details
from ..domain.date_parser import extract_date_range, extract_explicit_dates
from ..application.services.weather_service import (
    extract_weather_operations,
    normalize_weather_prompt,
)
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
_PLAN_ID_REPLY_RE = re.compile(
    r"(?:plant_season_id|plan_id|计划id|计划编号|id)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_PLAN_TASK_NAME_ALIASES = (
    "施基肥",
    "施蘖肥",
    "施穗肥",
    "送嫁肥",
    "封闭除草",
    "封杀除草",
    "水整地",
    "晒田",
    "收割前晒田",
    "收割",
    "播种",
    "移栽",
    "施肥",
    "追肥",
    "打药",
    "喷药",
    "除草",
)
_PLAN_TASK_TYPE_ALIASES = {
    "施肥": "施肥",
    "追肥": "施肥",
    "叶面肥": "施肥",
    "打药": "打药",
    "喷药": "打药",
    "喷施": "打药",
    "除草": "打药",
    "杀虫": "打药",
    "杀菌": "打药",
}
_TASK_OPERATOR_RE = re.compile(r"(?:操作人|执行人|负责人|作业人)[:：]?\s*([^\s，。；,;]+)")
_TASK_WORK_DESC_RE = re.compile(r"(?:备注|说明|详情|内容|工作内容)[:：]\s*(.+)$")


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


def extract_field_overrides(
    text: str,
    allowed_fields: Iterable[str],
    *,
    variety_matcher: Optional[Callable[[str], Optional[str]]] = None,
) -> dict[str, object]:
    prompt = str(text or "").strip()
    fields = {str(field).strip() for field in allowed_fields if str(field).strip()}
    if not prompt or not fields:
        return {}
    overrides: dict[str, object] = {}
    weather_fields = {
        "region",
        "start_date",
        "end_date",
        "requested_operations",
        "granularity",
        "include_advice",
    }
    if fields & weather_fields:
        overrides.update(_extract_weather_field_overrides(prompt, fields))
    planting_fields = {
        "region_id",
        "region_choice",
        "variety",
        "culti_type",
        "planting_method",
        "sowing_date",
        "transplant_date",
        "crop",
    }
    if fields & planting_fields:
        planting_overrides = extract_planting_field_overrides(
            prompt,
            include_variety="variety" in fields,
            include_dates=bool({"sowing_date", "transplant_date"} & fields),
            include_crop="crop" in fields,
            variety_matcher=variety_matcher,
        )
        for key, value in planting_overrides.items():
            if key in fields:
                overrides[key] = value
        if "region_choice" in fields and planting_overrides.get("region_id") not in (None, ""):
            overrides["region_choice"] = planting_overrides["region_id"]
    if {"plan_id", "plant_season_id"} & fields:
        plan_id = _extract_plan_id_like(prompt)
        if plan_id:
            if "plan_id" in fields:
                overrides["plan_id"] = plan_id
            if "plant_season_id" in fields:
                overrides["plant_season_id"] = plan_id
    if {"name", "task_type", "operator", "work_desc"} & fields:
        overrides.update(_extract_plan_task_field_overrides(prompt, fields))
    if "date" in fields:
        explicit_dates = extract_explicit_dates(prompt)
        if explicit_dates:
            overrides["date"] = explicit_dates[0].isoformat()
    return overrides


def _extract_weather_field_overrides(
    prompt: str, fields: set[str]
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    supported_ops, unsupported_ops = extract_weather_operations(
        prompt, require_suitability_cues=False
    )
    _, query = normalize_weather_prompt(prompt)
    if query is not None:
        payload = query.model_dump(mode="json")
        for key in ("region", "start_date", "end_date", "granularity", "include_advice"):
            if key in fields:
                value = payload.get(key)
                if value not in (None, ""):
                    overrides[key] = value
    else:
        parsed_range = extract_date_range(prompt)
        if parsed_range:
            if "start_date" in fields:
                overrides["start_date"] = parsed_range[0].isoformat()
            if "end_date" in fields:
                overrides["end_date"] = parsed_range[1].isoformat()
        if "region" in fields and not supported_ops and not unsupported_ops:
            region = extract_region_followup_hint(prompt)
            if region:
                overrides["region"] = region
    if "requested_operations" in fields and supported_ops:
        overrides["requested_operations"] = supported_ops
    return overrides


def _extract_plan_id_like(prompt: str) -> Optional[str]:
    match = _PLAN_ID_REPLY_RE.search(prompt)
    if match:
        return str(match.group(1) or "").strip() or None
    if prompt.isdigit():
        return prompt
    return None


def _extract_plan_task_field_overrides(
    prompt: str, fields: set[str]
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if "name" in fields:
        for name in sorted(_PLAN_TASK_NAME_ALIASES, key=len, reverse=True):
            if name in prompt:
                overrides["name"] = name
                break
    if "task_type" in fields:
        for key, value in sorted(
            _PLAN_TASK_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if key in prompt:
                overrides["task_type"] = value
                break
    if "operator" in fields:
        operator_match = _TASK_OPERATOR_RE.search(prompt)
        if operator_match:
            operator = str(operator_match.group(1) or "").strip()
            if operator:
                overrides["operator"] = operator
    if "work_desc" in fields:
        work_desc_match = _TASK_WORK_DESC_RE.search(prompt)
        if work_desc_match:
            work_desc = str(work_desc_match.group(1) or "").strip()
            if work_desc:
                overrides["work_desc"] = work_desc
    return overrides
