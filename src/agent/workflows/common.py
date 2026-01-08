"""
Shared helpers for LangGraph workflows.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ...infra.config import get_config
from ...infra.llm_extract import llm_structured_extract
from ...infra.variety_store import build_variety_hint
from ...schemas import PlantingDetails, PlantingDetailsDraft


UNKNOWN_MARKERS = ["不知道", "不清楚", "不确定", "记不清", "不记得", "忘了"]
MEMORY_ACCEPT_MARKERS = [
    "沿用",
    "同上",
    "用上次",
    "还是上次",
    "按上次",
    "继续用",
    "用之前的",
]
MEMORY_DENY_MARKERS = [
    "不用",
    "不沿用",
    "不用上次",
    "不要用",
    "不用之前的",
    "不用了",
    "重新",
]
MEMORY_YES_MARKERS = ["是", "好", "好的", "可以", "行"]
MEMORY_NO_MARKERS = ["否", "不是", "不可以", "不需要"]
MEMORY_METHOD_LABELS = {
    "direct_seeding": "直播",
    "transplanting": "移栽",
}


class PlantingExtract(BaseModel):
    crop: Optional[str] = None
    variety: Optional[str] = None
    planting_method: Optional[str] = Field(
        default=None, description="direct_seeding 或 transplanting"
    )
    sowing_date: Optional[date] = None
    transplant_date: Optional[date] = None
    region: Optional[str] = None
    planting_location: Optional[str] = None
    notes: Optional[str] = None


def infer_unknown_fields(
    prompt: str, missing_fields: List[str], field_labels: Dict[str, str]
) -> List[str]:
    if not missing_fields:
        return []
    if not any(marker in prompt for marker in UNKNOWN_MARKERS):
        return []

    unknown_fields: List[str] = []
    for field in missing_fields:
        label = field_labels.get(field)
        if label and label in prompt:
            unknown_fields.append(field)

    return unknown_fields or list(missing_fields)


def llm_extract_planting(
    prompt: str, *, schema: Type[BaseModel] = PlantingExtract
) -> Dict[str, object]:
    system_prompt = (
        "你是农事助手，负责从用户描述中抽取种植信息。"
        "只输出可确定的信息；不确定或未提及时保持为空。"
        "种植方式使用 direct_seeding 或 transplanting。"
        "日期格式为 YYYY-MM-DD。"
    )
    hint = build_variety_hint(prompt)
    if hint:
        system_prompt = f"{system_prompt}{hint}"
    return llm_structured_extract(
        prompt,
        schema=schema,
        system_prompt=system_prompt,
    )


def build_fallback_planting(draft: PlantingDetailsDraft) -> PlantingDetails:
    cfg = get_config()
    today = date.today()
    return PlantingDetails(
        crop=draft.crop or "水稻",
        variety=draft.variety,
        planting_method=draft.planting_method or "direct_seeding",
        sowing_date=draft.sowing_date or today,
        transplant_date=draft.transplant_date,
        region=draft.region or cfg.default_region,
        planting_location=draft.planting_location,
    )


def format_missing_question(
    missing_fields: List[str],
    field_labels: Dict[str, str],
    prefix: str,
    *,
    allow_unknown: bool = True,
) -> str:
    labels = [field_labels.get(field, field) for field in missing_fields]
    joined = "、".join(labels)
    message = f"{prefix}{joined}。"
    if allow_unknown:
        message = (
            f"{message}如果不清楚，可以直接回复“不知道/不确定”，我会使用默认值继续。"
        )
    return message


def classify_memory_confirmation(
    prompt: str, *, prompted: bool = False
) -> Optional[bool]:
    if not prompt:
        return None
    if any(marker in prompt for marker in MEMORY_ACCEPT_MARKERS):
        return True
    if any(marker in prompt for marker in MEMORY_DENY_MARKERS):
        return False
    if prompted:
        cleaned = prompt.strip()
        if cleaned in MEMORY_YES_MARKERS:
            return True
        if cleaned in MEMORY_NO_MARKERS:
            return False
    return None


def _format_memory_parts(planting: PlantingDetails) -> List[str]:
    parts = [f"作物: {planting.crop}"]
    if planting.variety:
        parts.append(f"品种: {planting.variety}")
    method_key = (
        planting.planting_method.value
        if hasattr(planting.planting_method, "value")
        else str(planting.planting_method)
    )
    parts.append(f"方式: {MEMORY_METHOD_LABELS.get(method_key, method_key)}")
    parts.append(f"播种日期: {planting.sowing_date.isoformat()}")
    if planting.region:
        parts.append(f"地区: {planting.region}")
    if planting.planting_location:
        parts.append(f"地块: {planting.planting_location}")
    return parts


def format_memory_confirmation(planting: PlantingDetails) -> str:
    info = "，".join(_format_memory_parts(planting))
    return (
        f"检测到上次种植信息：{info}。是否沿用？"
        "回复“是/沿用/同上”继续，回复“否/不用”将重新询问。"
    )


def apply_memory_to_draft(
    draft: PlantingDetailsDraft, memory: PlantingDetails
) -> PlantingDetailsDraft:
    data = draft.model_dump()
    assumptions = list(data.get("assumptions") or [])
    for field in (
        "crop",
        "variety",
        "planting_method",
        "sowing_date",
        "transplant_date",
        "region",
        "planting_location",
    ):
        if data.get(field) is None:
            value = getattr(memory, field, None)
            if field == "planting_method" and hasattr(value, "value"):
                value = value.value
            if value is not None:
                data[field] = value
                assumptions.append(f"{field}: 沿用上次记忆 {value}")
    data["assumptions"] = assumptions
    return PlantingDetailsDraft(**data)
