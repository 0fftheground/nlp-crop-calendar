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
