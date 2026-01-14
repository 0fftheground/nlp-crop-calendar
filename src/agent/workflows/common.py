"""
Shared helpers for LangGraph workflows.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ...infra.config import get_config
from ...infra.llm_extract import llm_structured_extract
from ...infra.variety_store import build_variety_hint
from ...prompts.planting_extract import build_planting_extract_prompt
from ...schemas import PlantingDetails, PlantingDetailsDraft, WeatherSeries


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


def coerce_planting_draft(value: object) -> Optional[PlantingDetailsDraft]:
    if value is None:
        return None
    if isinstance(value, PlantingDetailsDraft):
        return value
    if isinstance(value, dict):
        try:
            return PlantingDetailsDraft.model_validate(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            try:
                return PlantingDetailsDraft.model_validate(payload)
            except Exception:
                return None
    return None


def coerce_weather_series(
    data: Dict[str, object], *, region: str, source: str = "workflow"
) -> WeatherSeries:
    if data:
        try:
            return WeatherSeries.model_validate(data)
        except Exception:
            pass
    return WeatherSeries(
        region=region or "unknown",
        granularity="daily",
        start_date=None,
        end_date=None,
        points=[],
        source=source,
    )


def summarize_weather_series(weather_series: WeatherSeries) -> Dict[str, object]:
    return {
        "region": weather_series.region,
        "start_date": (
            weather_series.start_date.isoformat()
            if weather_series.start_date
            else None
        ),
        "end_date": (
            weather_series.end_date.isoformat() if weather_series.end_date else None
        ),
        "points": len(weather_series.points),
        "source": weather_series.source,
    }


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
    hint = build_variety_hint(prompt)
    system_prompt = build_planting_extract_prompt(hint)
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
