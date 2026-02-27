"""
Shared helpers for LangGraph workflows.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ...infra.llm_extract import llm_structured_extract
from ...prompts.planting_extract import build_planting_extract_prompt
from ...schemas import PlantingDetails, PlantingDetailsDraft, WeatherSeries


UNKNOWN_MARKERS = ["不知道", "不清楚", "不确定", "记不清", "不记得", "忘了"]


class PlantingExtract(BaseModel):
    crop: Optional[str] = None
    variety: Optional[str] = None
    culti_type: Optional[str] = Field(
        default=None, description="稻作类型/熟制，如早稻、晚稻、双季晚稻"
    )
    planting_method: Optional[str] = Field(
        default=None, description="direct_seeding 或 transplanting"
    )
    sowing_date: Optional[date] = None
    transplant_date: Optional[date] = None
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
    system_prompt = build_planting_extract_prompt()
    return llm_structured_extract(
        prompt,
        schema=schema,
        system_prompt=system_prompt,
    )


def build_fallback_planting(draft: PlantingDetailsDraft) -> PlantingDetails:
    today = date.today()
    method = draft.planting_method or "direct_seeding"
    sowing_date = draft.sowing_date or today
    return PlantingDetails(
        crop=draft.crop or "水稻",
        variety=draft.variety,
        culti_type=draft.culti_type,
        planting_method=method,
        sowing_date=sowing_date,
        transplant_date=draft.transplant_date,
    )
