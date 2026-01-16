"""
Shared helpers for LangGraph workflows.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Set, Type

from pydantic import BaseModel, Field

from ...infra.config import get_config
from ...infra.llm_extract import llm_structured_extract
from ...infra.variety_store import build_variety_hint
from ...prompts.planting_extract import build_planting_extract_prompt
from ...schemas import PlantingDetails, PlantingDetailsDraft, WeatherSeries


UNKNOWN_MARKERS = ["不知道", "不清楚", "不确定", "记不清", "不记得", "忘了"]
EXPERIENCE_FIELDS = [
    "crop",
    "variety",
    "planting_method",
    "sowing_date",
    "transplant_date",
    "region",
    "planting_location",
]
EXPERIENCE_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "transplant_date": "移栽日期",
    "region": "地区",
    "planting_location": "地块",
}
EXPERIENCE_CHANGE_MARKERS = [
    "更换",
    "换个",
    "换一个",
    "换成",
    "改成",
    "改一下",
    "重新",
    "不用",
    "不要",
    "不沿用",
]
EXPERIENCE_FIELD_MARKERS = {
    "crop": ["作物"],
    "variety": ["品种"],
    "planting_method": ["种植方式", "方式", "播种方式"],
    "sowing_date": ["播种日期", "播期", "播种时间", "播种"],
    "transplant_date": ["移栽日期", "插秧日期", "移栽时间", "插秧时间"],
    "region": ["地区", "区域", "地方"],
    "planting_location": ["地块", "位置", "地址"],
}
EXPERIENCE_ASSUMPTION_PREFIX = "沿用历史记录"


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


def detect_experience_change_fields(prompt: str) -> List[str]:
    if not prompt:
        return []
    text = prompt.strip()
    if not any(marker in text for marker in EXPERIENCE_CHANGE_MARKERS):
        return []
    fields: Set[str] = set()
    for field, markers in EXPERIENCE_FIELD_MARKERS.items():
        if any(marker in text for marker in markers):
            fields.add(field)
    if not fields:
        return list(EXPERIENCE_FIELDS)
    return list(fields)


def _format_experience_assumption(field: str, value: object) -> str:
    return f"{field}: {EXPERIENCE_ASSUMPTION_PREFIX} {value}"


def _is_experience_assumption(
    assumption: str, fields: Sequence[str]
) -> bool:
    for field in fields:
        prefix = f"{field}: {EXPERIENCE_ASSUMPTION_PREFIX}"
        if assumption.startswith(prefix):
            return True
    return False


def clear_experience_fields(
    draft: PlantingDetailsDraft, fields: Iterable[str]
) -> PlantingDetailsDraft:
    data = draft.model_dump()
    field_list = [field for field in fields if field in data]
    for field in field_list:
        data[field] = None
    assumptions = list(data.get("assumptions") or [])
    if field_list:
        assumptions = [
            item
            for item in assumptions
            if not _is_experience_assumption(item, field_list)
        ]
    data["assumptions"] = assumptions
    return PlantingDetailsDraft(**data)


def apply_experience_choice_to_draft(
    draft: PlantingDetailsDraft,
    planting: PlantingDetails,
    *,
    skip_fields: Optional[Set[str]] = None,
) -> tuple[PlantingDetailsDraft, List[str]]:
    data = draft.model_dump()
    assumptions = list(data.get("assumptions") or [])
    applied: List[str] = []
    skip = set(skip_fields or set())
    for field in EXPERIENCE_FIELDS:
        if field in skip:
            continue
        if data.get(field) is not None:
            continue
        value = getattr(planting, field, None)
        if value is None:
            continue
        if field == "planting_method" and hasattr(value, "value"):
            value = value.value
        data[field] = value
        applied.append(field)
        assumption = _format_experience_assumption(field, value)
        if assumption not in assumptions:
            assumptions.append(assumption)
    data["assumptions"] = assumptions
    return PlantingDetailsDraft(**data), applied


def build_experience_notice(
    planting: PlantingDetails, applied_fields: Sequence[str]
) -> Optional[str]:
    if not applied_fields:
        return None
    labels = [
        EXPERIENCE_FIELD_LABELS.get(field, field) for field in applied_fields
    ]
    joined = "、".join(labels)
    note = f"已默认沿用上次{joined}"
    if "sowing_date" in applied_fields and planting.sowing_date:
        note = (
            f"{note}（播种日期: {planting.sowing_date.isoformat()}）"
        )
    note = f"{note}。如需更换请回复“更换”或“更换+字段”。"
    return note
