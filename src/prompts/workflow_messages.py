from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

from ..schemas import PlantingDetails, Recommendation


CROP_CALENDAR_MISSING_PREFIX = "为了给出农事推荐，还需要补充："
GROWTH_STAGE_MISSING_PREFIX = "生育期预测还需要补充："
HISTORICAL_WEATHER_NOTE = "说明: 当前仅使用历史气象数据，结果仅适用于历史期。"
FUTURE_WEATHER_WARNING = "提示: 暂不支持未来气象数据，无法获取未来日期对应气象。"
GROWTH_STAGE_ORDER = [
    "三叶一心",
    "返青",
    "分蘖期",
    "有效分蘖终止期",
    "拔节期",
    "幼穗分化1期",
    "幼穗分化2期",
    "幼穗分化4期",
    "孕穗期",
    "破口期",
    "始穗期",
    "抽穗期",
    "齐穗期",
    "成熟期",
]
PLANTING_METHOD_LABELS = {
    "direct_seeding": "直播",
    "transplanting": "移栽",
}

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


def build_crop_calendar_missing_question(
    missing_fields: List[str],
    field_labels: Dict[str, str],
    *,
    allow_unknown: bool = True,
) -> str:
    return format_missing_question(
        missing_fields,
        field_labels,
        CROP_CALENDAR_MISSING_PREFIX,
        allow_unknown=allow_unknown,
    )


def build_growth_stage_missing_question(
    missing_fields: List[str],
    field_labels: Dict[str, str],
    *,
    allow_unknown: bool = True,
) -> str:
    return format_missing_question(
        missing_fields,
        field_labels,
        GROWTH_STAGE_MISSING_PREFIX,
        allow_unknown=allow_unknown,
    )


def build_future_weather_warning(
    sowing_date: Optional[date],
    *,
    threshold_year: int = 2026,
) -> Optional[str]:
    if sowing_date and sowing_date.year >= threshold_year:
        return FUTURE_WEATHER_WARNING
    return None


def format_crop_calendar_plan_message(
    planting: PlantingDetails,
    recommendations: List[Recommendation],
    assumptions: List[str],
    weather_note: Optional[str] = None,
    variety_note: Optional[str] = None,
    recommendation_note: Optional[str] = None,
) -> str:
    info_parts = [f"作物: {planting.crop}"]
    if planting.variety:
        info_parts.append(f"品种: {planting.variety}")
    if planting.region:
        info_parts.append(f"区域: {planting.region}")
    method_key = (
        planting.planting_method.value
        if hasattr(planting.planting_method, "value")
        else str(planting.planting_method)
    )
    method_label = PLANTING_METHOD_LABELS.get(method_key, method_key)
    info_parts.append(f"方式: {method_label}")
    info_parts.append(f"播种日期: {planting.sowing_date.isoformat()}")

    lines = ["已生成农事推荐。", HISTORICAL_WEATHER_NOTE]
    warning = build_future_weather_warning(planting.sowing_date)
    if warning:
        lines.append(warning)
    lines.append("种植信息: " + "，".join(info_parts))
    if weather_note:
        lines.append(f"气象信息: {weather_note}")
    if variety_note:
        lines.append(f"品种信息: {variety_note}")
    if recommendation_note:
        lines.append(f"推荐摘要: {recommendation_note}")
    if recommendations:
        lines.append("推荐操作:")
        for idx, rec in enumerate(recommendations, start=1):
            line = f"{idx}. {rec.title} - {rec.description}"
            lines.append(line)

    if assumptions:
        lines.append("默认/假设: " + "；".join(assumptions))

    return "\n".join(lines)


def format_growth_stage_message(
    planting: PlantingDetails,
    stages: Dict[str, str],
    *,
    weather_note: str = "",
    variety_note: str = "",
) -> str:
    info_parts = [f"作物: {planting.crop}"]
    if planting.variety:
        info_parts.append(f"品种: {planting.variety}")
    if planting.region:
        info_parts.append(f"地区: {planting.region}")
    info_parts.append(f"播种日期: {planting.sowing_date.isoformat()}")
    lines = ["种植信息: " + "，".join(info_parts)]
    predicted = stages.get("predicted_stage")
    next_stage = stages.get("estimated_next_stage")
    gdd = stages.get("gdd_accumulated")
    gdd_required = stages.get("gdd_required_maturity")
    base_temp = stages.get("base_temperature")
    stage_dates = stages.get("stage_dates")
    if stage_dates:
        try:
            payload = json.loads(stage_dates)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            ordered = []
            seen = set()
            for name in GROWTH_STAGE_ORDER:
                value = payload.get(name)
                if isinstance(value, str) and value:
                    ordered.append((name, value))
                    seen.add(name)
            for name, value in payload.items():
                if name in seen:
                    continue
                if isinstance(value, str) and value:
                    ordered.append((name, value))
            entries = ordered
            if entries:
                lines.append("生育期阶段日期:")
                for name, value in entries:
                    lines.append(f"{name}: {value}")
    return "\n".join(lines)


def format_planting_validation_error(error: Exception) -> str:
    return f"种植信息校验失败: {error}"
