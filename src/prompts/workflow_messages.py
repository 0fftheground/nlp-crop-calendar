from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

from ..schemas import PlantingDetails, Recommendation


CROP_CALENDAR_MISSING_PREFIX = "为了给出农事推荐，还需要补充："
GROWTH_STAGE_MISSING_PREFIX = "生育期结果查询还需要补充："
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
    optional_fields: Optional[List[str]] = None,
) -> str:
    labels = [field_labels.get(field, field) for field in missing_fields]
    if optional_fields:
        for field in optional_fields:
            if field in missing_fields:
                continue
            label = field_labels.get(field, field)
            labels.append(f"{label}(可选)")
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
    optional_fields: Optional[List[str]] = None,
) -> str:
    return format_missing_question(
        missing_fields,
        field_labels,
        CROP_CALENDAR_MISSING_PREFIX,
        allow_unknown=allow_unknown,
        optional_fields=optional_fields,
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
    method_key = (
        planting.planting_method.value
        if hasattr(planting.planting_method, "value")
        else str(planting.planting_method)
    )
    method_label = PLANTING_METHOD_LABELS.get(method_key, method_key)

    lines = ["已生成农事推荐。"]
    if weather_note:
        lines.append(HISTORICAL_WEATHER_NOTE)
        warning = build_future_weather_warning(planting.sowing_date)
        if warning:
            lines.append(warning)

    lines.append("")
    lines.append("【种植信息】")
    lines.append(f"作物：{planting.crop}")
    if planting.variety:
        lines.append(f"品种：{planting.variety}")
    culti_type = getattr(planting, "culti_type", None)
    if culti_type:
        lines.append(f"稻作类型：{culti_type}")
    lines.append(f"播种方式：{method_label}")
    lines.append(f"播种日期：{planting.sowing_date.isoformat()}")
    if planting.transplant_date:
        lines.append(f"移栽日期：{planting.transplant_date.isoformat()}")

    if weather_note:
        lines.append("")
        lines.append("【气象信息】")
        lines.append(str(weather_note))
    if variety_note:
        lines.append("")
        lines.append("【品种信息】")
        lines.append(str(variety_note))
    if recommendation_note:
        lines.append("")
        lines.append("【推荐摘要】")
        lines.append(str(recommendation_note))
    if recommendations:
        lines.append("")
        lines.append("【推荐操作】")
        for idx, rec in enumerate(recommendations, start=1):
            line = f"{idx}. {rec.title} - {rec.description}"
            lines.append(line)

    if assumptions:
        lines.append("")
        lines.append("【默认/假设】")
        lines.append("；".join(assumptions))

    return "\n".join(lines).strip()


def format_growth_stage_message(
    planting: PlantingDetails,
    stages: Dict[str, str],
    *,
    weather_note: str = "",
    variety_note: str = "",
) -> str:
    planting_method = getattr(planting, "planting_method", None)
    if planting_method:
        method_label = (
            "直播"
            if planting_method == "direct_seeding"
            else "插秧"
            if planting_method == "transplanting"
            else str(planting_method)
        )
    else:
        method_label = ""

    lines = ["【种植信息】", f"作物：{planting.crop}"]
    if planting.variety:
        lines.append(f"品种：{planting.variety}")
    culti_type = getattr(planting, "culti_type", None)
    if culti_type:
        lines.append(f"稻作类型：{culti_type}")
    if method_label:
        lines.append(f"播种方式：{method_label}")
    lines.append(f"播种日期：{planting.sowing_date.isoformat()}")
    if planting.transplant_date:
        lines.append(f"移栽日期：{planting.transplant_date.isoformat()}")

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
                lines.append("")
                lines.append("【生育期预测结果】")
                for name, value in entries:
                    lines.append(f"{name}：{value}")

    return "\n".join(lines).strip()
