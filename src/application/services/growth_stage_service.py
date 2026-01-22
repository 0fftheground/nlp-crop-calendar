from __future__ import annotations

import json
import sqlite3
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ...domain.enums import PlantingMethod
from ...infra.config import get_config
from ...infra.gdd_store import get_gdd_db_path, get_gdd_records
from ...infra.geocode_service import (
    annotate_geocode_gdd_region,
    geocode_with_amap,
)
from ...infra.rice_region_matcher import match_gdd_region
from ...observability.logging_utils import log_event
from ...schemas import GrowthStageResult, PredictGrowthStageInput, WeatherDataPoint

GDD_STAGE_COLUMNS = [
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
GDD_REGIONS = {"湖南", "安徽", "长江中下游", "黑龙江省"}
RICE_TYPES = {"一季晚稻", "一季稻", "中稻", "双季晚稻", "早稻", "麦茬稻"}
RICE_TYPE_ALIASES = {
    "单晚稻": "一季晚稻",
    "单季稻": "一季晚稻",
}
SUBSPECIES_TYPES = {"籼", "粳"}
MATURITY_TYPES = {"中熟", "中迟熟", "早中熟", "早熟", "迟熟"}
_REGION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}(?:省|市|州|盟|地区|区|县)")
_PROVINCE_TOKENS = [
    "安徽",
    "北京",
    "重庆",
    "福建",
    "甘肃",
    "广东",
    "广西",
    "贵州",
    "海南",
    "河北",
    "河南",
    "黑龙江",
    "湖北",
    "湖南",
    "吉林",
    "江苏",
    "江西",
    "辽宁",
    "内蒙古",
    "宁夏",
    "青海",
    "山东",
    "上海",
    "山西",
    "陕西",
    "四川",
    "天津",
    "新疆",
    "西藏",
    "云南",
    "浙江",
]


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _normalize_region(value: Optional[str]) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    if "黑龙江" in text:
        return "黑龙江省"
    for region in GDD_REGIONS:
        if region in text:
            return region
    return ""


def _normalize_rice_type(value: Optional[str]) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    for alias, canonical in RICE_TYPE_ALIASES.items():
        if alias in text:
            return canonical
    for rice_type in RICE_TYPES:
        if rice_type in text:
            return rice_type
    return text


def _normalize_subspecies(value: Optional[str]) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    for subspecies in SUBSPECIES_TYPES:
        if subspecies in text:
            return subspecies
    return text


def _normalize_maturity(value: Optional[str]) -> str:
    text = _normalize_text(value)
    if not text or text in {"-", "—"}:
        return ""
    for maturity in MATURITY_TYPES:
        if maturity in text:
            return maturity
    return text


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_region_hint(region: Optional[str]) -> Tuple[str, Optional[str]]:
    text = _normalize_text(region)
    if not text:
        return "", None
    if match_gdd_region(text):
        return text, None
    geocode = geocode_with_amap(text, raw_region=region)
    if geocode and geocode.get("province"):
        province = _normalize_text(geocode.get("province"))
        if province:
            return province, text
    return text, text if geocode else None


def _get_variety_db_path() -> Optional[Path]:
    cfg = get_config()
    if cfg.variety_db_path:
        return Path(cfg.variety_db_path)
    return (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "rice_variety_approvals.sqlite3"
    )


def _parse_approval_year(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit() and len(text) == 4:
        return int(text)
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else 0


def _extract_region_tokens(text: Optional[str]) -> List[str]:
    value = _normalize_text(text)
    if not value:
        return []
    tokens = _REGION_TOKEN_RE.findall(value)
    if "国审" in value and "国审" not in tokens:
        tokens.append("国审")
    for province in _PROVINCE_TOKENS:
        if province in value and province not in tokens:
            tokens.append(province)
    return tokens


def _score_region_match(
    approval_region: str, planting_tokens: List[str]
) -> int:
    if not approval_region:
        return 0
    score = 0
    region_tokens = _extract_region_tokens(approval_region)
    if "国审" in approval_region:
        score = max(score, 1)
    for token in planting_tokens:
        if token in region_tokens:
            return 3
    for token in planting_tokens:
        if token and token in approval_region:
            score = max(score, 2)
    return score


def _lookup_variety_records(variety_name: Optional[str]) -> List[Dict[str, object]]:
    if not variety_name:
        return []
    path = _get_variety_db_path()
    if not path or not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT approval_region, rice_type, subspecies_type, maturity, "
                "control_variety, days_vs_control, approval_year "
                "FROM variety_approvals WHERE variety_name = ? "
                "ORDER BY approval_year DESC",
                (variety_name,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def _resolve_variety_meta(
    input: PredictGrowthStageInput, region_hint: str
) -> Tuple[Dict[str, object], str]:
    if input.variety_record:
        return input.variety_record, "confirmed_record"
    records = _lookup_variety_records(input.variety)
    return _select_variety_record(records, region_hint), "auto_selected"


def _select_variety_record(
    records: List[Dict[str, object]], planting_region: Optional[str]
) -> Dict[str, object]:
    if not records:
        return {}
    planting_tokens = _extract_region_tokens(planting_region)
    best_score = -1
    best_year = -1
    best_record: Optional[Dict[str, object]] = None
    for record in records:
        approval_region = _normalize_text(record.get("approval_region"))
        score = (
            _score_region_match(approval_region, planting_tokens)
            if planting_tokens
            else 0
        )
        year = _parse_approval_year(record.get("approval_year"))
        if score > best_score:
            best_score = score
            best_year = year
            best_record = record
            continue
        if score == best_score and year > best_year:
            best_year = year
            best_record = record
    return best_record or records[0]


def _filter_records(
    records: Iterable[Dict[str, object]],
    criteria: Dict[str, str],
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    for record in records:
        ok = True
        for key, expected in criteria.items():
            if not expected:
                continue
            actual = _normalize_text(record.get(key))
            if actual != expected:
                ok = False
                break
        if ok:
            matches.append(record)
    return matches


def _select_gdd_record(
    records: List[Dict[str, object]],
    *,
    variety_name: Optional[str],
    approval_region: str,
    rice_type: str,
    subspecies: str,
    maturity: str,
    control_variety: str,
) -> Tuple[Dict[str, object], str]:
    if variety_name and "品种" in records[0]:
        matches = _filter_records(
            records,
            {"品种": variety_name, "审定地区": approval_region},
        )
        if matches:
            return matches[0], "variety"
    if control_variety and rice_type:
        matches = _filter_records(
            records,
            {"标准品种": control_variety, "稻作类型": rice_type},
        )
        if matches:
            return matches[0], "control_variety"
    if maturity and approval_region and rice_type and subspecies:
        matches = _filter_records(
            records,
            {
                "熟制": maturity,
                "审定地区": approval_region,
                "稻作类型": rice_type,
                "亚种": subspecies,
            },
        )
        if matches:
            return matches[0], "maturity_region"
    if approval_region and rice_type and subspecies:
        matches = _filter_records(
            records,
            {
                "审定地区": approval_region,
                "稻作类型": rice_type,
                "亚种": subspecies,
            },
        )
        if matches:
            return matches[0], "region"
    if rice_type and subspecies:
        matches = _filter_records(
            records,
            {
                "审定地区": "长江中下游",
                "稻作类型": rice_type,
                "亚种": subspecies,
            },
        )
        if matches:
            return matches[0], "fallback_yangtze"
    raise ValueError(
        "未匹配到积温参数: 标准品种={0}, 稻作类型={1}, 亚种={2}, 审定地区={3}, 熟制={4}".format(
            control_variety or "-",
            rice_type or "-",
            subspecies or "-",
            approval_region or "-",
            maturity or "-",
        )
    )


def _extract_stage_thresholds(record: Dict[str, object]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for stage in GDD_STAGE_COLUMNS:
        value = _parse_float(record.get(stage))
        if value is None:
            continue
        thresholds[stage] = float(value)
    return thresholds


def _select_temperature(point: WeatherDataPoint) -> Optional[float]:
    if point.temperature is not None:
        return float(point.temperature)
    if point.temperature_max is not None and point.temperature_min is not None:
        return (float(point.temperature_max) + float(point.temperature_min)) / 2
    if point.temperature_max is not None:
        return float(point.temperature_max)
    if point.temperature_min is not None:
        return float(point.temperature_min)
    return None


def _build_daily_temperatures(
    points: Iterable[WeatherDataPoint],
) -> List[Tuple[date, float]]:
    totals: Dict[date, float] = {}
    counts: Dict[date, int] = {}
    for point in points:
        if point.timestamp is None:
            continue
        avg_temp = _select_temperature(point)
        if avg_temp is None:
            continue
        day = point.timestamp.date()
        totals[day] = totals.get(day, 0.0) + avg_temp
        counts[day] = counts.get(day, 0) + 1
    daily = [(day, totals[day] / counts[day]) for day in totals]
    return sorted(daily, key=lambda item: item[0])


def _calc_gdd(avg_temp: float, base_temp: float) -> float:
    capped = max(min(avg_temp, 40.0), base_temp)
    return capped - base_temp


def _resolve_start_date(input: PredictGrowthStageInput) -> date:
    planting = input.planting
    method = planting.planting_method
    method_value = (
        method.value if hasattr(method, "value") else str(method)
    )
    if (
        method_value == PlantingMethod.TRANSPLANTING.value
        and planting.transplant_date
    ):
        return planting.transplant_date
    return planting.sowing_date


def _resolve_base_temp(subspecies: str) -> float:
    if subspecies == "粳":
        return 10.0
    return 12.0


def predict_growth_stage_local(
    input: PredictGrowthStageInput,
) -> GrowthStageResult:
    records = get_gdd_records()
    if not records:
        raise ValueError("GDD 积温表为空，无法预测生育期。")

    region_hint, geocode_key = _resolve_region_hint(input.region)
    variety_meta, variety_meta_source = _resolve_variety_meta(
        input, region_hint
    )
    approval_region = _normalize_region(variety_meta.get("approval_region"))
    if not approval_region:
        approval_region = match_gdd_region(region_hint)
    if geocode_key and approval_region:
        annotate_geocode_gdd_region(geocode_key, approval_region)
    rice_type = _normalize_rice_type(variety_meta.get("rice_type"))
    subspecies = _normalize_subspecies(variety_meta.get("subspecies_type"))
    maturity = _normalize_maturity(variety_meta.get("maturity"))
    control_variety = _normalize_text(variety_meta.get("control_variety"))
    control_days = _parse_float(variety_meta.get("days_vs_control")) or 0.0

    record, rule = _select_gdd_record(
        records,
        variety_name=input.variety,
        approval_region=approval_region,
        rice_type=rice_type,
        subspecies=subspecies,
        maturity=maturity,
        control_variety=control_variety,
    )
    thresholds = _extract_stage_thresholds(record)
    if not thresholds:
        raise ValueError("积温文件缺少生育期阈值。")

    base_temp = _resolve_base_temp(subspecies)
    start_date = _resolve_start_date(input)
    daily = _build_daily_temperatures(input.weatherSeries.points)
    daily = [(day, temp) for day, temp in daily if day >= start_date]

    cumulative: List[Tuple[date, float]] = []
    running = 0.0
    for day, avg_temp in daily:
        running += _calc_gdd(avg_temp, base_temp)
        cumulative.append((day, running))

    gdd_accumulated = cumulative[-1][1] if cumulative else 0.0
    stage_dates: Dict[str, str] = {}
    for stage in GDD_STAGE_COLUMNS:
        required = thresholds.get(stage)
        if required is None:
            continue
        stage_date = next(
            (day for day, total in cumulative if total >= required),
            None,
        )
        if stage_date and control_days:
            stage_date = stage_date + timedelta(days=int(round(control_days)))
        stage_dates[stage] = stage_date.isoformat() if stage_date else ""

    predicted_stage = ""
    estimated_next_stage = ""
    for stage in GDD_STAGE_COLUMNS:
        required = thresholds.get(stage)
        if required is None:
            continue
        if gdd_accumulated >= required:
            predicted_stage = stage
        elif not estimated_next_stage:
            estimated_next_stage = stage
            break

    log_event(
        "growth_stage_local_match",
        variety=input.variety,
        region=input.region,
        approval_region=approval_region,
        rice_type=rice_type,
        subspecies=subspecies,
        maturity=maturity,
        variety_record_source=variety_meta_source,
        rule=rule,
        gdd_path=str(get_gdd_db_path()),
    )

    stages: Dict[str, str] = {
        "predicted_stage": predicted_stage,
        "estimated_next_stage": estimated_next_stage,
        "gdd_accumulated": f"{gdd_accumulated:.1f}",
        "gdd_required_maturity": (
            f"{thresholds.get('成熟期'):.1f}"
            if thresholds.get("成熟期") is not None
            else ""
        ),
        "base_temperature": f"{base_temp:.1f}",
        "start_date": start_date.isoformat(),
        "stage_dates": json.dumps(stage_dates, ensure_ascii=False),
        "match_rule": rule,
    }
    return GrowthStageResult(stages=stages)
