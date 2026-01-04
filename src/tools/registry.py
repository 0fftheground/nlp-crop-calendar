import json
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from langchain_core.tools import BaseTool, tool as lc_tool
from ..infra.config import get_config
from ..infra.llm_extract import llm_structured_extract
from ..infra.variety_store import build_variety_hint
from ..infra.variety_store import retrieve_variety_candidates
from ..infra.tool_provider import maybe_intranet_tool, normalize_provider
from ..domain.services import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ..schemas.models import (
    GrowthStageResult,
    OperationItem,
    OperationPlanResult,
    PlantingDetailsDraft,
    ToolInvocation,
    WeatherDataPoint,
    WeatherSeries,
)


TOOLS: List[BaseTool] = []
TOOL_INDEX: Dict[str, BaseTool] = {}
HIDDEN_TOOL_NAMES = {"farming_recommendation", "growth_stage_prediction"}

VARIETY_PATTERN = re.compile(r"(?:品种|品名|品系)[:：\s]*([\w\u4e00-\u9fff-]{2,20})")
VARIETY_FALLBACK_PATTERN = re.compile(r"([\w\u4e00-\u9fff-]{2,20}号)")

DEFAULT_STAGE_BOUNDARIES = [
    ("germination", 10),
    ("tillering", 35),
    ("jointing", 60),
    ("heading", 90),
    ("maturity", 120),
]
DEFAULT_STAGE_TOTAL_DAYS = DEFAULT_STAGE_BOUNDARIES[-1][1]
DEFAULT_GDD_PER_DAY = 10.0
DEFAULT_GDD_BASE_TEMP = 10.0
GDD_BASE_TEMPS = {
    "水稻": 10.0,
    "小麦": 5.0,
    "玉米": 10.0,
    "大豆": 10.0,
    "油菜": 5.0,
    "棉花": 12.0,
    "花生": 10.0,
}


PLANTING_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "region": "地区",
}


}


def register_tool(tool: BaseTool) -> None:
    """Register a custom LangChain tool for routing."""
    TOOLS.append(tool)
    TOOL_INDEX[tool.name] = tool


def clear_tools() -> None:
    """Utility for tests or dynamic reloads."""
    TOOLS.clear()
    TOOL_INDEX.clear()


def list_tool_specs() -> List[Dict[str, str]]:
    """
    Return tool metadata for the selector prompt.

    Each BaseTool should define `name` and `description`.
    """
    return [
        {"name": t.name, "description": t.description or ""}
        for t in TOOLS
        if t.name not in HIDDEN_TOOL_NAMES
    ]


def auto_register_tool(*tool_args, **tool_kwargs):
    """
    Decorator that wraps LangChain's `@tool` and registers the tool automatically.

    Usage:

        @auto_register_tool("name", description="...")
        def handler(prompt: str) -> ToolInvocation:
            ...
    """

    def decorator(func):
        description = tool_kwargs.pop("description", None)
        if description:
            func.__doc__ = description
        langchain_tool = lc_tool(*tool_args, **tool_kwargs)(func)
        register_tool(langchain_tool)
        return langchain_tool

    return decorator


def _extract_variety(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    match = VARIETY_PATTERN.search(prompt)
    if match:
        return match.group(1)
    match = VARIETY_FALLBACK_PATTERN.search(prompt)
    if match:
        return match.group(1)
    return None


def _infer_crop_and_variety(prompt: str) -> Tuple[str, str]:
    crop_keywords = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
    crop = next((item for item in crop_keywords if item in prompt), "水稻")
    variety = _extract_variety(prompt)
    if not variety:
        candidates = retrieve_variety_candidates(prompt, limit=1)
        if candidates:
            variety = candidates[0]
    if not variety:
        variety = "美香占2号" if crop == "水稻" else f"{crop}示例品种"
    return crop, variety


def _llm_extract_planting_details(prompt: str) -> PlantingDetailsDraft:
    system_prompt = (
        "你是农事助手，负责从用户描述中抽取种植信息。"
        "只输出可确定的信息；不确定或未提及时保持为空。"
        "种植方式使用 direct_seeding 或 transplanting。"
        "日期格式为 YYYY-MM-DD。"
    )
    hint = build_variety_hint(prompt)
    if hint:
        system_prompt = f"{system_prompt}{hint}"
    payload = llm_structured_extract(
        prompt,
        schema=PlantingDetailsDraft,
        system_prompt=system_prompt,
    )
    allowed = {
        "crop",
        "variety",
        "planting_method",
        "sowing_date",
        "transplant_date",
        "region",
        "planting_location",
        "notes",
        "confidence",
    }
    filtered = {k: v for k, v in payload.items() if k in allowed}
    data: Dict[str, object] = {"source_text": prompt, **filtered}
    return PlantingDetailsDraft(**data)


def _parse_followup_payload(prompt: str) -> Optional[Dict[str, object]]:
    if not prompt:
        return None
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    followup = payload.get("followup")
    return followup if isinstance(followup, dict) else None


def _merge_followup_draft(
    prompt: str, prior_draft: Optional[Dict[str, object]]
) -> PlantingDetailsDraft:
    try:
        seed = PlantingDetailsDraft(**(prior_draft or {}))
    except Exception:
        seed = PlantingDetailsDraft()
    if not prompt:
        return seed
    fresh = _llm_extract_planting_details(prompt)
    answers = fresh.model_dump(exclude_none=True)
    return merge_planting_answers(seed, answers=answers)


def _format_missing_planting_question(missing_fields: List[str]) -> str:
    labels = [PLANTING_FIELD_LABELS.get(field, field) for field in missing_fields]
    joined = "、".join(labels)
    return f"生育期预测还需要补充：{joined}。请提供后继续。"


def _normalize_growth_days(value: object, default: int = DEFAULT_STAGE_TOTAL_DAYS) -> int:
    try:
        if value is None:
            return default
        days = int(value)
        return days if days > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_weather_series(data: Dict[str, object], *, region: str) -> WeatherSeries:
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
        source="workflow",
    )


def _average_temperature(point: WeatherDataPoint) -> Optional[float]:
    if point.temperature is not None:
        return float(point.temperature)
    if point.temperature_max is not None and point.temperature_min is not None:
        return (float(point.temperature_max) + float(point.temperature_min)) / 2
    return None


def _sum_gdd(points: List[WeatherDataPoint], base_temp: float) -> float:
    total = 0.0
    for point in points:
        avg_temp = _average_temperature(point)
        if avg_temp is None:
            continue
        total += max(0.0, avg_temp - base_temp)
    return total


def _build_gdd_thresholds(total_required_gdd: float) -> List[Tuple[str, float]]:
    total = total_required_gdd or (DEFAULT_GDD_PER_DAY * DEFAULT_STAGE_TOTAL_DAYS)
    thresholds: List[Tuple[str, float]] = []
    for stage, day in DEFAULT_STAGE_BOUNDARIES:
        thresholds.append((stage, total * (day / DEFAULT_STAGE_TOTAL_DAYS)))
    return thresholds


def _infer_stage(
    cumulative_gdd: float, thresholds: List[Tuple[str, float]]
) -> Tuple[str, str]:
    predicted = "harvest"
    next_stage = ""
    for index, (stage, boundary) in enumerate(thresholds):
        if cumulative_gdd <= boundary:
            predicted = stage
            next_stage = (
                thresholds[index + 1][0] if index + 1 < len(thresholds) else "harvest"
            )
            break
    return predicted, next_stage


def _estimate_stage_dates(
    sowing_date: date,
    avg_gdd_per_day: float,
    thresholds: List[Tuple[str, float]],
) -> Dict[str, str]:
    if avg_gdd_per_day <= 0:
        return {}
    stage_dates: Dict[str, str] = {}
    for stage, boundary in thresholds:
        days_needed = int(round(boundary / avg_gdd_per_day))
        stage_dates[f"{stage}_date"] = (
            sowing_date + timedelta(days=days_needed)
        ).isoformat()
    return stage_dates


@auto_register_tool(
    "variety_lookup",
    description=(
        "查询品种基础信息。仅用于用户明确询问品种特性/抗性/生育期等单点信息；"
        "不用于完整种植方案。"
    ),
)
def variety_lookup(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.variety_provider)
    intranet = maybe_intranet_tool(
        "variety_lookup",
        prompt,
        provider,
        cfg.variety_api_url,
        cfg.variety_api_key,
    )
    if intranet:
        return intranet
    crop, variety = _infer_crop_and_variety(prompt)
    payload = {
        "query": prompt,
        "crop": crop,
        "variety": variety,
        "growth_duration_days": 120,
        "traits": {
            "yield_level": "中高产",
            "lodging_resistance": "中等",
            "disease_resistance": ["稻瘟病", "纹枯病"],
        },
        "source": "mock",
    }
    return ToolInvocation(
        name="variety_lookup",
        message=f"已返回 {crop} 品种 {variety} 的模拟信息。",
        data=payload,
    )


@auto_register_tool(
    "weather_lookup",
    description="查询指定地区气象数据。仅用于获取天气数据本身；不生成农事建议或计划。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.weather_provider)
    intranet = maybe_intranet_tool(
        "weather_lookup",
        prompt,
        provider,
        cfg.weather_api_url,
        cfg.weather_api_key,
    )
    if intranet:
        return intranet
    start = date.today()
    points: List[WeatherDataPoint] = []
    for offset in range(3):
        current = start + timedelta(days=offset)
        points.append(
            WeatherDataPoint(
                timestamp=datetime.combine(current, time.min),
                temperature=20 + offset,
                temperature_max=26 + offset,
                temperature_min=16 + offset,
                humidity=60 + offset * 2,
                precipitation=0.5 * offset,
                wind_speed=2.0 + 0.3 * offset,
                condition="sunny" if offset < 2 else "cloudy",
            )
        )
    series = WeatherSeries(
        region=prompt or "unknown",
        granularity="daily",
        start_date=start,
        end_date=start + timedelta(days=2),
        points=points,
        source="mock",
    )
    return ToolInvocation(
        name="weather_lookup",
        message="已返回 3 天游模拟气象数据。",
        data=series.model_dump(mode="json"),
    )


@auto_register_tool(
    "growth_stage_prediction",
    description=(
        "基于自然语言抽取种植信息（作物/品种/方式/播期/地区），"
        "依次调用品种/气象工具并用积温预测生育期。仅用于生育期预测。"
    ),
)
def growth_stage_prediction(prompt: str) -> ToolInvocation:
    followup = _parse_followup_payload(prompt)
    followup_count = 0
    if followup:
        prompt = str(followup.get("prompt") or "")
        try:
            prior_count = int(followup.get("followup_count") or 0)
        except (TypeError, ValueError):
            prior_count = 0
        followup_count = prior_count + 1
        prior_draft = followup.get("draft")
        draft = _merge_followup_draft(prompt, prior_draft)
    else:
        draft = _llm_extract_planting_details(prompt)
    missing_fields = list_missing_required_fields(draft)
    if not draft.region:
        missing_fields.append("region")
    missing_fields = list(dict.fromkeys(missing_fields))

    trace = [
        f"extract draft={draft.model_dump(exclude_none=True)}",
        f"extract missing={missing_fields} followup_count={followup_count}",
    ]

    if missing_fields:
        message = _format_missing_planting_question(missing_fields)
        return ToolInvocation(
            name="growth_stage_prediction",
            message=message,
            data={
                "missing_fields": missing_fields,
                "draft": draft.model_dump(exclude_none=True),
                "followup_count": followup_count,
                "trace": trace,
            },
        )

    try:
        planting = normalize_and_validate_planting(draft)
    except MissingPlantingInfoError as exc:
        message = _format_missing_planting_question(exc.missing_fields)
        return ToolInvocation(
            name="growth_stage_prediction",
            message=message,
            data={
                "missing_fields": exc.missing_fields,
                "draft": draft.model_dump(exclude_none=True),
                "followup_count": followup_count,
                "trace": trace,
            },
        )
    except ValueError as exc:
        return ToolInvocation(
            name="growth_stage_prediction",
            message=f"种植信息校验失败: {exc}",
            data={"draft": draft.model_dump(exclude_none=True), "trace": trace},
        )

    crop = planting.crop
    variety = planting.variety or f"{crop}示例品种"
    region = planting.region or "unknown"

    variety_prompt = f"{crop} {variety}".strip()
    variety_result = execute_tool("variety_lookup", variety_prompt)
    if variety_result:
        variety_data = variety_result.data
        trace.append("variety_lookup ok")
    else:
        variety_data = {}
        trace.append("variety_lookup missing")

    weather_result = execute_tool("weather_lookup", region)
    if weather_result:
        weather_data = weather_result.data
        trace.append("weather_lookup ok")
    else:
        weather_data = {}
        trace.append("weather_lookup missing")

    weather_series = _coerce_weather_series(weather_data, region=region)
    gdd_base = GDD_BASE_TEMPS.get(crop, DEFAULT_GDD_BASE_TEMP)
    cumulative_gdd = _sum_gdd(weather_series.points, gdd_base)
    observed_days = len(weather_series.points)
    avg_gdd_per_day = cumulative_gdd / observed_days if observed_days else 0.0

    growth_days = _normalize_growth_days(variety_data.get("growth_duration_days"))
    total_required_gdd = growth_days * DEFAULT_GDD_PER_DAY
    thresholds = _build_gdd_thresholds(total_required_gdd)
    predicted_stage, next_stage = _infer_stage(cumulative_gdd, thresholds)

    sowing_date = planting.sowing_date
    days_since = (date.today() - sowing_date).days

    stage_dates = _estimate_stage_dates(sowing_date, avg_gdd_per_day, thresholds)
    result = GrowthStageResult(
        stages={
            "predicted_stage": predicted_stage,
            "estimated_next_stage": next_stage,
            "days_since_sowing": str(days_since),
            "gdd_accumulated": f"{cumulative_gdd:.1f}",
            "gdd_required_maturity": f"{total_required_gdd:.1f}",
            "base_temperature": f"{gdd_base:.1f}",
            "growth_duration_days": str(growth_days),
            "sowing_date": sowing_date.isoformat(),
            "method": "gdd_workflow",
            **stage_dates,
        }
    )
    payload = result.model_dump(mode="json")
    payload["workflow"] = {
        "inputs": {
            "planting": planting.model_dump(exclude_none=True, mode="json"),
        },
        "variety": variety_data,
        "weather_summary": {
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
        },
        "trace": trace,
    }
    return ToolInvocation(
        name="growth_stage_prediction",
        message="已完成品种查询、气象查询与积温计算，返回生育期预测结果。",
        data=payload,
    )


@auto_register_tool(
    "farming_recommendation",
    description=(
        "生成简短农事建议（单段建议）。仅在用户已提供作物/地区/时间或生育期时使用；"
        "若用户要完整种植计划/全流程安排，请走 workflow。"
    ),
)
def farming_recommendation(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.recommendation_provider)
    intranet = maybe_intranet_tool(
        "farming_recommendation",
        prompt,
        provider,
        cfg.recommendation_api_url,
        cfg.recommendation_api_key,
    )
    if intranet:
        return intranet
    crop, _ = _infer_crop_and_variety(prompt)
    ops = [
        OperationItem(
            stage="field_preparation",
            title="清沟排水",
            description="播种前疏通田间沟系，避免积水。",
            window="播种前 7 天",
            priority="medium",
        ),
        OperationItem(
            stage="seedling",
            title="查苗补苗",
            description="出苗后 10-15 天查苗，缺株处补播。",
            window="出苗后 10 天",
            priority="high",
        ),
        OperationItem(
            stage="fertilization",
            title="分蘖肥",
            description="分蘖期追施氮肥 5-8 公斤/亩。",
            window="出苗后 20-30 天",
            priority="medium",
        ),
    ]
    plan = OperationPlanResult(
        crop=crop,
        summary=f"{crop} 农事推荐（mock 数据）。",
        operations=ops,
        metadata={"source": "mock"},
    )
    return ToolInvocation(
        name="farming_recommendation",
        message="已返回模拟农事推荐。",
        data=plan.model_dump(mode="json"),
    )


def initialize_tools() -> None:
    """
    Hook for triggering tool registration.

    Importing this module elsewhere会执行装饰器，从而注册全部工具。
    """
    return None


def execute_tool(name: str, prompt: str) -> Optional[ToolInvocation]:
    tool = TOOL_INDEX.get(name)
    if not tool:
        return None
    result = tool.invoke(prompt)
    if isinstance(result, ToolInvocation):
        return result
    raise TypeError(
        f"Tool '{name}' returned unsupported type {type(result)!r}; "
        "please return ToolInvocation."
    )


def build_agent_tools() -> List[BaseTool]:
    """Build agent-friendly tools that return JSON strings for LLM consumption."""
    tools: List[BaseTool] = []

    def _make_tool_impl(tool_name: str, tool_description: str):
        def _tool_impl(prompt: str) -> str:
            result = execute_tool(tool_name, prompt)
            if not result:
                payload = {
                    "name": tool_name,
                    "message": "tool not found",
                    "data": {},
                }
            else:
                payload = result.model_dump(mode="json")
            return json.dumps(payload, ensure_ascii=True, default=str)

        _tool_impl.__doc__ = tool_description or ""
        return _tool_impl

    for spec in list_tool_specs():
        name = spec["name"]
        description = spec["description"]
        tool_impl = _make_tool_impl(name, description)
        tools.append(lc_tool(name)(tool_impl))
    return tools
