import json
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from langchain_core.tools import BaseTool, tool as lc_tool
from ..infra.config import get_config
from ..infra.llm_extract import llm_structured_extract
from ..infra.variety_store import build_variety_hint
from ..infra.variety_store import retrieve_variety_candidates
from ..infra.tool_cache import get_tool_result_cache
from ..infra.tool_provider import maybe_intranet_tool, normalize_provider
from ..domain.services import (
    MissingPlantingInfoError,
    list_missing_required_fields,
    merge_planting_answers,
    normalize_and_validate_planting,
)
from ..observability.logging_utils import log_event
from ..schemas.models import (
    OperationItem,
    OperationPlanResult,
    PlantingDetailsDraft,
    PredictGrowthStageInput,
    ToolInvocation,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)


TOOLS: List[BaseTool] = []
TOOL_INDEX: Dict[str, BaseTool] = {}
HIDDEN_TOOL_NAMES = {"farming_recommendation", "growth_stage_prediction"}

VARIETY_PATTERN = re.compile(r"(?:品种|品名|品系)[:：\s]*([\w\u4e00-\u9fff-]{2,20})")
VARIETY_FALLBACK_PATTERN = re.compile(r"([\w\u4e00-\u9fff-]{2,20}号)")

PLANTING_FIELD_LABELS = {
    "crop": "作物",
    "variety": "品种",
    "planting_method": "种植方式",
    "sowing_date": "播种日期",
    "region": "地区",
}

TOOL_CACHEABLE = {"variety_lookup", "weather_lookup", "farming_recommendation"}


def register_tool(tool: BaseTool) -> None:
    """Register a custom LangChain tool for routing."""
    TOOLS.append(tool)
    TOOL_INDEX[tool.name] = tool


def _should_cache_tool_result(result: ToolInvocation) -> bool:
    return bool(result.data)


def _get_cached_tool_result(
    tool_name: str, provider: str, prompt: str
) -> Optional[ToolInvocation]:
    if tool_name not in TOOL_CACHEABLE:
        return None
    cache = get_tool_result_cache()
    payload = cache.get(tool_name, provider, prompt)
    if not payload:
        return None
    try:
        return ToolInvocation(**payload)
    except Exception:
        return None


def _store_tool_result(
    tool_name: str, provider: str, prompt: str, result: ToolInvocation
) -> None:
    if tool_name not in TOOL_CACHEABLE:
        return None
    if not _should_cache_tool_result(result):
        return None
    cache = get_tool_result_cache()
    cache.set(tool_name, provider, prompt, result.model_dump(mode="json"))


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


def _summarize_tool_output(result: ToolInvocation) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "name": result.name,
        "message": result.message,
    }
    data = result.data or {}
    if isinstance(data, dict):
        summary["data_keys"] = list(data.keys())
        points = data.get("points")
        if isinstance(points, list):
            summary["points_count"] = len(points)
        operations = data.get("operations")
        if isinstance(operations, list):
            summary["operations_count"] = len(operations)
        recommendations = data.get("recommendations")
        if isinstance(recommendations, list):
            summary["recommendations_count"] = len(recommendations)
    return summary


def _parse_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, date):
        return value.year
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) == 4:
            return int(text)
        try:
            return date.fromisoformat(text).year
        except ValueError:
            return None
    return None


def _build_weather_query_from_payload(
    payload: Dict[str, object],
) -> Optional[WeatherQueryInput]:
    region = payload.get("region")
    if not region:
        return None
    year = _parse_year(payload.get("year"))
    if year is None:
        year = _parse_year(payload.get("start_date")) or _parse_year(payload.get("end_date"))
    if year is None:
        return None
    data: Dict[str, object] = {"region": region, "year": year}
    granularity = payload.get("granularity")
    if granularity in {"hourly", "daily"}:
        data["granularity"] = granularity
    include_advice = payload.get("include_advice")
    if isinstance(include_advice, bool):
        data["include_advice"] = include_advice
    try:
        return WeatherQueryInput(**data)
    except Exception:
        return None


def _normalize_weather_prompt(
    prompt: str,
) -> Tuple[str, Optional[WeatherQueryInput]]:
    if not prompt:
        return "", None
    text = prompt.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            query = WeatherQueryInput(region=text, year=date.today().year)
        except Exception:
            return text, None
        canonical = json.dumps(
            query.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        return canonical, query
    if not isinstance(payload, dict):
        return text, None
    query = _build_weather_query_from_payload(payload)
    if query is None:
        return text, None
    canonical = json.dumps(
        query.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
    return canonical, query


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
    cached = _get_cached_tool_result("variety_lookup", provider, prompt)
    if cached:
        return cached
    intranet = maybe_intranet_tool(
        "variety_lookup",
        prompt,
        provider,
        cfg.variety_api_url,
        cfg.variety_api_key,
    )
    if intranet:
        _store_tool_result("variety_lookup", provider, prompt, intranet)
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
    result = ToolInvocation(
        name="variety_lookup",
        message=f"已返回 {crop} 品种 {variety} 的模拟信息。",
        data=payload,
    )
    _store_tool_result("variety_lookup", provider, prompt, result)
    return result


@auto_register_tool(
    "weather_lookup",
    description="查询指定地区气象数据。仅用于获取天气数据本身；不生成农事建议或计划。",
)
def weather_lookup(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.weather_provider)
    prompt = prompt or ""
    cache_prompt, query = _normalize_weather_prompt(prompt)
    cached = _get_cached_tool_result("weather_lookup", provider, cache_prompt)
    if cached:
        return cached
    intranet_prompt = cache_prompt if query else prompt
    intranet = maybe_intranet_tool(
        "weather_lookup",
        intranet_prompt,
        provider,
        cfg.weather_api_url,
        cfg.weather_api_key,
    )
    if intranet:
        _store_tool_result("weather_lookup", provider, cache_prompt, intranet)
        return intranet
    if query:
        start = date(query.year, 1, 1)
        end = date(query.year, 12, 31)
        granularity = query.granularity or "daily"
        region = query.region
        total_days = max(1, (end - start).days + 1)
    else:
        start = date.today()
        end = start + timedelta(days=2)
        granularity = "daily"
        region = prompt or "unknown"
        total_days = 3
    points: List[WeatherDataPoint] = []
    for offset in range(total_days):
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
        region=region,
        granularity=granularity,
        start_date=start,
        end_date=end,
        points=points,
        source="mock",
    )
    result = ToolInvocation(
        name="weather_lookup",
        message="已返回模拟气象数据。",
        data=series.model_dump(mode="json"),
    )
    _store_tool_result("weather_lookup", provider, cache_prompt, result)
    return result


@auto_register_tool(
    "growth_stage_prediction",
    description=(
        "基于自然语言抽取种植信息（作物/品种/方式/播期/地区），"
        "补齐后调用内网生育期接口完成预测。仅用于生育期预测。"
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
    weather_summary = {
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
    request = PredictGrowthStageInput(planting=planting, weatherSeries=weather_series)
    request_payload = json.dumps(
        request.model_dump(mode="json"), ensure_ascii=True, default=str
    )
    cfg = get_config()
    provider = normalize_provider(cfg.growth_stage_provider)
    intranet = maybe_intranet_tool(
        "growth_stage_prediction",
        request_payload,
        provider,
        cfg.growth_stage_api_url,
        cfg.growth_stage_api_key,
    )
    trace.append(
        "growth_stage_intranet ok" if intranet else "growth_stage_intranet missing"
    )
    workflow_payload = {
        "inputs": {
            "planting": planting.model_dump(exclude_none=True, mode="json"),
        },
        "variety": variety_data,
        "weather_summary": weather_summary,
        "trace": trace,
    }
    if intranet:
        payload = dict(intranet.data or {})
        payload.setdefault("workflow", workflow_payload)
        return ToolInvocation(
            name=intranet.name,
            message=intranet.message,
            data=payload,
        )
    return ToolInvocation(
        name="growth_stage_prediction",
        message="生育期预测需要内网接口，请配置 GROWTH_STAGE_PROVIDER=intranet。",
        data={"workflow": workflow_payload},
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
    cached = _get_cached_tool_result("farming_recommendation", provider, prompt)
    if cached:
        return cached
    intranet = maybe_intranet_tool(
        "farming_recommendation",
        prompt,
        provider,
        cfg.recommendation_api_url,
        cfg.recommendation_api_key,
    )
    if intranet:
        _store_tool_result("farming_recommendation", provider, prompt, intranet)
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
    result = ToolInvocation(
        name="farming_recommendation",
        message="已返回模拟农事推荐。",
        data=plan.model_dump(mode="json"),
    )
    _store_tool_result("farming_recommendation", provider, prompt, result)
    return result


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
        log_event("tool_output", tool=_summarize_tool_output(result))
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
