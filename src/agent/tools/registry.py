import json
import re
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain_core.tools import BaseTool, tool as lc_tool
from ...infra.config import get_config
from ...infra.cache_keys import parse_planting_cache_key
from ...infra.llm import get_chat_model
from ...infra.variety_store import retrieve_variety_candidates
from ...infra.tool_cache import get_tool_result_cache
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...domain.services import DEFAULT_CROP
from ...observability.logging_utils import log_event
from ...schemas.models import (
    OperationItem,
    OperationPlanResult,
    ToolInvocation,
    WeatherDataPoint,
    WeatherQueryInput,
    WeatherSeries,
)


TOOLS: List[BaseTool] = []
TOOL_INDEX: Dict[str, BaseTool] = {}
HIDDEN_TOOL_NAMES = {"farming_recommendation", "growth_stage_prediction"}


TOOL_CACHEABLE = {
    "variety_lookup",
    "weather_lookup",
    "farming_recommendation",
    "growth_stage_prediction",
}
VARIETY_DB_TABLE = "variety_approvals"
VARIETY_DB_OUTPUT_FIELDS = (
    "variety_name",
    "approval_year",
    "approval_region",
    "suitable_region",
    "rice_type",
    "subspecies_type",
    "maturity",
    "control_variety",
    "days_vs_control",
)
VARIETY_DB_FIELD_LABELS = {
    "variety_name": "品种名称",
    "approval_year": "审定年份",
    "approval_region": "审定区域",
    "suitable_region": "适种地区",
    "rice_type": "稻作类型",
    "subspecies_type": "亚种类型",
    "maturity": "熟期",
    "control_variety": "对照品种",
    "days_vs_control": "比对照长(天)",
}


class VarietyMatchDecision(BaseModel):
    index: int
    reason: Optional[str] = None


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


def get_cached_tool_result(
    tool_name: str, provider: str, prompt: str
) -> Optional[ToolInvocation]:
    provider = normalize_provider(provider)
    return _get_cached_tool_result(tool_name, provider, prompt)


def cache_tool_result(
    tool_name: str, provider: str, prompt: str, result: ToolInvocation
) -> None:
    _store_tool_result(tool_name, provider, prompt, result)


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
    candidates = retrieve_variety_candidates(prompt, limit=1)
    return candidates[0] if candidates else None


def _infer_crop_and_variety(prompt: str) -> Tuple[str, str]:
    crop_keywords = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
    crop = next((item for item in crop_keywords if item in prompt), "水稻")
    variety = _extract_variety(prompt)
    if not variety:
        variety = "美香占2号" if crop == "水稻" else f"{crop}示例品种"
    return crop, variety


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


def _get_variety_db_path() -> Optional[Path]:
    cfg = get_config()
    if cfg.variety_db_path:
        return Path(cfg.variety_db_path)
    return Path(__file__).resolve().parents[2] / "resources" / "rice_variety_approvals.sqlite3"


def _query_variety_db_by_name(
    conn: sqlite3.Connection, name: str, limit: int
) -> List[sqlite3.Row]:
    rows = conn.execute(
        f"SELECT {', '.join(VARIETY_DB_OUTPUT_FIELDS)} "
        f"FROM {VARIETY_DB_TABLE} WHERE variety_name = ?",
        (name,),
    ).fetchall()
    if rows:
        return rows
    like_prefix = f"{name}%"
    rows = conn.execute(
        f"SELECT {', '.join(VARIETY_DB_OUTPUT_FIELDS)} "
        f"FROM {VARIETY_DB_TABLE} WHERE variety_name LIKE ? LIMIT ?",
        (like_prefix, limit),
    ).fetchall()
    if rows:
        return rows
    like_any = f"%{name}%"
    return conn.execute(
        f"SELECT {', '.join(VARIETY_DB_OUTPUT_FIELDS)} "
        f"FROM {VARIETY_DB_TABLE} WHERE variety_name LIKE ? LIMIT ?",
        (like_any, limit),
    ).fetchall()


def _query_variety_db_by_prompt(
    conn: sqlite3.Connection, prompt: str, limit: int
) -> List[sqlite3.Row]:
    return conn.execute(
        f"SELECT {', '.join(VARIETY_DB_OUTPUT_FIELDS)} "
        f"FROM {VARIETY_DB_TABLE} "
        "WHERE ? LIKE '%' || variety_name || '%' "
        "ORDER BY LENGTH(variety_name) DESC LIMIT ?",
        (prompt, limit),
    ).fetchall()


def _rows_to_variety_records(rows: List[sqlite3.Row]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        record: Dict[str, object] = {}
        for field in VARIETY_DB_OUTPUT_FIELDS:
            record[VARIETY_DB_FIELD_LABELS[field]] = row[field]
        records.append(record)
    return records


def _lookup_variety_records(prompt: str, *, limit: int = 5) -> List[Dict[str, object]]:
    path = _get_variety_db_path()
    if not path or not path.exists():
        return []
    prompt_text = prompt or ""
    variety_name = _extract_variety(prompt_text)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows: List[sqlite3.Row] = []
            if variety_name:
                rows = _query_variety_db_by_name(conn, variety_name, limit)
            if not rows and prompt_text:
                rows = _query_variety_db_by_prompt(conn, prompt_text, limit)
    except Exception:
        return []
    return _rows_to_variety_records(rows)


def _normalize_region_token(value: str) -> str:
    return re.sub(r"(省|市|州|盟|地区|区|县)$", "", value or "").strip()


def _extract_region_tokens(
    prompt: str, records: List[Dict[str, object]]
) -> List[str]:
    tokens: List[str] = []
    if prompt:
        tokens.extend(
            re.findall(r"[\u4e00-\u9fff]{2,8}(?:省|市|州|盟|地区|区|县)", prompt)
        )
    approval_regions = {
        str(record.get("审定区域") or "").strip()
        for record in records
        if record.get("审定区域")
    }
    if prompt:
        for region in approval_regions:
            if region and region in prompt:
                tokens.append(region)
    unique: List[str] = []
    seen = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _parse_approval_year(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) == 4:
        return int(text)
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def _score_record_by_region(
    record: Dict[str, object], region_tokens: List[str]
) -> int:
    approval_region = str(record.get("审定区域") or "")
    suitable_region = str(record.get("适种地区") or "")
    best = 0
    for token in region_tokens:
        normalized = _normalize_region_token(token)
        if token and token in suitable_region:
            best = max(best, 100)
        elif normalized and normalized in suitable_region:
            best = max(best, 90)
        if token and token in approval_region:
            best = max(best, 80)
        elif normalized and normalized in approval_region:
            best = max(best, 70)
    return best


def _pick_latest_year_record(
    records: List[Dict[str, object]], indices: Optional[List[int]] = None
) -> int:
    best_index = (indices[0] if indices else 0)
    best_year = -1
    for idx, record in enumerate(records):
        if indices and idx not in indices:
            continue
        year = _parse_approval_year(record.get("审定年份")) or 0
        if year > best_year:
            best_year = year
            best_index = idx
            continue
        if year == best_year:
            region = str(record.get("审定区域") or "")
            current = str(records[best_index].get("审定区域") or "")
            if region == "国审" and current != "国审":
                best_index = idx
    return best_index


def _llm_choose_variety_record(
    prompt: str,
    candidates: List[Dict[str, object]],
    region_tokens: List[str],
) -> Optional[VarietyMatchDecision]:
    try:
        llm = get_chat_model()
    except Exception:
        return None
    system_prompt = (
        "你是品种审定记录选择器，根据用户种植地点与审定信息选择最匹配的一条记录。"
        "只输出 JSON：index(候选列表序号)、reason(简短理由)。"
    )
    payload = {
        "prompt": prompt,
        "region_tokens": region_tokens,
        "candidates": candidates,
    }
    try:
        chooser = llm.with_structured_output(VarietyMatchDecision)
        result = chooser.invoke(
            [
                ("system", system_prompt),
                ("human", json.dumps(payload, ensure_ascii=True, default=str)),
            ]
        )
        decision = (
            result
            if isinstance(result, VarietyMatchDecision)
            else VarietyMatchDecision.model_validate(result)
        )
    except Exception:
        return None
    if decision.index < 0 or decision.index >= len(candidates):
        return None
    return decision


def _select_best_variety_record(
    prompt: str, records: List[Dict[str, object]]
) -> Tuple[Dict[str, object], str]:
    region_tokens = _extract_region_tokens(prompt, records)
    if region_tokens:
        scored = [
            (_score_record_by_region(record, region_tokens), idx)
            for idx, record in enumerate(records)
        ]
        max_score = max(score for score, _ in scored)
        best_indices = [idx for score, idx in scored if score == max_score]
        if max_score > 0 and len(best_indices) == 1:
            return records[best_indices[0]], "规则匹配"
        if max_score > 0 and len(best_indices) > 1:
            candidates = [
                {
                    "index": i,
                    "审定区域": records[i].get("审定区域"),
                    "适种地区": records[i].get("适种地区"),
                    "审定年份": records[i].get("审定年份"),
                    "稻作类型": records[i].get("稻作类型"),
                    "亚种类型": records[i].get("亚种类型"),
                    "熟期": records[i].get("熟期"),
                    "对照品种": records[i].get("对照品种"),
                }
                for i in best_indices
            ]
            decision = _llm_choose_variety_record(
                prompt, candidates, region_tokens
            )
            if decision:
                selected = records[decision.index]
                reason = decision.reason or "LLM 匹配"
                log_event(
                    "variety_match_llm_choice",
                    selected_index=decision.index,
                    reason=reason,
                )
                return selected, reason
            fallback = _pick_latest_year_record(records, best_indices)
            return records[fallback], "年份优先"
    fallback = _pick_latest_year_record(records)
    return records[fallback], "年份优先"


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
    records = _lookup_variety_records(prompt)
    if records:
        selected, reason = _select_best_variety_record(prompt, records)
        variety = selected.get("品种名称") or _extract_variety(prompt) or "未知"
        approval_regions = sorted(
            {r.get("审定区域") for r in records if r.get("审定区域")}
        )
        region_note = f"（{len(approval_regions)}个区域）" if approval_regions else ""
        payload = {
            "query": prompt,
            "crop": DEFAULT_CROP,
            "variety": variety,
            "selected": selected,
            "selection_reason": reason,
            "matches": records,
            "source": "sqlite",
        }
        message = f"已返回品种 {variety} 的审定信息{region_note}。"
        result = ToolInvocation(
            name="variety_lookup",
            message=message,
            data=payload,
        )
        _store_tool_result("variety_lookup", provider, prompt, result)
        return result
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
        "仅用于读取已缓存的生育期预测结果。"
        "缺少缓存时需走生育期预测 workflow。"
    ),
)
def growth_stage_prediction(prompt: str) -> ToolInvocation:
    provider = "workflow"
    cache_key = parse_planting_cache_key(prompt)
    if cache_key:
        cached = _get_cached_tool_result(
            "growth_stage_prediction", provider, cache_key
        )
        if cached:
            return cached
    return ToolInvocation(
        name="growth_stage_prediction",
        message="生育期预测必须走工作流，当前仅支持返回历史缓存结果。",
        data={"cache_hit": False},
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
