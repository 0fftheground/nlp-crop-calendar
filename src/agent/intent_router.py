from __future__ import annotations

import re
import time
from collections import OrderedDict
from datetime import date
from threading import Lock
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..infra.llm import get_extractor_model
from ..observability.llm_usage import log_llm_error, log_llm_request, log_llm_response
from ..observability.logging_utils import log_event
from .intent_boundaries import normalize_prompt
from .planner import ActionPlan


FAST_INTENT_MIN_CONFIDENCE = 0.7
FAST_INTENT_MIN_CONFIDENCE_NONE = 0.85
INTENT_CACHE_MIN_PROMPT_LEN = 2
_QUERY_WRAPPER_TOOLS = {
    "variety_lookup",
    "sowing_suitability_lookup",
    "growth_stage_lookup",
    "plant_task_create",
}
_WEATHER_REGION_RE = re.compile(
    r"(?P<region>[\u4e00-\u9fa5]{2,10})(?:的)?"
    r"(?:天气|气温|气象|降雨|降水|湿度|风速|预报)"
)
_YEAR_RE = re.compile(r"(20\d{2})\s*年?")
_DATE_TEXT_RE = re.compile(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})")
_DATE_TEXT_COMPACT_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_SOWING_QUERY_TOKENS = (
    "播种",
    "播期",
    "适播",
    "适合播",
    "什么时候播",
    "何时播",
)
_PLAN_WORKFLOW_TOKENS = ("方案", "计划", "建立", "生成", "创建")
_NON_AGRI_LIFE_TOKENS = ("旅游", "景点", "酒店", "穿衣", "洗车", "出行")


class _NonAgriBoundaryResult(BaseModel):
    out_of_scope: bool
    confidence: float
    reason: Optional[str] = None


class _IntentPlanCache:
    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self._max_items = max(1, int(max_items))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: "OrderedDict[str, tuple[dict, int]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[dict]:
        if not key:
            return None
        now = int(time.time())
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            payload, expires_at = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return dict(payload)

    def set(self, key: str, payload: dict) -> None:
        if not key:
            return None
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock:
            self._items[key] = (dict(payload), expires_at)
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)


class IntentRouter:
    def __init__(
        self,
        *,
        tool_names: set[str],
        workflow_names: set[str],
        planner,
        rule_engine,
        fast_router,
        cache_max_items: int,
        cache_ttl_seconds: int,
    ) -> None:
        self._tool_names = tool_names
        self._workflow_names = workflow_names
        self._planner = planner
        self._rule_engine = rule_engine
        self._fast_router = fast_router
        self._intent_cache = _IntentPlanCache(cache_max_items, cache_ttl_seconds)
        self._boundary_cache = _IntentPlanCache(cache_max_items, cache_ttl_seconds)
        self._non_agri_boundary_prompt = (
            "你是农业种植助手的域边界判定器。\n"
            "判断用户输入是否明显不属于农业种植辅助系统支持范围。\n"
            "如果问题的目标是旅游、酒店、景点、交通、穿衣、洗车等生活场景，"
            "且没有明确农业任务结构，则 out_of_scope=true。\n"
            "如果问题属于或可能属于农业任务（天气适宜度、播期推荐、品种信息、"
            "生育期、种植计划、农事记录等），则 out_of_scope=false。\n"
            "输出严格 JSON："
            '{"out_of_scope":true|false,"confidence":0-1,"reason":"..."}'
        )
        try:
            self._boundary_llm = get_extractor_model()
        except Exception as exc:
            self._boundary_llm = None
            log_event("intent_boundary_llm_unavailable", error=str(exc))

    def plan(
        self,
        prompt: str,
        *,
        pending: Optional[dict],
        intent_mode: str,
    ) -> Optional[ActionPlan]:
        if intent_mode == "llm_only":
            plan = self._planner.plan(prompt, pending=pending)
            return self._normalize_plan_for_boundaries(plan, prompt)
        plan = self._rule_route(prompt)
        if plan:
            log_event(
                "intent_rule_hit",
                action=plan.action,
                name=plan.name,
                reason=plan.reason,
            )
            return self._normalize_plan_for_boundaries(plan, prompt)
        cached_plan = self._get_cached_plan(prompt)
        if cached_plan:
            log_event(
                "intent_cache_hit",
                action=cached_plan.action,
                name=cached_plan.name,
            )
            return self._normalize_plan_for_boundaries(cached_plan, prompt)
        keyword_plan = self._keyword_route(prompt)
        if keyword_plan:
            log_event(
                "intent_keyword_hit",
                action=keyword_plan.action,
                name=keyword_plan.name,
                reason=keyword_plan.reason,
            )
            keyword_plan = self._normalize_plan_for_boundaries(keyword_plan, prompt)
            self._cache_plan(prompt, keyword_plan)
            return keyword_plan
        fast_plan = self._fast_intent_route(prompt, pending)
        if fast_plan:
            log_event("intent_fast_hit", action=fast_plan.action, name=fast_plan.name)
            fast_plan = self._normalize_plan_for_boundaries(fast_plan, prompt)
            self._cache_plan(prompt, fast_plan)
            return fast_plan
        plan = self._planner.plan(prompt, pending=pending)
        if plan:
            plan = self._normalize_plan_for_boundaries(plan, prompt)
            self._cache_plan(prompt, plan)
        return plan

    def _rule_route(self, prompt: str) -> Optional[ActionPlan]:
        text = normalize_prompt(prompt)
        if not text:
            return None
        if self._is_non_agri_out_of_scope(text):
            return ActionPlan(
                action="none",
                reason="boundary:non_agri_life_query",
            )
        lowered = text.lower()
        for name in self._tool_names:
            if name.lower() in lowered:
                return ActionPlan(
                    action="tool",
                    name=name,
                    input=self._default_plan_input("tool", name, text),
                )
        for name in self._workflow_names:
            if name.lower() in lowered:
                return ActionPlan(
                    action="workflow",
                    name=name,
                    input=self._default_plan_input("workflow", name, text),
                )
        rule = self._rule_engine.match(text)
        if rule:
            return self._plan_from_rule(rule, text)
        return None

    def _keyword_route(self, prompt: str) -> Optional[ActionPlan]:
        text = normalize_prompt(prompt)
        if not text:
            return None
        if self._looks_like_sowing_query(text):
            return ActionPlan(
                action="tool",
                name="sowing_suitability_lookup",
                input=self._default_plan_input("tool", "sowing_suitability_lookup", text),
                reason="keyword:sowing",
            )
        if self._looks_like_crop_calendar_workflow_prompt(text):
            return ActionPlan(
                action="workflow",
                name="crop_calendar_workflow",
                input=self._default_plan_input("workflow", "crop_calendar_workflow", text),
                reason="keyword:crop_calendar",
            )
        return None

    @staticmethod
    def _looks_like_sowing_query(text: str) -> bool:
        prompt = normalize_prompt(text)
        if not prompt:
            return False
        if any(token in prompt for token in _PLAN_WORKFLOW_TOKENS):
            return False
        return any(token in prompt for token in _SOWING_QUERY_TOKENS)

    @staticmethod
    def _looks_like_crop_calendar_workflow_prompt(text: str) -> bool:
        prompt = normalize_prompt(text)
        if not prompt:
            return False
        if not any(token in prompt for token in _PLAN_WORKFLOW_TOKENS):
            return False
        return any(token in prompt for token in ("种植", "方案", "计划", "移栽", "直播", "品种"))

    @staticmethod
    def _looks_like_non_agri_life_query(text: str) -> bool:
        prompt = normalize_prompt(text)
        if not prompt:
            return False
        if not any(token in prompt for token in _NON_AGRI_LIFE_TOKENS):
            return False
        return not any(
            token in prompt
            for token in ("施肥", "打药", "播种", "播期", "生育期", "种植", "农事", "计划")
        )

    def _fast_intent_route(
        self, prompt: str, pending: Optional[dict]
    ) -> Optional[ActionPlan]:
        if not self._fast_router:
            return None
        result = self._fast_router.classify(prompt, pending=pending)
        if not result:
            return None
        if result.action == "none":
            if result.confidence < FAST_INTENT_MIN_CONFIDENCE_NONE:
                return None
            return ActionPlan(action="none", reason=result.reason)
        if result.confidence < FAST_INTENT_MIN_CONFIDENCE:
            return None
        if not result.name:
            return None
        plan_input = result.input
        if plan_input in (None, "", {}, []):
            plan_input = self._default_plan_input(result.action, result.name, prompt)
        else:
            plan_input = self._merge_default_plan_input(
                result.action, result.name, plan_input, prompt
            )
        return ActionPlan(
            action=result.action,
            name=result.name,
            input=plan_input,
            reason=result.reason,
        )

    def _default_plan_input(self, action: str, name: str, prompt: str) -> object:
        if action == "workflow":
            return {"prompt": prompt}
        if action != "tool":
            return None
        if name in _QUERY_WRAPPER_TOOLS:
            return {"query": prompt}
        if name == "weather_lookup":
            return self._build_weather_payload(prompt)
        if name == "memory_clear":
            return {}
        return prompt

    def _merge_default_plan_input(
        self, action: str, name: str, payload: object, prompt: str
    ) -> object:
        if not isinstance(payload, dict):
            return payload
        merged = dict(payload)
        if action == "workflow" and "prompt" not in merged:
            merged["prompt"] = prompt
            return merged
        if action == "tool" and name in _QUERY_WRAPPER_TOOLS and "query" not in merged:
            merged["query"] = prompt
        return merged

    def _plan_from_rule(self, rule, prompt: str) -> Optional[ActionPlan]:
        if rule.action == "none":
            return ActionPlan(action="none", reason=f"rule:{rule.id}")
        if rule.action in {"tool", "workflow"} and not rule.name:
            return None
        plan_input = None
        handler = (rule.handler or "").lower()
        if handler == "weather":
            plan_input = self._build_weather_payload(prompt)
            if not plan_input:
                return None
        else:
            plan_input = self._default_plan_input(rule.action, rule.name, prompt)
        return ActionPlan(
            action=rule.action,
            name=rule.name,
            input=plan_input,
            reason=f"rule:{rule.id}",
        )

    def _build_weather_payload(self, prompt: str) -> Optional[dict]:
        text = normalize_prompt(prompt)
        if not text:
            return None
        match = _WEATHER_REGION_RE.search(text)
        region = match.group("region") if match else None
        payload: dict = {}
        if region:
            payload["region"] = region
        dates = []
        for match in _DATE_TEXT_RE.finditer(text):
            try:
                dates.append(
                    date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                )
            except ValueError:
                continue
        for match in _DATE_TEXT_COMPACT_RE.finditer(text):
            try:
                dates.append(
                    date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                )
            except ValueError:
                continue
        if dates:
            payload["start_date"] = dates[0].isoformat()
            if len(dates) >= 2:
                payload["end_date"] = dates[1].isoformat()
        year_match = _YEAR_RE.search(text)
        if year_match:
            try:
                payload["year"] = int(year_match.group(1))
            except ValueError:
                pass
        return payload or {}

    def _normalize_plan_for_boundaries(
        self, plan: Optional[ActionPlan], prompt: str
    ) -> Optional[ActionPlan]:
        if plan is None:
            return None
        if self._is_non_agri_out_of_scope(prompt):
            if plan.action != "none":
                return ActionPlan(
                    action="none",
                    reason="boundary:non_agri_life_query",
                )
            return plan
        text = normalize_prompt(prompt)
        if (
            plan.action == "tool"
            and plan.name == "weather_lookup"
            and self._looks_like_sowing_query(text)
        ):
            return ActionPlan(
                action="tool",
                name="sowing_suitability_lookup",
                input=self._default_plan_input(
                    "tool", "sowing_suitability_lookup", prompt
                ),
                reason="boundary:sowing_not_weather",
            )
        if (
            plan.action == "tool"
            and plan.name == "weather_lookup"
            and self._looks_like_crop_calendar_workflow_prompt(text)
        ):
            return ActionPlan(
                action="workflow",
                name="crop_calendar_workflow",
                input=self._default_plan_input("workflow", "crop_calendar_workflow", prompt),
                reason="boundary:workflow_not_weather",
            )
        return plan

    def _build_boundary_cache_key(self, prompt: str) -> str:
        text = normalize_prompt(prompt)
        if len(text) < INTENT_CACHE_MIN_PROMPT_LEN:
            return ""
        return f"boundary:non_agri:{text.lower()}"

    def _is_non_agri_out_of_scope(self, prompt: str) -> bool:
        text = normalize_prompt(prompt)
        if not text:
            return False
        if self._looks_like_non_agri_life_query(text):
            return True
        cache_key = self._build_boundary_cache_key(text)
        if cache_key:
            cached = self._boundary_cache.get(cache_key)
            if isinstance(cached, dict):
                value = cached.get("out_of_scope")
                if isinstance(value, bool):
                    return value
        result = self._classify_non_agri_out_of_scope(text)
        if cache_key:
            self._boundary_cache.set(cache_key, {"out_of_scope": result})
        return result

    def _classify_non_agri_out_of_scope(self, prompt: str) -> bool:
        if self._boundary_llm is None:
            return False
        try:
            extractor = self._boundary_llm.with_structured_output(
                _NonAgriBoundaryResult
            )
            log_llm_request(
                "intent_boundary_non_agri",
                model=self._boundary_llm,
                system_prompt=self._non_agri_boundary_prompt,
                user_prompt=prompt,
            )
            started_at = time.perf_counter()
            raw_result = extractor.invoke(
                [
                    SystemMessage(content=self._non_agri_boundary_prompt),
                    HumanMessage(content=prompt),
                ]
            )
            log_llm_response(
                "intent_boundary_non_agri",
                model=self._boundary_llm,
                result=raw_result,
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                response_text=getattr(raw_result, "content", raw_result),
            )
            payload = (
                raw_result.model_dump(exclude_none=True)
                if hasattr(raw_result, "model_dump")
                else raw_result
            )
            result = _NonAgriBoundaryResult.model_validate(payload)
            log_event(
                "intent_boundary_non_agri_result",
                out_of_scope=result.out_of_scope,
                confidence=result.confidence,
                reason=result.reason,
            )
            if result.out_of_scope and float(result.confidence or 0.0) >= 0.8:
                return True
            return False
        except Exception as exc:
            log_llm_error(
                "intent_boundary_non_agri",
                model=self._boundary_llm,
                system_prompt=self._non_agri_boundary_prompt,
                user_prompt=prompt,
                error=exc,
            )
            log_event(
                "intent_boundary_non_agri_fallback",
                error=str(exc),
                fallback="planner",
            )
            return False

    def _build_intent_cache_key(self, prompt: str) -> str:
        text = normalize_prompt(prompt)
        if len(text) < INTENT_CACHE_MIN_PROMPT_LEN:
            return ""
        return text.lower()

    def _get_cached_plan(self, prompt: str) -> Optional[ActionPlan]:
        key = self._build_intent_cache_key(prompt)
        if not key:
            return None
        payload = self._intent_cache.get(key)
        if not payload:
            return None
        try:
            return ActionPlan.model_validate(payload)
        except Exception:
            return None

    def _cache_plan(self, prompt: str, plan: ActionPlan) -> None:
        if plan.action == "none":
            return None
        if not plan.name:
            return None
        key = self._build_intent_cache_key(prompt)
        if not key:
            return None
        payload = plan.model_dump(mode="json")
        self._intent_cache.set(key, payload)
