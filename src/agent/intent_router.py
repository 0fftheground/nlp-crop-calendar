from __future__ import annotations

import re
import time
from collections import OrderedDict
from datetime import date
from threading import Lock
from typing import Optional

from ..observability.logging_utils import log_event
from .planner import ActionPlan


FAST_INTENT_MIN_CONFIDENCE = 0.7
FAST_INTENT_MIN_CONFIDENCE_NONE = 0.85
INTENT_CACHE_MIN_PROMPT_LEN = 2
_WEATHER_REGION_RE = re.compile(
    r"(?P<region>[\u4e00-\u9fa5]{2,10})(?:的)?"
    r"(?:天气|气温|气象|降雨|降水|湿度|风速|预报)"
)
_YEAR_RE = re.compile(r"(20\d{2})\s*年?")
_DATE_TEXT_RE = re.compile(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})")
_DATE_TEXT_COMPACT_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_SOWING_QUERY_CUES = (
    "播种",
    "播期",
    "适播",
)
_SOWING_INTENT_CUES = (
    "适合",
    "适宜",
    "什么时候",
    "何时",
    "窗口",
    "推荐",
    "怎么播",
    "播吗",
    "播嘛",
    "播呢",
)
_PLAN_QUERY_CUES = ("计划", "方案", "生成", "制定", "新增", "创建")


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_like_sowing_query(text: str) -> bool:
    prompt = _normalize_prompt(text)
    if not prompt:
        return False
    if any(token in prompt for token in _PLAN_QUERY_CUES):
        return False
    if not any(token in prompt for token in _SOWING_QUERY_CUES):
        return False
    if any(token in prompt for token in _SOWING_INTENT_CUES):
        return True
    return len(prompt) <= 12


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
        text = _normalize_prompt(prompt)
        if not text:
            return None
        if _looks_like_sowing_query(text):
            return ActionPlan(
                action="tool",
                name="sowing_suitability_lookup",
                input=self._default_plan_input(
                    "tool", "sowing_suitability_lookup", text
                ),
                reason="rule:sowing_query",
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
        if plan_input is None or plan_input == "":
            plan_input = self._default_plan_input(result.action, result.name, prompt)
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
        if name == "variety_lookup":
            return {"query": prompt}
        if name == "sowing_suitability_lookup":
            return {"query": prompt}
        if name == "weather_lookup":
            return self._build_weather_payload(prompt)
        if name == "memory_clear":
            return {}
        return prompt

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
        text = _normalize_prompt(prompt)
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
        if _looks_like_sowing_query(prompt):
            if plan.action != "tool" or plan.name != "sowing_suitability_lookup":
                return ActionPlan(
                    action="tool",
                    name="sowing_suitability_lookup",
                    input=self._default_plan_input(
                        "tool", "sowing_suitability_lookup", _normalize_prompt(prompt)
                    ),
                    reason="boundary:sowing_query",
                )
        return plan

    def _build_intent_cache_key(self, prompt: str) -> str:
        text = _normalize_prompt(prompt)
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
