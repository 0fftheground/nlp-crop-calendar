from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ..observability.logging_utils import log_event


class IntentRule(BaseModel):
    id: str
    action: Literal["tool", "workflow", "none"]
    name: Optional[str] = None
    priority: int = 0
    exact: List[str] = Field(default_factory=list)
    full_regex: List[str] = Field(default_factory=list)
    enabled: bool = True
    handler: Optional[str] = None


class IntentRuleSet(BaseModel):
    version: Optional[str] = None
    rules: List[IntentRule] = Field(default_factory=list)


@dataclass(frozen=True)
class _CompiledRule:
    rule: IntentRule
    exact: List[str]
    full_regex: List[object]


class IntentRuleEngine:
    def __init__(self, path: Path, reload_seconds: int = 5) -> None:
        self._path = path
        self._reload_seconds = max(1, int(reload_seconds))
        self._last_check = 0.0
        self._last_mtime: Optional[float] = None
        self._rules: List[_CompiledRule] = []

    def match(self, prompt: str) -> Optional[IntentRule]:
        self._reload_if_needed()
        if not self._rules:
            return None
        text = (prompt or "").strip()
        if not text:
            return None
        for compiled in self._rules:
            rule = compiled.rule
            if not rule.enabled:
                continue
            exact_hit = text in compiled.exact
            regex_hit = False
            if compiled.full_regex:
                for regex in compiled.full_regex:
                    if regex.fullmatch(text):
                        regex_hit = True
                        break
            if not exact_hit and not regex_hit:
                continue
            if rule.action in {"tool", "workflow"} and not rule.name:
                continue
            return rule
        return None

    def _reload_if_needed(self) -> None:
        now = time.time()
        if now - self._last_check < self._reload_seconds:
            return None
        self._last_check = now
        if not self._path:
            return None
        if not self._path.exists():
            return None
        mtime = self._path.stat().st_mtime
        if self._last_mtime is not None and mtime <= self._last_mtime:
            return None
        try:
            ruleset = self._load_ruleset(self._path)
        except Exception as exc:
            log_event("intent_rules_load_error", error=str(exc))
            return None
        compiled = self._compile_rules(ruleset)
        self._rules = compiled
        self._last_mtime = mtime
        log_event(
            "intent_rules_loaded",
            path=str(self._path),
            count=len(compiled),
            version=ruleset.version,
        )

    @staticmethod
    def _load_ruleset(path: Path) -> IntentRuleSet:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return IntentRuleSet.model_validate(payload)

    @staticmethod
    def _compile_rules(ruleset: IntentRuleSet) -> List[_CompiledRule]:
        compiled: List[_CompiledRule] = []
        for rule in sorted(ruleset.rules, key=lambda r: r.priority, reverse=True):
            full_regex_list: List[object] = []
            for pattern in rule.full_regex:
                try:
                    full_regex_list.append(re.compile(pattern))
                except Exception:
                    continue
            compiled.append(
                _CompiledRule(
                    rule=rule,
                    exact=[text.strip() for text in rule.exact if text and text.strip()],
                    full_regex=full_regex_list,
                )
            )
        return compiled
