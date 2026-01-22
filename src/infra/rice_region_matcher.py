from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional


def _normalize_text(value: Optional[str]) -> str:
    return str(value or "").strip().replace(" ", "")


def _load_region_map(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _get_region_map() -> Dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "resources" / "rice_region_map.json"
    if not path.exists():
        return {}
    try:
        return _load_region_map(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_tokens(values: object) -> Iterable[str]:
    if isinstance(values, list):
        for item in values:
            token = _normalize_text(item)
            if token:
                yield token


def match_gdd_region(region: Optional[str]) -> str:
    text = _normalize_text(region)
    if not text:
        return ""
    region_map = _get_region_map()
    token_map = region_map.get("gdd_region_tokens", {})
    if isinstance(token_map, dict):
        for gdd_region in token_map.keys():
            if _normalize_text(gdd_region) in text:
                return _normalize_text(gdd_region)
        for gdd_region, tokens in token_map.items():
            for token in _iter_tokens(tokens):
                if token in text:
                    return _normalize_text(gdd_region)
    if "长江中下游" in text:
        return "长江中下游"
    return ""
