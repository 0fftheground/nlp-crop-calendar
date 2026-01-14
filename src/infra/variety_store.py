from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .config import get_config


_TOKEN_SPLIT_RE = re.compile(r"[，,。；;、\s]+")
_VARIETY_DB_TABLE = "variety_approvals"


def _default_store_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "varieties.json"


def _get_variety_db_path() -> Optional[Path]:
    cfg = get_config()
    if cfg.variety_db_path:
        return Path(cfg.variety_db_path)
    default_path = Path(__file__).resolve().parents[2] / "resources" / "rice_variety_approvals.sqlite3"
    return default_path if default_path.exists() else None


@lru_cache(maxsize=1)
def load_variety_names(path: Path | None = None) -> List[str]:
    if path is None:
        path = _get_variety_db_path()
    if path and path.suffix == ".sqlite3" and path.exists():
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT variety_name FROM {_VARIETY_DB_TABLE}"
            ).fetchall()
        return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
    store_path = path or _default_store_path()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    names = []
    for item in payload.get("varieties", []):
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def retrieve_variety_candidates(
    query: str,
    *,
    limit: int = 5,
    threshold: float = 0.6,
    semantic: bool = True,
) -> List[str]:
    if not query:
        return []
    tokens = [t for t in _TOKEN_SPLIT_RE.split(query) if t]
    names = load_variety_names()
    matches = [name for name in names if name and name in query]
    if not matches and tokens:
        token_set = set(tokens)
        matches = [name for name in names if name in token_set]
    matches.sort(key=len, reverse=True)
    return matches[:limit]


def build_variety_hint(
    query: str,
    *,
    limit: int = 5,
    threshold: float = 0.6,
) -> str:
    candidates = retrieve_variety_candidates(
        query, limit=limit, threshold=threshold, semantic=True
    )
    if not candidates:
        return ""
    joined = "、".join(candidates)
    return f"可参考的品种候选：{joined}。仅在与用户描述匹配时填写。"
