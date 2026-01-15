from __future__ import annotations

import difflib
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .config import get_config


_TOKEN_SPLIT_RE = re.compile(r"[，,。；;、\s]+")
_VARIETY_DB_TABLE = "variety_approvals"
_VARIETY_QUERY_STOPWORDS = {
    "水稻",
    "小麦",
    "玉米",
    "大豆",
    "油菜",
    "棉花",
    "花生",
    "品种",
    "品系",
    "抗性",
    "特性",
    "生育期",
    "熟期",
    "审定",
    "信息",
    "查询",
    "查",
    "帮我",
    "请",
    "一下",
    "是什么",
    "多长",
    "多少",
    "的",
    "和",
    "与",
    "及",
}
_VARIETY_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")


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
    query_text = str(query).strip()
    tokens = extract_variety_tokens(query_text)
    if not tokens:
        tokens = [t for t in _TOKEN_SPLIT_RE.split(query_text) if t]
    names = load_variety_names()
    matches = [name for name in names if name and name in query_text]
    fuzzy_used = False
    if not matches and tokens:
        token_set = set(tokens)
        matches = [name for name in names if name in token_set]
    if not matches:
        matches = _get_fuzzy_matches(
            query_text, tokens, names, limit=limit, threshold=threshold
        )
        fuzzy_used = True
    if not fuzzy_used:
        matches.sort(key=len, reverse=True)
    return matches[:limit]


def extract_variety_tokens(prompt: str) -> List[str]:
    if not prompt:
        return []
    text = str(prompt)
    for word in _VARIETY_QUERY_STOPWORDS:
        text = text.replace(word, " ")
    candidates = _VARIETY_TOKEN_RE.findall(text)
    tokens: List[str] = []
    seen = set()
    for token in candidates:
        token = token.strip()
        if not token or token in _VARIETY_QUERY_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) > 20:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    tokens.sort(key=len, reverse=True)
    return tokens


def _get_fuzzy_matches(
    query: str,
    tokens: List[str],
    names: List[str],
    *,
    limit: int,
    threshold: float,
) -> List[str]:
    candidates = []
    if tokens:
        for name in names:
            if not name:
                continue
            best = 0.0
            for token in tokens:
                score = _fuzzy_score(name, token)
                if score > best:
                    best = score
                if best >= 1.0:
                    break
            if best >= threshold:
                candidates.append((best, name))
    else:
        for name in names:
            if not name:
                continue
            score = _fuzzy_score(name, query)
            if score >= threshold:
                candidates.append((score, name))
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [name for _, name in candidates[:limit]]


def _fuzzy_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.95
    return difflib.SequenceMatcher(None, left, right).ratio()


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
