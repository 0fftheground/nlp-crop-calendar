from __future__ import annotations

import json
import math
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from langchain_openai import OpenAIEmbeddings

from .config import get_config


_TOKEN_SPLIT_RE = re.compile(r"[，,。；;、\s]+")
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_QDRANT_COLLECTION = "varieties"


def _default_store_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "varieties.json"


@lru_cache(maxsize=1)
def load_variety_names(path: Path | None = None) -> List[str]:
    store_path = path or _default_store_path()
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    names = []
    for item in payload.get("varieties", []):
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


@lru_cache(maxsize=1)
def _get_embedding_client() -> OpenAIEmbeddings:
    cfg = get_config()
    api_key = cfg.extractor_api_key or cfg.openai_api_key
    if not api_key:
        raise ValueError("缺少 OpenAI API Key，无法进行语义检索。")
    base_url = cfg.extractor_api_base or cfg.openai_api_base
    model = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _qdrant_settings() -> Tuple[Optional[str], Optional[str]]:
    url = os.getenv("QDRANT_URL")
    raw_map = os.getenv("QDRANT_COLLECTIONS", "")
    collection = None
    if raw_map:
        try:
            mapping = json.loads(raw_map)
            if isinstance(mapping, dict):
                collection = mapping.get("variety")
        except json.JSONDecodeError:
            collection = None
    if not url and collection:
        url = _DEFAULT_QDRANT_URL
    if not collection and url:
        collection = _DEFAULT_QDRANT_COLLECTION
    if not url or not collection:
        return None, None
    return url.rstrip("/"), collection


@lru_cache(maxsize=1)
def _load_variety_embeddings() -> Tuple[List[str], List[List[float]]]:
    names = load_variety_names()
    embeddings = _get_embedding_client().embed_documents(names)
    return names, embeddings


def _semantic_candidates(
    query: str,
    *,
    limit: int,
    threshold: float,
) -> List[str]:
    names, embeddings = _load_variety_embeddings()
    query_vector = _get_embedding_client().embed_query(query)
    scored: List[Tuple[float, str]] = []
    for name, vector in zip(names, embeddings):
        score = _cosine_similarity(query_vector, vector)
        if score >= threshold:
            scored.append((score, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


def _qdrant_candidates(
    query: str,
    *,
    limit: int,
    threshold: float,
) -> List[str]:
    url, collection = _qdrant_settings()
    if not url or not collection:
        return []
    query_vector = _get_embedding_client().embed_query(query)
    payload = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
    }
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        response = client.post(
            f"{url}/collections/{collection}/points/search",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    results = data.get("result", [])
    candidates: List[str] = []
    for item in results:
        score = float(item.get("score", 0.0))
        if score < threshold:
            continue
        payload = item.get("payload") or {}
        name = str(payload.get("name", "")).strip()
        if name:
            candidates.append(name)
    return candidates[:limit]


def retrieve_variety_candidates(
    query: str,
    *,
    limit: int = 5,
    threshold: float = 0.6,
    semantic: bool = True,
) -> List[str]:
    if not query:
        return []
    if semantic:
        try:
            candidates = _qdrant_candidates(
                query, limit=limit, threshold=threshold
            )
            if candidates:
                return candidates
            candidates = _semantic_candidates(
                query, limit=limit, threshold=threshold
            )
            if candidates:
                return candidates
        except Exception:
            pass
    tokens = [t for t in _TOKEN_SPLIT_RE.split(query) if t]
    names = load_variety_names()
    scored: List[Tuple[float, str]] = []
    for name in names:
        if name in query:
            scored.append((1.0, name))
            continue
        best = 0.0
        for token in tokens:
            if len(token) < 2:
                continue
            best = max(best, SequenceMatcher(None, token, name).ratio())
        if best >= threshold:
            scored.append((best, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


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
