"""Persist request/response interactions for audit and analytics."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List, Optional

from ..observability.logging_utils import get_trace_id, summarize_text
from ..schemas.models import HandleResponse, UserRequest
from .config import get_config


class InteractionStore:
    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        raise NotImplementedError


class NoopInteractionStore(InteractionStore):
    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        return None


class MemoryInteractionStore(InteractionStore):
    def __init__(self, max_items: int) -> None:
        self._max_items = max(1, int(max_items))
        self._items: List[dict] = []
        self._lock = Lock()

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        session_id = request.session_id or request.user_id or "default"
        created_at = int(time.time())
        trace_id = get_trace_id()
        request_summary = _summarize_request(request)
        response_summary = _summarize_response(response)
        request_summary, response_summary = _attach_raw_payload(
            request_summary,
            response_summary,
            request=request,
            response=response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            created_at=created_at,
        )
        item = {
            "created_at": created_at,
            "trace_id": trace_id,
            "session_id": session_id,
            "prompt": request_summary.get("prompt_summary"),
            "region": request_summary.get("region"),
            "mode": response_summary.get("mode"),
            "latency_ms": latency_ms,
            "request": request_summary,
            "response": response_summary,
        }
        with self._lock:
            self._items.append(item)
            if len(self._items) > self._max_items:
                self._items = self._items[-self._max_items :]


class SqliteInteractionStore(InteractionStore):
    def __init__(self, path: Path, ttl_days: int) -> None:
        self._path = path
        self._ttl_days = max(0, int(ttl_days))
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS interactions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "created_at INTEGER NOT NULL, "
                "trace_id TEXT, "
                "session_id TEXT, "
                "prompt TEXT, "
                "region TEXT, "
                "mode TEXT, "
                "latency_ms INTEGER, "
                "request_json TEXT, "
                "response_json TEXT)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_created "
                "ON interactions (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_session "
                "ON interactions (session_id)"
            )
            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN trace_id TEXT")
            except Exception:
                pass
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interactions_trace "
                    "ON interactions (trace_id)"
                )
            except Exception:
                pass

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        created_at = int(time.time())
        trace_id = get_trace_id()
        request_summary = _summarize_request(request)
        response_summary = _summarize_response(response)
        request_summary, response_summary = _attach_raw_payload(
            request_summary,
            response_summary,
            request=request,
            response=response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            created_at=created_at,
        )
        request_json = json.dumps(
            request_summary, ensure_ascii=False, default=str
        )
        response_json = json.dumps(
            response_summary, ensure_ascii=False, default=str
        )
        session_id = request.session_id or request.user_id or "default"
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (created_at, trace_id, session_id, prompt, region, mode, latency_ms, "
                "request_json, response_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    trace_id,
                    session_id,
                    request_summary.get("prompt_summary"),
                    request_summary.get("region"),
                    response_summary.get("mode"),
                    latency_ms,
                    request_json,
                    response_json,
                ),
            )
            if self._ttl_days > 0:
                cutoff = created_at - (self._ttl_days * 86400)
                conn.execute(
                    "DELETE FROM interactions WHERE created_at < ?",
                    (cutoff,),
                )


def _summarize_request(request: UserRequest) -> dict:
    prompt = request.prompt or ""
    return {
        "prompt_summary": summarize_text(prompt),
        "prompt_len": len(prompt),
        "region": request.region,
        "user_id": request.user_id,
        "session_id": request.session_id,
    }


def _summarize_response(response: HandleResponse) -> dict:
    message = ""
    if response.mode == "tool" and response.tool:
        message = response.tool.message or ""
    elif response.plan:
        message = response.plan.message or ""
    summary = {
        "mode": response.mode,
        "message_summary": summarize_text(message),
    }
    if response.tool:
        summary["tool_name"] = response.tool.name
        if isinstance(response.tool.data, dict):
            summary["tool_data_keys"] = list(response.tool.data.keys())
    if response.plan:
        summary["recommendations_count"] = len(response.plan.recommendations or [])
        summary["trace_count"] = len(response.plan.trace or [])
    return summary


def _raw_store_path(trace_id: str, created_at: int) -> Path:
    cfg = get_config()
    if cfg.interaction_raw_dir:
        base = Path(cfg.interaction_raw_dir)
    else:
        base = Path(__file__).resolve().parents[2] / ".cache" / "interaction_raw"
    filename = f"{trace_id}_{created_at}.json"
    return base / filename


def _build_raw_payload(
    request: UserRequest,
    response: HandleResponse,
    latency_ms: int,
    trace_id: str,
    created_at: int,
) -> dict:
    return {
        "trace_id": trace_id,
        "created_at": created_at,
        "latency_ms": latency_ms,
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }


def _attach_raw_payload(
    request_summary: dict,
    response_summary: dict,
    *,
    request: UserRequest,
    response: HandleResponse,
    latency_ms: int,
    trace_id: str,
    created_at: Optional[int] = None,
) -> tuple[dict, dict]:
    cfg = get_config()
    created_at = int(time.time()) if created_at is None else int(created_at)
    raw_payload = _build_raw_payload(
        request, response, latency_ms, trace_id, created_at
    )
    raw_text = json.dumps(raw_payload, ensure_ascii=False, default=str)
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_size = len(raw_text)
    request_summary["raw_hash"] = raw_hash
    request_summary["raw_size"] = raw_size
    response_summary["raw_hash"] = raw_hash
    response_summary["raw_size"] = raw_size
    if raw_size <= max(0, int(cfg.interaction_raw_max_chars)):
        request_summary["raw"] = raw_payload
        response_summary["raw"] = raw_payload
        return request_summary, response_summary
    raw_path = _raw_store_path(trace_id, created_at)
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text, encoding="utf-8")
        request_summary["raw_ref"] = str(raw_path)
        response_summary["raw_ref"] = str(raw_path)
    except Exception:
        request_summary["raw_ref"] = None
        response_summary["raw_ref"] = None
    return request_summary, response_summary


def build_interaction_store() -> InteractionStore:
    cfg = get_config()
    store = (cfg.interaction_store or "disabled").lower()
    if store in {"off", "disabled", "none"}:
        return NoopInteractionStore()
    if store == "sqlite":
        if cfg.interaction_store_path:
            path = Path(cfg.interaction_store_path)
        else:
            root = Path(__file__).resolve().parents[2]
            path = root / ".cache" / "interactions.sqlite3"
        return SqliteInteractionStore(path=path, ttl_days=cfg.interaction_store_ttl_days)
    return MemoryInteractionStore(max_items=cfg.interaction_store_max_items)


@lru_cache(maxsize=1)
def get_interaction_store() -> InteractionStore:
    return build_interaction_store()
