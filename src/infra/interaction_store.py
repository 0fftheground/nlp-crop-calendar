"""Persist request/response interactions for audit and analytics."""

from __future__ import annotations

import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List

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
        item = {
            "created_at": int(time.time()),
            "session_id": session_id,
            "prompt": request.prompt,
            "region": request.region,
            "mode": response.mode,
            "latency_ms": latency_ms,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
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

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        created_at = int(time.time())
        request_json = json.dumps(
            request.model_dump(mode="json"), ensure_ascii=False, default=str
        )
        response_json = json.dumps(
            response.model_dump(mode="json"), ensure_ascii=False, default=str
        )
        session_id = request.session_id or request.user_id or "default"
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (created_at, session_id, prompt, region, mode, latency_ms, "
                "request_json, response_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    session_id,
                    request.prompt,
                    request.region,
                    response.mode,
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
