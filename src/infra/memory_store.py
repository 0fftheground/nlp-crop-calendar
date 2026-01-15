"""Persistence for per-identity planting memory with TTL support."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

from ..schemas import PlantingDetails
from .config import get_config


@dataclass(frozen=True)
class SessionMemory:
    planting: PlantingDetails
    updated_at: int


class SessionMemoryStore:
    def get(self, session_id: str) -> Optional[SessionMemory]:
        raise NotImplementedError

    def set(self, session_id: str, planting: PlantingDetails) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError


def _coerce_session_memory(payload: dict) -> Optional[SessionMemory]:
    if not isinstance(payload, dict):
        return None
    planting_payload = payload.get("planting")
    if not isinstance(planting_payload, dict):
        return None
    try:
        planting = PlantingDetails.model_validate(planting_payload)
    except Exception:
        return None
    updated_at = payload.get("updated_at")
    try:
        updated_at = int(updated_at)
    except (TypeError, ValueError):
        updated_at = int(time.time())
    return SessionMemory(planting=planting, updated_at=updated_at)


class InMemorySessionMemoryStore(SessionMemoryStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: dict[str, Tuple[dict, int]] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Optional[SessionMemory]:
        now = int(time.time())
        with self._lock:
            item = self._items.get(session_id)
            if not item:
                return None
            payload, expires_at = item
            if expires_at <= now:
                self._items.pop(session_id, None)
                return None
        return _coerce_session_memory(payload)

    def set(self, session_id: str, planting: PlantingDetails) -> None:
        payload = {
            "planting": planting.model_dump(mode="json"),
            "updated_at": int(time.time()),
        }
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock:
            self._items[session_id] = (payload, expires_at)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)


class SqliteSessionMemoryStore(SessionMemoryStore):
    def __init__(self, path: Path, ttl_seconds: int) -> None:
        self._path = path
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS session_memory ("
                "session_id TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_memory_expires "
                "ON session_memory (expires_at)"
            )

    def get(self, session_id: str) -> Optional[SessionMemory]:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM session_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM session_memory WHERE session_id = ?",
                    (session_id,),
                )
                return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return _coerce_session_memory(payload)

    def set(self, session_id: str, planting: PlantingDetails) -> None:
        payload = {
            "planting": planting.model_dump(mode="json"),
            "updated_at": int(time.time()),
        }
        payload_json = json.dumps(payload, ensure_ascii=True, default=str)
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO session_memory (session_id, payload, expires_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (session_id, payload_json, expires_at),
            )

    def delete(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM session_memory WHERE session_id = ?",
                (session_id,),
            )


def build_memory_store() -> SessionMemoryStore:
    cfg = get_config()
    store = (cfg.memory_store or "sqlite").lower()
    ttl_seconds = int(cfg.memory_store_ttl_days) * 86400
    if store == "sqlite":
        if cfg.memory_store_path:
            path = Path(cfg.memory_store_path)
        else:
            root = Path(__file__).resolve().parents[2]
            path = root / ".cache" / "session_memory.sqlite3"
        return SqliteSessionMemoryStore(path=path, ttl_seconds=ttl_seconds)
    return InMemorySessionMemoryStore(ttl_seconds=ttl_seconds)
