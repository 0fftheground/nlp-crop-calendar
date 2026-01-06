"""Persistence for follow-up workflow state with TTL support."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from .config import get_config


class PendingFollowupStore:
    def get(self, session_id: str) -> Optional[dict]:
        raise NotImplementedError

    def set(self, session_id: str, payload: dict) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError


class MemoryPendingFollowupStore(PendingFollowupStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: Dict[str, Tuple[dict, int]] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Optional[dict]:
        now = int(time.time())
        with self._lock:
            item = self._items.get(session_id)
            if not item:
                return None
            payload, expires_at = item
            if expires_at <= now:
                self._items.pop(session_id, None)
                return None
            return dict(payload)

    def set(self, session_id: str, payload: dict) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock:
            self._items[session_id] = (dict(payload), expires_at)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)


class SqlitePendingFollowupStore(PendingFollowupStore):
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
                "CREATE TABLE IF NOT EXISTS pending_followups ("
                "session_id TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_expires "
                "ON pending_followups (expires_at)"
            )

    def get(self, session_id: str) -> Optional[dict]:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM pending_followups WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM pending_followups WHERE session_id = ?",
                    (session_id,),
                )
                return None
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

    def set(self, session_id: str, payload: dict) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        payload_json = json.dumps(payload, ensure_ascii=True, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO pending_followups (session_id, payload, expires_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (session_id, payload_json, expires_at),
            )

    def delete(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_followups WHERE session_id = ?",
                (session_id,),
            )


def build_pending_followup_store() -> PendingFollowupStore:
    cfg = get_config()
    store = (cfg.pending_store or "memory").lower()
    ttl_seconds = cfg.pending_store_ttl_seconds
    if store == "sqlite":
        if cfg.pending_store_path:
            path = Path(cfg.pending_store_path)
        else:
            root = Path(__file__).resolve().parents[2]
            path = root / ".cache" / "pending_followups.sqlite3"
        return SqlitePendingFollowupStore(path=path, ttl_seconds=ttl_seconds)
    return MemoryPendingFollowupStore(ttl_seconds=ttl_seconds)
