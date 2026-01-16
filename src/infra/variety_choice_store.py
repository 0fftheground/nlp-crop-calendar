"""Persist user variety choices with TTL support."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from .config import get_config


@dataclass(frozen=True)
class VarietyChoice:
    variety: str
    region_choice: Optional[str]
    updated_at: int


class VarietyChoiceStore:
    def get(self, user_id: str, query_key: str) -> Optional[VarietyChoice]:
        raise NotImplementedError

    def set(
        self, user_id: str, query_key: str, variety: str, region_choice: Optional[str]
    ) -> None:
        raise NotImplementedError

    def delete(self, user_id: str, query_key: str) -> None:
        raise NotImplementedError

    def delete_user(self, user_id: str) -> None:
        raise NotImplementedError


def _coerce_choice(payload: dict) -> Optional[VarietyChoice]:
    if not isinstance(payload, dict):
        return None
    variety = payload.get("variety")
    if not isinstance(variety, str) or not variety.strip():
        return None
    region_choice = payload.get("region_choice")
    if region_choice is not None and not isinstance(region_choice, str):
        region_choice = None
    updated_at = payload.get("updated_at")
    try:
        updated_at = int(updated_at)
    except (TypeError, ValueError):
        updated_at = int(time.time())
    return VarietyChoice(
        variety=variety.strip(),
        region_choice=region_choice,
        updated_at=updated_at,
    )


class SqliteVarietyChoiceStore(VarietyChoiceStore):
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
                "CREATE TABLE IF NOT EXISTS variety_choice ("
                "user_id TEXT NOT NULL, "
                "query_key TEXT NOT NULL, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, "
                "PRIMARY KEY (user_id, query_key))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_variety_choice_expires "
                "ON variety_choice (expires_at)"
            )

    def get(self, user_id: str, query_key: str) -> Optional[VarietyChoice]:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM variety_choice "
                "WHERE user_id = ? AND query_key = ?",
                (user_id, query_key),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return _coerce_choice(payload)

    def set(
        self, user_id: str, query_key: str, variety: str, region_choice: Optional[str]
    ) -> None:
        payload = {
            "variety": variety,
            "region_choice": region_choice,
            "updated_at": int(time.time()),
        }
        payload_json = json.dumps(payload, ensure_ascii=True, default=str)
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO variety_choice (user_id, query_key, payload, expires_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, query_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (user_id, query_key, payload_json, expires_at),
            )

    def delete(self, user_id: str, query_key: str) -> None:
        expires_at = int(time.time()) - 1
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE variety_choice SET expires_at = ? "
                "WHERE user_id = ? AND query_key = ?",
                (expires_at, user_id, query_key),
            )

    def delete_user(self, user_id: str) -> None:
        expires_at = int(time.time()) - 1
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE variety_choice SET expires_at = ? WHERE user_id = ?",
                (expires_at, user_id),
            )


def build_variety_choice_store() -> VarietyChoiceStore:
    cfg = get_config()
    ttl_seconds = int(cfg.memory_store_ttl_days) * 86400
    root = Path(__file__).resolve().parents[2]
    path = root / ".cache" / "variety_choices.sqlite3"
    return SqliteVarietyChoiceStore(path=path, ttl_seconds=ttl_seconds)


@lru_cache(maxsize=1)
def get_variety_choice_store() -> VarietyChoiceStore:
    return build_variety_choice_store()
