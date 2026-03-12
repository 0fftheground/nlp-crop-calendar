"""Tool result cache with TTL support."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from .postgres import connect_postgres


class ToolResultCache:
    def get(self, tool_name: str, provider: str, prompt: str) -> Optional[dict]:
        raise NotImplementedError

    def set(self, tool_name: str, provider: str, prompt: str, payload: dict) -> None:
        raise NotImplementedError


def _make_cache_key(tool_name: str, provider: str, prompt: str) -> str:
    key = f"{tool_name}|{provider}|{prompt}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MemoryToolResultCache(ToolResultCache):
    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self._max_items = max(1, int(max_items))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: "OrderedDict[str, tuple[dict, int]]" = OrderedDict()
        self._lock = Lock()

    def get(self, tool_name: str, provider: str, prompt: str) -> Optional[dict]:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        now = int(time.time())
        with self._lock:
            item = self._items.get(cache_key)
            if not item:
                return None
            payload, expires_at = item
            if expires_at <= now:
                self._items.pop(cache_key, None)
                return None
            self._items.move_to_end(cache_key)
            return dict(payload)

    def set(self, tool_name: str, provider: str, prompt: str, payload: dict) -> None:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock:
            self._items[cache_key] = (dict(payload), expires_at)
            self._items.move_to_end(cache_key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)


class SqliteToolResultCache(ToolResultCache):
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
                "CREATE TABLE IF NOT EXISTS tool_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "tool_name TEXT NOT NULL, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_expires "
                "ON tool_cache (expires_at)"
            )

    def get(self, tool_name: str, provider: str, prompt: str) -> Optional[dict]:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM tool_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM tool_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

    def set(self, tool_name: str, provider: str, prompt: str, payload: dict) -> None:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        expires_at = int(time.time()) + self._ttl_seconds
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_cache (cache_key, tool_name, payload, expires_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (cache_key, tool_name, payload_json, expires_at),
            )


class PostgresToolResultCache(ToolResultCache):
    def __init__(self, url: str, ttl_seconds: int) -> None:
        self._url = url
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._lock = Lock()
        self._init_db()

    def _connect(self):
        return connect_postgres(self._url)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tool_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "tool_name TEXT NOT NULL, "
                "payload JSONB NOT NULL, "
                "expires_at BIGINT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_expires "
                "ON tool_cache (expires_at)"
            )

    def get(self, tool_name: str, provider: str, prompt: str) -> Optional[dict]:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM tool_cache WHERE cache_key = %s",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                payload = row.get("payload")
                expires_at = row.get("expires_at")
            else:
                payload, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM tool_cache WHERE cache_key = %s",
                    (cache_key,),
                )
                return None
            if isinstance(payload, dict):
                return dict(payload)
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
            return None

    def set(self, tool_name: str, provider: str, prompt: str, payload: dict) -> None:
        cache_key = _make_cache_key(tool_name, provider, prompt)
        expires_at = int(time.time()) + self._ttl_seconds
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_cache (cache_key, tool_name, payload, expires_at) "
                "VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (cache_key, tool_name, payload_json, expires_at),
            )


class NoopToolResultCache(ToolResultCache):
    def get(self, tool_name: str, provider: str, prompt: str) -> Optional[dict]:
        return None

    def set(self, tool_name: str, provider: str, prompt: str, payload: dict) -> None:
        return None


def build_tool_result_cache() -> ToolResultCache:
    # Caching disabled: always return a no-op cache.
    return NoopToolResultCache()


@lru_cache(maxsize=1)
def get_tool_result_cache() -> ToolResultCache:
    return build_tool_result_cache()
