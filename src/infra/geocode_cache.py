"""Local cache for geocoding lookups."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from .config import get_config


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    return re.sub(r"\s+", "", text)


def make_geocode_cache_key(address: str, city: Optional[str]) -> str:
    payload = f"{_normalize_text(address)}|{_normalize_text(city)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqliteGeocodeCacheStore:
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
                "CREATE TABLE IF NOT EXISTS geocode_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_geocode_expires "
                "ON geocode_cache (expires_at)"
            )

    def get(self, cache_key: str) -> Optional[dict]:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM geocode_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM geocode_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

    def set(self, cache_key: str, payload: dict) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO geocode_cache (cache_key, payload, expires_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (cache_key, payload_json, expires_at),
            )


def _build_store() -> SqliteGeocodeCacheStore:
    cfg = get_config()
    ttl_days = max(1, int(cfg.geocode_cache_ttl_days))
    ttl_seconds = ttl_days * 86400
    if cfg.geocode_cache_path:
        path = Path(cfg.geocode_cache_path)
    else:
        root = Path(__file__).resolve().parents[2]
        path = root / ".cache" / "geocode_cache.sqlite3"
    return SqliteGeocodeCacheStore(path=path, ttl_seconds=ttl_seconds)


@lru_cache(maxsize=1)
def get_geocode_cache() -> SqliteGeocodeCacheStore:
    return _build_store()


def get_geocode_cached(address: str, city: Optional[str] = None) -> Optional[dict]:
    cache_key = make_geocode_cache_key(address, city)
    return get_geocode_cache().get(cache_key)


def set_geocode_cached(
    address: str, city: Optional[str], payload: dict
) -> None:
    cache_key = make_geocode_cache_key(address, city)
    get_geocode_cache().set(cache_key, payload)
