"""Cache for weather series data with optional persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple
from uuid import uuid4

from ..schemas import WeatherQueryInput, WeatherSeries
from .config import get_config


class WeatherCacheStore:
    def get(self, cache_key: str) -> Optional[WeatherSeries]:
        raise NotImplementedError

    def set(self, cache_key: str, series: WeatherSeries) -> None:
        raise NotImplementedError


class MemoryWeatherCacheStore(WeatherCacheStore):
    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self._max_items = max(1, int(max_items))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: "OrderedDict[str, Tuple[WeatherSeries, int]]" = OrderedDict()
        self._lock = Lock()

    def get(self, cache_key: str) -> Optional[WeatherSeries]:
        now = int(time.time())
        with self._lock:
            item = self._items.get(cache_key)
            if not item:
                return None
            series, expires_at = item
            if expires_at <= now:
                self._items.pop(cache_key, None)
                return None
            self._items.move_to_end(cache_key)
            return series

    def set(self, cache_key: str, series: WeatherSeries) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock:
            self._items[cache_key] = (series, expires_at)
            self._items.move_to_end(cache_key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)


class SqliteWeatherCacheStore(WeatherCacheStore):
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
                "CREATE TABLE IF NOT EXISTS weather_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_weather_expires "
                "ON weather_cache (expires_at)"
            )

    def get(self, cache_key: str) -> Optional[WeatherSeries]:
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM weather_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if expires_at <= now:
                conn.execute(
                    "DELETE FROM weather_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                return None
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            try:
                return WeatherSeries.model_validate(payload)
            except Exception:
                return None

    def set(self, cache_key: str, series: WeatherSeries) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        payload_json = json.dumps(series.model_dump(mode="json"), ensure_ascii=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO weather_cache (cache_key, payload, expires_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (cache_key, payload_json, expires_at),
            )


def _build_store() -> WeatherCacheStore:
    cfg = get_config()
    store = (cfg.weather_cache_store or "memory").lower()
    ttl_seconds = cfg.weather_cache_ttl_seconds
    if store == "sqlite":
        if cfg.weather_cache_path:
            path = Path(cfg.weather_cache_path)
        else:
            root = Path(__file__).resolve().parents[2]
            path = root / ".cache" / "weather_cache.sqlite3"
        return SqliteWeatherCacheStore(path=path, ttl_seconds=ttl_seconds)
    return MemoryWeatherCacheStore(
        max_items=cfg.weather_cache_max_items,
        ttl_seconds=ttl_seconds,
    )


_STORE = _build_store()


def make_weather_cache_key(query: WeatherQueryInput) -> str:
    payload = query.model_dump(mode="json")
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_weather_series(
    series: WeatherSeries, *, cache_key: Optional[str] = None
) -> str:
    """Store a weather series and return a cache reference key."""
    cache_key = cache_key or uuid4().hex
    _STORE.set(cache_key, series)
    return cache_key


def get_weather_series(cache_key: Optional[str]) -> Optional[WeatherSeries]:
    """Retrieve a cached weather series by key."""
    if not cache_key:
        return None
    return _STORE.get(cache_key)


def get_weather_series_by_query(
    query: WeatherQueryInput,
) -> Optional[WeatherSeries]:
    cache_key = make_weather_cache_key(query)
    return get_weather_series(cache_key)


def store_weather_series_by_query(
    query: WeatherQueryInput, series: WeatherSeries
) -> str:
    cache_key = make_weather_cache_key(query)
    _STORE.set(cache_key, series)
    return cache_key
