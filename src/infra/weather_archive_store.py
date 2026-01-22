"""Archive store for historical weather data files."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from .config import get_config


GRID_RESOLUTION_DEGREES = 0.05


def _snap_to_grid(value: float, *, resolution: float) -> float:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    scaled = value / resolution
    if scaled >= 0:
        snapped = math.floor(scaled + 0.5)
    else:
        snapped = math.ceil(scaled - 0.5)
    return snapped * resolution


def _normalize_coord(value: float) -> float:
    snapped = _snap_to_grid(float(value), resolution=GRID_RESOLUTION_DEGREES)
    return round(snapped, 6)


def _normalize_region(region: str) -> str:
    return region.strip()


def _build_archive_token(
    region: str, lat: float, lon: float, year: int
) -> str:
    lat_key = _normalize_coord(lat)
    lon_key = _normalize_coord(lon)
    payload = f"{_normalize_region(region)}|{lat_key:.6f}|{lon_key:.6f}|{year}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_weather_archive_dir() -> Path:
    cfg = get_config()
    if cfg.weather_archive_dir:
        return Path(cfg.weather_archive_dir)
    root = Path(__file__).resolve().parents[2]
    return root / ".cache" / "weather_archive"


def build_weather_archive_path(
    region: str, lat: float, lon: float, year: int
) -> Path:
    archive_dir = get_weather_archive_dir() / str(year)
    token = _build_archive_token(region, lat, lon, year)
    return archive_dir / f"{token}.csv"


class SqliteWeatherArchiveStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS weather_archive ("
                "region TEXT NOT NULL, "
                "lat REAL NOT NULL, "
                "lon REAL NOT NULL, "
                "year INTEGER NOT NULL, "
                "data_path TEXT NOT NULL, "
                "updated_at INTEGER NOT NULL, "
                "PRIMARY KEY (region, lat, lon, year))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_weather_archive_year "
                "ON weather_archive (year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_weather_archive_grid "
                "ON weather_archive (lat, lon, year)"
            )

    def get(
        self, *, region: str, lat: float, lon: float, year: int
    ) -> Optional[str]:
        lat_key = _normalize_coord(lat)
        lon_key = _normalize_coord(lon)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data_path FROM weather_archive "
                "WHERE lat = ? AND lon = ? AND year = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (lat_key, lon_key, int(year)),
            ).fetchone()
        if not row:
            return None
        data_path = row[0]
        return data_path if isinstance(data_path, str) else None

    def set(
        self,
        *,
        region: str,
        lat: float,
        lon: float,
        year: int,
        data_path: str,
    ) -> None:
        normalized = _normalize_region(region)
        lat_key = _normalize_coord(lat)
        lon_key = _normalize_coord(lon)
        updated_at = int(time.time())
        with self._lock, self._connect() as conn:
            updated = conn.execute(
                "UPDATE weather_archive SET "
                "region = ?, data_path = ?, updated_at = ? "
                "WHERE lat = ? AND lon = ? AND year = ?",
                (
                    normalized,
                    data_path,
                    updated_at,
                    lat_key,
                    lon_key,
                    int(year),
                ),
            ).rowcount
            if updated == 0:
                conn.execute(
                    "INSERT INTO weather_archive "
                    "(region, lat, lon, year, data_path, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        normalized,
                        lat_key,
                        lon_key,
                        int(year),
                        data_path,
                        updated_at,
                    ),
                )


def _build_store() -> SqliteWeatherArchiveStore:
    cfg = get_config()
    if cfg.weather_archive_path:
        path = Path(cfg.weather_archive_path)
    else:
        root = Path(__file__).resolve().parents[2]
        path = root / ".cache" / "weather_archive.sqlite3"
    return SqliteWeatherArchiveStore(path=path)


@lru_cache(maxsize=1)
def get_weather_archive_store() -> SqliteWeatherArchiveStore:
    return _build_store()
