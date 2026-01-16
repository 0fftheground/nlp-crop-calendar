"""Persist planting choices per user/crop/region with TTL support."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

from ..schemas import PlantingDetails
from .config import get_config


@dataclass(frozen=True)
class PlantingChoice:
    planting: PlantingDetails
    updated_at: int


class PlantingChoiceStore:
    def get(
        self, user_id: str, crop: str, region: str
    ) -> Optional[PlantingChoice]:
        raise NotImplementedError

    def set(
        self, user_id: str, crop: str, region: str, planting: PlantingDetails
    ) -> None:
        raise NotImplementedError

    def delete(self, user_id: str, crop: str, region: str) -> None:
        raise NotImplementedError

    def delete_user(self, user_id: str) -> None:
        raise NotImplementedError


def _normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _normalize_pair(
    crop: Optional[str], region: Optional[str]
) -> Optional[Tuple[str, str]]:
    crop_key = _normalize_key(crop)
    region_key = _normalize_key(region)
    if not crop_key or not region_key:
        return None
    return crop_key, region_key


def build_choice_key(crop: Optional[str], region: Optional[str]) -> Optional[str]:
    pair = _normalize_pair(crop, region)
    if not pair:
        return None
    crop_key, region_key = pair
    return f"{crop_key}::{region_key}"


def _coerce_choice(payload: dict) -> Optional[PlantingChoice]:
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
    return PlantingChoice(planting=planting, updated_at=updated_at)


class SqlitePlantingChoiceStore(PlantingChoiceStore):
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
                "CREATE TABLE IF NOT EXISTS planting_choice ("
                "user_id TEXT NOT NULL, "
                "crop_key TEXT NOT NULL, "
                "region_key TEXT NOT NULL, "
                "payload TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, "
                "PRIMARY KEY (user_id, crop_key, region_key))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planting_choice_expires "
                "ON planting_choice (expires_at)"
            )

    def get(
        self, user_id: str, crop: str, region: str
    ) -> Optional[PlantingChoice]:
        pair = _normalize_pair(crop, region)
        if not pair:
            return None
        crop_key, region_key = pair
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM planting_choice "
                "WHERE user_id = ? AND crop_key = ? AND region_key = ?",
                (user_id, crop_key, region_key),
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
        self, user_id: str, crop: str, region: str, planting: PlantingDetails
    ) -> None:
        pair = _normalize_pair(crop, region)
        if not pair:
            return None
        crop_key, region_key = pair
        payload = {
            "planting": planting.model_dump(mode="json"),
            "updated_at": int(time.time()),
        }
        payload_json = json.dumps(payload, ensure_ascii=True, default=str)
        expires_at = int(time.time()) + self._ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO planting_choice "
                "(user_id, crop_key, region_key, payload, expires_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, crop_key, region_key) DO UPDATE SET "
                "payload = excluded.payload, "
                "expires_at = excluded.expires_at",
                (user_id, crop_key, region_key, payload_json, expires_at),
            )

    def delete(self, user_id: str, crop: str, region: str) -> None:
        pair = _normalize_pair(crop, region)
        if not pair:
            return None
        crop_key, region_key = pair
        expires_at = int(time.time()) - 1
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE planting_choice SET expires_at = ? "
                "WHERE user_id = ? AND crop_key = ? AND region_key = ?",
                (expires_at, user_id, crop_key, region_key),
            )

    def delete_user(self, user_id: str) -> None:
        expires_at = int(time.time()) - 1
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE planting_choice SET expires_at = ? WHERE user_id = ?",
                (expires_at, user_id),
            )


def build_planting_choice_store() -> PlantingChoiceStore:
    cfg = get_config()
    ttl_seconds = int(cfg.memory_store_ttl_days) * 86400
    root = Path(__file__).resolve().parents[2]
    path = root / ".cache" / "planting_choices.sqlite3"
    return SqlitePlantingChoiceStore(path=path, ttl_seconds=ttl_seconds)


@lru_cache(maxsize=1)
def get_planting_choice_store() -> PlantingChoiceStore:
    return build_planting_choice_store()
