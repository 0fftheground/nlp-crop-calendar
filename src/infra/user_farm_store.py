from __future__ import annotations

from typing import Optional

from .config import get_config
from .postgres import execute, fetch_all, quote_identifier


def _require_db_url() -> str:
    cfg = get_config()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取用户农场映射。")
    return cfg.agri_db_url


def _get_table_name() -> str:
    cfg = get_config()
    return cfg.user_farm_table or "user_farm_map"


def _ensure_table() -> None:
    url = _require_db_url()
    table = quote_identifier(_get_table_name())
    sql = (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        "user_id TEXT PRIMARY KEY, "
        "farm_id TEXT NOT NULL, "
        "created_at TIMESTAMPTZ DEFAULT NOW(), "
        "updated_at TIMESTAMPTZ DEFAULT NOW()"
        ")"
    )
    execute(url, sql)


def get_farm_id_for_user(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    _ensure_table()
    url = _require_db_url()
    table = quote_identifier(_get_table_name())
    rows = fetch_all(
        url,
        f"SELECT farm_id FROM {table} WHERE user_id = %s LIMIT 1",
        (user_id,),
    )
    if not rows:
        return None
    value = rows[0].get("farm_id")
    return str(value) if value is not None else None


def set_farm_id_for_user(user_id: str, farm_id: str) -> None:
    if not user_id or not farm_id:
        return
    _ensure_table()
    url = _require_db_url()
    table = quote_identifier(_get_table_name())
    execute(
        url,
        f"INSERT INTO {table} (user_id, farm_id) VALUES (%s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET "
        "farm_id = EXCLUDED.farm_id, updated_at = NOW()",
        (user_id, farm_id),
    )
