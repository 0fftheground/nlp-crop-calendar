from __future__ import annotations

from typing import Dict, List

from .config import get_config
from .db_catalog import TABLE_KEY_GROWTH_STAGE_FORECAST, resolve_db_table
from .postgres import fetch_all, quote_identifier


def _get_gdd_db_url() -> str | None:
    cfg = get_config()
    return cfg.agri_db_url


def _get_gdd_db_table() -> str:
    return resolve_db_table(get_config(), TABLE_KEY_GROWTH_STAGE_FORECAST)


def _require_db_url() -> str:
    url = _get_gdd_db_url()
    if not url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取积温数据。")
    return url


def get_gdd_source() -> str:
    table = _get_gdd_db_table()
    return f"postgres:{table}"


def _fetch_gdd_records() -> List[Dict[str, object]]:
    url = _require_db_url()
    table = _get_gdd_db_table()
    try:
        sql = f"SELECT * FROM {quote_identifier(table)}"
        return fetch_all(url, sql)
    except Exception as exc:
        raise RuntimeError(f"GDD Postgres 读取失败: {exc}") from exc


def get_gdd_records() -> List[Dict[str, object]]:
    return _fetch_gdd_records()


def clear_gdd_cache() -> None:
    return None
