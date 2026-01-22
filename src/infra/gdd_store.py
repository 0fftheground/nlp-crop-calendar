from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .config import get_config

GDD_TABLE_NAME = "gdd_stages"


def get_gdd_db_path() -> Path:
    cfg = get_config()
    if cfg.growth_stage_db_path:
        return Path(cfg.growth_stage_db_path)
    return Path(__file__).resolve().parents[2] / "resources" / "gdd.sqlite3"


def _fetch_gdd_records() -> List[Dict[str, object]]:
    path = get_gdd_db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"GDD SQLite 不存在: {path}. 请先运行 scripts/import_gdd_to_sqlite.py 导入。"
        )
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM {GDD_TABLE_NAME}"
            ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"GDD SQLite 读取失败: {exc}") from exc
    return [dict(row) for row in rows]


@lru_cache(maxsize=1)
def get_gdd_records() -> List[Dict[str, object]]:
    return _fetch_gdd_records()


def clear_gdd_cache() -> None:
    get_gdd_records.cache_clear()
