from __future__ import annotations

import re
from typing import Sequence


_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


def quote_identifier(value: str) -> str:
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in value.split("."))


def _ensure_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Postgres 连接需要安装 psycopg[binary]。"
            "请在 requirements.txt 中加入 psycopg[binary]。"
        ) from exc
    return psycopg, dict_row


def connect_postgres(url: str):
    psycopg, dict_row = _ensure_psycopg()
    return psycopg.connect(url, row_factory=dict_row)


def fetch_all(
    url: str, sql: str, params: Sequence[object] | None = None
) -> list[dict[str, object]]:
    with connect_postgres(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def execute(url: str, sql: str, params: Sequence[object] | None = None) -> None:
    with connect_postgres(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
        conn.commit()
