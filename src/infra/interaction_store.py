"""Persist request/response interactions for audit and analytics."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from ..observability.logging_utils import get_trace_id, summarize_text
from ..observability.interaction_context import get_interaction_context
from ..schemas.models import HandleResponse, UserRequest
from .config import get_config
from .postgres import connect_postgres


class InteractionStore:
    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        raise NotImplementedError

    def get_latest_session_lineage(self, session_id: str) -> Optional[dict]:
        return None


def _row_value(row: object, key: object, default: object = None) -> object:
    if row is None:
        return default
    if isinstance(key, str):
        if isinstance(row, dict):
            return row.get(key, default)
        try:
            return row[key]
        except Exception:
            return default
    try:
        return row[key]
    except Exception:
        return default


class NoopInteractionStore(InteractionStore):
    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        return None


class MemoryInteractionStore(InteractionStore):
    def __init__(self, max_items: int) -> None:
        self._max_items = max(1, int(max_items))
        self._items: List[dict] = []
        self._lock = Lock()
        self._next_id = 1

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        session_id = request.session_id or request.user_id or "default"
        created_at = int(time.time())
        trace_id = get_trace_id()
        request_summary = _summarize_request(request)
        response_summary = _summarize_response(response)
        request_summary, response_summary = _attach_raw_payload(
            request_summary,
            response_summary,
            request=request,
            response=response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            created_at=created_at,
        )
        item = {
            "id": self._next_id,
            "created_at": created_at,
            "trace_id": trace_id,
            "session_id": session_id,
            **_current_interaction_fields(),
            "prompt": request_summary.get("prompt_summary"),
            "region": request_summary.get("region"),
            "mode": response_summary.get("mode"),
            "latency_ms": latency_ms,
            "request": request_summary,
            "response": response_summary,
        }
        with self._lock:
            self._items.append(item)
            self._next_id += 1
            if len(self._items) > self._max_items:
                self._items = self._items[-self._max_items :]

    def get_latest_session_lineage(self, session_id: str) -> Optional[dict]:
        with self._lock:
            for item in reversed(self._items):
                if str(item.get("session_id") or "") != session_id:
                    continue
                return _extract_lineage_payload(
                    int(item.get("id") or 0),
                    item.get("request") or {},
                    item.get("response") or {},
                    top_level=item,
                )
        return None


class SqliteInteractionStore(InteractionStore):
    def __init__(self, path: Path, ttl_days: int) -> None:
        self._path = path
        self._ttl_days = max(0, int(ttl_days))
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS interactions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "created_at INTEGER NOT NULL, "
                "trace_id TEXT, "
                "session_id TEXT, "
                "request_id TEXT, "
                "thread_id TEXT, "
                "parent_interaction_id INTEGER, "
                "continuity_type TEXT, "
                "continuity_source TEXT, "
                "dialogue_act TEXT, "
                "task_type TEXT, "
                "prompt TEXT, "
                "region TEXT, "
                "mode TEXT, "
                "latency_ms INTEGER, "
                "request_json TEXT, "
                "response_json TEXT)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_created "
                "ON interactions (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_session "
                "ON interactions (session_id)"
            )
            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN trace_id TEXT")
            except Exception:
                pass
            for statement in (
                "ALTER TABLE interactions ADD COLUMN request_id TEXT",
                "ALTER TABLE interactions ADD COLUMN thread_id TEXT",
                "ALTER TABLE interactions ADD COLUMN parent_interaction_id INTEGER",
                "ALTER TABLE interactions ADD COLUMN continuity_type TEXT",
                "ALTER TABLE interactions ADD COLUMN continuity_source TEXT",
                "ALTER TABLE interactions ADD COLUMN dialogue_act TEXT",
                "ALTER TABLE interactions ADD COLUMN task_type TEXT",
            ):
                try:
                    conn.execute(statement)
                except Exception:
                    pass
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interactions_trace "
                    "ON interactions (trace_id)"
                )
            except Exception:
                pass
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interactions_thread "
                    "ON interactions (thread_id)"
                )
            except Exception:
                pass

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        created_at = int(time.time())
        trace_id = get_trace_id()
        request_summary = _summarize_request(request)
        response_summary = _summarize_response(response)
        request_summary, response_summary = _attach_raw_payload(
            request_summary,
            response_summary,
            request=request,
            response=response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            created_at=created_at,
        )
        request_json = json.dumps(
            request_summary, ensure_ascii=False, default=str
        )
        response_json = json.dumps(
            response_summary, ensure_ascii=False, default=str
        )
        session_id = request.session_id or request.user_id or "default"
        interaction_fields = _current_interaction_fields()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (created_at, trace_id, session_id, request_id, thread_id, "
                "parent_interaction_id, continuity_type, continuity_source, dialogue_act, task_type, "
                "prompt, region, mode, latency_ms, request_json, response_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    trace_id,
                    session_id,
                    interaction_fields.get("request_id"),
                    interaction_fields.get("thread_id"),
                    interaction_fields.get("parent_interaction_id"),
                    interaction_fields.get("continuity_type"),
                    interaction_fields.get("continuity_source"),
                    interaction_fields.get("dialogue_act"),
                    interaction_fields.get("task_type"),
                    request_summary.get("prompt_summary"),
                    request_summary.get("region"),
                    response_summary.get("mode"),
                    latency_ms,
                    request_json,
                    response_json,
                ),
                )
            if self._ttl_days > 0:
                cutoff = created_at - (self._ttl_days * 86400)
                conn.execute(
                    "DELETE FROM interactions WHERE created_at < ?",
                    (cutoff,),
                )

    def get_latest_session_lineage(self, session_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, request_id, thread_id, parent_interaction_id, continuity_type, continuity_source, "
                "dialogue_act, task_type, "
                "request_json, response_json FROM interactions "
                "WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return _extract_lineage_payload(
            int(_row_value(row, 0) or 0),
            _normalize_payload_json(_row_value(row, 8)),
            _normalize_payload_json(_row_value(row, 9)),
            top_level={
                "request_id": _row_value(row, 1),
                "thread_id": _row_value(row, 2),
                "parent_interaction_id": _row_value(row, 3),
                "continuity_type": _row_value(row, 4),
                "continuity_source": _row_value(row, 5),
                "dialogue_act": _row_value(row, 6),
                "task_type": _row_value(row, 7),
            },
        )


class PostgresInteractionStore(InteractionStore):
    def __init__(self, url: str, ttl_days: int) -> None:
        self._url = url
        self._ttl_days = max(0, int(ttl_days))
        self._lock = Lock()
        self._init_db()

    def _connect(self):
        return connect_postgres(self._url)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS interactions ("
                "id BIGSERIAL PRIMARY KEY, "
                "created_at BIGINT NOT NULL, "
                "trace_id TEXT, "
                "session_id TEXT, "
                "request_id TEXT, "
                "thread_id TEXT, "
                "parent_interaction_id BIGINT, "
                "continuity_type TEXT, "
                "continuity_source TEXT, "
                "dialogue_act TEXT, "
                "task_type TEXT, "
                "prompt TEXT, "
                "region TEXT, "
                "mode TEXT, "
                "latency_ms INTEGER, "
                "request_json JSONB, "
                "response_json JSONB)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_created "
                "ON interactions (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_session "
                "ON interactions (session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_trace "
                "ON interactions (trace_id)"
            )
            for statement in (
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS request_id TEXT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS thread_id TEXT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS parent_interaction_id BIGINT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS continuity_type TEXT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS continuity_source TEXT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS dialogue_act TEXT",
                "ALTER TABLE interactions ADD COLUMN IF NOT EXISTS task_type TEXT",
            ):
                conn.execute(statement)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_thread "
                "ON interactions (thread_id)"
            )

    def record(self, request: UserRequest, response: HandleResponse, latency_ms: int) -> None:
        created_at = int(time.time())
        trace_id = get_trace_id()
        request_summary = _summarize_request(request)
        response_summary = _summarize_response(response)
        request_summary, response_summary = _attach_raw_payload(
            request_summary,
            response_summary,
            request=request,
            response=response,
            latency_ms=latency_ms,
            trace_id=trace_id,
            created_at=created_at,
        )
        request_json = json.dumps(
            request_summary, ensure_ascii=False, default=str
        )
        response_json = json.dumps(
            response_summary, ensure_ascii=False, default=str
        )
        session_id = request.session_id or request.user_id or "default"
        interaction_fields = _current_interaction_fields()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (created_at, trace_id, session_id, request_id, thread_id, "
                "parent_interaction_id, continuity_type, continuity_source, dialogue_act, task_type, "
                "prompt, region, mode, latency_ms, request_json, response_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                (
                    created_at,
                    trace_id,
                    session_id,
                    interaction_fields.get("request_id"),
                    interaction_fields.get("thread_id"),
                    interaction_fields.get("parent_interaction_id"),
                    interaction_fields.get("continuity_type"),
                    interaction_fields.get("continuity_source"),
                    interaction_fields.get("dialogue_act"),
                    interaction_fields.get("task_type"),
                    request_summary.get("prompt_summary"),
                    request_summary.get("region"),
                    response_summary.get("mode"),
                    latency_ms,
                    request_json,
                    response_json,
                ),
            )
            if self._ttl_days > 0:
                cutoff = created_at - (self._ttl_days * 86400)
                conn.execute(
                    "DELETE FROM interactions WHERE created_at < %s",
                    (cutoff,),
                )

    def get_latest_session_lineage(self, session_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, request_id, thread_id, parent_interaction_id, continuity_type, continuity_source, "
                    "dialogue_act, task_type, "
                    "request_json, response_json FROM interactions "
                    "WHERE session_id = %s ORDER BY created_at DESC, id DESC LIMIT 1",
                    (session_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return _extract_lineage_payload(
            int(_row_value(row, "id", _row_value(row, 0)) or 0),
            _normalize_payload_json(
                _row_value(row, "request_json", _row_value(row, 8))
            ),
            _normalize_payload_json(
                _row_value(row, "response_json", _row_value(row, 9))
            ),
            top_level={
                "request_id": _row_value(row, "request_id", _row_value(row, 1)),
                "thread_id": _row_value(row, "thread_id", _row_value(row, 2)),
                "parent_interaction_id": _row_value(
                    row, "parent_interaction_id", _row_value(row, 3)
                ),
                "continuity_type": _row_value(
                    row, "continuity_type", _row_value(row, 4)
                ),
                "continuity_source": _row_value(
                    row, "continuity_source", _row_value(row, 5)
                ),
                "dialogue_act": _row_value(
                    row, "dialogue_act", _row_value(row, 6)
                ),
                "task_type": _row_value(
                    row, "task_type", _row_value(row, 7)
                ),
            },
        )

def _summarize_request(request: UserRequest) -> dict:
    prompt = request.prompt or ""
    interaction = get_interaction_context()
    return {
        "prompt_summary": summarize_text(prompt),
        "prompt_len": len(prompt),
        "region": request.region,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "request_id": interaction.get("request_id"),
        "thread_id": interaction.get("thread_id"),
        "parent_interaction_id": interaction.get("parent_interaction_id"),
        "continuity_type": interaction.get("continuity_type"),
        "continuity_source": interaction.get("continuity_source"),
        "dialogue_act": interaction.get("dialogue_act"),
        "task_type": interaction.get("task_type"),
    }


def _summarize_response(response: HandleResponse) -> dict:
    message = ""
    interaction = get_interaction_context()
    if response.mode == "tool" and response.tool:
        message = response.tool.message or ""
    elif response.plan:
        message = response.plan.message or ""
    summary = {
        "mode": response.mode,
        "message_summary": summarize_text(message),
        "request_id": interaction.get("request_id"),
        "thread_id": interaction.get("thread_id"),
        "parent_interaction_id": interaction.get("parent_interaction_id"),
        "continuity_type": interaction.get("continuity_type"),
        "continuity_source": interaction.get("continuity_source"),
        "dialogue_act": interaction.get("dialogue_act"),
        "task_type": interaction.get("task_type"),
    }
    if response.tool:
        summary["tool_name"] = response.tool.name
        if isinstance(response.tool.data, dict):
            summary["tool_data_keys"] = list(response.tool.data.keys())
    if response.plan:
        summary["recommendations_count"] = len(response.plan.recommendations or [])
        summary["trace_count"] = len(response.plan.trace or [])
        workflow_name = str((response.plan.data or {}).get("workflow_name") or "").strip()
        if workflow_name:
            summary["workflow_name"] = workflow_name
    return summary


def _raw_store_path(trace_id: str, created_at: int) -> Path:
    cfg = get_config()
    if cfg.interaction_raw_dir:
        base = Path(cfg.interaction_raw_dir)
    else:
        base = Path(__file__).resolve().parents[2] / ".cache" / "interaction_raw"
    filename = f"{trace_id}_{created_at}.json"
    return base / filename


def _build_raw_payload(
    request: UserRequest,
    response: HandleResponse,
    latency_ms: int,
    trace_id: str,
    created_at: int,
) -> dict:
    interaction = get_interaction_context()
    return {
        "trace_id": trace_id,
        "request_id": interaction.get("request_id"),
        "thread_id": interaction.get("thread_id"),
        "parent_interaction_id": interaction.get("parent_interaction_id"),
        "continuity_type": interaction.get("continuity_type"),
        "continuity_source": interaction.get("continuity_source"),
        "dialogue_act": interaction.get("dialogue_act"),
        "task_type": interaction.get("task_type"),
        "created_at": created_at,
        "latency_ms": latency_ms,
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }


def _attach_raw_payload(
    request_summary: dict,
    response_summary: dict,
    *,
    request: UserRequest,
    response: HandleResponse,
    latency_ms: int,
    trace_id: str,
    created_at: Optional[int] = None,
) -> tuple[dict, dict]:
    cfg = get_config()
    created_at = int(time.time()) if created_at is None else int(created_at)
    raw_payload = _build_raw_payload(
        request, response, latency_ms, trace_id, created_at
    )
    raw_text = json.dumps(raw_payload, ensure_ascii=False, default=str)
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_size = len(raw_text)
    request_summary["raw_hash"] = raw_hash
    request_summary["raw_size"] = raw_size
    response_summary["raw_hash"] = raw_hash
    response_summary["raw_size"] = raw_size
    if raw_size <= max(0, int(cfg.interaction_raw_max_chars)):
        request_summary["raw"] = raw_payload
        response_summary["raw"] = raw_payload
        return request_summary, response_summary
    raw_path = _raw_store_path(trace_id, created_at)
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text, encoding="utf-8")
        request_summary["raw_ref"] = str(raw_path)
        response_summary["raw_ref"] = str(raw_path)
    except Exception:
        request_summary["raw_ref"] = None
        response_summary["raw_ref"] = None
    return request_summary, response_summary


def build_interaction_store() -> InteractionStore:
    cfg = get_config()
    store = (cfg.interaction_store or "disabled").lower()
    if store in {"off", "disabled", "none"}:
        return NoopInteractionStore()
    if store == "postgres":
        if not cfg.cache_db_url:
            raise RuntimeError("缺少 CACHE_DB_URL，无法写入外部审计存储。")
        return PostgresInteractionStore(
            url=cfg.cache_db_url, ttl_days=cfg.interaction_store_ttl_days
        )
    if store == "sqlite":
        if cfg.interaction_store_path:
            path = Path(cfg.interaction_store_path)
        else:
            root = Path(__file__).resolve().parents[2]
            path = root / ".cache" / "interactions.sqlite3"
        return SqliteInteractionStore(path=path, ttl_days=cfg.interaction_store_ttl_days)
    return MemoryInteractionStore(max_items=cfg.interaction_store_max_items)


@lru_cache(maxsize=1)
def get_interaction_store() -> InteractionStore:
    return build_interaction_store()


def _current_interaction_fields() -> Dict[str, Any]:
    interaction = get_interaction_context()
    return {
        "request_id": interaction.get("request_id"),
        "thread_id": interaction.get("thread_id"),
        "parent_interaction_id": interaction.get("parent_interaction_id"),
        "continuity_type": interaction.get("continuity_type"),
        "continuity_source": interaction.get("continuity_source"),
        "dialogue_act": interaction.get("dialogue_act"),
        "task_type": interaction.get("task_type"),
    }


def _normalize_payload_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_lineage_payload(
    interaction_id: int,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    *,
    top_level: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    top_level = dict(top_level or {})
    request_id = str(
        top_level.get("request_id") or request_payload.get("request_id") or ""
    ).strip()
    thread_id = str(
        top_level.get("thread_id")
        or request_payload.get("thread_id")
        or response_payload.get("thread_id")
        or ""
    ).strip()
    return {
        "interaction_id": interaction_id,
        "request_id": request_id or None,
        "thread_id": thread_id or None,
        "parent_interaction_id": top_level.get("parent_interaction_id"),
        "continuity_type": str(top_level.get("continuity_type") or "").strip() or None,
        "continuity_source": str(top_level.get("continuity_source") or "").strip() or None,
        "dialogue_act": str(top_level.get("dialogue_act") or "").strip() or None,
        "task_type": str(top_level.get("task_type") or "").strip() or None,
    }
