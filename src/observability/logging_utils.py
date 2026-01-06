from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextvars import ContextVar
from typing import Any, Dict, Optional


_TRACE_ID_CTX: ContextVar[str] = ContextVar("trace_id", default="unknown")
_LOGGER = logging.getLogger("observability")
_INITIALIZED = False


def init_logging(*, log_path: Optional[str] = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    handlers = []
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        )
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    _LOGGER.setLevel(logging.INFO)
    _INITIALIZED = True


def set_trace_id(trace_id: str):
    return _TRACE_ID_CTX.set(trace_id)


def reset_trace_id(token) -> None:
    _TRACE_ID_CTX.reset(token)


def get_trace_id() -> str:
    value = _TRACE_ID_CTX.get()
    return value or "unknown"


def summarize_text(text: str, limit: int = 400) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _build_payload(event: str, fields: Dict[str, Any]) -> str:
    payload = {"event": event, "trace_id": get_trace_id(), **fields}
    return json.dumps(payload, ensure_ascii=True, default=str)


def log_event(event: str, **fields: Any) -> None:
    _LOGGER.info(_build_payload(event, fields))
