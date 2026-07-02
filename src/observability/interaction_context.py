from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional


_INTERACTION_CTX: ContextVar[Dict[str, Any]] = ContextVar(
    "interaction_context", default={}
)


def set_interaction_context(payload: Dict[str, Any]):
    return _INTERACTION_CTX.set(dict(payload))


def reset_interaction_context(token) -> None:
    _INTERACTION_CTX.reset(token)


def get_interaction_context() -> Dict[str, Any]:
    return dict(_INTERACTION_CTX.get() or {})


def update_interaction_context(**updates: Any) -> Dict[str, Any]:
    payload = get_interaction_context()
    for key, value in updates.items():
        if value is None and key not in payload:
            continue
        payload[key] = value
    _INTERACTION_CTX.set(payload)
    return dict(payload)


def build_initial_interaction_context(
    *,
    request_id: str,
    session_id: str,
    user_id: Optional[str] = None,
    previous_interaction_id: Optional[int] = None,
    previous_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "session_id": session_id,
        "user_id": user_id,
        "previous_interaction_id": previous_interaction_id,
        "previous_thread_id": previous_thread_id,
        "thread_id": request_id,
        "parent_interaction_id": None,
        "continuity_type": "standalone",
        "continuity_source": "none",
        "dialogue_act": "start_new_task",
        "task_type": "none",
    }
