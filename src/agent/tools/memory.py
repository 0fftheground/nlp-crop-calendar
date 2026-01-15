from __future__ import annotations

from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "memory_clear",
    description="清除种植记忆（user_id 优先，回退到 session_id）。",
)
def memory_clear(prompt: str) -> ToolInvocation:
    _ = prompt
    return ToolInvocation(
        name="memory_clear",
        message="已清除记忆。",
        data={},
    )
