from __future__ import annotations

from ...schemas.models import ToolInvocation
from .registry import auto_register_tool


@auto_register_tool(
    "memory_clear",
    description="清除历史经验记录（user_id 优先，回退到 session_id）。",
)
def memory_clear(prompt: str) -> ToolInvocation:
    """返回清空记忆的工具响应；实际清理逻辑由上层会话机制接管。"""
    _ = prompt
    return ToolInvocation(
        name="memory_clear",
        message="已清除历史经验记录。",
        data={},
    )
