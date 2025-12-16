from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, tool as lc_tool

from .models import ToolInvocation


TOOLS: List[BaseTool] = []
TOOL_INDEX: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """Register a custom LangChain tool for routing."""
    TOOLS.append(tool)
    TOOL_INDEX[tool.name] = tool


def clear_tools() -> None:
    """Utility for tests or dynamic reloads."""
    TOOLS.clear()
    TOOL_INDEX.clear()


def list_tool_specs() -> List[Dict[str, str]]:
    """
    Return tool metadata for the selector prompt.

    Each BaseTool should define `name` and `description`.
    """
    return [{"name": t.name, "description": t.description or ""} for t in TOOLS]


def auto_register_tool(*tool_args, **tool_kwargs):
    """
    Decorator that wraps LangChain's `@tool` and registers the tool automatically.

    Usage:

        @auto_register_tool("name", description="...")
        def handler(prompt: str) -> ToolInvocation:
            ...
    """

    def decorator(func):
        langchain_tool = lc_tool(*tool_args, **tool_kwargs)(func)
        register_tool(langchain_tool)
        return langchain_tool

    return decorator


@auto_register_tool("sample_weather", description="示例工具：返回固定天气信息。")
def sample_weather(prompt: str) -> ToolInvocation:
    """示例：返回固定的天气信息，演示工具结构。"""
    return ToolInvocation(
        name="sample_weather",
        message="示例天气：未来 3 天晴到多云，最高 26℃，最低 18℃。",
        data={"original_prompt": prompt},
    )


def initialize_tools() -> None:
    """
    Hook for triggering tool registration.

    只需 import 包含工具定义的模块即可触发自动注册：

        from . import tools  # noqa: F401
    """
    pass


def execute_tool(name: str, prompt: str) -> Optional[ToolInvocation]:
    tool = TOOL_INDEX.get(name)
    if not tool:
        return None
    result = tool.invoke(prompt)
    if isinstance(result, ToolInvocation):
        return result
    raise TypeError(
        f"Tool '{name}' returned unsupported type {type(result)!r}; "
        "please return ToolInvocation."
    )
