from __future__ import annotations

import importlib
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, tool as lc_tool

from ...infra.tool_cache import get_tool_result_cache
from ...infra.tool_provider import normalize_provider
from ...observability.logging_utils import log_event
from ...schemas.models import ToolInvocation


TOOLS: List[BaseTool] = []
TOOL_INDEX: Dict[str, BaseTool] = {}
HIDDEN_TOOL_NAMES = {"farming_recommendation", "growth_stage_prediction"}


TOOL_CACHEABLE = {
    "variety_lookup",
    "weather_lookup",
    "farming_recommendation",
    "growth_stage_prediction",
}
_TOOL_MODULES = ("variety", "weather", "growth_stage", "recommendation", "memory")
_TOOLS_INITIALIZED = False


def register_tool(tool: BaseTool) -> None:
    """Register a custom LangChain tool for routing."""
    TOOLS.append(tool)
    TOOL_INDEX[tool.name] = tool


def _should_cache_tool_result(result: ToolInvocation) -> bool:
    return bool(result.data)


def _get_cached_tool_result(
    tool_name: str, provider: str, prompt: str
) -> Optional[ToolInvocation]:
    if tool_name not in TOOL_CACHEABLE:
        return None
    cache = get_tool_result_cache()
    payload = cache.get(tool_name, provider, prompt)
    if not payload:
        return None
    try:
        return ToolInvocation(**payload)
    except Exception:
        return None


def _store_tool_result(
    tool_name: str, provider: str, prompt: str, result: ToolInvocation
) -> None:
    if tool_name not in TOOL_CACHEABLE:
        return None
    if not _should_cache_tool_result(result):
        return None
    cache = get_tool_result_cache()
    cache.set(tool_name, provider, prompt, result.model_dump(mode="json"))


def get_cached_tool_result(
    tool_name: str, provider: str, prompt: str
) -> Optional[ToolInvocation]:
    provider = normalize_provider(provider)
    return _get_cached_tool_result(tool_name, provider, prompt)


def cache_tool_result(
    tool_name: str, provider: str, prompt: str, result: ToolInvocation
) -> None:
    _store_tool_result(tool_name, provider, prompt, result)


def clear_tools() -> None:
    """Utility for tests or dynamic reloads."""
    TOOLS.clear()
    TOOL_INDEX.clear()


def list_tool_specs() -> List[Dict[str, str]]:
    """
    Return tool metadata for the selector prompt.

    Each BaseTool should define `name` and `description`.
    """
    initialize_tools()
    return [
        {"name": t.name, "description": t.description or ""}
        for t in TOOLS
        if t.name not in HIDDEN_TOOL_NAMES
    ]


def auto_register_tool(*tool_args, **tool_kwargs):
    """
    Decorator that wraps LangChain's `@tool` and registers the tool automatically.

    Usage:

        @auto_register_tool("name", description="...")
        def handler(prompt: str) -> ToolInvocation:
            ...
    """

    def decorator(func):
        description = tool_kwargs.pop("description", None)
        if description:
            func.__doc__ = description
        langchain_tool = lc_tool(*tool_args, **tool_kwargs)(func)
        register_tool(langchain_tool)
        return langchain_tool

    return decorator


def _summarize_tool_output(result: ToolInvocation) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "name": result.name,
        "message": result.message,
    }
    data = result.data or {}
    if isinstance(data, dict):
        summary["data_keys"] = list(data.keys())
        points = data.get("points")
        if isinstance(points, list):
            summary["points_count"] = len(points)
        operations = data.get("operations")
        if isinstance(operations, list):
            summary["operations_count"] = len(operations)
        recommendations = data.get("recommendations")
        if isinstance(recommendations, list):
            summary["recommendations_count"] = len(recommendations)
    return summary


def _import_tool_modules() -> List[object]:
    modules = []
    package = __package__ or "src.agent.tools"
    for name in _TOOL_MODULES:
        modules.append(importlib.import_module(f"{package}.{name}"))
    return modules


def initialize_tools() -> None:
    """
    Trigger tool registration by importing tool modules.
    """
    global _TOOLS_INITIALIZED
    if _TOOLS_INITIALIZED and TOOLS:
        return None
    modules = _import_tool_modules()
    if not TOOLS:
        for module in modules:
            importlib.reload(module)
    _TOOLS_INITIALIZED = True
    return None


def execute_tool(name: str, prompt: str) -> Optional[ToolInvocation]:
    initialize_tools()
    tool = TOOL_INDEX.get(name)
    if not tool:
        return None
    result = tool.invoke(prompt)
    if isinstance(result, ToolInvocation):
        log_event("tool_output", tool=_summarize_tool_output(result))
        return result
    raise TypeError(
        f"Tool '{name}' returned unsupported type {type(result)!r}; "
        "please return ToolInvocation."
    )
