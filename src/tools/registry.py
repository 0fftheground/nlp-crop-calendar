from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, tool as lc_tool

from ..schemas.models import ToolInvocation


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


def _not_available(tool_name: str, prompt: str) -> ToolInvocation:
    """统一的工具占位符，提示需要在内网环境中实现真实调用。"""
    return ToolInvocation(
        name=tool_name,
        message="该工具仅在内网环境提供真实数据，当前 demo 环境返回占位响应。",
        data={"original_prompt": prompt, "status": "not_available_in_demo"},
    )


@auto_register_tool("variety_lookup", description="查询品种信息（占位，实现请接入内网 API）。")
def variety_lookup(prompt: str) -> ToolInvocation:
    """
    查询品种信息工具占位符。

    TODO: 在内网环境实现具体逻辑，例如根据品种名称返回成熟期、抗性等。
    """
    return _not_available("variety_lookup", prompt)


@auto_register_tool("weather_lookup", description="查询气象数据（占位，实现请接入自动站服务）。")
def weather_lookup(prompt: str) -> ToolInvocation:
    """
    查询气象数据工具占位符。

    TODO: 在内网环境实现具体逻辑，例如调用气象中心 API 返回逐小时预报。
    """
    return _not_available("weather_lookup", prompt)


@auto_register_tool("growth_stage_prediction", description="生育期预测（占位，实现请接入模型服务）。")
def growth_stage_prediction(prompt: str) -> ToolInvocation:
    """
    生育期预测工具占位符。

    TODO: 在内网环境实现具体逻辑，例如推理作物进度并返回阶段概率。
    """
    return _not_available("growth_stage_prediction", prompt)


@auto_register_tool("farming_recommendation", description="农事推荐（占位，实现请接入知识库/LLM）。")
def farming_recommendation(prompt: str) -> ToolInvocation:
    """
    农事推荐工具占位符。

    TODO: 在内网环境实现具体逻辑，例如结合知识库返回操作清单。
    """
    return _not_available("farming_recommendation", prompt)


def initialize_tools() -> None:
    """
    Hook for triggering tool registration.

    Importing this module elsewhere会执行装饰器，从而注册全部工具。
    """
    return None


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
