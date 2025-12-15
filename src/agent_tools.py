"""
Agent 工具定义
定义 LLM Agent 可以使用的工具和函数
"""

from typing import Dict, Any
from langchain_core.tools import tool
import logging
from src.api_caller import APICaller, MOCK_API_HOST
from src.intent_recognition_manager import IntentRecognitionManager

logger = logging.getLogger(__name__)

# 全局工具实例
api_caller = APICaller()
intent_manager = IntentRecognitionManager(use_llm=True)


@tool
def call_api(endpoint: str, method: str = "GET", params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    调用外部 API

    Args:
        endpoint: API 端点 URL
        method: HTTP 方法 (GET 或 POST)
        params: 请求参数字典

    Returns:
        API 响应结果
    """
    if params is None:
        params = {}

    logger.info(f"调用 API: {method} {endpoint}")

    response = api_caller.call_api(endpoint=endpoint, method=method, params=params)

    return {
        "success": response.success,
        "data": response.data,
        "error": response.error,
        "status_code": response.status_code,
    }


@tool
def recognize_intent(user_input: str) -> Dict[str, Any]:
    """
    使用 LLM 识别用户输入的意图，考虑可用的工具

    Args:
        user_input: 用户输入的文本

    Returns:
        识别结果，包括意图名称、置信度、关联工具等
    """
    logger.info(f"识别意图: {user_input}")
    
    # 获取所有可用的工具信息并传递给意图识别器
    available_tools = list(intent_manager.tools_info.keys())
    result = intent_manager.recognize(user_input, available_tools=available_tools)
    
    return {
        "intent": result.get("intent"),
        "confidence": result.get("confidence", 0.0),
        "method": result.get("method", "llm"),
        "success": result.get("success", False),
        "associated_tool": result.get("associated_tool"),
        "tool_schema": result.get("tool_schema"),
        "details": result.get("details", {})
    }


@tool
def search_information(query: str, search_type: str = "general") -> Dict[str, Any]:
    """
    搜索信息 (演示工具)

    Args:
        query: 搜索查询
        search_type: 搜索类型 (general, news, images 等)

    Returns:
        搜索结果
    """
    logger.info(f"搜索: {query} (类型: {search_type})")

    # 这是一个演示实现，实际应用中应该调用真实搜索 API
    return {
        "query": query,
        "type": search_type,
        "results": [
            {"title": f"搜索结果 1", "snippet": "相关内容摘要..."},
            {"title": f"搜索结果 2", "snippet": "相关内容摘要..."},
        ],
        "count": 2,
    }


@tool
def get_weather(location: str, units: str = "metric") -> Dict[str, Any]:
    """
    获取天气信息

    Args:
        location: 地点名称
        units: 温度单位 (metric 或 imperial)

    Returns:
        天气信息
    """
    logger.info(f"获取天气: {location}")

    # 调用天气 API
    api_endpoint = "https://api.weatherapi.com/v1/current.json"
    params = {"q": location, "aqi": "no"}

    response = api_caller.call_api(endpoint=api_endpoint, method="GET", params=params)

    if response.success:
        return {
            "success": True,
            "location": location,
            "data": response.data,
        }
    else:
        return {
            "success": False,
            "location": location,
            "error": response.error,
        }


@tool
def translate_text(
    text: str,
    source_language: str = "zh",
    target_language: str = "en",
) -> Dict[str, Any]:
    """
    翻译文本

    Args:
        text: 需要翻译的文本
        source_language: 源语言代码
        target_language: 目标语言代码

    Returns:
        翻译结果
    """
    logger.info(f"翻译文本: {text[:50]}...")

    # 调用翻译 API
    api_endpoint = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"{source_language}|{target_language}",
    }

    response = api_caller.call_api(endpoint=api_endpoint, method="GET", params=params)

    if response.success:
        return {
            "success": True,
            "original_text": text,
            "translated_text": response.data.get("responseData", {}).get("translatedText"),
            "source_language": source_language,
            "target_language": target_language,
        }


@tool
def mock_weather_service(location: str = "Beijing") -> Dict[str, Any]:
    """
    访问内置 Mock 天气接口，避免真实网络请求

    Args:
        location: 查询地点
    """
    endpoint = f"{MOCK_API_HOST}/weather"
    params = {"q": location}
    response = api_caller.call_api(endpoint=endpoint, method="GET", params=params)
    return {
        "endpoint": endpoint,
        "success": response.success,
        "data": response.data,
        "error": response.error,
    }


@tool
def mock_translate_service(text: str, source_language: str = "zh", target_language: str = "en") -> Dict[str, Any]:
    """
    访问内置 Mock 翻译接口
    """
    endpoint = f"{MOCK_API_HOST}/translate"
    params = {
        "q": text,
        "langpair": f"{source_language}|{target_language}",
    }
    response = api_caller.call_api(endpoint=endpoint, method="GET", params=params)
    return {
        "endpoint": endpoint,
        "success": response.success,
        "data": response.data,
        "error": response.error,
    }


@tool
def mock_information_service(query: str = "latest news") -> Dict[str, Any]:
    """
    访问内置 Mock 信息/搜索接口
    """
    endpoint = f"{MOCK_API_HOST}/info"
    params = {"q": query}
    response = api_caller.call_api(endpoint=endpoint, method="GET", params=params)
    return {
        "endpoint": endpoint,
        "success": response.success,
        "data": response.data,
        "error": response.error,
    }
    else:
        return {
            "success": False,
            "error": response.error,
        }


@tool
def get_available_intents() -> Dict[str, Any]:
    """
    获取意图识别的说明和建议

    Returns:
        意图识别的信息和可用功能
    """
    return {
        "description": "已启用 LLM 驱动的意图识别",
        "capabilities": [
            "自动理解用户意图",
            "自动提取参数",
            "支持多语言",
            "生成澄清问题",
            "提供意图建议"
        ],
        "features": {
            "recognize_intent": "使用 LLM 识别用户意图",
            "get_intent_suggestions": "获取多个可能的意图",
            "extract_parameters": "从用户输入中提取参数"
        }
    }


@tool
def get_tool_information() -> Dict[str, Any]:
    """
    获取所有可用工具的信息

    Returns:
        工具列表和描述
    """
    tools_info = {
        "call_api": "调用外部 API，支持 GET 和 POST 请求",
        "recognize_intent": "识别用户输入的自然语言意图",
        "search_information": "搜索信息和数据",
        "get_weather": "获取指定地点的天气信息",
        "translate_text": "翻译文本到指定语言",
        "get_available_intents": "获取系统支持的所有意图",
        "get_tool_information": "获取所有可用工具的信息",
        "toggle_mock_mode": "切换 Mock 模式和真实 API 模式",
        "demo_collect": "演示表单收集接口，要求提供 name/email/city",
    }

    return {
        "tools": tools_info,
        "count": len(tools_info),
    }


@tool
def toggle_mock_mode(enabled: bool = None) -> Dict[str, Any]:
    """
    切换 Mock 模式和真实 API 模式

    Args:
        enabled: True 启用 Mock 模式，False 禁用 Mock 模式，None 返回当前状态

    Returns:
        当前模式状态
    """
    if enabled is not None:
        api_caller.set_mock_mode(enabled)
    
    mode_str = "已启用 Mock 模式" if api_caller.use_mock else "已启用真实 API 模式"
    
    return {
        "mock_mode_enabled": api_caller.use_mock,
        "status": mode_str,
        "description": "使用 Mock 模式可以在没有网络连接或外网限制的环境下测试应用"
    }


@tool
def demo_collect(name: str = None, email: str = None, city: str = None) -> Dict[str, Any]:
    """
    演示表单收集接口，测试缺参提示与回显

    Args:
        name: 姓名
        email: 邮箱
        city: 城市
    """
    missing = [k for k, v in {"name": name, "email": email, "city": city}.items() if not v]
    if missing:
        return {
            "success": False,
            "missing": missing,
            "message": f"缺少字段: {', '.join(missing)}"
        }
    return {
        "success": True,
        "data": {"name": name, "email": email, "city": city},
        "message": "已收到表单信息（演示接口）"
    }


def get_all_tools():
    """
    获取所有工具列表

    Returns:
        工具列表
    """
    return [
        call_api,
        recognize_intent,
        search_information,
        get_weather,
        translate_text,
        mock_weather_service,
        mock_translate_service,
        mock_information_service,
        get_available_intents,
        get_tool_information,
        toggle_mock_mode,
        demo_collect,
    ]
