"""
LLM 意图识别工具和辅助函数
简化 LLM 驱动的意图识别的使用
"""

import logging
from typing import Dict, Any, Optional, List
from src.llm_config import init_llm
from src.llm_intent_recognizer import LLMIntentRecognizer

logger = logging.getLogger(__name__)

# Agent Tools 的标准格式定义
AGENT_TOOLS_SCHEMA = {
    "get_growth_stage_info": {
        "name": "get_growth_stage_info",
        "description": "根据",
        "parameters": {
            "endpoint": "API 端点 URL",
            "method": "HTTP 方法 (GET 或 POST)",
            "params": "请求参数字典"
        }
    },
    "recognize_intent": {
        "name": "recognize_intent",
        "description": "使用 LLM 识别用户输入的意图",
        "parameters": {
            "user_input": "用户输入的文本"
        }
    },
    "search_information": {
        "name": "search_information",
        "description": "搜索信息",
        "parameters": {
            "query": "搜索查询",
            "search_type": "搜索类型"
        }
    },
    "get_weather": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
            "location": "地点名称",
            "units": "温度单位"
        }
    },
    "translate_text": {
        "name": "translate_text",
        "description": "翻译文本",
        "parameters": {
            "text": "需要翻译的文本",
            "source_language": "源语言代码",
            "target_language": "目标语言代码"
        }
    },
    "get_available_intents": {
        "name": "get_available_intents",
        "description": "获取意图识别的说明和建议",
        "parameters": {}
    },
    "get_tool_information": {
        "name": "get_tool_information",
        "description": "获取所有可用工具的信息",
        "parameters": {}
    },
    "mock_weather_service": {
        "name": "mock_weather_service",
        "description": "访问内置 Mock 天气接口 (https://mock.api.local/weather)",
        "parameters": {
            "location": "城市或地点名称"
        }
    },
    "mock_translate_service": {
        "name": "mock_translate_service",
        "description": "访问 Mock 翻译接口 (https://mock.api.local/translate)",
        "parameters": {
            "text": "需要翻译的文本",
            "source_language": "源语言代码",
            "target_language": "目标语言代码"
        }
    },
    "mock_information_service": {
        "name": "mock_information_service",
        "description": "访问 Mock 搜索/资讯接口 (https://mock.api.local/info)",
        "parameters": {
            "query": "想查询的关键词"
        }
    },
    "toggle_mock_mode": {
        "name": "toggle_mock_mode",
        "description": "切换 Mock 模式和真实 API 模式",
        "parameters": {
            "enabled": "True 启用 Mock，False 禁用，None 返回状态"
        }
    },
    "demo_collect": {
        "name": "demo_collect",
        "description": "演示表单收集接口，需用户补充信息",
        "parameters": {
            "name": "用户姓名",
            "email": "邮箱地址",
            "city": "城市"
        }
    }
}


class IntentRecognitionManager:
    """
    意图识别管理器
    自动选择使用 LLM 意图识别或传统方式
    集成 Agent Tools 的 API 调用格式
    """

    def __init__(self, use_llm: bool = True, llm_provider: str = "mock", available_tools: Optional[List[str]] = None):
        """
        初始化意图识别管理器

        Args:
            use_llm: 是否使用 LLM 意图识别
            llm_provider: LLM 提供商 (openai, ollama, mock)
            available_tools: 可用的工具列表名称 (若为 None，则使用所有默认工具)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.recognizer = None
        
        # 初始化可用的工具列表
        if available_tools is None:
            self.available_tools = list(AGENT_TOOLS_SCHEMA.keys())
        else:
            self.available_tools = available_tools
        
        # 构建工具信息（用于传递给 LLM）
        self.tools_info = self._build_tools_info()
        
        if use_llm:
            try:
                llm = init_llm(llm_provider)
                self.recognizer = LLMIntentRecognizer(llm)
                logger.info(f"LLM 意图识别已初始化，可用工具: {', '.join(self.available_tools)}")
            except Exception as e:
                logger.warning(f"LLM 意图识别初始化失败: {str(e)}")
                self.use_llm = False

    def _build_tools_info(self) -> Dict[str, Any]:
        """
        构建工具信息字典，用于传递给 LLM

        Returns:
            工具信息字典
        """
        tools_info = {}
        for tool_name in self.available_tools:
            if tool_name in AGENT_TOOLS_SCHEMA:
                tools_info[tool_name] = AGENT_TOOLS_SCHEMA[tool_name]
        return tools_info

    def get_tools_info(self) -> Dict[str, Any]:
        """
        获取工具信息（供外部使用）

        Returns:
            工具信息字典
        """
        return self.tools_info

    def recognize(
        self,
        user_input: str,
        available_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        识别用户意图

        Args:
            user_input: 用户输入
            available_tools: 可用的工具列表（若提供则覆盖初始化时的设置）

        Returns:
            识别结果字典，包含意图、置信度、关联的工具等信息
        """
        if not self.use_llm or not self.recognizer:
            return self._fallback_recognition(user_input)
        
        try:
            # 如果提供了 available_tools，则更新工具信息
            tools_info = available_tools if available_tools else self.tools_info
            
            intent, confidence, details = self.recognizer.recognize_intent(
                user_input,
                available_tools=list(tools_info.keys()) if isinstance(tools_info, dict) else tools_info
            )
            
            # 增强结果：添加关联的工具信息
            associated_tool = self._find_associated_tool(intent, details)
            
            return {
                "success": True,
                "method": "llm",
                "intent": intent,
                "confidence": confidence,
                "associated_tool": associated_tool,
                "tool_schema": AGENT_TOOLS_SCHEMA.get(associated_tool) if associated_tool else None,
                "details": details
            }
        except Exception as e:
            logger.error(f"LLM 意图识别失败: {str(e)}")
            return self._fallback_recognition(user_input)

    def _find_associated_tool(self, intent: str, details: Dict[str, Any]) -> Optional[str]:
        """
        根据识别的意图找到关联的工具

        Args:
            intent: 识别的意图
            details: 意图详情

        Returns:
            关联的工具名称
        """
        # 直接匹配意图名称与工具名称
        if intent in AGENT_TOOLS_SCHEMA:
            return intent
        
        # 尝试从 details 中查找工具信息
        if details.get("tool"):
            tool_name = details["tool"]
            if tool_name in AGENT_TOOLS_SCHEMA:
                return tool_name
        
        # 根据意图描述匹配工具
        intent_lower = intent.lower()
        for tool_name, tool_schema in AGENT_TOOLS_SCHEMA.items():
            if intent_lower in tool_schema.get("description", "").lower():
                return tool_name
        
        return None

    def _fallback_recognition(self, user_input: str) -> Dict[str, Any]:
        """
        降级到基础识别方式

        Args:
            user_input: 用户输入

        Returns:
            识别结果
        """
        return {
            "success": False,
            "method": "fallback",
            "intent": "unknown",
            "confidence": 0.0,
            "message": "使用 LLM 失败，请配置 LLM 或使用传统模式"
        }

    def get_suggestions(
        self,
        user_input: str,
        num_suggestions: int = 3
    ) -> list:
        """
        获取意图建议

        Args:
            user_input: 用户输入
            num_suggestions: 建议数量

        Returns:
            意图建议列表
        """
        if not self.use_llm or not self.recognizer:
            return []
        
        try:
            return self.recognizer.get_intent_suggestions(
                user_input,
                num_suggestions
            )
        except Exception as e:
            logger.error(f"获取意图建议失败: {str(e)}")
            return []

    def extract_params(
        self,
        user_input: str,
        intent: str,
        required_params: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        从用户输入中提取参数

        Args:
            user_input: 用户输入
            intent: 意图
            required_params: 需要的参数列表

        Returns:
            提取的参数字典
        """
        if not self.use_llm or not self.recognizer:
            return {}
        
        try:
            return self.recognizer.extract_parameters(
                user_input,
                intent,
                required_params
            )
        except Exception as e:
            logger.error(f"参数提取失败: {str(e)}")
            return {}


# 全局管理器实例
_intent_manager = None


def get_intent_manager(use_llm: bool = True, llm_provider: str = "mock") -> IntentRecognitionManager:
    """
    获取全局意图识别管理器

    Args:
        use_llm: 是否使用 LLM
        llm_provider: LLM 提供商

    Returns:
        意图识别管理器实例
    """
    global _intent_manager
    if _intent_manager is None:
        _intent_manager = IntentRecognitionManager(use_llm, llm_provider)
    return _intent_manager


def quick_recognize(user_input: str) -> str:
    """
    快速识别意图（简化接口）

    Args:
        user_input: 用户输入

    Returns:
        识别的意图名称
    """
    manager = get_intent_manager()
    result = manager.recognize(user_input)
    return result.get("intent", "unknown")


def quick_get_suggestions(user_input: str, num: int = 3) -> list:
    """
    快速获取意图建议

    Args:
        user_input: 用户输入
        num: 建议数量

    Returns:
        意图建议列表
    """
    manager = get_intent_manager()
    return manager.get_suggestions(user_input, num)
