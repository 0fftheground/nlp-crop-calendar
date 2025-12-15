"""
LLM 驱动的意图识别模块
使用 LLM 替代手动模式匹配进行意图识别
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LLMIntentRecognizer:
    """
    基于 LLM 的意图识别器
    使用 LLM 自动理解用户意图，无需手动定义模式
    """

    def __init__(self, llm: BaseLanguageModel):
        """
        初始化 LLM 意图识别器

        Args:
            llm: LangChain 语言模型实例
        """
        self.llm = llm
        self.intent_memory = []  # 存储识别历史用于学习

    def recognize_intent(
        self,
        user_input: str,
        available_tools: Optional[list] = None,
        context: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        使用 LLM 识别用户意图

        Args:
            user_input: 用户输入文本
            available_tools: 可用的工具列表（如果有）
            context: 对话上下文

        Returns:
            (意图名称, 置信度, 详细信息)
        """
        # 当使用 Mock LLM 时，走本地简易匹配，避免依赖模型输出 JSON
        if getattr(self.llm, "_llm_type", "") == "mock":
            return self._mock_recognize(user_input, available_tools)

        system_prompt = self._build_system_prompt(available_tools)
        
        # 构建消息
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户说: {user_input}")
        ]
        
        try:
            # 调用 LLM
            response = self.llm.invoke(messages)
            
            # 解析响应
            result = self._parse_llm_response(response.content)
            
            logger.info(f"意图识别: {user_input} -> {result['intent']}")
            
            return (
                result.get("intent", "unknown"),
                result.get("confidence", 0.0),
                result
            )
            
        except Exception as e:
            logger.error(f"LLM 意图识别失败: {str(e)}")
            return ("unknown", 0.0, {"error": str(e)})

    def _mock_recognize(
        self,
        user_input: str,
        available_tools: Optional[list]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Mock LLM 下的简易意图匹配，基于关键词模糊映射到工具
        """
        text = user_input.lower()
        tool_candidates = [
            ("get_weather", ["weather", "天气", "下雨", "温度", "预报"]),
            ("translate_text", ["translate", "翻译", "translate", "英文", "英文翻译"]),
            ("search_information", ["search", "查一下", "搜索", "find", "找一下"]),
            ("call_api", ["api", "接口", "调用"]),
            ("recognize_intent", ["意图", "intent"]),
            ("get_available_intents", ["意图列表", "有哪些意图"]),
            ("get_tool_information", ["工具列表", "tools", "有哪些工具"]),
            ("toggle_mock_mode", ["mock", "测试模式", "切换模式"]),
            ("demo_collect", ["demo", "表单", "报名", "登记", "测试接口"]),
        ]

        def tool_allowed(name: str) -> bool:
            if available_tools is None:
                return True
            if isinstance(available_tools, dict):
                return name in available_tools.keys()
            return name in available_tools

        matched_tool = None
        for tool_name, keywords in tool_candidates:
            if any(k in text for k in keywords) and tool_allowed(tool_name):
                matched_tool = tool_name
                break

        if matched_tool is None:
            return ("unknown", 0.0, {"description": "未匹配到工具"})

        # 给予较高置信度，便于链路继续执行
        details = {
            "tool": matched_tool,
            "description": f"Mock 匹配到工具 {matched_tool}",
            "required_params": {},
        }
        return (matched_tool, 0.8, details)

    def _build_system_prompt(self, available_tools: Optional[list] = None) -> str:
        """
        构建系统提示，包含可用工具的详细信息

        Args:
            available_tools: 可用的工具列表或工具字典

        Returns:
            系统提示文本
        """
        tools_info = ""
        if available_tools:
            if isinstance(available_tools, dict):
                # 如果是字典格式，直接使用
                tools_info = f"""
可用的工具/API 调用格式:
{json.dumps(available_tools, ensure_ascii=False, indent=2)}

请根据用户的输入，识别他们想要使用哪个工具，并提供相应的参数。
"""
            else:
                # 如果是列表，构建描述
                tools_list = ", ".join(available_tools)
                tools_info = f"""
可用的工具: {tools_list}

请根据用户的输入，识别他们想要使用哪个工具。
"""
        
        prompt = f"""你是一个智能意图识别助手，具备工具调用能力。
你的任务是理解用户的自然语言输入，识别他们的真实意图，并推荐相应的工具调用。

{tools_info}

请以 JSON 格式返回以下信息：
{{
    "intent": "识别的意图（推荐使用 tool 名称，如 get_weather, translate_text 等）",
    "confidence": 0.0 到 1.0 之间的置信度,
    "tool": "关联的工具名称（如果有的话）",
    "description": "对用户意图的简短描述",
    "required_params": {{"param_name": "用户提供的值或需求描述"}},
    "clarification": "如果需要澄清，提供问题；否则为 null"
}}

重要：
1. 只返回 JSON，不要其他文字
2. 置信度应该基于用户表述的清晰度
3. 如果不确定，置信度应该较低
4. 如果用户的需求匹配某个工具，在 "tool" 字段中指定工具名称
5. 尽可能准确地识别用户想要的服务
6. 参数应该按照工具的要求格式化
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析 LLM 的 JSON 响应

        Args:
            response_text: LLM 返回的文本

        Returns:
            解析后的字典
        """
        try:
            # 尝试找到 JSON 部分
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                return {
                    "intent": "unknown",
                    "confidence": 0.0,
                    "error": "无法解析 LLM 响应"
                }
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误: {str(e)}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": f"JSON 解析失败: {str(e)}"
            }

    def extract_parameters(
        self,
        user_input: str,
        intent: str,
        required_params: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        从用户输入中提取参数

        Args:
            user_input: 用户输入
            intent: 识别的意图
            required_params: 需要的参数列表

        Returns:
            提取的参数字典
        """
        if not required_params:
            return {}
        
        system_prompt = f"""你是参数提取专家。
从用户的输入中提取以下参数: {', '.join(required_params)}

返回 JSON 格式:
{{
    "extracted_params": {{"param_name": "extracted_value"}},
    "missing_params": ["param_name"]
}}

只返回 JSON，不要其他文字。
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户说: {user_input}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content)
            return result.get("extracted_params", {})
        except Exception as e:
            logger.error(f"参数提取失败: {str(e)}")
            return {}

    def clarify_intent(
        self,
        user_input: str,
        current_intent: str,
        confidence: float
    ) -> Optional[str]:
        """
        当置信度不足时，询问用户以澄清意图

        Args:
            user_input: 用户输入
            current_intent: 当前识别的意图
            confidence: 置信度

        Returns:
            澄清问题，如果不需要则返回 None
        """
        if confidence >= 0.7:
            return None
        
        system_prompt = """你是对话协助助手。
用户的输入不够清晰。请生成一个简洁的澄清问题。
只返回问题本身，不要其他文字。
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户说: {user_input}\n当前理解: {current_intent}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"生成澄清问题失败: {str(e)}")
            return None

    def batch_recognize_intents(
        self,
        user_inputs: list
    ) -> list:
        """
        批量识别多个用户输入的意图

        Args:
            user_inputs: 用户输入列表

        Returns:
            意图识别结果列表
        """
        results = []
        for user_input in user_inputs:
            intent, confidence, details = self.recognize_intent(user_input)
            results.append({
                "input": user_input,
                "intent": intent,
                "confidence": confidence,
                "details": details
            })
        return results

    def get_intent_suggestions(
        self,
        user_input: str,
        num_suggestions: int = 3
    ) -> list:
        """
        获取前 N 个最可能的意图建议

        Args:
            user_input: 用户输入
            num_suggestions: 建议数量

        Returns:
            意图建议列表
        """
        system_prompt = f"""你是意图预测专家。
分析用户输入，列出前 {num_suggestions} 个最可能的意图。

返回 JSON 格式:
{{
    "suggestions": [
        {{"intent": "...", "confidence": 0.9, "reason": "..."}},
        ...
    ]
}}

只返回 JSON。
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户说: {user_input}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content)
            return result.get("suggestions", [])
        except Exception as e:
            logger.error(f"获取意图建议失败: {str(e)}")
            return []

    def adapt_to_user_feedback(
        self,
        user_input: str,
        feedback: str,
        llm_result: Dict[str, Any]
    ) -> None:
        """
        根据用户反馈适应和改进

        Args:
            user_input: 原始用户输入
            feedback: 用户反馈
            llm_result: LLM 之前的识别结果
        """
        # 记录反馈用于后续改进
        self.intent_memory.append({
            "user_input": user_input,
            "feedback": feedback,
            "previous_result": llm_result,
            "corrected": True
        })
        
        logger.info(f"已记录用户反馈: {feedback}")
