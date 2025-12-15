"""
主应用程序
集成 LangChain Agent 的自然语言处理应用
"""

import json
import logging
from typing import Dict, Any, Optional
from src.intent_recognition_manager import IntentRecognitionManager, AGENT_TOOLS_SCHEMA
from src.api_caller import APICaller
from src.agent import NLPAgent, MultiTurnAgent


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NLPApp:
    """
    自然语言处理应用
    集成意图识别、API调用和 LangChain Agent 功能
    """

    def __init__(
        self,
        intent_config_path: str = None,
        use_agent: bool = True,
        llm_provider: str = "mock",
        agent_type: str = "openai_functions",
        use_llm_intent_recognition: bool = True,
    ):
        """
        初始化应用

        Args:
            intent_config_path: 意图配置文件路径（已废弃，保留用于向后兼容性）
            use_agent: 是否使用 LLM Agent
            llm_provider: LLM 提供商
            agent_type: Agent 类型
            use_llm_intent_recognition: 是否使用 LLM 进行意图识别（现已为标准方法，默认为 True）
        """
        # 初始化 LLM 意图识别管理器（现已为唯一方法）
        self.intent_manager = IntentRecognitionManager(
            use_llm=True,
            llm_provider=llm_provider
        )
        self.use_llm_intent = True
        
        self.api_caller = APICaller()
        self.use_agent = use_agent

        # 初始化 Agent（如果启用）
        if use_agent:
            try:
                self.agent = MultiTurnAgent(
                    llm_provider=llm_provider,
                    agent_type=agent_type,
                    verbose=True,
                )
                logger.info(f"LLM Agent 已初始化 (提供商: {llm_provider})")
            except Exception as e:
                logger.warning(f"Agent 初始化失败: {str(e)}，将使用传统模式")
                self.use_agent = False
                self.agent = None
        else:
            self.agent = None

        logger.info("NLP应用已初始化 (意图识别: LLM 驱动)")

    def process_user_input(
        self, user_input: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理用户输入

        Args:
            user_input: 用户输入的自然语言文本
            custom_params: 自定义API参数

        Returns:
            处理结果字典
        """
        logger.info(f"处理用户输入: {user_input}")

        # 如果启用 Agent，优先使用 Agent
        if self.use_agent and self.agent:
            try:
                response = self.agent.chat_with_history(user_input)
                return {
                    "success": True,
                    "mode": "agent",
                    "user_input": user_input,
                    "response": response,
                }
            except Exception as e:
                logger.error(f"Agent 处理失败: {str(e)}，转换为传统模式")

        # 降级到传统意图识别模式
        return self._process_with_intent_recognition(user_input, custom_params)

    def process_input(
        self, user_input: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        兼容旧接口别名，调用 process_user_input
        """
        return self.process_user_input(user_input, custom_params)

    def _process_with_intent_recognition(
        self, user_input: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        使用意图识别处理用户输入

        Args:
            user_input: 用户输入的自然语言文本
            custom_params: 自定义API参数

        Returns:
            处理结果字典
        """
        # 第一步: 识别意图
        recognition_result = self.intent_manager.recognize(user_input)
        
        if not recognition_result.get("success", False):
            return {
                "success": False,
                "mode": "llm_intent",
                "user_input": user_input,
                "error": "LLM 意图识别失败",
            }
        
        intent_name = recognition_result.get("intent")
        confidence = recognition_result.get("confidence", 0.0)
        details = recognition_result.get("details", {})
        intent_config = {"intent": intent_name, "details": details}

        # 根据工具 schema 推导所需参数
        associated_tool = details.get("tool") or intent_name
        tool_schema = AGENT_TOOLS_SCHEMA.get(associated_tool)
        required_params = list(tool_schema.get("parameters", {}).keys()) if tool_schema else []

        if intent_name is None or intent_name == "unknown":
            logger.warning(f"无法识别意图，最高匹配分数: {confidence:.2%}")
            return {
                "success": False,
                "mode": "intent_recognition",
                "message": "无法理解您的请求，请重新表述",
                "user_input": user_input,
                "confidence": confidence,
            }

        logger.info(f"识别的意图: {intent_name} (置信度: {confidence:.2%}")

        # 第二步: 获取API配置
        api_endpoint = intent_config.get("api_endpoint")
        method = intent_config.get("method", "GET")

        # 如果有必填参数但尚未补充，提示用户补全（仅针对无法自动填充的字段）
        auto_fillable = {"location", "query", "category", "timezone", "units", "source_language", "target_language", "text"}
        missing_params = [
            p for p in required_params
            if (custom_params or {}).get(p) is None and p not in auto_fillable
        ]
        if missing_params:
            return {
                "success": False,
                "mode": "intent_recognition",
                "user_input": user_input,
                "intent": intent_name,
                "confidence": confidence,
                "tool": associated_tool,
                "missing_params": missing_params,
                "message": f"请提供以下参数: {', '.join(missing_params)}",
                "details": details,
            }

        # 如果没有可调用的 API，直接返回识别结果
        if not api_endpoint:
            logger.info("意图未配置 API 端点，跳过 API 调用")
            return {
                "success": True,
                "mode": "intent_recognition",
                "user_input": user_input,
                "intent": intent_name,
                "confidence": confidence,
                "api_endpoint": None,
                "api_method": None,
                "api_params": None,
                "api_response": None,
                "message": "该意图未配置 API，已返回识别结果",
                "details": details,
            }

        # 第三步: 准备API参数
        api_params = self._prepare_api_params(
            user_input, required_params, custom_params
        )

        # 第四步: 调用API
        logger.info(f"调用API: {method} {api_endpoint}")
        api_response = self.api_caller.call_api(
            endpoint=api_endpoint, method=method, params=api_params
        )

        # 第五步: 返回结果
        result = {
            "success": api_response.success,
            "mode": "intent_recognition",
            "user_input": user_input,
            "intent": intent_name,
            "confidence": confidence,
            "api_endpoint": api_endpoint,
            "api_method": method,
            "api_params": api_params,
            "api_response": api_response.data if api_response.success else None,
            "error": api_response.error,
        }

        if api_response.success:
            logger.info(f"处理成功: {intent_name}")
        else:
            logger.error(f"API调用失败: {api_response.error}")

        return result

    def _prepare_api_params(
        self,
        user_input: str,
        required_params: list,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        准备API参数

        Args:
            user_input: 用户输入
            required_params: 所需参数列表
            custom_params: 自定义参数

        Returns:
            API参数字典
        """
        params = custom_params.copy() if custom_params else {}

        # 为缺失的参数提供默认值
        for param in required_params:
            if param not in params:
                if param == "location":
                    params[param] = "Beijing"
                elif param == "query":
                    params[param] = user_input
                elif param == "category":
                    params[param] = "general"
                elif param == "timezone":
                    params[param] = "Asia/Shanghai"
                elif param == "units":
                    params[param] = "metric"
                elif param in ["source_language", "target_language"]:
                    params[param] = "zh" if param == "source_language" else "en"
                elif param == "text":
                    params[param] = user_input
                else:
                    params[param] = ""

        return params

    def list_available_intents(self) -> Dict[str, Any]:
        """
        列出 LLM 意图识别的功能和能力

        Returns:
            意图识别功能信息
        """
        return {
            "recognition_method": "LLM 驱动",
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
                "extract_parameters": "从用户输入中提取参数",
                "clarify_intent": "生成澄清问题"
            }
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """
        获取 Agent 信息

        Returns:
            Agent 配置信息
        """
        if self.agent:
            return self.agent.get_agent_info()
        return {"enabled": False, "message": "Agent 未启用"}

    def get_chat_history(self) -> str:
        """获取对话历史"""
        if self.agent and isinstance(self.agent, MultiTurnAgent):
            return self.agent.get_history_text()
        return ""

    def clear_chat_history(self):
        """清除对话历史"""
        if self.agent and isinstance(self.agent, MultiTurnAgent):
            self.agent.clear_history()
            logger.info("对话历史已清除")

    def close(self):
        """关闭应用资源"""
        self.api_caller.close()
        if self.agent:
            logger.info("Agent 已关闭")
        logger.info("应用已关闭")


def main():
    """
    主函数 - 演示应用的交互式使用
    """
    # 初始化应用 (使用 Mock LLM 用于演示，可改为 'openai')
    app = NLPApp(use_agent=True, llm_provider="mock", agent_type="openai_functions")

    print("=" * 50)
    print("欢迎使用 NLP 自然语言意图识别应用 (LangChain 版)")
    print("=" * 50)

    # 显示 Agent 信息
    agent_info = app.get_agent_info()
    if agent_info.get("enabled", True):
        print(f"\nAgent 配置: {agent_info}")
    else:
        print(f"\n{agent_info['message']}")

    print("\n可用意图:")
    for intent in app.list_available_intents():
        print(f"  - {intent}")

    print("\n输入 'quit' 或 'exit' 退出应用")
    print("输入 'history' 查看对话历史")
    print("输入 'clear' 清除对话历史\n")

    # 交互循环
    while True:
        try:
            user_input = input("请输入您的需求: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("感谢使用，再见！")
                break

            if user_input.lower() == "history":
                history = app.get_chat_history()
                print("\n对话历史:")
                print(history if history else "暂无历史记录")
                print()
                continue

            if user_input.lower() == "clear":
                app.clear_chat_history()
                print("对话历史已清除\n")
                continue

            if not user_input:
                continue

            # 处理用户输入
            result = app.process_user_input(user_input)

            # 显示结果
            print("\n" + "=" * 50)
            if result["success"]:
                print("处理成功!")
                print("=" * 50)
                if result.get("mode") == "agent":
                    print(f"响应: {result['response']}")
                else:
                    print(f"意图: {result['intent']}")
                    print(f"置信度: {result['confidence']:.2%}")
                    print(f"API 响应: {json.dumps(result['api_response'], indent=2, ensure_ascii=False)}")
            else:
                print("处理失败")
                print("=" * 50)
                print(f"错误: {result.get('error') or result.get('message')}")
            print()

        except KeyboardInterrupt:
            print("\n\n应用已中断，再见！")
            break
        except Exception as e:
            logger.error(f"处理出错: {str(e)}")
            print(f"错误: {str(e)}\n")

    # 关闭应用
    app.close()


if __name__ == "__main__":
    main()
