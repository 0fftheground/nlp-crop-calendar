"""
LangChain Agent 实现
基于 ReAct 和 OpenAI Functions 的 Agent 架构
"""

import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain.agents import (
    create_openai_functions_agent,
    create_react_agent,
    AgentExecutor,
)
from src.llm_config import get_llm
from src.agent_tools import get_all_tools

logger = logging.getLogger(__name__)


class NLPAgent:
    """
    基于 LangChain 的 NLP Agent
    支持多种 Agent 类型和 LLM 提供商
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        agent_type: str = "openai_functions",
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        初始化 Agent

        Args:
            llm_provider: LLM 提供商 ('openai', 'ollama', 'mock')
            agent_type: Agent 类型 ('openai_functions', 'react')
            max_iterations: 最大迭代次数
            verbose: 是否打印详细日志
        """
        self.llm_provider = llm_provider
        self.agent_type = agent_type
        self.max_iterations = max_iterations
        self.verbose = verbose

        # 初始化 LLM
        logger.info(f"初始化 LLM: {llm_provider}")
        self.llm = get_llm(provider=llm_provider)

        # 获取工具
        self.tools = get_all_tools()
        logger.info(f"加载工具: {len(self.tools)} 个")

        # 创建 Agent
        self.agent = self._create_agent()
        # 预处理输入，确保 agent_scratchpad 存在且为列表
        self.agent = RunnableLambda(self._normalize_agent_input) | self.agent
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=False,  # 禁用自动字符串兜底，避免 scratchpad 类型异常
        )

        logger.info("Agent 初始化完成")

    def _create_agent(self):
        """
        创建 Agent

        Returns:
            Agent 实例
        """
        if self.agent_type == "openai_functions":
            return self._create_openai_functions_agent()
        elif self.agent_type == "react":
            return self._create_react_agent()
        else:
            raise ValueError(f"不支持的 Agent 类型: {self.agent_type}")

    @staticmethod
    def _normalize_agent_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        确保传入 Agent 的字段类型符合预期
        - chat_history 缺省为空列表
        - agent_scratchpad 必须为列表，若不是则置空列表
        """
        if inputs.get("chat_history") is None:
            inputs["chat_history"] = []
        scratchpad = inputs.get("agent_scratchpad")
        if not isinstance(scratchpad, list):
            inputs["agent_scratchpad"] = []
        return inputs

    def _create_openai_functions_agent(self):
        """
        创建 OpenAI Functions Agent

        Returns:
            OpenAI Functions Agent
        """
        return create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
        )

    def _create_react_agent(self):
        """
        创建 ReAct Agent

        Returns:
            ReAct Agent
        """
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
        )

    def process_user_input(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        处理用户输入

        Args:
            user_input: 用户输入的文本
            chat_history: 对话历史

        Returns:
            处理结果
        """
        logger.info(f"处理用户输入: {user_input}")

        try:
            # 准备输入
            agent_input = {
                "input": user_input,
                "chat_history": chat_history or [],
            }
            logger.debug(
                "Agent 输入准备完成 | chat_history=%d 条",
                len(agent_input["chat_history"]),
            )

            # 执行 Agent
            result = self.agent_executor.invoke(agent_input)

            return {
                "success": True,
                "user_input": user_input,
                "response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
            }

        except Exception as e:
            logger.error(
                "处理出错: %s | chat_history_type=%s",
                str(e),
                type(chat_history).__name__,
                exc_info=True,
            )
            return {
                "success": False,
                "user_input": user_input,
                "error": str(e),
                "response": "抱歉，处理您的请求时出现错误。请重试。",
            }

    def stream_user_input(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ):
        """
        流式处理用户输入

        Args:
            user_input: 用户输入的文本
            chat_history: 对话历史

        Yields:
            处理过程中的中间步骤
        """
        agent_input = {
            "input": user_input,
            "chat_history": chat_history or [],
            "agent_scratchpad": [],
        }

        for event in self.agent_executor.stream(agent_input):
            yield event

    def get_agent_info(self) -> Dict[str, Any]:
        """
        获取 Agent 信息

        Returns:
            Agent 配置和能力信息
        """
        return {
            "llm_provider": self.llm_provider,
            "agent_type": self.agent_type,
            "max_iterations": self.max_iterations,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools],
        }

    def chat(self, user_input: str) -> str:
        """
        简单的聊天接口

        Args:
            user_input: 用户输入

        Returns:
            Agent 响应
        """
        result = self.process_user_input(user_input)
        return result["response"] if result["success"] else result.get("error", "")


class MultiTurnAgent(NLPAgent):
    """
    支持多轮对话的 Agent
    """

    def __init__(self, *args, **kwargs):
        """初始化多轮对话 Agent"""
        super().__init__(*args, **kwargs)
        self.chat_history: List[BaseMessage] = []

    def add_to_history(self, message: BaseMessage):
        """添加消息到对话历史"""
        self.chat_history.append(message)

    def clear_history(self):
        """清除对话历史"""
        self.chat_history = []

    def chat_with_history(self, user_input: str) -> str:
        """
        支持历史记录的聊天

        Args:
            user_input: 用户输入

        Returns:
            Agent 响应
        """
        # 添加用户输入到历史
        user_message = HumanMessage(content=user_input)
        self.add_to_history(user_message)

        # 处理用户输入
        result = self.process_user_input(user_input, chat_history=self.chat_history)

        # 添加响应到历史
        if result["success"]:
            from langchain_core.messages import AIMessage

            ai_message = AIMessage(content=result["response"])
            self.add_to_history(ai_message)

        return result["response"] if result["success"] else result.get("error", "")

    def get_history(self) -> List[BaseMessage]:
        """获取对话历史"""
        return self.chat_history.copy()

    def get_history_text(self) -> str:
        """获取对话历史的文本形式"""
        history_text = ""
        for msg in self.chat_history:
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            history_text += f"{role}: {msg.content}\n"
        return history_text
