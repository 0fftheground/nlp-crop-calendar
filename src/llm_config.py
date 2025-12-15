"""
LLM 配置和初始化模块
支持多种 LLM 提供商：OpenAI, Ollama, Claude 等
"""

from functools import lru_cache
from typing import Optional, Iterable, Sequence, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.language_models import BaseLanguageModel


class LLMSettings(BaseSettings):
    """LLM 配置，优先从 .env 与环境变量读取"""

    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")

    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama2", env="OLLAMA_MODEL")

    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    llm_timeout: int = Field(30, env="LLM_TIMEOUT")
    default_llm_provider: str = Field("mock", env="DEFAULT_LLM_PROVIDER")
    use_mock_api: bool = Field(False, env="USE_MOCK_API")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> LLMSettings:
    """获取全局配置，启用缓存避免重复读取 .env"""
    return LLMSettings()


class LLMConfig:
    """
    向后兼容的配置类快照，保留旧版静态属性用法。
    建议新代码使用 get_settings()/LLMSettings。
    """

    _settings = get_settings()

    # OpenAI 配置
    OPENAI_API_KEY = _settings.openai_api_key
    OPENAI_MODEL = _settings.openai_model

    # Ollama 配置
    OLLAMA_BASE_URL = _settings.ollama_base_url
    OLLAMA_MODEL = _settings.ollama_model

    # 其他配置
    DEFAULT_TEMPERATURE = _settings.llm_temperature
    MAX_TOKENS = _settings.llm_max_tokens
    TIMEOUT = _settings.llm_timeout


def init_openai_llm() -> BaseLanguageModel:
    """
    初始化 OpenAI LLM

    Returns:
        OpenAI 语言模型实例

    Raises:
        ValueError: 如果未设置 API Key
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API Key 未设置。请设置环境变量 OPENAI_API_KEY"
        )

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            request_timeout=settings.llm_timeout,
        )
        return llm
    except ImportError:
        raise ImportError(
            "langchain-openai 未安装。请运行: pip install langchain-openai"
        )


def init_ollama_llm() -> BaseLanguageModel:
    """
    初始化 Ollama LLM (本地模型)

    Returns:
        Ollama 语言模型实例
    """
    settings = get_settings()

    try:
        from langchain_community.llms import Ollama

        llm = Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
        )
        return llm
    except ImportError:
        raise ImportError(
            "langchain-community 未安装。请运行: pip install langchain-community"
        )


def init_mock_llm() -> BaseLanguageModel:
    """
    初始化 Mock LLM (用于测试)

    Returns:
        Mock 语言模型实例
    """
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.callbacks import (
        CallbackManagerForLLMRun,
        AsyncCallbackManagerForLLMRun,
    )
    from langchain_core.outputs import LLMResult, ChatGeneration
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
    )
    from langchain_core.prompt_values import PromptValue

    class MockLLM(BaseLanguageModel):
        """模拟 LLM，用于测试和演示"""

        default_response: str = "这是一个模拟的 LLM 响应"

        @property
        def _llm_type(self) -> str:
            return "mock"

        # ---------- 基础工具 ----------
        def _ensure_messages(self, data) -> List[BaseMessage]:
            if isinstance(data, list):
                return data
            if isinstance(data, BaseMessage):
                return [data]
            return [HumanMessage(content=str(data))]

        def _build_result(self, messages: Sequence[BaseMessage]) -> LLMResult:
            # 如果提示要求 JSON（意图识别等场景），返回可解析的固定 JSON
            joined = " ".join([getattr(m, "content", "") for m in messages])
            if "JSON" in joined or "intent" in joined.lower():
                content = (
                    '{"intent": "mock_intent", "confidence": 0.62, '
                    '"tool": "search_information", "description": "Mock 意图识别结果", '
                    '"required_params": {}, "clarification": null}'
                )
            else:
                # 找到最近的人类消息，给出可读的 mock 文本
                content = self.default_response
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        content = f"[Mock Response] 已收到: {msg.content}"
                        break
            ai_message = AIMessage(content=content)
            generation = ChatGeneration(message=ai_message)
            return LLMResult(generations=[[generation]])

        # ---------- LLM 接口实现 ----------
        def _generate(
            self,
            messages: Sequence[BaseMessage],
            stop: Optional[Iterable[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs,
        ) -> LLMResult:
            return self._build_result(messages)

        async def _agenerate(
            self,
            messages: Sequence[BaseMessage],
            stop: Optional[Iterable[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs,
        ) -> LLMResult:
            return self._build_result(messages)

        # ---------- Runnable 便捷方法 ----------
        def invoke(self, input, *, stop: Optional[Iterable[str]] = None, **kwargs):
            messages = self._ensure_messages(input)
            return self._build_result(messages).generations[0][0].message

        async def ainvoke(self, input, *, stop: Optional[Iterable[str]] = None, **kwargs):
            return self.invoke(input, stop=stop, **kwargs)

        def predict(self, text: str, *, stop: Optional[Iterable[str]] = None, **kwargs) -> str:
            return self.invoke(text, stop=stop, **kwargs).content

        async def apredict(self, text: str, *, stop: Optional[Iterable[str]] = None, **kwargs) -> str:
            return self.predict(text, stop=stop, **kwargs)

        def predict_messages(
            self,
            messages: Sequence[BaseMessage],
            *,
            stop: Optional[Iterable[str]] = None,
            **kwargs,
        ) -> BaseMessage:
            return self.invoke(messages, stop=stop, **kwargs)

        async def apredict_messages(
            self,
            messages: Sequence[BaseMessage],
            *,
            stop: Optional[Iterable[str]] = None,
            **kwargs,
        ) -> BaseMessage:
            return self.predict_messages(messages, stop=stop, **kwargs)

        def generate_prompt(
            self,
            prompts: Sequence[PromptValue],
            *,
            stop: Optional[Iterable[str]] = None,
            **kwargs,
        ) -> LLMResult:
            generations = []
            for prompt in prompts:
                prompt_messages = prompt.to_messages()
                result = self._build_result(prompt_messages)
                generations.append(result.generations[0])
            return LLMResult(generations=generations)

        async def agenerate_prompt(
            self,
            prompts: Sequence[PromptValue],
            *,
            stop: Optional[Iterable[str]] = None,
            **kwargs,
        ) -> LLMResult:
            return self.generate_prompt(prompts, stop=stop, **kwargs)

    return MockLLM()


def get_llm(provider: str = "openai") -> BaseLanguageModel:
    """
    获取 LLM 实例

    Args:
        provider: LLM 提供商，支持 'openai', 'ollama', 'mock'

    Returns:
        语言模型实例

    Raises:
        ValueError: 如果提供商不支持
    """
    provider = provider.lower()

    if provider == "openai":
        return init_openai_llm()
    elif provider == "ollama":
        return init_ollama_llm()
    elif provider == "mock":
        return init_mock_llm()
    else:
        raise ValueError(
            f"不支持的 LLM 提供商: {provider}。支持: openai, ollama, mock"
        )


# 向后兼容别名
def init_llm(provider: str = "openai") -> BaseLanguageModel:
    """
    兼容旧代码的初始化别名

    Args:
        provider: LLM 提供商

    Returns:
        语言模型实例
    """
    return get_llm(provider)
