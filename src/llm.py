from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import get_config


def get_chat_model() -> BaseChatModel:
    cfg = get_config()
    if cfg.llm_provider != "openai":
        raise ValueError("仅支持 OpenAI 作为 LLM 提供商，请设置 LLM_PROVIDER=openai")
    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY 未配置，无法调用 OpenAI API")
    return ChatOpenAI(
        api_key=cfg.openai_api_key,
        temperature=0.2,
        model="gpt-4o-mini",
    )
