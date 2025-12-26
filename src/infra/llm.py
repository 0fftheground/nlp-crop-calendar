from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import get_config


def get_chat_model() -> BaseChatModel:
    cfg = get_config()
    if cfg.llm_provider != "openai":
        raise ValueError("仅支持 OpenAI 作为 LLM 提供商，请设置 LLM_PROVIDER=openai")
    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY 未配置，无法调用 OpenAI API")
    kwargs = {
        "api_key": cfg.openai_api_key,
        "temperature": 0.2,
        "model": "gpt-4.1-mini",
    }
    if cfg.openai_api_base:
        kwargs["base_url"] = cfg.openai_api_base
    return ChatOpenAI(**kwargs)


def get_extractor_model() -> BaseChatModel:
    cfg = get_config()
    if cfg.extractor_provider != "openai":
        raise ValueError(
            "仅支持 OpenAI 作为抽取模型提供商，请设置 EXTRACTOR_PROVIDER=openai"
        )
    api_key = cfg.extractor_api_key or cfg.openai_api_key
    if not api_key:
        raise ValueError("EXTRACTOR_API_KEY 未配置，无法调用 OpenAI API")
    kwargs = {
        "api_key": api_key,
        "temperature": cfg.extractor_temperature,
        "model": cfg.extractor_model,
    }
    base_url = cfg.extractor_api_base or cfg.openai_api_base
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
