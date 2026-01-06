from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(
        default=None, validation_alias="OPENAI_API_BASE"
    )
    extractor_provider: str = Field(
        default="openai", validation_alias="EXTRACTOR_PROVIDER"
    )
    extractor_model: str = Field(
        default="gpt-4.1-mini", validation_alias="EXTRACTOR_MODEL"
    )
    extractor_api_key: Optional[str] = Field(
        default=None, validation_alias="EXTRACTOR_API_KEY"
    )
    extractor_api_base: Optional[str] = Field(
        default=None, validation_alias="EXTRACTOR_API_BASE"
    )
    extractor_temperature: float = Field(
        default=0.0, validation_alias="EXTRACTOR_TEMPERATURE"
    )
    default_region: str = Field(default="global", validation_alias="DEFAULT_REGION")
    fastapi_port: int = Field(default=8000, validation_alias="FASTAPI_PORT")
    variety_provider: str = Field(
        default="mock", validation_alias="VARIETY_PROVIDER"
    )
    variety_api_url: Optional[str] = Field(
        default=None, validation_alias="VARIETY_API_URL"
    )
    variety_api_key: Optional[str] = Field(
        default=None, validation_alias="VARIETY_API_KEY"
    )
    weather_provider: str = Field(
        default="mock", validation_alias="WEATHER_PROVIDER"
    )
    weather_api_url: Optional[str] = Field(
        default=None, validation_alias="WEATHER_API_URL"
    )
    weather_api_key: Optional[str] = Field(
        default=None, validation_alias="WEATHER_API_KEY"
    )
    growth_stage_provider: str = Field(
        default="mock", validation_alias="GROWTH_STAGE_PROVIDER"
    )
    growth_stage_api_url: Optional[str] = Field(
        default=None, validation_alias="GROWTH_STAGE_API_URL"
    )
    growth_stage_api_key: Optional[str] = Field(
        default=None, validation_alias="GROWTH_STAGE_API_KEY"
    )
    recommendation_provider: str = Field(
        default="mock", validation_alias="RECOMMENDATION_PROVIDER"
    )
    recommendation_api_url: Optional[str] = Field(
        default=None, validation_alias="RECOMMENDATION_API_URL"
    )
    recommendation_api_key: Optional[str] = Field(
        default=None, validation_alias="RECOMMENDATION_API_KEY"
    )
    pending_store: str = Field(default="sqlite", validation_alias="PENDING_STORE")
    pending_store_ttl_seconds: int = Field(
        default=1800, validation_alias="PENDING_STORE_TTL_SECONDS"
    )
    pending_store_path: Optional[str] = Field(
        default=None, validation_alias="PENDING_STORE_PATH"
    )
    weather_cache_store: str = Field(
        default="sqlite", validation_alias="WEATHER_CACHE_STORE"
    )
    weather_cache_ttl_seconds: int = Field(
        default=21600, validation_alias="WEATHER_CACHE_TTL_SECONDS"
    )
    weather_cache_path: Optional[str] = Field(
        default=None, validation_alias="WEATHER_CACHE_PATH"
    )
    weather_cache_max_items: int = Field(
        default=128, validation_alias="WEATHER_CACHE_MAX_ITEMS"
    )
    tool_cache_store: str = Field(default="sqlite", validation_alias="TOOL_CACHE_STORE")
    tool_cache_ttl_seconds: int = Field(
        default=3600, validation_alias="TOOL_CACHE_TTL_SECONDS"
    )
    tool_cache_path: Optional[str] = Field(
        default=None, validation_alias="TOOL_CACHE_PATH"
    )
    tool_cache_max_items: int = Field(
        default=512, validation_alias="TOOL_CACHE_MAX_ITEMS"
    )
    interaction_store: str = Field(
        default="sqlite", validation_alias="INTERACTION_STORE"
    )
    interaction_store_ttl_days: int = Field(
        default=30, validation_alias="INTERACTION_STORE_TTL_DAYS"
    )
    interaction_store_path: Optional[str] = Field(
        default=None, validation_alias="INTERACTION_STORE_PATH"
    )
    interaction_store_max_items: int = Field(
        default=2000, validation_alias="INTERACTION_STORE_MAX_ITEMS"
    )

    @field_validator("llm_provider", mode="after")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        return value.lower() if value else value

    @field_validator("extractor_provider", mode="after")
    @classmethod
    def normalize_extractor_provider(cls, value: str) -> str:
        return value.lower() if value else value

    @field_validator(
        "variety_provider",
        "weather_provider",
        "growth_stage_provider",
        "recommendation_provider",
        mode="after",
    )
    @classmethod
    def normalize_tool_provider(cls, value: str) -> str:
        return value.lower() if value else value

    @field_validator(
        "pending_store",
        "weather_cache_store",
        "tool_cache_store",
        "interaction_store",
        mode="after",
    )
    @classmethod
    def normalize_pending_store(cls, value: str) -> str:
        return value.lower() if value else value


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
