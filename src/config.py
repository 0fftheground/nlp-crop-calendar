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
    default_region: str = Field(default="global", validation_alias="DEFAULT_REGION")
    fastapi_port: int = Field(default=8000, validation_alias="FASTAPI_PORT")

    @field_validator("llm_provider", mode="after")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        return value.lower() if value else value


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
