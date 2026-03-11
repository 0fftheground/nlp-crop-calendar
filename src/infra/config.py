import json
from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
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
    agri_db_url: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("AGRI_DB_URL", "DB_URL")
    )
    agri_db_host: Optional[str] = Field(
        default=None, validation_alias="AGRI_DB_HOST"
    )
    agri_db_port: int = Field(default=5432, validation_alias="AGRI_DB_PORT")
    agri_db_name: Optional[str] = Field(
        default=None, validation_alias="AGRI_DB_NAME"
    )
    agri_db_user: Optional[str] = Field(
        default=None, validation_alias="AGRI_DB_USER"
    )
    agri_db_password: Optional[str] = Field(
        default=None, validation_alias="AGRI_DB_PASSWORD"
    )
    agri_db_sslmode: Optional[str] = Field(
        default=None, validation_alias="AGRI_DB_SSLMODE"
    )
    cache_db_url: Optional[str] = Field(
        default=None, validation_alias="CACHE_DB_URL"
    )
    default_farm_id: Optional[str] = Field(
        default=None, validation_alias="DEFAULT_FARM_ID"
    )
    business_api_base_url: Optional[str] = Field(
        default=None, validation_alias="BUSINESS_API_BASE_URL"
    )
    business_api_key: Optional[str] = Field(
        default=None, validation_alias="BUSINESS_API_KEY"
    )
    planting_plan_search_api_url: Optional[str] = Field(
        default=None, validation_alias="PLANTING_PLAN_SEARCH_API_URL"
    )
    planting_plan_active_api_url: Optional[str] = Field(
        default=None, validation_alias="PLANTING_PLAN_ACTIVE_API_URL"
    )
    planting_plan_detail_api_url: Optional[str] = Field(
        default=None, validation_alias="PLANTING_PLAN_DETAIL_API_URL"
    )
    farm_weather_api_url: Optional[str] = Field(
        default=None, validation_alias="FARM_WEATHER_API_URL"
    )
    sowing_suitability_api_url: Optional[str] = Field(
        default=None, validation_alias="SOWING_SUITABILITY_API_URL"
    )
    variety_provider: str = Field(
        default="local", validation_alias="VARIETY_PROVIDER"
    )
    variety_api_url: Optional[str] = Field(
        default=None, validation_alias="VARIETY_API_URL"
    )
    variety_api_key: Optional[str] = Field(
        default=None, validation_alias="VARIETY_API_KEY"
    )
    db_region_lookup_candidates: list[dict[str, str]] = Field(
        default_factory=list, validation_alias="DB_REGION_LOOKUP_CANDIDATES"
    )
    variety_db_table: Optional[str] = Field(
        default=None, validation_alias="VARIETY_DB_TABLE"
    )
    public_base_url: Optional[str] = Field(
        default=None, validation_alias="PUBLIC_BASE_URL"
    )
    growth_stage_provider: str = Field(
        default="local", validation_alias="GROWTH_STAGE_PROVIDER"
    )
    growth_stage_api_url: Optional[str] = Field(
        default=None, validation_alias="GROWTH_STAGE_API_URL"
    )
    growth_stage_api_key: Optional[str] = Field(
        default=None, validation_alias="GROWTH_STAGE_API_KEY"
    )
    region_db_table: Optional[str] = Field(
        default=None, validation_alias="REGION_DB_TABLE"
    )
    region_db_id_column: str = Field(
        default="region_id", validation_alias="REGION_DB_ID_COLUMN"
    )
    region_db_name_column: str = Field(
        default="region_name", validation_alias="REGION_DB_NAME_COLUMN"
    )
    crop_calendar_provider: str = Field(
        default="mock",
        validation_alias=AliasChoices(
            "CROP_CALENDAR_PROVIDER",
            "RECOMMENDATION_PROVIDER",
        ),
    )
    crop_calendar_api_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "CROP_CALENDAR_API_URL",
            "RECOMMENDATION_API_URL",
        ),
    )
    crop_calendar_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "CROP_CALENDAR_API_KEY",
            "RECOMMENDATION_API_KEY",
        ),
    )
    crop_calendar_save_api_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "CROP_CALENDAR_SAVE_API_URL",
            "RECOMMENDATION_SAVE_API_URL",
        ),
    )
    crop_calendar_delete_api_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "CROP_CALENDAR_DELETE_API_URL",
            "RECOMMENDATION_DELETE_API_URL",
        ),
    )
    pending_store: str = Field(default="sqlite", validation_alias="PENDING_STORE")
    pending_store_ttl_seconds: int = Field(
        default=1800, validation_alias="PENDING_STORE_TTL_SECONDS"
    )
    pending_store_path: Optional[str] = Field(
        default=None, validation_alias="PENDING_STORE_PATH"
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
    interaction_raw_max_chars: int = Field(
        default=2000, validation_alias="INTERACTION_RAW_MAX_CHARS"
    )
    interaction_raw_dir: Optional[str] = Field(
        default=None, validation_alias="INTERACTION_RAW_DIR"
    )
    intent_rules_path: Optional[str] = Field(
        default=None, validation_alias="INTENT_RULES_PATH"
    )
    intent_rules_reload_seconds: int = Field(
        default=5, validation_alias="INTENT_RULES_RELOAD_SECONDS"
    )
    intent_routing_mode: str = Field(
        default="llm_only", validation_alias="INTENT_ROUTING_MODE"
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
        "growth_stage_provider",
        "crop_calendar_provider",
        mode="after",
    )
    @classmethod
    def normalize_tool_provider(cls, value: str) -> str:
        return value.lower() if value else value

    @field_validator(
        "pending_store",
        "tool_cache_store",
        "interaction_store",
        mode="after",
    )
    @classmethod
    def normalize_pending_store(cls, value: str) -> str:
        return value.lower() if value else value

    @field_validator("db_region_lookup_candidates", mode="before")
    @classmethod
    def parse_db_region_lookup_candidates(
        cls, value: object
    ) -> list[dict[str, str]] | object:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return parsed
            return []
        return value

    @model_validator(mode="after")
    def build_agri_db_url(self) -> "AppConfig":
        if self.agri_db_url:
            return self
        if not (self.agri_db_host and self.agri_db_name and self.agri_db_user):
            return self
        host = self.agri_db_host
        port = self.agri_db_port or 5432
        user = self.agri_db_user
        password = self.agri_db_password or ""
        auth = f"{user}:{password}" if password else user
        url = f"postgresql://{auth}@{host}:{port}/{self.agri_db_name}"
        if self.agri_db_sslmode:
            url = f"{url}?sslmode={self.agri_db_sslmode}"
        self.agri_db_url = url
        return self


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
