from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="agents-eval-framework", description="Application name for logs and metrics")
    app_env: str = Field(default="development", description="development | staging | production")
    debug: bool = Field(default=False, description="Enable debug mode")

    api_prefix: str = Field(default="/v1", description="URL prefix for all API routes")

    fireworks_api_key: str = Field(description="Fireworks API key")
    llm_model: str = Field(default="fireworks_ai/accounts/fireworks/models/minimax-m3", description="Primary LLM model")
    llm_timeout_seconds: int = Field(default=60, description="LLM call timeout")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
