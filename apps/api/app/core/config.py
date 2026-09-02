from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NexusIQ API"
    environment: str = "development"

    database_url: str = (
        "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"
    )

    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str | None = None
    google_api_key: str | None = None

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
