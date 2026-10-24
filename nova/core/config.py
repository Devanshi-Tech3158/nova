"""Environment-based configuration settings for Nova."""

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Configuration loaded from environment variables."""

    api_key: str | None = Field(default=None)
    db_path: str = Field(default="nova.db")
    database_url: str | None = Field(default=None)
    llm_provider: str = Field(default="fake")
    llm_model: str = Field(default="fake-model")
    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with defaults."""
        return cls(
            api_key=os.getenv("NOVA_API_KEY") or os.getenv("API_KEY"),
            db_path=os.getenv("NOVA_DB_PATH", "nova.db"),
            database_url=os.getenv("DATABASE_URL"),
            llm_provider=os.getenv("NOVA_LLM_PROVIDER", "fake"),
            llm_model=os.getenv("NOVA_LLM_MODEL", "fake-model"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def get_settings() -> Settings:
    """Return application settings initialized from the current environment."""
    return Settings.from_env()
