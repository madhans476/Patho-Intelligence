"""Centralized app configuration via pydantic-settings.

All environment-dependent values live here — nothing hardcoded, nothing
scattered across modules. See docs/ADR for why this pattern was chosen.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    environment: str = "development"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/pathointelligence"

    # Model artifacts
    model_checkpoint_path: str = "models/checkpoints/best_model.pt"

    # LLM reporting
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # Object storage (local path for MVP; swap for S3-compatible in Phase 9)
    artifact_storage_path: str = "data/artifacts"

    # Model decision threshold — see docs/ADR/0009
    decision_threshold: float = 0.211


@lru_cache
def get_settings() -> Settings:
    return Settings()
