"""AuditLens Application Configuration Settings."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration managed via environment variables."""

    # Project metadata
    PROJECT_NAME: str = "AuditLens API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server settings
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    BACKEND_CORS_ORIGINS: list[str] | str = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Database & Storage
    DATABASE_URL: str = "sqlite:///./auditlens.db"
    MODEL_ARTIFACTS_DIR: str = "./models"
    DATA_RAW_DIR: str = "./data/raw"
    DATA_PROCESSED_DIR: str = "./data/processed"
    DATA_SAMPLE_DIR: str = "./data/sample"

    # LLM Settings (optional / offline fallback by default)
    LLM_PROVIDER: str = "none"
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gemini-1.5-flash"

    # Vector store
    VECTOR_STORE_PATH: str = "./data/vector_store"
    EMBEDDING_PROVIDER: str = "local"

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(f"Invalid CORS origins value: {v}")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
