"""Application Configuration Module."""

import json
from pathlib import Path
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "AuditLens"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API Routing
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def resolve_database_url(cls, v: str) -> str:
        if v.startswith("sqlite:///./"):
            rel_path = v.replace("sqlite:///./", "")
            abs_path = (ROOT_DIR / rel_path).resolve().as_posix()
            return f"sqlite:///{abs_path}"
        elif v.startswith("sqlite:///") and not v.startswith("sqlite:///:memory:"):
            # Ensure forward slashes for Windows
            raw_path = v.replace("sqlite:///", "")
            posix_path = Path(raw_path).resolve().as_posix()
            return f"sqlite:///{posix_path}"
        return v

    # Storage Paths
    DATA_RAW_DIR: str = str(ROOT_DIR / "data" / "raw")
    DATA_PROCESSED_DIR: str = str(ROOT_DIR / "data" / "processed")
    DATA_FIXTURES_DIR: str = str(ROOT_DIR / "data" / "fixtures")
    DATA_REFERENCE_DOCS_DIR: str = str(ROOT_DIR / "data" / "reference_docs")
    ARTIFACTS_DIR: str = str(ROOT_DIR / "artifacts")
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'data' / 'auditlens.db').as_posix()}"

    # Optional LLM
    LLM_PROVIDER: str = "offline"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    EMBEDDING_PROVIDER: str = "local"

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
