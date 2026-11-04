"""Health check endpoint with live component verification."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    timestamp: str
    python_version: str
    services: Dict[str, str]


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system health status and component metadata with live verification."""
    services = {
        "api": "healthy",
        "database": "sqlite_ready",
        "ml_engine": "ready",
        "llm_mode": settings.LLM_PROVIDER,
    }
    overall_status = "ok"

    # 1. Live Database connectivity check
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception:
        services["database"] = "degraded_unreachable"
        overall_status = "degraded"

    # 2. RAG vector store check
    index_file = Path(settings.ARTIFACTS_DIR) / "rag_index.joblib"
    services["rag_index"] = "indexed" if index_file.exists() else "ready_unindexed"

    return HealthResponse(
        status=overall_status,
        app=settings.APP_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split(" ")[0],
        services=services,
    )
