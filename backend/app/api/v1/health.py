"""Health check endpoint router."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from backend.app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
async def get_health() -> dict[str, Any]:
    """Health check endpoint reporting API status, version, and environment."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(UTC).isoformat(),
    }
