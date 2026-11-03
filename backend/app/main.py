"""Main FastAPI application entry point for AuditLens."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.audit import router as audit_router
from backend.app.api.v1.cases import router as cases_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.policy import router as policy_router
from backend.app.api.v1.transactions import router as transactions_router
from backend.app.api.v1.upload import router as upload_router
from backend.app.core.config import settings
from backend.app.core.errors import register_error_handlers
from backend.app.core.logging import setup_logging
from backend.app.db.session import init_db

# Setup application logging with sensitive-field masking
setup_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger("auditlens.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifespan handler."""
    logger.info(
        f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]"
    )
    # Ensure database schema is initialized
    init_db()
    logger.info("Relational database schema initialized successfully.")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Financial Anomaly & Fraud Investigation Copilot API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
origins = settings.BACKEND_CORS_ORIGINS
if isinstance(origins, str):
    origins = [origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register global error handling
register_error_handlers(app)

# Include API routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(transactions_router, prefix="/api/v1")
app.include_router(cases_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(policy_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
# Also expose /health directly at the root for standard load balancer checks
app.include_router(health_router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """Root endpoint welcoming clients and directing to documentation."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": "/health",
    }
