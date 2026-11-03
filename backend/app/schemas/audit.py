"""Pydantic schemas for audit events and trail queries."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AuditEventResponse(BaseModel):
    """Single immutable audit log entry."""

    id: int
    event_type: str
    entity_type: str
    entity_id: str
    actor: str
    details: dict[str, Any] | None = None
    timestamp: datetime

    class Config:
        from_attributes = True


class AuditEventListResponse(BaseModel):
    """Paginated collection of audit log entries."""

    items: list[AuditEventResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
