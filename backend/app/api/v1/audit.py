"""FastAPI router for querying immutable audit trail logs."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.audit import AuditEventListResponse
from backend.app.services.audit_service import list_audit_events

router = APIRouter(prefix="/audit", tags=["Audit Trail"])


@router.get("/events", response_model=AuditEventListResponse)
def get_audit_events(
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Events per page")] = 50,
    entity_type: Annotated[
        str | None, Query(description="Filter by target entity type")
    ] = None,
    entity_id: Annotated[
        str | None, Query(description="Filter by target entity identifier")
    ] = None,
    actor: Annotated[
        str | None, Query(description="Filter by analyst identity")
    ] = None,
) -> AuditEventListResponse:
    """Retrieve immutable audit events for security and compliance audits."""
    items, total = list_audit_events(
        db=db,
        page=page,
        page_size=page_size,
        entity_type=entity_type,
        entity_id=entity_id,
        actor=actor,
    )
    total_pages = max(1, (total + page_size - 1) // page_size)
    return AuditEventListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
