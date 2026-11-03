"""Service for creating and querying immutable audit trail events."""

import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import AuditEventModel


def record_audit_event(
    db: Session,
    event_type: str,
    entity_type: str,
    entity_id: str,
    actor: str = "analyst_system",
    details: Mapping[str, Any] | None = None,
) -> AuditEventModel:
    """Record an immutable event in the audit trail.

    Args:
        db: Database session.
        event_type: Type of action (e.g. CASE_CREATED, CASE_STATUS_UPDATED).
        entity_type: Domain entity (case, transaction, dataset).
        entity_id: Identifier of the target entity.
        actor: Identity of the analyst or automated worker.
        details: Optional structured dictionary of contextual changes.

    Returns:
        AuditEventModel: The persisted audit event record.
    """
    details_str = json.dumps(details) if details is not None else None
    event = AuditEventModel(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=str(entity_id),
        actor=actor,
        details=details_str,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_audit_events(
    db: Session,
    page: int = 1,
    page_size: int = 50,
    entity_type: str | None = None,
    entity_id: str | None = None,
    actor: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Retrieve paginated audit trail events with optional filtering.

    Returns:
        tuple[list[dict], int]: List of parsed event dictionaries and total count.
    """
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    query = select(AuditEventModel)
    if entity_type:
        query = query.where(AuditEventModel.entity_type == entity_type)
    if entity_id:
        query = query.where(AuditEventModel.entity_id == entity_id)
    if actor:
        query = query.where(AuditEventModel.actor == actor)

    # Count total
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    # Paginate descending
    query = (
        query.order_by(desc(AuditEventModel.timestamp))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    records = db.scalars(query).all()

    items = []
    for r in records:
        details_obj = None
        if r.details:
            try:
                details_obj = json.loads(r.details)
            except (json.JSONDecodeError, TypeError):
                details_obj = {"raw": r.details}

        items.append(
            {
                "id": r.id,
                "event_type": r.event_type,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "actor": r.actor,
                "details": details_obj,
                "timestamp": r.timestamp,
            }
        )

    return items, total
