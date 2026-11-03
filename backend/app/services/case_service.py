"""Service managing investigation cases, note histories, and status transitions."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.db.models import CaseModel, CaseNoteModel, CaseTransactionModel
from backend.app.schemas.case import (
    CaseCreate,
    CaseListResponse,
    CaseNoteCreate,
    CaseNoteResponse,
    CasePriority,
    CaseResponse,
    CaseStatus,
    CaseUpdate,
)
from backend.app.services.audit_service import record_audit_event


def generate_case_id(db: Session) -> str:
    """Generate a sequential, human-readable case identifier (e.g. CASE-20261103-0001)."""
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"CASE-{date_str}-"

    # Query highest sequence for today
    count_today = (
        db.scalar(
            select(func.count(CaseModel.id)).where(CaseModel.case_id.like(f"{prefix}%"))
        )
        or 0
    )
    return f"{prefix}{count_today + 1:04d}"


def case_to_schema(case: CaseModel) -> CaseResponse:
    """Transform an ORM CaseModel into a Pydantic CaseResponse schema."""
    tx_ids = [ct.transaction_id for ct in case.transactions]
    note_schemas = [
        CaseNoteResponse(
            id=n.id,
            case_id=n.case_id,
            author=n.author,
            content=n.content,
            created_at=n.created_at,
        )
        for n in case.notes
    ]

    return CaseResponse(
        id=case.id,
        case_id=case.case_id,
        title=case.title,
        description=case.description,
        status=CaseStatus(case.status),
        priority=CasePriority(case.priority),
        assigned_to=case.assigned_to,
        transaction_ids=tx_ids,
        notes=note_schemas,
        created_at=case.created_at,
        updated_at=case.updated_at,
    )


def create_case(
    db: Session,
    payload: CaseCreate,
    actor: str = "analyst_system",
) -> CaseResponse:
    """Create a new investigation case, link transactions, and record an audit event."""
    cid = generate_case_id(db)

    case = CaseModel(
        case_id=cid,
        title=payload.title,
        description=payload.description or "",
        status=CaseStatus.NEW.value,
        priority=payload.priority.value,
        assigned_to=payload.assigned_to,
    )
    db.add(case)
    db.flush()

    for tx_id in payload.transaction_ids:
        link = CaseTransactionModel(
            case_id=case.id,
            transaction_id=str(tx_id),
        )
        db.add(link)

    db.commit()

    # Re-query with eager loading
    stmt = (
        select(CaseModel)
        .options(selectinload(CaseModel.transactions), selectinload(CaseModel.notes))
        .where(CaseModel.id == case.id)
    )
    persisted_case = db.scalar(stmt)
    assert persisted_case is not None

    record_audit_event(
        db=db,
        event_type="CASE_CREATED",
        entity_type="case",
        entity_id=cid,
        actor=actor,
        details={
            "case_id": cid,
            "title": payload.title,
            "priority": payload.priority.value,
            "linked_transactions": payload.transaction_ids,
        },
    )

    return case_to_schema(persisted_case)


def get_case(db: Session, case_identifier: str | int) -> CaseResponse | None:
    """Retrieve an investigation case by integer id or string case_id."""
    stmt = select(CaseModel).options(
        selectinload(CaseModel.transactions), selectinload(CaseModel.notes)
    )

    if isinstance(case_identifier, int) or (
        isinstance(case_identifier, str) and case_identifier.isdigit()
    ):
        case = db.scalar(stmt.where(CaseModel.id == int(case_identifier)))
        if case:
            return case_to_schema(case)

    case = db.scalar(stmt.where(CaseModel.case_id == str(case_identifier)))
    if case:
        return case_to_schema(case)

    return None


def list_cases(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    status: CaseStatus | None = None,
    priority: CasePriority | None = None,
    assigned_to: str | None = None,
) -> CaseListResponse:
    """Retrieve filtered and paginated investigation cases."""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    query = select(CaseModel).options(
        selectinload(CaseModel.transactions), selectinload(CaseModel.notes)
    )

    if status:
        query = query.where(CaseModel.status == status.value)
    if priority:
        query = query.where(CaseModel.priority == priority.value)
    if assigned_to:
        query = query.where(CaseModel.assigned_to == assigned_to)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    query = (
        query.order_by(desc(CaseModel.updated_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    cases = db.scalars(query).all()
    items = [case_to_schema(c) for c in cases]

    return CaseListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


def update_case(
    db: Session,
    case_identifier: str | int,
    payload: CaseUpdate,
    actor: str = "analyst_system",
) -> CaseResponse:
    """Update case status, priority, or assignee with audit logging."""
    stmt = select(CaseModel).options(
        selectinload(CaseModel.transactions), selectinload(CaseModel.notes)
    )
    if isinstance(case_identifier, int) or (
        isinstance(case_identifier, str) and case_identifier.isdigit()
    ):
        case = db.scalar(stmt.where(CaseModel.id == int(case_identifier)))
    else:
        case = db.scalar(stmt.where(CaseModel.case_id == str(case_identifier)))

    if not case:
        raise ValueError(f"Case not found: {case_identifier}")

    changes: dict[str, Any] = {}

    if payload.title is not None and payload.title != case.title:
        changes["title"] = {"old": case.title, "new": payload.title}
        case.title = payload.title

    if payload.description is not None and payload.description != case.description:
        changes["description"] = {
            "old": case.description,
            "new": payload.description,
        }
        case.description = payload.description

    if payload.status is not None and payload.status.value != case.status:
        changes["status"] = {"old": case.status, "new": payload.status.value}
        case.status = payload.status.value

    if payload.priority is not None and payload.priority.value != case.priority:
        changes["priority"] = {
            "old": case.priority,
            "new": payload.priority.value,
        }
        case.priority = payload.priority.value

    if payload.assigned_to is not None and payload.assigned_to != case.assigned_to:
        changes["assigned_to"] = {
            "old": case.assigned_to,
            "new": payload.assigned_to,
        }
        case.assigned_to = payload.assigned_to

    if changes:
        case.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(case)

        event_type = "CASE_STATUS_UPDATED" if "status" in changes else "CASE_UPDATED"
        record_audit_event(
            db=db,
            event_type=event_type,
            entity_type="case",
            entity_id=case.case_id,
            actor=actor,
            details=changes,
        )

    return case_to_schema(case)


def add_case_note(
    db: Session,
    case_identifier: str | int,
    payload: CaseNoteCreate,
    actor: str = "analyst_system",
) -> CaseNoteResponse:
    """Add a timestamped analyst note to an investigation case."""
    stmt = select(CaseModel).where(
        CaseModel.id == int(case_identifier)
        if (
            isinstance(case_identifier, int)
            or (isinstance(case_identifier, str) and case_identifier.isdigit())
        )
        else CaseModel.case_id == str(case_identifier)
    )
    case = db.scalar(stmt)
    if not case:
        raise ValueError(f"Case not found: {case_identifier}")

    author = payload.author or actor
    note = CaseNoteModel(
        case_id=case.id,
        author=author,
        content=payload.content,
    )
    db.add(note)
    case.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(note)

    record_audit_event(
        db=db,
        event_type="CASE_NOTE_ADDED",
        entity_type="case",
        entity_id=case.case_id,
        actor=actor,
        details={
            "note_id": note.id,
            "author": author,
            "excerpt": payload.content[:100],
        },
    )

    return CaseNoteResponse(
        id=note.id,
        case_id=note.case_id,
        author=note.author,
        content=note.content,
        created_at=note.created_at,
    )
