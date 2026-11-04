"""Service layer for Case Management and Immutable Audit Logging."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.models.audit import AuditEvent
from app.models.case import Case, CaseNote, CaseTransaction
from app.schemas.case import CaseCreate, CaseNoteCreate, CaseStatusUpdate, CaseTransactionCreate

ALLOWED_STATUSES = {"NEW", "UNDER_REVIEW", "ESCALATED", "CLOSED_BENIGN", "CLOSED_SUSPICIOUS"}
ALLOWED_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def generate_case_number() -> str:
    """Generate a readable, timestamped case number."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:6].upper()
    return f"CASE-{now_str}-{short_id}"


def log_audit_event(
    db: Session,
    entity_type: str,
    entity_id: str,
    action: str,
    actor: str = "analyst",
    details: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    """Record an append-only audit event in the database."""
    details_str = json.dumps(details) if isinstance(details, dict) else details
    event = AuditEvent(
        entity_type=entity_type,
        entity_id=str(entity_id),
        action=action,
        actor=actor,
        details=details_str,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def create_case(db: Session, case_in: CaseCreate, actor: str = "analyst") -> Case:
    """Create a new investigation case, optionally link initial transactions, and log audit event."""
    priority = case_in.priority.upper()
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{case_in.priority}'. Allowed: {sorted(ALLOWED_PRIORITIES)}"
        )

    case = Case(
        case_number=generate_case_number(),
        title=case_in.title,
        description=case_in.description,
        priority=priority,
        status="NEW",
        assigned_analyst=case_in.assigned_analyst,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(case)
    db.flush()

    if case_in.initial_transactions:
        for tx in case_in.initial_transactions:
            case_tx = CaseTransaction(
                case_id=case.id,
                transaction_index=tx.transaction_index,
                amount=tx.amount,
                fraud_probability=tx.fraud_probability,
                anomaly_score=tx.anomaly_score,
                added_reason=tx.added_reason,
                created_at=datetime.now(timezone.utc),
            )
            db.add(case_tx)

    db.commit()
    db.refresh(case)

    log_audit_event(
        db,
        entity_type="case",
        entity_id=case.case_number,
        action="created",
        actor=actor,
        details={
            "case_id": case.id,
            "title": case.title,
            "priority": case.priority,
            "transactions_count": len(case_in.initial_transactions or []),
        },
    )

    return case


def list_cases(
    db: Session,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """List investigation cases with counts for notes and linked transactions."""
    query = db.query(Case)
    if status:
        query = query.filter(Case.status == status.upper())
    if priority:
        query = query.filter(Case.priority == priority.upper())

    cases = query.order_by(Case.created_at.desc()).offset(skip).limit(limit).all()

    # Efficiently load counts
    results = []
    for c in cases:
        notes_count = (
            db.query(func.count(CaseNote.id)).filter(CaseNote.case_id == c.id).scalar() or 0
        )
        txs_count = (
            db.query(func.count(CaseTransaction.id))
            .filter(CaseTransaction.case_id == c.id)
            .scalar()
            or 0
        )
        results.append(
            {
                "id": c.id,
                "case_number": c.case_number,
                "title": c.title,
                "description": c.description,
                "status": c.status,
                "priority": c.priority,
                "assigned_analyst": c.assigned_analyst,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "notes_count": notes_count,
                "transactions_count": txs_count,
            }
        )
    return results


def get_case(db: Session, case_id: int) -> Optional[Case]:
    """Retrieve full case details including sorted notes and transactions."""
    return (
        db.query(Case)
        .options(joinedload(Case.notes), joinedload(Case.transactions))
        .filter(Case.id == case_id)
        .first()
    )


def update_case_status(
    db: Session,
    case_id: int,
    status_update: CaseStatusUpdate,
) -> Case:
    """Transition a case's status, add an audit event, and optionally record an explanatory note."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise ValueError(f"Case with ID {case_id} not found.")

    new_status = status_update.status.upper()
    if new_status not in ALLOWED_STATUSES:
        raise ValueError(
            f"Invalid status '{status_update.status}'. Allowed: {sorted(ALLOWED_STATUSES)}"
        )

    old_status = case.status
    case.status = new_status
    case.updated_at = datetime.now(timezone.utc)

    if status_update.note:
        note = CaseNote(
            case_id=case.id,
            author=status_update.actor,
            content=f"Status changed from {old_status} to {new_status}: {status_update.note}",
            created_at=datetime.now(timezone.utc),
        )
        db.add(note)

    db.commit()
    db.refresh(case)

    log_audit_event(
        db,
        entity_type="case",
        entity_id=case.case_number,
        action="status_changed",
        actor=status_update.actor,
        details={
            "old_status": old_status,
            "new_status": new_status,
            "has_note": bool(status_update.note),
        },
    )

    return case


def add_case_note(
    db: Session,
    case_id: int,
    note_in: CaseNoteCreate,
) -> CaseNote:
    """Add an analyst investigation note to a case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise ValueError(f"Case with ID {case_id} not found.")

    note = CaseNote(
        case_id=case.id,
        author=note_in.author,
        content=note_in.content,
        created_at=datetime.now(timezone.utc),
    )
    case.updated_at = datetime.now(timezone.utc)
    db.add(note)
    db.commit()
    db.refresh(note)

    log_audit_event(
        db,
        entity_type="case",
        entity_id=case.case_number,
        action="note_added",
        actor=note_in.author,
        details={"note_id": note.id, "snippet": note.content[:100]},
    )

    return note


def link_case_transaction(
    db: Session,
    case_id: int,
    tx_in: CaseTransactionCreate,
    actor: str = "analyst",
) -> CaseTransaction:
    """Link a specific transaction and its model scores to an investigation case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise ValueError(f"Case with ID {case_id} not found.")

    tx = CaseTransaction(
        case_id=case.id,
        transaction_index=tx_in.transaction_index,
        amount=tx_in.amount,
        fraud_probability=tx_in.fraud_probability,
        anomaly_score=tx_in.anomaly_score,
        added_reason=tx_in.added_reason,
        created_at=datetime.now(timezone.utc),
    )
    case.updated_at = datetime.now(timezone.utc)
    db.add(tx)
    db.commit()
    db.refresh(tx)

    log_audit_event(
        db,
        entity_type="case",
        entity_id=case.case_number,
        action="transaction_linked",
        actor=actor,
        details={
            "transaction_index": tx.transaction_index,
            "amount": tx.amount,
            "fraud_probability": tx.fraud_probability,
        },
    )

    return tx


def list_audit_events(
    db: Session,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[AuditEvent]:
    """Retrieve immutable audit events, optionally filtered by entity."""
    query = db.query(AuditEvent)
    if entity_type:
        query = query.filter(AuditEvent.entity_type == entity_type)
    if entity_id:
        query = query.filter(AuditEvent.entity_id == entity_id)

    return query.order_by(AuditEvent.timestamp.desc()).offset(skip).limit(limit).all()
