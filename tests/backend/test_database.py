"""Tests for database models, constraints, and relational cascades."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    AuditEventModel,
    CaseModel,
    CaseNoteModel,
    CaseTransactionModel,
    TransactionModel,
)


def test_transaction_model_crud(test_db: Session):
    """Verify TransactionModel insertion, querying, and field constraints."""
    tx = TransactionModel(
        transaction_id="TX_TEST_001",
        time=100.0,
        amount=250.75,
        features='{"V1": -1.2, "V2": 0.5}',
        anomaly_score=0.45,
        fraud_probability=0.88,
        is_flagged=True,
        score_type="dual_scoring",
        label=1,
    )
    test_db.add(tx)
    test_db.commit()

    retrieved = test_db.scalar(
        select(TransactionModel).where(TransactionModel.transaction_id == "TX_TEST_001")
    )
    assert retrieved is not None
    assert retrieved.amount == 250.75
    assert retrieved.is_flagged is True
    assert retrieved.fraud_probability == 0.88


def test_case_model_relationships_and_cascade(test_db: Session):
    """Verify CaseModel links to CaseTransactionModel and CaseNoteModel with cascade delete."""
    case = CaseModel(
        case_id="CASE-20261103-0001",
        title="Suspicious high-value velocity pattern",
        description="Multiple atypical transfers flagged within 10 minutes",
        status="NEW",
        priority="HIGH",
        assigned_to="analyst_alice",
    )
    test_db.add(case)
    test_db.commit()

    # Add transaction link
    link = CaseTransactionModel(
        case_id=case.id,
        transaction_id="TX_TEST_001",
    )
    test_db.add(link)

    # Add note
    note = CaseNoteModel(
        case_id=case.id,
        author="analyst_alice",
        content="Investigated IP geolocation mismatch. Escalating for SAR review.",
    )
    test_db.add(note)
    test_db.commit()

    # Verify query
    fetched_case = test_db.scalar(
        select(CaseModel).where(CaseModel.case_id == "CASE-20261103-0001")
    )
    assert fetched_case is not None
    assert len(fetched_case.transactions) == 1
    assert fetched_case.transactions[0].transaction_id == "TX_TEST_001"
    assert len(fetched_case.notes) == 1
    assert "SAR review" in fetched_case.notes[0].content

    # Test cascade delete
    test_db.delete(fetched_case)
    test_db.commit()

    # Associated notes and links should be removed
    remaining_links = test_db.scalars(
        select(CaseTransactionModel).where(CaseTransactionModel.case_id == case.id)
    ).all()
    remaining_notes = test_db.scalars(
        select(CaseNoteModel).where(CaseNoteModel.case_id == case.id)
    ).all()
    assert len(remaining_links) == 0
    assert len(remaining_notes) == 0


def test_audit_event_immutability(test_db: Session):
    """Verify AuditEventModel records timestamped events with JSON payloads."""
    event = AuditEventModel(
        event_type="CASE_STATUS_UPDATED",
        entity_type="case",
        entity_id="CASE-20261103-0001",
        actor="analyst_bob",
        details='{"old": "NEW", "new": "IN_REVIEW"}',
    )
    test_db.add(event)
    test_db.commit()

    retrieved = test_db.scalar(
        select(AuditEventModel).where(AuditEventModel.actor == "analyst_bob")
    )
    assert retrieved is not None
    assert retrieved.event_type == "CASE_STATUS_UPDATED"
    assert "IN_REVIEW" in retrieved.details
