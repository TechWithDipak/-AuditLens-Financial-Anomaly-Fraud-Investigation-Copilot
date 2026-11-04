"""Unit Tests for Case Management and Audit Logging Service."""

import pytest
from sqlalchemy.orm import Session

from app.schemas.case import CaseCreate, CaseNoteCreate, CaseStatusUpdate, CaseTransactionCreate
from app.services import case_service


def test_create_case_basic(db_session: Session):
    case_in = CaseCreate(
        title="Suspicious Structuring Pattern",
        description="Multiple high velocity transactions observed near $10,000 threshold.",
        priority="HIGH",
        assigned_analyst="analyst_smith",
    )
    case = case_service.create_case(db=db_session, case_in=case_in, actor="analyst_smith")

    assert case.id is not None
    assert case.case_number.startswith("CASE-")
    assert case.title == "Suspicious Structuring Pattern"
    assert case.priority == "HIGH"
    assert case.status == "NEW"
    assert case.assigned_analyst == "analyst_smith"

    # Verify audit event was logged
    audit_events = case_service.list_audit_events(
        db_session, entity_type="case", entity_id=case.case_number
    )
    assert len(audit_events) == 1
    assert audit_events[0].action == "created"
    assert audit_events[0].actor == "analyst_smith"


def test_create_case_with_transactions(db_session: Session):
    case_in = CaseCreate(
        title="Card Not Present Anomaly",
        priority="MEDIUM",
        initial_transactions=[
            CaseTransactionCreate(
                transaction_index=42,
                amount=540.25,
                fraud_probability=0.88,
                anomaly_score=-0.21,
                added_reason="High probability from XGBoost",
            ),
            CaseTransactionCreate(
                transaction_index=43,
                amount=1200.00,
                fraud_probability=0.92,
                anomaly_score=-0.35,
                added_reason="Sequence anomaly in HMM",
            ),
        ],
    )
    case = case_service.create_case(db=db_session, case_in=case_in)
    detail = case_service.get_case(db_session, case.id)

    assert detail is not None
    assert len(detail.transactions) == 2
    assert detail.transactions[0].transaction_index == 42
    assert detail.transactions[0].amount == 540.25
    assert detail.transactions[1].amount == 1200.00


def test_create_case_invalid_priority(db_session: Session):
    case_in = CaseCreate(
        title="Invalid Case",
        priority="SUPER_CRITICAL",  # Not in ALLOWED_PRIORITIES
    )
    with pytest.raises(ValueError, match="Invalid priority"):
        case_service.create_case(db=db_session, case_in=case_in)


def test_list_cases_filter(db_session: Session):
    case1 = case_service.create_case(db_session, CaseCreate(title="Case A", priority="LOW"))
    case2 = case_service.create_case(db_session, CaseCreate(title="Case B", priority="HIGH"))

    case_service.update_case_status(
        db_session, case2.id, CaseStatusUpdate(status="UNDER_REVIEW", note="Investigating")
    )

    all_cases = case_service.list_cases(db_session)
    assert len(all_cases) >= 2

    under_review = case_service.list_cases(db_session, status="UNDER_REVIEW")
    assert any(c["id"] == case2.id for c in under_review)
    assert not any(c["id"] == case1.id for c in under_review)

    high_cases = case_service.list_cases(db_session, priority="HIGH")
    assert any(c["id"] == case2.id for c in high_cases)


def test_update_case_status_with_note(db_session: Session):
    case = case_service.create_case(
        db_session, CaseCreate(title="Lifecycle Test", priority="MEDIUM")
    )
    updated = case_service.update_case_status(
        db_session,
        case.id,
        CaseStatusUpdate(status="ESCALATED", actor="lead_analyst", note="Escalating to fraud team"),
    )

    assert updated.status == "ESCALATED"
    detail = case_service.get_case(db_session, case.id)
    assert len(detail.notes) == 1
    assert "Escalating to fraud team" in detail.notes[0].content
    assert detail.notes[0].author == "lead_analyst"

    # Audit events should have 'created' and 'status_changed'
    events = case_service.list_audit_events(
        db_session, entity_type="case", entity_id=case.case_number
    )
    assert len(events) == 2
    actions = [e.action for e in events]
    assert "created" in actions
    assert "status_changed" in actions


def test_update_case_status_invalid(db_session: Session):
    case = case_service.create_case(db_session, CaseCreate(title="Status Test", priority="LOW"))
    with pytest.raises(ValueError, match="Invalid status"):
        case_service.update_case_status(
            db_session, case.id, CaseStatusUpdate(status="NON_EXISTENT_STATUS")
        )


def test_add_case_note(db_session: Session):
    case = case_service.create_case(db_session, CaseCreate(title="Note Test", priority="LOW"))
    note = case_service.add_case_note(
        db_session,
        case.id,
        CaseNoteCreate(
            author="analyst_jane", content="Confirmed identity with cardholder via phone."
        ),
    )

    assert note.id is not None
    assert note.case_id == case.id
    assert note.author == "analyst_jane"
    assert "Confirmed identity" in note.content

    detail = case_service.get_case(db_session, case.id)
    assert len(detail.notes) == 1


def test_link_case_transaction(db_session: Session):
    case = case_service.create_case(db_session, CaseCreate(title="Tx Link Test", priority="HIGH"))
    tx = case_service.link_case_transaction(
        db_session,
        case.id,
        CaseTransactionCreate(
            transaction_index=105,
            amount=999.99,
            fraud_probability=0.75,
            anomaly_score=-0.12,
            added_reason="Correlated with compromised merchant token",
        ),
        actor="analyst_jane",
    )

    assert tx.id is not None
    assert tx.case_id == case.id
    assert tx.transaction_index == 105
    assert tx.amount == 999.99

    detail = case_service.get_case(db_session, case.id)
    assert len(detail.transactions) == 1
    assert detail.transactions[0].transaction_index == 105
