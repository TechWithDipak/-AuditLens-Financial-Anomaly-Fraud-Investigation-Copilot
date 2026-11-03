"""Tests for Audit Trail API endpoint."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.services.audit_service import record_audit_event


def test_get_audit_events_and_filtering(client: TestClient, test_db: Session):
    """Verify audit events endpoint lists entries and supports actor / entity filters."""
    # Seed audit events
    record_audit_event(
        db=test_db,
        event_type="CASE_CREATED",
        entity_type="case",
        entity_id="CASE-001",
        actor="analyst_alice",
        details={"reason": "test"},
    )
    record_audit_event(
        db=test_db,
        event_type="TRANSACTION_SCORED",
        entity_type="transaction",
        entity_id="TX_999",
        actor="analyst_bob",
        details={"score": 0.95},
    )

    # Fetch all
    res = client.get("/api/v1/audit/events")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2

    # Filter by entity_type=case
    res_case = client.get("/api/v1/audit/events?entity_type=case")
    assert res_case.status_code == 200
    data_case = res_case.json()
    assert data_case["total"] == 1
    assert data_case["items"][0]["entity_type"] == "case"

    # Filter by actor=analyst_bob
    res_bob = client.get("/api/v1/audit/events?actor=analyst_bob")
    assert res_bob.status_code == 200
    data_bob = res_bob.json()
    assert data_bob["total"] == 1
    assert data_bob["items"][0]["actor"] == "analyst_bob"

