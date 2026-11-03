"""Tests for Case management endpoints, status transitions, notes, and audit tracking."""

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import AuditEventModel


def test_create_case_success_and_audit(client: TestClient, test_db: Session):
    """Verify creating a case generates sequential ID, links transactions, and creates audit event."""
    payload = {
        "title": "Suspected mule account rapid dispersion",
        "description": "5 outbound transactions just below AML reporting thresholds.",
        "priority": "HIGH",
        "transaction_ids": ["TX_TEST_0001", "TX_TEST_0002"],
        "assigned_to": "analyst_sarah",
    }
    response = client.post(
        "/api/v1/cases",
        json=payload,
        headers={"X-Analyst-ID": "analyst_supervisor"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "CASE-" in data["case_id"]
    assert data["title"] == payload["title"]
    assert data["status"] == "NEW"
    assert data["priority"] == "HIGH"
    assert len(data["transaction_ids"]) == 2

    # Verify audit event in DB
    events = test_db.scalars(
        select(AuditEventModel).where(AuditEventModel.entity_id == data["case_id"])
    ).all()
    assert len(events) >= 1
    assert events[0].event_type == "CASE_CREATED"
    assert events[0].actor == "analyst_supervisor"


def test_create_case_validation_error(client: TestClient):
    """Verify case creation with invalid or missing fields is rejected with 422."""
    payload = {
        "title": "AB",  # Too short (< 3 chars)
        "priority": "INVALID_PRIORITY",
    }
    response = client.post("/api/v1/cases", json=payload)
    assert response.status_code == 422


def test_list_and_filter_cases(client: TestClient):
    """Verify listing cases with status and priority filtering."""
    # Create two cases
    client.post(
        "/api/v1/cases",
        json={"title": "Case Priority Low", "priority": "LOW"},
    )
    client.post(
        "/api/v1/cases",
        json={"title": "Case Priority High", "priority": "HIGH"},
    )

    # Filter priority=HIGH
    res_high = client.get("/api/v1/cases?priority=HIGH")
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert data_high["total"] >= 1
    for c in data_high["items"]:
        assert c["priority"] == "HIGH"


def test_get_case_detail_and_404(client: TestClient):
    """Verify retrieving case details and 404 for unknown case."""
    create_res = client.post(
        "/api/v1/cases",
        json={"title": "Case For Detail Fetch", "priority": "MEDIUM"},
    )
    case_id = create_res.json()["case_id"]

    # Fetch detail
    res = client.get(f"/api/v1/cases/{case_id}")
    assert res.status_code == 200
    assert res.json()["case_id"] == case_id

    # Unknown case
    res_404 = client.get("/api/v1/cases/CASE-99999999-9999")
    assert res_404.status_code == 404


def test_update_case_status_and_audit(client: TestClient, test_db: Session):
    """Verify updating status records transition in audit log."""
    create_res = client.post(
        "/api/v1/cases",
        json={"title": "Case for transition", "priority": "MEDIUM"},
    )
    case_id = create_res.json()["case_id"]

    # Transition to IN_REVIEW
    update_res = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "IN_REVIEW", "assigned_to": "analyst_bob"},
        headers={"X-Analyst-ID": "analyst_bob"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "IN_REVIEW"
    assert update_res.json()["assigned_to"] == "analyst_bob"

    # Transition to CLOSED_CONFIRMED_FRAUD
    close_res = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "CLOSED_CONFIRMED_FRAUD"},
        headers={"X-Analyst-ID": "analyst_bob"},
    )
    assert close_res.status_code == 200
    assert close_res.json()["status"] == "CLOSED_CONFIRMED_FRAUD"

    # Check audit events
    audit_events = test_db.scalars(
        select(AuditEventModel).where(AuditEventModel.entity_id == case_id)
    ).all()
    event_types = [e.event_type for e in audit_events]
    assert "CASE_CREATED" in event_types
    assert "CASE_STATUS_UPDATED" in event_types


def test_add_note_to_case(client: TestClient):
    """Verify adding notes to a case."""
    create_res = client.post(
        "/api/v1/cases",
        json={"title": "Case with notes", "priority": "MEDIUM"},
    )
    case_id = create_res.json()["case_id"]

    note_payload = {
        "author": "analyst_investigator",
        "content": "Requested cardholder transaction verification via SMS auth.",
    }
    note_res = client.post(f"/api/v1/cases/{case_id}/notes", json=note_payload)
    assert note_res.status_code == 201
    data = note_res.json()
    assert data["author"] == "analyst_investigator"
    assert "SMS auth" in data["content"]

    # Verify note appears in case detail
    case_res = client.get(f"/api/v1/cases/{case_id}")
    assert len(case_res.json()["notes"]) == 1
    assert case_res.json()["notes"][0]["content"] == note_payload["content"]

