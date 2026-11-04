"""Integration Tests for Cases and Audit Endpoints."""

from fastapi.testclient import TestClient


def test_create_and_get_case_api(client: TestClient):
    payload = {
        "title": "Suspected Mule Account Activity",
        "description": "Rapid inflows followed by immediate atm withdrawals.",
        "priority": "HIGH",
        "assigned_analyst": "analyst_bob",
        "initial_transactions": [
            {
                "transaction_index": 77,
                "amount": 2500.0,
                "fraud_probability": 0.82,
                "anomaly_score": -0.4,
                "added_reason": "High fraud probability",
            }
        ],
    }
    response = client.post("/api/v1/cases/", json=payload)
    assert response.status_code == 201
    data = response.json()
    case_id = data["id"]
    assert data["case_number"].startswith("CASE-")
    assert data["title"] == "Suspected Mule Account Activity"
    assert data["priority"] == "HIGH"
    assert data["status"] == "NEW"
    assert data["transactions_count"] == 1

    # Fetch detail
    detail_res = client.get(f"/api/v1/cases/{case_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == case_id
    assert len(detail["transactions"]) == 1
    assert detail["transactions"][0]["amount"] == 2500.0


def test_create_case_invalid_priority_api(client: TestClient):
    payload = {
        "title": "Invalid Priority Case",
        "priority": "EXTREME",
    }
    response = client.post("/api/v1/cases/", json=payload)
    assert response.status_code == 400
    assert "Invalid priority" in response.json()["detail"]


def test_update_case_status_api(client: TestClient):
    # First create
    res = client.post(
        "/api/v1/cases/",
        json={"title": "Status Transition Flow", "priority": "MEDIUM"},
    )
    case_id = res.json()["id"]

    # Transition to UNDER_REVIEW
    patch_res = client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"status": "UNDER_REVIEW", "actor": "reviewer_alex", "note": "Review in progress"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "UNDER_REVIEW"

    # Transition to CLOSED_BENIGN
    patch_res2 = client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={
            "status": "CLOSED_BENIGN",
            "actor": "reviewer_alex",
            "note": "Customer verified travel pattern",
        },
    )
    assert patch_res2.status_code == 200
    assert patch_res2.json()["status"] == "CLOSED_BENIGN"

    # Invalid status
    invalid_patch = client.patch(
        f"/api/v1/cases/{case_id}/status",
        json={"status": "UNKNOWN_STATUS"},
    )
    assert invalid_patch.status_code == 400


def test_add_case_notes_and_transactions_api(client: TestClient):
    res = client.post(
        "/api/v1/cases/",
        json={"title": "Notes and Transactions Case", "priority": "LOW"},
    )
    case_id = res.json()["id"]

    # Add note
    note_res = client.post(
        f"/api/v1/cases/{case_id}/notes",
        json={"author": "analyst_carol", "content": "Initial assessment complete."},
    )
    assert note_res.status_code == 201
    assert note_res.json()["author"] == "analyst_carol"

    # Add transaction
    tx_res = client.post(
        f"/api/v1/cases/{case_id}/transactions",
        json={
            "transaction_index": 12,
            "amount": 49.99,
            "fraud_probability": 0.15,
            "anomaly_score": 0.05,
            "added_reason": "Baseline control transaction",
        },
    )
    assert tx_res.status_code == 201
    assert tx_res.json()["transaction_index"] == 12

    # Verify detail has both
    detail_res = client.get(f"/api/v1/cases/{case_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["notes"]) == 1
    assert len(detail["transactions"]) == 1


def test_list_audit_events_api(client: TestClient):
    # Check audit events endpoint
    events_res = client.get("/api/v1/cases/audit/events")
    assert events_res.status_code == 200
    events = events_res.json()
    assert isinstance(events, list)
    assert len(events) > 0
    assert "entity_type" in events[0]
    assert "action" in events[0]
