"""Integration Tests for AI Investigation Report Assistant API."""

from fastapi.testclient import TestClient


def test_generate_draft_api(client: TestClient):
    payload = {
        "transaction_index": 12,
        "analyst_hypothesis": "Observed sudden spike in outflow velocity.",
        "provider": "offline",
    }
    response = client.post("/api/v1/assistant/generate-draft", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["report_id"].startswith("REP-")
    assert "Tx #12" in data["report_title"]
    assert len(data["observed_facts"]) == 1
    assert data["observed_facts"][0]["transaction_index"] == 12
    assert "fraud_probability" in data["model_scores"]
    assert len(data["suggested_follow_up_questions"]) > 0
    assert data["is_ai_generated"] is True
    assert "HUMAN" in data["mandatory_advisory_label"]
    assert data["verification_passed"] is True


def test_generate_draft_invalid_index_api(client: TestClient):
    payload = {
        "transaction_index": 99999999,  # Out of range
        "provider": "offline",
    }
    response = client.post("/api/v1/assistant/generate-draft", json=payload)
    assert response.status_code == 404


def test_verify_evidence_api(client: TestClient):
    # First generate a valid draft
    gen_res = client.post(
        "/api/v1/assistant/generate-draft",
        json={"transaction_index": 8, "provider": "offline"},
    )
    draft_data = gen_res.json()
    actual_amt = draft_data["observed_facts"][0]["amount"]
    actual_prob = draft_data["model_scores"]["fraud_probability"]

    # Verify matching
    verify_res = client.post(
        "/api/v1/assistant/verify-evidence",
        json={
            "draft": draft_data,
            "actual_amount": actual_amt,
            "actual_fraud_probability": actual_prob,
        },
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["verification_passed"] is True

    # Verify mismatch
    mismatch_res = client.post(
        "/api/v1/assistant/verify-evidence",
        json={
            "draft": draft_data,
            "actual_amount": actual_amt + 5000.0,
            "actual_fraud_probability": actual_prob,
        },
    )
    assert mismatch_res.status_code == 200
    assert mismatch_res.json()["verification_passed"] is False
    assert mismatch_res.json()["discrepancies_detected"] > 0
