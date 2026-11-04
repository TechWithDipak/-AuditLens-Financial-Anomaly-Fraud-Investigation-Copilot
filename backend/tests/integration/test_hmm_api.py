"""Integration tests for Unit 4 HMM API endpoints."""

from fastapi.testclient import TestClient


def test_hmm_forward_api_endpoint(client: TestClient):
    """Verify Forward API calculates sequence log-likelihood."""
    payload = {
        "observations": [0, 1, 0, 2, 1, 0],
    }
    response = client.post("/api/v1/algorithms/hmm/forward", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_length"] == 6
    assert data["log_likelihood"] < 0.0
    assert len(data["scaled_factors"]) == 6
    assert "Sequence likelihood reflects transition consistency" in data["message"]


def test_hmm_viterbi_api_endpoint(client: TestClient):
    """Verify Viterbi API decodes state path and returns state labels."""
    payload = {
        "observations": [0, 0, 1, 3, 2, 0],
    }
    response = client.post("/api/v1/algorithms/hmm/viterbi", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sequence_length"] == 6
    assert len(data["best_path"]) == 6
    assert len(data["state_labels"]) == 6
    assert "Inferred hidden states are statistical" in data["message"]


def test_account_timeline_endpoint(client: TestClient):
    """Verify timeline endpoint extracts account sequence and decodes behavioral states."""
    response = client.get("/api/v1/algorithms/hmm/account-timeline?customer_id=ACC_0001")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "ACC_0001"
    assert data["sequence_length"] == 50  # 50 transactions per account in fixture
    assert len(data["timeline"]) == 50
    assert data["timeline"][0]["step"] == 0
    assert (
        "Normal Activity" in data["state_distribution"]
        or "Elevated / Travel Activity" in data["state_distribution"]
    )


def test_account_timeline_nonexistent_account(client: TestClient):
    """Verify requesting an unknown customer returns HTTP 400 with helpful error."""
    response = client.get("/api/v1/algorithms/hmm/account-timeline?customer_id=ACC_NONEXISTENT")
    assert response.status_code == 400
    assert "No transactions found" in response.json()["detail"]
