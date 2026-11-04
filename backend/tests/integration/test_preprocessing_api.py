"""Integration tests for EDA and Preprocessing API endpoints."""

from fastapi.testclient import TestClient


def test_get_eda_endpoint(client: TestClient):
    """Verify EDA endpoint returns complete distribution and time statistics."""
    response = client.get("/api/v1/preprocessing/eda?dataset_name=creditcard_sample_fixture.csv")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 1000
    assert data["fraud_count"] == 5
    assert data["fraud_rate_pct"] == 0.5
    assert data["time_analysis"] is not None
    assert len(data["feature_distributions"]) == 30  # Time + 28 V-features + Amount


def test_split_stratified_endpoint(client: TestClient):
    """Verify split endpoint executes stratified split on sample fixture."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "strategy": "stratified",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    response = client.post("/api/v1/preprocessing/split", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "stratified"
    assert data["train"]["sample_count"] == 700
    assert data["val"]["sample_count"] == 100
    assert data["test"]["sample_count"] == 200
    assert len(data["feature_names"]) == 30


def test_split_chronological_endpoint(client: TestClient):
    """Verify split endpoint executes chronological split with non-overlapping times."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "strategy": "chronological",
        "test_size": 0.2,
        "val_size": 0.1,
    }
    response = client.post("/api/v1/preprocessing/split", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "chronological"
    assert data["train"]["max_time"] <= data["val"]["min_time"]
    assert data["val"]["max_time"] <= data["test"]["min_time"]
