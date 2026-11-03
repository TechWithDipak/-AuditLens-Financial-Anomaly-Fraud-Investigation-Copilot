"""Tests for Model metadata and evaluation report API endpoints."""

from fastapi.testclient import TestClient


def test_get_models_metadata(client: TestClient):
    """Verify models metadata returns loaded models, thresholds, and feature count."""
    response = client.get("/api/v1/models/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "inference_engine_ready" in data
    assert "operational_threshold" in data
    assert "feature_count" in data
    assert isinstance(data["models"], dict)
    assert "notice" in data


def test_get_model_evaluation(client: TestClient):
    """Verify evaluation endpoint returns comparative benchmarking metrics."""
    response = client.get("/api/v1/models/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "operational_threshold" in data
    assert isinstance(data["models_evaluated"], list)
    assert isinstance(data["limitations"], list)

