"""Integration tests for Unit 3 Algorithms API endpoints."""

from fastapi.testclient import TestClient


def test_kmeans_api_endpoint(client: TestClient):
    """Verify K-Means execution endpoint produces cluster summaries and elbow curve."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "k": 3,
        "compute_elbow": True,
        "random_state": 42,
    }
    response = client.post("/api/v1/algorithms/kmeans/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["k"] == 3
    assert data["inertia"] > 0
    assert len(data["cluster_summaries"]) == 3
    assert len(data["elbow_curve"]) > 0
    assert "Cluster membership indicates behavioral grouping" in data["message"]


def test_gmm_api_endpoint(client: TestClient):
    """Verify GMM execution endpoint returns converged EM model with log-likelihoods."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "n_components": 3,
        "covariance_type": "full",
        "unusual_percentile": 2.0,
        "random_state": 42,
    }
    response = client.post("/api/v1/algorithms/gmm/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_components"] == 3
    assert data["converged"] is True
    assert data["unusual_sample_count"] > 0
    assert "Low log-likelihood reflects statistical rarity" in data["message"]


def test_pca_api_endpoint(client: TestClient):
    """Verify PCA endpoint returns variance ratios and 2D projection."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "n_components": 2,
        "random_state": 42,
    }
    response = client.post("/api/v1/algorithms/pca/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_components"] == 2
    assert len(data["explained_variance_ratio"]) == 2
    assert data["total_explained_variance"] > 0
    assert len(data["projection_preview"][0]) == 2
