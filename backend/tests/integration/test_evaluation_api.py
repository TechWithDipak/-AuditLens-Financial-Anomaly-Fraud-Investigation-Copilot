"""Integration tests for Evaluation, Explainability, and Isolation Forest APIs."""

from fastapi.testclient import TestClient


def test_isolation_forest_api(client: TestClient):
    """Verify Isolation Forest execution endpoint."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "contamination": 0.02,
        "n_estimators": 50,
        "random_state": 42,
    }
    response = client.post("/api/v1/evaluation/isolation-forest/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "IsolationForest"
    assert data["anomalous_sample_count"] > 0
    assert len(data["anomaly_scores_preview"]) > 0


def test_explain_transaction_api(client: TestClient):
    """Verify SHAP explanation API endpoint returns feature attributions."""
    # First train XGBoost to guarantee model artifact exists
    train_payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    client.post("/api/v1/algorithms/supervised/xgboost/train", json=train_payload)

    # Sample transaction features
    features = {"Time": 100.0, "Amount": 85.50, "V1": -2.3, "V2": 1.4}
    response = client.post("/api/v1/evaluation/explain?model_name=xgboost", json=features)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBoost_Classifier"
    assert 0.0 <= data["predicted_probability"] <= 1.0
    assert len(data["top_attributions"]) > 0
    assert "SHAP values measure statistical" in data["caveat"]


def test_thresholds_api(client: TestClient):
    """Verify threshold tuning curve API returns trade-off points."""
    response = client.get("/api/v1/evaluation/thresholds?model_name=xgboost")
    assert response.status_code == 200
    data = response.json()
    assert len(data["points"]) == 12
    assert 0.0 <= data["optimal_f1_threshold"] <= 1.0
    assert 0.0 <= data["high_recall_threshold"] <= 1.0
