"""Integration tests for Unit 5 Supervised & Ensemble API endpoints."""

from fastapi.testclient import TestClient


def test_cart_train_api_endpoint(client: TestClient):
    """Verify CART training API endpoint returns metrics and decision rules."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    response = client.post("/api/v1/algorithms/supervised/cart/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "CART_DecisionTree"
    assert data["train_samples"] == 700
    assert data["test_samples"] == 200
    assert data["metrics"]["pr_auc"] >= 0.0
    assert data["rules_text"] is not None


def test_xgboost_train_api_endpoint(client: TestClient):
    """Verify XGBoost training API returns metrics and feature importances."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    response = client.post("/api/v1/algorithms/supervised/xgboost/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBoost_Classifier"
    assert data["metrics"]["pr_auc"] >= 0.0
    assert len(data["feature_importances"]) > 0


def test_ensemble_voting_api_endpoint(client: TestClient):
    """Verify Ensemble training API with soft voting."""
    payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    response = client.post(
        "/api/v1/algorithms/supervised/ensemble/train?ensemble_type=voting", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Ensemble_Voting"
    assert data["metrics"]["pr_auc"] >= 0.0


def test_model_comparison_api_endpoint(client: TestClient):
    """Verify comparison endpoint compares CART, XGBoost, and Ensemble on identical test set."""
    response = client.get(
        "/api/v1/algorithms/supervised/compare?dataset_name=creditcard_sample_fixture.csv"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["test_sample_count"] == 200
    assert len(data["comparison"]) == 3
    assert data["best_model_by_prauc"] in [
        "CART_DecisionTree",
        "XGBoost_Classifier",
        "Ensemble_Voting",
    ]
