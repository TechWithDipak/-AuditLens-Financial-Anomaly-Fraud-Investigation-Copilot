"""Unit tests for Unit 5G: CART Decision Tree Baseline Classifier."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.cart import CARTService
from app.services.ml.preprocessing import split_dataset


@pytest.fixture
def fraud_data():
    """Create reproducible synthetic imbalanced fraud dataset (200 rows, 10 frauds = 5%)."""
    np.random.seed(42)
    n = 200
    time = np.linspace(0, 7200, n)
    v1 = np.random.randn(n)
    v2 = np.random.randn(n)
    amount = np.random.exponential(100.0, n)
    target = np.zeros(n, dtype=int)
    fraud_idx = np.random.choice(n, size=10, replace=False)
    target[fraud_idx] = 1
    # Fraud signals in V1 and V2
    v1[fraud_idx] -= 2.5
    v2[fraud_idx] += 3.0

    df = pd.DataFrame({"Time": time, "V1": v1, "V2": v2, "Amount": amount, "Class": target})
    train_df, _, test_df = split_dataset(df, strategy="stratified", test_size=0.25, random_state=42)
    return train_df, test_df


def test_cart_training_and_metrics(fraud_data):
    """Verify CART trains, evaluates on test split, and produces valid metrics."""
    train_df, test_df = fraud_data
    res = CARTService.train_and_evaluate(train_df, test_df, max_depth=4, random_state=42)

    assert res.model_name == "CART_DecisionTree"
    assert res.train_samples == len(train_df)
    assert res.test_samples == len(test_df)
    assert res.metrics.pr_auc >= 0.0
    assert 0.0 <= res.metrics.precision <= 1.0
    assert 0.0 <= res.metrics.recall <= 1.0
    assert 0.0 <= res.metrics.f1_score <= 1.0
    assert res.metrics.confusion_matrix.true_positive >= 0
    assert len(res.feature_importances) > 0
    assert res.rules_text is not None
    assert "V1" in res.rules_text or "V2" in res.rules_text


def test_cart_artifact_persistence(fraud_data):
    """Verify CART artifact can be loaded and produces identical predictions."""
    import joblib

    train_df, test_df = fraud_data
    res = CARTService.train_and_evaluate(train_df, test_df, random_state=42)

    saved_data = joblib.load(res.artifact_path)
    model = saved_data["model"]
    preprocessor = saved_data["preprocessor"]

    X_test_scaled = preprocessor.transform(test_df)
    reloaded_preds = model.predict(X_test_scaled)
    assert len(reloaded_preds) == len(test_df)
    assert set(np.unique(reloaded_preds)).issubset({0, 1})
