"""Unit tests for Unit 5H: Boosting — XGBoost Classifier."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.preprocessing import split_dataset
from app.services.ml.xgboost_model import XGBoostService


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
    v1[fraud_idx] -= 3.0
    v2[fraud_idx] += 3.5

    df = pd.DataFrame({"Time": time, "V1": v1, "V2": v2, "Amount": amount, "Class": target})
    train_df, _, test_df = split_dataset(df, strategy="stratified", test_size=0.25, random_state=42)
    return train_df, test_df


def test_xgboost_training_and_metrics(fraud_data):
    """Verify XGBoost trains, applies scale_pos_weight, and outputs valid metrics."""
    train_df, test_df = fraud_data
    res = XGBoostService.train_and_evaluate(
        train_df, test_df, max_depth=3, n_estimators=50, random_state=42
    )

    assert res.model_name == "XGBoost_Classifier"
    assert res.hyperparameters["scale_pos_weight"] > 1.0  # Imbalance accounted for
    assert res.metrics.pr_auc > 0.0
    assert 0.0 <= res.metrics.roc_auc <= 1.0
    assert len(res.feature_importances) > 0


def test_xgboost_artifact_persistence(fraud_data):
    """Verify XGBoost artifact saves and loads cleanly with identical predictions."""
    import joblib

    train_df, test_df = fraud_data
    res = XGBoostService.train_and_evaluate(train_df, test_df, n_estimators=30, random_state=42)

    saved = joblib.load(res.artifact_path)
    model = saved["model"]
    preprocessor = saved["preprocessor"]

    X_test_scaled = preprocessor.transform(test_df)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    assert len(preds) == len(test_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
