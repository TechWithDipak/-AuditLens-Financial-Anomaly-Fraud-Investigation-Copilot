"""Unit tests for Phase 7: Explainability, Thresholds, and Isolation Forest."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.cart import CARTService
from app.services.ml.explainability import ExplainabilityService
from app.services.ml.isolation_forest import IsolationForestService
from app.services.ml.preprocessing import split_dataset
from app.services.ml.thresholds import ThresholdService
from app.services.ml.xgboost_model import XGBoostService


@pytest.fixture(scope="module")
def trained_models():
    """Train and persist CART and XGBoost models for explainability testing."""
    np.random.seed(42)
    n = 150
    time = np.linspace(0, 3600, n)
    v1 = np.random.randn(n)
    v2 = np.random.randn(n)
    amount = np.random.exponential(50.0, n)
    target = np.zeros(n, dtype=int)
    target[np.random.choice(n, size=8, replace=False)] = 1
    v1[target == 1] -= 3.0

    df = pd.DataFrame({"Time": time, "V1": v1, "V2": v2, "Amount": amount, "Class": target})
    train_df, _, test_df = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    cart_res = CARTService.train_and_evaluate(train_df, test_df, random_state=42)
    xgb_res = XGBoostService.train_and_evaluate(train_df, test_df, n_estimators=20, random_state=42)

    return df, cart_res, xgb_res


def test_isolation_forest():
    """Verify Isolation Forest generates valid anomaly scores and flags outliers."""
    np.random.seed(42)
    normal = np.random.randn(50, 2)
    outlier = np.array([[15.0, 15.0], [-15.0, -15.0]])
    X = np.vstack([normal, outlier])

    res = IsolationForestService.run_isolation_forest(X, contamination=0.05, random_state=42)
    assert res.anomalous_sample_count >= 1
    assert len(res.anomaly_scores_preview) == 20
    # Outliers should have negative decision scores
    assert 50 in res.anomaly_indices_preview or 51 in res.anomaly_indices_preview


def test_threshold_service():
    """Verify threshold evaluation computes expected operational trade-offs."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.85, 0.95])

    res = ThresholdService.compute_threshold_curve(y_true, y_prob, model_name="TestModel")
    assert len(res.points) == 12
    assert 0.0 <= res.optimal_f1_threshold <= 1.0

    # Test trade-off: high threshold (0.9) has lower or equal recall than low threshold (0.1)
    low_th = next(p for p in res.points if p.threshold == 0.10)
    high_th = next(p for p in res.points if p.threshold == 0.90)
    assert low_th.recall >= high_th.recall


def test_shap_explanation_xgboost(trained_models):
    """Verify SHAP explanation computes local attributions for a transaction instance."""
    df, _, _ = trained_models
    sample_row = df.iloc[0].drop("Class").to_dict()

    res = ExplainabilityService.explain_transaction("xgboost", sample_row)
    assert res.model_name == "XGBoost_Classifier"
    assert 0.0 <= res.predicted_probability <= 1.0
    assert len(res.top_attributions) > 0

    # Ensure top attribution is sorted by absolute value
    abs_shaps = [abs(a.shap_value) for a in res.top_attributions]
    assert abs_shaps == sorted(abs_shaps, reverse=True)


def test_shap_explanation_cart(trained_models):
    """Verify SHAP explanation on CART Decision Tree."""
    df, _, _ = trained_models
    sample_row = df.iloc[1].drop("Class").to_dict()

    res = ExplainabilityService.explain_transaction("cart", sample_row)
    assert res.model_name == "CART_DecisionTree"
    assert 0.0 <= res.predicted_probability <= 1.0
    assert len(res.raw_shap_values) == 4  # Time, V1, V2, Amount
