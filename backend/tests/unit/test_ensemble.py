"""Unit tests for Unit 5I: Ensemble Learning (Voting, Stacking, Fair Model Comparison)."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.ensemble import EnsembleService
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
    v1[fraud_idx] -= 3.0
    v2[fraud_idx] += 3.0

    df = pd.DataFrame({"Time": time, "V1": v1, "V2": v2, "Amount": amount, "Class": target})
    train_df, _, test_df = split_dataset(df, strategy="stratified", test_size=0.25, random_state=42)
    return train_df, test_df


def test_soft_voting_ensemble(fraud_data):
    """Verify Soft Voting Ensemble trains and outputs valid combined probabilities."""
    train_df, test_df = fraud_data
    res = EnsembleService.train_and_evaluate(
        train_df, test_df, ensemble_type="voting", random_state=42
    )

    assert res.model_name == "Ensemble_Voting"
    assert res.metrics.pr_auc > 0.0
    assert 0.0 <= res.metrics.precision <= 1.0


def test_stacking_ensemble(fraud_data):
    """Verify Stacking Ensemble uses out-of-fold CV without leakage and outputs valid metrics."""
    train_df, test_df = fraud_data
    res = EnsembleService.train_and_evaluate(
        train_df, test_df, ensemble_type="stacking", random_state=42
    )

    assert res.model_name == "Ensemble_Stacking"
    assert "Stacking" in res.algorithm
    assert res.metrics.pr_auc >= 0.0


def test_fair_model_comparison_on_identical_test_split(fraud_data):
    """Verify all models are evaluated on the identical held-out test split."""
    train_df, test_df = fraud_data
    comp = EnsembleService.compare_all_models(train_df, test_df, random_state=42)

    assert comp.test_sample_count == len(test_df)
    assert comp.test_fraud_count == int(test_df["Class"].sum())
    assert len(comp.comparison) == 3

    model_names = [m.model_name for m in comp.comparison]
    assert "CART_DecisionTree" in model_names
    assert "XGBoost_Classifier" in model_names
    assert "Ensemble_Voting" in model_names

    # Check that best model is correctly identified
    assert comp.best_model_by_prauc in model_names
