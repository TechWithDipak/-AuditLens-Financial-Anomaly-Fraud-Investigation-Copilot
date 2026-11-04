"""Unit tests for EDA, reproducible splitting, and leakage-safe preprocessing."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from app.services.ml.preprocessing import (
    LeakageSafePreprocessor,
    compute_eda,
    split_dataset,
)


@pytest.fixture
def sample_data():
    """Create reproducible test dataset with 100 rows, Time, Amount, V1, and Class."""
    np.random.seed(42)
    time = np.linspace(0, 3600, 100)
    v1 = np.random.randn(100)
    amount = np.random.exponential(scale=100.0, size=100)
    # 5% fraud rate (5 frauds out of 100)
    target = np.zeros(100, dtype=int)
    target[np.random.choice(100, size=5, replace=False)] = 1

    return pd.DataFrame(
        {
            "Time": time,
            "V1": v1,
            "Amount": amount,
            "Class": target,
        }
    )


def test_compute_eda_distributions(sample_data):
    """Verify EDA generates correct distribution stats and temporal rates."""
    eda = compute_eda(sample_data)
    assert eda.total_records == 100
    assert eda.total_features == 4
    assert eda.fraud_count == 5
    assert eda.fraud_rate_pct == 5.0
    assert eda.time_analysis is not None
    assert eda.time_analysis.duration_hours == 1.0
    assert len(eda.feature_distributions) == 3  # Time, V1, Amount


def test_stratified_split_preserves_class_ratio(sample_data):
    """Verify stratified split maintains equal fraud proportions across partitions."""
    train_df, val_df, test_df = split_dataset(
        sample_data, strategy="stratified", test_size=0.2, val_size=0.1, random_state=42
    )

    assert len(train_df) == 70
    assert len(val_df) == 10
    assert len(test_df) == 20

    # Ensure each split received fraud cases according to stratification
    assert train_df["Class"].sum() >= 1
    assert test_df["Class"].sum() >= 1


def test_chronological_split_prevents_lookahead_leakage(sample_data):
    """Verify chronological split strictly partitions time with zero future-to-past overlap."""
    train_df, val_df, test_df = split_dataset(
        sample_data, strategy="chronological", test_size=0.2, val_size=0.1
    )

    assert len(train_df) == 70
    assert len(val_df) == 10
    assert len(test_df) == 20

    # Strict temporal boundary check: max(train) <= min(val) <= max(val) <= min(test)
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_leakage_safe_preprocessor(sample_data):
    """Verify preprocessor fits strictly on train and transforms without leaking test data."""
    train_df, _, test_df = split_dataset(sample_data, strategy="stratified", test_size=0.2)

    preprocessor = LeakageSafePreprocessor()
    # Before fit, transform must fail
    with pytest.raises(RuntimeError, match="must be fitted on training data first"):
        preprocessor.transform(test_df)

    # Fit on train only (default test_size=0.2, val_size=0.1 leaves 70 rows for train)
    X_train_scaled = preprocessor.fit_transform(train_df)
    assert X_train_scaled.shape == (70, 3)

    # Scaler center must match training median, NOT test median
    train_amount_median = float(train_df["Amount"].median())
    test_amount_median = float(test_df["Amount"].median())
    assert preprocessor.amount_scaler.center_[0] == pytest.approx(train_amount_median)
    assert preprocessor.amount_scaler.center_[0] != pytest.approx(test_amount_median)

    # Transform test set using train scaler
    X_test_scaled = preprocessor.transform(test_df)
    assert X_test_scaled.shape == (20, 3)
    assert np.isfinite(X_test_scaled).all()


def test_preprocessor_serialization(sample_data):
    """Verify fitted preprocessor can be saved and reloaded with identical transformations."""
    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(sample_data)
    orig_transformed = preprocessor.transform(sample_data)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        preprocessor.save(tmp_path)
        loaded = LeakageSafePreprocessor.load(tmp_path)
        reloaded_transformed = loaded.transform(sample_data)
        np.testing.assert_array_almost_equal(orig_transformed, reloaded_transformed)
    finally:
        import os

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
