"""Unit tests to verify zero future-to-past data leakage and causal integrity."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_causal_customer_features,
    engineer_base_features,
    FeaturePipeline,
)


def create_mock_customer_txns():
    """Create a controlled sequence of customer transactions."""
    times = pd.date_range("2020-01-01 10:00:00", periods=5, freq="2h")
    df = pd.DataFrame(
        {
            "cc_num": [1001] * 5,
            "trans_date_trans_time": times,
            "amt": [100.0, 150.0, 200.0, 250.0, 300.0],
            "lat": [40.0] * 5,
            "long": [-74.0] * 5,
            "merch_lat": [40.1] * 5,
            "merch_long": [-74.1] * 5,
            "dob": ["1985-05-15"] * 5,
            "gender": ["M"] * 5,
            "city_pop": [50000] * 5,
            "category": ["grocery_pos"] * 5,
            "is_fraud": [0, 0, 0, 0, 0],
        }
    )
    return df


def test_first_transaction_has_zero_past_history():
    """First transaction must have 0 count in 24h, default time gap, and 0 z-score."""
    df = create_mock_customer_txns()
    res = compute_causal_customer_features(df)

    # First transaction checks
    assert res.loc[0, "txn_count_last_24h"] == 0.0
    assert res.loc[0, "amt_zscore_vs_customer_history"] == 0.0
    assert res.loc[0, "secs_since_last_txn"] == 86400.0 * 30.0


def test_no_future_leakage_when_future_txns_added():
    """Adding future transactions must NOT change features of earlier transactions."""
    df_early = create_mock_customer_txns().iloc[:3].copy()
    res_early = compute_causal_customer_features(df_early)

    # Add 2 future transactions with extreme amounts
    future_times = pd.date_range("2020-01-01 16:00:00", periods=2, freq="1h")
    df_future = pd.DataFrame(
        {
            "cc_num": [1001, 1001],
            "trans_date_trans_time": future_times,
            "amt": [99999.0, 88888.0],
            "lat": [40.0, 40.0],
            "long": [-74.0, -74.0],
            "merch_lat": [40.1, 40.1],
            "merch_long": [-74.1, -74.1],
            "dob": ["1985-05-15", "1985-05-15"],
            "gender": ["M", "M"],
            "city_pop": [50000, 50000],
            "category": ["grocery_pos", "grocery_pos"],
            "is_fraud": [1, 1],
        }
    )
    df_combined = pd.concat([df_early, df_future], ignore_index=True)
    res_combined = compute_causal_customer_features(df_combined)

    # Earlier 3 rows must be mathematically identical
    for col in ["secs_since_last_txn", "amt_zscore_vs_customer_history", "txn_count_last_24h"]:
        np.testing.assert_allclose(
            res_early[col].values,
            res_combined.iloc[:3][col].values,
            err_msg=f"Leakage detected in {col}: future transactions altered past features!",
        )


def test_current_transaction_excluded_from_own_history():
    """Current transaction must not be included in its own z-score or 24h count."""
    df = create_mock_customer_txns()
    res = compute_causal_customer_features(df)

    # For transaction 1 (second txn, amt=150):
    # Past history contains ONLY txn 0 (amt=100)
    # Mean of past = 100.0, std = 0.0 -> z-score = 0.0
    assert res.loc[1, "txn_count_last_24h"] == 1.0

    # For transaction 2 (third txn, amt=200):
    # Past history contains txn 0 (100) and txn 1 (150).
    # Past mean = 125.0, past std = sqrt(((100-125)^2 + (150-125)^2)/2) = 25.0
    # Expected z-score = (200 - 125) / 25 = 3.0
    past_mean = (100.0 + 150.0) / 2.0
    past_std = np.std([100.0, 150.0])  # population std (n denominator)
    expected_zscore = (200.0 - past_mean) / (past_std + 1e-4)

    np.testing.assert_allclose(
        res.loc[2, "amt_zscore_vs_customer_history"],
        expected_zscore,
        rtol=1e-3,
        err_msg="Current transaction was improperly included in historical baseline!",
    )


def test_forbidden_columns_not_in_features():
    """Ensure cc_num, trans_num, names, street, unix_time are never model features."""
    df = create_mock_customer_txns()
    df["first"] = "John"
    df["last"] = "Doe"
    df["street"] = "123 Main St"
    df["trans_num"] = "abc123xyz"
    df["unix_time"] = 1577872800

    feat_df = engineer_base_features(df)
    pipe = FeaturePipeline()
    X = pipe.fit_transform(feat_df)

    forbidden = ["cc_num", "trans_num", "first", "last", "street", "unix_time", "Unnamed: 0"]
    for col in forbidden:
        assert col not in pipe.feature_names, f"Forbidden column {col} found in feature names!"
        assert col not in X.columns, f"Forbidden column {col} found in transformed output!"


def test_pipeline_fits_on_train_only():
    """Verify preprocessor parameters only derive from train data."""
    train_df = engineer_base_features(create_mock_customer_txns())
    pipe = FeaturePipeline()
    pipe.fit(train_df)

    train_mean = pipe.scaler.mean_.copy()

    # Create dummy test df with 100x values
    test_df = train_df.copy()
    test_df["amt"] = test_df["amt"] * 1000.0

    # Transforming test must NOT change pipeline's learned parameters
    pipe.transform(test_df)
    np.testing.assert_allclose(
        pipe.scaler.mean_,
        train_mean,
        err_msg="Pipeline scaler parameters mutated during transform of test data!",
    )
