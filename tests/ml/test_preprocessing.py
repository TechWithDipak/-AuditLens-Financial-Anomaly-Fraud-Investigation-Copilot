"""Tests for leakage-safe preprocessing pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.pipeline import AuditLensPreprocessor
from ml.preprocessing.splitter import split_dataset

SAMPLE_DIR = Path("./data/sample")


def test_preprocessor_fit_transform():
    """Verify preprocessor fits on training data and transforms both train and test partitions."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    preprocessor = AuditLensPreprocessor(scale_amount=True, encode_time_cyclic=True)
    preprocessor.fit(split.X_train)

    assert preprocessor.is_fitted_ is True
    assert preprocessor.metadata_["n_features_in"] == 30
    assert (
        preprocessor.metadata_["n_features_out"] == 31
    )  # 2 time cyclic + 28 PCA + 1 amount

    X_train_trans = preprocessor.transform(split.X_train)
    X_test_trans = preprocessor.transform(split.X_test)

    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape == (120, 31)
    assert X_test_trans.shape == (30, 31)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_preprocessor_rejects_target_column():
    """Verify preprocessor raises ValueError if target column Class is passed."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    preprocessor = AuditLensPreprocessor()

    with pytest.raises(ValueError, match="Target column 'Class' detected"):
        preprocessor.fit(df)  # DataFrame containing 'Class'


def test_preprocessor_handles_missing_values_at_inference():
    """Verify preprocessor imputes missing values on unseen test records without crashing."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    preprocessor = AuditLensPreprocessor()
    preprocessor.fit(split.X_train)

    # Corrupt test set with NaNs in Amount and PCA columns
    corrupted_test = split.X_test.copy()
    corrupted_test.loc[corrupted_test.index[0], "Amount"] = np.nan
    corrupted_test.loc[corrupted_test.index[1], "V5"] = np.nan

    transformed = preprocessor.transform(corrupted_test)
    assert transformed.shape == (30, 31)
    assert not np.isnan(transformed).any()


def test_preprocessor_robust_amount_scaling():
    """Verify RobustScaler centers Amount around training median."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, test_size=0.2, random_state=42)

    preprocessor = AuditLensPreprocessor(scale_amount=True, encode_time_cyclic=False)
    preprocessor.fit(split.X_train)

    transformed = preprocessor.transform(split.X_train)
    # The last feature is Amount
    amount_col_idx = -1
    scaled_amounts = transformed[:, amount_col_idx]

    # RobustScaler with unit_variance scales by IQR; median should be centered around 0
    train_amount_median = split.X_train["Amount"].median()
    assert preprocessor.metadata_["amount_median"] == float(train_amount_median)
    assert abs(np.median(scaled_amounts)) < 1e-5


def test_preprocessor_save_and_load_reproducibility():
    """Verify saved and reloaded preprocessor produces bit-for-bit identical transforms."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, test_size=0.2, random_state=42)

    preprocessor = AuditLensPreprocessor(scale_amount=True, encode_time_cyclic=True)
    preprocessor.fit(split.X_train)
    original_output = preprocessor.transform(split.X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "preprocessor.joblib"
        preprocessor.save(save_path)

        assert save_path.exists()
        assert save_path.with_suffix(".json").exists()

        loaded = AuditLensPreprocessor.load(save_path)
        loaded_output = loaded.transform(split.X_test)

        np.testing.assert_allclose(original_output, loaded_output, rtol=1e-7, atol=1e-7)
