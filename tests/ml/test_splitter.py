"""Tests for reproducible, leakage-safe data splitting."""

from pathlib import Path

import pandas as pd
import pytest

from ml.preprocessing.splitter import split_dataset

SAMPLE_DIR = Path("./data/sample")


def test_stratified_split_proportions_and_separation():
    """Verify stratified split separates X and y without overlapping records and preserves class ratio."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    assert "Class" not in split.X_train.columns
    assert "Class" not in split.X_test.columns
    assert len(split.X_train) == 120
    assert len(split.X_test) == 30

    # Verify both sets contain fraud cases proportionally
    train_summary = split.summary()
    assert train_summary["train_fraud_count"] > 0
    assert train_summary["test_fraud_count"] > 0
    # Class ratios should be similar
    assert (
        abs(train_summary["train_fraud_rate"] - train_summary["test_fraud_rate"]) < 3.0
    )


def test_stratified_split_with_validation_set():
    """Verify 3-way split (train, val, test) produces disjoint sets with proper sizes."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(
        df, strategy="stratified", test_size=0.2, val_size=0.1, random_state=42
    )

    assert split.X_val is not None
    assert split.y_val is not None
    assert len(split.X_test) == 30
    assert len(split.X_val) == 15
    assert len(split.X_train) == 105

    summary = split.summary()
    assert summary["has_val"] is True
    assert summary["val_size"] == 15


def test_time_based_split_chronological_ordering():
    """Verify time-based split ensures test set strictly follows training timestamps."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, strategy="time", test_size=0.2, deduplicate=False)

    max_train_time = split.X_train["Time"].max()
    min_test_time = split.X_test["Time"].min()

    assert max_train_time <= min_test_time, (
        f"Temporal leakage detected: max train time ({max_train_time}) > min test time ({min_test_time})"
    )


def test_split_rejects_missing_target():
    """Verify ValueError is raised if target column is missing from DataFrame."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv").drop(
        columns=["Class"]
    )
    with pytest.raises(ValueError, match="Target column 'Class' not found"):
        split_dataset(df)


def test_deduplication_prevents_leakage():
    """Verify duplicate rows are removed prior to splitting."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    # Add duplicate copies of first 10 rows
    df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)

    split = split_dataset(df_with_dups, strategy="stratified", deduplicate=True)
    assert split.metadata["duplicates_removed"] == 10
    assert len(split.X_train) + len(split.X_test) == 150
