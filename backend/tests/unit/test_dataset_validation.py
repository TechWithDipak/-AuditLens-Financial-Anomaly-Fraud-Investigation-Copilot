"""Unit tests for dataset loading, validation, and data quality checks."""

import pandas as pd
import pytest

from app.services.dataset_service import (
    DatasetService,
    DatasetValidationError,
)


def test_valid_dataframe_profile():
    """Verify clean dataframe generates expected profile without warnings."""
    df = pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0],
            "V1": [1.2, -0.5, 0.3],
            "Amount": [50.0, 120.0, 15.0],
            "Class": [0, 0, 1],
        }
    )
    res = DatasetService.profile_dataframe(df, filename="test.csv")
    assert res.status in ["valid", "warning"]
    assert res.quality.row_count == 3
    assert res.quality.column_count == 4
    assert res.quality.duplicate_rows == 0
    assert res.quality.missing_cells_total == 0
    assert len(res.preview_rows) == 3


def test_empty_dataframe_raises():
    """Verify empty dataframe raises validation error."""
    df = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="is empty"):
        DatasetService.profile_dataframe(df, filename="empty.csv")


def test_non_csv_rejected():
    """Verify non-csv file extension is rejected."""
    with pytest.raises(DatasetValidationError, match="Only CSV files are supported"):
        DatasetService.validate_csv_bytes(b"col1,col2\n1,2", filename="test.json")


def test_duplicate_rows_detected():
    """Verify duplicate row counts are properly flagged."""
    df = pd.DataFrame(
        {
            "Time": [0.0, 0.0],
            "V1": [1.0, 1.0],
            "Amount": [10.0, 10.0],
            "Class": [0, 0],
        }
    )
    res = DatasetService.profile_dataframe(df, filename="duplicates.csv")
    assert res.quality.duplicate_rows == 1
    assert any("duplicate" in w.lower() for w in res.quality.warnings)


def test_missing_values_detected():
    """Verify missing values are reported per column and in total."""
    df = pd.DataFrame(
        {
            "Time": [0.0, None],
            "V1": [1.0, 2.0],
            "Amount": [None, 50.0],
            "Class": [0, 1],
        }
    )
    res = DatasetService.profile_dataframe(df, filename="missing.csv")
    assert res.quality.missing_cells_total == 2
    time_col = next(c for c in res.quality.columns if c.name == "Time")
    assert time_col.missing_count == 1
    assert time_col.missing_pct == 50.0


def test_negative_amount_warning():
    """Verify negative transaction amounts trigger audit warning."""
    df = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Amount": [100.0, -25.0],
            "Class": [0, 0],
        }
    )
    res = DatasetService.profile_dataframe(df, filename="negative_amt.csv")
    assert any("negative transaction amounts" in w.lower() for w in res.quality.warnings)


def test_class_imbalance_detection():
    """Verify extreme class imbalance triggers evaluation metric guidance."""
    # 1 fraud out of 100 transactions (1% positive)
    classes = [0] * 99 + [1]
    df = pd.DataFrame(
        {
            "Time": list(range(100)),
            "Amount": [10.0] * 100,
            "Class": classes,
        }
    )
    res = DatasetService.profile_dataframe(df, filename="imbalance.csv")
    assert res.quality.class_imbalance is not None
    assert res.quality.class_imbalance.is_imbalanced is True
    assert any("imbalance" in w.lower() for w in res.quality.warnings)
