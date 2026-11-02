"""Tests for transaction and CSV schema validation layer."""

from pathlib import Path

import pandas as pd

from ml.data.validator import validate_csv_bytes, validate_dataframe

SAMPLE_DIR = Path("./data/sample")


def test_validate_valid_dataframe():
    """Verify clean DataFrame passes validation."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    result = validate_dataframe(df)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.row_count == 150
    assert result.column_count == 31


def test_validate_empty_dataframe():
    """Verify empty DataFrame fails validation with explicit message."""
    empty_df = pd.DataFrame()
    result = validate_dataframe(empty_df)
    assert result.is_valid is False
    assert any("empty" in err.lower() for err in result.errors)


def test_validate_missing_columns():
    """Verify DataFrame missing required PCA features fails validation."""
    df = pd.read_csv(SAMPLE_DIR / "corrupted_missing_columns.csv")
    result = validate_dataframe(df)
    assert result.is_valid is False
    assert len(result.missing_columns) > 0
    assert any("V3" in col for col in result.missing_columns)
    assert any("Missing required columns" in err for err in result.errors)


def test_validate_corrupted_types():
    """Verify DataFrame with strings in numeric columns fails validation."""
    df = pd.read_csv(SAMPLE_DIR / "corrupted_types.csv")
    result = validate_dataframe(df)
    assert result.is_valid is False
    assert any("non-numeric" in err for err in result.errors)


def test_validate_negative_amounts():
    """Verify negative transaction amounts are detected and rejected."""
    df = pd.read_csv(SAMPLE_DIR / "corrupted_negative_amount.csv")
    result = validate_dataframe(df)
    assert result.is_valid is False
    assert any("Negative transaction amounts" in err for err in result.errors)


def test_validate_training_mode_requires_target():
    """Verify is_training=True requires Class column."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    df_no_target = df.drop(columns=["Class"])

    # Inference mode: valid without Class
    assert validate_dataframe(df_no_target, is_training=False).is_valid is True

    # Training mode: invalid without Class
    res_train = validate_dataframe(df_no_target, is_training=True)
    assert res_train.is_valid is False
    assert "Class" in res_train.missing_columns


def test_validate_duplicate_rows_warning():
    """Verify duplicated rows generate operational warning without crashing."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    # Duplicate first 5 rows
    df_dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
    result = validate_dataframe(df_dup)
    assert result.is_valid is True
    assert result.duplicate_count == 5
    assert len(result.warnings) > 0
    assert any("duplicate rows" in w for w in result.warnings)


def test_validate_csv_bytes_oversized():
    """Verify CSV byte validation rejects content exceeding max size."""
    # Create 200 bytes of dummy content with a 100 byte limit
    dummy_bytes = b"Time,Amount\n1,10\n" * 10
    df, result = validate_csv_bytes(dummy_bytes, filename="test.csv", max_size_bytes=50)
    assert df is None
    assert result.is_valid is False
    assert any("exceeds maximum allowed size" in err for err in result.errors)


def test_validate_csv_bytes_empty():
    """Verify 0 byte content is rejected."""
    df, result = validate_csv_bytes(b"", filename="empty.csv")
    assert df is None
    assert result.is_valid is False
    assert any("empty" in err for err in result.errors)
