"""Tests for repeatable data loading module."""

from pathlib import Path

import pytest

from ml.data.loader import load_dataset

SAMPLE_DIR = Path("./data/sample")


def test_load_valid_sample_dataset():
    """Verify loading a valid synthetic dataset produces DataFrame and clean validation."""
    valid_path = SAMPLE_DIR / "synthetic_creditcard_sample.csv"
    assert valid_path.exists(), "Sample fixture must exist"

    df, validation = load_dataset(valid_path)
    assert not df.empty
    assert df.shape[0] == 150
    assert df.shape[1] == 31  # 30 features + Class
    assert validation.is_valid is True
    assert len(validation.errors) == 0


def test_load_dataset_nonexistent_file():
    """Verify loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/sample/does_not_exist.csv")


def test_load_dataset_unsupported_extension():
    """Verify loading a non-CSV file raises ValueError."""
    dummy_txt = SAMPLE_DIR / "dummy.txt"
    dummy_txt.write_text("dummy text content")
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(dummy_txt)
    finally:
        if dummy_txt.exists():
            dummy_txt.unlink()


def test_load_dataset_nrows_limit():
    """Verify nrows parameter restricts loaded rows cleanly."""
    valid_path = SAMPLE_DIR / "synthetic_creditcard_sample.csv"
    df, _ = load_dataset(valid_path, nrows=10)
    assert len(df) == 10
