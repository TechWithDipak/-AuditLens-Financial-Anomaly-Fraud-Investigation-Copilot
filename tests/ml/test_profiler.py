"""Tests for dataset profiling and statistical inspection."""

from pathlib import Path

import pandas as pd

from ml.data.profiler import generate_profiling_report, profile_dataset

SAMPLE_DIR = Path("./data/sample")


def test_profile_dataset_metrics():
    """Verify profile_dataset extracts dimensions, quality, amount, and class statistics."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    profile = profile_dataset(df)

    # Dimensions
    assert profile["dimensions"]["rows"] == 150
    assert profile["dimensions"]["columns"] == 31
    assert profile["dimensions"]["memory_mb"] > 0

    # Quality
    assert profile["quality"]["total_missing_values"] == 0

    # Amount stats
    amt = profile["amount_statistics"]
    assert amt["min"] >= 0
    assert amt["max"] > amt["min"]
    assert amt["mean"] > 0
    assert amt["median"] > 0

    # Time coverage
    time_cov = profile["time_coverage"]
    assert time_cov["min_seconds"] >= 0
    assert time_cov["max_seconds"] > 0
    assert time_cov["span_hours"] > 0

    # Class distribution
    cls_dist = profile["class_distribution"]
    assert cls_dist["legitimate_count"] > 0
    assert cls_dist["fraud_count"] > 0
    assert cls_dist["fraud_percentage"] > 0

    # PCA features summary
    assert len(profile["pca_features"]) == 28


def test_generate_profiling_report():
    """Verify markdown report generation produces well-formatted headings and statistics."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    report = generate_profiling_report(df, title="Synthetic Profiling Smoke Test")

    assert "# Synthetic Profiling Smoke Test" in report
    assert "## 1. Overview & Dimensions" in report
    assert "## 2. Data Quality & Missingness" in report
    assert "## 3. Time Coverage" in report
    assert "## 4. Amount Statistics" in report
    assert "## 5. Class Distribution" in report
    assert "**Row count**: 150" in report
