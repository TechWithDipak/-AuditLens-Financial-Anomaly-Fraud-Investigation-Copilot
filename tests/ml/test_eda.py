"""Tests for exploratory data analysis (EDA) and leakage checks."""

from pathlib import Path

import pandas as pd

from ml.eda.analysis import (
    analyze_class_imbalance,
    analyze_feature_correlations,
    analyze_time_patterns,
    check_target_leakage,
    generate_eda_markdown,
)

SAMPLE_DIR = Path("./data/sample")


def test_analyze_class_imbalance():
    """Verify class imbalance calculation accurately identifies minority class."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    imb = analyze_class_imbalance(df)

    assert imb["total_records"] == 150
    assert imb["legitimate_count"] == 141
    assert imb["fraud_count"] == 9
    assert imb["fraud_percentage"] == 6.0
    assert "15.67:1" in imb["imbalance_ratio"]


def test_analyze_feature_correlations():
    """Verify correlation calculation correctly ranks predictive features."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    corr_info = analyze_feature_correlations(df)

    assert "all_target_correlations" in corr_info
    assert len(corr_info["all_target_correlations"]) == 30  # 30 input features
    assert len(corr_info["top_positive_features"]) > 0
    assert len(corr_info["top_negative_features"]) > 0


def test_check_target_leakage_clean_data():
    """Verify clean dataset triggers zero target leakage alarms."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    leakages = check_target_leakage(df)
    assert len(leakages) == 0


def test_check_target_leakage_detects_leaked_feature():
    """Verify synthetic perfectly correlated feature is flagged as a leakage hazard."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv").copy()
    # Create an artificial post-outcome leak (e.g. direct copy of target)
    df["suspicious_chargeback_flag"] = df["Class"]

    leakages = check_target_leakage(df)
    assert len(leakages) == 1
    assert leakages[0]["column"] == "suspicious_chargeback_flag"
    assert leakages[0]["absolute_correlation"] == 1.0


def test_analyze_time_patterns():
    """Verify time pattern analysis calculates hourly distribution."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    patterns = analyze_time_patterns(df)

    assert patterns["has_time"] is True
    assert patterns["span_hours"] > 0
    assert "fraud_by_hour" in patterns


def test_generate_eda_markdown():
    """Verify markdown report generation produces structured headers."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    report = generate_eda_markdown(df)

    assert "# AuditLens — Exploratory Data Analysis Report" in report
    assert "## 2. Class Imbalance Analysis" in report
    assert "## 4. Target Leakage & Post-Outcome Field Audit" in report
    assert "## 5. Temporal Patterns & Coverage" in report
