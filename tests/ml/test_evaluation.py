"""Tests for model evaluation metrics, threshold selection, and comparisons."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.comparison import analyze_prediction_errors, evaluate_model_suite
from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_threshold_curve,
    find_optimal_threshold,
)
from ml.evaluation.reporter import save_evaluation_reports
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.models.random_forest import RandomForestFraudClassifier


def test_calculate_classification_metrics_known_example():
    """Verify metrics calculation against hand-verified truth matrix."""
    # 4 transactions: 2 legit, 2 fraud
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.9])

    # At threshold 0.5:
    # y_pred = [0, 1, 0, 1] -> TN=1, FP=1, FN=1, TP=1
    metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_metrics_handles_perfect_and_zero_predictions():
    """Verify metrics calculation gracefully handles edge cases."""
    y_true = np.array([0, 0, 1, 1])
    # Perfect predictions
    perfect_probs = np.array([0.0, 0.0, 1.0, 1.0])
    perf = calculate_classification_metrics(y_true, perfect_probs, threshold=0.5)
    assert perf["precision"] == 1.0
    assert perf["recall"] == 1.0
    assert perf["f1"] == 1.0
    assert perf["confusion_matrix"]["false_positives"] == 0
    assert perf["confusion_matrix"]["false_negatives"] == 0

    # Naive all-zero predictions
    zero_probs = np.array([0.0, 0.0, 0.0, 0.0])
    zero_m = calculate_classification_metrics(y_true, zero_probs, threshold=0.5)
    assert zero_m["precision"] == 0.0
    assert zero_m["recall"] == 0.0
    assert zero_m["f1"] == 0.0
    assert zero_m["confusion_matrix"]["false_negatives"] == 2


def test_calculate_threshold_curve_monotonicity():
    """Verify threshold curve calculates trade-offs across range."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.6, 0.9])

    curve = calculate_threshold_curve(y_true, y_prob, thresholds=[0.15, 0.5, 0.85])
    assert len(curve) == 3

    # As threshold increases, predicted fraud count should decrease or remain equal
    preds = [item["predicted_fraud_count"] for item in curve]
    assert preds[0] >= preds[1] >= preds[2]


def test_find_optimal_threshold_f1():
    """Verify threshold optimization selects the cutoff maximizing F1 score."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])

    optimal = find_optimal_threshold(y_true, y_prob, criterion="f1")
    # Cutoff between 0.4 and 0.7 yields 100% precision and 100% recall
    assert 0.4 <= optimal["selected_threshold"] <= 0.7
    assert optimal["metrics_at_selected_threshold"]["f1"] == 1.0


def test_random_forest_fraud_classifier():
    """Verify RandomForestFraudClassifier fits and predicts probabilities in [0.0, 1.0]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 10))
    y = np.zeros(60, dtype=int)
    y[:10] = 1

    rf = RandomForestFraudClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)
    assert rf.is_fitted is True

    probs = rf.predict_fraud_probabilities(X)
    assert len(probs) == 60
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "rf.joblib"
        rf.save(model_path)
        loaded = RandomForestFraudClassifier.load(model_path)
        loaded_probs = loaded.predict_fraud_probabilities(X)
        np.testing.assert_allclose(probs, loaded_probs, rtol=1e-6)


def test_evaluate_model_suite_integration():
    """Verify evaluate_model_suite evaluates multiple candidate models."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 15))
    y_train = np.zeros(100, dtype=int)
    y_train[:15] = 1

    X_test = rng.standard_normal((30, 15))
    y_test = np.zeros(30, dtype=int)
    y_test[:5] = 1

    iso = IsolationForestAnomalyDetector(n_estimators=20, random_state=42).fit(X_train)
    logreg = LogisticRegressionFraudClassifier(random_state=42).fit(X_train, y_train)

    comparison = evaluate_model_suite(
        X_test,
        y_test,
        models={"isolation_forest": iso, "logistic_regression": logreg},
        threshold=0.5,
    )

    assert "naive_majority" in comparison["models"]
    assert "isolation_forest" in comparison["models"]
    assert "logistic_regression" in comparison["models"]
    assert comparison["test_sample_count"] == 30


def test_analyze_prediction_errors_isolation():
    """Verify false positive and false negative analysis isolates error cases."""
    X_raw = pd.DataFrame(
        {
            "Time": [10.0, 20.0, 30.0, 40.0],
            "Amount": [50.0, 120.0, 15.0, 500.0],
            "V4": [0.1, 2.5, 0.2, 3.8],
        }
    )
    y_true = np.array([0, 0, 1, 1])
    # Case 1: true=0, prob=0.9 (FP)
    # Case 2: true=1, prob=0.1 (FN)
    y_prob = np.array([0.05, 0.9, 0.1, 0.95])

    errors = analyze_prediction_errors(X_raw, y_true, y_prob, threshold=0.5)

    assert errors["total_false_positives"] == 1
    assert errors["total_false_negatives"] == 1
    assert errors["top_false_positives"][0]["index"] == 1
    assert errors["top_false_negatives"][0]["index"] == 2


def test_save_evaluation_reports():
    """Verify markdown and JSON evaluation reports are correctly written."""
    dummy_comparison = {
        "test_sample_count": 20,
        "actual_fraud_count": 2,
        "models": {
            "logistic_regression": {
                "model_name": "Logistic Regression Baseline",
                "model_type": "supervised_classification",
                "default_metrics": {
                    "pr_auc": 0.85,
                    "roc_auc": 0.92,
                    "f1": 0.80,
                    "precision": 0.80,
                    "recall": 0.80,
                    "false_positive_rate": 0.05,
                    "false_negative_rate": 0.20,
                    "confusion_matrix": {
                        "true_positives": 2,
                        "false_positives": 1,
                        "true_negatives": 17,
                        "false_negatives": 0,
                    },
                },
                "optimal_threshold": {
                    "selected_threshold": 0.45,
                    "metrics_at_selected_threshold": {
                        "precision": 0.85,
                        "recall": 0.9,
                        "f1": 0.87,
                    },
                },
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = Path(tmpdir) / "report.md"
        json_file = Path(tmpdir) / "report.json"

        save_evaluation_reports(
            dummy_comparison, output_md_path=md_file, output_json_path=json_file
        )

        assert md_file.exists()
        assert json_file.exists()
        content = md_file.read_text(encoding="utf-8")
        assert "# AuditLens — Model Evaluation & Benchmark Comparison Report" in content
        assert "| **Logistic Regression Baseline** |" in content
