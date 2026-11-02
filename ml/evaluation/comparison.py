"""Comparative model benchmarking and error pattern analysis."""

from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_threshold_curve,
    find_optimal_threshold,
)
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.models.random_forest import RandomForestFraudClassifier


def evaluate_model_suite(
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict[str, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate multiple fraud detection models on a held-out test split.

    Args:
        X_test: Preprocessed test features.
        y_test: Ground truth binary target (0=legit, 1=fraud).
        models: Dictionary of models, e.g. {'logistic_regression': ..., 'random_forest': ..., 'isolation_forest': ...}
        threshold: Standard operational decision threshold.

    Returns:
        dict[str, Any]: Multi-model comparison report.
    """
    y_test = np.asarray(y_test).astype(int)
    comparison: dict[str, Any] = {
        "test_sample_count": len(y_test),
        "actual_fraud_count": int(np.sum(y_test)),
        "models": {},
    }

    # 1. Naive Baseline (predicts all legitimate class 0)
    naive_probs = np.zeros(len(y_test))
    comparison["models"]["naive_majority"] = {
        "model_name": "Naive Majority Baseline",
        "description": "Always predicts non-fraud (Class=0). Demonstrates why accuracy is misleading.",
        "metrics": calculate_classification_metrics(y_test, naive_probs, threshold=0.5),
    }

    # 2. Evaluate supplied models
    for key, model in models.items():
        if isinstance(model, IsolationForestAnomalyDetector):
            scores, _is_anomaly = model.score_transactions(X_test)
            metrics = calculate_classification_metrics(
                y_test, scores, threshold=threshold
            )
            opt_thresh = find_optimal_threshold(y_test, scores, criterion="f1")
            curve = calculate_threshold_curve(y_test, scores)

            comparison["models"][key] = {
                "model_name": model.model_name,
                "version": model.model_version,
                "model_type": "anomaly_detection",
                "default_metrics": metrics,
                "optimal_threshold": opt_thresh,
                "threshold_curve": curve,
            }

        elif isinstance(
            model, (LogisticRegressionFraudClassifier, RandomForestFraudClassifier)
        ):
            probs = model.predict_fraud_probabilities(X_test)
            metrics = calculate_classification_metrics(
                y_test, probs, threshold=threshold
            )
            opt_thresh = find_optimal_threshold(y_test, probs, criterion="f1")
            curve = calculate_threshold_curve(y_test, probs)

            comparison["models"][key] = {
                "model_name": model.model_name,
                "version": model.model_version,
                "model_type": "supervised_classification",
                "default_metrics": metrics,
                "optimal_threshold": opt_thresh,
                "threshold_curve": curve,
            }

    return comparison


def analyze_prediction_errors(
    X_raw: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Isolate and analyze False Positives and False Negatives for triage review."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    # Rank False Positives by highest predicted probability
    fp_sorted = (
        fp_indices[np.argsort(y_prob[fp_indices])[::-1]] if len(fp_indices) > 0 else []
    )
    # Rank False Negatives by lowest predicted probability (most severe misses)
    fn_sorted = (
        fn_indices[np.argsort(y_prob[fn_indices])] if len(fn_indices) > 0 else []
    )

    top_fp = []
    for idx in fp_sorted[:top_n]:
        row_dict = X_raw.iloc[idx].to_dict() if isinstance(X_raw, pd.DataFrame) else {}
        top_fp.append(
            {
                "index": int(idx),
                "predicted_probability": round(float(y_prob[idx]), 4),
                "actual_class": 0,
                "sample_features": {
                    k: round(float(v), 2)
                    for k, v in row_dict.items()
                    if k in ["Time", "Amount", "V4", "V11", "V14"]
                },
            }
        )

    top_fn = []
    for idx in fn_sorted[:top_n]:
        row_dict = X_raw.iloc[idx].to_dict() if isinstance(X_raw, pd.DataFrame) else {}
        top_fn.append(
            {
                "index": int(idx),
                "predicted_probability": round(float(y_prob[idx]), 4),
                "actual_class": 1,
                "sample_features": {
                    k: round(float(v), 2)
                    for k, v in row_dict.items()
                    if k in ["Time", "Amount", "V4", "V11", "V14"]
                },
            }
        )

    return {
        "threshold": threshold,
        "total_false_positives": len(fp_indices),
        "total_false_negatives": len(fn_indices),
        "top_false_positives": top_fp,
        "top_false_negatives": top_fn,
    }
