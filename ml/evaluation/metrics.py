"""Classification metric computation and threshold trade-off analysis."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification under class imbalance.

    Args:
        y_true: Ground truth binary array (0 = legit, 1 = fraud).
        y_prob: Continuous predicted probabilities for positive class (Class=1).
        threshold: Decision threshold for converting probabilities into binary predictions.

    Returns:
        dict[str, Any]: Complete metrics dictionary including confusion matrix, PR-AUC, and ROC-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Shape mismatch: y_true ({len(y_true)}) != y_prob ({len(y_prob)})"
        )

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_pos = tp + fn
    total_neg = tn + fp

    # Rates
    fpr = float(fp / total_neg) if total_neg > 0 else 0.0
    fnr = float(fn / total_pos) if total_pos > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(tn / total_neg) if total_neg > 0 else 0.0

    # Area under curve metrics
    # PR-AUC / Average Precision is the primary operational metric for imbalanced fraud
    pr_auc = float(average_precision_score(y_true, y_prob)) if total_pos > 0 else 0.0

    # ROC-AUC is secondary; can be over-optimistic under severe class imbalance
    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5

    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
        "sample_counts": {
            "total_samples": len(y_true),
            "actual_fraud": int(total_pos),
            "actual_legitimate": int(total_neg),
            "predicted_fraud": int(tp + fp),
        },
    }


def calculate_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate performance across an array of decision thresholds to analyze triage trade-offs."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]

    curve = []
    for t in thresholds:
        metrics = calculate_classification_metrics(y_true, y_prob, threshold=t)
        curve.append(
            {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positives": metrics["confusion_matrix"]["false_positives"],
                "false_negatives": metrics["confusion_matrix"]["false_negatives"],
                "predicted_fraud_count": metrics["sample_counts"]["predicted_fraud"],
            }
        )
    return curve


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    target_value: float = 0.8,
) -> dict[str, Any]:
    """Find the operational decision threshold based on an explicit analyst objective.

    Criteria:
    - 'f1': Maximizes the F1 harmonic mean between precision and recall.
    - 'recall_at_precision': Finds the lowest threshold attaining at least target_value precision.
    - 'precision_at_recall': Finds the highest threshold capturing at least target_value recall.
    """
    threshold_range = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -1.0
    best_metrics: dict[str, Any] = {}

    for t in threshold_range:
        m = calculate_classification_metrics(y_true, y_prob, threshold=float(t))

        if criterion == "f1":
            score = m["f1"]
            if score > best_score:
                best_score = score
                best_t = float(t)
                best_metrics = m

        elif criterion == "recall_at_precision":
            # Must satisfy precision >= target_value, then maximize recall
            if m["precision"] >= target_value:
                score = m["recall"]
                if score > best_score:
                    best_score = score
                    best_t = float(t)
                    best_metrics = m

        elif criterion == "precision_at_recall":
            # Must satisfy recall >= target_value, then maximize precision
            if m["recall"] >= target_value:
                score = m["precision"]
                if score > best_score:
                    best_score = score
                    best_t = float(t)
                    best_metrics = m

    if not best_metrics:
        # Fallback to threshold=0.5 if constraint could not be met
        best_t = 0.5
        best_metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    return {
        "criterion": criterion,
        "target_value": target_value if criterion != "f1" else None,
        "selected_threshold": round(best_t, 3),
        "metrics_at_selected_threshold": best_metrics,
    }
