"""Standardized Evaluation Metrics for Imbalanced Fraud Classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.schemas.supervised import ConfusionMatrix, EvaluationMetrics


def calculate_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> EvaluationMetrics:
    """Compute rigorous imbalanced evaluation metrics prioritizing PR-AUC and F1 over accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = round(fp / max(fp + tn, 1), 4)
    fnr = round(fn / max(fn + tp, 1), 4)

    # Primary metrics: Precision, Recall, F1, PR-AUC
    prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    # PR-AUC / Average Precision (crucial for imbalanced fraud)
    try:
        pr_auc = round(float(average_precision_score(y_true, y_prob)), 4)
    except Exception:
        pr_auc = 0.0

    # ROC-AUC as secondary metric
    try:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        roc_auc = 0.5

    acc = round(float(accuracy_score(y_true, y_pred)), 4)

    return EvaluationMetrics(
        precision=prec,
        recall=rec,
        f1_score=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        accuracy=acc,
        confusion_matrix=ConfusionMatrix(
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            true_positive=tp,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
        ),
    )
