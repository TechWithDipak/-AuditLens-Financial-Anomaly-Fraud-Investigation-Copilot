"""Threshold Tuning & Operational Trade-off Analysis."""

from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from app.schemas.evaluation import ThresholdCurveResponse, ThresholdPoint


class ThresholdService:
    @staticmethod
    def compute_threshold_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "XGBoost_Classifier",
    ) -> ThresholdCurveResponse:
        """Compute precision-recall-F1 operational trade-offs across decision thresholds."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        points: List[ThresholdPoint] = []

        best_f1 = -1.0
        best_f1_thresh = 0.5
        high_recall_thresh = 0.1  # Fallback

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            points.append(
                ThresholdPoint(
                    threshold=th,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                    true_positive=tp,
                    false_positive=fp,
                    false_negative=fn,
                    true_negative=tn,
                )
            )

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = th

            # Track highest threshold that still guarantees >= 90% recall
            if rec >= 0.90:
                high_recall_thresh = th

        return ThresholdCurveResponse(
            model_name=model_name,
            points=points,
            optimal_f1_threshold=best_f1_thresh,
            high_recall_threshold=high_recall_thresh,
        )
