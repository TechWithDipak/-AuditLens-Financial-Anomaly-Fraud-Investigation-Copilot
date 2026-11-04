"""Supporting Unsupervised Algorithm: Isolation Forest Anomaly Detection."""

import numpy as np
from sklearn.ensemble import IsolationForest

from app.schemas.evaluation import IsolationForestRunResponse


class IsolationForestService:
    @staticmethod
    def run_isolation_forest(
        X: np.ndarray,
        contamination: float = 0.01,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> IsolationForestRunResponse:
        """Fit Isolation Forest and compute tree partition depth anomaly scores."""
        n_samples, n_features = X.shape
        if n_samples < 10:
            raise ValueError("Sample count must be at least 10 for Isolation Forest.")
        if not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or infinite values.")

        iso = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        )
        preds = iso.fit_predict(X)  # 1: inliers, -1: outliers
        scores = iso.decision_function(X)  # Lower values indicate higher abnormality

        anomalous_indices = np.where(preds == -1)[0].tolist()

        return IsolationForestRunResponse(
            contamination=contamination,
            n_estimators=n_estimators,
            anomalous_sample_count=len(anomalous_indices),
            anomaly_scores_preview=[round(float(s), 4) for s in scores[:20]],
            anomaly_indices_preview=anomalous_indices[:20],
        )
