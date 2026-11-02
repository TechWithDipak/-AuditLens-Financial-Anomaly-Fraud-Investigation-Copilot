"""Unsupervised anomaly detection baseline using Isolation Forest."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ml.models.base import BaseFraudModel, ModelMetadata


class IsolationForestAnomalyDetector(BaseFraudModel):
    """Unsupervised anomaly detector identifying transactions that deviate from learned distributions.

    IMPORTANT ARCHITECTURAL CAVEAT:
    An anomaly is a statistical outlier, NOT confirmed fraud. High anomaly scores
    indicate rare or atypical feature combinations requiring analyst inspection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: str | float = "auto",
        random_state: int = 42,
        model_version: str = "iso_forest_v0.1.0",
    ):
        super().__init__(
            model_name="Isolation Forest Anomaly Detector",
            model_version=model_version,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._score_min: float | None = None
        self._score_max: float | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "IsolationForestAnomalyDetector":
        """Fit Isolation Forest strictly on unlabeled feature matrix X.

        Args:
            X: 2D numeric array of preprocessed features.
            y: Ignored (strictly unsupervised).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Input X must be a 2D array, got ndim={X.ndim}")

        if X.shape[0] == 0:
            raise ValueError("Cannot fit model on empty array X")

        # Fit scikit-learn Isolation Forest
        self.model.fit(X)

        # Calibrate baseline score range for normalization
        # In scikit-learn, score_samples outputs negative values; lower = more anomalous
        raw_scores = -self.model.score_samples(X)
        self._score_min = float(np.min(raw_scores))
        self._score_max = float(np.max(raw_scores))

        self.is_fitted = True
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            algorithm="IsolationForest",
            model_type="anomaly_detection",
            hyperparameters={
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
            },
            feature_names=[f"feat_{i}" for i in range(X.shape[1])],
            random_state=self.random_state,
            training_samples=int(X.shape[0]),
            training_metrics={
                "score_min": self._score_min,
                "score_max": self._score_max,
                "score_mean": float(np.mean(raw_scores)),
            },
        )
        return self

    def score_transactions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalized anomaly scores and outlier prediction flags.

        Args:
            X: 2D feature matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - anomaly_scores: Float array [0.0, 1.0] where 1.0 = most atypical/isolated.
                - is_anomaly: Boolean array where True indicates outlier based on contamination threshold.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before calling score_transactions()."
            )

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Raw scores: scikit-learn score_samples produces negative values; invert so higher = more anomalous
        raw_scores = -self.model.score_samples(X)

        # Normalize to [0.0, 1.0] using fitted training bounds with clip
        denom = (
            (self._score_max - self._score_min)
            if (
                self._score_max
                and self._score_min
                and self._score_max > self._score_min
            )
            else 1.0
        )
        normalized_scores = np.clip(
            (raw_scores - (self._score_min or 0.0)) / denom, 0.0, 1.0
        )
        normalized_scores = np.round(normalized_scores, 4)

        # Binary outlier flag: predict() returns -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X)
        is_anomaly = raw_preds == -1

        return normalized_scores, is_anomaly

    def save(self, filepath: str | Path) -> Path:
        """Persist model artifact and metadata JSON to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

        meta_path = path.with_suffix(".json")
        if self.metadata:
            meta_path.write_text(
                json.dumps(self.metadata.to_dict(), indent=2), encoding="utf-8"
            )

        return path

    @classmethod
    def load(cls, filepath: str | Path) -> "IsolationForestAnomalyDetector":
        """Load fitted anomaly detector artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded
