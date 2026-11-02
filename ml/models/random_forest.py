"""Supervised Random Forest fraud classifier for model comparison."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.models.base import BaseFraudModel, ModelMetadata


class RandomForestFraudClassifier(BaseFraudModel):
    """Non-linear ensemble classifier capturing complex feature interactions.

    Uses balanced class weighting to handle severe fraud imbalance.
    Predicts calibrated continuous fraud probabilities: P(Fraud=1 | X).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 4,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        model_version: str = "rf_v0.1.0",
    ):
        super().__init__(
            model_name="Random Forest Fraud Classifier",
            model_version=model_version,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestFraudClassifier":
        """Fit Random Forest on preprocessed features X and binary labels y."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D, got ndim={X.ndim}")

        if y.ndim != 1:
            y = y.ravel()

        if len(X) != len(y):
            raise ValueError(
                f"Shape mismatch: X has {len(X)} rows, y has {len(y)} elements"
            )

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Supervised model requires both classes (0 and 1) for training. Got: {unique_labels}"
            )

        self.model.fit(X, y)
        self.is_fitted = True

        feature_importances = {
            f"feat_{i}": round(float(imp), 4)
            for i, imp in enumerate(self.model.feature_importances_)
        }

        self.metadata = ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            algorithm="RandomForestClassifier",
            model_type="supervised_classification",
            hyperparameters={
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "class_weight": str(self.class_weight),
            },
            feature_names=[f"feat_{i}" for i in range(X.shape[1])],
            random_state=self.random_state,
            training_samples=int(X.shape[0]),
            training_metrics={
                "n_features": int(X.shape[1]),
                "feature_importances": feature_importances,
            },
        )
        return self

    def predict_fraud_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Estimate continuous probability that each transaction is fraud."""
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before calling predict_fraud_probabilities()."
            )

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        probabilities = self.model.predict_proba(X)[:, 1]
        return np.round(probabilities, 4)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classify transactions based on an operational decision threshold."""
        probabilities = self.predict_fraud_probabilities(X)
        return (probabilities >= threshold).astype(int)

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
    def load(cls, filepath: str | Path) -> "RandomForestFraudClassifier":
        """Load fitted classifier artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded
