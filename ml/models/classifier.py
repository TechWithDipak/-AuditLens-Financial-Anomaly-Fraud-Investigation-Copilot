"""Supervised fraud classification baseline using Logistic Regression."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.models.base import BaseFraudModel, ModelMetadata


class LogisticRegressionFraudClassifier(BaseFraudModel):
    """Linear supervised baseline estimating fraud probability from labeled examples.

    Uses balanced class weighting to handle extreme class imbalance.
    Predicts calibrated continuous fraud probabilities: P(Fraud=1 | X).
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        model_version: str = "logreg_v0.1.0",
    ):
        super().__init__(
            model_name="Logistic Regression Fraud Classifier",
            model_version=model_version,
            random_state=random_state,
        )
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight

        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionFraudClassifier":
        """Fit supervised classifier on preprocessed features X and ground-truth binary labels y.

        Args:
            X: 2D numeric array of preprocessed features.
            y: 1D binary target array (0 = legitimate, 1 = fraud).
        """
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

        # Extract training metrics and coefficients
        coefficients = {
            f"feat_{i}": round(float(c), 4) for i, c in enumerate(self.model.coef_[0])
        }

        self.metadata = ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            algorithm="LogisticRegression",
            model_type="supervised_classification",
            hyperparameters={
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "class_weight": str(self.class_weight),
            },
            feature_names=[f"feat_{i}" for i in range(X.shape[1])],
            random_state=self.random_state,
            training_samples=int(X.shape[0]),
            training_metrics={
                "intercept": round(float(self.model.intercept_[0]), 4),
                "n_iter": int(self.model.n_iter_[0]),
                "class_distribution": {
                    int(k): int(v)
                    for k, v in zip(*np.unique(y, return_counts=True), strict=False)
                },
                "coefficients": coefficients,
            },
        )
        return self

    def predict_fraud_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Estimate continuous probability that each transaction is fraudulent: P(Class=1 | X).

        Returns:
            np.ndarray: Array of probabilities in range [0.0, 1.0].
        """
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
    def load(cls, filepath: str | Path) -> "LogisticRegressionFraudClassifier":
        """Load fitted classifier artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded
