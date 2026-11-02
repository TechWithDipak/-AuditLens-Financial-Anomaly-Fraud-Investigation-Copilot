"""Abstract base model interfaces and prediction data contracts."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PredictionRecord:
    """Individual transaction prediction contract.

    Architectural requirement:
    Anomaly scores (statistical deviation from normal behavior) and
    supervised fraud probabilities (estimated likelihood of confirmed fraud)
    must remain strictly separate fields to prevent misleading analysts.
    """

    transaction_id: int | str
    anomaly_score: float  # Normalized anomaly score [0.0, 1.0], higher = more atypical
    is_anomaly: bool  # Flag indicating statistical outlier
    fraud_probability: float | None = None  # Supervised fraud probability [0.0, 1.0]
    predicted_class: int | None = None  # Binary class label (0=normal, 1=fraud)
    anomaly_model_version: str = "unknown"
    classifier_model_version: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction record to dictionary."""
        return asdict(self)


@dataclass
class ModelMetadata:
    """Audit metadata tracking provenance, hyperparameters, and environment."""

    model_name: str
    model_version: str
    algorithm: str
    model_type: str  # 'anomaly_detection' or 'supervised_classification'
    hyperparameters: dict[str, Any]
    feature_names: list[str]
    random_state: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    training_samples: int = 0
    training_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseFraudModel(ABC):
    """Abstract base class for all AuditLens ML models."""

    def __init__(self, model_name: str, model_version: str, random_state: int = 42):
        self.model_name = model_name
        self.model_version = model_version
        self.random_state = random_state
        self.is_fitted: bool = False
        self.metadata: ModelMetadata | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseFraudModel":
        """Fit model on feature matrix."""

    @abstractmethod
    def save(self, filepath: str | Path) -> Path:
        """Persist model and metadata to disk."""

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path) -> "BaseFraudModel":
        """Load fitted model artifact from disk."""
