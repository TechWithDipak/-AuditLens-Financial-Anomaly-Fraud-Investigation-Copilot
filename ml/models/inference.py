"""Inference service executing dual scoring (anomaly and supervised fraud probability)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.data.schema import REQUIRED_FEATURE_COLUMNS
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.base import PredictionRecord
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.preprocessing.pipeline import AuditLensPreprocessor


class AuditLensInferenceEngine:
    """Production-grade scoring engine executing preprocessing, anomaly detection, and fraud classification.

    Separates anomaly scores (atypical pattern indicators) and supervised fraud
    probabilities (calibrated probability estimates).
    """

    def __init__(
        self,
        preprocessor: AuditLensPreprocessor,
        anomaly_model: IsolationForestAnomalyDetector | None = None,
        classifier_model: LogisticRegressionFraudClassifier | None = None,
    ):
        self.preprocessor = preprocessor
        self.anomaly_model = anomaly_model
        self.classifier_model = classifier_model

    @classmethod
    def load_from_directory(
        cls, model_dir: str | Path = "./models"
    ) -> "AuditLensInferenceEngine":
        """Instantiate engine by loading serialized artifacts from an artifact directory."""
        dir_path = Path(model_dir)

        prep_file = dir_path / "preprocessor.joblib"
        if not prep_file.exists():
            raise FileNotFoundError(f"Preprocessor artifact missing in: {dir_path}")
        preprocessor = AuditLensPreprocessor.load(prep_file)

        anomaly_file = dir_path / "isolation_forest.joblib"
        anomaly_model = (
            IsolationForestAnomalyDetector.load(anomaly_file)
            if anomaly_file.exists()
            else None
        )

        clf_file = dir_path / "logistic_regression.joblib"
        classifier_model = (
            LogisticRegressionFraudClassifier.load(clf_file)
            if clf_file.exists()
            else None
        )

        return cls(
            preprocessor=preprocessor,
            anomaly_model=anomaly_model,
            classifier_model=classifier_model,
        )

    def score_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> list[PredictionRecord]:
        """Score a DataFrame of transactions.

        Args:
            df: DataFrame containing required feature columns.
            threshold: Probability decision threshold for supervised classification.

        Returns:
            list[PredictionRecord]: Per-transaction scored records with decoupled scores.
        """
        # Validate required columns
        missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required features: {missing}")

        # Drop target if present to prevent leakage
        feature_df = df[REQUIRED_FEATURE_COLUMNS].copy()

        # Transform features
        X_trans = self.preprocessor.transform(feature_df)

        # 1. Anomaly Scoring
        if self.anomaly_model:
            anomaly_scores, is_anomaly_flags = self.anomaly_model.score_transactions(
                X_trans
            )
            anomaly_version = self.anomaly_model.model_version
        else:
            anomaly_scores = np.zeros(len(df))
            is_anomaly_flags = np.zeros(len(df), dtype=bool)
            anomaly_version = "none"

        # 2. Supervised Fraud Probability
        if self.classifier_model:
            fraud_probs = self.classifier_model.predict_fraud_probabilities(X_trans)
            predicted_classes = (fraud_probs >= threshold).astype(int)
            clf_version = self.classifier_model.model_version
        else:
            fraud_probs = [None] * len(df)
            predicted_classes = [None] * len(df)
            clf_version = None

        # Build prediction records
        records: list[PredictionRecord] = []
        for idx in range(len(df)):
            tx_id = df.index[idx]
            rec = PredictionRecord(
                transaction_id=str(tx_id),
                anomaly_score=float(anomaly_scores[idx]),
                is_anomaly=bool(is_anomaly_flags[idx]),
                fraud_probability=float(fraud_probs[idx])
                if fraud_probs[idx] is not None
                else None,
                predicted_class=int(predicted_classes[idx])
                if predicted_classes[idx] is not None
                else None,
                anomaly_model_version=anomaly_version,
                classifier_model_version=clf_version,
            )
            records.append(rec)

        return records

    def score_single(
        self,
        transaction: dict[str, Any] | pd.Series,
        threshold: float = 0.5,
    ) -> PredictionRecord:
        """Score a single transaction."""
        df = pd.DataFrame([transaction])
        return self.score_batch(df, threshold=threshold)[0]
