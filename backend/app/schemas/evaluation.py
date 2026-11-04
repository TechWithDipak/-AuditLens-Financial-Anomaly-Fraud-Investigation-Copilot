"""Schemas for Integrated Evaluation, SHAP Explainability, and Threshold Tuning."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureAttribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    contribution: str  # "increases_risk" or "decreases_risk"


class SHAPExplanationResponse(BaseModel):
    model_name: str
    model_version: str = "1.0.0"
    base_value: float
    predicted_probability: float
    decision_threshold: float = 0.5
    prediction_label: int
    top_attributions: List[FeatureAttribution]
    raw_shap_values: Dict[str, float]
    caveat: str = "SHAP values measure statistical model feature contribution, not proof of causation or legal liability."


class ThresholdPoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1_score: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


class ThresholdCurveResponse(BaseModel):
    model_name: str
    points: List[ThresholdPoint]
    optimal_f1_threshold: float
    high_recall_threshold: float  # e.g., threshold achieving >= 90% recall
    message: str = "Threshold tuning balances investigator review capacity against fraud leakage."


class IsolationForestRunRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    contamination: float = Field(default=0.01, gt=0.0, lt=0.5)
    n_estimators: int = 100
    random_state: int = 42


class IsolationForestRunResponse(BaseModel):
    model_name: str = "IsolationForest"
    contamination: float
    n_estimators: int
    anomalous_sample_count: int
    anomaly_scores_preview: List[float]
    anomaly_indices_preview: List[int]
    message: str = "Isolation Forest anomaly score is an unsupervised metric distinct from supervised fraud probability."


class IntegratedTransactionScores(BaseModel):
    transaction_id: str
    fraud_probability: Optional[float] = Field(None, description="Supervised probability [0, 1]")
    gmm_log_likelihood: Optional[float] = Field(
        None, description="Density rarity under mixture (-inf, 0]"
    )
    kmeans_cluster_id: Optional[int] = Field(
        None, description="Peer grouping cluster index [0..K-1]"
    )
    isolation_forest_score: Optional[float] = Field(
        None, description="Unsupervised isolation score [-1, 1]"
    )
    caveat: str = "Heterogeneous scores represent distinct mathematical semantics; they are not conflated into a single composite score."
