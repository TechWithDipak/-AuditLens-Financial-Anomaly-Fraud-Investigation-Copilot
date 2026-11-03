"""Pydantic schemas for model metadata and evaluation reports."""

from typing import Any

from pydantic import BaseModel, Field


class ModelInfoSchema(BaseModel):
    """Metadata regarding a single loaded model."""

    model_name: str
    model_version: str
    model_type: str  # unsupervised_anomaly or supervised_classifier
    description: str
    features: list[str]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class ModelMetadataResponse(BaseModel):
    """Active inference engine status, loaded models, and operational thresholds."""

    inference_engine_ready: bool
    operational_threshold: float
    feature_count: int
    models: dict[str, ModelInfoSchema]
    notice: str = "Decision-support advisory. Models produce likelihoods, not proof."


class ModelEvaluationResponse(BaseModel):
    """Consolidated benchmark and comparative model performance metrics."""

    generated_at: str | None = None
    best_model_name: str | None = None
    operational_threshold: float | None = None
    models_evaluated: list[dict[str, Any]] = Field(default_factory=list)
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

