"""Schemas for Unit 5 Supervised & Ensemble Algorithms (CART, XGBoost, Ensemble)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConfusionMatrix(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    false_positive_rate: float
    false_negative_rate: float


class EvaluationMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    pr_auc: float  # Average Precision (primary metric for imbalanced data)
    roc_auc: float
    accuracy: float  # Reported as secondary, but not primary
    confusion_matrix: ConfusionMatrix


class SupervisedTrainRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class SupervisedTrainResponse(BaseModel):
    model_name: str
    algorithm: str
    version: str = "1.0.0"
    hyperparameters: Dict[str, Any]
    train_samples: int
    test_samples: int
    fraud_count_test: int
    metrics: EvaluationMetrics
    feature_importances: Dict[str, float]
    rules_text: Optional[str] = None
    artifact_path: Optional[str] = None
    message: str = "Supervised fraud probability estimates require human analyst verification."


class ModelComparisonItem(BaseModel):
    model_name: str
    algorithm_type: str
    pr_auc: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float


class ModelComparisonResponse(BaseModel):
    test_sample_count: int
    test_fraud_count: int
    comparison: List[ModelComparisonItem]
    best_model_by_prauc: str
