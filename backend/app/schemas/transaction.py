"""Pydantic schemas for transactions, scoring requests, and explainability."""

from datetime import datetime

from pydantic import BaseModel, Field


class TransactionBase(BaseModel):
    """Core transaction fields."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    time: float = Field(..., description="Seconds elapsed since reference epoch")
    amount: float = Field(
        ..., ge=0.0, description="Transaction monetary amount (must be non-negative)"
    )


class TransactionCreate(TransactionBase):
    """Payload for creating or ingesting a transaction."""

    features: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary containing PCA features V1-V28",
    )
    label: int | None = Field(
        None, ge=0, le=1, description="Optional ground truth class (0 or 1)"
    )


class TransactionResponse(TransactionBase):
    """Analyst-facing transaction record with dual scores and metadata."""

    id: int
    features: dict[str, float] = Field(default_factory=dict)
    anomaly_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Unsupervised outlier score [0.0, 1.0]"
    )
    fraud_probability: float | None = Field(
        None, ge=0.0, le=1.0, description="Supervised fraud probability [0.0, 1.0]"
    )
    is_flagged: bool = Field(
        False, description="Whether transaction exceeds operational review threshold"
    )
    score_type: str = Field(
        "unscored", description="Type of scoring applied (supervised/anomaly/dual)"
    )
    label: int | None = Field(None, description="Ground truth label if known")
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Paginated collection of transaction records."""

    items: list[TransactionResponse]
    total: int = Field(..., description="Total matching transactions in database")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., description="Total available pages")


class TransactionScoreRequest(BaseModel):
    """Request payload to score arbitrary transaction feature values."""

    transaction_id: str = Field(
        "TEMP_TX_001", description="Identifier for this scoring query"
    )
    time: float = Field(0.0, description="Seconds elapsed")
    amount: float = Field(..., ge=0.0, description="Monetary transaction amount")
    features: dict[str, float] = Field(
        ...,
        description="Map of feature names to float values (must contain V1 through V28)",
    )


class TransactionScoreResponse(BaseModel):
    """Scoring evaluation result providing separate anomaly and fraud probabilities."""

    transaction_id: str
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Unsupervised anomaly severity (statistical distance)",
    )
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Supervised probability of fraud"
    )
    is_flagged: bool = Field(
        ..., description="True if fraud_probability exceeds review threshold"
    )
    decision_threshold: float = Field(
        ..., description="Active operational triage threshold"
    )
    score_type: str = "dual_scoring"
    model_version_anomaly: str
    model_version_classifier: str
    notice: str = (
        "Advisory decision support. Does not constitute autonomous freeze action."
    )


class FeatureContributionSchema(BaseModel):
    """Feature attribution item for explainability."""

    feature_name: str
    transformed_value: float
    contribution: float
    direction: str  # increases_risk, decreases_risk, neutral
    raw_value: float | None = None
    relative_magnitude: float = 0.0
    plain_text_description: str = ""


class TransactionExplanationResponse(BaseModel):
    """Complete explainability report for an individual transaction."""

    transaction_id: str
    model_name: str
    model_version: str
    explanation_method: str
    score_type: str
    prediction_score: float
    decision_threshold: float
    is_flagged: bool
    base_value: float
    top_contributions: list[FeatureContributionSchema]
    all_contributions: dict[str, float]
    summary_text: str
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    timestamp: str
