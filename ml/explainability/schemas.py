"""Typed contracts and data structures for model explainability."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class FeatureContribution:
    """Individual feature attribution towards an instance prediction."""

    feature_name: str
    transformed_value: float
    contribution: float  # Signed attribution: positive = increases fraud score, negative = decreases
    direction: str  # 'increases_risk', 'decreases_risk', or 'neutral'
    raw_value: float | None = None
    relative_magnitude: float = 0.0  # Normalized relative importance [0.0, 1.0]
    plain_text_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionExplanation:
    """Complete analyst-facing explainability report for an individual transaction.

    LEGAL & OPERATIONAL NOTICE:
    Feature contributions explain why the statistical model assigned its score.
    They represent mathematical correlations, NOT legal proof of intent or causation.
    """

    transaction_id: str | int
    model_name: str
    model_version: str
    explanation_method: str  # e.g. 'shap_linear', 'shap_tree', 'fallback_attribution'
    score_type: str  # 'fraud_probability' or 'anomaly_score'
    prediction_score: float
    decision_threshold: float
    is_flagged: bool
    base_value: float  # Expected model output across background population
    top_contributions: list[FeatureContribution] = field(default_factory=list)
    all_contributions: dict[str, float] = field(default_factory=dict)
    summary_text: str = ""
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["top_contributions"] = [c.to_dict() for c in self.top_contributions]
        return res


@dataclass
class GlobalFeatureImportance:
    """Global feature importance summary across the entire model."""

    model_name: str
    model_version: str
    method: str
    feature_rankings: list[dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
