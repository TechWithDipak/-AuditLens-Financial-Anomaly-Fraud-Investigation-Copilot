"""Pydantic Schemas for AI Investigation Report Assistant."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class ObservedTransactionFact(BaseModel):
    transaction_index: int
    amount: float
    time_seconds: Optional[float] = None
    merchant_category: Optional[str] = None
    ground_truth_class: Optional[int] = None
    context: Optional[str] = None


class ModelScoreBreakdown(BaseModel):
    supervised_model_name: str
    fraud_probability: float = Field(..., description="Calibrated supervised probability [0, 1]")
    gmm_log_likelihood: Optional[float] = Field(None, description="Unsupervised density rarity")
    kmeans_cluster_id: Optional[int] = Field(None, description="Spending behavior peer cluster")
    hmm_decoded_state: Optional[str] = Field(
        None, description="Markovian sequence behavioral state"
    )
    isolation_forest_score: Optional[float] = Field(
        None, description="Tree partition depth anomaly score"
    )


class ExplanationSignal(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"


class CitedRegulatoryPassage(BaseModel):
    doc_id: str
    title: str
    issuing_body: str
    section: str
    source_url: str
    similarity_score: float
    excerpt: str
    has_injection_risk: bool = False


class InvestigationReportDraft(BaseModel):
    report_id: str
    case_id: Optional[int] = None
    report_title: str
    case_summary: str
    observed_facts: List[ObservedTransactionFact]
    model_scores: ModelScoreBreakdown
    explanation_signals: List[ExplanationSignal]
    regulatory_citations: List[CitedRegulatoryPassage]
    missing_evidence_and_uncertainty: List[str]
    suggested_follow_up_questions: List[str]
    is_ai_generated: bool = True
    mandatory_advisory_label: str = (
        "AI-GENERATED INVESTIGATION DRAFT REQUIRING HUMAN ANALYST REVIEW. "
        "Under no circumstances should model outputs or drafted narratives be treated as definitive proof "
        "or used to automate adverse or legal action."
    )
    verification_passed: bool
    verification_warnings: List[str] = []
    generation_provider: str  # "offline_rule_engine", "openai", "gemini"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerateDraftRequest(BaseModel):
    transaction_index: int = Field(..., description="Transaction index in dataset to investigate")
    case_id: Optional[int] = Field(None, description="Optional linked investigation case ID")
    analyst_hypothesis: Optional[str] = Field(
        None, description="Optional hypothesis or preliminary notes"
    )
    provider: Optional[str] = Field(
        "offline", description="Provider adapter: 'offline', 'gemini', 'openai'"
    )


class VerifyEvidenceRequest(BaseModel):
    draft: InvestigationReportDraft
    actual_amount: float
    actual_fraud_probability: float
