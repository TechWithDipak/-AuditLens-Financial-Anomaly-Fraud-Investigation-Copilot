"""Schemas for Unit 4 Hidden Markov Model (HMM) algorithms."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HMMForwardRequest(BaseModel):
    observations: List[int] = Field(
        ..., min_length=1, description="Discrete observation symbol indices"
    )
    initial_probs: Optional[List[float]] = None
    transition_matrix: Optional[List[List[float]]] = None
    emission_matrix: Optional[List[List[float]]] = None


class HMMForwardResponse(BaseModel):
    sequence_length: int
    log_likelihood: float
    likelihood: float
    scaled_factors: List[float]
    is_unusual_sequence: bool
    message: str = (
        "Sequence likelihood reflects transition consistency, not a fraud classification."
    )


class HMMViterbiRequest(BaseModel):
    observations: List[int] = Field(
        ..., min_length=1, description="Discrete observation symbol indices"
    )
    initial_probs: Optional[List[float]] = None
    transition_matrix: Optional[List[List[float]]] = None
    emission_matrix: Optional[List[List[float]]] = None


class HMMViterbiResponse(BaseModel):
    sequence_length: int
    best_path: List[int]
    best_path_log_prob: float
    state_labels: List[str]
    message: str = (
        "Inferred hidden states are statistical behavioural patterns, not fraud verdicts."
    )


class AccountTimelinePoint(BaseModel):
    step: int
    timestamp: int
    amount: float
    tx_type: str
    observed_symbol: int
    inferred_state: int
    inferred_state_label: str
    is_fraud: int
    # Frontend aliases
    merchant_category: Optional[int] = None
    is_anomaly: Optional[int] = None
    decoded_state: Optional[int] = None
    decoded_state_name: Optional[str] = None


class AccountTimelineResponse(BaseModel):
    customer_id: str
    sequence_length: int
    sequence_log_likelihood: float
    timeline: List[AccountTimelinePoint]
    state_distribution: Dict[str, int]
    # Frontend aliases
    account_id: Optional[str] = None
    total_steps: Optional[int] = None
    anomaly_count: Optional[int] = None
