"""Pydantic Schemas for Case Management and Audit Logging."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CaseTransactionBase(BaseModel):
    transaction_index: int = Field(..., description="Index/row identifier of the transaction")
    amount: float = Field(..., description="Transaction monetary amount")
    fraud_probability: Optional[float] = Field(
        None, description="Supervised fraud probability if evaluated"
    )
    anomaly_score: Optional[float] = Field(
        None, description="Unsupervised anomaly score if evaluated"
    )
    added_reason: Optional[str] = Field(
        None, max_length=255, description="Brief justification for linking"
    )


class CaseTransactionCreate(CaseTransactionBase):
    pass


class CaseTransactionResponse(CaseTransactionBase):
    id: int
    case_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CaseNoteCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Analyst note or rationale")
    author: str = Field(default="analyst", max_length=128, description="Author identifier")


class CaseNoteResponse(BaseModel):
    id: int
    case_id: int
    author: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CaseBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Case title or subject")
    description: Optional[str] = Field(None, description="Detailed case background or hypothesis")
    priority: str = Field(
        default="MEDIUM", description="Case priority: LOW, MEDIUM, HIGH, CRITICAL"
    )
    assigned_analyst: str = Field(default="analyst_default", max_length=128)


class CaseCreate(CaseBase):
    initial_transactions: Optional[List[CaseTransactionCreate]] = Field(
        default=None, description="Optional transactions to link upon case creation"
    )


class CaseStatusUpdate(BaseModel):
    status: str = Field(
        ..., description="NEW, UNDER_REVIEW, ESCALATED, CLOSED_BENIGN, CLOSED_SUSPICIOUS"
    )
    actor: str = Field(default="analyst", max_length=128)
    note: Optional[str] = Field(
        None, description="Optional explanation note attached to the transition"
    )


class CaseResponse(CaseBase):
    id: int
    case_number: str
    status: str
    created_at: datetime
    updated_at: datetime
    notes_count: int = 0
    transactions_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class CaseDetailResponse(CaseResponse):
    notes: List[CaseNoteResponse] = []
    transactions: List[CaseTransactionResponse] = []

    model_config = ConfigDict(from_attributes=True)


class AuditEventResponse(BaseModel):
    id: int
    entity_type: str
    entity_id: str
    action: str
    actor: str
    details: Optional[str] = None
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
