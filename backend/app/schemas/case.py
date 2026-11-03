"""Pydantic schemas for case management, investigator notes, and status workflows."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    """Lifecycle statuses for fraud investigation cases."""

    NEW = "NEW"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    CLOSED_FALSE_POSITIVE = "CLOSED_FALSE_POSITIVE"
    CLOSED_CONFIRMED_FRAUD = "CLOSED_CONFIRMED_FRAUD"


class CasePriority(str, Enum):
    """Urgency / triage priorities for fraud cases."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseNoteCreate(BaseModel):
    """Payload to add an analyst note to a case."""

    author: str = Field(
        ..., min_length=1, max_length=128, description="Analyst name or ID"
    )
    content: str = Field(..., min_length=1, description="Audit note text or rationale")


class CaseNoteResponse(BaseModel):
    """Persisted case note."""

    id: int
    case_id: int
    author: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class CaseCreate(BaseModel):
    """Payload to create an investigation case."""

    title: str = Field(
        ..., min_length=3, max_length=256, description="Short summary of investigation"
    )
    description: str = Field("", description="Detailed background or context")
    priority: CasePriority = Field(
        CasePriority.MEDIUM, description="Initial triage priority"
    )
    transaction_ids: list[str] = Field(
        default_factory=list, description="List of transaction IDs linked to this case"
    )
    assigned_to: str | None = Field(
        None, max_length=128, description="Assignee analyst ID"
    )


class CaseUpdate(BaseModel):
    """Payload to update an investigation case."""

    title: str | None = Field(None, min_length=3, max_length=256)
    description: str | None = None
    status: CaseStatus | None = None
    priority: CasePriority | None = None
    assigned_to: str | None = None


class CaseResponse(BaseModel):
    """Full investigation case detail."""

    id: int
    case_id: str
    title: str
    description: str
    status: CaseStatus
    priority: CasePriority
    assigned_to: str | None
    transaction_ids: list[str] = Field(default_factory=list)
    notes: list[CaseNoteResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CaseListResponse(BaseModel):
    """Paginated collection of investigation cases."""

    items: list[CaseResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
