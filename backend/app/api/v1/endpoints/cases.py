"""FastAPI Endpoints for Investigation Cases and Audit Trails."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.security import AnalystRole, require_role
from app.db.session import get_db
from app.schemas.case import (
    AuditEventResponse,
    CaseCreate,
    CaseDetailResponse,
    CaseNoteCreate,
    CaseNoteResponse,
    CaseResponse,
    CaseStatusUpdate,
    CaseTransactionCreate,
    CaseTransactionResponse,
)
from app.services import case_service

router = APIRouter()


@router.post("/", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case_endpoint(
    case_in: CaseCreate,
    actor: str = Query("analyst", description="User or system triggering creation"),
    current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST)),
    db: Session = Depends(get_db),
):
    """Create a new investigation case (requires Analyst privileges or higher)."""
    try:
        case = case_service.create_case(db=db, case_in=case_in, actor=actor)
        return CaseResponse(
            id=case.id,
            case_number=case.case_number,
            title=case.title,
            description=case.description,
            status=case.status,
            priority=case.priority,
            assigned_analyst=case.assigned_analyst,
            created_at=case.created_at,
            updated_at=case.updated_at,
            notes_count=0,
            transactions_count=len(case_in.initial_transactions or []),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case: {str(e)}",
        ) from e


@router.get("/", response_model=List[CaseResponse])
def list_cases_endpoint(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    priority_filter: Optional[str] = Query(
        None, alias="priority", description="Filter by priority"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all investigation cases with filtering and pagination."""
    try:
        cases = case_service.list_cases(
            db=db,
            status=status_filter,
            priority=priority_filter,
            skip=skip,
            limit=limit,
        )
        return cases
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cases: {str(e)}",
        ) from e


@router.get("/audit/events", response_model=List[AuditEventResponse])
def list_audit_events_endpoint(
    entity_type: Optional[str] = Query(None, description="Entity type filter e.g. case"),
    entity_id: Optional[str] = Query(None, description="Entity ID filter"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Retrieve immutable action audit events for governance."""
    try:
        events = case_service.list_audit_events(
            db=db,
            entity_type=entity_type,
            entity_id=entity_id,
            skip=skip,
            limit=limit,
        )
        return events
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list audit events: {str(e)}",
        ) from e


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case_detail_endpoint(
    case_id: int,
    db: Session = Depends(get_db),
):
    """Retrieve complete case details, notes history, and linked transactions."""
    case = case_service.get_case(db=db, case_id=case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Case {case_id} not found"
        )

    return CaseDetailResponse(
        id=case.id,
        case_number=case.case_number,
        title=case.title,
        description=case.description,
        status=case.status,
        priority=case.priority,
        assigned_analyst=case.assigned_analyst,
        created_at=case.created_at,
        updated_at=case.updated_at,
        notes_count=len(case.notes),
        transactions_count=len(case.transactions),
        notes=[CaseNoteResponse.model_validate(n) for n in case.notes],
        transactions=[CaseTransactionResponse.model_validate(t) for t in case.transactions],
    )


@router.patch("/{case_id}/status", response_model=CaseResponse)
def update_case_status_endpoint(
    case_id: int,
    update_in: CaseStatusUpdate,
    current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST)),
    db: Session = Depends(get_db),
):
    """Update case lifecycle status (NEW, UNDER_REVIEW, ESCALATED, CLOSED_BENIGN, CLOSED_SUSPICIOUS)."""
    try:
        case = case_service.update_case_status(db=db, case_id=case_id, status_update=update_in)
        return CaseResponse(
            id=case.id,
            case_number=case.case_number,
            title=case.title,
            description=case.description,
            status=case.status,
            priority=case.priority,
            assigned_analyst=case.assigned_analyst,
            created_at=case.created_at,
            updated_at=case.updated_at,
            notes_count=len(case.notes),
            transactions_count=len(case.transactions),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update case status: {str(e)}",
        ) from e


@router.post(
    "/{case_id}/notes", response_model=CaseNoteResponse, status_code=status.HTTP_201_CREATED
)
def add_case_note_endpoint(
    case_id: int,
    note_in: CaseNoteCreate,
    current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST)),
    db: Session = Depends(get_db),
):
    """Add an analyst note to an investigation case."""
    try:
        note = case_service.add_case_note(db=db, case_id=case_id, note_in=note_in)
        return CaseNoteResponse.model_validate(note)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add note: {str(e)}",
        ) from e


@router.post(
    "/{case_id}/transactions",
    response_model=CaseTransactionResponse,
    status_code=status.HTTP_201_CREATED,
)
def link_transaction_endpoint(
    case_id: int,
    tx_in: CaseTransactionCreate,
    actor: str = Query("analyst", description="Author or system linking the transaction"),
    current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST)),
    db: Session = Depends(get_db),
):
    """Link a flagged transaction and its model scores to a case."""
    try:
        tx = case_service.link_case_transaction(db=db, case_id=case_id, tx_in=tx_in, actor=actor)
        return CaseTransactionResponse.model_validate(tx)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to link transaction: {str(e)}",
        ) from e
