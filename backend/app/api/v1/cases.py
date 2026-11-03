"""FastAPI router for investigation case management, status tracking, and notes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.case import (
    CaseCreate,
    CaseListResponse,
    CaseNoteCreate,
    CaseNoteResponse,
    CasePriority,
    CaseResponse,
    CaseStatus,
    CaseUpdate,
)
from backend.app.services.case_service import (
    add_case_note,
    create_case,
    get_case,
    list_cases,
    update_case,
)

router = APIRouter(prefix="/cases", tags=["Cases"])


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_investigation_case(
    payload: CaseCreate,
    db: Annotated[Session, Depends(get_db)],
    x_analyst_id: Annotated[
        str | None, Header(description="Identity of creating analyst")
    ] = None,
) -> CaseResponse:
    """Open a new fraud investigation case linking one or more flagged transactions."""
    actor = x_analyst_id or payload.assigned_to or "analyst_system"
    return create_case(db=db, payload=payload, actor=actor)


@router.get("", response_model=CaseListResponse)
def get_cases(
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Cases per page")] = 20,
    status: Annotated[
        CaseStatus | None, Query(description="Filter by case lifecycle status")
    ] = None,
    priority: Annotated[
        CasePriority | None, Query(description="Filter by triage priority")
    ] = None,
    assigned_to: Annotated[
        str | None, Query(description="Filter by assigned analyst ID")
    ] = None,
) -> CaseListResponse:
    """Retrieve filtered, paginated investigation cases."""
    return list_cases(
        db=db,
        page=page,
        page_size=page_size,
        status=status,
        priority=priority,
        assigned_to=assigned_to,
    )


@router.get("/{case_identifier}", response_model=CaseResponse)
def get_investigation_case(
    case_identifier: str,
    db: Annotated[Session, Depends(get_db)],
) -> CaseResponse:
    """Retrieve a single investigation case including linked transactions and notes."""
    case = get_case(db=db, case_identifier=case_identifier)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_identifier}' not found",
        )
    return case


@router.patch("/{case_identifier}", response_model=CaseResponse)
def update_investigation_case(
    case_identifier: str,
    payload: CaseUpdate,
    db: Annotated[Session, Depends(get_db)],
    x_analyst_id: Annotated[
        str | None, Header(description="Identity of updating analyst")
    ] = None,
) -> CaseResponse:
    """Update case status, priority, description, or assigned analyst."""
    actor = x_analyst_id or "analyst_system"
    try:
        return update_case(
            db=db,
            case_identifier=case_identifier,
            payload=payload,
            actor=actor,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err


@router.post(
    "/{case_identifier}/notes",
    response_model=CaseNoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_note_to_case(
    case_identifier: str,
    payload: CaseNoteCreate,
    db: Annotated[Session, Depends(get_db)],
    x_analyst_id: Annotated[
        str | None, Header(description="Identity of note author")
    ] = None,
) -> CaseNoteResponse:
    """Attach an auditable analyst comment or finding to an investigation case."""
    actor = x_analyst_id or payload.author or "analyst_system"
    try:
        return add_case_note(
            db=db,
            case_identifier=case_identifier,
            payload=payload,
            actor=actor,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
