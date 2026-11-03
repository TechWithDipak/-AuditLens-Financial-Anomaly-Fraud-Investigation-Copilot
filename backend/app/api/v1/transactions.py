"""FastAPI router for transaction queries, scoring, and explainability."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.transaction import (
    TransactionExplanationResponse,
    TransactionListResponse,
    TransactionResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from backend.app.services.transaction_service import (
    explain_transaction_by_id,
    get_transaction_by_id,
    list_transactions,
    score_transaction_features,
)

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.get("", response_model=TransactionListResponse)
def get_transactions(
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 50,
    min_amount: Annotated[
        float | None, Query(ge=0.0, description="Minimum amount filter")
    ] = None,
    max_amount: Annotated[
        float | None, Query(ge=0.0, description="Maximum amount filter")
    ] = None,
    is_flagged: Annotated[
        bool | None, Query(description="Filter by flagged status")
    ] = None,
    min_score: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Minimum fraud probability"),
    ] = None,
    max_score: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Maximum fraud probability"),
    ] = None,
    score_type: Annotated[str | None, Query(description="Filter by score type")] = None,
) -> TransactionListResponse:
    """Retrieve filtered, paginated transaction records."""
    return list_transactions(
        db=db,
        page=page,
        page_size=page_size,
        min_amount=min_amount,
        max_amount=max_amount,
        is_flagged=is_flagged,
        min_score=min_score,
        max_score=max_score,
        score_type=score_type,
    )


@router.get("/{tx_identifier}", response_model=TransactionResponse)
def get_transaction(
    tx_identifier: str,
    db: Annotated[Session, Depends(get_db)],
) -> TransactionResponse:
    """Retrieve an individual transaction record by integer ID or string transaction_id."""
    tx = get_transaction_by_id(db, tx_identifier)
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{tx_identifier}' not found",
        )

    import json

    feat_dict = {}
    try:
        feat_dict = json.loads(tx.features)
    except (json.JSONDecodeError, TypeError):
        feat_dict = {}

    return TransactionResponse(
        id=tx.id,
        transaction_id=tx.transaction_id,
        time=tx.time,
        amount=tx.amount,
        features=feat_dict,
        anomaly_score=tx.anomaly_score,
        fraud_probability=tx.fraud_probability,
        is_flagged=tx.is_flagged,
        score_type=tx.score_type,
        label=tx.label,
        created_at=tx.created_at,
    )


@router.post("/score", response_model=TransactionScoreResponse)
def score_transaction(
    payload: TransactionScoreRequest,
    x_analyst_id: Annotated[
        str | None, Header(description="Identity of requesting analyst")
    ] = None,
) -> TransactionScoreResponse:
    """Evaluate real-time dual scoring (unsupervised anomaly + supervised fraud probability)."""
    return score_transaction_features(
        time=payload.time,
        amount=payload.amount,
        features=payload.features,
        transaction_id=payload.transaction_id,
    )


@router.get(
    "/{tx_identifier}/explanation",
    response_model=TransactionExplanationResponse,
)
def get_transaction_explanation(
    tx_identifier: str,
    db: Annotated[Session, Depends(get_db)],
) -> TransactionExplanationResponse:
    """Retrieve SHAP or fallback feature attribution explanation for a transaction."""
    try:
        return explain_transaction_by_id(db, tx_identifier)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except RuntimeError as r_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(r_err),
        ) from r_err
