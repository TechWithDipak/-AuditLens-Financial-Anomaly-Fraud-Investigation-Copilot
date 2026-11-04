"""FastAPI Endpoints for AI Investigation Report Assistant."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.assistant import (
    GenerateDraftRequest,
    InvestigationReportDraft,
    VerifyEvidenceRequest,
)
from app.services.assistant.assistant_service import assistant_service
from app.services.assistant.evidence_verifier import verify_draft_evidence

router = APIRouter()


@router.post(
    "/generate-draft",
    response_model=InvestigationReportDraft,
    status_code=status.HTTP_200_OK,
)
def generate_draft_endpoint(request: GenerateDraftRequest):
    """Generate structured, evidence-grounded investigation report draft with claim verification."""
    try:
        return assistant_service.generate_draft(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate investigation report draft: {str(e)}",
        ) from e


@router.post("/verify-evidence")
def verify_evidence_endpoint(request: VerifyEvidenceRequest):
    """Verify arbitrary draft claims against actual ground-truth facts and scores."""
    try:
        is_verified, warnings = verify_draft_evidence(
            draft=request.draft,
            actual_amount=request.actual_amount,
            actual_probability=request.actual_fraud_probability,
        )
        return {
            "verification_passed": is_verified,
            "discrepancies_detected": len(warnings),
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify report evidence: {str(e)}",
        ) from e
