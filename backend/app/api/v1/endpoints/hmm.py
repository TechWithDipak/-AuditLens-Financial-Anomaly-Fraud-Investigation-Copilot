"""API Endpoints for Unit 4 Hidden Markov Model (Forward, Viterbi, and Account Timelines)."""

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.hmm import (
    AccountTimelineResponse,
    HMMForwardRequest,
    HMMForwardResponse,
    HMMViterbiRequest,
    HMMViterbiResponse,
)
from app.services.ml.hmm import HMMService

router = APIRouter()


@router.post("/forward", response_model=HMMForwardResponse)
def compute_forward_likelihood(req: HMMForwardRequest) -> HMMForwardResponse:
    """Compute observation sequence log-likelihood using the Forward Algorithm with scaling."""
    try:
        return HMMService.score_sequence_forward(
            observations=req.observations,
            pi=req.initial_probs,
            A=req.transition_matrix,
            B=req.emission_matrix,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/viterbi", response_model=HMMViterbiResponse)
def decode_viterbi_path(req: HMMViterbiRequest) -> HMMViterbiResponse:
    """Decode the most likely latent behavioral state path using the Viterbi Algorithm in log-space."""
    try:
        return HMMService.decode_sequence_viterbi(
            observations=req.observations,
            pi=req.initial_probs,
            A=req.transition_matrix,
            B=req.emission_matrix,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/account-timeline", response_model=AccountTimelineResponse)
def get_account_timeline(
    customer_id: str = Query(
        "ACC_0001", description="Account identifier (e.g. ACC_0001 to ACC_0010)"
    ),
) -> AccountTimelineResponse:
    """Extract chronological account transaction sequence, compute Forward score, and decode Viterbi state path."""
    try:
        return HMMService.get_account_timeline(customer_id=customer_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/accounts/{account_id}/timeline", response_model=AccountTimelineResponse)
def get_account_timeline_by_path(
    account_id: str,
) -> AccountTimelineResponse:
    """Alias for account timeline with path parameter for RESTful frontend access."""
    try:
        return HMMService.get_account_timeline(customer_id=account_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
