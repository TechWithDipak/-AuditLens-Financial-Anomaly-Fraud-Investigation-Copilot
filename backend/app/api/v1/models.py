"""FastAPI router for model provenance, runtime configuration, and evaluation reports."""

from fastapi import APIRouter

from backend.app.schemas.model import (
    ModelEvaluationResponse,
    ModelMetadataResponse,
)
from backend.app.services.model_service import (
    get_evaluation_report,
    get_models_metadata,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_model_metadata() -> ModelMetadataResponse:
    """Retrieve runtime ML model versions, feature count, and operational decision threshold."""
    return get_models_metadata()


@router.get("/evaluation", response_model=ModelEvaluationResponse)
def get_model_evaluation() -> ModelEvaluationResponse:
    """Retrieve benchmark evaluation report comparing models across imbalanced metrics."""
    return get_evaluation_report()

