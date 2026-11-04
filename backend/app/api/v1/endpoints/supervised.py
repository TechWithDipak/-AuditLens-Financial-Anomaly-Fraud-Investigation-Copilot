"""API Endpoints for Unit 5 Supervised & Ensemble Algorithms (CART, XGBoost, Ensembles)."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.schemas.supervised import (
    ModelComparisonResponse,
    SupervisedTrainRequest,
    SupervisedTrainResponse,
)
from app.services.ml.cart import CARTService
from app.services.ml.ensemble import EnsembleService
from app.services.ml.preprocessing import split_dataset
from app.services.ml.xgboost_model import XGBoostService

router = APIRouter()


def _load_and_split(
    dataset_name: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42
):
    """Helper to load dataset and perform reproducible stratified split."""
    safe_name = Path(dataset_name).name
    path = Path(settings.DATA_FIXTURES_DIR) / safe_name
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / safe_name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{safe_name}' not found."
        )

    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supervised learning requires ground-truth 'Class' column in dataset.",
        )

    train_df, val_df, test_df = split_dataset(
        df, strategy="stratified", test_size=test_size, val_size=val_size, random_state=random_state
    )
    return train_df, val_df, test_df


@router.post("/cart/train", response_model=SupervisedTrainResponse)
def train_cart(req: SupervisedTrainRequest) -> SupervisedTrainResponse:
    """Train CART Decision Tree baseline and evaluate on held-out test split."""
    train_df, _, test_df = _load_and_split(
        req.dataset_name, req.test_size, req.val_size, req.random_state
    )
    try:
        return CARTService.train_and_evaluate(train_df, test_df, random_state=req.random_state)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/xgboost/train", response_model=SupervisedTrainResponse)
def train_xgboost(req: SupervisedTrainRequest) -> SupervisedTrainResponse:
    """Train XGBoost gradient boosted trees with imbalance weighting and evaluate on test split."""
    train_df, _, test_df = _load_and_split(
        req.dataset_name, req.test_size, req.val_size, req.random_state
    )
    try:
        return XGBoostService.train_and_evaluate(train_df, test_df, random_state=req.random_state)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/ensemble/train", response_model=SupervisedTrainResponse)
def train_ensemble(
    req: SupervisedTrainRequest,
    ensemble_type: str = Query("voting", pattern="^(voting|stacking)$"),
) -> SupervisedTrainResponse:
    """Train soft-voting or stacking ensemble combining CART and XGBoost."""
    train_df, _, test_df = _load_and_split(
        req.dataset_name, req.test_size, req.val_size, req.random_state
    )
    try:
        return EnsembleService.train_and_evaluate(
            train_df, test_df, ensemble_type=ensemble_type, random_state=req.random_state
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/compare", response_model=ModelComparisonResponse)
def compare_models(
    dataset_name: str = Query("creditcard_sample_fixture.csv"),
    test_size: float = Query(0.2),
    random_state: int = Query(42),
) -> ModelComparisonResponse:
    """Evaluate CART, XGBoost, and Ensemble on the identical held-out test split for fair comparison."""
    train_df, _, test_df = _load_and_split(
        dataset_name, test_size=test_size, random_state=random_state
    )
    try:
        return EnsembleService.compare_all_models(train_df, test_df, random_state=random_state)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
