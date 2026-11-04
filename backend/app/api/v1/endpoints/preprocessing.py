"""API Endpoints for Exploratory Data Analysis and Leakage-Safe Splits."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, status

from app.core.config import settings
from app.schemas.preprocessing import EDAResponse, SplitRequest, SplitResponse
from app.services.ml.preprocessing import (
    compute_eda,
    get_partition_info,
    split_dataset,
)

router = APIRouter()


def _load_dataset_by_name(name: str) -> pd.DataFrame:
    """Helper to safely resolve and load a dataset from fixtures or raw."""
    safe_name = Path(name).name
    path = Path(settings.DATA_FIXTURES_DIR) / safe_name
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / safe_name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{safe_name}' not found.",
        )
    return pd.read_csv(path)


@router.get("/eda", response_model=EDAResponse)
@router.post("/eda", response_model=EDAResponse)
def get_eda(
    dataset_name: str = Query("creditcard_sample_fixture.csv", description="Dataset filename"),
    payload: Optional[Dict[str, Any]] = Body(None, description="Optional POST payload"),
) -> EDAResponse:
    """Generate exploratory data analysis, distribution statistics, and time analysis."""
    target_name = dataset_name
    if payload:
        if payload.get("dataset_path"):
            target_name = Path(payload["dataset_path"]).name
        elif payload.get("dataset_name"):
            target_name = Path(payload["dataset_name"]).name

    df = _load_dataset_by_name(target_name)
    return compute_eda(df)


@router.post("/split", response_model=SplitResponse)
def run_split(req: SplitRequest) -> SplitResponse:
    """Execute reproducible train/val/test data partitioning without leakage."""
    dataset_name = req.dataset_name
    if req.dataset_path:
        dataset_name = Path(req.dataset_path).name

    strategy = req.split_type or req.strategy or "stratified"

    df = _load_dataset_by_name(dataset_name)
    try:
        train_df, val_df, test_df = split_dataset(
            df=df,
            strategy=strategy,
            test_size=req.test_size,
            val_size=req.val_size,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    features = [c for c in df.columns if c != "Class"]

    return SplitResponse(
        strategy=strategy,
        random_state=req.random_state,
        train=get_partition_info(train_df),
        val=get_partition_info(val_df),
        test=get_partition_info(test_df),
        feature_names=features,
    )
