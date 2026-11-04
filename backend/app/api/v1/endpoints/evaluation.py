"""API Endpoints for Model Explainability, Threshold Tuning, and Isolation Forest."""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, status

from app.core.config import settings
from app.schemas.evaluation import (
    IsolationForestRunRequest,
    IsolationForestRunResponse,
    SHAPExplanationResponse,
    ThresholdCurveResponse,
)
from app.services.ml.explainability import ExplainabilityService
from app.services.ml.isolation_forest import IsolationForestService
from app.services.ml.preprocessing import LeakageSafePreprocessor, split_dataset
from app.services.ml.thresholds import ThresholdService

router = APIRouter()


@router.post("/explain", response_model=SHAPExplanationResponse)
def explain_transaction_shap(
    model_name: str = Query("xgboost", description="Model name ('xgboost' or 'cart')"),
    features: Dict[str, float] = Body(..., description="Transaction feature dictionary"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> SHAPExplanationResponse:
    """Compute local SHAP feature attributions for a single transaction."""
    try:
        return ExplainabilityService.explain_transaction(
            model_name=model_name,
            transaction_features=features,
            decision_threshold=threshold,
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf)) from fnf
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/thresholds", response_model=ThresholdCurveResponse)
def get_threshold_curve(
    model_name: str = Query("xgboost", description="Model name ('xgboost' or 'cart')"),
    dataset_name: str = Query("creditcard_sample_fixture.csv"),
) -> ThresholdCurveResponse:
    """Compute precision-recall operational trade-off curve across decision thresholds."""
    artifacts_dir = Path(settings.ARTIFACTS_DIR)
    model_filename = "xgboost_model.joblib" if "xgb" in model_name.lower() else "cart_model.joblib"
    artifact_path = artifacts_dir / model_filename

    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trained model '{model_filename}' not found. Please train model first.",
        )

    saved = joblib.load(artifact_path)
    model = saved["model"]
    preprocessor = saved["preprocessor"]

    # Load test split
    path = Path(settings.DATA_FIXTURES_DIR) / Path(dataset_name).name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_name}' not found."
        )
    df = pd.read_csv(path)
    _, _, test_df = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    X_test_scaled = preprocessor.transform(test_df)
    y_test = test_df["Class"].to_numpy(dtype=int)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return ThresholdService.compute_threshold_curve(
        y_test, y_prob, model_name=model_filename.split(".")[0]
    )


@router.post("/isolation-forest/run", response_model=IsolationForestRunResponse)
def run_isolation_forest(req: IsolationForestRunRequest) -> IsolationForestRunResponse:
    """Run supporting Isolation Forest benchmark for unsupervised anomaly score ranking."""
    safe_name = Path(req.dataset_name).name
    path = Path(settings.DATA_FIXTURES_DIR) / safe_name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{safe_name}' not found."
        )
    df = pd.read_csv(path)
    preprocessor = LeakageSafePreprocessor()
    X_scaled = preprocessor.fit_transform(df)

    try:
        return IsolationForestService.run_isolation_forest(
            X=X_scaled,
            contamination=req.contamination,
            n_estimators=req.n_estimators,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
