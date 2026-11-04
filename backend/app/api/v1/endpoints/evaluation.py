"""API Endpoints for Model Explainability, Threshold Tuning, and Isolation Forest."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, status

from app.core.config import settings
from app.schemas.evaluation import (
    ExplainTransactionRequest,
    IsolationForestRunRequest,
    IsolationForestRunResponse,
    SHAPExplanationResponse,
    ThresholdAnalysisRequest,
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
        res = ExplainabilityService.explain_transaction(
            model_name=model_name,
            transaction_features=features,
            decision_threshold=threshold,
        )
        res.advisory_notice = res.caveat
        res.features = [
            {
                "feature": attr.feature_name,
                "value": attr.feature_value,
                "shap_value": attr.shap_value,
                "impact": attr.contribution,
            }
            for attr in res.top_attributions
        ]
        return res
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf)) from fnf
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/explain-transaction", response_model=SHAPExplanationResponse)
def explain_transaction_by_index(
    req: ExplainTransactionRequest,
) -> SHAPExplanationResponse:
    """Compute local SHAP feature attributions for a transaction by index from standard fixture."""
    path = Path(settings.DATA_FIXTURES_DIR) / "creditcard_sample_fixture.csv"
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / "creditcard_sample_fixture.csv"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credit card sample fixture not found.",
        )

    df = pd.read_csv(path)
    idx = req.transaction_index
    if idx < 0 or idx >= len(df):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction index {idx} out of range [0, {len(df)-1}].",
        )

    row = df.iloc[idx]
    features = {str(col): float(row[col]) for col in df.columns if col != "Class"}

    try:
        res = ExplainabilityService.explain_transaction(
            model_name=req.model_name,
            transaction_features=features,
            decision_threshold=req.decision_threshold,
        )
        res.transaction_index = idx
        res.advisory_notice = res.caveat
        res.features = [
            {
                "feature": attr.feature_name,
                "value": attr.feature_value,
                "shap_value": attr.shap_value,
                "impact": attr.contribution,
            }
            for attr in res.top_attributions
        ]
        return res
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf)) from fnf
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


def _calculate_thresholds_response(
    model_name: str,
    dataset_name: str,
    cost_fp: float,
    cost_fn: float,
) -> ThresholdCurveResponse:
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

    path = Path(settings.DATA_FIXTURES_DIR) / Path(dataset_name).name
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / Path(dataset_name).name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_name}' not found."
        )
    df = pd.read_csv(path)
    _, _, test_df = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    X_test_scaled = preprocessor.transform(test_df)
    y_test = test_df["Class"].to_numpy(dtype=int)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    curve = ThresholdService.compute_threshold_curve(
        y_test, y_prob, model_name=model_filename.split(".")[0]
    )

    for pt in curve.points:
        exp_cost = round(float(pt.false_positive * cost_fp + pt.false_negative * cost_fn), 2)
        pt.expected_cost = exp_cost
        pt.flagged_count = pt.true_positive + pt.false_positive
        pt.f1 = pt.f1_score

    min_pt = min(curve.points, key=lambda p: p.expected_cost if p.expected_cost is not None else float("inf"))
    curve.min_cost_threshold = round(min_pt.threshold, 2)
    curve.thresholds = [p.model_dump() for p in curve.points]
    return curve


@router.get("/thresholds", response_model=ThresholdCurveResponse)
def get_threshold_curve(
    model_name: str = Query("xgboost", description="Model name ('xgboost' or 'cart')"),
    dataset_name: str = Query("creditcard_sample_fixture.csv"),
    cost_fp: float = Query(10.0),
    cost_fn: float = Query(500.0),
) -> ThresholdCurveResponse:
    """Compute precision-recall operational trade-off curve across decision thresholds."""
    return _calculate_thresholds_response(model_name, dataset_name, cost_fp, cost_fn)


@router.post("/thresholds", response_model=ThresholdCurveResponse)
def post_threshold_curve(
    req: Optional[ThresholdAnalysisRequest] = Body(None),
) -> ThresholdCurveResponse:
    """Compute operational threshold trade-offs and cost minimization curve via POST."""
    cost_fp = req.cost_fp if req else 10.0
    cost_fn = req.cost_fn if req else 500.0
    model_name = req.model_name if req else "xgboost"
    dataset_name = req.dataset_name if req else "creditcard_sample_fixture.csv"
    return _calculate_thresholds_response(model_name, dataset_name, cost_fp, cost_fn)


@router.post("/isolation-forest", response_model=IsolationForestRunResponse)
@router.post("/isolation-forest/run", response_model=IsolationForestRunResponse)
def run_isolation_forest(req: IsolationForestRunRequest) -> IsolationForestRunResponse:
    """Run supporting Isolation Forest benchmark for unsupervised anomaly score ranking."""
    safe_name = Path(req.dataset_name).name
    path = Path(settings.DATA_FIXTURES_DIR) / safe_name
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / safe_name
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
