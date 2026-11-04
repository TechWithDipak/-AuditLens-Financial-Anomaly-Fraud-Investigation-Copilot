"""API Endpoints for Unit 3 Syllabus Algorithms (K-Means, GMM via EM, PCA)."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.schemas.algorithms import (
    GMMRunRequest,
    GMMRunResponse,
    KMeansRunRequest,
    KMeansRunResponse,
    PCARunRequest,
    PCARunResponse,
)
from app.services.ml.gmm_em import GMMService
from app.services.ml.kmeans import KMeansService
from app.services.ml.pca import PCAService
from app.services.ml.preprocessing import LeakageSafePreprocessor

router = APIRouter()


def _load_and_preprocess(name: str):
    """Helper to load dataset and prepare scaled feature array."""
    safe_name = Path(name).name
    path = Path(settings.DATA_FIXTURES_DIR) / safe_name
    if not path.exists():
        path = Path(settings.DATA_RAW_DIR) / safe_name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{safe_name}' not found.",
        )
    df = pd.read_csv(path)
    preprocessor = LeakageSafePreprocessor()
    X_scaled = preprocessor.fit_transform(df)
    return df, X_scaled, preprocessor.feature_columns


@router.post("/kmeans/run", response_model=KMeansRunResponse)
def run_kmeans(req: KMeansRunRequest) -> KMeansRunResponse:
    """Execute K-Means clustering and return cluster summaries with inertia and silhouette."""
    df, X_scaled, _ = _load_and_preprocess(req.dataset_name)
    try:
        return KMeansService.run_kmeans(
            X=X_scaled,
            df_original=df,
            k=req.k,
            compute_elbow=req.compute_elbow,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/gmm/run", response_model=GMMRunResponse)
def run_gmm(req: GMMRunRequest) -> GMMRunResponse:
    """Fit Gaussian Mixture Model via EM and return log-likelihoods, responsibilities, and rarity rankings."""
    _, X_scaled, _ = _load_and_preprocess(req.dataset_name)
    try:
        return GMMService.run_gmm(
            X=X_scaled,
            n_components=req.n_components,
            covariance_type=req.covariance_type,
            unusual_percentile=req.unusual_percentile,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/pca/run", response_model=PCARunResponse)
def run_pca(req: PCARunRequest) -> PCARunResponse:
    """Fit PCA on feature matrix and return explained variance ratios and 2D projection."""
    _, X_scaled, feature_cols = _load_and_preprocess(req.dataset_name)
    try:
        return PCAService.run_pca(
            X=X_scaled,
            feature_names=feature_cols,
            n_components=req.n_components,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
