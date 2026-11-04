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


@router.post("/kmeans", response_model=KMeansRunResponse)
@router.post("/kmeans/run", response_model=KMeansRunResponse)
def run_kmeans(req: KMeansRunRequest) -> KMeansRunResponse:
    """Execute K-Means clustering and return cluster summaries with inertia and silhouette."""
    df, X_scaled, _ = _load_and_preprocess(req.dataset_name)
    k_val = req.n_clusters if req.n_clusters is not None else req.k
    try:
        res = KMeansService.run_kmeans(
            X=X_scaled,
            df_original=df,
            k=k_val,
            compute_elbow=req.compute_elbow,
            random_state=req.random_state,
        )
        res.n_clusters = res.k
        res.cluster_sizes = {str(cs.cluster_id): cs.sample_count for cs in res.cluster_summaries}
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/gmm", response_model=GMMRunResponse)
@router.post("/gmm/run", response_model=GMMRunResponse)
def run_gmm(req: GMMRunRequest) -> GMMRunResponse:
    """Fit Gaussian Mixture Model via EM and return log-likelihoods, responsibilities, and rarity rankings."""
    _, X_scaled, _ = _load_and_preprocess(req.dataset_name)
    try:
        res = GMMService.run_gmm(
            X=X_scaled,
            n_components=req.n_components,
            covariance_type=req.covariance_type,
            unusual_percentile=req.unusual_percentile,
            random_state=req.random_state,
        )
        res.lower_bound_log_likelihood = res.unusual_threshold_log_likelihood
        res.weights = [round(1.0 / res.n_components, 4)] * res.n_components
        res.top_anomalous_indices = [
            {
                "index": idx,
                "log_likelihood": float(res.log_likelihoods_preview[i]) if i < len(res.log_likelihoods_preview) else 0.0,
            }
            for i, idx in enumerate(res.unusual_indices_preview[:10])
        ]
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/pca", response_model=PCARunResponse)
@router.post("/pca/run", response_model=PCARunResponse)
def run_pca(req: PCARunRequest) -> PCARunResponse:
    """Fit PCA on feature matrix and return explained variance ratios and 2D projection."""
    df, X_scaled, feature_cols = _load_and_preprocess(req.dataset_name)
    try:
        res = PCAService.run_pca(
            X=X_scaled,
            feature_names=feature_cols,
            n_components=req.n_components,
            random_state=req.random_state,
        )
        res.cumulative_variance = res.cumulative_variance_ratio
        samples = []
        for i, p in enumerate(res.projection_preview[:100]):
            class_lbl = 0
            if "Class" in df.columns and i < len(df):
                class_lbl = int(df["Class"].iloc[i])
            samples.append({
                "x": float(p[0]) if len(p) > 0 else 0.0,
                "y": float(p[1]) if len(p) > 1 else 0.0,
                "class_label": class_lbl,
            })
        res.projection_samples = samples
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
