"""Schemas for Unit 3 Algorithm executions (K-Means, GMM, PCA)."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# K-Means Schemas
class KMeansRunRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    k: int = Field(default=3, ge=2, le=20)
    compute_elbow: bool = True
    random_state: int = 42


class ClusterSummary(BaseModel):
    cluster_id: int
    sample_count: int
    proportion_pct: float
    feature_means: Dict[str, float]


class KMeansRunResponse(BaseModel):
    k: int
    inertia: float
    silhouette_score: Optional[float] = None
    elbow_curve: Optional[List[Dict[str, float]]] = None
    cluster_summaries: List[ClusterSummary]
    cluster_assignments_preview: List[int]
    message: str = "Cluster membership indicates behavioral grouping, not fraud status."


# GMM Schemas
class GMMRunRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    n_components: int = Field(default=3, ge=2, le=15)
    covariance_type: str = Field(default="full", pattern="^(full|tied|diag|spherical)$")
    unusual_percentile: float = Field(default=1.0, gt=0.0, lt=50.0)
    random_state: int = 42


class GMMRunResponse(BaseModel):
    n_components: int
    covariance_type: str
    converged: bool
    n_iter: int
    aic: float
    bic: float
    unusual_threshold_log_likelihood: float
    unusual_sample_count: int
    responsibilities_preview: List[List[float]]
    log_likelihoods_preview: List[float]
    unusual_indices_preview: List[int]
    message: str = "Low log-likelihood reflects statistical rarity under the fitted mixture, not proof of fraud."


# PCA Schemas
class PCARunRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    n_components: int = Field(default=2, ge=2, le=30)
    random_state: int = 42


class PCARunResponse(BaseModel):
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_variance_ratio: List[float]
    total_explained_variance: float
    projection_preview: List[List[float]]
    feature_names: List[str]
    message: str = "PCA projections are for dimensional visualization and exploratory analysis."
