"""Unit tests for Unit 3A: K-Means Clustering."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.kmeans import KMeansService


@pytest.fixture
def clustering_data():
    """Create synthetic clustered 2D data with 3 distinct centers."""
    np.random.seed(42)
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    c3 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([c1, c2, c3])
    df = pd.DataFrame(X, columns=["V1", "V2"])
    return X, df


def test_kmeans_valid_fit(clustering_data):
    """Verify K-Means produces valid clusters, inertia, and silhouette score."""
    X, df = clustering_data
    res = KMeansService.run_kmeans(X, df_original=df, k=3, compute_elbow=True, random_state=42)

    assert res.k == 3
    assert res.inertia > 0.0
    assert res.silhouette_score is not None
    assert 0.4 <= res.silhouette_score <= 1.0  # Well-separated clusters
    assert len(res.cluster_summaries) == 3
    assert sum(c.sample_count for c in res.cluster_summaries) == 90
    assert res.elbow_curve is not None
    assert len(res.elbow_curve) > 0


def test_kmeans_reproducibility(clustering_data):
    """Verify identical random state produces exact identical cluster assignments."""
    X, df = clustering_data
    res1 = KMeansService.run_kmeans(X, df, k=3, random_state=42)
    res2 = KMeansService.run_kmeans(X, df, k=3, random_state=42)

    assert res1.inertia == res2.inertia
    assert res1.cluster_assignments_preview == res2.cluster_assignments_preview


def test_kmeans_invalid_inputs(clustering_data):
    """Verify invalid parameters or corrupted arrays raise ValueError."""
    X, df = clustering_data
    # k > n_samples
    with pytest.raises(ValueError, match="Sample count .* must be greater"):
        KMeansService.run_kmeans(X[:2], df.iloc[:2], k=5)

    # NaN in input
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN or infinite"):
        KMeansService.run_kmeans(X_nan, df, k=3)
