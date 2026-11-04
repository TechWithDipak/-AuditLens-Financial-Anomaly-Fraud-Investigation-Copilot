"""Unit tests for Unit 3C: Principal Component Analysis (PCA)."""

import numpy as np
import pytest

from app.services.ml.pca import PCAService


@pytest.fixture
def pca_data():
    """Create synthetic data with dominant first principal axis."""
    np.random.seed(42)
    # x has variance 10, y has variance 1, z has variance 0.1
    x = np.random.normal(0, 10, 100)
    y = np.random.normal(0, 1, 100)
    z = np.random.normal(0, 0.1, 100)
    X = np.column_stack([x, y, z])
    feature_names = ["feat_x", "feat_y", "feat_z"]
    return X, feature_names


def test_pca_explained_variance(pca_data):
    """Verify PCA explained variance is monotonic decreasing and sum is <= 1.0."""
    X, feature_names = pca_data
    res = PCAService.run_pca(X, feature_names=feature_names, n_components=3, random_state=42)

    assert res.n_components == 3
    assert len(res.explained_variance_ratio) == 3

    # Monotonic decreasing variance explained: component 0 > component 1 > component 2
    assert res.explained_variance_ratio[0] > res.explained_variance_ratio[1]
    assert res.explained_variance_ratio[1] > res.explained_variance_ratio[2]

    # Cumulative variance should end at ~1.0 for full components
    assert res.cumulative_variance_ratio[-1] == pytest.approx(1.0, abs=1e-3)
    assert len(res.projection_preview) == 20
    assert len(res.projection_preview[0]) == 3


def test_pca_invalid_components(pca_data):
    """Verify PCA rejects invalid component counts."""
    X, feature_names = pca_data
    with pytest.raises(ValueError, match="must be between 1 and"):
        PCAService.run_pca(X, feature_names, n_components=10)  # Only 3 features exist

    with pytest.raises(ValueError, match="must be between 1 and"):
        PCAService.run_pca(X, feature_names, n_components=0)
