"""Unit tests for Unit 3B: Gaussian Mixture Model (GMM) via Expectation-Maximization (EM)."""

import numpy as np
import pytest

from app.services.ml.gmm_em import GMMService


@pytest.fixture
def gmm_data():
    """Create synthetic mixture of two Gaussians with an outlier."""
    np.random.seed(42)
    g1 = np.random.randn(50, 2) + np.array([-3, 0])
    g2 = np.random.randn(50, 2) + np.array([3, 0])
    outlier = np.array([[20.0, 20.0]])
    return np.vstack([g1, g2, outlier])


def test_gmm_em_convergence_and_responsibilities(gmm_data):
    """Verify EM parameter estimation converges and posterior responsibilities sum to 1."""
    res = GMMService.run_gmm(gmm_data, n_components=2, covariance_type="full", random_state=42)

    assert res.converged is True
    assert res.n_iter > 0
    assert res.aic is not None
    assert res.bic is not None

    # Check that responsibilities sum to ~1 for each sample
    for row in res.responsibilities_preview:
        assert sum(row) == pytest.approx(1.0, abs=1e-3)

    # Check finite log-likelihoods
    for ll in res.log_likelihoods_preview:
        assert np.isfinite(ll)
        assert ll <= 0.0  # Log-likelihood is non-positive for typical density scales


def test_gmm_unusualness_ranking(gmm_data):
    """Verify extreme outlier point (index 100) is ranked unusual due to low log-likelihood."""
    res = GMMService.run_gmm(gmm_data, n_components=2, unusual_percentile=2.0, random_state=42)

    # The outlier is at index 100
    assert 100 in res.unusual_indices_preview
    assert res.unusual_sample_count >= 1


def test_gmm_invalid_inputs(gmm_data):
    """Verify GMM raises ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="must be greater than component count"):
        GMMService.run_gmm(gmm_data[:2], n_components=5)

    gmm_nan = gmm_data.copy()
    gmm_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN or non-finite"):
        GMMService.run_gmm(gmm_nan, n_components=2)
