"""Unit tests for hand-written EM-GMM and mathematical equivalence with scikit-learn."""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from src.em_gmm import HandWrittenGMM


@pytest.fixture
def synthetic_gaussian_data():
    """Create a 2-cluster 2D dataset."""
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(100, 2) + np.array([2.0, 2.0])
    cluster2 = rng.randn(100, 2) + np.array([-2.0, -2.0])
    X = np.vstack([cluster1, cluster2])
    return X


def test_em_log_likelihood_monotonic_increase(synthetic_gaussian_data):
    """Log-likelihood must increase or stay within tolerance at each EM iteration."""
    X = synthetic_gaussian_data
    gmm = HandWrittenGMM(n_components=2, max_iter=30, random_state=42)
    gmm.fit(X)

    history = gmm.log_likelihood_history_
    assert len(history) > 1

    # Check monotonic increase
    for i in range(1, len(history)):
        assert history[i] >= history[i - 1] - 1e-3, (
            f"EM log-likelihood decreased at iter {i}: {history[i-1]} -> {history[i]}"
        )


def test_score_samples_matches_sklearn(synthetic_gaussian_data):
    """Verify log-likelihood score_samples matches sklearn GaussianMixture given same parameters."""
    X = synthetic_gaussian_data
    n_components = 2

    # Fit sklearn GMM
    sk_gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=42,
        max_iter=30,
    )
    sk_gmm.fit(X)
    sk_scores = sk_gmm.score_samples(X)

    # Instantiate hand-written GMM and set exact same parameters
    hw_gmm = HandWrittenGMM(
        n_components=n_components,
        reg_covar=1e-6,
        random_state=42,
    )
    hw_gmm.weights_ = sk_gmm.weights_.copy()
    hw_gmm.means_ = sk_gmm.means_.copy()
    hw_gmm.covariances_ = sk_gmm.covariances_.copy()

    hw_scores = hw_gmm.score_samples(X)

    # Assert analytical equivalence
    np.testing.assert_allclose(
        hw_scores,
        sk_scores,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Hand-written GMM log-likelihood differs from scikit-learn!",
    )


def test_weights_sum_to_one(synthetic_gaussian_data):
    """Fitted mixture weights must sum to 1.0 and be non-negative."""
    X = synthetic_gaussian_data
    gmm = HandWrittenGMM(n_components=3, max_iter=20, random_state=42)
    gmm.fit(X)

    assert np.isclose(np.sum(gmm.weights_), 1.0)
    assert np.all(gmm.weights_ >= 0.0)
    assert gmm.means_.shape == (3, 2)
    assert gmm.covariances_.shape == (3, 2, 2)
