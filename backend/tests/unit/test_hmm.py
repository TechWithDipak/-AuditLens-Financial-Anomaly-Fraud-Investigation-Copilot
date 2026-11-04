"""Unit tests for Unit 4: Hidden Markov Model (HMM), Forward, and Viterbi Algorithms."""

import math

import numpy as np
import pytest

from app.services.ml.hmm_forward import forward_algorithm_scaled
from app.services.ml.hmm_viterbi import viterbi_algorithm_log


@pytest.fixture
def toy_hmm():
    """Analytically verified 2-state, 2-observation toy HMM model.

    States: {0, 1}
    Observations: {0, 1}
    pi = [0.6, 0.4]
    A = [[0.7, 0.3], [0.4, 0.6]]
    B = [[0.8, 0.2], [0.1, 0.9]]
    Sequence: O = [0, 1, 0]

    Analytical derivations:
      - Forward Likelihood: P(O) = 0.098992 => log P(O) = -2.312716
      - Viterbi Best Path: [0, 1, 0] with path prob P* = 0.041472 => log P* = -3.182705
    """
    pi = np.array([0.6, 0.4], dtype=np.float64)
    A = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    B = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    obs = [0, 1, 0]
    return pi, A, B, obs


def test_forward_algorithm_hand_calculated_verification(toy_hmm):
    """Verify Forward algorithm against hand-calculated analytical derivation."""
    pi, A, B, obs = toy_hmm
    alpha_hat, log_ll, c_factors = forward_algorithm_scaled(pi, A, B, obs)

    # Hand-calculated analytical total probability: 0.098992
    expected_likelihood = 0.098992
    expected_log_ll = math.log(expected_likelihood)

    assert math.isclose(log_ll, expected_log_ll, rel_tol=1e-5)
    assert math.isclose(math.exp(log_ll), expected_likelihood, rel_tol=1e-5)
    assert len(c_factors) == 3
    assert alpha_hat.shape == (3, 2)


def test_viterbi_algorithm_hand_calculated_verification(toy_hmm):
    """Verify Viterbi algorithm against hand-calculated analytical state path."""
    pi, A, B, obs = toy_hmm
    best_path, log_prob = viterbi_algorithm_log(pi, A, B, obs)

    expected_path = [0, 1, 0]
    expected_path_prob = 0.041472
    expected_log_prob = math.log(expected_path_prob)

    assert best_path == expected_path
    assert math.isclose(log_prob, expected_log_prob, rel_tol=1e-5)


def test_numerical_stability_on_long_sequences(toy_hmm):
    """Verify Forward scaling and Viterbi log-space prevent underflow on T=2000 steps."""
    pi, A, B, _ = toy_hmm
    np.random.seed(42)
    # Generate 2,000 observations
    long_obs = list(np.random.choice([0, 1], size=2000, p=[0.7, 0.3]))

    # Forward must return finite negative log-likelihood without float underflow to 0.0
    _, log_ll, c_factors = forward_algorithm_scaled(pi, A, B, long_obs)
    assert np.isfinite(log_ll)
    assert log_ll < -100.0  # Naturally small likelihood accumulated in log-space
    assert all(np.isfinite(c) and c > 0 for c in c_factors)

    # Viterbi must decode state path of length 2,000 without underflow
    best_path, best_log_prob = viterbi_algorithm_log(pi, A, B, long_obs)
    assert len(best_path) == 2000
    assert np.isfinite(best_log_prob)
    assert all(state in [0, 1] for state in best_path)


def test_hmm_input_validation(toy_hmm):
    """Verify strict validation of stochastic matrices, empty sequence, and out-of-bounds symbols."""
    pi, A, B, obs = toy_hmm

    # Empty sequence
    with pytest.raises(ValueError, match="cannot be empty"):
        forward_algorithm_scaled(pi, A, B, [])

    # Row sum of transition matrix != 1.0
    bad_A = A.copy()
    bad_A[0, 0] = 0.9  # row 0 sum becomes 1.2
    with pytest.raises(ValueError, match="must sum to 1.0"):
        forward_algorithm_scaled(pi, bad_A, B, obs)

    # Out of bounds observation symbol (symbol 5 when M=2)
    with pytest.raises(ValueError, match="out of bounds"):
        viterbi_algorithm_log(pi, A, B, [0, 5, 0])
