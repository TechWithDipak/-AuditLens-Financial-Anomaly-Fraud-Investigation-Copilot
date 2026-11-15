"""Unit tests for Hidden Markov Model, Forward and Viterbi equivalence with hmmlearn."""

import numpy as np
import pytest
from hmmlearn.hmm import CategoricalHMM

from src.hmm import (
    encode_discrete_symbols,
    forward_log_space,
    viterbi_log_space,
    CustomerHMM,
    N_SYMBOLS,
    N_STATES,
)


def test_symbol_encoding_ranges_and_bins():
    """Verify symbols fall strictly within [0, 11] and map correctly."""
    amt_zscores = np.array([0.2, 1.5, 4.0, -0.5, 2.5, 5.0])
    is_night = np.array([0, 0, 0, 1, 1, 1])
    distances = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])

    symbols = encode_discrete_symbols(amt_zscores, is_night, distances)
    assert len(symbols) == 6
    assert np.all(symbols >= 0) and np.all(symbols < N_SYMBOLS)

    # Low amt (0), Day (0), Near (0) -> 0
    assert symbols[0] == 0
    # Elevated amt (1), Day (0), Near (0) -> 1 * 4 = 4
    assert symbols[1] == 4
    # High amt (2), Day (0), Near (0) -> 2 * 4 = 8
    assert symbols[2] == 8
    # Low amt (0), Night (1), Far (1) -> 0*4 + 1*2 + 1 = 3
    assert symbols[3] == 3
    # Elevated amt (1), Night (1), Far (1) -> 1*4 + 1*2 + 1 = 7
    assert symbols[4] == 7
    # High amt (2), Night (1), Far (1) -> 2*4 + 1*2 + 1 = 11
    assert symbols[5] == 11


def test_forward_matches_hmmlearn_score():
    """Verify hand-written forward algorithm matches hmmlearn score to high precision."""
    rng = np.random.RandomState(42)
    K = 3
    M = 12

    model = CategoricalHMM(n_components=K, random_state=42)
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    model.emissionprob_ = rng.dirichlet(np.ones(M), size=K)

    obs = np.array([0, 1, 4, 7, 11, 2, 0, 3, 8, 9])
    obs_2d = obs.reshape(-1, 1)

    sk_score = model.score(obs_2d)
    hw_score_log = forward_log_space(
        obs,
        model.startprob_,
        model.transmat_,
        model.emissionprob_,
    )
    from src.hmm import forward_scaled
    hw_score_scaled = forward_scaled(
        obs,
        model.startprob_,
        model.transmat_,
        model.emissionprob_,
    )

    np.testing.assert_allclose(
        hw_score_log,
        sk_score,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Hand-written Forward log-space differs from hmmlearn!",
    )
    np.testing.assert_allclose(
        hw_score_scaled,
        sk_score,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Hand-written Forward scaled differs from hmmlearn!",
    )


def test_viterbi_matches_hmmlearn_decode():
    """Verify hand-written Viterbi path and log-prob match hmmlearn decode exactly."""
    rng = np.random.RandomState(42)
    K = 3
    M = 12

    model = CategoricalHMM(n_components=K, random_state=42)
    model.startprob_ = np.array([0.4, 0.4, 0.2])
    model.transmat_ = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    model.emissionprob_ = rng.dirichlet(np.ones(M), size=K)

    obs = np.array([2, 5, 11, 1, 0, 4, 8, 3, 7, 10])
    obs_2d = obs.reshape(-1, 1)

    sk_vit_ll, sk_path = model.decode(obs_2d, algorithm="viterbi")
    hw_vit_ll, hw_path = viterbi_log_space(
        obs,
        model.startprob_,
        model.transmat_,
        model.emissionprob_,
    )

    np.testing.assert_allclose(
        hw_vit_ll,
        sk_vit_ll,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Hand-written Viterbi log-probability differs from hmmlearn!",
    )
    np.testing.assert_array_equal(
        hw_path,
        sk_path,
        err_msg="Hand-written Viterbi state path differs from hmmlearn!",
    )


def test_customer_hmm_wrapper():
    """Verify end-to-end CustomerHMM fit, scoring, and verify method."""
    seq1 = np.array([0, 1, 0, 2, 1, 0])
    seq2 = np.array([3, 4, 5, 3, 4, 3, 2])
    seq3 = np.array([11, 10, 11, 9, 8])

    chmm = CustomerHMM(n_components=3, random_state=42)
    chmm.fit([seq1, seq2, seq3])

    test_seq = np.array([0, 1, 2, 4, 0])
    fwd_ok, vit_ok, msg = chmm.verify_against_hmmlearn(test_seq)
    assert fwd_ok, f"Forward verification failed: {msg}"
    assert vit_ok, f"Viterbi verification failed: {msg}"
