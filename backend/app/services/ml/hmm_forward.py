"""Unit 4E: Forward Algorithm with numerical scaling for sequence likelihood."""

from typing import List, Tuple

import numpy as np


def validate_hmm_parameters(
    pi: np.ndarray, A: np.ndarray, B: np.ndarray, observations: List[int]
) -> None:
    """Validate stochastic matrix shapes, normalization, and symbol ranges."""
    n_states = len(pi)
    if A.shape != (n_states, n_states):
        raise ValueError(f"Transition matrix A must be of shape ({n_states}, {n_states}).")
    if B.shape[0] != n_states:
        raise ValueError(f"Emission matrix B row count must match state count ({n_states}).")
    if not np.allclose(np.sum(pi), 1.0, atol=1e-4):
        raise ValueError("Initial probability vector pi must sum to 1.0.")
    if not np.allclose(np.sum(A, axis=1), 1.0, atol=1e-4):
        raise ValueError("Each row of transition matrix A must sum to 1.0.")
    if not np.allclose(np.sum(B, axis=1), 1.0, atol=1e-4):
        raise ValueError("Each row of emission matrix B must sum to 1.0.")
    if len(observations) == 0:
        raise ValueError("Observation sequence cannot be empty.")
    n_symbols = B.shape[1]
    for idx, obs in enumerate(observations):
        if not (0 <= obs < n_symbols):
            raise ValueError(
                f"Observation at index {idx} ({obs}) is out of bounds [0, {n_symbols - 1}]."
            )


def forward_algorithm_scaled(
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    observations: List[int],
) -> Tuple[np.ndarray, float, List[float]]:
    """Compute Forward variables and sequence log-likelihood with numerical underflow scaling.

    Returns:
      alpha_hat: Scaled forward trellis (T, N)
      log_likelihood: Total sequence log-likelihood log P(O | lambda)
      c_factors: Normalization scale factors at each time step
    """
    validate_hmm_parameters(pi, A, B, observations)

    T = len(observations)
    N = len(pi)
    alpha_hat = np.zeros((T, N), dtype=np.float64)
    c_factors: List[float] = []

    # 1. Initialization (t = 0)
    o_0 = observations[0]
    alpha_0 = pi * B[:, o_0]
    sum_0 = float(np.sum(alpha_0))
    if sum_0 <= 0.0:
        raise ValueError("Initial observation has zero probability under the model.")
    c_0 = 1.0 / sum_0
    alpha_hat[0, :] = alpha_0 * c_0
    c_factors.append(c_0)

    # 2. Induction (t = 1 ... T-1)
    for t in range(1, T):
        o_t = observations[t]
        alpha_t = np.dot(alpha_hat[t - 1, :], A) * B[:, o_t]
        sum_t = float(np.sum(alpha_t))
        if sum_t <= 0.0:
            raise ValueError(f"Observation at step {t} has zero transition/emission probability.")
        c_t = 1.0 / sum_t
        alpha_hat[t, :] = alpha_t * c_t
        c_factors.append(c_t)

    # 3. Log-Likelihood calculation
    # log P(O | lambda) = - sum_{t=0}^{T-1} log(c_t)
    log_likelihood = -float(np.sum(np.log(c_factors)))

    return alpha_hat, log_likelihood, c_factors
