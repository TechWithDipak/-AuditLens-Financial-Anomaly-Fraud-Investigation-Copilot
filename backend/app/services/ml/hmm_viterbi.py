"""Unit 4F: Viterbi Algorithm in log-space for latent state sequence decoding."""

from typing import List, Tuple

import numpy as np

from app.services.ml.hmm_forward import validate_hmm_parameters

LOG_ZERO = -1e12  # Representation of log(0)


def safe_log(x: np.ndarray) -> np.ndarray:
    """Compute elementwise log with negative infinity floor for numerical stability."""
    with np.errstate(divide="ignore"):
        res = np.log(x)
        res[~np.isfinite(res)] = LOG_ZERO
    return res


def viterbi_algorithm_log(
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    observations: List[int],
) -> Tuple[List[int], float]:
    """Find the single most probable latent state path using dynamic programming in log-space.

    Returns:
      best_path: Optimal state path sequence [q_0*, q_1*, ..., q_{T-1}*]
      best_log_prob: Joint log-probability of the optimal path and observations
    """
    validate_hmm_parameters(pi, A, B, observations)

    T = len(observations)
    N = len(pi)

    log_pi = safe_log(pi)
    log_A = safe_log(A)
    log_B = safe_log(B)

    # Viterbi trellis: V[t, j] stores max log prob up to step t in state j
    V = np.zeros((T, N), dtype=np.float64)
    # Backpointer matrix
    psi = np.zeros((T, N), dtype=np.int32)

    # 1. Initialization (t = 0)
    o_0 = observations[0]
    V[0, :] = log_pi + log_B[:, o_0]
    psi[0, :] = 0

    # 2. Recursion (t = 1 ... T-1)
    for t in range(1, T):
        o_t = observations[t]
        for j in range(N):
            # Candidate scores from each previous state i to current state j
            candidate_scores = V[t - 1, :] + log_A[:, j]
            best_prev_state = int(np.argmax(candidate_scores))
            V[t, j] = candidate_scores[best_prev_state] + log_B[j, o_t]
            psi[t, j] = best_prev_state

    # 3. Termination
    best_last_state = int(np.argmax(V[T - 1, :]))
    best_log_prob = float(V[T - 1, best_last_state])

    # 4. Backtracking
    best_path = [0] * T
    best_path[T - 1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = int(psi[t + 1, best_path[t + 1]])

    return best_path, best_log_prob
