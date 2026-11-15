"""Hidden Markov Model for sequence anomaly detection with hand-written Forward and Viterbi."""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn.hmm import CategoricalHMM
import joblib

from src.config import ARTIFACTS_DIR, RANDOM_SEED


# Symbol dictionary mappings
N_SYMBOLS = 12
N_STATES = 3


def encode_discrete_symbols(
    amt_zscore: np.ndarray,
    is_night: np.ndarray,
    distance_km: np.ndarray,
) -> np.ndarray:
    """
    Map transaction attributes to 12 discrete symbols (0 to 11):
    - amt_zscore: Low (<= 1.0) -> 0, Elevated (1.0 < z <= 3.0) -> 1, High (> 3.0) -> 2
    - is_night: Day (0) -> 0, Night (1) -> 1
    - distance_km: Near (<= 50km) -> 0, Far (> 50km) -> 1
    Total = amt_bin * 4 + night_bin * 2 + dist_bin
    """
    amt_zscore = np.asarray(amt_zscore, dtype=float)
    is_night = np.asarray(is_night, dtype=int)
    distance_km = np.asarray(distance_km, dtype=float)

    amt_bin = np.zeros_like(amt_zscore, dtype=int)
    amt_bin[amt_zscore > 1.0] = 1
    amt_bin[amt_zscore > 3.0] = 2

    night_bin = (is_night > 0).astype(int)
    dist_bin = (distance_km > 50.0).astype(int)

    symbols = amt_bin * 4 + night_bin * 2 + dist_bin
    return np.clip(symbols, 0, N_SYMBOLS - 1)


def forward_scaled(
    obs_seq: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
) -> float:
    """
    Hand-written Forward algorithm with Rabiner (1989) scaling.
    Computes total sequence log-likelihood ln P(O | lambda).
    """
    obs_seq = np.asarray(obs_seq, dtype=int)
    T = len(obs_seq)
    if T == 0:
        return 0.0

    alpha = startprob * emissionprob[:, obs_seq[0]]
    c0 = np.sum(alpha)
    if c0 <= 0:
        return -100.0
    alpha /= c0
    log_likelihood = np.log(c0)

    for t in range(1, T):
        alpha = (alpha @ transmat) * emissionprob[:, obs_seq[t]]
        ct = np.sum(alpha)
        if ct <= 0:
            ct = 1e-15
        alpha /= ct
        log_likelihood += np.log(ct)

    return float(log_likelihood)


def compute_customer_sequence_scores(
    df: pd.DataFrame,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute sequence anomaly score for each transaction using hand-written
    scaled forward algorithm across the customer's chronological sequence.
    Returns per-transaction rolling average log-likelihood over the last up to 10 transactions.
    """
    # Requires cc_num and symbols
    df_sorted = df[["cc_num", "symbol"]].copy()
    df_sorted["__orig_idx"] = np.arange(len(df))
    # Group by customer
    scores = np.zeros(len(df), dtype=float)

    for cc, group in df_sorted.groupby("cc_num"):
        symbols = group["symbol"].values
        orig_indices = group["__orig_idx"].values
        T = len(symbols)
        if T == 0:
            continue

        alpha = startprob * emissionprob[:, symbols[0]]
        c0 = np.sum(alpha)
        if c0 <= 0:
            c0 = 1e-15
        alpha /= c0
        log_c = np.empty(T)
        log_c[0] = np.log(c0)

        for t in range(1, T):
            alpha = (alpha @ transmat) * emissionprob[:, symbols[t]]
            ct = np.sum(alpha)
            if ct <= 0:
                ct = 1e-15
            alpha /= ct
            log_c[t] = np.log(ct)

        # Rolling mean of last min(t, 10) steps
        cumsum = np.cumsum(np.insert(log_c, 0, 0.0))
        w_start = np.maximum(0, np.arange(T) - window + 1)
        w_len = np.arange(T) - w_start + 1
        cust_scores = (cumsum[1:] - cumsum[w_start]) / w_len
        scores[orig_indices] = cust_scores

    return scores


def forward_log_space(
    obs_seq: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
) -> float:
    """
    Hand-written Forward algorithm in log-space.
    Computes total sequence log-likelihood ln P(O | lambda).
    """
    obs_seq = np.asarray(obs_seq, dtype=int)
    T = len(obs_seq)
    if T == 0:
        return 0.0

    K = len(startprob)
    log_start = np.log(np.maximum(startprob, 1e-15))
    log_trans = np.log(np.maximum(transmat, 1e-15))
    log_emiss = np.log(np.maximum(emissionprob, 1e-15))

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_start + log_emiss[:, obs_seq[0]]

    for t in range(1, T):
        sym = obs_seq[t]
        for j in range(K):
            log_alpha[t, j] = log_emiss[j, sym] + logsumexp(
                log_alpha[t - 1] + log_trans[:, j]
            )

    return float(logsumexp(log_alpha[-1]))


def viterbi_log_space(
    obs_seq: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Hand-written Viterbi algorithm in log-space.
    Finds optimal hidden state sequence: argmax_Q P(Q, O | lambda).
    """
    obs_seq = np.asarray(obs_seq, dtype=int)
    T = len(obs_seq)
    if T == 0:
        return 0.0, np.array([], dtype=int)

    K = len(startprob)
    log_start = np.log(np.maximum(startprob, 1e-15))
    log_trans = np.log(np.maximum(transmat, 1e-15))
    log_emiss = np.log(np.maximum(emissionprob, 1e-15))

    V = np.empty((T, K))
    ptr = np.empty((T, K), dtype=int)

    # Initialization
    V[0] = log_start + log_emiss[:, obs_seq[0]]
    ptr[0] = 0

    # Recursion
    for t in range(1, T):
        sym = obs_seq[t]
        for j in range(K):
            vals = V[t - 1] + log_trans[:, j]
            best_prev = int(np.argmax(vals))
            ptr[t, j] = best_prev
            V[t, j] = vals[best_prev] + log_emiss[j, sym]

    # Termination
    best_last = int(np.argmax(V[-1]))
    best_logprob = float(V[-1, best_last])

    # Backtracking
    path = np.empty(T, dtype=int)
    path[-1] = best_last
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return best_logprob, path


class CustomerHMM:
    """Wrapper managing Baum-Welch training and custom Forward/Viterbi inference."""

    def __init__(self, n_components: int = N_STATES, random_state: int = RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.model: Optional[CategoricalHMM] = None

    @property
    def startprob_(self) -> np.ndarray:
        return self.model.startprob_

    @property
    def transmat_(self) -> np.ndarray:
        return self.model.transmat_

    @property
    def emissionprob_(self) -> np.ndarray:
        return self.model.emissionprob_

    def fit(self, symbol_sequences: List[np.ndarray]) -> "CustomerHMM":
        """Train HMM using Baum-Welch on concatenated observation sequences."""
        # Filter empty sequences
        valid_seqs = [s for s in symbol_sequences if len(s) > 0]
        if not valid_seqs:
            raise ValueError("No valid symbol sequences to train HMM.")

        lengths = [len(s) for s in valid_seqs]
        X = np.concatenate(valid_seqs).reshape(-1, 1)

        self.model = CategoricalHMM(
            n_components=self.n_components,
            n_iter=50,
            tol=1e-4,
            random_state=self.random_state,
        )
        self.model.fit(X, lengths)
        return self

    def score_sequence(self, obs_seq: np.ndarray) -> float:
        """Normalized log-likelihood using hand-written Forward algorithm."""
        if len(obs_seq) == 0:
            return 0.0
        ll = forward_log_space(
            obs_seq,
            self.startprob_,
            self.transmat_,
            self.emissionprob_,
        )
        return ll / float(len(obs_seq))

    def decode_sequence(self, obs_seq: np.ndarray) -> Tuple[float, np.ndarray]:
        """Decode most probable state path using hand-written Viterbi."""
        if len(obs_seq) == 0:
            return 0.0, np.array([], dtype=int)
        return viterbi_log_space(
            obs_seq,
            self.startprob_,
            self.transmat_,
            self.emissionprob_,
        )

    def verify_against_hmmlearn(self, obs_seq: np.ndarray) -> Tuple[bool, bool, str]:
        """Verify that hand-written Forward and Viterbi match hmmlearn exactly."""
        if len(obs_seq) == 0:
            return True, True, "Empty sequence."

        obs_2d = obs_seq.reshape(-1, 1)

        # hmmlearn forward & viterbi
        sk_ll = float(self.model.score(obs_2d))
        sk_vit_ll, sk_path = self.model.decode(obs_2d, algorithm="viterbi")

        # Hand-written
        hw_ll = forward_log_space(
            obs_seq,
            self.startprob_,
            self.transmat_,
            self.emissionprob_,
        )
        hw_vit_ll, hw_path = viterbi_log_space(
            obs_seq,
            self.startprob_,
            self.transmat_,
            self.emissionprob_,
        )

        fwd_match = bool(np.isclose(sk_ll, hw_ll, atol=1e-4))
        vit_match = bool(
            np.isclose(sk_vit_ll, hw_vit_ll, atol=1e-4) and np.array_equal(sk_path, hw_path)
        )
        msg = f"Forward match: {fwd_match} (hmmlearn={sk_ll:.4f}, hw={hw_ll:.4f}) | Viterbi match: {vit_match}"
        return fwd_match, vit_match, msg

    def save(self, file_path: Optional[Path] = None) -> Path:
        if file_path is None:
            file_path = ARTIFACTS_DIR / "hmm_model.joblib"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "CustomerHMM":
        if file_path is None:
            file_path = ARTIFACTS_DIR / "hmm_model.joblib"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing HMM artifact: {file_path}")
        return joblib.load(file_path)
