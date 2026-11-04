"""Unit 4D: Hidden Markov Model (HMM) Framework and Account Sequence Modeling."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.hmm import (
    AccountTimelinePoint,
    AccountTimelineResponse,
    HMMForwardResponse,
    HMMViterbiResponse,
)
from app.services.ml.hmm_forward import forward_algorithm_scaled
from app.services.ml.hmm_viterbi import viterbi_algorithm_log

# Canonical 3-State Model parameters for financial behavior
# State 0: Normal, State 1: Elevated / Travel, State 2: High-Risk Burst
DEFAULT_PI = np.array([0.75, 0.20, 0.05], dtype=np.float64)

# Transition Matrix A: High persistence in Normal, moderate in Elevated, low persistence in Burst
DEFAULT_A = np.array(
    [
        [0.85, 0.12, 0.03],
        [0.25, 0.65, 0.10],
        [0.30, 0.20, 0.50],
    ],
    dtype=np.float64,
)

# Emission Matrix B over 4 discrete symbols:
# Symbol 0: Small Debit/Payment (< $120)
# Symbol 1: Medium Payment ($120 - $600)
# Symbol 2: Large Transfer ($600 - $2000)
# Symbol 3: Urgent Cash-Out / Extreme Burst (> $2000)
DEFAULT_B = np.array(
    [
        [0.80, 0.15, 0.04, 0.01],
        [0.20, 0.60, 0.15, 0.05],
        [0.05, 0.15, 0.45, 0.35],
    ],
    dtype=np.float64,
)

STATE_LABELS = {
    0: "Normal Activity",
    1: "Elevated / Travel Activity",
    2: "High-Risk Burst",
}


def discretize_transaction(amount: float, tx_type: str) -> int:
    """Discretize continuous transaction into emission symbol index [0..3]."""
    if tx_type in ["CASH_OUT", "TRANSFER"] and amount > 2000.0:
        return 3
    elif amount > 600.0 or tx_type == "TRANSFER":
        return 2
    elif amount > 120.0:
        return 1
    else:
        return 0


class HMMService:
    @classmethod
    def score_sequence_forward(
        cls,
        observations: List[int],
        pi: Optional[List[float]] = None,
        A: Optional[List[List[float]]] = None,
        B: Optional[List[List[float]]] = None,
    ) -> HMMForwardResponse:
        """Calculate observation sequence likelihood using Forward Algorithm with scaling."""
        pi_arr = np.array(pi if pi else DEFAULT_PI, dtype=np.float64)
        A_arr = np.array(A if A else DEFAULT_A, dtype=np.float64)
        B_arr = np.array(B if B else DEFAULT_B, dtype=np.float64)

        _, log_ll, c_factors = forward_algorithm_scaled(pi_arr, A_arr, B_arr, observations)
        likelihood = float(np.exp(log_ll)) if log_ll > -700.0 else 0.0

        # Sequence anomaly heuristic: average log likelihood per step < -2.5
        avg_ll_per_step = log_ll / len(observations)
        is_unusual = avg_ll_per_step < -2.5

        return HMMForwardResponse(
            sequence_length=len(observations),
            log_likelihood=round(log_ll, 4),
            likelihood=round(likelihood, 6),
            scaled_factors=[round(float(c), 4) for c in c_factors[:20]],
            is_unusual_sequence=is_unusual,
        )

    @classmethod
    def decode_sequence_viterbi(
        cls,
        observations: List[int],
        pi: Optional[List[float]] = None,
        A: Optional[List[List[float]]] = None,
        B: Optional[List[List[float]]] = None,
    ) -> HMMViterbiResponse:
        """Find most probable latent behavioral state path using Viterbi Algorithm in log-space."""
        pi_arr = np.array(pi if pi else DEFAULT_PI, dtype=np.float64)
        A_arr = np.array(A if A else DEFAULT_A, dtype=np.float64)
        B_arr = np.array(B if B else DEFAULT_B, dtype=np.float64)

        best_path, log_prob = viterbi_algorithm_log(pi_arr, A_arr, B_arr, observations)
        labels = [STATE_LABELS.get(s, f"State {s}") for s in best_path]

        return HMMViterbiResponse(
            sequence_length=len(observations),
            best_path=best_path,
            best_path_log_prob=round(log_prob, 4),
            state_labels=labels,
        )

    @classmethod
    def get_account_timeline(cls, customer_id: str = "ACC_0001") -> AccountTimelineResponse:
        """Extract customer transactions from sequence fixture, score sequence, and decode states."""
        seq_path = Path(settings.DATA_FIXTURES_DIR) / "sequence_account_fixture.csv"
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequential fixture not found at {seq_path}")

        df = pd.read_csv(seq_path)
        customer_df = df[df["customer_id"] == customer_id].sort_values(by="timestamp")

        if customer_df.empty:
            raise ValueError(f"No transactions found for customer ID '{customer_id}'.")

        # Discretize observations
        symbols: List[int] = []
        for _, row in customer_df.iterrows():
            sym = discretize_transaction(float(row["amount"]), str(row["tx_type"]))
            symbols.append(sym)

        # Forward score and Viterbi path
        _, log_ll, _ = forward_algorithm_scaled(DEFAULT_PI, DEFAULT_A, DEFAULT_B, symbols)
        best_path, _ = viterbi_algorithm_log(DEFAULT_PI, DEFAULT_A, DEFAULT_B, symbols)

        timeline_points: List[AccountTimelinePoint] = []
        state_dist: Dict[str, int] = {}

        for idx, (_, row) in enumerate(customer_df.iterrows()):
            state = best_path[idx]
            label = STATE_LABELS.get(state, f"State {state}")
            state_dist[label] = state_dist.get(label, 0) + 1

            timeline_points.append(
                AccountTimelinePoint(
                    step=int(row["step"]),
                    timestamp=int(row["timestamp"]),
                    amount=round(float(row["amount"]), 2),
                    tx_type=str(row["tx_type"]),
                    observed_symbol=symbols[idx],
                    inferred_state=state,
                    inferred_state_label=label,
                    is_fraud=int(row["is_fraud"]),
                )
            )

        return AccountTimelineResponse(
            customer_id=customer_id,
            sequence_length=len(timeline_points),
            sequence_log_likelihood=round(log_ll, 4),
            timeline=timeline_points,
            state_distribution=state_dist,
        )
