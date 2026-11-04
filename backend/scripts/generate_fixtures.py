"""Generate deterministic synthetic test fixtures for Phase 2 and tests."""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

fixtures_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures"
fixtures_dir.mkdir(parents=True, exist_ok=True)

# 1. Credit Card Tabular Sample Fixture (1,000 rows, 31 columns)
n_samples = 1000
n_fraud = 5
time_vals = np.sort(np.random.uniform(0, 172800, n_samples))
v_features = np.random.randn(n_samples, 28)
amounts = np.round(np.random.exponential(scale=88.0, size=n_samples), 2)
classes = np.zeros(n_samples, dtype=int)

fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
classes[fraud_indices] = 1
v_features[fraud_indices, 0] -= 3.5
v_features[fraud_indices, 1] += 3.0
v_features[fraud_indices, 2] -= 4.0
amounts[fraud_indices] = np.round(np.random.uniform(250, 1200, size=n_fraud), 2)

cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
data = np.column_stack([time_vals, v_features, amounts, classes])
df_cc = pd.DataFrame(data, columns=cols)
df_cc["Class"] = df_cc["Class"].astype(int)

cc_path = fixtures_dir / "creditcard_sample_fixture.csv"
df_cc.to_csv(cc_path, index=False)
print(f"Generated {cc_path} with shape {df_cc.shape} and {df_cc['Class'].sum()} frauds")

# 2. Sequential Account Fixture (500 rows, 10 accounts, chronological)
accounts = [f"ACC_{i:04d}" for i in range(1, 11)]
rows = []

for acc in accounts:
    n_tx = 50
    curr_time = 1000
    state = 0
    for step in range(n_tx):
        if state == 0:
            state = int(np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))
        elif state == 1:
            state = int(np.random.choice([0, 1, 2], p=[0.25, 0.65, 0.10]))
        else:
            state = int(np.random.choice([0, 1, 2], p=[0.30, 0.20, 0.50]))

        curr_time += int(np.random.exponential(scale=3600))
        if state == 0:
            amt = round(float(np.random.uniform(5.0, 120.0)), 2)
            t_type = str(np.random.choice(["PAYMENT", "DEBIT"], p=[0.8, 0.2]))
            is_fraud = 0
        elif state == 1:
            amt = round(float(np.random.uniform(100.0, 600.0)), 2)
            t_type = str(np.random.choice(["PAYMENT", "TRANSFER"], p=[0.6, 0.4]))
            is_fraud = 1 if np.random.rand() < 0.05 else 0
        else:
            amt = round(float(np.random.uniform(800.0, 4500.0)), 2)
            t_type = str(np.random.choice(["TRANSFER", "CASH_OUT"], p=[0.7, 0.3]))
            is_fraud = 1 if np.random.rand() < 0.60 else 0

        rows.append(
            {
                "customer_id": acc,
                "timestamp": curr_time,
                "step": step,
                "tx_type": t_type,
                "amount": amt,
                "latent_state": state,
                "is_fraud": is_fraud,
            }
        )

df_seq = pd.DataFrame(rows)
seq_path = fixtures_dir / "sequence_account_fixture.csv"
df_seq.to_csv(seq_path, index=False)
print(f"Generated {seq_path} with shape {df_seq.shape} and {df_seq['is_fraud'].sum()} frauds")
