"""Data loading, verification, and chronological splitting for AuditLens."""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

from src.config import (
    TRAIN_RAW_FILE,
    TEST_RAW_FILE,
    TRAIN_SPLIT_RATIO,
    DOCS_DIR,
)


def check_raw_data_exists() -> Tuple[bool, str]:
    """Check if the required Sparkov dataset files exist."""
    missing = []
    if not TRAIN_RAW_FILE.exists():
        missing.append(str(TRAIN_RAW_FILE))
    if not TEST_RAW_FILE.exists():
        missing.append(str(TEST_RAW_FILE))

    if missing:
        msg = (
            "Required dataset files are missing:\n"
            + "\n".join(f"- {p}" for p in missing)
            + "\nPlease place 'fraudTrain.csv' and 'fraudTest.csv' into 'data/raw/'."
        )
        return False, msg
    return True, "Dataset files located."


def load_raw_dataset(
    file_path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a raw CSV file and ensure chronological ordering."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    # Use low_memory=False for consistent dtype inference
    df = pd.read_csv(file_path, nrows=nrows)

    # Clean unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure trans_date_trans_time is datetime
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        # Ensure chronological sort
        df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    return df


def split_train_val(
    df_train_full: pd.DataFrame,
    split_ratio: float = TRAIN_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split:
    Train = earlier split_ratio (80%) of fraudTrain.
    Validation = remaining (20%) of fraudTrain.
    Never shuffle across time.
    """
    # Verify chronological ordering
    if not df_train_full["trans_date_trans_time"].is_monotonic_increasing:
        df_train_full = df_train_full.sort_values("trans_date_trans_time").reset_index(drop=True)

    split_idx = int(len(df_train_full) * split_ratio)
    train_df = df_train_full.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df_train_full.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, val_df


def compute_split_stats(df: pd.DataFrame, name: str) -> Dict:
    """Calculate verified statistics for a given split."""
    total_rows = len(df)
    fraud_rows = int(df["is_fraud"].sum())
    normal_rows = total_rows - fraud_rows
    fraud_rate = (fraud_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    min_date = df["trans_date_trans_time"].min()
    max_date = df["trans_date_trans_time"].max()

    return {
        "name": name,
        "total_rows": total_rows,
        "normal_rows": normal_rows,
        "fraud_rows": fraud_rows,
        "fraud_rate": fraud_rate,
        "min_date": str(min_date),
        "max_date": str(max_date),
    }


def generate_data_docs(stats_dict: Dict[str, Dict], output_path: Optional[Path] = None) -> str:
    """Generate docs/DATA.md with real verified counts and synthetic disclosure."""
    if output_path is None:
        output_path = DOCS_DIR / "DATA.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# Dataset Specification: Sparkov Credit Card Fraud

## Overview
- **Source**: Kaggle `kartik2112/fraud-detection` (Sparkov synthetic credit card transaction generator).
- **Nature of Data**: **SYNTHETIC**. The transactions, names, merchant labels, and fraud patterns are generated using a simulation framework.
- **Purpose in AuditLens**: Training, validating, and testing machine learning models to detect fraudulent transactions and explain suspicious patterns.

## Verified Dataset Splits

| Split | Rows | Date Start | Date End | Normal Rows | Fraud Rows | Fraud Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (80% of fraudTrain)** | {stats_dict['train']['total_rows']:,} | {stats_dict['train']['min_date']} | {stats_dict['train']['max_date']} | {stats_dict['train']['normal_rows']:,} | {stats_dict['train']['fraud_rows']:,} | {stats_dict['train']['fraud_rate']:.4f}% |
| **Validation (20% of fraudTrain)** | {stats_dict['val']['total_rows']:,} | {stats_dict['val']['min_date']} | {stats_dict['val']['max_date']} | {stats_dict['val']['normal_rows']:,} | {stats_dict['val']['fraud_rows']:,} | {stats_dict['val']['fraud_rate']:.4f}% |
| **Full fraudTrain** | {stats_dict['train_full']['total_rows']:,} | {stats_dict['train_full']['min_date']} | {stats_dict['train_full']['max_date']} | {stats_dict['train_full']['normal_rows']:,} | {stats_dict['train_full']['fraud_rows']:,} | {stats_dict['train_full']['fraud_rate']:.4f}% |
| **Test (fraudTest)** | {stats_dict['test']['total_rows']:,} | {stats_dict['test']['min_date']} | {stats_dict['test']['max_date']} | {stats_dict['test']['normal_rows']:,} | {stats_dict['test']['fraud_rows']:,} | {stats_dict['test']['fraud_rate']:.4f}% |

## Split Methodology
- **Temporal Strictness**: Chronological time splits are strictly enforced. Shuffling across time is prohibited to prevent lookahead bias.
- **Train Window**: Earlier 80% of chronological transactions from `fraudTrain.csv`.
- **Validation Window**: Subsequent 20% of chronological transactions from `fraudTrain.csv`, used strictly for decision threshold tuning and hyperparameter selection.
- **Holdout Test**: Entire `fraudTest.csv` dataset, representing unseen future time period for final model evaluation.

## Feature Exclusions
The following fields are strictly excluded from predictive model inputs to prevent overfitting and synthetic artifact leakage:
- Identifier fields: `cc_num`, `trans_num`, `Unnamed: 0`
- Personal identification: `first`, `last`, `street`, `dob`
- System timestamps: `unix_time`
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    exists, msg = check_raw_data_exists()
    if not exists:
        print(msg)
        exit(1)

    print("Loading raw datasets for verification...")
    df_train_full = load_raw_dataset(TRAIN_RAW_FILE)
    df_test = load_raw_dataset(TEST_RAW_FILE)

    train_df, val_df = split_train_val(df_train_full)

    stats = {
        "train": compute_split_stats(train_df, "Train"),
        "val": compute_split_stats(val_df, "Validation"),
        "train_full": compute_split_stats(df_train_full, "Full fraudTrain"),
        "test": compute_split_stats(df_test, "Test"),
    }

    doc = generate_data_docs(stats)
    print("Verification complete. DATA.md written.")
    for k, v in stats.items():
        print(f"[{v['name']}] Rows: {v['total_rows']}, Dates: {v['min_date']} to {v['max_date']}, Fraud: {v['fraud_rows']} ({v['fraud_rate']:.4f}%)")
