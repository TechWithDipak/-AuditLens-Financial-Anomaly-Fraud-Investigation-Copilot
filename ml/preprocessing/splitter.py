"""Leakage-safe, reproducible data splitting module."""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.schema import TARGET_COLUMN


@dataclass
class SplitData:
    """Encapsulates partitioned feature matrices and target vectors."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    strategy: str = "stratified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary of split sizes and class distributions."""
        train_fraud = int(self.y_train.sum()) if self.y_train is not None else 0
        test_fraud = int(self.y_test.sum()) if self.y_test is not None else 0
        val_fraud = int(self.y_val.sum()) if self.y_val is not None else 0

        return {
            "strategy": self.strategy,
            "train_size": len(self.X_train),
            "train_fraud_count": train_fraud,
            "train_fraud_rate": round(train_fraud / max(1, len(self.X_train)) * 100, 3),
            "test_size": len(self.X_test),
            "test_fraud_count": test_fraud,
            "test_fraud_rate": round(test_fraud / max(1, len(self.X_test)) * 100, 3),
            "has_val": self.X_val is not None,
            "val_size": len(self.X_val) if self.X_val is not None else 0,
            "val_fraud_count": val_fraud,
            "val_fraud_rate": (
                round(val_fraud / max(1, len(self.X_val)) * 100, 3)
                if self.X_val is not None
                else 0
            ),
        }


def split_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    strategy: Literal["stratified", "time"] = "stratified",
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
    deduplicate: bool = True,
) -> SplitData:
    """Split dataset into reproducible train/validation/test sets.

    Guarantees:
    - Target column is strictly separated from feature matrix X.
    - Zero index or row overlap between train and test splits.
    - Optional deduplication before splitting to prevent identical duplicate transactions leaking across splits.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if test_size <= 0 or test_size >= 1.0:
        raise ValueError(f"Invalid test_size: {test_size}. Must be between 0 and 1.")

    if val_size < 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            f"Invalid val_size: {val_size}. Total test + val must be < 1.0."
        )

    # 1. Deduplication to prevent cross-split leakage
    working_df = df.copy()
    duplicates_removed = 0
    if deduplicate:
        init_len = len(working_df)
        working_df = working_df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = init_len - len(working_df)

    # 2. Separate X and y
    X = working_df.drop(columns=[target_col])
    y = working_df[target_col]

    metadata: dict[str, Any] = {
        "initial_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "clean_rows": len(working_df),
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "strategy": strategy,
    }

    if strategy == "stratified":
        # Check that both classes are present
        if y.nunique() < 2:
            raise ValueError(
                "Stratified split requires at least two distinct classes in target."
            )

        if val_size > 0:
            # First split off test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            # Then split temp into train and val
            adjusted_val_size = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=adjusted_val_size,
                random_state=random_state,
                stratify=y_temp,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            X_val, y_val = None, None

    elif strategy == "time":
        if "Time" not in X.columns:
            raise ValueError("Time-based split requires 'Time' column in DataFrame.")

        # Sort strictly by Time
        sorted_indices = working_df.sort_values(by="Time").index
        X_sorted = X.loc[sorted_indices].reset_index(drop=True)
        y_sorted = y.loc[sorted_indices].reset_index(drop=True)

        n = len(working_df)
        n_test = int(n * test_size)
        n_val = int(n * val_size) if val_size > 0 else 0
        n_train = n - n_test - n_val

        X_train = X_sorted.iloc[:n_train].reset_index(drop=True)
        y_train = y_sorted.iloc[:n_train].reset_index(drop=True)

        if n_val > 0:
            X_val = X_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True)
            y_val = y_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True)
        else:
            X_val, y_val = None, None

        X_test = X_sorted.iloc[n_train + n_val :].reset_index(drop=True)
        y_test = y_sorted.iloc[n_train + n_val :].reset_index(drop=True)

    else:
        raise ValueError(
            f"Unknown split strategy: '{strategy}'. Choose 'stratified' or 'time'."
        )

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        strategy=strategy,
        metadata=metadata,
    )
