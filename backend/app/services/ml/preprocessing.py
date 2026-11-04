"""Exploratory Data Analysis and Leakage-Safe Preprocessing Pipeline."""

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from app.schemas.preprocessing import (
    EDAResponse,
    FeatureDistribution,
    SplitPartitionInfo,
    TimeAnalysis,
)


def compute_eda(df: pd.DataFrame) -> EDAResponse:
    """Compute distribution statistics, skewness, and temporal characteristics."""
    total_records, total_features = df.shape

    # Class imbalance
    fraud_count = int(df["Class"].sum()) if "Class" in df.columns else 0
    fraud_rate_pct = round((fraud_count / total_records) * 100.0, 4) if total_records > 0 else 0.0

    # Time analysis
    time_info = None
    if "Time" in df.columns and pd.api.types.is_numeric_dtype(df["Time"]):
        min_t = float(df["Time"].min())
        max_t = float(df["Time"].max())
        duration_hrs = round((max_t - min_t) / 3600.0, 2)
        rate = round(total_records / max(duration_hrs, 0.01), 2)
        time_info = TimeAnalysis(
            min_time=min_t,
            max_time=max_t,
            duration_hours=duration_hrs,
            transaction_count=total_records,
            hourly_rate_avg=rate,
        )

    # Feature distributions
    distributions: List[FeatureDistribution] = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Class"]

    for col in numeric_cols:
        s = df[col].dropna()
        if not s.empty:
            distributions.append(
                FeatureDistribution(
                    feature=col,
                    mean=round(float(s.mean()), 4),
                    std=round(float(s.std()), 4),
                    median=round(float(s.median()), 4),
                    min=round(float(s.min()), 4),
                    max=round(float(s.max()), 4),
                    skewness=round(float(s.skew()), 4),
                )
            )

    return EDAResponse(
        total_records=total_records,
        total_features=total_features,
        fraud_count=fraud_count,
        fraud_rate_pct=fraud_rate_pct,
        time_analysis=time_info,
        feature_distributions=distributions,
    )


def split_dataset(
    df: pd.DataFrame,
    strategy: str = "stratified",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute reproducible train/val/test split.

    Strategies:
      - 'stratified': Preserves class ratio across splits using random_state.
      - 'chronological': Strictly sorts by Time; allocates past to train, middle to val, future to test.
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError("Invalid split fractions: test_size + val_size must be < 1.0.")

    target_col = "Class" if "Class" in df.columns else None

    if strategy == "chronological":
        if "Time" not in df.columns:
            raise ValueError("Chronological split requires 'Time' column.")
        sorted_df = df.sort_values(by="Time").reset_index(drop=True)
        n = len(sorted_df)
        train_end = int(n * (1.0 - test_size - val_size))
        val_end = int(n * (1.0 - test_size))

        train_df = sorted_df.iloc[:train_end].copy()
        val_df = sorted_df.iloc[train_end:val_end].copy()
        test_df = sorted_df.iloc[val_end:].copy()
        return train_df, val_df, test_df

    elif strategy == "stratified":
        if target_col is None:
            # Fall back to uniform random split if no target exists
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            adj_val_size = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(
                train_val_df, test_size=adj_val_size, random_state=random_state
            )
        else:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[target_col],
            )
            adj_val_size = val_size / (1.0 - test_size)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adj_val_size,
                random_state=random_state,
                stratify=train_val_df[target_col],
            )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    else:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. Use 'stratified' or 'chronological'."
        )


def get_partition_info(df: pd.DataFrame) -> SplitPartitionInfo:
    """Helper to summarize a split partition."""
    n = len(df)
    fraud_cnt = int(df["Class"].sum()) if "Class" in df.columns else 0
    fraud_pct = round((fraud_cnt / n) * 100.0, 4) if n > 0 else 0.0
    min_t = float(df["Time"].min()) if "Time" in df.columns and not df.empty else None
    max_t = float(df["Time"].max()) if "Time" in df.columns and not df.empty else None
    return SplitPartitionInfo(
        sample_count=n,
        fraud_count=fraud_cnt,
        fraud_rate_pct=fraud_pct,
        min_time=min_t,
        max_time=max_t,
    )


class LeakageSafePreprocessor:
    """Scales skewed features (Amount, Time) using RobustScaler fitted strictly on train data."""

    def __init__(self, scale_amount: bool = True, scale_time: bool = True):
        self.scale_amount = scale_amount
        self.scale_time = scale_time
        self.amount_scaler = RobustScaler() if scale_amount else None
        self.time_scaler = RobustScaler() if scale_time else None
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "LeakageSafePreprocessor":
        """Fit scalers strictly on training features."""
        self.feature_columns = [c for c in X_train.columns if c != "Class"]

        if self.scale_amount and "Amount" in X_train.columns:
            self.amount_scaler.fit(X_train[["Amount"]])

        if self.scale_time and "Time" in X_train.columns:
            self.time_scaler.fit(X_train[["Time"]])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using parameters learned solely from training data."""
        if not self.is_fitted:
            raise RuntimeError("LeakageSafePreprocessor must be fitted on training data first.")

        X_copy = X[self.feature_columns].copy()

        if self.scale_amount and "Amount" in X_copy.columns:
            X_copy["Amount"] = self.amount_scaler.transform(X_copy[["Amount"]])

        if self.scale_time and "Time" in X_copy.columns:
            X_copy["Time"] = self.time_scaler.transform(X_copy[["Time"]])

        return X_copy.to_numpy(dtype=np.float64)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on train and transform."""
        return self.fit(X_train).transform(X_train)

    def save(self, filepath: str) -> None:
        """Serialize fitted preprocessor artifact."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str) -> "LeakageSafePreprocessor":
        """Deserialize preprocessor artifact."""
        return joblib.load(filepath)
