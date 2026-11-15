"""Feature engineering and causal transformation pipeline for AuditLens."""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.config import ARTIFACTS_DIR


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate haversine distance in kilometers between two sets of GPS coordinates."""
    R = 6371.0  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


def compute_causal_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute causal historical features per customer strictly chronologically.
    Guaranteed past-only:
    - secs_since_last_txn
    - amt_zscore_vs_customer_history (past only)
    - txn_count_last_24h (past only)
    """
    # Keep original index to restore order
    orig_cols = list(df.columns)
    work_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(work_df["trans_date_trans_time"]):
        work_df["trans_date_trans_time"] = pd.to_datetime(work_df["trans_date_trans_time"])

    # Sort strictly by customer and timestamp
    work_df = work_df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(names=["__orig_index"])

    # 1. secs_since_last_txn: past only (time since previous txn)
    # Default 30 days (2,592,000s) for first transaction
    DEFAULT_GAP_SECS = 86400.0 * 30.0
    time_diff = work_df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    work_df["secs_since_last_txn"] = time_diff.fillna(DEFAULT_GAP_SECS).astype(float)

    # 2. amt_zscore_vs_customer_history: past only
    # For txn i, mean and std are computed over [0, i-1]
    past_amt = work_df.groupby("cc_num")["amt"].shift(1)
    past_count = work_df.groupby("cc_num").cumcount()
    past_sum = past_amt.groupby(work_df["cc_num"]).cumsum().fillna(0.0)
    past_sq_sum = (past_amt ** 2).groupby(work_df["cc_num"]).cumsum().fillna(0.0)

    past_mean = np.where(past_count > 0, past_sum / np.maximum(past_count, 1), 0.0)
    past_var = np.where(
        past_count > 1,
        (past_sq_sum / np.maximum(past_count, 1)) - (past_mean ** 2),
        0.0,
    )
    past_std = np.sqrt(np.maximum(past_var, 0.0))

    # If first transaction or std is near zero, z-score is 0.0
    amt_zscore = np.where(
        (past_count > 0) & (past_std > 1e-4),
        (work_df["amt"] - past_mean) / (past_std + 1e-4),
        0.0,
    )
    work_df["amt_zscore_vs_customer_history"] = np.clip(amt_zscore, -10.0, 10.0).astype(float)

    # 3. txn_count_last_24h: past only (window: [t - 24h, t))
    # closed='left' strictly excludes current transaction
    rolling_24h = (
        work_df.groupby("cc_num")
        .rolling("24h", on="trans_date_trans_time", closed="left")["amt"]
        .count()
        .fillna(0.0)
        .values
    )
    work_df["txn_count_last_24h"] = rolling_24h.astype(float)

    # Restore original row order
    work_df = work_df.sort_values("__orig_index").reset_index(drop=True)
    work_df = work_df.drop(columns=["__orig_index"])

    return work_df


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base deterministic features:
    amt, log_amt, hour, day_of_week, is_night, distance_km, age, gender, city_pop.
    Assumes causal customer features (secs_since_last_txn, amt_zscore, txn_count_last_24h)
    are already present or computes them if missing.
    """
    df = df.copy()

    if "secs_since_last_txn" not in df.columns:
        df = compute_causal_customer_features(df)

    if not pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["log_amt"] = np.log1p(df["amt"].clip(lower=0.0)).astype(float)
    df["hour"] = df["trans_date_trans_time"].dt.hour.astype(int)
    df["day_of_week"] = df["trans_date_trans_time"].dt.day_of_week.astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Haversine distance
    df["distance_km"] = haversine_distance(
        df["lat"].values,
        df["long"].values,
        df["merch_lat"].values,
        df["merch_long"].values,
    ).astype(float)

    # Age
    dob_dt = pd.to_datetime(df["dob"])
    df["age"] = (
        (df["trans_date_trans_time"] - dob_dt).dt.days / 365.25
    ).fillna(40).astype(int)

    # Gender binary (M=1, F=0)
    df["gender"] = (df["gender"].astype(str).str.upper() == "M").astype(int)

    # Ensure numeric city_pop
    df["city_pop"] = df["city_pop"].astype(float)

    return df


class FeaturePipeline:
    """Preprocesses features: fits scalers and encoders on train data only."""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.num_cols: List[str] = [
            "amt",
            "log_amt",
            "hour",
            "day_of_week",
            "is_night",
            "distance_km",
            "age",
            "gender",
            "city_pop",
            "secs_since_last_txn",
            "amt_zscore_vs_customer_history",
            "txn_count_last_24h",
        ]
        self.cat_cols: List[str] = ["category"]
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit OneHotEncoder and StandardScaler on train data only."""
        # Fit categorical encoder
        cat_data = df[self.cat_cols].fillna("unknown")
        self.encoder.fit(cat_data)
        encoded_cats = list(self.encoder.get_feature_names_out(self.cat_cols))

        # Fit numerical scaler
        num_data = df[self.num_cols].fillna(0.0)
        self.scaler.fit(num_data)

        self.feature_names = self.num_cols + encoded_cats
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features into scaled matrix + one-hot encoded categories."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before transform.")

        num_scaled = self.scaler.transform(df[self.num_cols].fillna(0.0))
        cat_encoded = self.encoder.transform(df[self.cat_cols].fillna("unknown"))

        X_all = np.hstack([num_scaled, cat_encoded])
        return pd.DataFrame(X_all, columns=self.feature_names, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, file_path: Optional[Path] = None) -> Path:
        if file_path is None:
            file_path = ARTIFACTS_DIR / "feature_pipeline.joblib"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "FeaturePipeline":
        if file_path is None:
            file_path = ARTIFACTS_DIR / "feature_pipeline.joblib"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing pipeline artifact: {file_path}")
        return joblib.load(file_path)
