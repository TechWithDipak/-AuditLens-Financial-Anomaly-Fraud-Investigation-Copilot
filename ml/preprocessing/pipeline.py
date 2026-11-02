"""Leakage-safe, repeatable preprocessing pipeline."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.data.schema import PCA_COLUMNS, REQUIRED_FEATURE_COLUMNS, TARGET_COLUMN


class AuditLensPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe preprocessing pipeline for financial fraud transactions.

    Key Features:
    - Fits transformations (imputations, scalers) strictly on training data.
    - Robust scaling on 'Amount' (median-centered, IQR-scaled) to handle heavy skewness and extreme outliers.
    - Cyclic transformation for 'Time' to capture periodic diurnal purchasing patterns.
    - Robust median imputation on PCA dimensions to protect against missing values during inference.
    - Zero target leakage: rejects inputs containing target column 'Class'.
    """

    def __init__(
        self,
        scale_amount: bool = True,
        encode_time_cyclic: bool = True,
        scale_pca: bool = False,
    ):
        self.scale_amount = scale_amount
        self.encode_time_cyclic = encode_time_cyclic
        self.scale_pca = scale_pca

        # Internal transformers (fitted in fit())
        self.imputer_: SimpleImputer | None = None
        self.amount_scaler_: RobustScaler | None = None
        self.pca_scaler_: StandardScaler | None = None
        self.feature_names_in_: list[str] = []
        self.output_feature_names_: list[str] = []
        self.is_fitted_: bool = False
        self.metadata_: dict[str, Any] = {}

    def _extract_time_features(
        self, time_series: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert continuous elapsed seconds into cyclic 24-hour sin/cos signals."""
        hours = (time_series // 3600) % 24
        radians = 2 * np.pi * hours / 24.0
        return np.sin(radians).to_numpy(), np.cos(radians).to_numpy()

    def fit(self, X: pd.DataFrame, y=None) -> "AuditLensPreprocessor":
        """Fit preprocessing transformations exclusively on training features X.

        Args:
            X: Training DataFrame containing required feature columns.
            y: Ignored (prevent target leakage).

        Raises:
            ValueError: If target column 'Class' is detected in X, or required features are missing.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=REQUIRED_FEATURE_COLUMNS)

        if TARGET_COLUMN in X.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' detected in feature matrix X. "
                "Target must be separated prior to preprocessing to prevent target leakage."
            )

        missing_cols = [col for col in REQUIRED_FEATURE_COLUMNS if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        self.feature_names_in_ = list(X.columns)

        # 1. Fit Imputer on all input features
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X[REQUIRED_FEATURE_COLUMNS])

        # 2. Fit Amount Scaler (RobustScaler is ideal for skewed amounts)
        if self.scale_amount:
            self.amount_scaler_ = RobustScaler(unit_variance=True)
            self.amount_scaler_.fit(X[["Amount"]])

        # 3. Optional PCA scaling
        if self.scale_pca:
            self.pca_scaler_ = StandardScaler()
            self.pca_scaler_.fit(X[PCA_COLUMNS])

        # 4. Determine output feature names
        out_names = []
        if self.encode_time_cyclic:
            out_names.extend(["time_sin", "time_cos"])
        else:
            out_names.append("Time")

        out_names.extend(PCA_COLUMNS)
        out_names.append("Amount")
        self.output_feature_names_ = out_names

        # 5. Record metadata for auditability
        self.metadata_ = {
            "n_features_in": len(self.feature_names_in_),
            "n_features_out": len(self.output_feature_names_),
            "output_features": self.output_feature_names_,
            "amount_median": (
                float(self.amount_scaler_.center_[0]) if self.amount_scaler_ else None
            ),
            "amount_scale": (
                float(self.amount_scaler_.scale_[0]) if self.amount_scaler_ else None
            ),
            "imputer_statistics": {
                col: float(val)
                for col, val in zip(
                    REQUIRED_FEATURE_COLUMNS, self.imputer_.statistics_, strict=False
                )
            },
        }

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using pre-fitted parameters.

        Args:
            X: Input DataFrame.

        Returns:
            np.ndarray: Clean, scaled numeric array ready for model consumption.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "AuditLensPreprocessor must be fitted before calling transform()."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if TARGET_COLUMN in X.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' found in transform input. Exclude target before transformation."
            )

        # Impute missing values
        imputed_array = self.imputer_.transform(X[REQUIRED_FEATURE_COLUMNS])
        imputed_df = pd.DataFrame(
            imputed_array, columns=REQUIRED_FEATURE_COLUMNS, index=X.index
        )

        # Transform Time
        if self.encode_time_cyclic:
            t_sin, t_cos = self._extract_time_features(imputed_df["Time"])
            time_features = np.column_stack([t_sin, t_cos])
        else:
            time_features = imputed_df[["Time"]].to_numpy()

        # Transform PCA features
        if self.scale_pca and self.pca_scaler_:
            pca_features = self.pca_scaler_.transform(imputed_df[PCA_COLUMNS])
        else:
            pca_features = imputed_df[PCA_COLUMNS].to_numpy()

        # Transform Amount
        if self.scale_amount and self.amount_scaler_:
            amount_features = self.amount_scaler_.transform(imputed_df[["Amount"]])
        else:
            amount_features = imputed_df[["Amount"]].to_numpy()

        # Combine into final array: [Time features, V1..V28, Amount]
        return np.hstack([time_features, pca_features, amount_features])

    def save(self, filepath: str | Path) -> Path:
        """Persist fitted preprocessor artifact and companion metadata JSON."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted preprocessor.")

        artifact_path = Path(filepath)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, artifact_path)

        meta_path = artifact_path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.metadata_, indent=2), encoding="utf-8")
        return artifact_path

    @classmethod
    def load(cls, filepath: str | Path) -> "AuditLensPreprocessor":
        """Load fitted preprocessor artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at: {path.resolve()}"
            )
        preprocessor = joblib.load(path)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return preprocessor
