"""Dataset Service: CSV Loading, Schema Validation, and Quality Profiling."""

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.dataset import (
    ClassImbalanceReport,
    ColumnProfile,
    DataQualityReport,
    DatasetValidationResult,
)

logger = logging.getLogger("auditlens.datasets")

# Default expected schema for primary fraud detection tabular data
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB bound


class DatasetValidationError(Exception):
    """Raised when dataset fails critical schema or security validation."""

    pass


class DatasetService:
    @staticmethod
    def profile_dataframe(
        df: pd.DataFrame, filename: str, file_size_bytes: int = 0
    ) -> DatasetValidationResult:
        """Compute exhaustive data quality profile without exposing raw row values in logs."""
        warnings = []
        row_count, col_count = df.shape

        if row_count == 0:
            raise DatasetValidationError(f"Dataset '{filename}' is empty.")

        # Check duplicate rows
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            warnings.append(f"Detected {duplicate_count} duplicate rows.")

        # Column-by-column profile
        columns_profile = []
        total_missing = 0

        for col in df.columns:
            s = df[col]
            missing_cnt = int(s.isna().sum())
            total_missing += missing_cnt
            missing_pct = round((missing_cnt / row_count) * 100.0, 3)

            min_val: Optional[float] = None
            max_val: Optional[float] = None
            mean_val: Optional[float] = None

            if pd.api.types.is_numeric_dtype(s):
                valid_s = s.dropna()
                if not valid_s.empty:
                    min_val = round(float(valid_s.min()), 4)
                    max_val = round(float(valid_s.max()), 4)
                    mean_val = round(float(valid_s.mean()), 4)

            columns_profile.append(
                ColumnProfile(
                    name=col,
                    dtype=str(s.dtype),
                    missing_count=missing_cnt,
                    missing_pct=missing_pct,
                    min_val=min_val,
                    max_val=max_val,
                    mean_val=mean_val,
                )
            )

        # Range checks
        if "Amount" in df.columns and pd.api.types.is_numeric_dtype(df["Amount"]):
            if (df["Amount"] < 0).any():
                warnings.append("Detected negative transaction amounts.")

        if "Time" in df.columns and pd.api.types.is_numeric_dtype(df["Time"]):
            if (df["Time"] < 0).any():
                warnings.append("Detected negative timestamp values.")

        # Class distribution check
        imbalance_report = None
        target_col = None
        for candidate in ["Class", "is_fraud", "target", "label"]:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col:
            val_counts = df[target_col].value_counts().to_dict()
            counts_str = {str(k): int(v) for k, v in val_counts.items()}
            pos_count = counts_str.get("1", 0)
            pos_ratio = round(pos_count / row_count, 6)
            is_imbalanced = pos_ratio < 0.05

            imbalance_report = ClassImbalanceReport(
                label_column=target_col,
                class_counts=counts_str,
                positive_class_ratio=pos_ratio,
                is_imbalanced=is_imbalanced,
            )

            if is_imbalanced:
                warnings.append(
                    f"Extreme class imbalance detected ({round(pos_ratio * 100, 3)}% positive). "
                    "Accuracy is not a reliable evaluation metric; use PR-AUC and F1."
                )

        # Check required columns
        missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_expected:
            warnings.append(f"Missing expected standard fraud columns: {missing_expected[:5]}...")

        status = (
            "error"
            if total_missing > (row_count * col_count * 0.5)
            else ("warning" if warnings else "valid")
        )

        quality = DataQualityReport(
            row_count=row_count,
            column_count=col_count,
            duplicate_rows=duplicate_count,
            missing_cells_total=total_missing,
            columns=columns_profile,
            class_imbalance=imbalance_report,
            warnings=warnings,
            is_valid_for_ml=len(missing_expected) == 0 and total_missing == 0,
        )

        # Safe preview: sanitize NaN/Inf for JSON serialization
        preview_df = df.head(10).replace({np.nan: None, np.inf: None, -np.inf: None})
        preview_rows = preview_df.to_dict(orient="records")

        # Sanitized audit logging (counts and metadata only, no raw transaction data)
        logger.info(
            f"Dataset profiled: filename={filename}, rows={row_count}, "
            f"cols={col_count}, duplicates={duplicate_count}, warnings={len(warnings)}"
        )

        return DatasetValidationResult(
            status=status,
            filename=filename,
            file_size_bytes=file_size_bytes,
            quality=quality,
            preview_rows=preview_rows,
            records=preview_rows,
            total_records=row_count,
            columns=list(df.columns),
        )

    @classmethod
    def validate_csv_bytes(cls, content: bytes, filename: str) -> DatasetValidationResult:
        """Validate uploaded CSV content with size and structure limits."""
        if not filename.lower().endswith(".csv"):
            raise DatasetValidationError("Only CSV files are supported (.csv extension required).")

        size_bytes = len(content)
        if size_bytes > MAX_UPLOAD_BYTES:
            raise DatasetValidationError(
                f"File exceeds maximum upload limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )

        if size_bytes == 0:
            raise DatasetValidationError("Uploaded file is empty (0 bytes).")

        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise DatasetValidationError(f"Malformed CSV: unable to parse ({str(e)})") from e

        return cls.profile_dataframe(df, filename=filename, file_size_bytes=size_bytes)

    @classmethod
    def load_fixture_dataset(
        cls, fixture_name: str = "creditcard_sample_fixture.csv"
    ) -> DatasetValidationResult:
        """Load and profile a standard synthetic fixture from data/fixtures/."""
        fixture_path = Path(settings.DATA_FIXTURES_DIR) / fixture_name
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found at {fixture_path}")

        df = pd.read_csv(fixture_path)
        size_bytes = fixture_path.stat().st_size
        return cls.profile_dataframe(df, filename=fixture_name, file_size_bytes=size_bytes)

    @classmethod
    def get_transaction_record(
        cls, index: int, fixture_name: str = "creditcard_sample_fixture.csv"
    ) -> dict:
        """Fetch a single transaction record by row index."""
        fixture_path = Path(settings.DATA_FIXTURES_DIR) / fixture_name
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found at {fixture_path}")

        df = pd.read_csv(fixture_path)
        if index < 0 or index >= len(df):
            raise IndexError(f"Transaction index {index} out of bounds (total rows: {len(df)})")
        return df.iloc[index].to_dict()


dataset_service = DatasetService()
