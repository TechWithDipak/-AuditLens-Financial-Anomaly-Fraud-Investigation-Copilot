"""Data validation layer for transactions and CSV uploads."""

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml.data.schema import (
    MAX_UPLOAD_SIZE_BYTES,
    REQUIRED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class ValidationResult:
    """Structured report of data validation checks."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    columns_present: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns_present": self.columns_present,
            "missing_columns": self.missing_columns,
            "duplicate_count": self.duplicate_count,
            "summary": self.summary,
        }


def validate_dataframe(
    df: pd.DataFrame,
    is_training: bool = False,
) -> ValidationResult:
    """Validate a transaction DataFrame against schema and domain constraints.

    Args:
        df: DataFrame containing transaction records.
        is_training: If True, requires TARGET_COLUMN ('Class') to be present.

    Returns:
        ValidationResult: Detailed validation outcome.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Check for empty DataFrame
    if df is None or df.empty:
        return ValidationResult(
            is_valid=False,
            errors=["DataFrame is empty or contains no records."],
            row_count=0,
            column_count=0,
        )

    row_count, col_count = df.shape
    columns_present = list(df.columns)

    # 2. Check for required feature columns
    required_cols = list(REQUIRED_FEATURE_COLUMNS)
    if is_training:
        required_cols.append(TARGET_COLUMN)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required columns ({len(missing_cols)}): {', '.join(missing_cols)}"
        )

    # If missing required columns, cannot safely proceed with deep numeric checks
    if missing_cols:
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            row_count=row_count,
            column_count=col_count,
            columns_present=columns_present,
            missing_columns=missing_cols,
        )

    # 3. Check for non-numeric types in required feature columns
    for col in REQUIRED_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try converting to numeric to see if there are string corruptions
            coerced = pd.to_numeric(df[col], errors="coerce")
            invalid_count = int(coerced.isna().sum())
            if invalid_count > 0:
                errors.append(
                    f"Column '{col}' contains {invalid_count} non-numeric or malformed values."
                )

    # 4. Check for NaN / Missing values
    nan_counts = df[required_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        for col, count in cols_with_nans.items():
            errors.append(f"Column '{col}' has {int(count)} missing (NaN) values.")

    # 5. Check Amount domain constraints (Amount >= 0)
    if "Amount" in df.columns and pd.api.types.is_numeric_dtype(df["Amount"]):
        negative_amounts = int((df["Amount"] < 0).sum())
        if negative_amounts > 0:
            errors.append(
                f"Negative transaction amounts detected ({negative_amounts} records). Amounts must be >= 0."
            )

    # 6. Check Class domain constraints (must be 0 or 1)
    if TARGET_COLUMN in df.columns:
        if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
            errors.append(f"Target column '{TARGET_COLUMN}' must be numeric (0 or 1).")
        else:
            unique_classes = set(df[TARGET_COLUMN].dropna().unique())
            invalid_classes = unique_classes - {0, 1}
            if invalid_classes:
                errors.append(
                    f"Target column '{TARGET_COLUMN}' contains invalid labels: {invalid_classes}. Must be 0 or 1."
                )

    # 7. Check for duplicate rows (operational warning)
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(
            f"Dataset contains {duplicate_count} duplicate rows ({round(duplicate_count / row_count * 100, 2)}%)."
        )

    # Summary statistics
    summary: dict[str, Any] = {
        "features_verified": len(REQUIRED_FEATURE_COLUMNS),
        "has_target": TARGET_COLUMN in df.columns,
    }
    if TARGET_COLUMN in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        class_counts = df[TARGET_COLUMN].value_counts().to_dict()
        summary["class_distribution"] = {
            int(k): int(v) for k, v in class_counts.items()
        }

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        column_count=col_count,
        columns_present=columns_present,
        missing_columns=missing_cols,
        duplicate_count=duplicate_count,
        summary=summary,
    )


def validate_csv_bytes(
    content: bytes,
    filename: str = "upload.csv",
    is_training: bool = False,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> tuple[pd.DataFrame | None, ValidationResult]:
    """Validate raw CSV upload bytes, enforce size limits, and parse DataFrame safely.

    Args:
        content: Raw bytes from uploaded file.
        filename: Name of the uploaded file.
        is_training: Whether dataset is uploaded for training (requires Class).
        max_size_bytes: Maximum permitted size in bytes.

    Returns:
        tuple[pd.DataFrame | None, ValidationResult]
    """
    # 1. Check file size
    size_bytes = len(content)
    if size_bytes > max_size_bytes:
        return None, ValidationResult(
            is_valid=False,
            errors=[
                (
                    f"File '{filename}' exceeds maximum allowed size of {max_size_bytes // (1024 * 1024)}MB "
                    f"(received {size_bytes / (1024 * 1024):.2f}MB)."
                )
            ],
        )

    # 2. Check for empty content
    if size_bytes == 0:
        return None, ValidationResult(
            is_valid=False,
            errors=[f"File '{filename}' is empty (0 bytes)."],
        )

    # 3. Parse CSV safely
    try:
        # Protect against malformed encodings
        text_stream = io.StringIO(content.decode("utf-8"))
        df = pd.read_csv(text_stream)
    except UnicodeDecodeError:
        return None, ValidationResult(
            is_valid=False,
            errors=[f"File '{filename}' is not valid UTF-8 text."],
        )
    except pd.errors.EmptyDataError:
        return None, ValidationResult(
            is_valid=False,
            errors=[f"File '{filename}' contains no readable tabular data."],
        )
    except (pd.errors.ParserError, ValueError, OSError, TypeError) as exc:
        return None, ValidationResult(
            is_valid=False,
            errors=[f"Failed to parse CSV in file '{filename}': {exc!s}"],
        )

    # 4. Perform domain validation
    validation = validate_dataframe(df, is_training=is_training)
    return (df if validation.is_valid else None), validation
