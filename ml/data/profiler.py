"""Data profiling and statistical inspection module."""

from typing import Any

import pandas as pd

from ml.data.schema import PCA_COLUMNS, TARGET_COLUMN


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect schema, data types, missingness, duplicates, class distribution, and ranges.

    Args:
        df: Input transaction DataFrame.

    Returns:
        dict[str, Any]: Comprehensive profile metrics dictionary.
    """
    row_count, col_count = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    # Missingness
    missing_counts = df.isna().sum().to_dict()
    total_missing = int(sum(missing_counts.values()))

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / max(1, row_count)) * 100, 2)

    # Time coverage
    time_profile: dict[str, Any] = {}
    if "Time" in df.columns and pd.api.types.is_numeric_dtype(df["Time"]):
        min_time = float(df["Time"].min())
        max_time = float(df["Time"].max())
        time_profile = {
            "min_seconds": min_time,
            "max_seconds": max_time,
            "span_hours": round((max_time - min_time) / 3600, 2),
            "is_sorted": bool(df["Time"].is_monotonic_increasing),
        }

    # Amount statistics
    amount_profile: dict[str, Any] = {}
    if "Amount" in df.columns and pd.api.types.is_numeric_dtype(df["Amount"]):
        amount_profile = {
            "min": round(float(df["Amount"].min()), 2),
            "max": round(float(df["Amount"].max()), 2),
            "mean": round(float(df["Amount"].mean()), 2),
            "std": round(float(df["Amount"].std()), 2),
            "median": round(float(df["Amount"].median()), 2),
            "q25": round(float(df["Amount"].quantile(0.25)), 2),
            "q75": round(float(df["Amount"].quantile(0.75)), 2),
            "zero_amount_count": int((df["Amount"] == 0).sum()),
        }

    # Class distribution
    class_profile: dict[str, Any] = {}
    if TARGET_COLUMN in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        legit = int(counts.get(0, counts.get(0.0, 0)))
        fraud = int(counts.get(1, counts.get(1.0, 0)))
        total = legit + fraud
        fraud_rate = round((fraud / max(1, total)) * 100, 4)
        imbalance_ratio = f"{round(legit / max(1, fraud), 1)}:1" if fraud > 0 else "N/A"
        class_profile = {
            "legitimate_count": legit,
            "fraud_count": fraud,
            "total_labeled": total,
            "fraud_percentage": fraud_rate,
            "imbalance_ratio": imbalance_ratio,
        }

    # Summary of PCA features
    pca_summary: dict[str, dict[str, float]] = {}
    for col in PCA_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            pca_summary[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }

    return {
        "dimensions": {
            "rows": row_count,
            "columns": col_count,
            "memory_mb": memory_mb,
        },
        "quality": {
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": duplicate_pct,
        },
        "time_coverage": time_profile,
        "amount_statistics": amount_profile,
        "class_distribution": class_profile,
        "pca_features": pca_summary,
    }


def generate_profiling_report(
    df: pd.DataFrame, title: str = "Dataset Profiling Report"
) -> str:
    """Generate a clean markdown report of the dataset profile."""
    profile = profile_dataset(df)
    dims = profile["dimensions"]
    qual = profile["quality"]
    amt = profile["amount_statistics"]
    cls_dist = profile["class_distribution"]
    time_cov = profile["time_coverage"]

    lines = [
        f"# {title}",
        "",
        "## 1. Overview & Dimensions",
        f"- **Row count**: {dims['rows']:,}",
        f"- **Column count**: {dims['columns']}",
        f"- **Memory footprint**: {dims['memory_mb']} MB",
        "",
        "## 2. Data Quality & Missingness",
        f"- **Total missing (NaN) values**: {qual['total_missing_values']}",
        f"- **Duplicate rows**: {qual['duplicate_rows']} ({qual['duplicate_percentage']}%)",
        "",
    ]

    if time_cov:
        lines.extend(
            [
                "## 3. Time Coverage",
                f"- **Span**: {time_cov.get('span_hours', 'N/A')} hours ({time_cov.get('min_seconds', 0)} to {time_cov.get('max_seconds', 0)} seconds)",
                f"- **Monotonically ordered**: {time_cov.get('is_sorted', False)}",
                "",
            ]
        )

    if amt:
        lines.extend(
            [
                "## 4. Amount Statistics",
                f"- **Mean amount**: {amt['mean']}",
                f"- **Standard deviation**: {amt['std']}",
                f"- **Median (50%)**: {amt['median']}",
                f"- **Interquartile range**: [{amt['q25']}, {amt['q75']}]",
                f"- **Min / Max**: {amt['min']} / {amt['max']}",
                f"- **Zero-amount transactions**: {amt['zero_amount_count']}",
                "",
            ]
        )

    if cls_dist:
        lines.extend(
            [
                "## 5. Class Distribution (Target Imbalance)",
                f"- **Legitimate transactions (`Class = 0`)**: {cls_dist['legitimate_count']:,}",
                f"- **Fraudulent transactions (`Class = 1`)**: {cls_dist['fraud_count']:,}",
                f"- **Fraud rate**: {cls_dist['fraud_percentage']}%",
                f"- **Imbalance ratio**: {cls_dist['imbalance_ratio']}",
                "",
            ]
        )

    return "\n".join(lines)
