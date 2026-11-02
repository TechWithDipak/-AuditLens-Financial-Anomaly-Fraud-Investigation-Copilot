"""AuditLens Exploratory Data Analysis (EDA) Module."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.data.schema import TARGET_COLUMN


def analyze_class_imbalance(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> dict[str, Any]:
    """Calculate class counts, percentages, and imbalance ratios."""
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not present"}

    counts = df[target_col].value_counts().to_dict()
    legit_count = int(counts.get(0, counts.get(0.0, 0)))
    fraud_count = int(counts.get(1, counts.get(1.0, 0)))
    total = legit_count + fraud_count

    fraud_rate = (fraud_count / max(1, total)) * 100
    ratio_str = (
        f"{round(legit_count / max(1, fraud_count), 2)}:1" if fraud_count > 0 else "N/A"
    )

    return {
        "total_records": total,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_rate, 4),
        "imbalance_ratio": ratio_str,
        "is_imbalanced": fraud_rate < 5.0,  # Highly imbalanced threshold
    }


def analyze_feature_correlations(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute Pearson correlations with the fraud target and inter-feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return {"target_correlations": {}, "top_positive": [], "top_negative": []}

    corr_series = numeric_df.corr()[target_col].drop(index=target_col)
    corr_series = corr_series.dropna()

    sorted_corr = corr_series.sort_values(ascending=False)
    top_pos = [
        {"feature": str(k), "correlation": round(float(v), 4)}
        for k, v in sorted_corr.head(top_n).items()
        if v > 0
    ]
    top_neg = [
        {"feature": str(k), "correlation": round(float(v), 4)}
        for k, v in sorted_corr.tail(top_n).items()
        if v < 0
    ]

    return {
        "all_target_correlations": {
            str(k): round(float(v), 4) for k, v in corr_series.items()
        },
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
    }


def check_target_leakage(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    """Scan feature columns for potential target leakage or near-perfect correlation."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return []

    leakage_suspects: list[dict[str, Any]] = []
    correlations = numeric_df.corr()[target_col].drop(index=target_col).abs()

    for col, corr_val in correlations.items():
        if corr_val >= threshold or np.isnan(corr_val):
            leakage_suspects.append(
                {
                    "column": str(col),
                    "absolute_correlation": round(float(corr_val), 4)
                    if not np.isnan(corr_val)
                    else "NaN",
                    "risk": "High - Near-perfect correlation suggests target leakage or degenerate feature",
                }
            )

    return leakage_suspects


def analyze_time_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze temporal distribution and fraud concentration over elapsed time."""
    if "Time" not in df.columns:
        return {"has_time": False}

    time_col = df["Time"]
    min_sec = float(time_col.min())
    max_sec = float(time_col.max())
    span_hours = (max_sec - min_sec) / 3600

    # Hourly buckets
    hourly_bucket = (time_col // 3600).astype(int)
    counts_by_hour = hourly_bucket.value_counts().sort_index().to_dict()

    result: dict[str, Any] = {
        "has_time": True,
        "min_seconds": min_sec,
        "max_seconds": max_sec,
        "span_hours": round(span_hours, 2),
        "total_hourly_buckets": len(counts_by_hour),
    }

    if TARGET_COLUMN in df.columns:
        fraud_by_hour = (
            df[df[TARGET_COLUMN] == 1].groupby(hourly_bucket).size().to_dict()
        )
        result["fraud_by_hour"] = {int(k): int(v) for k, v in fraud_by_hour.items()}

    return result


def run_full_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Execute complete EDA pipeline and compile structured metrics."""
    return {
        "record_count": len(df),
        "feature_count": df.shape[1],
        "class_imbalance": analyze_class_imbalance(df),
        "correlations": analyze_feature_correlations(df),
        "leakage_risks": check_target_leakage(df),
        "temporal_patterns": analyze_time_patterns(df),
    }


def generate_eda_markdown(
    df: pd.DataFrame, title: str = "Exploratory Data Analysis Report"
) -> str:
    """Generate comprehensive Markdown EDA report for documentation."""
    eda = run_full_eda(df)
    imb = eda["class_imbalance"]
    corr = eda["correlations"]
    leakage = eda["leakage_risks"]
    time_p = eda["temporal_patterns"]

    lines = [
        f"# AuditLens — {title}",
        "",
        "## 1. Overview & Dataset Dimensions",
        f"- **Total Transactions**: {eda['record_count']:,}",
        f"- **Total Features**: {eda['feature_count']}",
        "",
        "## 2. Class Imbalance Analysis",
        f"- **Legitimate Count (`Class = 0`)**: {imb.get('legitimate_count', 0):,}",
        f"- **Fraud Count (`Class = 1`)**: {imb.get('fraud_count', 0):,}",
        f"- **Fraud Percentage**: {imb.get('fraud_percentage', 0)}%",
        f"- **Imbalance Ratio**: {imb.get('imbalance_ratio', 'N/A')}",
        (
            "> **Operational Constraint**: Standard accuracy cannot be used as an evaluation metric. "
            "A naive model predicting legitimate for every transaction would attain 99%+ accuracy while missing all fraud. "
            "PR-AUC (Average Precision), Recall@k, and precision/recall trade-offs must be prioritized."
        ),
        "",
        "## 3. Top Predictive Features (Target Correlation)",
        "### Top Positive Correlators (Elevated in Fraud):",
    ]

    for item in corr.get("top_positive_features", [])[:5]:
        lines.append(f"- **{item['feature']}**: $+{item['correlation']}$")

    lines.append("\n### Top Negative Correlators (Depressed in Fraud):")
    for item in corr.get("top_negative_features", [])[:5]:
        lines.append(f"- **{item['feature']}**: ${item['correlation']}$")

    lines.extend(
        [
            "",
            "## 4. Target Leakage & Post-Outcome Field Audit",
        ]
    )
    if not leakage:
        lines.append(
            "- **Zero target leakage detected**: No feature exhibits suspiciously perfect correlation ($|r| \\ge 0.98$) with the target variable."
        )
    else:
        for leak in leakage:
            lines.append(
                f"- **WARNING - Suspicious Column**: `{leak['column']}` with correlation {leak['absolute_correlation']}"
            )

    if time_p.get("has_time"):
        lines.extend(
            [
                "",
                "## 5. Temporal Patterns & Coverage",
                f"- **Observed Time Window**: {time_p['span_hours']} hours ({time_p['min_seconds']}s to {time_p['max_seconds']}s)",
                "- **Split Strategy Recommendation**: For models evaluating temporal drift, time-based sequential splits (`Time` ordering) should be supported alongside stratified splits.",
                "",
            ]
        )

    return "\n".join(lines)


def save_eda_report(
    df: pd.DataFrame, output_path: str | Path = "./docs/EDA_REPORT.md"
) -> Path:
    """Save EDA markdown report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_md = generate_eda_markdown(df)
    path.write_text(report_md, encoding="utf-8")
    return path
