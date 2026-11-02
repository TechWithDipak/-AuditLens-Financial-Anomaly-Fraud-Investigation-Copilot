"""AuditLens Model Evaluation and Comparison Package."""

from ml.evaluation.comparison import (
    analyze_prediction_errors,
    evaluate_model_suite,
)
from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_threshold_curve,
    find_optimal_threshold,
)
from ml.evaluation.reporter import (
    format_evaluation_markdown,
    save_evaluation_reports,
)

__all__ = [
    "analyze_prediction_errors",
    "calculate_classification_metrics",
    "calculate_threshold_curve",
    "evaluate_model_suite",
    "find_optimal_threshold",
    "format_evaluation_markdown",
    "save_evaluation_reports",
]
