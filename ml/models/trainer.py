"""Reusable baseline model training pipeline orchestrator."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.data.loader import load_dataset
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.preprocessing.pipeline import AuditLensPreprocessor
from ml.preprocessing.splitter import split_dataset


def train_baseline_models(
    data_path: str | Path,
    output_dir: str | Path = "./models",
    test_size: float = 0.2,
    random_state: int = 42,
    train_anomaly: bool = True,
    train_supervised: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, and persist baseline ML models with complete metadata.

    Args:
        data_path: Path to dataset CSV file.
        output_dir: Directory where trained artifacts and metadata are saved.
        test_size: Proportion of dataset reserved for evaluation.
        random_state: Random seed for reproducibility.
        train_anomaly: Whether to train Isolation Forest anomaly detector.
        train_supervised: Whether to train Logistic Regression classifier.

    Returns:
        dict[str, Any]: Training report including artifact paths and baseline metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df, validation = load_dataset(data_path, is_training=train_supervised)
    if not validation.is_valid:
        raise ValueError(f"Cannot train models on invalid data: {validation.errors}")

    # 2. Partition data safely (deduplicated, stratified)
    split = split_dataset(
        df,
        strategy="stratified",
        test_size=test_size,
        random_state=random_state,
        deduplicate=True,
    )

    # 3. Fit Preprocessing Pipeline strictly on training features
    preprocessor = AuditLensPreprocessor(scale_amount=True, encode_time_cyclic=True)
    preprocessor.fit(split.X_train)
    X_train_trans = preprocessor.transform(split.X_train)
    X_test_trans = preprocessor.transform(split.X_test)

    prep_path = out_dir / "preprocessor.joblib"
    preprocessor.save(prep_path)

    results: dict[str, Any] = {
        "dataset_path": str(data_path),
        "split_summary": split.summary(),
        "random_state": random_state,
        "preprocessor_path": str(prep_path),
        "models": {},
        "baseline_metrics": {},
    }

    # 4. Train Anomaly Detection Baseline
    if train_anomaly:
        anomaly_model = IsolationForestAnomalyDetector(random_state=random_state)
        anomaly_model.fit(X_train_trans)
        anomaly_path = out_dir / "isolation_forest.joblib"
        anomaly_model.save(anomaly_path)

        # Evaluate on test set
        test_scores, test_is_anomaly = anomaly_model.score_transactions(X_test_trans)
        results["models"]["anomaly_detector"] = {
            "version": anomaly_model.model_version,
            "path": str(anomaly_path),
            "test_outlier_rate": round(float(np.mean(test_is_anomaly)), 4),
            "test_mean_score": round(float(np.mean(test_scores)), 4),
        }

    # 5. Train Supervised Baseline
    if train_supervised:
        y_train = split.y_train.to_numpy()
        y_test = split.y_test.to_numpy()

        classifier = LogisticRegressionFraudClassifier(random_state=random_state)
        classifier.fit(X_train_trans, y_train)
        clf_path = out_dir / "logistic_regression.joblib"
        classifier.save(clf_path)

        # Baseline evaluation on test set
        test_probs = classifier.predict_fraud_probabilities(X_test_trans)
        test_preds = classifier.predict(X_test_trans, threshold=0.5)

        pr_auc = float(average_precision_score(y_test, test_probs))
        roc_auc = (
            float(roc_auc_score(y_test, test_probs))
            if len(np.unique(y_test)) > 1
            else None
        )
        prec = float(precision_score(y_test, test_preds, zero_division=0))
        rec = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))

        clf_metrics = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc else None,
            "precision_at_0_5": round(prec, 4),
            "recall_at_0_5": round(rec, 4),
            "f1_at_0_5": round(f1, 4),
        }

        results["models"]["fraud_classifier"] = {
            "version": classifier.model_version,
            "path": str(clf_path),
            "metrics": clf_metrics,
        }
        results["baseline_metrics"] = clf_metrics

    # Save consolidated evaluation summary to JSON
    eval_file = out_dir / "baseline_evaluation.json"
    eval_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["evaluation_report_path"] = str(eval_file)

    return results
