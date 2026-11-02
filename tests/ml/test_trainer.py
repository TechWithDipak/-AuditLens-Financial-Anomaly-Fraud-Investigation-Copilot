"""Tests for baseline model trainer workflow."""

import tempfile
from pathlib import Path

from ml.models.trainer import train_baseline_models

SAMPLE_DATA = Path("./data/sample/synthetic_creditcard_sample.csv")


def test_train_baseline_models_pipeline():
    """Verify train_baseline_models executes end-to-end and outputs artifacts and metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = train_baseline_models(
            data_path=SAMPLE_DATA,
            output_dir=tmpdir,
            test_size=0.2,
            random_state=42,
            train_anomaly=True,
            train_supervised=True,
        )

        assert "models" in results
        assert "anomaly_detector" in results["models"]
        assert "fraud_classifier" in results["models"]
        assert "baseline_metrics" in results

        metrics = results["baseline_metrics"]
        assert "pr_auc" in metrics
        assert "f1_at_0_5" in metrics

        # Verify artifacts exist on disk
        out_path = Path(tmpdir)
        assert (out_path / "preprocessor.joblib").exists()
        assert (out_path / "isolation_forest.joblib").exists()
        assert (out_path / "logistic_regression.joblib").exists()
        assert (out_path / "baseline_evaluation.json").exists()
