"""Tests for unified inference service and prediction output contracts."""

from pathlib import Path

import pandas as pd
import pytest

from ml.models.inference import AuditLensInferenceEngine

SAMPLE_DIR = Path("./data/sample")
MODELS_DIR = Path("./models")


def test_inference_engine_from_directory():
    """Verify loading preprocessor and models from directory and scoring transactions."""
    engine = AuditLensInferenceEngine.load_from_directory(MODELS_DIR)

    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    records = engine.score_batch(df.head(10))

    assert len(records) == 10
    for rec in records:
        # Crucial architectural test: anomaly score and fraud probability must be separate fields
        assert isinstance(rec.anomaly_score, float)
        assert 0.0 <= rec.anomaly_score <= 1.0
        assert isinstance(rec.is_anomaly, bool)
        assert isinstance(rec.fraud_probability, float)
        assert 0.0 <= rec.fraud_probability <= 1.0
        assert rec.predicted_class in {0, 1}
        assert rec.anomaly_model_version == "iso_forest_v0.1.0"
        assert rec.classifier_model_version == "logreg_v0.1.0"


def test_inference_engine_score_single():
    """Verify single row scoring works seamlessly."""
    engine = AuditLensInferenceEngine.load_from_directory(MODELS_DIR)
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    row_dict = df.iloc[0].to_dict()

    rec = engine.score_single(row_dict)
    assert rec.anomaly_score >= 0.0
    assert rec.fraud_probability is not None


def test_inference_engine_rejects_missing_features():
    """Verify inference engine raises ValueError if required columns are missing."""
    engine = AuditLensInferenceEngine.load_from_directory(MODELS_DIR)
    corrupted_df = pd.DataFrame({"Amount": [10.0], "Time": [100.0]})

    with pytest.raises(ValueError, match="missing required features"):
        engine.score_batch(corrupted_df)
