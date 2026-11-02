"""Tests for Isolation Forest anomaly detection baseline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.models.anomaly import IsolationForestAnomalyDetector


def test_fit_and_score_transactions():
    """Verify anomaly detector fits on numeric features and outputs normalized scores."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 31))
    X_test = rng.standard_normal((20, 31))

    detector = IsolationForestAnomalyDetector(n_estimators=50, random_state=42)
    detector.fit(X_train)

    assert detector.is_fitted is True
    scores, is_anomaly = detector.score_transactions(X_test)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (20,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert is_anomaly.dtype == bool
    assert is_anomaly.shape == (20,)


def test_score_unfitted_raises_runtime_error():
    """Verify scoring before fitting raises RuntimeError."""
    detector = IsolationForestAnomalyDetector()
    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.score_transactions(np.ones((5, 10)))


def test_fit_invalid_dimensions():
    """Verify non-2D input raises ValueError."""
    detector = IsolationForestAnomalyDetector()
    with pytest.raises(ValueError, match="Input X must be a 2D array"):
        detector.fit(np.ones(10))


def test_anomaly_model_save_and_load_reproducibility():
    """Verify saved and loaded anomaly model yields identical scores."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 20))

    detector = IsolationForestAnomalyDetector(n_estimators=30, random_state=42)
    detector.fit(X)
    orig_scores, orig_anomalies = detector.score_transactions(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "detector.joblib"
        detector.save(model_path)

        assert model_path.exists()
        assert model_path.with_suffix(".json").exists()

        loaded = IsolationForestAnomalyDetector.load(model_path)
        loaded_scores, loaded_anomalies = loaded.score_transactions(X)

        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(orig_anomalies, loaded_anomalies)
