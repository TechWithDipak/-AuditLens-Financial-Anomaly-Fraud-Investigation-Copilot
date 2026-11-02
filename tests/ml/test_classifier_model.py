"""Tests for Logistic Regression fraud classification baseline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml.models.classifier import LogisticRegressionFraudClassifier


def test_fit_and_predict_probabilities():
    """Verify supervised classifier outputs continuous probabilities in [0.0, 1.0]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 31))
    # Imbalanced labels: 95 legit, 5 fraud
    y = np.zeros(100, dtype=int)
    y[rng.choice(100, size=5, replace=False)] = 1

    clf = LogisticRegressionFraudClassifier(random_state=42)
    clf.fit(X, y)

    assert clf.is_fitted is True
    probs = clf.predict_fraud_probabilities(X)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (100,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    preds = clf.predict(X, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_unfitted_raises_runtime_error():
    """Verify inference before fitting raises RuntimeError."""
    clf = LogisticRegressionFraudClassifier()
    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.predict_fraud_probabilities(np.ones((5, 10)))


def test_fit_single_class_raises_value_error():
    """Verify fitting on dataset with only one class raises ValueError."""
    X = np.ones((10, 5))
    y = np.zeros(10, dtype=int)  # Only class 0

    clf = LogisticRegressionFraudClassifier()
    with pytest.raises(ValueError, match="requires both classes"):
        clf.fit(X, y)


def test_classifier_save_and_load_reproducibility():
    """Verify saved and loaded classifier produces identical predictions."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 15))
    y = np.zeros(60, dtype=int)
    y[:10] = 1

    clf = LogisticRegressionFraudClassifier(random_state=42)
    clf.fit(X, y)
    orig_probs = clf.predict_fraud_probabilities(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "clf.joblib"
        clf.save(model_path)

        assert model_path.exists()
        assert model_path.with_suffix(".json").exists()

        loaded = LogisticRegressionFraudClassifier.load(model_path)
        loaded_probs = loaded.predict_fraud_probabilities(X)

        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-6, atol=1e-6)
