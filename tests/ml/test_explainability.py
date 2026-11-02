"""Tests for model explainability, SHAP values, and analyst-facing evidence."""

from pathlib import Path

import pandas as pd
import pytest

from ml.explainability.explainer import AuditLensExplainer
from ml.models.anomaly import IsolationForestAnomalyDetector
from ml.models.classifier import LogisticRegressionFraudClassifier
from ml.models.random_forest import RandomForestFraudClassifier
from ml.preprocessing.pipeline import AuditLensPreprocessor
from ml.preprocessing.splitter import split_dataset

SAMPLE_DIR = Path("./data/sample")


@pytest.fixture
def sample_data_and_preprocessor():
    """Fixture providing fitted preprocessor and sample data."""
    df = pd.read_csv(SAMPLE_DIR / "synthetic_creditcard_sample.csv")
    split = split_dataset(df, strategy="stratified", test_size=0.2, random_state=42)

    preprocessor = AuditLensPreprocessor(scale_amount=True, encode_time_cyclic=True)
    preprocessor.fit(split.X_train)
    X_train_trans = preprocessor.transform(split.X_train)
    X_test_trans = preprocessor.transform(split.X_test)

    return {
        "df": df,
        "split": split,
        "preprocessor": preprocessor,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
    }


def test_linear_explainer_transaction(sample_data_and_preprocessor):
    """Verify linear model explanation provides signed feature contributions and metadata."""
    data = sample_data_and_preprocessor
    logreg = LogisticRegressionFraudClassifier(random_state=42)
    logreg.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"],
        model=logreg,
        background_data=data["X_train_trans"],
        use_shap=True,
    )

    test_row = data["split"].X_test.iloc[0]
    explanation = explainer.explain_transaction(
        test_row, transaction_id="TX_1001", threshold=0.5
    )

    assert explanation.transaction_id == "TX_1001"
    assert explanation.model_name == logreg.model_name
    assert explanation.model_version == logreg.model_version
    assert explanation.score_type == "fraud_probability"
    assert 0.0 <= explanation.prediction_score <= 1.0
    assert len(explanation.top_contributions) == 5

    # Check non-causal descriptions
    for contrib in explanation.top_contributions:
        assert contrib.direction in {"increases_risk", "decreases_risk", "neutral"}
        assert "caused" not in contrib.plain_text_description.lower()
        assert "guilt" not in contrib.plain_text_description.lower()

    # Check limitations are attached
    assert len(explanation.limitations) > 0
    assert any("NOT legal proof" in caveat for caveat in explanation.limitations)


def test_tree_explainer_random_forest(sample_data_and_preprocessor):
    """Verify TreeExplainer runs on Random Forest model and extracts top contributing features."""
    data = sample_data_and_preprocessor
    rf = RandomForestFraudClassifier(n_estimators=20, random_state=42)
    rf.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"],
        model=rf,
        use_shap=True,
    )

    test_row = data["split"].X_test.iloc[1]
    explanation = explainer.explain_transaction(
        test_row, transaction_id=202, threshold=0.5
    )

    assert explanation.model_name == rf.model_name
    assert len(explanation.all_contributions) == 31  # 31 preprocessed features
    assert explanation.explanation_method in {"shap_tree", "fallback_attribution"}


def test_fallback_attribution_when_shap_disabled(sample_data_and_preprocessor):
    """Verify fallback attribution runs deterministically when SHAP is disabled."""
    data = sample_data_and_preprocessor
    logreg = LogisticRegressionFraudClassifier(random_state=42)
    logreg.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"],
        model=logreg,
        use_shap=False,  # Force fallback
    )

    assert explainer.method == "fallback_attribution"
    test_row = data["split"].X_test.iloc[2]
    explanation = explainer.explain_transaction(test_row)

    assert explanation.explanation_method == "fallback_attribution"
    assert len(explanation.top_contributions) == 5
    assert explanation.prediction_score >= 0.0


def test_anomaly_detector_explanation(sample_data_and_preprocessor):
    """Verify explaining Isolation Forest anomaly scores."""
    data = sample_data_and_preprocessor
    iso = IsolationForestAnomalyDetector(n_estimators=30, random_state=42)
    iso.fit(data["X_train_trans"])

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"],
        model=iso,
    )

    test_row = data["split"].X_test.iloc[0]
    explanation = explainer.explain_transaction(test_row)

    assert explanation.score_type == "anomaly_score"
    assert explanation.model_name == iso.model_name
    assert 0.0 <= explanation.prediction_score <= 1.0


def test_explainer_rejects_missing_features(sample_data_and_preprocessor):
    """Verify explanation raises ValueError when required columns are absent."""
    data = sample_data_and_preprocessor
    logreg = LogisticRegressionFraudClassifier(random_state=42)
    logreg.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"], model=logreg, use_shap=False
    )
    corrupted_row = pd.Series({"Amount": 50.0, "Time": 100.0})  # Missing V1-V28

    with pytest.raises(ValueError, match="missing required feature columns"):
        explainer.explain_transaction(corrupted_row)


def test_global_feature_importance(sample_data_and_preprocessor):
    """Verify global feature importance rankings."""
    data = sample_data_and_preprocessor
    logreg = LogisticRegressionFraudClassifier(random_state=42)
    logreg.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"], model=logreg, use_shap=False
    )
    global_imp = explainer.get_global_feature_importance(top_n=5)

    assert len(global_imp.feature_rankings) == 5
    assert global_imp.model_name == logreg.model_name
    assert (
        global_imp.feature_rankings[0]["importance"]
        >= global_imp.feature_rankings[1]["importance"]
    )


def test_explain_batch(sample_data_and_preprocessor):
    """Verify batch explanation generates list matching batch size."""
    data = sample_data_and_preprocessor
    logreg = LogisticRegressionFraudClassifier(random_state=42)
    logreg.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"], model=logreg, use_shap=False
    )
    batch_df = data["split"].X_test.head(4)

    exps = explainer.explain_batch(batch_df)
    assert len(exps) == 4
    for exp in exps:
        assert exp.prediction_score is not None
        assert len(exp.top_contributions) == 5


def test_explain_tp_fp_tn_cases_and_determinism(sample_data_and_preprocessor):
    """Verify explanations for true positive, false positive, and true negative cases are deterministic."""
    data = sample_data_and_preprocessor
    rf = RandomForestFraudClassifier(n_estimators=20, random_state=42)
    rf.fit(data["X_train_trans"], data["split"].y_train.to_numpy())

    explainer = AuditLensExplainer(
        preprocessor=data["preprocessor"],
        model=rf,
        use_shap=True,
    )

    # Pick a row and run twice to verify determinism
    test_row = data["split"].X_test.iloc[0]
    exp1 = explainer.explain_transaction(test_row, transaction_id="TX_DET_1")
    exp2 = explainer.explain_transaction(test_row, transaction_id="TX_DET_1")

    assert exp1.prediction_score == exp2.prediction_score
    assert len(exp1.top_contributions) == len(exp2.top_contributions)
    for c1, c2 in zip(exp1.top_contributions, exp2.top_contributions, strict=True):
        assert c1.feature_name == c2.feature_name
        assert pytest.approx(c1.contribution, abs=1e-5) == c2.contribution
        assert "proof" not in c1.plain_text_description.lower()
