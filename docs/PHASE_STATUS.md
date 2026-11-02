# Phase 4 — Baseline Machine-Learning Models

## Changes made
- Created baseline models architecture in `ml/models/`:
  - `ml/models/base.py`: Defined `PredictionRecord` ensuring strict decoupling between unsupervised anomaly scores and supervised fraud probabilities. Defined `ModelMetadata` for full provenance and hyperparameter tracking.
  - `ml/models/anomaly.py`: Implemented `IsolationForestAnomalyDetector` (`iso_forest_v0.1.0`), fitting strictly on unlabeled preprocessed features with continuous normalized $[0.0, 1.0]$ score output and binary outlier flags.
  - `ml/models/classifier.py`: Implemented `LogisticRegressionFraudClassifier` (`logreg_v0.1.0`) with `class_weight='balanced'`, predicting calibrated continuous fraud probabilities $P(\text{Class}=1 \mid X) \in [0.0, 1.0]$.
  - `ml/models/trainer.py`: Implemented automated training pipeline executing dataset loading, stratified splitting, preprocessing fit, model training, held-out test evaluation, and artifact serialization.
  - `ml/models/inference.py`: Implemented `AuditLensInferenceEngine` supporting batch and single-transaction scoring with input validation and typed prediction contracts.
- Persisted baseline model artifacts in `models/` with companion JSON metadata:
  - `models/isolation_forest.joblib` and `models/isolation_forest.json`
  - `models/logistic_regression.joblib` and `models/logistic_regression.json`
  - `models/preprocessor.joblib` and `models/preprocessor.json`
  - `models/baseline_evaluation.json`
- Documented model architectures, intended decision-support usage, feature inputs, and baseline metrics in `docs/MODEL_CARD.md`.
- Implemented comprehensive unit and integration test suite across anomaly detector, supervised classifier, inference engine, and trainer.

## Files created or modified
- `models/.gitkeep`
- `docs/MODEL_CARD.md`
- `ml/models/__init__.py`
- `ml/models/base.py`
- `ml/models/anomaly.py`
- `ml/models/classifier.py`
- `ml/models/trainer.py`
- `ml/models/inference.py`
- `tests/ml/test_anomaly_model.py`
- `tests/ml/test_classifier_model.py`
- `tests/ml/test_inference_engine.py`
- `tests/ml/test_trainer.py`
- `docs/project-memory/02_EXECUTION_LOG.md`
- `docs/project-memory/03_LESSONS_LEARNED.md`
- `docs/PHASE_STATUS.md`

## Verification performed
| Check | Command / Method | Result |
|---|---|---|
| Backend & ML Unit tests | `.venv/bin/pytest -v` | **PASS** (58 passed in 1.74s) |
| Backend & ML Linting | `.venv/bin/ruff check backend ml tests` | **PASS** (All checks passed) |
| Frontend tests & build | `cd frontend && npm test && npm run lint && npm run build` | **PASS** (Zero regression) |
| Anomaly score bounds test | `test_fit_and_score_transactions` | **PASS** (Scores strictly in $[0.0, 1.0]$) |
| Classifier probability test | `test_fit_and_predict_probabilities` | **PASS** (Probabilities strictly in $[0.0, 1.0]$) |
| Score decoupling contract test | `test_inference_engine_from_directory` | **PASS** (`anomaly_score` and `fraud_probability` verified as separate fields) |
| Artifact persistence reproducibility | `test_anomaly_model_save_and_load_reproducibility`, `test_classifier_save_and_load_reproducibility` | **PASS** (Exact bit-for-bit prediction match on reload) |
| Missing feature rejection test | `test_inference_engine_rejects_missing_features` | **PASS** (ValueError raised on incomplete feature vectors) |
| End-to-end training pipeline test | `test_train_baseline_models_pipeline` | **PASS** (Artifacts generated, baseline metrics recorded) |
| Model artifact git isolation | `git status -s --ignored models/` | **PASS** (Large `.joblib` artifacts ignored by Git) |

## Issues and limitations
- Logistic Regression is a linear classifier; non-linear models (e.g. Random Forest or gradient-boosted trees) and detailed operational threshold curves are deferred to Phase 5.
- Isolation Forest scores represent statistical distance, not proven fraud.

## Exit gate
**PASS**

## Next phase
Phase 5 — Model evaluation, comparison, and threshold selection.
Do not begin until the user explicitly approves.
