# Phase 6 — Explainability and Analyst-Facing Evidence

## Changes made
- Implemented explainability data contracts in `ml/explainability/schemas.py`:
  - `FeatureContribution`: Encapsulates individual feature attributions, signed directional indicators (`increases_risk`, `decreases_risk`, `neutral`), raw/transformed values, relative normalized magnitudes, and non-causal descriptions.
  - `TransactionExplanation`: Encapsulates full transaction explanation report containing model metadata (name, version), prediction score, score type (`fraud_probability` vs `anomaly_score`), threshold, base expected value, top-$k$ contributions, all feature contributions, plain-language summary, and regulatory disclaimers/caveats.
  - `GlobalFeatureImportance`: Encapsulates model-wide feature ranking and importance metrics.
- Implemented explainability engine `AuditLensExplainer` in `ml/explainability/explainer.py`:
  - Native SHAP integration with model-specific explainers: `shap.LinearExplainer` for linear models (`LogisticRegressionFraudClassifier`), `shap.TreeExplainer` for tree ensembles (`RandomForestFraudClassifier`).
  - Robust mathematical fallback attribution engine using standardized feature deviations when SHAP is disabled or unsupported.
  - Transparent anomaly detector explanation: attributes isolation forest anomaly scores by measuring feature divergence from expected normal background centroids.
  - Plain-language non-causal descriptions enforcing strict compliance guidelines: phrases state that features "increased/decreased model score" rather than claiming "proof of fraud" or "guilt".
  - Metadata and operational guardrails: every explanation attaches the exact model version, decision threshold, scoring timestamp, and standard regulatory caveats.
  - Global feature importance computation across model coefficients or tree importance vectors.
  - Batch explanation method `explain_batch` for high-throughput batch auditing.
- Implemented unit tests in `tests/ml/test_explainability.py` (8 test cases):
  - Linear model SHAP explanations and signed attribution validation.
  - Random Forest tree SHAP explanations.
  - Deterministic fallback attribution when SHAP is disabled.
  - Isolation Forest anomaly score explanation.
  - Schema validation rejecting inputs missing required features.
  - Global feature importance ordering.
  - Batch transaction explanation consistency.
  - Determinism across repeated explanation evaluations on identical inputs.

## Files created or modified
- `ml/explainability/__init__.py`
- `ml/explainability/schemas.py`
- `ml/explainability/explainer.py`
- `tests/ml/test_explainability.py`
- `docs/project-memory/02_EXECUTION_LOG.md`
- `docs/project-memory/03_LESSONS_LEARNED.md`
- `docs/PHASE_STATUS.md`

## Verification performed
| Check | Command / Method | Result |
|---|---|---|
| Full backend & ML test suite | `.venv/bin/pytest -v` | **PASS** (74 passed in 2.66s) |
| Backend & ML Linting & Formatting | `.venv/bin/ruff check backend ml tests && .venv/bin/ruff format --check backend ml tests` | **PASS** (Zero errors) |
| Frontend tests & build | `cd frontend && npm test && npm run lint && npm run build` | **PASS** (Zero regression) |
| Linear explainer test | `test_linear_explainer_transaction` | **PASS** (Signed contributions, score in $[0.0, 1.0]$, disclaimers attached) |
| Tree explainer test | `test_tree_explainer_random_forest` | **PASS** (TreeExplainer runs, generates 31 feature attributions) |
| Fallback attribution test | `test_fallback_attribution_when_shap_disabled` | **PASS** (Runs without SHAP, top 5 features extracted) |
| Anomaly explanation test | `test_anomaly_detector_explanation` | **PASS** (Attributes Isolation Forest score correctly as `anomaly_score`) |
| Missing feature rejection | `test_explainer_rejects_missing_features` | **PASS** (Rejects missing columns with clear `ValueError`) |
| Global feature ranking | `test_global_feature_importance` | **PASS** (Sorted descending by global attribution weight) |
| Batch explanation test | `test_explain_batch` | **PASS** (All rows in batch explained) |
| Determinism test | `test_explain_tp_fp_tn_cases_and_determinism` | **PASS** (Exact reproducible scores & attributions on repeated runs) |

## Issues and limitations
- For PCA-transformed columns `V1`–`V28`, feature explanations provide mathematical directional evidence (e.g. `V14 = -4.21 increased model score by +0.32`) rather than human-domain business semantics (e.g. merchant type). In production systems with un-anonymized features, semantic feature dictionaries will map raw feature names to human-readable domain descriptors.
- Explanations are strictly decision-support aids and do not constitute legal proof of fraud or regulatory culpability.

## Exit gate
**PASS**

## Next phase
Phase 7 — Backend API and Case Management.
Do not begin until the user explicitly approves.
