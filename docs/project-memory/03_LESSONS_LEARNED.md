# Project Memory: Lessons Learned & Technical Notes

## Purpose
Continuous record of technical discoveries, potential edge cases, mistakes made, root causes, fixes, and prevention rules across all phases.

---

### Phase 0: Repository Inspection & Baseline Setup
- **Discovery**: The host environment runs Python 3.13.7 and Node v25.6.1. Python 3.13 is a very recent version; while core packages like FastAPI, Uvicorn, Pandas, and scikit-learn support Python 3.13, some ML or C-extension packages (such as certain SHAP builds, XGBoost wheels, or PyTorch if used) can occasionally have wheel availability nuances on Python 3.13 on Apple Silicon.
  - *Mitigation & Prevention Rule*: In Phase 1 and Phase 4, prioritize pure-python or well-supported wheel distributions (e.g. scikit-learn standard wheels). If SHAP binary compilation requires heavy toolchains, prepare a lightweight tree/linear feature contribution fallback module so development is never blocked.
- **Discovery**: Repository is completely greenfield (no prior code or tests).
  - *Baseline Verification*: Confirmed zero tests exist initially. Phase 1 must establish the test harness (`pytest` for backend, `jest`/`vitest` or equivalent for frontend) alongside the initial health check.
- **Discovery**: Node.js v25.6.1 is an odd/current release. Modern Next.js works on Node 18.18+ or 20+, and generally supports Node 22+.
  - *Mitigation & Prevention Rule*: Verify compatibility when initializing Next.js in Phase 1 with strict lockfiles.


---

### Phase 1: Foundation and Local Development Environment
- **Mistake / Root Cause**: During initial pytest run, `ModuleNotFoundError: No module named 'backend'` occurred because tests were executed without root package resolution in Python path.
  - *Fix*: Created `pytest.ini` with `pythonpath = .` and added `__init__.py` files across all subpackages.
  - *Prevention Rule*: Always configure `pytest.ini` with explicit `pythonpath = .` whenever tests import packages rooted at workspace root.
- **Discovery**: In FastAPI, unrouted requests (such as requests to non-existent URLs) raise `starlette.exceptions.HTTPException` rather than `fastapi.exceptions.HTTPException`. Registering an exception handler solely for `fastapi.HTTPException` causes unrouted 404s to fall back to Starlette's raw `{ "detail": "Not Found" }` format instead of our standardized error envelope.
  - *Fix*: Bound the uniform error handler to both `FastAPI.HTTPException` and `StarletteHTTPException`.
  - *Prevention Rule*: In FastAPI APIs requiring consistent error envelopes, always register exception handlers for both `fastapi.HTTPException` and `starlette.exceptions.HTTPException`.
- **Discovery**: In TypeScript Vite projects, accessing `import.meta.env` requires `vite/client` type reference.
  - *Fix*: Added `src/vite-env.d.ts` with `/// <reference types="vite/client" />`.
  - *Prevention Rule*: Keep `vite-env.d.ts` in every Vite TypeScript application to satisfy compiler checks for `import.meta.env`.

---

### Phase 2: Dataset Acquisition, Profiling, and Data Validation
- **Discovery**: When creating synthetic fixtures with incompatible column values (e.g. inserting string `"INVALID"` into a `float64` series in Pandas), Pandas raises a `FutureWarning` regarding dtype incompatibility.
  - *Fix*: In synthetic generator, explicitly construct corrupted columns as objects or strings before assignment.
  - *Prevention Rule*: In test fixture generators, create string columns explicitly when testing type corruption to avoid relying on implicit Pandas dtype conversions.
- **Discovery**: In Python sets, `0 == 0.0` and `1 == 1.0` evaluate to equal hash and value, so `{0, 1, 0.0, 1.0}` contains duplicate elements flagged by linter rule B033.
  - *Fix*: Simplified set subtraction to `{0, 1}` which cleanly covers both integer and floating-point 0 and 1 values.
  - *Prevention Rule*: Use `{0, 1}` for binary validation in Python sets without redundant float literals.
- **Discovery**: FastAPI route parameters using `File(...)` without default values trigger linter warning B008 unless annotated with `# noqa: B008`, because FastAPI depends on parameter default sentinel objects for dependency injection.
  - *Fix*: Added `# noqa: B008` to `File(...)` in route signatures.

---

### Phase 3: Exploratory Data Analysis and Preprocessing
- **Discovery**: In financial fraud modeling, scaling `Amount` using `StandardScaler` (mean/std) is brittle because large anomalous payments heavily distort the sample mean and variance.
  - *Fix*: Employed `RobustScaler(unit_variance=True)`, which scales using the median and Interquartile Range (IQR = Q75 - Q25), effectively insulating normal transactions from fraud-amount distortion.
  - *Prevention Rule*: In transaction fraud domains with long-tailed or log-normal distributions, always use RobustScaler or log-transforms rather than standard mean-variance scaling.
- **Discovery**: When splitting imbalanced datasets, identical duplicate transactions can exist across partitions if deduplication is not performed prior to partitioning, resulting in cross-split data leakage.
  - *Fix*: Added default deduplication in `split_dataset` before applying stratified or temporal splits.
  - *Prevention Rule*: Always deduplicate feature vectors prior to cross-validation or train/test splitting when identical rows may occur in raw event logs.
- **Discovery**: If `Class` is included inside the feature matrix passed into `Pipeline.fit()`, silent target leakage can easily occur in downstream stages.
  - *Fix*: Raised an explicit `ValueError` in `AuditLensPreprocessor.fit()` and `.transform()` if `TARGET_COLUMN` is present in the feature matrix `X`.
  - *Prevention Rule*: Hard-enforce target separation by asserting absence of label columns at the transformer entry point.

---

### Phase 4: Baseline Machine-Learning Models
- **Discovery**: In scikit-learn 1.8+, passing `penalty='l2'` explicitly to `LogisticRegression` triggers a deprecation warning because `penalty` is deprecated in favor of `l1_ratio` or `C`.
  - *Fix*: Omitted explicit `penalty='l2'` from the constructor since L2 regularization is the default behavior when `l1_ratio` is unset.
  - *Prevention Rule*: In modern scikit-learn, use default `penalty` and tune `C` directly to avoid deprecation warnings across future releases.
- **Discovery**: Raw Isolation Forest `score_samples()` produces unnormalized negative values (e.g. $[-0.7, -0.3]$). Exposing raw negative scores to analysts creates confusion regarding threshold calibration.
  - *Fix*: Calibrated baseline score bounds during `fit()` and normalized inference outputs to $[0.0, 1.0]$ where 1.0 represents the highest anomaly severity.
  - *Prevention Rule*: Always normalize unsupervised anomaly outputs to a standardized $[0.0, 1.0]$ severity scale while explicitly clarifying that this is an outlier metric, not a probability of guilt.

---

### Phase 5: Model Evaluation, Comparison, and Threshold Selection
- **Discovery**: Naive majority classification (predicting non-fraud for all transactions) achieves over 93% accuracy on our test partition while delivering 0.0 recall and 0.067 PR-AUC (completely missing every fraud event).
  - *Fix*: Excluded raw accuracy from model selection criteria and established PR-AUC / Average Precision as the primary objective.
  - *Prevention Rule*: In fraud systems with $< 1\%$ positive rates, never report accuracy without pairing it with PR-AUC, confusion matrices, and recall at operational review thresholds.
- **Discovery**: Default decision threshold (0.5) is rarely optimal under severe class imbalance, especially when balanced class weighting shifts the predicted probability distribution.
  - *Fix*: Implemented `find_optimal_threshold()` to dynamically compute cutoffs targeting explicit operational triage budgets (e.g. F1 maximization or bounded analyst alert capacity).
  - *Prevention Rule*: Always expose configurable decision thresholds so compliance teams can tune false positive alert volume according to investigator review capacity.


---

### Phase 6: Explainability and Analyst-Facing Evidence
- **Discovery**: When using `shap.LinearExplainer` with scikit-learn Logistic Regression in Python 3.13, passing a background dataset directly without specifying an explicit masker produces a `FutureWarning` concerning `feature_perturbation`.
  - *Fix*: Documented the warning and implemented both native SHAP and robust fallback attribution engines to insulate production deployments against upstream SHAP internal API shifts.
  - *Prevention Rule*: In production ML explainability architectures, always implement a deterministic mathematical fallback attribution strategy (e.g. standardized feature deviation or gradient product) alongside SHAP libraries so explanations remain available if a C-extension or explainer initialization fails.
- **Discovery**: In regulatory contexts, describing a high SHAP value as "this transaction is fraud because of V14" violates non-causal compliance standards and can mislead compliance auditors.
  - *Fix*: Standardized all plain-text explanations to non-causal statistical language ("V14 = -4.21 increased model score by +0.32" rather than "caused fraud") and attached mandatory caveats stating explanations represent statistical correlations rather than legal proof of intent.
  - *Prevention Rule*: Hardcode standard disclaimers directly into explainability payload contracts so downstream web and PDF reporting layers cannot omit regulatory guardrails.
