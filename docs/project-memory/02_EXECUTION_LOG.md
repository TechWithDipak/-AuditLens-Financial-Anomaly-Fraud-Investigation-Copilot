# Project Memory: Execution Log

## Format Reference
- Entry: Timestamp, Phase, Action, Files Touched, Commands Run, Verification Result, Deviations / Noticed Items.

---

### Phase 0: Repository Inspection & Technical Plan
- **Date**: 2026-10-31
- **Status**: Completed (PASS)
- **Goal**: Inspect existing repository, identify system toolchains, define architecture and directory structure, assess technical risks and dependencies, and establish baseline test status.

#### Actions & Commands Executed:
1. `ls -la`: Checked root workspace contents.
   - *Result*: Found only `.git` and `AUDITLENS_PHASED_IMPLEMENTATION_PLAN.md`.
2. `git status && git log -n 5 --oneline`: Checked git status and revision history.
   - *Result*: Branch `main`, no commits yet, untracked `AUDITLENS_PHASED_IMPLEMENTATION_PLAN.md`.
3. `python3 --version && which python3 && node --version && which node && npm --version && which npm`:
   - *Result*: Python 3.13.7 (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`), Node v25.6.1 (`/opt/homebrew/bin/node`), npm 11.9.0.
4. `python3 -m pytest --version`: Checked if pytest exists in base python.
   - *Result*: No module named pytest in base Python. Confirmed Python virtual environment will be required in Phase 1.
5. `which sqlite3 && sqlite3 --version && python3 -c "import venv; print('venv available')"`:
   - *Result*: SQLite 3.54.0 installed, `venv` module verified available.
6. `git log --all --full-history -- .`:
   - *Result*: Confirmed no previous commits or leaked secrets exist in git history.
7. Baseline test run check:
   - *Result*: No existing test suites or test files in repository. Baseline test count: 0 passed, 0 failed, 0 skipped.

#### Files Created:
- `docs/project-memory/01_PROBLEM_STATEMENT.md`: Documented project context, goals, scope, constraints, and success criteria.
- `docs/project-memory/02_EXECUTION_LOG.md`: Initialized execution log.
- `docs/project-memory/03_LESSONS_LEARNED.md`: Initialized lessons learned and environment notes.
- `docs/PROJECT_PLAN.md`: Comprehensive breakdown of all project phases and delivery milestones.
- `docs/ARCHITECTURE.md`: Detailed system architecture, component boundaries, and data flow.
- `docs/DECISIONS.md`: Initial architecture decision records (ADRs).
- `.env.example`: Secure environment variable template with placeholders only.
- `.gitignore`: Standard gitignore preventing commits of secrets, virtual environments, node modules, raw data, and model artifacts.
- `docs/PHASE_STATUS.md`: Phase 0 exit gate report formatted according to implementation plan specifications.

#### Deviations from Plan:
- None.

#### Noticed Items (Outside Scope for Phase 0):
- Base Python 3.13 environment has minimal packages (`pip 26.1.2`). A self-contained virtual environment (`.venv`) should be created during Phase 1 to house FastAPI, scikit-learn, pytest, etc.
- Node.js version is v25.6.1; will verify Next.js / React compatibility when bootstrapping frontend in Phase 1.


---

### Phase 1: Project Foundation and Local Development Environment
- **Date**: 2026-10-31
- **Status**: Completed (PASS)
- **Goal**: Create a clean, reproducible project skeleton for backend and frontend that starts locally with safe configuration, health-check endpoint, structured logging with PII masking, error envelopes, and automated tests.

#### Actions & Commands Executed:
1. `python3 -m venv .venv && .venv/bin/pip install --upgrade pip`: Initialized isolated Python 3.13 virtual environment.
2. Created `backend/requirements.txt`: Specified FastAPI, Uvicorn, Pydantic, pydantic-settings, python-dotenv, httpx, pytest, pytest-asyncio.
3. `.venv/bin/pip install -r backend/requirements.txt`: Installed backend dependencies.
4. Created backend structure:
   - `backend/app/main.py`: Main FastAPI entrypoint with CORS, error handlers, and lifespan logging.
   - `backend/app/core/config.py`: Central Pydantic Settings with safe defaults.
   - `backend/app/core/logging.py`: Structured logging with `SensitiveDataMaskingFilter` masking PAN cards, SSNs, passwords, CVVs, and Bearer tokens.
   - `backend/app/core/errors.py`: Standardized JSON error response envelope handling `APIError`, `HTTPException`, `StarletteHTTPException`, and unhandled 500 exceptions.
   - `backend/app/api/v1/health.py`: `GET /health` and `GET /api/v1/health` returning system health status.
5. Created backend tests:
   - `tests/backend/test_health.py`: Health endpoint status and schema tests.
   - `tests/backend/test_logging.py`: Sensitive data masking filter tests.
   - `tests/backend/test_errors.py`: Error envelope and internal exception shielding tests.
6. `.venv/bin/pip install ruff`: Installed Ruff linter and code formatter.
7. Fixed import collection and Starlette 404 error handler:
   - Added `pytest.ini` with `pythonpath = .`
   - Added `StarletteHTTPException` handler in `backend/app/core/errors.py` to ensure unrouted 404s match error response envelope.
8. `.venv/bin/ruff check backend tests` and `.venv/bin/pytest -v`:
   - *Result*: 10 passed in 0.14s; all Ruff checks passed.
9. Created frontend structure (`frontend/`):
   - `frontend/package.json`: React 18, TypeScript, Vite.
   - `frontend/vite.config.ts`: Proxying `/api` and `/health` to backend on port 8000.
   - `frontend/src/services/api.ts`: API client with typed health response.
   - `frontend/src/components/Navbar.tsx`: Header navigation with live backend status indicator.
   - `frontend/src/App.tsx`: Analyst copilot shell and layout.
   - `frontend/src/styles.css`: Dark-mode analyst dashboard styling.
   - `frontend/src/vite-env.d.ts`: Vite TypeScript definitions.
   - `frontend/test/api.test.js`: Frontend unit test suite.
10. `npm install` in `frontend/`:
    - *Result*: 67 packages installed cleanly in 5 seconds.
11. `npm run build` and `npm test` and `npm run lint`:
    - *Result*: Production build completed in 234ms; 2 tests passed; lint/type check clean.
12. Live smoke tests:
    - Backend: Started uvicorn on `http://127.0.0.1:8000`. Verified `GET /health`, `GET /api/v1/health`, and `GET /non-existent`.
    - Frontend: Verified preview server serving `http://localhost:3000`.
13. Updated `README.md` with full local setup instructions.

#### Files Created / Modified:
- `backend/requirements.txt`
- `backend/__init__.py`
- `backend/app/__init__.py`
- `backend/app/core/__init__.py`
- `backend/app/core/config.py`
- `backend/app/core/logging.py`
- `backend/app/core/errors.py`
- `backend/app/api/__init__.py`
- `backend/app/api/v1/__init__.py`
- `backend/app/api/v1/health.py`
- `backend/app/main.py`
- `pytest.ini`
- `tests/__init__.py`
- `tests/backend/__init__.py`
- `tests/backend/test_health.py`
- `tests/backend/test_logging.py`
- `tests/backend/test_errors.py`
- `frontend/package.json`
- `frontend/tsconfig.json`
- `frontend/vite.config.ts`
- `frontend/index.html`
- `frontend/.env.example`
- `frontend/src/styles.css`
- `frontend/src/vite-env.d.ts`
- `frontend/src/services/api.ts`
- `frontend/src/components/Navbar.tsx`
- `frontend/src/App.tsx`
- `frontend/src/main.tsx`
- `frontend/test/api.test.js`
- `README.md`
- `docs/project-memory/02_EXECUTION_LOG.md`
- `docs/project-memory/03_LESSONS_LEARNED.md`
- `docs/PHASE_STATUS.md`

#### Deviations from Plan:
- None.

---

### Phase 2: Dataset Acquisition, Profiling, and Data Validation
- **Date**: 2026-10-31
- **Status**: Completed (PASS)
- **Goal**: Establish a trustworthy data input and profiling pipeline before training models, document dataset provenance and schema, build a repeatable loader and validation layer, create synthetic test fixtures, and implement safe CSV upload endpoints with error rejection.

#### Actions & Commands Executed:
1. Updated `backend/requirements.txt` with `pandas>=2.2.0`, `numpy>=2.0.0`, and `python-multipart>=0.0.12`.
2. `.venv/bin/pip install -r backend/requirements.txt`: Successfully installed `pandas-2.3.3`, `numpy-2.5.4`, and `python-multipart-0.0.32`.
3. Created `docs/DATASET_CARD.md`: Documented dataset provenance (ULB ML Group), license (ODbL v1.0), 284,807 transactions, 31 columns, 492 fraud cases (0.172% fraud rate), PCA anonymization rationale, and strict usage boundaries.
4. Created `docs/DATA_DICTIONARY.md`: Documented full field-level data dictionary for `Time`, `V1`–`V28`, `Amount`, and `Class`, including data types, value domains, and missingness expectations.
5. Implemented `ml/data/schema.py`: Central schema definition with required features, target column, expected dtypes, and maximum upload size limits (50MB).
6. Implemented `ml/data/synthetic.py`: Synthetic generator and created 5 fixtures in `data/sample/`:
   - `synthetic_creditcard_sample.csv` (150 rows, 9 fraud cases, statistical shifts on key components)
   - `corrupted_missing_columns.csv` (missing V3–V28)
   - `corrupted_types.csv` (corrupted strings in numeric columns)
   - `corrupted_negative_amount.csv` (negative transaction amounts)
   - `empty.csv` (0-byte file)
7. Implemented `ml/data/validator.py`: Comprehensive validation layer checking empty inputs, schema completeness, numeric dtypes, positive amounts, valid class labels (0 or 1), and duplicate row detection.
8. Implemented `ml/data/loader.py`: Repeatable data loading module with format checking and schema verification.
9. Implemented `ml/data/profiler.py`: Statistical profiling module extracting dimensions, missingness, duplicate count, time span, amount statistics, class distribution, and markdown report generation.
10. Implemented `backend/app/api/v1/upload.py`: Safe CSV upload endpoint (`POST /api/v1/data/upload`) enforcing file extensions, size limits, schema validation, and structured error responses.
11. Implemented test suites:
    - `tests/ml/test_data_loader.py`: Tested valid dataset loading, nonexistent files, unsupported formats, and row limits.
    - `tests/ml/test_data_validator.py`: Tested valid frames, empty frames, missing columns, corrupted types, negative amounts, training mode target requirements, duplicate warnings, and byte limits.
    - `tests/ml/test_profiler.py`: Tested metric extraction and markdown report generation.
    - `tests/backend/test_upload_api.py`: Tested valid upload, unsupported media types (415), empty CSV (422), missing columns (422), and corrupted types (422).
12. Executed `.venv/bin/pytest -v` and `.venv/bin/ruff check backend ml tests`:
    - *Result*: 30 passed in 0.33s; all Ruff checks passed.
13. Live smoke tests:
    - Verified `POST /api/v1/data/upload` with `synthetic_creditcard_sample.csv` -> HTTP 200 with 150 rows verified.
    - Verified `POST /api/v1/data/upload` with `corrupted_missing_columns.csv` -> HTTP 422 with structured rejection details.

#### Files Created / Modified:
- `backend/requirements.txt`
- `docs/DATASET_CARD.md`
- `docs/DATA_DICTIONARY.md`
- `ml/__init__.py`
- `ml/data/__init__.py`
- `ml/data/schema.py`
- `ml/data/synthetic.py`
- `ml/data/validator.py`
- `ml/data/loader.py`
- `ml/data/profiler.py`
- `data/sample/synthetic_creditcard_sample.csv`
- `data/sample/corrupted_missing_columns.csv`
- `data/sample/corrupted_types.csv`
- `data/sample/corrupted_negative_amount.csv`
- `data/sample/empty.csv`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `backend/app/api/v1/upload.py`
- `backend/app/main.py`
- `tests/ml/__init__.py`
- `tests/ml/test_data_loader.py`
- `tests/ml/test_data_validator.py`
- `tests/ml/test_profiler.py`
- `tests/backend/test_upload_api.py`
- `docs/project-memory/02_EXECUTION_LOG.md`
- `docs/project-memory/03_LESSONS_LEARNED.md`
- `docs/PHASE_STATUS.md`

#### Deviations from Plan:
- None.

---

### Phase 3: Exploratory Data Analysis and Preprocessing
- **Date**: 2026-11-02
- **Status**: Completed (PASS)
- **Goal**: Understand the data, calculate class distributions and correlations, audit for target leakage, build leakage-safe reproducible train/validation/test splits (stratified and time-based), and create a persistent preprocessing pipeline that fits transformations strictly on training data.

#### Actions & Commands Executed:
1. Updated `backend/requirements.txt` with `scikit-learn>=1.5.0` and `joblib>=1.4.0`.
2. `.venv/bin/pip install -r backend/requirements.txt`: Successfully installed `scikit-learn-1.9.1`, `scipy-1.18.1`, `joblib-1.6.0`, `narwhals-2.28.0`, `threadpoolctl-3.7.0`, and `cloudpickle-3.1.2`.
3. Created `ml/eda/analysis.py` and `ml/eda/__init__.py`:
   - `analyze_class_imbalance`: Evaluated class counts, fraud percentage (6.0% on sample, 0.172% benchmark), and documented operational constraints against naive accuracy metrics.
   - `analyze_feature_correlations`: Ranked top positive and negative correlators with target `Class`.
   - `check_target_leakage`: Audited features for suspicious near-perfect correlations ($|r| \ge 0.98$).
   - `analyze_time_patterns`: Analyzed temporal coverage and hourly fraud distribution.
   - `save_eda_report`: Generated initial `docs/EDA_REPORT.md`.
4. Created `ml/preprocessing/splitter.py`:
   - `split_dataset`: Implemented reproducible stratified and time-based splitters.
   - Enforced deduplication prior to split to prevent identical transactions across partitions.
   - Enforced strict target separation (`Class` strictly excluded from feature matrix `X`).
   - Verified zero index overlap between splits.
5. Created `ml/preprocessing/pipeline.py`:
   - Implemented `AuditLensPreprocessor` adhering to scikit-learn transformer protocol.
   - Fits `SimpleImputer(strategy="median")` and `RobustScaler` on `Amount` strictly on training data.
   - Encodes `Time` into cyclic sinusoidal and cosinusoidal signals (`time_sin`, `time_cos`) to capture diurnal fraud cycles.
   - Rejects inputs containing target column `Class` to prevent target leakage.
   - Implemented artifact persistence (`save` and `load`) using `joblib` and companion JSON metadata.
6. Implemented Phase 3 unit tests:
   - `tests/ml/test_eda.py`: Tested class imbalance, correlation ranking, leakage detection on synthetic leak feature, and markdown generation.
   - `tests/ml/test_splitter.py`: Tested stratified split ratios, 3-way validation split, time-based chronological ordering, missing target rejection, and deduplication.
   - `tests/ml/test_preprocessing.py`: Tested fit/transform behavior, target column rejection, missing value imputation on unseen inference records, robust amount scaling, and exact bit-for-bit reload reproducibility.
7. Executed `.venv/bin/pytest -v` and `.venv/bin/ruff check backend ml tests`:
   - *Result*: 46 tests passed in 6.28s; all Ruff checks passed.
8. Executed frontend test and build verification:
   - *Result*: 2 frontend tests passed; TypeScript check passed; production bundle built cleanly in 238ms.

#### Files Created / Modified:
- `backend/requirements.txt`
- `docs/EDA_REPORT.md`
- `ml/eda/__init__.py`
- `ml/eda/analysis.py`
- `ml/preprocessing/__init__.py`
- `ml/preprocessing/splitter.py`
- `ml/preprocessing/pipeline.py`
- `tests/ml/test_eda.py`
- `tests/ml/test_splitter.py`
- `tests/ml/test_preprocessing.py`
- `docs/project-memory/02_EXECUTION_LOG.md`
- `docs/project-memory/03_LESSONS_LEARNED.md`
- `docs/PHASE_STATUS.md`

#### Deviations from Plan:
- None.

---

### Phase 4: Baseline Machine-Learning Models
- **Date**: 2026-11-02
- **Status**: Completed (PASS)
- **Goal**: Implement unsupervised anomaly detection and supervised fraud classification baselines, keep anomaly scores and fraud probabilities strictly distinct, build reusable training/inference modules, store hyperparameters and metadata, and persist model artifacts.

#### Actions & Commands Executed:
1. Created `models/` directory and `.gitkeep` (with `models/*` ignored in `.gitignore`).
2. Implemented `ml/models/base.py`:
   - `PredictionRecord`: Data contract ensuring `anomaly_score` and `fraud_probability` remain decoupled.
   - `ModelMetadata`: Provenance tracking schema recording algorithm, hyperparameters, feature list, and training metrics.
   - `BaseFraudModel`: Abstract model interface.
3. Implemented `ml/models/anomaly.py`:
   - `IsolationForestAnomalyDetector`: Wraps `IsolationForest` (`n_estimators=100`, `contamination=0.05`, `random_state=42`).
   - Normalizes anomaly score to continuous $[0.0, 1.0]$ range where higher values indicate more isolated transactions.
   - Version identifier: `iso_forest_v0.1.0`.
   - Persists model artifact (`isolation_forest.joblib`) and metadata (`isolation_forest.json`).
4. Implemented `ml/models/classifier.py`:
   - `LogisticRegressionFraudClassifier`: Regularized linear classifier with `class_weight='balanced'`, `C=1.0`, `solver='lbfgs'`, `random_state=42`.
   - Predicts calibrated continuous fraud probability $P(\text{Class}=1 \mid X) \in [0.0, 1.0]$.
   - Version identifier: `logreg_v0.1.0`.
   - Persists model artifact (`logistic_regression.joblib`) and metadata (`logistic_regression.json`).
5. Implemented `ml/models/trainer.py`:
   - `train_baseline_models`: Automated training pipeline executing data loading, stratified splitting, preprocessing fit, anomaly training, supervised classifier training, and baseline evaluation on held-out test data.
   - Saves consolidated evaluation report to `models/baseline_evaluation.json`.
6. Implemented `ml/models/inference.py`:
   - `AuditLensInferenceEngine`: Production inference service supporting single-transaction and batch scoring.
   - Validates feature alignment, executes preprocessing, generates dual scores, and outputs typed `PredictionRecord` instances.
7. Created documentation:
   - `docs/MODEL_CARD.md`: Documented architecture, version identifiers, intended decision-support use, feature schema, test evaluation metrics, and operational limitations.
8. Implemented Phase 4 unit tests:
   - `tests/ml/test_anomaly_model.py`: Tested fitting, score bounds $[0.0, 1.0]$, dimensions, and exact reload reproducibility.
   - `tests/ml/test_classifier_model.py`: Tested probability calibration $[0.0, 1.0]$, threshold classification, single-class error rejection, and reload reproducibility.
   - `tests/ml/test_inference_engine.py`: Tested directory artifact loading, dual score decoupling in prediction records, single-transaction scoring, and missing feature rejection.
   - `tests/ml/test_trainer.py`: Tested end-to-end training execution, artifact creation, and evaluation report generation.
9. Executed `.venv/bin/pytest -v` and `.venv/bin/ruff check backend ml tests`:
   - *Result*: 58 tests passed in 1.74s; all Ruff checks passed.
10. Executed frontend verification:
    - *Result*: 2 frontend tests passed; lint/type check clean; bundle built in 234ms.

#### Files Created / Modified:
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

#### Deviations from Plan:
- None.
