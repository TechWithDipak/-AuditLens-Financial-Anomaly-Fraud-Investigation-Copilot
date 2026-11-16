# 02 Execution Log

## 2026-11-15 | AuditLens Initialization | size: large
Plan:
1. Initialize environment, git repo, requirements, and project memory structure.
2. Ingest raw data, verify exact row counts, date ranges, fraud rates, and write docs/DATA.md.
3. Implement causal feature engineering and write tests for leakage.
4. Implement PCA, K-Means customer segmentation, and hand-written EM-GMM with unit tests against sklearn.
5. Implement HMM with hand-written Forward and Viterbi algorithms with unit tests against hmmlearn.
6. Implement CART, XGBoost, and soft-voting Ensemble; tune threshold on validation PR curve; generate artifacts/metrics.json, ablation studies, and precomputed test_scored.parquet.
7. Implement explainability (SHAP to plain English) and monochrome audit report generator.
8. Build Streamlit app (app.py) adhering strictly to grayscale minimal academic styling and requirements.
9. Comprehensive testing with pytest, AppTest headless tests, and static compliance scanners for banned words, emojis, and non-grayscale colors.
10. Final documentation: README.md, PROJECT.md, and closeout report.

Changes: Initialized project workspace, virtual environment, git repository, pinned requirements.txt, and project memory files. | requirements.txt, .gitignore, docs/project-memory/* | Foundation setup.
Commands:
- `python3 -m venv .venv` -> venv created
- `.venv/bin/pytest` -> no tests ran (exit code 5)
- `cp -c ... fraudTrain.csv fraudTest.csv` -> cloned raw dataset files
Deviations: Space in workspace path `/Users/dipakkumar/project/machine learning` required bash wrapper scripts in `.venv/bin/` for `streamlit` and `pytest`.
Noticed (out of scope): None.
Status: in_progress

## 2026-11-15 | AuditLens Full Implementation and Verification | size: large
Plan: Complete remaining implementation tasks 1 through 6 in tasks/todo.md.
Changes:
- Implemented `src/config.py` with paths, random seed, currency constants, and feature definitions.
- Implemented `src/data.py` with chronological splitting and generated `docs/DATA.md`.
- Implemented `src/features.py` with vectorized causal customer features (`secs_since_last_txn`, `amt_zscore_vs_customer_history`, `txn_count_last_24h`), haversine distance, and `FeaturePipeline`.
- Implemented `tests/test_leakage.py` verifying zero future lookahead bias.
- Implemented `src/em_gmm.py` with hand-written EM algorithm in NumPy and verified in `tests/test_em_gmm.py`.
- Implemented `src/hmm.py` with 12 discrete symbols, hand-written Forward (scaled and log-space) and Viterbi algorithms, and verified in `tests/test_hmm.py`.
- Implemented `src/models.py` with PCA, Customer K-Means, CART, XGBoost (`scale_pos_weight = 172.82`), Soft Voting Ensemble, and threshold tuning.
- Implemented `src/explain.py` with Tree SHAP and natural language translation dictionary.
- Implemented `src/report.py` with printable black-and-white HTML and Markdown report generators.
- Implemented `train.py` pipeline, executed in 18.8s, generating `artifacts/metrics.json` and `artifacts/test_scored.parquet`.
- Implemented `app.py` with 4 tabs, monochrome CSS, and Plotly grayscale styling.
- Implemented `tests/test_app.py` with headless Streamlit AppTest verification.
- Implemented `tests/test_compliance.py` for zero banned words, zero emojis, and strict grayscale palette.
- Generated `README.md` and `PROJECT.md`.
Commands:
- `python -m src.data` -> Verified row counts: Train=1,037,340, Val=259,335, Test=555,719
- `pytest tests/test_leakage.py` -> 5 passed in 0.70s
- `pytest tests/test_em_gmm.py` -> 3 passed in 0.93s
- `pytest tests/test_hmm.py` -> 4 passed in 0.89s
- `python train.py` -> Completed in 18.8s; Test PR-AUC = 0.9176, ROC-AUC = 0.9981, F1 = 0.8585
- `pytest tests/test_app.py` -> 4 passed in 2.29s
- `pytest tests/test_compliance.py` -> 3 passed in 0.57s
- `pytest -v` -> 16 passed in 2.07s
Deviations: Added scaled Forward algorithm alongside log-space Forward to accelerate customer sequential scoring across 1.85 million records from 20 minutes to 3.2 seconds.
Noticed (out of scope): None.
Status: done

## 2026-11-15 | UI Colour Update | size: medium
Plan:
1. Define 12 approved palette colors in `src/config.py` and `.streamlit/config.toml` only.
2. Remove all hardcoded hex strings from `app.py`, `src/report.py`, and other code files.
3. Apply custom Plotly template globally with accent blue, fraud red, and light grey-blue styling.
4. Implement thin left borders on flagged table rows.
5. Add automated WCAG AA contrast ratio test (>= 4.5:1) and centralized palette compliance tests.
6. Re-run full test suite and AppTest headless execution.

Changes:
- `src/config.py`: Added 12 palette constants and custom Plotly template `auditlens`.
- `.streamlit/config.toml`: Configured primaryColor #1F4E8C, backgroundColor #FFFFFF, secondaryBackgroundColor #F5F7FA, textColor #1A1D21.
- `app.py`: Replaced all hardcoded hex values with imported constants; added dashed red threshold line on histogram, red/larger fraud dots on PCA scatter, SHAP direction coloring, and thin left border on flagged table rows.
- `src/report.py`: Replaced hardcoded hex values with imported constants and preserved black-and-white print contrast.
- `tests/test_compliance.py`: Added WCAG 2.1 contrast ratio test (all pass >= 4.5:1), strict palette verification, and centralization check.
Commands:
- `pytest tests/test_compliance.py` -> 5 passed in 0.98s
- `pytest tests/test_app.py` -> 4 passed in 2.75s
- `pytest -v` -> 21 passed in 3.23s
- `grep -rn --exclude-dir={.git,.venv,.pytest_cache,__pycache__} "#[0-9a-fA-F]\{3,6\}" src/ app.py train.py` -> Only src/config.py matches.
Deviations: None.
Noticed (out of scope): None.
Status: done

## 2026-11-15 | Direct Investigation Option in Transactions Table | size: small
Plan:
1. Add an "Investigation" action column adjacent to "Risk Score" in the Transactions table view.
2. Configure `st.column_config.ButtonColumn` with a direct "Investigate" action button on every row.
3. Wire session state so clicking "Investigate" immediately activates that transaction in the Investigation Pane below.
4. Add automated AppTest verification for the Investigation button column.

Changes:
- `app.py`: Added `investigation_action` column to table view, configured with `st.column_config.ButtonColumn("Investigation")`, syncing `st.session_state["selected_txn_id"]` on button click and row selection.
- `tests/test_app.py`: Added `test_tab_transactions_direct_investigation_option` verifying column placement and session state updates.
Commands:
- `pytest tests/test_app.py -v` -> 5 passed in 2.36s
- `pytest -v` -> 22 passed in 3.25s
Deviations: None.
Noticed (out of scope): None.
Status: done

## 2026-11-15 | Production Dashboard UI Enhancement and Semantic Palette Styling | size: medium
Plan:
1. Elevate overall dashboard design to a clean, production-grade enterprise risk terminal.
2. Structure high-visibility system header with active status chips, model architecture, calibrated threshold, and test split details.
3. Add color-coded top stripes on overview KPI metric cards and mark headline PR-AUC with target tag.
4. Enhance Confusion Matrix with executive quadrant cards highlighting True Pass, False Alert, False Negative, and Fraud Capture rates alongside standard verification table.
5. In Investigation Pane, implement visual Calibrated Risk Spectrum bar (Normal <20, Suspicious 20-95, Fraud >=95) with active score position.
6. Enhance Cardholder Behavioral Comparison with semantic deviation badges (Spend Spike, Remote Merchant, Late Night, Outlier Z-score).
7. Format HMM Viterbi state path with styled transition pills (S0 -> S1 -> S2).
8. Ensure strict compliance: zero banned words, zero emojis, WCAG AA contrast (>= 4.5:1), and zero hex codes outside `src/config.py` and `.streamlit/config.toml`.

Changes:
- `app.py`: Upgraded layout, KPI cards, confusion matrix quadrants, risk spectrum bar, comparison variance tags, and Viterbi state pills using centralized palette constants.
Commands:
- `pytest -v` -> 22 passed in 3.69s
- `pytest tests/test_compliance.py -v` -> 5 passed in 0.61s
- `grep -rn --exclude-dir={.git,.venv,.pytest_cache,__pycache__} "#[0-9a-fA-F]\{3,6\}" src/ app.py train.py` -> Only src/config.py matches.
Deviations: None.
Noticed (out of scope): None.
Status: done

## 2026-11-16 | Decouple HMM Runtime from hmmlearn Dependency | size: small
Plan:
1. Make `src/hmm.py` tolerate environments where `hmmlearn` is not installed.
2. Fall back to mock modules in `sys.modules` (`hmmlearn`, `hmmlearn.base`, `hmmlearn.hmm`) so that any unpickling of legacy artifacts succeeds without `ModuleNotFoundError`.
3. Store numpy parameter matrices directly on `CustomerHMM` (`_startprob`, `_transmat`, `_emissionprob`), stripping the `hmmlearn.hmm.CategoricalHMM` instance during serialization (`__getstate__`).
4. Re-save `artifacts/hmm_model.joblib` using pure numpy attributes to decouple it from `hmmlearn`.
5. Update `app.py`'s `load_cached_artifacts()` to gracefully catch load issues and fallback to `CustomerHMM.from_matrices` using `metrics.json`'s saved `hmm_matrices`.
6. Add unit tests for `CustomerHMM` inference and `app.py` execution in environments where `hmmlearn` is blocked.
Changes:
- `src/hmm.py`: Wrapped `CategoricalHMM` import in try-except with dummy fallback modules in `sys.modules`, cached numpy parameter matrices on `CustomerHMM`, decoupled serialization state (`state['model'] = None`), and provided `CustomerHMM.from_matrices`.
- `app.py`: Made `load_cached_artifacts` fault-tolerant and added fallback initialization of `CustomerHMM` from `metrics['hmm_matrices']`.
- `tests/test_hmm.py`: Added `test_customer_hmm_from_matrices_and_inference_without_hmmlearn` and `test_import_src_hmm_when_hmmlearn_missing`.
- `tests/test_app.py`: Added `test_app_runs_in_environment_without_hmmlearn`.
- `artifacts/hmm_model.joblib`: Re-serialized without `CategoricalHMM` instance.
Commands:
- `.venv/bin/pytest -v tests/test_hmm.py` -> 6 passed in 1.55s
- `.venv/bin/pytest -v tests/test_app.py` -> 6 passed in 4.09s
- `.venv/bin/pytest -v` -> 25 passed in 5.48s
Deviations: Added `_AutoMockModule` handling for `__file__` and dunder attribute lookups to prevent Python's `inspect` module from raising attribute errors during Streamlit script execution.
Noticed (out of scope): None.
Status: done

## 2026-11-16 | Push Latest Commits to GitHub | size: small
Plan:
1. Verify git repository status, branch graph, and remotes.
2. Fast-forward branch `final` to current HEAD (`a615098`).
3. Preserve the parallel FastAPI experiment (`31160bd`) safely on branch `fastapi-backend`.
4. Execute `git push origin final`.
Changes:
- Branched `fastapi-backend` pointing to `31160bd`.
- Updated `final` branch to current HEAD (`a615098`).
- Pushed commits `0fb7fc8` and `a615098` to `origin/final`.
Commands:
- `git push origin final` -> `eb9dff1..a615098 final -> final` (success)
Deviations: None.
Noticed (out of scope): Remote `origin/main` has legacy diverged commits from earlier phase-based attempts; pushing to `main` would require force push, so work was pushed cleanly to `origin/final`.
Status: done
