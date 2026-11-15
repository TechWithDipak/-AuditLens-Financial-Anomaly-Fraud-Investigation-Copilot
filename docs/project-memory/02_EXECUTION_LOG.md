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
