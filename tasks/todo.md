# Tasks: AuditLens Implementation

- [x] Task 1: Data Ingestion, Verification, and `docs/DATA.md`
  - Description: Implement `src/config.py` and `src/data.py` to ingest Sparkov dataset with 80/20 chronological train/validation split and holdout test. Calculate real row counts, date ranges, and fraud rates, and record in `docs/DATA.md`.
  - Verification: `python -m src.data` verified: Train=1,037,340, Val=259,335, Test=555,719.

- [x] Task 2: Causal Feature Engineering and Leakage Test
  - Description: Implement `src/features.py` computing causal features (`amt`, `log_amt`, `hour`, `day_of_week`, `is_night`, category one-hot, `distance_km`, `age`, `gender`, `city_pop`, `secs_since_last_txn`, `amt_zscore_vs_customer_history`, `txn_count_last_24h`). Exclude identifier columns (`cc_num`, `trans_num`, names, street, row index). Fit scalers on train only. Implement `tests/test_leakage.py`.
  - Verification: `pytest tests/test_leakage.py` passed (5/5 tests).

- [x] Task 3: EM-GMM and HMM Hand-written Algorithms
  - Description: Write EM-GMM by hand in `src/em_gmm.py` with E-step responsibilities, M-step updates, log-likelihood, and test against `sklearn.mixture.GaussianMixture` in `tests/test_em_gmm.py`. Implement HMM discrete symbol encoder, hand-written Forward (scaled and log-space) and Viterbi (log-space) in `src/hmm.py`, and test against `hmmlearn` in `tests/test_hmm.py`. Implement PCA and customer-level K-Means in `src/models.py`.
  - Verification: `pytest tests/test_em_gmm.py tests/test_hmm.py` passed (7/7 tests).

- [x] Task 4: Supervised Models, Training Pipeline, and Metrics
  - Description: Implement CART, XGBoost (`scale_pos_weight`), and soft-voting Ensemble in `src/models.py`. In `train.py`, fit models on train, tune decision threshold on validation PR curve, evaluate on full test set, run ablation (with and without unsupervised columns), save `artifacts/metrics.json` and `artifacts/test_scored.parquet`.
  - Verification: `python train.py` ran in 18.8s; Test PR-AUC = 0.9176, ROC-AUC = 0.9981, F1 = 0.8585.

- [x] Task 5: Explainability, Reporting, and UI Implementation
  - Description: Implement SHAP tree explanations with natural language translation dictionary in `src/explain.py`. Implement monochrome audit report generator in `src/report.py`. Implement `.streamlit/config.toml` and `app.py` with 4 tabs (`Overview`, `Transactions`, `Algorithms`, `How it works`), rupee formatting, grayscale Plotly styling, and interactive run buttons.
  - Verification: Streamlit UI runs cleanly and renders all 4 tabs and widgets.

- [x] Task 6: Full Verification, AppTest, Compliance Scans, and Docs
  - Description: Write headless AppTest in `tests/test_app.py` exercising all tabs, row selection, run buttons, and report download. Run full pytest suite. Execute static compliance scan for emojis, banned words, and non-grayscale colors. Write `README.md` and `PROJECT.md`.
  - Verification: All 16 pytest tests pass, compliance scan is 100% clean, documentation complete.
