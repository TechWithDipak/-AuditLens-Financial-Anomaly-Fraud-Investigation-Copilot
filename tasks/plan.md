# Implementation Plan: AuditLens

## Overview
AuditLens is a lightweight, high-performance credit card fraud review and investigation web application. It flags transactions into Normal, Suspicious, or Fraud, explains why using plain-English behavioral justifications, and generates formal monochrome audit reports. The project uses the Sparkov synthetic fraud dataset, strictly enforces chronological splitting and causal feature engineering, and features transparent implementations of K-Means, EM-GMM, HMM (Forward and Viterbi), PCA, CART, and XGBoost.

## Architecture Decisions
- **Refined Color Palette**: Minimal academic aesthetic using background (#FFFFFF), panels (#F5F7FA), borders (#D9DEE5), text (#1A1D21), secondary text (#5C6670), accent blue (#1F4E8C), and high-contrast risk colors (Normal #2E6B3F on #EAF3EC, Suspicious #8A5A00 on #FBF1DC, Fraud #9B1C1C on #FBE9E9). No emojis, no decorative icons.
- **Causal Feature Engineering**: Per-customer chronological accumulators compute transaction frequency, time since last transaction, and amount z-scores using historical transactions only. Scalers and categorical encodings are fitted strictly on the train split.
- **Hand-written Algorithms with Validation**: Hand-written Expectation-Maximization for GMM and Forward/Viterbi for HMM in NumPy, mathematically verified against scikit-learn and hmmlearn in automated unit tests.
- **Precomputed Test Artifacts**: All test-set transactions are scored and saved to Parquet during training to ensure instant UI responsiveness without raw CSV parsing.

## Task List

### Phase 1: Data Ingestion and Verification
- [ ] Task 1.1: Implement `src/config.py` and `src/data.py` with chronological splitting.
- [ ] Task 1.2: Compute exact dataset statistics and generate `docs/DATA.md`.
- [ ] Checkpoint 1: Verify data ingestion and split integrity.

### Phase 2: Feature Engineering and Leakage Prevention
- [ ] Task 2.1: Implement `src/features.py` with causal rolling customer statistics and haversine distance.
- [ ] Task 2.2: Implement `tests/test_leakage.py` verifying no future-to-past information leakage.
- [ ] Checkpoint 2: All leakage tests pass.

### Phase 3: Unsupervised & Sequential Algorithms
- [ ] Task 3.1: Implement hand-written EM-GMM in `src/em_gmm.py` with unit tests against sklearn in `tests/test_em_gmm.py`.
- [ ] Task 3.2: Implement HMM symbol encoder, Forward, and Viterbi in `src/hmm.py` with unit tests against hmmlearn in `tests/test_hmm.py`.
- [ ] Task 3.3: Implement PCA and customer-level K-Means in `src/models.py`.
- [ ] Checkpoint 3: EM-GMM and HMM math tests pass.

### Phase 4: Supervised Models, Ensemble, and Training Pipeline
- [ ] Task 4.1: Implement CART, XGBoost, and soft-voting Ensemble in `src/models.py`.
- [ ] Task 4.2: Implement `train.py` with validation threshold tuning, ablation study, and metric saving.
- [ ] Task 4.3: Execute `train.py`, generate `artifacts/metrics.json` and `artifacts/test_scored.parquet`.
- [ ] Checkpoint 4: Training pipeline succeeds and full test set metrics are recorded.

### Phase 5: Explainability, Reporting, and UI Development
- [ ] Task 5.1: Implement SHAP feature translation and plain-English reasons in `src/explain.py`.
- [ ] Task 5.2: Implement printable monochrome audit report generator in `src/report.py`.
- [ ] Task 5.3: Implement Streamlit configuration `.streamlit/config.toml` and main interface `app.py`.
- [ ] Checkpoint 5: App UI loads and renders all 4 tabs cleanly.

### Phase 6: Verification, Static Checks, and Documentation
- [ ] Task 6.1: Implement end-to-end headless UI testing via `streamlit.testing.v1.AppTest` in `tests/test_app.py`.
- [ ] Task 6.2: Run full pytest suite.
- [ ] Task 6.3: Static audit for emojis, banned hype words, and non-grayscale colors.
- [ ] Task 6.4: Write comprehensive `README.md` and `PROJECT.md`.
- [ ] Checkpoint 6: Ready for final review.
