# 01 Problem Statement: AuditLens

## 1. Problem
Credit card fraud detection systems often operate as black boxes, providing probability scores without actionable, human-interpretable justifications. Fraud investigators and risk analysts require clear, plain-language explanations of why a specific transaction is flagged, grounded in customer behavioral baselines, temporal patterns, and geographic anomalies, without overwhelming visual noise or opaque metrics.

## 2. Context
The system processes credit card transaction records from the Sparkov synthetic dataset. It serves analysts who need to review high-volume transactional flows, distinguish between normal activity, suspicious variations, and definite fraud, and generate exportable audit reports. The solution runs locally as a lightweight Python application using Streamlit for analysis and visualization.

## 3. Goal
Deliver AuditLens: a small, clean, high-performance web application that flags transactions into Normal, Suspicious, or Fraud, explains why in plain English using SHAP values and customer behavioral baselines, and exports formal audit reports, with transparent demonstration of all underlying machine learning and statistical algorithms.

## 4. Constraints
- **Dataset**: Strictly Kaggle `kartik2112/fraud-detection` (`data/raw/fraudTrain.csv` and `fraudTest.csv`). The dataset is synthetic.
- **Split Discipline**: Temporal chronological split (first 80% train, last 20% validation of fraudTrain; fraudTest as holdout test). Zero future-to-past leakage.
- **Technology Stack**: Python 3.13, pandas, numpy, scikit-learn, xgboost, shap, hmmlearn, Streamlit, Plotly. No database, no React, no external API server.
- **Design System**: Refined palette defined once in `src/config.py` and `.streamlit/config.toml` (Background #FFFFFF, panels #F5F7FA, borders #D9DEE5, text #1A1D21, secondary text #5C6670, accent blue #1F4E8C, and risk colors Normal #2E6B3F on #EAF3EC, Suspicious #8A5A00 on #FBF1DC, Fraud #9B1C1C on #FBE9E9). No emojis, no gradients, no decorative icons, no hype language.
- **Currency**: Display in Indian Rupees (₹) with Indian numbering format (1 USD = 83 INR configurable). Underlying model computes on raw amounts.
- **Performance**: Precomputed test evaluation scores saved in Parquet; no raw CSV parsing on UI interactions.

## 5. Assumptions
- **Verified**: Raw dataset `fraudTrain.csv` (1,296,675 rows) and `fraudTest.csv` (555,719 rows) are present in `data/raw/`.
- **Verified**: Python 3.13 virtual environment possesses all pinned requirements (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `hmmlearn`, `streamlit`, `plotly`, `pyarrow`, `joblib`, `pytest`).
- **Verified**: Haversine distance correctly captures merchant-to-cardholder geographic displacement using latitude and longitude coordinates.
- **Verified**: Customer behavioral baselines can be computed strictly chronologically using past transactions per cardholder.
- **Assumed**: Memory footprint on 1.29M rows can be handled efficiently with typed numeric columns and vectorized operations.

## 6. Out of Scope
- Real-time streaming ingestion (Kafka/Flink).
- External databases (PostgreSQL, MongoDB).
- User authentication and multi-tenant access control.
- Deep learning or neural network architectures (LSTM, Transformers).
- SMOTE or synthetic minority oversampling (explicitly forbidden; use `scale_pos_weight`).

## 7. Success Criteria
1. **Data Verification**: `docs/DATA.md` documents verified row counts, date ranges, and class balance for train, validation, and test splits. Check: `python -m src.data` / verification script.
2. **Feature Engineering & Leakage Protection**: Causal rolling customer aggregates with zero future leakage. Check: `pytest tests/test_leakage.py`.
3. **EM-GMM Implementation**: Hand-written Gaussian Mixture Model Expectation-Maximization in NumPy verified against `sklearn.mixture.GaussianMixture`. Check: `pytest tests/test_em_gmm.py`.
4. **HMM Implementation**: Hand-written Forward algorithm (scaled log-space) and Viterbi algorithm (log-space) verified against `hmmlearn.hmm.CategoricalHMM`. Check: `pytest tests/test_hmm.py`.
5. **Model Evaluation & Ablation**: Evaluation on the full test set reporting PR-AUC as headline metric, ROC-AUC, Precision, Recall, F1, confusion matrix, and feature ablation. Check: `artifacts/metrics.json` inspection.
6. **Interpretability & Reporting**: Plain-English rule explanations with feature-to-sentence mappings, customer historical comparisons, and clean black-and-white downloadable audit report. Check: Unit test and UI verification.
7. **Streamlit Application Verification**: 4 tabs (`Overview`, `Transactions`, `Algorithms`, `How it works`), grayscale styling, interactive transaction inspection, interactive algorithm execution, full headless execution using `AppTest`. Check: `pytest tests/test_app.py`.
8. **Code Cleanliness & Style Compliance**: Zero emojis, zero banned words, no non-grayscale color codes. Check: Static scan script.

## 8. How the Problem is Solved
- **Chosen Approach**:
  - Full chronological sequence computation per cardholder to calculate cumulative historical statistics (`secs_since_last_txn`, `amt_zscore_vs_customer_history`, `txn_count_last_24h`).
  - Feature dimensionality reduction via PCA (5 components).
  - Customer profile segmentation via K-Means on aggregated behavioral vectors.
  - Hand-crafted EM Gaussian Mixture Model computing transactional density anomaly scores.
  - Categorical Hidden Markov Model trained on 12-symbol discrete behavioral alphabet, using custom Forward (anomaly score) and Viterbi (state sequence).
  - Supervised learning with CART Decision Tree and XGBoost (with `scale_pos_weight`), augmented with unsupervised scores. Soft voting ensemble.
  - Threshold tuning on validation PR curve; holdout test evaluation.
  - SHAP TreeExplainer for local feature attributions translated into concise natural language sentences.
- **Alternatives Considered**:
  - *SMOTE / Oversampling*: Rejected due to artificial pattern distortion and explicit prompt constraint.
  - *Deep learning sequential models*: Rejected due to unnecessary complexity and slower training without interpretability benefits.
- **Rollback Plan**:
  - All artifacts versioned under `artifacts/`. If a training run degrades PR-AUC or fails validation checks, previous serialized artifacts remain untouched until new validation passes.

## 9. How the Project Works
- **Overview**: AuditLens is a card fraud detection and investigation system that couples unsupervised anomaly discovery (GMM, HMM, K-Means) with supervised decision trees and gradient boosting, surfacing plain-English reasons for flagged transactions.
- **Tech Stack**:
  - Python 3.13.7
  - pandas 3.1.1, numpy 2.5.4
  - scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, hmmlearn 0.3.3
  - streamlit 1.67.0, plotly 7.1.0, pyarrow 26.0.1, joblib 1.6.0, pytest 9.2.0
- **Folder Structure**:
  - `app.py`: Single Streamlit UI entry point with 4 tabs.
  - `train.py`: Full model training and evaluation script.
  - `src/config.py`: Central configurations, paths, parameters, currency constants.
  - `src/data.py`: Raw data ingestion and chronological splitting.
  - `src/features.py`: Causal feature engineering and transformation pipeline.
  - `src/em_gmm.py`: Hand-written NumPy EM algorithm for GMM anomaly scoring.
  - `src/hmm.py`: Discrete symbol encoder, hand-written Forward and Viterbi algorithms.
  - `src/models.py`: K-Means, PCA, CART, XGBoost, and soft voting ensemble.
  - `src/explain.py`: SHAP tree explainer and natural language sentence generator.
  - `src/report.py`: Printable monochrome HTML/Markdown audit report generator.
  - `tests/`: Automated test suite for leakage, mathematical equivalences, and UI.
  - `artifacts/`: Serialized models, scalers, precomputed test parquet, metrics.json.
  - `docs/`: Dataset documentation, architecture, and memory logs.
- **Entry Points**:
  - Pipeline training: `python train.py`
  - Web UI: `streamlit run app.py`
- **Architecture and Data Flow**:
  1. `data/raw/` CSVs -> `src/data.py` (chronological 80/20 train/val split, holdout test).
  2. `src/features.py` -> Causal rolling statistics per customer -> Scalers/Encoders fitted on train only.
  3. `src/models.py`, `src/em_gmm.py`, `src/hmm.py` -> Unsupervised feature scoring (PCA, K-Means, GMM, HMM).
  4. Supervised models (CART, XGBoost, Ensemble) trained on augmented features -> Validation PR-AUC threshold tuning.
  5. Test set evaluated on headline PR-AUC and secondary metrics -> `artifacts/metrics.json`.
  6. Precomputed test scored records -> `artifacts/test_scored.parquet`.
  7. `app.py` loads artifacts into interactive UI tabs.
- **Configuration**:
  - `USD_TO_INR`: Default 83.0 (configurable via UI sidebar).
  - Random seed: 42.

## 10. User Interface
- **UI Stack**: Streamlit 1.67.0 with custom monochrome CSS, Plotly (`plotly_white` template with grayscale styling).
- **Navigation Map**: 4 tabs on a single page:
  - Tab 1: `Overview`
  - Tab 2: `Transactions`
  - Tab 3: `Algorithms`
  - Tab 4: `How it works`
- **Tab Details**:
  - `Overview`: Key metrics (Total Evaluated, Flagged Transactions, Detected Fraud Rate, Headline Test PR-AUC, ROC-AUC, Precision, Recall, F1), Risk Score Distribution histogram, Confusion Matrix table.
  - `Transactions`: Filterable table by Risk Level (All, Fraud, Suspicious, Normal), Category, Date Range. Direct "Investigation" action button column adjacent to Risk Score in the table triggers instant loading of the transaction investigation pane upon click (also supports table row selection and search). Investigation pane details: Verdict badge, 0-100 score, 3-5 plain-English reasons, Customer Comparison table, Customer Recent Transactions, K-Means profile, HMM Viterbi state path, What to check next, Download Report button.
  - `Algorithms`: Interactive sections for K-Means, GMM, EM, PCA, HMM, Forward, Viterbi, CART, XGBoost, and Ensemble. Each section displays plain description, mathematical formula, rationale, interactive Run button, and real-time execution results.
  - `How it works`: End-to-end methodology, feature engineering logic, temporal split rationale, and synthetic dataset disclosure.

## 11. Revision Log
- **2026-11-15**: Initial specification and architecture definition.
- **2026-11-15**: Full implementation completed. Model training achieved 0.9176 Test PR-AUC, 0.9981 ROC-AUC, 85.42% precision, 86.29% recall, and 0.8585 F1 at validation-tuned threshold 0.9500. All 16 unit, math, leakage, compliance, and AppTest UI tests passed.
- **2026-11-15**: UI Colour Update completed. Centralized 12-color palette in `src/config.py` and `.streamlit/config.toml` with zero hex values in other source files. All text-on-color contrast ratios exceed 4.5:1 (up to 16.9:1). Applied custom Plotly template globally, thin left borders on flagged table rows, and updated all 21 automated tests.
- **2026-11-15**: Added direct "Investigation" button column next to Risk Score in the Transactions table using `st.column_config.ButtonColumn`, enabling immediate forensic investigation pane activation on single click.
- **2026-11-15**: Production UI enhancement. Added system metadata status bar, color-coded top stripes on KPI cards, headline objective tags, 4-quadrant executive confusion matrix cards, calibrated risk spectrum gauge in the investigation dossier, behavioral deviation variance tags, and styled HMM Viterbi state path pills. All 22 tests passing with full WCAG AA compliance.

