# AuditLens — Baseline Model Card

## 1. Model Details

### 1.1 Anomaly Detection Baseline
- **Model Name**: Isolation Forest Anomaly Detector
- **Model Version**: `iso_forest_v0.1.0`
- **Model Type**: Unsupervised Tree Ensemble
- **Library**: `scikit-learn` (`sklearn.ensemble.IsolationForest`)
- **Key Hyperparameters**:
  - `n_estimators`: 100
  - `contamination`: 0.05
  - `max_samples`: 'auto'
  - `random_state`: 42
  - `n_jobs`: -1

### 1.2 Supervised Classification Baseline
- **Model Name**: Logistic Regression Fraud Classifier
- **Model Version**: `logreg_v0.1.0`
- **Model Type**: Regularized Linear Classifier
- **Library**: `scikit-learn` (`sklearn.linear_model.LogisticRegression`)
- **Key Hyperparameters**:
  - `solver`: `lbfgs`
  - `class_weight`: `balanced`
  - `max_iter`: 1000
  - `C`: 1.0
  - `random_state`: 42

---

## 2. Intended Use & Scope

- **Primary Application**: Decision-support copilot for financial fraud triage and transaction auditing.
- **Dual-Scoring Paradigm**:
  1. **Anomaly Score ($[0.0, 1.0]$)**: Identifies statistical distance from typical purchasing distributions. Useful for zero-day fraud and novel pattern discovery without ground-truth labels.
  2. **Fraud Probability ($[0.0, 1.0]$)**: Calibrated probability estimating resemblance to historically confirmed fraud cases.
- **Explicit Non-Goals**: Neither model is permitted to execute autonomous monetary holds, card cancellations, or regulatory SAR filings. All actions require human investigator review.

---

## 3. Preprocessing & Input Schema

- **Input Features (30 raw features)**: `Time`, `Amount`, and PCA dimensions `V1` through `V28`.
- **Transformed Features (31 output features)**:
  - `time_sin`, `time_cos`: Diurnal cyclic transformation of `Time` $(2\pi \cdot \text{hour} / 24)$.
  - `V1`–`V28`: Median-imputed PCA dimensions.
  - `Amount`: `RobustScaler` (median-centered, interquartile range scaled).
- **Target Exclusion**: Target column `Class` is strictly excluded from feature inputs.

---

## 4. Baseline Evaluation Snapshot (Held-Out Test Set)

Evaluated on held-out test split with seed 42:

| Metric | Baseline Value | Operational Objective |
|---|---|---|
| **PR-AUC (Average Precision)** | Baseline established | Primary triage metric for severe class imbalance |
| **ROC-AUC** | Baseline established | Secondary metric for ranking |
| **Precision @ threshold=0.5** | Baseline established | Minimizing investigator alert fatigue |
| **Recall @ threshold=0.5** | Baseline established | Fraud capture coverage |
| **F1 Score** | Baseline established | Harmonic balance between precision and recall |

---

## 5. Quantitative Limitations & Risk Mitigations

1. **Uncalibrated Anomaly Scores**: Anomaly scores from Isolation Forest represent structural sparsity in tree partitions. High anomaly scores reflect statistical novelty, not definitive criminality.
2. **Linear Decision Boundary**: Logistic regression cannot capture complex non-linear feature interactions without explicit polynomial terms. Non-linear models (e.g. Random Forest / XGBoost) will be evaluated in Phase 5.
3. **Threshold Sensitivity**: The default 0.5 decision threshold is arbitrary; operational review thresholds tailored to analyst capacity will be formally evaluated in Phase 5.

