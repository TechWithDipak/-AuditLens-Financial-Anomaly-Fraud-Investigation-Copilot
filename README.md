# AuditLens

AuditLens is a credit card transaction fraud review and explanation application. It flags transactions into Normal, Suspicious, or Fraud, explains why a transaction looks suspicious in plain English, and generates exportable forensic audit reports.

**Notice on Dataset**: The underlying transactions originate from the Sparkov synthetic fraud generator (Kaggle `kartik2112/fraud-detection`). All transaction records, cardholder profiles, and merchant names are synthetic simulation records.

---

## Architecture Overview

AuditLens combines unsupervised anomaly detection (PCA, K-Means customer profiling, hand-written EM-GMM, and HMM sequence modeling) with supervised tree models (CART and XGBoost). Local feature attributions from Tree SHAP and customer behavioral baselines are translated into natural language explanations.

---

## Installation

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Set Up Virtual Environment

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Dataset Placement
Ensure the raw Sparkov dataset files are placed in `data/raw/`:
- `data/raw/fraudTrain.csv`
- `data/raw/fraudTest.csv`

---

## Running the Application

The system requires exactly two commands:

### Step 1: Train Models and Precompute Test Scores
```bash
python train.py
```
This fits all preprocessors on the chronological train split, tunes the decision threshold on validation PR curve, evaluates on the full 555,719 test set, and saves metrics to `artifacts/metrics.json` and scored transactions to `artifacts/test_scored.parquet`.

### Step 2: Launch the Web Interface
```bash
streamlit run app.py
```
The application opens in the default web browser at `http://localhost:8501`.

If artifacts are missing, the application displays a plain message advising the user to run `python train.py` first.

---

## Operating the Web Application

The interface is organized into four tabs:

1. **Overview**: Displays total evaluated test transactions (555,719), flagged counts, actual fraud rate (0.3860%), headline test PR-AUC (0.9176), ROC-AUC (0.9981), precision (85.42%), recall (86.29%), F1-score (0.8585), a risk score distribution histogram, and the confusion matrix.
2. **Transactions**: Filterable, searchable, and paginated transaction ledger. Selecting any transaction displays its classification verdict (Fraud, Suspicious, or Normal), 0 to 100 risk score, 3 to 5 plain-English reasons, a customer baseline comparison table, recent transactions, customer K-Means segment, HMM Viterbi sequential state path, investigative checklist, and a button to download the audit report.
3. **Algorithms**: Interactive walkthroughs of K-Means, GMM, EM, PCA, HMM, Forward, Viterbi, CART, XGBoost, and the Ensemble model with mathematical equations, rationale, interactive run buttons, and validation against scikit-learn and hmmlearn.
4. **How it works**: End-to-end explanation of the data pipeline, chronological temporal splits, causal feature engineering, and synthetic data rationale.

Sidebar controls allow adjusting the USD to INR conversion rate (default 83.0) and reviewing the operational glossary.

---

## Automated Testing

To run the full unit, leakage, mathematical equivalence, compliance, and headless UI test suite:

```bash
pytest -v
```

All 16 test cases verify:
- Causal feature engineering without lookahead bias (`tests/test_leakage.py`)
- EM-GMM convergence and analytical match against scikit-learn (`tests/test_em_gmm.py`)
- Forward and Viterbi exact numerical match against hmmlearn (`tests/test_hmm.py`)
- Headless Streamlit interface execution and report generation via AppTest (`tests/test_app.py`)
- Project-wide compliance against banned hype words, emojis, and non-grayscale colors (`tests/test_compliance.py`)
