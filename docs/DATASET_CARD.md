# AuditLens — Dataset Card

**Document:** Dataset Card & Provenance Specification  
**Version:** 1.0.0  
**Date:** 2026-11-04  
**Primary Dataset:** Kaggle Credit Card Fraud Detection (ULB)  
**Secondary Benchmark:** Synthetic Financial Account Sequence Dataset  

---

## 1. Primary Dataset: Credit Card Fraud Detection

### Overview
- **Source:** Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) / Kaggle.
- **Reference URL:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Licence:** Open Database License (ODbL) v1.0.
- **Permitted Use:** Open academic, commercial, and research use with attribution.
- **Expected Local Path:** `data/raw/creditcard.csv` (strictly excluded from Git via `.gitignore`).
- **Synthetic Test Fixture:** `data/fixtures/creditcard_sample_fixture.csv` (1,000 samples, tracked in Git for CI).

### Dataset Characteristics
- **Row Count:** 284,807 transactions.
- **Column Count:** 31 features.
- **Temporal Span:** Transactions occurred over 2 days in September 2013 by European cardholders.
- **Target Variable:** `Class` (Binary):
  - `0`: Legitimate transaction ($N = 284,315$, $99.828\%$).
  - `1`: Fraudulent transaction ($N = 492$, $0.172\%$).
- **Class Imbalance Ratio:** 1:578 (Extreme imbalance).

### Privacy & Feature Anonymization
Due to confidentiality and banking privacy regulations, the original raw financial features were transformed using Principal Component Analysis (PCA) by the dataset curators:
- Features `V1`, `V2`, ..., `V28` represent the principal components of the original feature space.
- Features `Time` and `Amount` were not transformed and represent original physical dimensions.
- No cardholder names, Primary Account Numbers (PANs), dates of birth, or geographic identifiers are contained in the dataset.

### Known Limitations
1. **Lack of Account Trajectories:** The dataset does not provide cardholder or account identifiers. Each row is an isolated transaction with elapsed seconds (`Time`) since the first recording. It cannot model multi-step sequential customer behavior over days or months.
2. **Feature Interpretability Tradeoff:** Because `V1`–`V28` are orthogonal linear combinations of undisclosed original features, feature attribution (e.g. SHAP) identifies *which component* influenced the prediction, but domain analysts cannot map `V1` directly to a specific merchant category code or terminal type.

---

## 2. Secondary Benchmark: Synthetic Financial Account Sequence Dataset

### Overview
- **Purpose:** Specifically created to demonstrate Unit 4 Sequential Algorithms (Hidden Markov Models, Forward Algorithm, Viterbi Algorithm) without violating dataset provenance or inventing false history on anonymized data.
- **Location:** `data/fixtures/sequence_account_fixture.csv`.
- **Row Count:** 500 transactions across 10 distinct customer accounts (`ACC_0001` through `ACC_0010`).
- **Chronological Ordering:** Sorted strictly by `customer_id` and ascending `timestamp`.

### Latent Behavioral States
1. **State 0 (Normal Day-to-Day):** Low-to-moderate transaction amounts ($5 – $120), predominantly `PAYMENT` or `DEBIT`, low transition probability to high-risk states.
2. **State 1 (Elevated Activity / Travel):** Moderate amounts ($100 – $600), increased transaction velocity, occasional `TRANSFER`.
3. **State 2 (High-Risk Burst / Account Takeover):** High-value amounts ($800 – $4,500), rapid succession of `TRANSFER` or `CASH_OUT`, high correlation with fraud labels.

---

## 3. Data Acquisition & Kaggle Download Instructions

To download the full 284,807-row Kaggle Credit Card dataset:

### Option A: Kaggle CLI (Recommended)
1. Install Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (on Windows: `C:\Users\<Username>\.kaggle\kaggle.json`).
   *Never commit `kaggle.json` to version control.*
3. Download the dataset into `data/raw/`:
   ```powershell
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```
4. Verify the file exists at `data/raw/creditcard.csv`.

### Option B: Web Browser Download
1. Visit `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`.
2. Click **Download (66 MB)**.
3. Extract `creditcard.csv` directly into the `data/raw/` directory.

