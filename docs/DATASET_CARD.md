# AuditLens — Dataset Card: Credit Card Fraud Detection Benchmark

## 1. Dataset Summary
- **Dataset Name**: Credit Card Fraud Detection
- **Original Publishers**: Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB), Belgium.
- **Reference Publications**:
  - Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
- **Source URL**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Licence**: Open Database License (ODbL) v1.0. Commercial and research use permitted with attribution.
- **Acquisition Method**: Direct download of `creditcard.csv` into local `data/raw/` (strictly excluded from Git tracking via `.gitignore`).

---

## 2. Dataset Composition & Dimensions
- **Total Transactions (Rows)**: 284,807
- **Total Features (Columns)**: 31
  - 30 input features (`Time`, `Amount`, `V1` through `V28`)
  - 1 binary ground-truth target (`Class`)
- **Time Horizon**: Transactions recorded over a period of 2 days (48 hours / 172,792 seconds) in September 2013 by European cardholders.
- **Class Distribution & Imbalance**:
  - Legitimate Transactions (`Class = 0`): 284,315 (99.828%)
  - Fraudulent Transactions (`Class = 1`): 492 (0.172%)
  - Severe class imbalance ratio: ~578:1 legitimate to fraud.

---

## 3. Data Transformation & Privacy Protection
- **PCA Anonymization**: Due to confidentiality agreements and European data protection laws, the original background features (e.g., merchant ID, cardholder demographic data, location, transaction history) were transformed using Principal Component Analysis (PCA).
- **Non-transformed Features**:
  - `Time`: Elapsed seconds from the initial transaction in the dataset.
  - `Amount`: Monetary transaction amount (in EUR).
- **Target Label**:
  - `Class`: Binary indicator where `1` represents confirmed fraudulent activity and `0` represents legitimate activity.
- **PII & Sensitivity Assessment**:
  - The dataset contains **zero raw PII** (names, addresses, phone numbers, raw 16-digit credit card PAN numbers, CVVs, or IP addresses).
  - All numerical features `V1`–`V28` are orthogonal projections resulting from PCA.

---

## 4. Known Limitations & Usage Constraints
1. **Lack of Semantic Feature Names**: Because `V1`–`V28` are anonymized PCA dimensions, domain-specific semantic rules (e.g., "high-risk merchant category" or "foreign card issuance") cannot be directly validated by feature names alone.
2. **Short Temporal Window**: Spanning only 48 hours, the dataset cannot reflect multi-month seasonality, holiday purchasing spikes, or evolving long-term fraud patterns.
3. **Severe Imbalance Constraint**: Standard evaluation metrics like classification accuracy or ROC-AUC can be misleading. PR-AUC (Average Precision), precision, recall, and cost-weighted confusion matrices are mandatory.
4. **Decision-Support Scope**: This dataset is used strictly to train and evaluate baseline predictive models. Model outputs represent statistical risk estimates, not definitive legal determinations.

---

## 5. Synthetic Fixtures Policy
For unit testing and local developer onboarding where raw data is not pre-downloaded:
- Small synthetic CSV files are maintained in `data/sample/`.
- Synthetic data is explicitly tagged with metadata `is_synthetic = True` and must never be reported as real-world detection metrics.

