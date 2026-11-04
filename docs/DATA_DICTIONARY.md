# AuditLens — Data Dictionary

**Document:** Data Dictionary & Schema Specification  
**Version:** 1.0.0  
**Date:** 2026-11-04  

---

## 1. Primary Tabular Dataset Schema (`creditcard.csv` & `creditcard_sample_fixture.csv`)

| Column Name | Data Type | Physical Range / Values | Unit | Description | Role in ML Pipeline |
|---|---|---|---|---|---|
| `Time` | `float64` | $[0.0, 172792.0]$ | Seconds | Number of seconds elapsed between this transaction and the first transaction in the dataset. | Temporal feature; used for chronological train/test split. |
| `V1` | `float64` | $[-56.41, 2.45]$ | Dimensionless | 1st Principal Component from PCA transformation. | Quantitative predictor (clustering, tree splitting, boosting). |
| `V2` | `float64` | $[-72.72, 22.06]$ | Dimensionless | 2nd Principal Component from PCA transformation. | Quantitative predictor. |
| `V3` | `float64` | $[-48.33, 9.38]$ | Dimensionless | 3rd Principal Component from PCA transformation. | Quantitative predictor (strong fraud differentiator). |
| `V4` | `float64` | $[-5.68, 16.88]$ | Dimensionless | 4th Principal Component from PCA transformation. | Quantitative predictor. |
| `V5` ... `V28` | `float64` | Unbounded real values | Dimensionless | Principal Components 5 through 28 from PCA transformation. | Quantitative predictors. |
| `Amount` | `float64` | $[0.0, 25691.16]$ | EUR / USD | Transaction monetary value. | Numerical feature; requires robust scaling or log transformation due to high skewness. |
| `Class` | `int64` | $\{0, 1\}$ | Binary Flag | Fraud status label: `0` = Legitimate, `1` = Fraudulent. | Ground truth target variable for supervised models (CART, XGBoost, Ensembles). Excluded from unsupervised modules. |

---

## 2. Sequential Benchmark Dataset Schema (`sequence_account_fixture.csv`)

| Column Name | Data Type | Permitted Values | Unit | Description | Role in Sequential Modeling (Unit 4) |
|---|---|---|---|---|---|
| `customer_id` | `string` | e.g. `ACC_0001` | Identifier | Unique account or cardholder identifier. | Sequence grouping key. |
| `timestamp` | `int64` | Positive integers | Seconds | Epoch or relative transaction timestamp. | Chronological sorting key within account trajectory. |
| `step` | `int64` | $[0, 49]$ | Sequence Index | Discrete step order of transaction for this account. | Step validation and ordering check. |
| `tx_type` | `string` | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT` | Discrete Category | Payment channel or operational transaction type. | Categorical emission observation in HMM. |
| `amount` | `float64` | $[5.0, 4500.0]$ | Currency | Transaction monetary value. | Continuous observation feature in HMM. |
| `latent_state` | `int64` | $\{0, 1, 2\}$ | Latent State | Ground truth Markov behavioral state: `0`=Normal, `1`=Elevated, `2`=High-Risk Burst. | Evaluation benchmark for Viterbi state path decoding. |
| `is_fraud` | `int64` | $\{0, 1\}$ | Binary Flag | Fraud indicator for the transaction. | Secondary evaluation label. |

