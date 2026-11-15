# Dataset Specification: Sparkov Credit Card Fraud

## Overview
- **Source**: Kaggle `kartik2112/fraud-detection` (Sparkov synthetic credit card transaction generator).
- **Nature of Data**: **SYNTHETIC**. The transactions, names, merchant labels, and fraud patterns are generated using a simulation framework.
- **Purpose in AuditLens**: Training, validating, and testing machine learning models to detect fraudulent transactions and explain suspicious patterns.

## Verified Dataset Splits

| Split | Rows | Date Start | Date End | Normal Rows | Fraud Rows | Fraud Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (80% of fraudTrain)** | 1,037,340 | 2019-01-01 00:00:18 | 2020-03-06 07:15:17 | 1,031,372 | 5,968 | 0.5753% |
| **Validation (20% of fraudTrain)** | 259,335 | 2020-03-06 07:16:43 | 2020-06-21 12:13:37 | 257,797 | 1,538 | 0.5931% |
| **Full fraudTrain** | 1,296,675 | 2019-01-01 00:00:18 | 2020-06-21 12:13:37 | 1,289,169 | 7,506 | 0.5789% |
| **Test (fraudTest)** | 555,719 | 2020-06-21 12:14:25 | 2020-12-31 23:59:34 | 553,574 | 2,145 | 0.3860% |

## Split Methodology
- **Temporal Strictness**: Chronological time splits are strictly enforced. Shuffling across time is prohibited to prevent lookahead bias.
- **Train Window**: Earlier 80% of chronological transactions from `fraudTrain.csv`.
- **Validation Window**: Subsequent 20% of chronological transactions from `fraudTrain.csv`, used strictly for decision threshold tuning and hyperparameter selection.
- **Holdout Test**: Entire `fraudTest.csv` dataset, representing unseen future time period for final model evaluation.

## Feature Exclusions
The following fields are strictly excluded from predictive model inputs to prevent overfitting and synthetic artifact leakage:
- Identifier fields: `cc_num`, `trans_num`, `Unnamed: 0`
- Personal identification: `first`, `last`, `street`, `dob`
- System timestamps: `unix_time`
