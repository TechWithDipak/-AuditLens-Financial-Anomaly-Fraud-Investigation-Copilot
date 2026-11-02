# AuditLens — Exploratory Data Analysis Report

## 1. Overview & Dataset Dimensions
- **Total Transactions**: 150
- **Total Features**: 31

## 2. Class Imbalance Analysis
- **Legitimate Count (`Class = 0`)**: 141
- **Fraud Count (`Class = 1`)**: 9
- **Fraud Percentage**: 6.0%
- **Imbalance Ratio**: 15.67:1
> **Operational Constraint**: Standard accuracy cannot be used as an evaluation metric. A naive model predicting legitimate for every transaction would attain 99%+ accuracy while missing all fraud. PR-AUC (Average Precision), Recall@k, and precision/recall trade-offs must be prioritized.

## 3. Top Predictive Features (Target Correlation)
### Top Positive Correlators (Elevated in Fraud):
- **V4**: $+0.6463$
- **V11**: $+0.5522$
- **V13**: $+0.1695$
- **Amount**: $+0.1591$
- **V20**: $+0.0923$

### Top Negative Correlators (Depressed in Fraud):
- **V16**: $-0.0156$
- **V19**: $-0.0193$
- **Time**: $-0.0322$
- **V6**: $-0.0471$
- **V3**: $-0.0529$

## 4. Target Leakage & Post-Outcome Field Audit
- **Zero target leakage detected**: No feature exhibits suspiciously perfect correlation ($|r| \ge 0.98$) with the target variable.

## 5. Temporal Patterns & Coverage
- **Observed Time Window**: 46.48 hours (1272.2s to 168587.5s)
- **Split Strategy Recommendation**: For models evaluating temporal drift, time-based sequential splits (`Time` ordering) should be supported alongside stratified splits.
