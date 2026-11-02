# AuditLens — Data Dictionary: Credit Card Transaction Schema

## 1. Schema Specifications

The schema defines the standard tabular input expected by AuditLens for transaction ingestion, validation, and scoring.

| Column Name | Storage Type | Physical Domain | Description | Value Range / Constraints | Pipeline Role |
|---|---|---|---|---|---|
| `Time` | `float64` / `int64` | Elapsed Seconds | Number of seconds elapsed between this transaction and the first transaction in the dataset. | $\ge 0.0$ (typically $0$ to $172,800$ for 48h) | Temporal ordering & feature engineering |
| `V1` | `float64` | PCA Dimension 1 | 1st Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-56.4, 2.45]$) | Feature input |
| `V2` | `float64` | PCA Dimension 2 | 2nd Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-72.7, 22.0]$) | Feature input |
| `V3` | `float64` | PCA Dimension 3 | 3rd Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-48.3, 9.38]$) | Feature input |
| `V4` | `float64` | PCA Dimension 4 | 4th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-5.68, 16.88]$) | Feature input |
| `V5` | `float64` | PCA Dimension 5 | 5th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-113.7, 34.8]$) | Feature input |
| `V6` | `float64` | PCA Dimension 6 | 6th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-26.2, 73.3]$) | Feature input |
| `V7` | `float64` | PCA Dimension 7 | 7th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-43.6, 120.6]$) | Feature input |
| `V8` | `float64` | PCA Dimension 8 | 8th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-73.2, 20.0]$) | Feature input |
| `V9` | `float64` | PCA Dimension 9 | 9th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-13.4, 15.6]$) | Feature input |
| `V10` | `float64` | PCA Dimension 10 | 10th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-24.6, 23.7]$) | Feature input |
| `V11` | `float64` | PCA Dimension 11 | 11th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-4.8, 12.0]$) | Feature input |
| `V12` | `float64` | PCA Dimension 12 | 12th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-18.7, 7.8]$) | Feature input |
| `V13` | `float64` | PCA Dimension 13 | 13th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-5.8, 7.1]$) | Feature input |
| `V14` | `float64` | PCA Dimension 14 | 14th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-19.2, 10.5]$) | Feature input |
| `V15` | `float64` | PCA Dimension 15 | 15th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-4.5, 8.9]$) | Feature input |
| `V16` | `float64` | PCA Dimension 16 | 16th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-14.1, 17.3]$) | Feature input |
| `V17` | `float64` | PCA Dimension 17 | 17th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-25.2, 9.3]$) | Feature input |
| `V18` | `float64` | PCA Dimension 18 | 18th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-9.5, 5.0]$) | Feature input |
| `V19` | `float64` | PCA Dimension 19 | 19th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-7.2, 5.6]$) | Feature input |
| `V20` | `float64` | PCA Dimension 20 | 20th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-54.5, 39.4]$) | Feature input |
| `V21` | `float64` | PCA Dimension 21 | 21st Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-34.8, 27.2]$) | Feature input |
| `V22` | `float64` | PCA Dimension 22 | 22nd Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-10.9, 10.5]$) | Feature input |
| `V23` | `float64` | PCA Dimension 23 | 23rd Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-44.8, 22.5]$) | Feature input |
| `V24` | `float64` | PCA Dimension 24 | 24th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-2.8, 4.6]$) | Feature input |
| `V25` | `float64` | PCA Dimension 25 | 25th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-10.3, 7.5]$) | Feature input |
| `V26` | `float64` | PCA Dimension 26 | 26th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-2.6, 3.5]$) | Feature input |
| `V27` | `float64` | PCA Dimension 27 | 27th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-22.6, 31.6]$) | Feature input |
| `V28` | `float64` | PCA Dimension 28 | 28th Principal Component derived from anonymized original transaction features. | Unbounded continuous ($\sim [-15.4, 33.8]$) | Feature input |
| `Amount` | `float64` | Currency (EUR) | Monetary value of the transaction. | $\ge 0.0$ ($\text{Mean: } \approx 88.35$, $\text{Max: } 25,691.16$) | Feature input |
| `Class` | `int64` | Binary Flag | Fraud status ground-truth label. | $0$ (Legitimate) or $1$ (Fraudulent) | Target label (supervised training only) |

---

## 2. Validation & Quality Rules

1. **Required Features**:
   - For **scoring/inference**: `Time`, `Amount`, and all `V1`–`V28` must be present. `Class` is optional.
   - For **supervised training**: `Class` is mandatory.
2. **Missing Values**:
   - The reference dataset has **0 missing values** across all 284,807 rows.
   - If missing values are detected in inference uploads, they must be flagged by the validator.
3. **Value Bounds**:
   - `Amount` must be $\ge 0.0$. Negative amounts indicate corrupt data or reversal errors and will be rejected.
   - `Class`, when provided, must strictly be in $\{0, 1\}$.
4. **Duplicates**:
   - The original dataset contains 1,081 duplicate rows. The pipeline must detect and document duplicates to prevent cross-split leakage.

