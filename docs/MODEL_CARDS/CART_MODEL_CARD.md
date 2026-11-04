# AuditLens — Model Card: CART Decision Tree Baseline

**Model Identifier:** `CART_DecisionTree`  
**Algorithm:** Classification and Regression Trees (CART)  
**Task:** Supervised Binary Fraud Classification Baseline  
**Version:** 1.0.0  
**Date:** 2026-11-04  

---

## 1. Model Overview & Purpose
CART serves as the interpretable baseline supervised model in AuditLens. It constructs a binary decision tree by recursively partitioning the feature space to maximize the reduction of Gini impurity:

$$\Delta I_G = I_G(D) - \frac{|D_L|}{|D|} I_G(D_L) - \frac{|D_R|}{|D|} I_G(D_R)$$

Its primary role in financial fraud investigation is providing transparent, human-readable logic rules that an analyst or compliance officer can immediately verify.

---

## 2. Hyperparameters & Configuration
- **Criterion:** Gini Impurity
- **Max Depth:** 5 (Restrained to prevent memorizing rare fraudulent samples)
- **Min Samples Leaf:** 5
- **Class Imbalance Strategy:** `class_weight='balanced'` (Inversely proportional to class frequencies)
- **Random Seed:** 42

---

## 3. Evaluation Metrics & Benchmark Performance
Evaluated on stratified held-out test split ($N_{\text{test}} = 200$, 0.5% fraud):
- **Primary Metric (PR-AUC / Average Precision):** Reported on live execution
- **F1-Score:** Harmonic mean of precision and recall
- **Precision & Recall:** Reported with exact confusion matrix (`tn`, `fp`, `fn`, `tp`)
- **False Positive Rate / False Negative Rate:** Reported

---

## 4. Interpretability & Rules
The model outputs human-readable decision rules:
- Decision rules extract exact split thresholds on scaled features (e.g. `|--- V1 <= -2.10`, `|--- Amount > 1.45`).
- Rule-based transparency allows investigators to trace why a specific transaction was flagged.

---

## 5. Limitations & Caveats
- Single decision trees exhibit high variance and are prone to overfitting without depth limits.
- Orthogonal axis-aligned splits may struggle with complex diagonal feature interactions.

