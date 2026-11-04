# AuditLens — Model Card: XGBoost Gradient Boosted Trees

**Model Identifier:** `XGBoost_Classifier`  
**Algorithm:** Extreme Gradient Boosting (XGBoost)  
**Task:** High-Performance Supervised Binary Fraud Classification  
**Version:** 1.0.0  
**Date:** 2026-11-04  

---

## 1. Model Overview & Purpose
XGBoost is the primary high-capacity supervised classifier in AuditLens. It fits additive decision trees sequentially by optimizing a regularized objective function combining convex loss and tree complexity:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

It captures non-linear feature interactions and high-order patterns while managing extreme class imbalance via explicit cost weighting.

---

## 2. Hyperparameters & Imbalance Strategy
- **Max Depth:** 4
- **Learning Rate ($\eta$):** 0.1
- **Number of Estimators:** 100
- **Evaluation Metric:** `eval_metric='aucpr'` (Area under the Precision-Recall curve)
- **Class Imbalance Weight:** `scale_pos_weight = N_neg / N_pos` (Empirically computed from training partition)
- **Random Seed:** 42

---

## 3. Evaluation & Metrics
- **Primary Metric:** Precision-Recall AUC (Average Precision)
- **Secondary Metrics:** F1-Score, Precision, Recall, ROC-AUC
- **Operational Metrics:** Confusion Matrix (`tn`, `fp`, `fn`, `tp`), False Positive Rate, False Negative Rate
- **Target Leakage Safeguard:** Preprocessing fitted strictly on training partition; zero test set parameter leakage.

---

## 4. Feature Importance
- Global importance computed across all boosted trees (gain and split weight).
- Integrated with TreeSHAP explainability in Phase 7 for local per-transaction attribution.

---

## 5. Limitations
- More complex to audit than a single decision tree.
- Requires careful calibration when estimating raw posterior probabilities under extreme `scale_pos_weight`.

