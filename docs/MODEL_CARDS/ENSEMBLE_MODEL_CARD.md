# AuditLens — Model Card: Ensemble Learning (Voting & Stacking)

**Model Identifier:** `Ensemble_Voting` / `Ensemble_Stacking`  
**Algorithm:** Model Combination via Soft Voting & Out-of-Fold Stacking  
**Task:** Robust Supervised Binary Fraud Classification  
**Version:** 1.0.0  
**Date:** 2026-11-04  

---

## 1. Model Overview & Purpose
The Ensemble module evaluates whether combining distinct algorithmic paradigms (Interpretable Trees + Gradient Boosted Trees) improves test set generalization and stability on highly imbalanced fraud distributions.

---

## 2. Combination Strategies
1. **Soft-Voting Ensemble:**
   Computes the unweighted or weighted average of posterior predicted probabilities:
   $$P_{\text{ens}}(y = 1 \mid x) = \frac{1}{M} \sum_{m=1}^M P_m(y = 1 \mid x)$$
2. **Stacking Classifier:**
   Employs a meta-learner (regularized Logistic Regression) trained strictly on 3-fold cross-validated out-of-fold predictions from base learners (CART and XGBoost). This prevents meta-learner data leakage.

---

## 3. Fair Benchmarking Protocol
To guarantee objective evaluation:
- Base models and ensembles are evaluated on the **exact same held-out test split**.
- Evaluation prioritizes PR-AUC and F1-Score over raw accuracy.
- Head-to-head comparison table surfaces whether the ensemble outperformed the strongest individual constituent model.

---

## 4. Limitations
- Inference latency is the sum of constituent base model evaluations.
- Soft voting requires calibrated probability outputs across diverse base learners.

