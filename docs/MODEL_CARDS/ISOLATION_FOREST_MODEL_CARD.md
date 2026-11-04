# Model Card: Isolation Forest Anomaly Detector

## 1. Model Details
- **Model Name:** Isolation Forest
- **Model Type:** Unsupervised Ensemble of Isolation Trees (iTrees)
- **Library:** `scikit-learn` (`sklearn.ensemble.IsolationForest`)
- **Version:** 1.0.0
- **Implementation File:** [`backend/app/services/ml/isolation_forest.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/isolation_forest.py)
- **Role in Platform:** Unsupervised statistical rarity benchmark to contrast against supervised fraud probability estimators.

---

## 2. Intended Use & Semantic Boundary
- **Intended Use:** Identifies transactions that are isolated quickly in feature space with few recursive splits.
- **Score Taxonomy Separation:** The Isolation Forest score is strictly an **unsupervised tree-depth anomaly score** in $[-1.0, 1.0]$, where negative values denote anomalies.
- **Out-of-Scope Use:** It is **not** a calibrated probability of fraud and must never be interpreted as proof that a transaction is fraudulent. Genuine legitimate transactions (e.g. rare high-value luxury purchases or unusual time-of-day orders) naturally produce high anomaly scores.

---

## 3. Mathematical Formulation
The anomaly score $s(x, n)$ for an observation $x$ across an ensemble of $n$ trees is:

$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$

where:
- $h(x)$ is the path length (number of edges traversed from root to terminal node).
- $c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree.

---

## 4. Hyperparameters
- `n_estimators`: 100
- `contamination`: 0.01 (estimated proportion of anomalies in dataset)
- `random_state`: 42
- `max_samples`: "auto"

---

## 5. Ethical Considerations
Because Isolation Forest is entirely unsupervised, it does not rely on human fraud labels. However, its isolation criteria can flag atypical but legitimate financial transactions, disproportionately impacting edge-case consumers without human adjudication.

