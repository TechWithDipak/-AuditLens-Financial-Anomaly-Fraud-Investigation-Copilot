# AuditLens: Project Technical Documentation

## 1. Executive Summary and Synthetic Data Disclosure
AuditLens is a credit card fraud detection, explanation, and reporting system. It flags incoming card authorizations into Normal, Suspicious, or Fraud, translates algorithmic decisions into plain-English sentences, and outputs printable forensic audit documents.

**Dataset Notice**: The data utilized throughout this system is strictly synthetic, generated via the Sparkov credit card simulation framework (Kaggle `kartik2112/fraud-detection`). No actual credit card numbers, personal identities, or financial records were processed.

---

## 2. System Architecture and Component Responsibilities

The codebase follows a flat, modular architecture:

- `app.py`: The single-entry Streamlit user interface structured across four dedicated tabs using a minimal monochrome aesthetic.
- `train.py`: Full training pipeline executing causal feature extraction, unsupervised fitting, supervised modeling, validation threshold tuning, test evaluation, and artifact serialization.
- `requirements.txt`: Exact pinned dependency specifications.
- `.streamlit/config.toml`: Streamlit theme configuration enforcing white background (#FFFFFF), dark text (#1A1D21), subtle gray panels (#F5F7FA), borders (#D9DEE5), and accent blue (#1F4E8C).
- `src/config.py`: Central repository paths, currency conversion constants (`USD_TO_INR = 83.0`), random seed (`42`), and column definitions.
- `src/data.py`: Raw dataset loading, verification, and chronological splitting (80% train, 20% validation, holdout test).
- `src/features.py`: Vectorized causal rolling customer statistics (`secs_since_last_txn`, `amt_zscore_vs_customer_history`, `txn_count_last_24h`), haversine distance calculation, and train-only `FeaturePipeline`.
- `src/em_gmm.py`: Hand-written Gaussian Mixture Model trained via Expectation-Maximization in NumPy, computing transaction density scores.
- `src/hmm.py`: Discrete 12-symbol observation encoder, Baum-Welch training wrapper, and hand-written Forward and Viterbi algorithms.
- `src/models.py`: Customer-level K-Means segmentation, PCA dimensionality reduction, CART decision tree, XGBoost classifier, soft-voting ensemble, and validation threshold optimizer.
- `src/explain.py`: SHAP TreeExplainer wrapper and natural language sentence translation engine.
- `src/report.py`: Printable monochrome HTML and Markdown audit report generators.
- `tests/`: Automated unit tests covering leakage prevention, EM-GMM equivalence with scikit-learn, Forward/Viterbi equivalence with hmmlearn, static compliance, and AppTest headless UI tests.
- `artifacts/`: Serialized models, scalers, `metrics.json`, and precomputed `test_scored.parquet`.
- `docs/`: Dataset documentation (`DATA.md`) and project memory logs.

---

## 3. Mathematical Foundations of Algorithms

### 3.1 Principal Component Analysis (PCA)
PCA projects scaled feature vectors $X \in \mathbb{R}^{N \times D}$ into an orthogonal subspace of lower dimensionality $d = 5$:

$$\max_{w} \frac{w^T X^T X w}{w^T w} \quad \text{subject to } w_i^T w_j = 0 \; (\forall i \ne j)$$

The first 5 components capture maximum feature variance while decoupling collinear variables, serving as continuous inputs to the Gaussian Mixture Model.

### 3.2 K-Means Customer Behavioral Clustering
Cardholders are aggregated into behavioral feature vectors summarizing average amount, variance, daily transaction rate, category shares, and average merchant distance:

$$\arg\min_{S} \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$$

Where $\mu_i$ represents the centroid of cluster $S_i$. K-Means establishes peer groups so that an authorization is evaluated relative to the customer's typical behavioral cohort.

### 3.3 Gaussian Mixture Model (GMM) with Hand-Written EM
The probability density of feature vector $x$ is modeled as:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where $\pi_k$ are mixture weights, $\mu_k$ are component means, and $\Sigma_k$ are covariance matrices.

#### Expectation Step (E-step):
Computes posterior responsibilities $\gamma_{ik}$ in log-space:

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

$$\ln \mathcal{L} = \sum_{i=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) \right)$$

#### Maximization Step (M-step):
Updates parameters using effective component counts $N_k = \sum_{i=1}^N \gamma_{ik}$:

$$\pi_k^{\text{new}} = \frac{N_k}{N}, \quad \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$

$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T + \epsilon I$$

Lower sample log-likelihoods $\ln p(x_i)$ indicate unusual transactions that fall into low-density regions.

### 3.4 Hidden Markov Model (HMM)
A 3-state HMM models sequential spending transitions over an alphabet of 12 discrete symbols combining amount z-score (3 levels), time of day (2 levels), and merchant distance (2 levels).

#### Baum-Welch Training:
Learns state transition matrix $A$, emission matrix $B$, and initial state distribution $\pi$.

#### Hand-Written Forward Algorithm (Scaled Form):
Calculates sequence likelihood $P(O | \lambda)$ for observations $O = (o_1, \dots, o_T)$ using Rabiner scaling factors $c_t$:

$$\alpha_1(j) = \pi_j B_j(o_1), \quad c_1 = \frac{1}{\sum_{j} \alpha_1(j)}, \quad \hat{\alpha}_1(j) = c_1 \alpha_1(j)$$

$$\alpha_t(j) = \left[ \sum_{i=1}^K \hat{\alpha}_{t-1}(i) A_{ij} \right] B_j(o_t), \quad c_t = \frac{1}{\sum_{j} \alpha_t(j)}, \quad \hat{\alpha}_t(j) = c_t \alpha_t(j)$$

$$\ln P(O | \lambda) = -\sum_{t=1}^T \ln(c_t)$$

#### Hand-Written Viterbi Algorithm (Log-Space Form):
Finds the most probable hidden state path $Q^* = (q_1^*, \dots, q_T^*)$:

$$V_1(i) = \ln \pi_i + \ln B_i(o_1)$$

$$V_t(j) = \max_{1 \le i \le K} [V_{t-1}(i) + \ln A_{ij}] + \ln B_j(o_t)$$

$$ptr_t(j) = \arg\max_{1 \le i \le K} [V_{t-1}(i) + \ln A_{ij}]$$

Backtracking from $q_T^* = \arg\max_i V_T(i)$ yields the optimal sequence of behavioral states.

### 3.5 CART Decision Tree
Builds binary recursive splits of depth 4 using Gini impurity with class balancing:

$$\text{Gini}(t) = 1 - \sum_{i=1}^C p(i|t)^2$$

Provides human-auditable logic rules for edge cases and baseline classification.

### 3.6 XGBoost Classifier
The primary supervised engine minimizes regularized objective:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

To handle extreme class imbalance (172 normal transactions per 1 fraud transaction), the loss function scales positive gradients by:

$$\text{scale\_pos\_weight} = \frac{N_{\text{normal}}}{N_{\text{fraud}}} = \frac{1,031,372}{5,968} \approx 172.82$$

### 3.7 Soft Voting Ensemble
Averages predicted probabilities from XGBoost and CART:

$$P_{\text{ens}}(y = 1 | x) = 0.5 \cdot P_{\text{xgb}}(y = 1 | x) + 0.5 \cdot P_{\text{cart}}(y = 1 | x)$$

---

## 4. Real Empirical Evaluation Metrics

All numbers below originate directly from `artifacts/metrics.json` recorded during execution on the full holdout test split (`fraudTest.csv`, 555,719 rows):

### 4.1 Model Comparison on Holdout Test Set

| Metric | XGBoost (Selected Final) | CART Decision Tree | Soft Voting Ensemble |
| :--- | :--- | :--- | :--- |
| **Validation PR-AUC** | **0.9375** | 0.5210 | 0.8845 |
| **Test PR-AUC (Headline)** | **0.9176** | 0.4627 | 0.8556 |
| **Test ROC-AUC** | **0.9981** | 0.9740 | 0.9950 |
| **Precision** | **85.42%** | 22.91% | 81.84% |
| **Recall** | **86.29%** | 71.84% | 81.12% |
| **F1-Score** | **0.8585** | 0.3474 | 0.8148 |

### 4.2 Test Confusion Matrix (XGBoost at Threshold 0.9500)

| Ground Truth | Predicted Normal | Predicted Fraud | Total |
| :--- | :--- | :--- | :--- |
| **Actual Normal** | 553,258 (TN) | 316 (FP) | 553,574 |
| **Actual Fraud** | 294 (FN) | 1,851 (TP) | 2,145 |

- False Positive Rate: $316 / 553,574 = 0.057\%$
- False Negative Rate: $294 / 2,145 = 13.71\%$
- Overall Accuracy: $99.89\%$ (not used as headline due to class imbalance)

### 4.3 Unsupervised Feature Ablation Study

| Model Configuration | Test PR-AUC | Gain |
| :--- | :--- | :--- |
| **Base XGBoost** (Standard features only) | 0.8185 | Baseline |
| **Augmented XGBoost** (+ GMM, HMM, K-Means) | 0.9176 | +0.0991 |

**Assessment**: The unsupervised anomaly scores (EM-GMM likelihood, HMM sequential score, and K-Means customer cluster distance) provide a +0.0991 PR-AUC boost. They supply temporal sequence context and multi-attribute density baselines that static tree splits cannot easily reconstruct from raw columns.

### 4.4 Unsupervised Validation
- K-Means Customer Silhouette Score: **0.2484** (computed across 964 cardholder vectors).
- EM Convergence: Log-likelihood strictly increased from $-427,756.6$ to $-329,570.8$ over 23 iterations.
- Forward and Viterbi Equivalence: Hand-written implementations matched `hmmlearn` to within $10^{-5}$ relative tolerance.

---

## 5. End-to-End Walkthrough of Application Tabs

### Tab 1: Overview
Presents high-level operational statistics:
- Key cards: Evaluated Transactions (555,719), Flagged Fraud, Flagged Suspicious, Ground-Truth Fraud Rate (0.3860%).
- Model performance metrics: PR-AUC (0.9176), ROC-AUC (0.9981), Precision (85.42%), Recall (86.29%), F1-Score (0.8585).
- Risk score distribution histogram: Grayscale logarithmic bar plot showing concentration of scores near zero with a distinct tail of high-risk transactions.
- Confusion matrix: Formatted table documenting True Negatives, False Positives, False Negatives, and True Positives.

### Tab 2: Transactions
Interactive investigative console:
- Multi-parameter filtering: Verdict (All, Fraud, Suspicious, Normal), Category, Cardholder ID, and Merchant name.
- Scored transactions table: Displaying Transaction ID, Timestamp, Cardholder ID, Category, Amount in Rupees (₹), Verdict badge, and Risk Score.
- Transaction Investigation Pane:
  - Classification verdict badge (Fraud = solid black, Suspicious = outlined, Normal = muted gray).
  - 3 to 5 plain-English reasons translated from SHAP feature attributions and customer comparisons.
  - Cardholder Behavioral Comparison table contrasting current transaction attributes with the cardholder's baseline.
  - Recent transactions list for the cardholder.
  - Cardholder K-Means segment profile.
  - HMM Viterbi optimal sequential state path.
  - Investigative action checklist.
  - One-click HTML audit report download button.

### Tab 3: Algorithms
Interactive mathematical laboratory:
- Covers all 10 algorithms: PCA, K-Means, GMM, EM, HMM, Forward, Viterbi, CART, XGBoost, and Ensemble.
- Displays plain-language explanation, LaTeX mathematical formulation, and system rationale.
- Interactive Run buttons executing live calculations:
  - PCA: Plots 2D scatter of normal vs fraud transactions.
  - K-Means: Displays customer cluster summary table.
  - EM: Runs hand-written optimization on a 300-point sample, displaying the log-likelihood ascent curve.
  - HMM Forward and Viterbi: Runs validation test, confirming "Matches hmmlearn: yes".
  - CART: Renders readable text tree rules.
  - XGBoost: Displays top 10 feature importance horizontal bar chart.
  - Ensemble & Ablation: Displays empirical validation comparison and ablation study.

### Tab 4: How it works
Comprehensive textual documentation covering data generation, chronological temporal splits, causal feature engineering, the hybrid anomaly architecture, and forensic interpretability.

---

## 6. System Limitations

1. **Synthetic Simulation**: While the Sparkov generator mimics legitimate and fraudulent credit card usage patterns, synthetic transactions do not fully capture adversary behavior such as coordinated bot attacks, card testing sprees, or merchant collusion.
2. **Batch Precomputation**: For instant responsiveness on 555,719 test transactions, test scores are precomputed into Parquet. Production deployment would require an inference microservice scoring authorizations in real time.
3. **Cold Start**: Newly issued cards with zero transaction history fall back to default population baselines until sufficient transactions accumulate.

---

## 7. Five-Minute Faculty Demonstration Script

This structured script is tailored for a technical faculty evaluation or viva presentation:

### Minute 0:00 - 1:00: Problem Statement and Architecture
- "Welcome. This is AuditLens, a credit card transaction fraud review and explanation application."
- "The challenge in financial fraud detection is twofold: extreme class imbalance (less than 0.4% fraud) and the need for human-auditable explanations for risk analysts."
- "We built a hybrid system: unsupervised statistical modeling (PCA, K-Means, EM-GMM, HMM) supplies behavioral baselines to an XGBoost classifier with `scale_pos_weight = 172.82`."
- "All data is from the Kaggle Sparkov synthetic fraud dataset, strictly split chronologically to prevent temporal lookahead bias."

### Minute 1:00 - 2:00: Tab 1 (Overview) & Test Metrics
- Navigate to the **Overview** tab.
- "On the holdout test set of 555,719 transactions, AuditLens achieves a headline PR-AUC of **0.9176** and an ROC-AUC of **0.9981**."
- "At our validation-tuned threshold of 0.9500, the system achieves **85.42% precision** and **86.29% recall**, capturing 1,851 out of 2,145 fraud attempts while generating only 316 false positives out of 553,574 normal transactions."
- "Notice the risk score distribution: normal activity clusters near zero, with a distinct separation for flagged transactions."

### Minute 2:00 - 3:15: Tab 2 (Transactions) & Forensic Investigation
- Navigate to the **Transactions** tab.
- Filter by `verdict: Fraud`. Select a flagged transaction from the list.
- "When an investigator selects a transaction, AuditLens does not simply output a probability score. It provides 3 to 5 plain-English justifications."
- Point out the reasons: e.g., 'Amount is 8.4 times this customer's average spend', 'Transaction initiated at 01:00 outside normal daytime activity', 'Distance is significantly farther than typical travel radius'.
- Review the customer comparison table, recent transactions, customer K-Means segment, and the HMM Viterbi sequential path.
- Click **Download audit report (HTML)** to demonstrate the clean, printable black-and-white forensic document with amounts converted to Rupees (₹).

### Minute 3:15 - 4:15: Tab 3 (Algorithms) & Mathematical Rigor
- Navigate to the **Algorithms** tab.
- Expand **3. Gaussian Mixture Model with Hand-Written EM**.
- Click **Run EM optimization on PCA sample**. Point out the real-time monotonic log-likelihood curve computed from our hand-written NumPy implementation.
- Expand **4. Hidden Markov Model**.
- Click **Run Forward and Viterbi Verification against hmmlearn**. Show the real output confirming "Matches hmmlearn: yes" for both Forward and Viterbi.
- Expand **7. Ensemble Selection and Ablation Study**. Show that adding unsupervised features delivered a **+0.0991 PR-AUC** gain over standard XGBoost.

### Minute 4:15 - 5:00: Tab 4 (How It Works), Code Quality, and Q&A
- Navigate to the **How it works** tab.
- Summarize the strict design discipline: monochrome academic styling (black, white, grays only), zero decorative emojis or hype terms, and robust testing.
- "We have 16 automated tests in pytest covering leakage prevention, mathematical equivalence, and full headless UI testing via Streamlit AppTest."
- Conclude and invite questions.
