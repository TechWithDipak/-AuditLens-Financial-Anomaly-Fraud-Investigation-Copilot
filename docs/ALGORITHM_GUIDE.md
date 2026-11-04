# AuditLens — Algorithm Guide

**Document:** Mathematical Formulations, Implementation Strategy & Safety Boundaries  
**Version:** 1.0.0  
**Date:** 2026-11-04  
**Scope:** Units 3, 4, and 5 Syllabus Machine Learning Algorithms  

---

# Unit 3: Mixture Models and Expectation-Maximization (EM)

## 1. K-Means Clustering (`services/ml/kmeans.py`)

### 1.1 Mathematical Formulation
K-Means partitions an unlabeled dataset $X = \{x_1, \dots, x_N\} \subset \mathbb{R}^D$ into $K$ disjoint spherical clusters $S = \{S_1, \dots, S_K\}$ by minimizing the within-cluster sum of squares (inertia):

$$\mathcal{J}(S, \mu) = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2$$

where $\mu_k = \frac{1}{|S_k|} \sum_{x \in S_k} x$ denotes the centroid of cluster $S_k$.

### 1.2 Iterative Lloyd's Procedure
1. **Initialization:** Centroids $\mu_1^{(0)}, \dots, \mu_K^{(0)}$ initialized using the `k-means++` seeding algorithm to ensure dispersed initial centers.
2. **Assignment Step:** Assign each observation $x_i$ to its nearest centroid:
   $$c_i^{(t)} = \arg\min_{k \in \{1,\dots,K\}} \|x_i - \mu_k^{(t)}\|^2$$
3. **Update Step:** Recalculate centroids as the arithmetic mean of assigned points:
   $$\mu_k^{(t+1)} = \frac{\sum_{i: c_i = k} x_i}{\sum_{i: c_i = k} 1}$$
4. **Convergence:** Terminate when centroids shift below tolerance $\epsilon$ or maximum iterations are reached.

### 1.3 Validation & Diagnostics
- **Elbow Analysis:** Plotting inertia $\mathcal{J}(K)$ as a function of $K$. The optimal number of clusters occurs at the "elbow" point where marginal reductions in within-cluster variance diminish.
- **Silhouette Coefficient:** For each sample $i$, let $a(i)$ be the mean intra-cluster distance and $b(i)$ be the minimum mean distance to any other cluster:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
  A higher average silhouette score indicates well-separated, dense clusters.

### 1.4 Decision Support Boundary
> **Critical Interpretation:** Cluster membership is strictly a peer-group behavioral segmentation. A cluster containing an unusual ratio of transactions is **not** a "fraud cluster" and must never be used as automated evidence of fraud.

---

## 2. Gaussian Mixture Models (GMM) via Expectation-Maximization (`services/ml/gmm_em.py`)

### 2.1 Probabilistic Formulation
GMM models the probability density of feature vectors as a convex combination of $K$ multivariate Gaussian distributions:

$$p(x \mid \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

subject to $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$. Here, $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ represents the multivariate Gaussian density:

$$\mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

### 2.2 The Expectation-Maximization (EM) Estimation Procedure
Because latent component assignments $z_i \in \{1, \dots, K\}$ are unobserved, direct maximum likelihood estimation lacks a closed-form solution. The EM algorithm iteratively optimizes the expected complete-data log-likelihood:

1. **E-Step (Expectation):** Calculate the posterior probability (responsibility) that component $k$ generated observation $x_i$:
   $$\gamma_{ik} = P(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}$$
   Note that $\sum_{k=1}^K \gamma_{ik} = 1.0$ for each observation $x_i$.

2. **M-Step (Maximization):** Update parameters using the computed responsibilities:
   $$N_k = \sum_{i=1}^N \gamma_{ik}$$
   $$\pi_k^{(t+1)} = \frac{N_k}{N}$$
   $$\mu_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$
   $$\Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T$$

### 2.3 Per-Sample Log-Likelihood & Unusualness Ranking
AuditLens computes the per-sample log-likelihood under the fitted mixture distribution:

$$\ell(x_i) = \log p(x_i \mid \hat{\theta}) = \log \sum_{k=1}^K \hat{\pi}_k \mathcal{N}(x_i \mid \hat{\mu}_k, \hat{\Sigma}_k)$$

Observations situated in the low-density tail of the distribution fall below a configurable percentile threshold $\tau_\alpha$ (e.g. $\alpha = 1.0\%$):

$$\text{Unusual}(x_i) \iff \ell(x_i) \le \tau_\alpha$$

### 2.4 Decision Support Boundary
> **Critical Interpretation:** Low log-likelihood denotes statistical rarity relative to the Gaussian components. Genuine legitimate transactions (such as rare overseas travel or holiday purchases) frequently exhibit low density. An unusualness ranking is an alert for human review, not proof of illicit intent.

---

## 3. Principal Component Analysis (PCA) (`services/ml/pca.py`)

### 3.1 Mathematical Formulation
PCA is an orthogonal linear transformation that projects continuous feature vectors $x \in \mathbb{R}^D$ into a lower-dimensional subspace $z \in \mathbb{R}^d$ ($d \ll D$) maximizing retained variance.

Given centered data matrix $\tilde{X} = X - \mu_X$, the empirical sample covariance matrix is:

$$\Sigma = \frac{1}{N-1} \tilde{X}^T \tilde{X}$$

Eigendecomposition yields orthogonal eigenvectors $v_1, \dots, v_D$ and corresponding eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$:

$$\Sigma v_j = \lambda_j v_j$$

### 3.2 Explained Variance
The proportion of total sample variance accounted for by the $j$-th principal component is:

$$\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^D \lambda_m}$$

The cumulative explained variance over the top $d$ components represents the fidelity of the reduced representation:

$$\text{Cumulative EVR}_d = \sum_{j=1}^d \text{EVR}_j$$

### 3.3 Projection & Visualization
The $d$-dimensional coordinate projection is obtained by matrix multiplication:

$$Z = \tilde{X} V_d$$

where $V_d = [v_1, v_2, \dots, v_d] \in \mathbb{R}^{D \times d}$. In AuditLens, 2D projections ($d=2$) provide financial analysts with intuitive geometric maps of customer transaction distributions and behavioral clusters.

---

## 4. Summary of Output Semantics

| Algorithm | Metric / Output | Valid Range | Interpretation |
|---|---|---|---|
| **K-Means** | `cluster_id` | $\{0, \dots, K-1\}$ | Behavioral peer group assignment |
| **K-Means** | `inertia` | $[0, \infty)$ | Within-cluster variance sum |
| **GMM via EM** | `log_likelihood` | $(-\infty, 0.0]$ | Log-density under fitted Gaussian mixture |
| **GMM via EM** | `responsibilities` | $[0.0, 1.0]^K$ | Posterior component assignment probabilities ($\sum = 1$) |
| **GMM via EM** | `unusual_indices` | Index list | Transactions falling into bottom $\alpha\%$ density tail |
| **PCA** | `projection_preview` | $\mathbb{R}^2$ | Low-dimensional orthogonal scatter coordinates |
| **PCA** | `explained_variance_ratio` | $[0.0, 1.0]^d$ | Variance contribution of each principal eigenvector |



---

# Unit 4: Hidden Markov Models (HMM)

## 1. Hidden Markov Model Framework (`services/ml/hmm.py`)

### 1.1 Model Definition
A first-order Hidden Markov Model is a generative temporal probabilistic model denoted by the parameter triplet $\lambda = (A, B, \pi)$:

1. **State Space:** Discrete unobserved behavioral states $S = \{s_1, \dots, s_N\}$.
   - In AuditLens: $N=3$ states representing financial behavior:
     - $s_0$: *Normal Day-to-Day Activity*
     - $s_1$: *Elevated Activity / Travel*
     - $s_2$: *High-Risk Burst / Account Takeover*
2. **Observation Space:** Observation sequence $O = (o_1, o_2, \dots, o_T)$ over discrete emission alphabet $V = \{v_1, \dots, v_M\}$ derived from transaction attributes (amount tiers, transaction types: `PAYMENT`, `TRANSFER`, `CASH_OUT`).
3. **Transition Probability Matrix:** $A = [a_{ij}] \in \mathbb{R}^{N \times N}$, where:
   $$a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i), \quad \sum_{j=1}^N a_{ij} = 1$$
4. **Emission Probability Matrix:** $B = [b_j(k)] \in \mathbb{R}^{N \times M}$, where:
   $$b_j(k) = P(o_t = v_k \mid q_t = s_j), \quad \sum_{k=1}^M b_j(k) = 1$$
5. **Initial State Distribution:** $\pi = [\pi_i] \in \mathbb{R}^N$, where:
   $$\pi_i = P(q_1 = s_i), \quad \sum_{i=1}^N \pi_i = 1$$

---

## 2. Forward Algorithm with Numerical Scaling (`services/ml/hmm_forward.py`)

### 2.1 The Evaluation Problem
Given model $\lambda$ and observation sequence $O = (o_1, \dots, o_T)$, compute the sequence likelihood $P(O \mid \lambda)$. The naive marginalization over all $N^T$ hidden state paths has exponential complexity $\mathcal{O}(N^T \cdot T)$. The Forward dynamic programming procedure computes this in polynomial time $\mathcal{O}(N^2 T)$.

### 2.2 Forward Variable & Underflow Failure Mode
The unscaled forward variable is:
$$\alpha_t(i) = P(o_1, \dots, o_t, q_t = s_i \mid \lambda)$$

Because $\alpha_t(i)$ multiplies $T$ conditional probabilities strictly bounded in $[0, 1]$, on sequences of length $T > 100$, floating-point values rapidly drop below the machine epsilon ($\approx 10^{-308}$ in 64-bit IEEE 754), resulting in catastrophic numerical underflow to zero.

### 2.3 Numerical Underflow Protection via Scaling
AuditLens implements normalized scale coefficients $c_t$ at every time step $t$:

1. **Initialization ($t = 1$):**
   $$\alpha_1(i) = \pi_i b_i(o_1)$$
   $$c_1 = \frac{1}{\sum_{i=1}^N \alpha_1(i)}$$
   $$\hat{\alpha}_1(i) = c_1 \alpha_1(i)$$

2. **Induction ($t = 2, \dots, T$):**
   $$\alpha_t(j) = \left[ \sum_{i=1}^N \hat{\alpha}_{t-1}(i) a_{ij} \right] b_j(o_t)$$
   $$c_t = \frac{1}{\sum_{j=1}^N \alpha_t(j)}$$
   $$\hat{\alpha}_t(j) = c_t \alpha_t(j)$$

3. **Log-Likelihood Calculation:**
   Using the identity $\prod_{t=1}^T c_t = \frac{1}{P(O \mid \lambda)}$, the exact log-likelihood is recovered in log-space without underflow:
   $$\log P(O \mid \lambda) = - \sum_{t=1}^T \log(c_t)$$

---

## 3. Viterbi Algorithm in Log-Space (`services/ml/hmm_viterbi.py`)

### 3.1 The Decoding Problem
Find the single optimal sequence of latent states $Q^* = (q_1^*, \dots, q_T^*)$ maximizing the joint probability $P(Q, O \mid \lambda)$.

### 3.2 Dynamic Programming Formulation in Log-Space
To eliminate underflow and maintain numerical precision, calculations are performed entirely in log-space:

1. **Initialization ($t = 1$):**
   $$V_1(i) = \log \pi_i + \log b_i(o_1), \quad 1 \le i \le N$$
   $$\psi_1(i) = 0$$

2. **Recursion ($t = 2, \dots, T$):**
   For each state $j \in \{1, \dots, N\}$:
   $$V_t(j) = \max_{1 \le i \le N} \left[ V_{t-1}(i) + \log a_{ij} \right] + \log b_j(o_t)$$
   $$\psi_t(j) = \arg\max_{1 \le i \le N} \left[ V_{t-1}(i) + \log a_{ij} \right]$$

3. **Termination:**
   $$P^* = \max_{1 \le i \le N} V_T(i)$$
   $$q_T^* = \arg\max_{1 \le i \le N} V_T(i)$$

4. **State Path Backtracking:**
   For $t = T-1, T-2, \dots, 1$:
   $$q_t^* = \psi_{t+1}(q_{t+1}^*)$$

---

## 4. Hand-Calculated Educational Verification Benchmark

To verify algorithmic correctness against manual calculation:
- $\pi = [0.6, 0.4]$
- $A = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}$, $B = \begin{bmatrix} 0.8 & 0.2 \\ 0.1 & 0.9 \end{bmatrix}$
- Observation sequence: $O = [0, 1, 0]$ ($T=3$)

### Manual Derivations:
- **Forward Variables:**
  - $t=1$: $\alpha_1 = [0.48, 0.04]$, $P(o_1) = 0.52$
  - $t=2$: $\alpha_2 = [0.0704, 0.1512]$, $P(o_1, o_2) = 0.2216$
  - $t=3$: $\alpha_3 = [0.087808, 0.011184]$
  - **Total Likelihood:** $P(O) = 0.087808 + 0.011184 = 0.098992$
  - **Log-Likelihood:** $\ln(0.098992) = -2.312716$
- **Viterbi Trellis:**
  - $t=1$: $V_1 = [0.48, 0.04]$
  - $t=2$: $V_2 = [0.0672, 0.1296]$, backpointer $\psi_2 = [0, 0]$
  - $t=3$: $V_3 = [0.041472, 0.007776]$, backpointer $\psi_3 = [1, 1]$
  - **Optimal Path:** $[0, 1, 0]$ with path probability $P^* = 0.041472 \implies \ln(P^*) = -3.182705$

Unit test [`backend/tests/unit/test_hmm.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/tests/unit/test_hmm.py) verifies these hand-derived values within $10^{-5}$ tolerance.

---

## 5. Decision Support Boundaries for Sequential Modeling

| Output Field | Mathematical Meaning | What it IS | What it IS NOT |
|---|---|---|---|
| `sequence_log_likelihood` | $\log P(O \mid \lambda)$ under HMM | Measure of sequence conformity to past behavioral transitions | Not a fraud probability score |
| `inferred_state` | $q_t^* = \arg\max P(Q \mid O, \lambda)$ | Decoded behavioral profile (Normal / Elevated / Burst) | Not a deterministic fraud accusation |


---

# Unit 5: Combining Models

## 1. Classification and Regression Trees (CART) (`services/ml/cart.py`)

### 1.1 Mathematical Formulation
CART builds binary decision trees by recursively searching for the feature $j$ and split threshold $\theta$ that maximizes the reduction in Gini impurity $\Delta I_G(s, D)$:

$$\Delta I_G(s, D) = I_G(D) - \frac{|D_L|}{|D|} I_G(D_L) - \frac{|D_R|}{|D|} I_G(D_R)$$

where the Gini impurity of dataset $D$ over classes $k \in \{0, 1\}$ is:

$$I_G(D) = 1 - \sum_{k=0}^1 p_k^2$$

### 1.2 Class Imbalance Strategy
In imbalanced fraud detection ($p_1 \ll p_0$), unweighted splits overwhelmingly favor the majority legitimate class. AuditLens uses balanced class weights:

$$w_k = \frac{N}{2 \cdot N_k}$$

This penalizes misclassifications on rare fraud instances inversely proportional to their empirical frequency.

---

## 2. Boosting — XGBoost (`services/ml/xgboost_model.py`)

### 2.1 Gradient Tree Boosting Formulation
XGBoost constructs an ensemble of $M$ regression trees sequentially:

$$\hat{y}_i = \sum_{m=1}^M f_m(x_i), \quad f_m \in \mathcal{F}$$

At iteration $t$, the objective function minimizes the second-order Taylor approximation of the loss plus tree regularization:

$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$.

### 2.2 Imbalance Compensation via `scale_pos_weight`
To prevent the boosting process from ignoring the positive fraud class, positive sample gradient contributions are scaled by:

$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$

---

## 3. Ensemble Learning: Voting and Stacking (`services/ml/ensemble.py`)

### 3.1 Conceptual Distinctions
- **CART:** A single interpretable decision tree model.
- **XGBoost:** A boosting ensemble that sequentially trains trees on the residual gradients of preceding trees.
- **Meta-Ensemble:** A model aggregation framework combining multiple distinct learning paradigms (e.g. CART + XGBoost) to reduce generalization error.

### 3.2 Combination Strategies
1. **Soft Voting:** Averages posterior predicted class probabilities:
   $$P_{\text{ens}}(Y=1 \mid x) = \frac{1}{M} \sum_{m=1}^M P_m(Y=1 \mid x)$$
2. **Stacking with Out-of-Fold Predictions:**
   Base learners generate out-of-fold cross-validated predictions on the training data. A meta-learner (Logistic Regression) is fitted on these predictions, eliminating the risk of data leakage.

### 3.3 Fair Benchmarking Protocol
In AuditLens, all models (CART, XGBoost, Ensembles) are evaluated on the **identical held-out test split** using the exact same metrics:
- Precision-Recall AUC (Average Precision) as the primary metric
- F1-Score, Precision, Recall, ROC-AUC
- Confusion matrix and false positive/negative rates


---

# Integrated Evaluation, Explainability & Supporting Algorithms

## 1. Model Explainability via SHAP (`services/ml/explainability.py`)

### 1.1 Cooperative Game Theory Foundations
SHAP assigns each feature an attribution value representing its contribution to the model's prediction for a specific instance. Grounded in cooperative game theory, Shapley values uniquely satisfy four fundamental axioms:
1. **Efficiency:** The sum of all feature attributions equals the difference between the model output and the expected base value:
   $$\sum_{j=1}^D \phi_j(x) = f(x) - \mathbb{E}[f(X)]$$
2. **Symmetry:** If two features contribute identically across all possible subsets, their attributions are equal.
3. **Dummy / Null Player:** Features that do not change the model output in any subset receive zero attribution ($\phi_j = 0$).
4. **Additivity:** When combining models (ensembles), the joint attribution equals the sum of the individual model attributions.

### 1.2 TreeSHAP Algorithm
For tree ensembles (XGBoost and CART), AuditLens implements `shap.TreeExplainer`, which computes exact Shapley values in polynomial time $\mathcal{O}(T L D^2)$ (where $T$ is the number of trees, $L$ is max leaves, and $D$ is tree depth), rather than exponential time over feature subsets.

### 1.3 Decision Support Boundary
> **Essential Regulatory Principle:** SHAP attributions reflect the internal mathematical mechanics of the machine learning model. A positive SHAP value indicates that a feature increased the model's score; it is **not legal proof of causation** or evidence that the transaction was fraudulent.

---

## 2. Decision Threshold Tuning (`services/ml/thresholds.py`)

### 2.1 Operational Context
In extreme class imbalance, the standard threshold $\theta = 0.5$ is frequently suboptimal. Lowering the decision threshold captures more fraudulent transactions (increasing Recall) at the expense of generating more false alarms (decreasing Precision).

### 2.2 AuditLens Threshold Protocol
AuditLens computes an operational trade-off curve across $\theta \in [0.05, 0.90]$:
- **Optimal $F_1$ Threshold:** Maximizes harmonic balance between Precision and Recall.
- **High-Recall Operating Point:** Identifies the highest decision threshold that maintains $\text{Recall} \ge 90\%$, enabling financial compliance teams to minimize unflagged fraud leaks while managing investigation capacity.

---

## 3. Supporting Benchmark: Isolation Forest (`services/ml/isolation_forest.py`)

### 3.1 Mathematical Principle
Isolation Forest isolates anomalies by randomly selecting a feature and randomly selecting a split value between the feature's minimum and maximum. Because anomalies require fewer recursive splits to isolate, they exhibit shorter path lengths in an ensemble of isolation trees (iTrees).

The anomaly score for instance $x$ across an ensemble of $n$ trees is:

$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$

where $h(x)$ is the path length and $c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree.

### 3.2 Score Taxonomy Enforcement
| Score Type | Algorithm | Range | Semantic Role |
|---|---|---|---|
| `fraud_probability` | Supervised (CART / XGBoost / Ensemble) | $[0.0, 1.0]$ | Estimated probability that $Y = 1$ given labels |
| `isolation_forest_score` | Unsupervised Isolation Forest | $[-1.0, 1.0]$ | Tree depth isolation rarity metric |
| `gmm_log_likelihood` | Unsupervised GMM via EM | $(-\infty, 0.0]$ | Log-density under fitted Gaussian components |
