# AuditLens — Financial Anomaly & Fraud Investigation Copilot

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![SQLite WAL](https://img.shields.io/badge/SQLite-WAL_Mode-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-105%20Passed%20(100%25)-success.svg)](backend/tests/)
[![Offline Mode](https://img.shields.io/badge/Offline-100%25%20Zero%20API%20Keys-brightgreen.svg)]()

AuditLens is a locally runnable, modular financial anomaly detection and fraud investigation copilot built to assist human compliance analysts. It operationalizes **all nine mandatory Machine Learning syllabus algorithms across Units 3, 4, and 5**, combined with local SHAP tree explainability, a deterministic regulatory RAG knowledge base, and an AI investigation assistant running 100% offline without requiring external API keys.

> **Decision Support Boundary:** AuditLens is strictly an advisory decision-support system. It never makes autonomous accusations, account freezes, or regulatory SAR filings. All model outputs strictly preserve their formal mathematical semantics to empower human adjudication.

---

## 🏛️ System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    AUDITLENS DASHBOARD                                 |
|          (React 18 + TypeScript + Vite + Tailwind CSS + Lucide Icons + Recharts)       |
|  [Overview] [Data Workspace] [Algorithm Lab (Units 3,4,5)] [Model Comparison]       |
|  [Transaction Explorer & SHAP Drawer] [Case Management & Audit] [RAG] [Assistant]     |
+---------------------------------------------------------------------------------------+
                                           |  (REST JSON over HTTP / 127.0.0.1:8000)
                                           v
+---------------------------------------------------------------------------------------+
|                                    FASTAPI BACKEND                                     |
|  +---------------------------------------------------------------------------------+  |
|  | Routers: /datasets, /preprocessing, /algorithms, /evaluation, /cases, /rag,     |  |
|  |          /assistant, /health                                                     |  |
|  +---------------------------------------------------------------------------------+  |
|  | Security & Governance:                                                          |  |
|  | - RBAC: VIEWER (1) < ANALYST (2) < LEAD_INVESTIGATOR (3) < ADMIN (4)            |  |
|  | - Upload limits: Streaming MAX_UPLOAD_BYTES = 100MB (HTTP 413)                  |  |
|  | - Filename sanitization & path traversal prevention                              |  |
|  | - Prompt injection defense ([BLOCKED_INJECTION_PAYLOAD])                        |  |
|  | - Zero PII/PAN logging hygiene                                                  |  |
|  +---------------------------------------------------------------------------------+  |
|  | Algorithmic Execution Engines (Local CPU & Memory):                              |  |
|  | - Unit 3: K-Means (Lloyd's), GMM via EM (Multivariate), PCA (Eigen)             |  |
|  | - Unit 4: HMM (3-State), Scaled Forward (ct), Log-Space Viterbi (Log-Treillis) |  |
|  | - Unit 5: CART (Balanced Gini), XGBoost (PR-AUC Tuned), Soft Voting & Stacking  |  |
|  | - Explainability: TreeSHAP (shap.TreeExplainer), 12-point Threshold Curve, IF   |  |
|  | - RAG Engine: Deterministic TF-IDF Cosine Similarity with weak-cutoff threshold |  |
|  | - Assistant: Deterministic Rule-Based Synthesis & Evidence Claim Verifier       |  |
+---------------------------------------------------------------------------------------+
                     |                                              |
                     v                                              v
+------------------------------------+             +------------------------------------+
|       LOCAL SQLITE WAL DATABASE     |             |       PERSISTED ARTIFACTS / DATA   |
| - Cases, Notes, Linked Transactions|             | - data/fixtures/ (CSV datasets)    |
| - Immutable Audit Events Ledger    |             | - data/reference_docs/ (AML Regs)  |
| - Alembic Schema Migrations        |             | - artifacts/ (Models & RAG Index)  |
+------------------------------------+             +------------------------------------+
```

---

## 📚 Mandatory Syllabus Algorithm Implementation Matrix

| Unit | Syllabus Algorithm | Implementation File | Mathematical Formulation & Role | Operational Semantics |
| :---: | :--- | :--- | :--- | :--- |
| **Unit 3** | **K-Means Clustering** | [`kmeans.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/kmeans.py) | $\min \sum \|x - \mu_k\|^2$ via Lloyd's algorithm with `k-means++` seeding; computes elbow inertia curve & silhouette scores. | Behavioral peer group segmentation (not fraud status). |
| **Unit 3** | **GMM via EM** | [`gmm_em.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/gmm_em.py) | $p(x) = \sum \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ via E-Step (responsibilities) & M-Step parameter re-estimation. | Per-sample log-likelihood density; unusualness tail ranking. |
| **Unit 3** | **PCA** | [`pca.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/pca.py) | Centered covariance eigendecomposition $\Sigma v = \lambda v$; calculates Explained Variance Ratio (EVR). | Orthogonal 2D projection for geometric visual inspection. |
| **Unit 4** | **HMM Framework** | [`hmm.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/hmm.py) | Parameter triplet $\lambda = (A, B, \pi)$ with 3 behavioral states (Normal, Elevated, Burst) over discretized emission alphabet. | Markov temporal state transition modeling. |
| **Unit 4** | **Forward Algorithm** | [`hmm_forward.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/hmm_forward.py) | Pure NumPy dynamic programming with scaling factors $c_t$; $\log P(O \mid \lambda) = -\sum \log c_t$. Underflow-proof up to $T=2,000$. | Sequence likelihood and temporal conformity evaluation. |
| **Unit 4** | **Viterbi Algorithm** | [`hmm_viterbi.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/hmm_viterbi.py) | Dynamic programming trellis in log-space: $V_t(j) = \max_i [V_{t-1}(i) + \log a_{ij}] + \log b_j(o_t)$. Verified against analytical toy model. | Optimal hidden behavioral state path decoding. |
| **Unit 5** | **CART** | [`cart.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/cart.py) | Binary recursive partitioning with balanced class weights $w_k = \frac{N}{2 N_k}$; extracts human-readable decision rules. | Interpretable decision logic trees for compliance. |
| **Unit 5** | **XGBoost** | [`xgboost_model.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/xgboost_model.py) | Regularized gradient boosted trees with `scale_pos_weight` and `eval_metric='aucpr'`. | Primary supervised fraud probability estimation. |
| **Unit 5** | **Ensembles (Voting & Stacking)** | [`ensemble.py`](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/backend/app/services/ml/ensemble.py) | Soft-voting posterior averaging & out-of-fold meta-learner stacking evaluated on identical held-out test split. | Multi-paradigm aggregation minimizing variance & bias. |

---

## ⚡ Quickstart & Reproducible Demonstration

### 1. Run Automated End-to-End Walkthrough Script (100% Offline)
The fastest way to verify all 9 algorithms, explainability, RAG retrieval, and persistence is to run the automated demonstration script:
```powershell
# Windows
.\.venv\Scripts\python.exe scripts/demo_walkthrough.py
```
```bash
# macOS / Linux
python3 scripts/demo_walkthrough.py
```
This script executes the entire 10-step pipeline in ~6 seconds with zero cloud dependencies.

### 2. Launching Interactive Dashboard

#### Terminal 1: Backend API (FastAPI)
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```
- **Backend API:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **System Health Check:** `http://127.0.0.1:8000/api/v1/health`

#### Terminal 2: Frontend Dashboard (Vite + React)
```powershell
cd frontend
npm run dev
```
- **Web UI:** `http://localhost:5173`

---

## 🧪 Comprehensive Verification & QA

AuditLens maintains a 100% pass rate across 105 automated unit, integration, and security tests:

```powershell
# Run full automated test suite (105 tests)
.\.venv\Scripts\pytest.exe backend/tests -v

# Run dedicated security & failure-mode regression suite (16 tests)
.\.venv\Scripts\pytest.exe backend/tests/security -v

# Run Python code quality and formatting check
.\.venv\Scripts\ruff.exe check backend/

# Build frontend production bundle
cd frontend && npm run build
```

---

## 🛡️ Security, Privacy & Compliance Architecture

- **Zero Mandatory Cloud Dependencies:** Core ML, SHAP, RAG, and AI drafting run entirely on local offline CPU without sending data to external APIs.
- **Strict Score Taxonomy Separation:** Never conflates supervised fraud probabilities ($[0, 1]$), GMM density log-likelihoods, K-Means clusters, Isolation Forest anomaly depths, or Markov states into an uncalibrated composite.
- **Role-Based Access Control (RBAC):** Hierarchical permissions (`VIEWER`, `ANALYST`, `LEAD_INVESTIGATOR`, `ADMIN`) protecting case updates and vector re-indexing.
- **Upload Boundaries & Path Traversal Prevention:** Enforces 100MB streaming upload bounds with `HTTP 413` and strips directory components from uploaded filenames.
- **Prompt Injection Defense:** Centralized regex scanner neutralizes adversarial prompt injection payloads in user hypotheses with `[BLOCKED_INJECTION_PAYLOAD]`.
- **Zero PII/PAN Logging:** Sensitive payment instrument identifiers and raw transaction arrays are strictly masked in logs.

---

## 📖 Complete Documentation Index

- [**SETUP Guide (`SETUP.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/SETUP.md) — Comprehensive installation, database migrations, and clean-install verification checklists.
- [**Final Project Report (`docs/FINAL_REPORT.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/FINAL_REPORT.md) — Exhaustive technical report and academic defense.
- [**Algorithm Guide (`docs/ALGORITHM_GUIDE.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/ALGORITHM_GUIDE.md) — Mathematical formulas, derivations, and decision boundaries for all 9 algorithms.
- [**System Architecture (`docs/ARCHITECTURE.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/ARCHITECTURE.md) — Component interactions, state management, and data flow.
- [**Security & Privacy Blueprint (`docs/SECURITY_AND_PRIVACY.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/SECURITY_AND_PRIVACY.md) — STRIDE threat analysis, RBAC hierarchy, and retention lifecycle.
- [**Architectural Decisions (`docs/DECISIONS.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/DECISIONS.md) — ADRs covering offline-first design, score taxonomy, SQLite WAL, and deterministic RAG.
- [**Dataset Card (`docs/DATASET_CARD.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/DATASET_CARD.md) & [**Data Dictionary (`docs/DATA_DICTIONARY.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/DATA_DICTIONARY.md) — Complete feature documentation and ethical use guidelines.
- [**RAG Sources & Provenance (`docs/RAG_SOURCES.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/RAG_SOURCES.md) — Regulatory citations, licensing, and metadata for FATF, FinCEN, and EBA reference documents.
- [**API Specification (`docs/API.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/API.md) — Detailed REST contracts and schemas.
- [**Model Cards (`docs/MODEL_CARDS/`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/MODEL_CARDS/) — Standardized model cards for CART, XGBoost, and Ensembles.
- [**Live Phase Status (`docs/PHASE_STATUS.md`)**](file:///c:/Users/harit/OneDrive/Desktop/projects/ML%20Project/ML%20Project/docs/PHASE_STATUS.md) — Detailed verification status across all phases (Phases 0 through 13).

---

## ⚖️ License & Ethical Advisory

AuditLens is released under the **MIT License**.  
*Advisory:* Machine learning anomaly detection models produce statistical estimates that reflect training data patterns. They must never be used autonomously to make adverse financial decisions, terminate accounts, or file SARs without human review.
