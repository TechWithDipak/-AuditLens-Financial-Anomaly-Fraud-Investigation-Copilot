# AuditLens — Project Implementation Plan

**Project Name:** AuditLens  
**Project Type:** Financial Anomaly & Fraud Investigation Copilot  
**Version:** 1.0.0-dev  
**Date:** 2026-11-04  
**Status:** In Progress (Phase 0 Complete)

---

## 1. Executive Summary

AuditLens is a locally runnable, modular, full-stack decision-support system designed to assist human financial fraud analysts. It combines unsupervised anomaly detection, sequential state modeling, and supervised classification across nine core Machine Learning syllabus algorithms from Units 3, 4, and 5. It augments quantitative models with SHAP-based local explainability, an evidence-grounded Retrieval-Augmented Generation (RAG) knowledge base, and an interactive analyst case management workflow.

AuditLens strictly operates as a **decision-support tool**. It never makes autonomous financial or legal accusations. All scores, probabilities, and sequence inferences are explicitly labeled by their mathematical semantics and presented to the human analyst for adjudication.

---

## 2. Nine Syllabus Algorithm Mapping

| Unit | Algorithm Item | Module Implementation | Evaluation & Metrics | Role in AuditLens |
|---|---|---|---|---|
| **Unit 3** | **K-Means Clustering** | `services/ml/kmeans.py` | Inertia, Elbow Curve, Silhouette Score | Customer/transaction peer group behavioral profiling |
| **Unit 3** | **Gaussian Mixture Models (GMM) via EM** | `services/ml/gmm_em.py` | Per-sample Log-Likelihood, Component Responsibilities, BIC/AIC | Unsupervised density-based anomaly ranking |
| **Unit 3** | **Principal Component Analysis (PCA)** | `services/ml/pca.py` | Explained Variance Ratio, Cumulative Variance, Reconstruction Error | Dimensionality reduction and 2D/3D behavioral projection |
| **Unit 4** | **Hidden Markov Model (HMM)** | `services/ml/hmm.py` | Transition matrix convergence, State persistence | Latent behavioral state modeling over account transaction history |
| **Unit 4** | **Forward Algorithm** | `services/ml/hmm_forward.py` | Sequence Log-Likelihood under fitted HMM | Sequence anomaly scoring; numerical scaling / log-space |
| **Unit 4** | **Viterbi Algorithm** | `services/ml/hmm_viterbi.py` | Optimal state path, Path log-probability | Inferring most likely chronological behavioral state path |
| **Unit 5** | **CART (Classification Tree)** | `services/ml/cart.py` | Precision, Recall, F1, PR-AUC, Decision Rules | Transparent, interpretable baseline supervised fraud tree |
| **Unit 5** | **Boosting — XGBoost** | `services/ml/xgboost_model.py` | Precision, Recall, F1, PR-AUC, ROC-AUC, SHAP values | High-performance gradient boosted classifier for fraud detection |
| **Unit 5** | **Ensemble Learning** | `services/ml/ensemble.py` | Held-out test set comparison, Out-of-fold calibration | Combining diverse models (Voting / Stacking) with fair benchmarking |

*Optional Supporting Algorithm:* **Isolation Forest** (`services/ml/isolation_forest.py`) will be implemented as an unsupervised tabular anomaly benchmark.

---

## 3. Phased Implementation Roadmap

AuditLens strictly adheres to the 14-phase implementation roadmap defined in `AUDITLENS_FULL_BUILD_PROMPT.md`. Each phase produces verified deliverables and halts for explicit user approval before the next phase commences.

```
+-----------------------------------------------------------------------------------+
| Phase 0: Workspace Inspection & Architecture Blueprint (Current)                  |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 1: Repository Foundation (FastAPI skeleton, Vite shell, setup scripts)      |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 2: Dataset Acquisition, Validation & Synthetic Test Fixtures                |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3: Exploratory Data Analysis & Leakage-Safe Preprocessing Pipeline          |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 4: Unit 3 Algorithms (K-Means, GMM via EM, PCA)                             |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 5: Unit 4 Sequential Algorithms (HMM, Forward Algorithm, Viterbi Algorithm) |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 6: Unit 5 Supervised & Ensemble Algorithms (CART, XGBoost, Ensemble)        |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 7: Integrated ML Evaluation, Threshold Tuning & SHAP Explainability         |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 8: Backend REST APIs, Persistence & Case Management Service                 |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 9: Frontend Financial Analyst Dashboard & Visualization Suite               |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 10: RAG Reference Knowledge Base & Vector Index                             |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 11: Optional LLM Investigation Assistant (Structured Grounded Drafts)       |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 12: Security Audit, Hardening & Complete Regression Testing                 |
+-----------------------------------------------------------------------------------+
                                         | [Approval]
                                         v
+-----------------------------------------------------------------------------------+
| Phase 13: Final Documentation, Verification & Demonstration Package               |
+-----------------------------------------------------------------------------------+
```

---

## 4. Phase-by-Phase Execution Details

### Phase 0: Workspace Inspection & Architecture Blueprint (Current)
- **Objective:** Verify workspace, validate runtime environment, establish architectural blueprints, document dataset strategy, address sequence data requirements, and define directory structure.
- **Deliverables:** `docs/ARCHITECTURE.md`, `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/PHASE_STATUS.md`, `SETUP.md`.
- **Exit Gate:** Architectural blueprints complete; zero baseline test failures (no pre-existing code); explicit user approval.

### Phase 1: Repository Foundation
- **Objective:** Establish the backend FastAPI project and frontend Vite React TypeScript project.
- **Deliverables:** Directory scaffolding, `backend/requirements.txt`, `frontend/package.json`, `.gitignore`, `.env.example`, FastAPI health check endpoint (`/api/v1/health`), basic frontend connectivity test, lint/type-check scripts.
- **Exit Gate:** Clean install succeeds; backend and frontend dev servers start; health check passes; unit tests execute.

### Phase 2: Dataset Acquisition & Validation
- **Objective:** Ingest primary dataset (Kaggle Credit Card Fraud Detection) and generate synthetic sequence fixtures for sequential behavioral modeling.
- **Deliverables:** CSV loader, streaming schema validator, missingness/range check, dataset card (`docs/DATASET_CARD.md`), data dictionary (`docs/DATA_DICTIONARY.md`), synthetic test fixtures in `data/fixtures/`.
- **Exit Gate:** Schema validation passes; dataset card documented; raw data excluded from Git.

### Phase 3: EDA & Leakage-Safe Preprocessing
- **Objective:** Compute data distributions, imbalance ratios, and chronological characteristics; implement fit-on-train-only scalers.
- **Deliverables:** Preprocessing pipeline (`StandardScaler`/`RobustScaler`), train/val/test splitters (time-based or stratified), zero-leakage test suite.
- **Exit Gate:** Preprocessing fitted strictly on train sets; split reproducibility verified by random seed; zero leakage.

### Phase 4: Unit 3 Algorithm Modules (K-Means, GMM/EM, PCA)
- **Objective:** Implement clustering, density anomaly detection, and dimensionality reduction.
- **Deliverables:** `services/ml/kmeans.py`, `services/ml/gmm_em.py`, `services/ml/pca.py`, algorithmic tests, visual summary helpers, `docs/ALGORITHM_GUIDE.md` (Unit 3 sections).
- **Exit Gate:** Unit tests pass with deterministic seeds; mathematically valid output shapes; cluster labels separated from fraud predictions.

### Phase 5: Unit 4 Algorithm Modules (HMM, Forward, Viterbi)
- **Objective:** Implement sequential behavioral modeling on account transaction sequences.
- **Deliverables:** `services/ml/hmm.py`, `services/ml/hmm_forward.py`, `services/ml/hmm_viterbi.py`, hand-calculated educational verification tests, sequence timeline visualizer.
- **Exit Gate:** Forward and Viterbi algorithms match hand-calculated analytical matrices; log-space scaling prevents underflow; sequence dataset limitations clearly documented.

### Phase 6: Unit 5 Algorithm Modules (CART, XGBoost, Ensemble)
- **Objective:** Supervised fraud classification with baseline tree, boosted trees, and ensemble combinations.
- **Deliverables:** `services/ml/cart.py`, `services/ml/xgboost_model.py`, `services/ml/ensemble.py`, model persistence (serialization/deserialization), model cards in `docs/MODEL_CARDS/`.
- **Exit Gate:** Models train and serialize reproducibly; evaluated on identical held-out test splits; PR-AUC and F1 reported.

### Phase 7: Integrated ML Evaluation & Explainability
- **Objective:** Integrate all model outputs into standardized schemas and generate SHAP explainability.
- **Deliverables:** Model scoring service with strictly segregated score semantics (probability vs. anomaly score vs. log-likelihood); SHAP TreeExplainer integration; fallback feature importances.
- **Exit Gate:** Model scores adhere to typed contracts; SHAP explanations link to exact model versions and features; explanations pass unit tests.

### Phase 8: Backend APIs & Persistence
- **Objective:** Implement FastAPI REST endpoints, SQLAlchemy ORM models, Alembic migrations, and case management.
- **Deliverables:** SQLite database (`auditlens.db`), Alembic migrations, endpoints for datasets, algorithms, predictions, cases, and audit logs.
- **Exit Gate:** API endpoints pass integration test suite; migrations apply and rollback cleanly; audit events record analyst actions.

### Phase 9: Frontend Dashboard
- **Objective:** Build an analyst dashboard in React, Tailwind CSS, and Recharts.
- **Deliverables:** Dark-themed financial UI: Overview, Data Workspace, Algorithm Lab, Model Comparison, Transaction Explorer, Detail & Explainability, and Case Management.
- **Exit Gate:** Frontend communicates with backend APIs; displays live model outputs and charts; handles loading, empty, and error states gracefully.

### Phase 10: RAG Regulatory & Reference Knowledge Base
- **Objective:** Ingest authoritative regulatory guidelines (FATF, FinCEN, Wolfsberg) into a local vector retrieval index.
- **Deliverables:** Document ingestion pipeline, text chunking, local vector store adapter (Chroma/SQLite FTS5), similarity search with source citations, prompt-injection defense tests.
- **Exit Gate:** Search returns cited excerpts with document and section references; returns "no relevant source found" on low confidence; passes adversarial prompt injection tests.

### Phase 11: Optional LLM Investigation Assistant
- **Objective:** Generate structured, grounded investigation report drafts using pluggable LLM adapters.
- **Deliverables:** Pluggable LLM interface (Mock/Offline adapter for zero-key usage, OpenAI/Gemini adapter when configured), structured JSON output validation, hallucination guardrails.
- **Exit Gate:** Works 100% in offline mode; drafts are explicitly marked as "AI-generated draft requiring human analyst adjudication"; API errors fail gracefully.

### Phase 12: Security Audit & Full Regression Suite
- **Objective:** End-to-end security review, dependency scanning, input sanitization, and full regression testing.
- **Deliverables:** Security assessment (`docs/SECURITY_AND_PRIVACY.md`), test coverage report, upload bounding, sanitization tests.
- **Exit Gate:** Zero high-severity vulnerabilities; all test suites pass; no sensitive financial data logged.

### Phase 13: Final Documentation & Demonstration Package
- **Objective:** Complete all project documentation, finalize verification scripts, and prepare demonstration walkthrough.
- **Deliverables:** Final `README.md`, `SETUP.md`, `docs/FINAL_REPORT.md`, end-to-end demo script.
- **Exit Gate:** Complete independent clean setup verification; all 9 syllabus algorithms verified on real executions.

---

## 5. Risk Assessment and Mitigation

| Risk | Impact | Probability | Mitigation Strategy |
|---|---|---|---|
| **Python 3.13 Wheel Availability on Windows** | High | Low | Core scientific libraries (NumPy 2.1+, Pandas 2.2+, Scikit-learn 1.5+, XGBoost 2.1+) have pre-built binary wheels for Python 3.13 on `win_amd64`. HMM algorithms will feature a standalone, high-performance, pure NumPy implementation to eliminate any C-extension compilation dependencies. |
| **Tabular Fraud Data Lacks Account Sequences** | High | High | Acknowledged upfront. Kaggle Credit Card Fraud does not contain customer IDs. We use Kaggle for Units 3 & 5, and pair it with a transparently labeled, synthetic sequential account dataset for Unit 4 HMM/Forward/Viterbi. |
| **Severe Class Imbalance (e.g. 0.17% Fraud)** | High | High | Avoid misleading accuracy metrics. Use PR-AUC (Average Precision), F1-Score, cost-sensitive weighting (`scale_pos_weight` in XGBoost), and stratified splits. |
| **Data Leakage in Preprocessing & Evaluation** | Critical | Medium | Strictly enforce fit-on-train-only policies for scalers, imputers, PCA, and models. Cross-validation will execute inside scikit-learn pipelines. |
| **LLM Hallucination or API Key Dependency** | Medium | Medium | Pluggable architecture: the system functions completely without an API key using deterministic rule-based report drafting. LLM drafts strictly cite retrieved RAG passages. |

