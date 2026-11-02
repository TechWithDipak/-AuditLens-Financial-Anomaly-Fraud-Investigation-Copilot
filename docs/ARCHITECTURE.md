# AuditLens — System Architecture

## 1. System Overview

AuditLens is built as a **modular monolith** optimized for local development, reproducible academic demonstrations, and enterprise extensibility. It cleanly separates the frontend user interface, backend REST API services, machine learning pipeline, data persistence layer, and retrieval/LLM services.

```
+-------------------------------------------------------------------+
|                        Analyst Web Browser                        |
|             (React / Next.js with TypeScript & Tailwind)          |
+---------------------------------+---------------------------------+
                                  | HTTP / JSON REST
                                  v
+-------------------------------------------------------------------+
|                           Backend API                             |
|              (FastAPI on Python 3.13 + Uvicorn)                   |
|                                                                   |
|   +-----------------------+     +-----------------------------+   |
|   | Auth & Validation     |     | Structured Logging & Masking|   |
|   +-----------------------+     +-----------------------------+   |
|                                                                   |
|   +-----------------------+     +-----------------------------+   |
|   | Data Ingestion &      |     | Case Management &           |   |
|   | Validation Service    |     | Audit Trail Service         |   |
|   +-----------------------+     +-----------------------------+   |
|                                                                   |
|   +-----------------------+     +-----------------------------+   |
|   | ML Scoring &          |     | RAG Retrieval & Report      |   |
|   | Explainability Engine |     | Draft Generator (Optional)  |   |
|   +-----------------------+     +-----------------------------+   |
+-------------------+-----------------------------+-----------------+
                    |                             |
                    v                             v
+-----------------------------+     +-------------------------------+
|     Persistence Layer       |     |      Vector Knowledge Base    |
|   SQLite (Local Relational) |     |  ChromaDB / SQLite Vector     |
|   SQLAlchemy / SQLModel     |     |  Policy & Regulatory Docs     |
+-----------------------------+     +-------------------------------+
                    ^                             ^
                    |                             |
+-------------------+-----------------------------+-----------------+
|                  Data & Machine Learning Pipeline                 |
|            - Ingestion & Profiling (Pandas / NumPy)               |
|            - Leakage-safe Preprocessing Pipelines                 |
|            - Unsupervised Anomaly Model (Isolation Forest)        |
|            - Supervised Classifier (Logistic Regression / RF)     |
|            - Explainability (SHAP / Tree-Linear Attributions)     |
|            - Serialized Artifacts (`models/`, `data/sample/`)     |
+-------------------------------------------------------------------+
```

---

## 2. Component Descriptions

### 2.1 Frontend (`frontend/`)
- **Technology**: Next.js / React with TypeScript and Tailwind CSS.
- **Responsibilities**:
  - Triage Dashboard: high-level KPI cards, alert trends, validation summaries.
  - Transaction Explorer: sortable, searchable, filterable data grid with pagination.
  - Detail & Investigation View: score breakdown (anomaly score vs. fraud probability), visual explanation waterfall (SHAP or feature contributions), and contextual caveats.
  - Case Workspace: analyst case creation, status updates (Open, Under Investigation, Escalated, Closed), analyst notes, and audit event views.
  - Report Review Pane: side-by-side view of facts, retrieved regulations, and draft findings.

### 2.2 Backend REST API (`backend/`)
- **Technology**: FastAPI, Pydantic v2, Uvicorn.
- **Responsibilities**:
  - Request validation and serialization.
  - Sensitive field masking in application logs.
  - Endpoints for dataset upload, transaction pagination/filtering, scoring, explainability, case lifecycle, and health monitoring (`GET /health`).
  - Error-handling middleware with structured error envelopes.

### 2.3 Machine Learning Engine (`ml/`)
- **Technology**: scikit-learn, Pandas, NumPy, SHAP (with fallback).
- **Core Design Principles**:
  - **Decoupled Tasks**:
    1. *Anomaly Detection (Unsupervised)*: Identifies transactions deviating from normal distributions (e.g., Isolation Forest). Outputs continuous anomaly scores and binary outlier flags.
    2. *Fraud Classification (Supervised)*: Predicts likelihood of confirmed fraud based on labeled historical cases. Outputs calibrated probabilities $[0.0, 1.0]$.
  - **Reproducibility**: Standardized random seeds (`RANDOM_STATE = 42`), versioned model artifact storage (`models/`), and JSON metadata tracking hyperparameters and training metrics.
  - **Explainability**: Local feature importance for each flagged transaction, distinguishing global model weights from instance-level signal contributions.

### 2.4 Data Management & Persistence (`data/`, `backend/db/`)
- **Technology**: SQLite, SQLAlchemy / SQLModel.
- **Directory Layout**:
  - `data/raw/`: Raw external datasets (strictly git-ignored).
  - `data/processed/`: Normalized, preprocessed arrays/splits (git-ignored).
  - `data/sample/`: Small, synthetic, anonymized test fixtures (committed for reproducible test runs).
- **Relational Schema**:
  - `cases`: Investigation case identifier, title, status, assigned analyst, priority, timestamps.
  - `transactions`: Normalized transaction records or reference hashes.
  - `audit_logs`: Immutable log of analyst actions (viewed record, changed status, added note, exported report).
  - `notes`: Timestamped investigator commentary.

### 2.5 Retrieval-Augmented Generation & Report Service (`backend/services/rag/`)
- **Technology**: Local Vector Index (ChromaDB or SQLite vector extension), provider-agnostic LLM interface.
- **Responsibilities**:
  - Chunking and indexing public regulatory guides (e.g., FinCEN guidance, AML thresholds, internal credit card dispute procedures).
  - Traceable passage retrieval with source URI, section, and publication date.
  - Report drafting: synthesizes observed facts, model scores, top contributing features, and retrieved citations into a structured draft.
  - Strict fallback: if no LLM key is configured, the system generates a template-based summary directly from facts and citations without external API calls.

---

## 3. Data Flow

```
1. Transaction Upload
   Analyst -> [POST /api/v1/data/upload] -> CSV Validator -> Store in DB/Cache

2. Analysis & Scoring
   Backend -> Load Preprocessing Pipeline -> Isolation Forest -> Anomaly Score
           -> Supervised Classifier      -> Fraud Probability
           -> Explainability Engine      -> Feature Attributions

3. Investigation & Triage
   Analyst inspects transaction -> [GET /api/v1/transactions/{id}]
   Analyst opens case          -> [POST /api/v1/cases]
   Analyst queries policy       -> [POST /api/v1/rag/query] -> Vector Store -> Citations

4. Case Resolution
   Analyst triggers draft report -> [POST /api/v1/cases/{id}/draft-report]
   LLM Service / Template Engine compiles draft -> Analyst edits & finalizes report
   Action recorded to Audit Log
```

---

## 4. Security, Privacy & Compliance Architecture

1. **Zero Hardcoded Secrets**: All configuration, ports, and API credentials loaded exclusively from environment variables via Pydantic Settings (`.env` ignored by git).
2. **Sensitive Data Masking**: PII, credit card numbers, and account identifiers are masked at the API ingestion boundary and in structured logs (e.g. `4532-XXXX-XXXX-1234`).
3. **Decision-Support Boundary**: The system contains no external integration points capable of executing monetary holds, freezing cards, or filing regulatory notices. All outputs are explicitly advisory.
4. **Model Uncertainty**: UIs and reports prominently display model version, score types, and operational limitations to prevent blind automation bias.

