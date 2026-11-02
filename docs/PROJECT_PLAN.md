# AuditLens — Project Plan

## 1. Project Overview & Vision
AuditLens is a financial anomaly and fraud investigation copilot designed for fraud analysts, investigators, and compliance professionals. It provides decision support by combining unsupervised anomaly detection, supervised fraud risk classification, transparent per-transaction explainability, regulatory document retrieval (RAG), and evidence-grounded report drafting.

AuditLens strictly operates as a decision-support tool. All final review, enforcement, account actions, and regulatory filings remain with human analysts.

---

## 2. Implementation Roadmap & Phases

The project is structured into 13 strictly sequential phases. Progression requires meeting every acceptance criterion and passing the formal exit gate.

| Phase | Title | Objective | Primary Deliverables | Exit Gate Criteria |
|---|---|---|---|---|
| **Phase 0** | Repository Inspection & Technical Plan | Baseline repository state, define architecture, document decisions, assess risks. | `PROJECT_PLAN.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `.env.example`, `PHASE_STATUS.md` | Workspace inspected, zero secrets, baseline recorded, approval gate reached. |
| **Phase 1** | Project Foundation & Local Dev Environment | Bootstrap clean skeleton for backend and frontend. | FastAPI app, health endpoint, React/Next.js shell, structured logging, `.gitignore`, initial test setup. | Backend and frontend start cleanly, `/health` returns 200, linting and baseline tests pass. |
| **Phase 2** | Dataset Acquisition, Profiling & Validation | Ingest reference dataset, profile features, create upload validation. | Dataset loader, schema validator, profiling report, synthetic CSV test fixtures. | Valid CSVs parsed, malformed/oversized uploads rejected with clear errors, no PII leakage. |
| **Phase 3** | EDA & Leakage-Safe Preprocessing | Analyze distributions, class balance, and build reproducible preprocessing pipelines. | EDA scripts/notebooks, train/test split (stratified/time-based), feature transformers. | Reproducible preprocessing, no target leakage, fitted only on training data, unit tests pass. |
| **Phase 4** | Baseline Machine Learning Models | Train simple, defensible anomaly and supervised fraud models. | Isolation Forest (anomaly), Logistic Regression/Random Forest (supervised), model artifacts. | Models reproducible with fixed seeds, inference functions tested, separate anomaly and fraud scores. |
| **Phase 5** | Evaluation, Comparison & Threshold Selection | Evaluate on held-out test split using imbalanced classification metrics. | PR-AUC, ROC-AUC, confusion matrices, operational threshold selection report. | Evaluation strictly reproducible on test set, threshold trade-offs documented, metrics validated. |
| **Phase 6** | Explainability & Analyst-Facing Evidence | Provide per-transaction feature contributions and rationale. | SHAP integration or fallback attribution, human-readable explanations, limitation caveats. | Explanations match model versions and feature values; clear warnings on uncertainty; tests pass. |
| **Phase 7** | Backend API & Case Management | Expose endpoints for upload, scoring, explanation, case persistence, and audit logging. | REST API endpoints, SQLite persistence, audit trail, pagination, request validation. | All endpoints unit- and integration-tested; database migrations run cleanly; no sensitive data leaks. |
| **Phase 8** | Dashboard & Analyst Workflow | Deliver responsive web UI for investigation workflow. | Overview metrics, upload view, transaction review table, case management UI. | End-to-end user flow working against real backend API; responsive; loading/error states handled. |
| **Phase 9** | Reference Document Ingestion & RAG | Enable semantic retrieval over public compliance/policy docs. | Ingestion pipeline, chunking, vector indexing (Chroma/local), traceable citation retriever. | Traceable citations with section/source metadata; safe handling of empty or ungrounded queries. |
| **Phase 10** | LLM Investigation Assistant & Report Drafting | Draft evidence-grounded investigation reports with human-in-the-loop controls. | Provider-agnostic LLM adapter, structured output schema, hallucination check, draft review UI. | Graceful degradation without API key; reports strictly grounded in transaction facts; draft status clear. |
| **Phase 11** | Security, Privacy, Reliability & E2E Validation | System hardening, input sanitization, PII masking verification, full regression pass. | Rate limiting, size limits, security review, regression test suite. | Zero high-severity vulnerabilities; PII masked in logs; complete end-to-end regression pass. |
| **Phase 12** | Documentation, Demo & Release Readiness | Complete documentation, reproducible demo walkthrough, and academic project report. | Updated README, demo script with synthetic data, system manual, final evaluation report. | Project reproducible from scratch by a third party; all documentation verified against working code. |

---

## 3. Technology Stack Summary

- **Backend Runtime & API**: Python 3.13.7, FastAPI, Uvicorn, Pydantic v2.
- **Frontend Application**: Next.js / React with TypeScript, Tailwind CSS.
- **Data & Modeling**: Pandas, NumPy, scikit-learn (Isolation Forest, Logistic Regression, Random Forest), SHAP.
- **Persistence**: SQLite (local development), SQLAlchemy / SQLModel.
- **Retrieval & RAG**: Local vector store (Chroma or lightweight sqlite-vec / numpy vector store), LangChain (if justified).
- **Testing & Quality**: `pytest` (backend), `vitest` / `jest` (frontend), `flake8` / `ruff`, `mypy`.
- **Packaging & Config**: Virtual environment (`.venv`), `npm`, `.env` via `python-dotenv`.

---

## 4. Key Risks & Mitigations

1. **Class Imbalance in Fraud Data**:
   - *Risk*: A trivial classifier predicting "non-fraud" achieves 99.8%+ accuracy while detecting zero fraud.
   - *Mitigation*: Never evaluate on raw accuracy. Use PR-AUC (Average Precision), F1-score, Precision-Recall curves, and operational threshold analysis.
2. **Data Leakage**:
   - *Risk*: Preprocessing transformations fitted on full data or temporal leakage across train/test splits.
   - *Mitigation*: Strictly fit preprocessing pipelines only on training splits. Use time-based splits when timestamps exist.
3. **LLM Hallucination & External Dependency**:
   - *Risk*: External LLMs generating fabricated regulatory articles or false transaction facts.
   - *Mitigation*: Rule-grounded schema validation. Direct extraction of observed facts from database. LLM operates strictly as an optional draft writer; system remains 100% functional without an API key.
4. **Data Privacy & PII Leakage**:
   - *Risk*: Exposing real financial or personal data in logs, UI screenshots, or API errors.
   - *Mitigation*: Strict masking middleware in logging; support synthetic datasets for development, testing, and demonstrations.
5. **System Compatibility**:
   - *Risk*: Newer Python (3.13.7) or Node (v25.6.1) compatibility with legacy C-extensions.
   - *Mitigation*: Prefer robust, widely maintained wheels (scikit-learn) and implement clear fallback logic (e.g. tree interpreter / coefficient attribution if native SHAP wheels fail to compile on Python 3.13 Apple Silicon).

---

## 5. Execution Protocol & Phase Gates

For each phase:
1. Inspect workspace and read `docs/PHASE_STATUS.md`.
2. State goals, files to change, and acceptance criteria.
3. Implement only what is in scope for the active phase.
4. Run automated tests, linting, and smoke checks. Fix any issues immediately.
5. Update `docs/PHASE_STATUS.md` and project memory logs (`01_PROBLEM_STATEMENT.md`, `02_EXECUTION_LOG.md`, `03_LESSONS_LEARNED.md`).
6. Present the phase report and halt execution. **Wait for explicit user approval before proceeding to the next phase.**

