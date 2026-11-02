# Project Memory: Problem Statement & Context

## 1. Context & Background
The project **AuditLens** is a decision-support copilot designed for financial fraud and anomaly investigation. In financial institutions, analysts must sift through thousands of transaction records daily, balancing detection speed, false positive triage, explainability, regulatory compliance, and evidence documentation.

This repository (`Machine learing project`) is a greenfield workspace governed by `AUDITLENS_PHASED_IMPLEMENTATION_PLAN.md`.

## 2. Problem Statement
Financial fraud investigation workflows currently suffer from:
1. **High False-Positive Noise**: Existing rules and monolithic black-box systems produce overwhelming alerts without calibrated uncertainty.
2. **Lack of Transparent Explainability**: Investigators struggle to understand *why* a particular transaction was flagged, leading to slow review times or blind acceptance.
3. **Disconnected Regulatory Context**: Cross-referencing compliance guidelines and anti-money laundering (AML) / fraud policies is manual and detached from the data scoring workflow.
4. **Tedious Documentation**: Compiling case findings into consistent, evidence-grounded reports requires substantial manual synthesis.

## 3. Project Goal
Deliver **AuditLens**: a web-based, modular financial anomaly and fraud investigation copilot that empowers analysts to:
- Ingest and validate transaction data safely.
- Run baseline anomaly detection (unsupervised) and fraud risk scoring (supervised).
- View per-transaction explainability (e.g. SHAP values / feature importance) with operational caveats.
- Retrieve relevant policy and regulatory reference excerpts via retrieval-augmented generation (RAG).
- Generate structured, evidence-grounded draft investigation summaries with an optional LLM.
- Manage investigation cases through an interactive analyst dashboard.

## 4. Scope & Boundaries
### In Scope
- Phased, test-driven implementation (Phases 0 through 12).
- FastAPI backend serving REST endpoints for validation, scoring, explainability, case management, and report generation.
- React/Next.js TypeScript dashboard for analyst interaction.
- Baseline ML models: Isolation Forest for anomaly detection, Logistic Regression/Random Forest for supervised classification.
- Local SQLite database for persistent case management and audit logging.
- Fallback modes ensuring core functionality runs without external LLM keys or network dependencies.
- Strict PII masking and security guardrails.

### Out of Scope (Explicit Non-Goals)
- **Autonomous Authority**: AuditLens will NOT freeze accounts, decline transactions automatically, submit Suspicious Activity Reports (SARs) directly to authorities, or make final determinations of guilt.
- **Premature Distributed Architecture**: No Kafka, Kubernetes, or microservice mesh for MVP local dev.
- **Unverified Third-Party Integrations**: No direct live connections to banking payment rails or production core-banking systems.

## 5. Constraints & Environmental Assumptions
- **Host Environment**: macOS (Darwin arm64) with Python 3.13.7, Node.js v25.6.1, npm 11.9.0, SQLite 3.54.0.
- **Reproducibility**: Preprocessing parameters, random seeds, package versions, and train/test splits must be strictly recorded.
- **Evaluation Discipline**: Fraud datasets are severely imbalanced; accuracy alone is prohibited. Evaluation must use PR-AUC (Average Precision), F1, Precision, Recall, ROC-AUC, and threshold trade-off analysis.
- **Security & Privacy**: Zero hardcoded secrets, parameterized queries, and masking of account numbers / PII in logs.

## 6. Testable Success Criteria
1. Backend health check endpoint returns `200 OK`.
2. Clean environment setup reproducible via documented instructions.
3. Unit, integration, and ML validation test suite passes with zero errors.
4. CSV upload endpoint validates schemas and gracefully rejects malformed, oversized, or missing-column inputs.
5. Baseline models output separate anomaly scores and supervised probabilities.
6. SHAP or documented fallback explainability correctly reflects feature contributions.
7. RAG service returns traceable citations or explicit "no relevant source found" responses.
8. Investigation reports clearly distinguish observed facts, model outputs, and AI-generated drafts.

## 7. Solution Approach
Follow the 13-phase plan (`AUDITLENS_PHASED_IMPLEMENTATION_PLAN.md`, Phases 0 to 12) strictly:
- Build modular components: `backend/`, `frontend/`, `ml/`, `data/`, `docs/`, `tests/`.
- Maintain strict phase gates: verify with tests, update phase status, and wait for user approval before moving to the next phase.

