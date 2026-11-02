# AuditLens — Financial Anomaly & Fraud Investigation Copilot

AuditLens is a web-based decision-support copilot that assists fraud analysts and compliance professionals with transaction data validation, unsupervised anomaly detection, supervised fraud scoring, explainable evidence generation, and regulatory policy retrieval.

> **Decision-Support Notice**: AuditLens is strictly advisory. It does not freeze accounts, decline transactions automatically, submit Suspicious Activity Reports (SARs) directly to authorities, or make autonomous legal/financial determinations. All actions remain with human analysts.

---

## Architecture & Technology Stack

- **Backend**: FastAPI (Python 3.13.7) with Pydantic v2, Uvicorn, and structured logging with PII/PAN masking.
- **Frontend**: React 18 + TypeScript + Vite with a dark-mode analyst triage layout.
- **ML Engine**: Scikit-Learn (Isolation Forest & supervised classifiers) and explainability modules (*implemented in subsequent phases*).
- **Persistence**: SQLite (local development) with clean relational schemas.
- **RAG & Policy**: Vector store retrieval over regulatory guidance (*implemented in Phase 9*).

---

## Getting Started (Local Setup)

### Prerequisites
- Python 3.13+ (or 3.11+)
- Node.js v20+ / v25+ and npm 10+
- SQLite 3

### 1. Backend Setup

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt
pip install ruff  # for formatting and linting

# 3. Copy environment configuration
cp .env.example .env

# 4. Start the backend development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **Health Check Endpoint**: `http://127.0.0.1:8000/health`

### 2. Frontend Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. (Optional) Copy frontend environment configuration
cp .env.example .env

# 4. Start frontend development server
npm run dev
```

The UI dashboard will run at `http://localhost:3000`.

---

## Verification & Testing Commands

### Backend Verification
```bash
# Run backend test suite (unit tests, logging filter tests, error handler tests)
.venv/bin/pytest -v

# Run code linter and formatting check
.venv/bin/ruff check backend tests
```

### Frontend Verification
```bash
cd frontend

# Type-check and lint
npm run lint

# Run frontend tests
npm test

# Build production bundle
npm run build
```

---

## Security & Privacy Guardrails
1. **Zero Secret Commits**: All credentials and API keys are read via environment variables. `.env` is ignored by Git.
2. **Sensitive Data Masking**: All application logging automatically masks PAN credit card numbers, SSNs, passwords, CVVs, and Bearer tokens.
3. **Structured Errors**: Unhandled 500 errors and internal stack traces are caught by error handlers and never leaked to API consumers.

