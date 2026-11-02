# AuditLens — Architecture Decision Records (ADRs)

## ADR-001: Architecture Style — Modular Monolith over Distributed Microservices
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: AuditLens requires a frontend UI, backend API, ML scoring, local persistence, vector retrieval, and optional LLM orchestration. Microservice architectures introduce substantial overhead (network latency, service discovery, container orchestration, distributed tracing) without justification for single-analyst local workflows or academic demonstrations.
- **Decision**: Adopt a modular monolith directory structure (`backend/`, `frontend/`, `ml/`, `data/`, `tests/`). The backend communicates with ML pipelines and persistence via well-defined in-process Python interfaces, and the frontend communicates via HTTP REST APIs.
- **Consequences**:
  - *Positive*: Easy local setup, straightforward unit/integration testing, simple zero-dependency deployment, no container orchestration required for development.
  - *Negative*: If independent horizontal scaling of scoring vs API is needed in future enterprise tiers, services will need separation.
  - *Mitigation*: Maintain clean dependency boundaries within `backend/services/` so components can be extracted if needed later.

---

## ADR-002: Backend Framework — FastAPI with Python 3.13
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: The backend needs high performance, native async support, automatic OpenAPI documentation, and direct integration with Python data science libraries (Pandas, scikit-learn).
- **Decision**: Use FastAPI with Pydantic v2 and Uvicorn.
- **Consequences**:
  - *Positive*: Built-in input validation, fast JSON serialization, native OpenAPI interactive docs (`/docs`), seamless integration with scientific Python stacks.
  - *Negative*: Python 3.13 is modern; C-extension packages must be vetted for wheel support.

---

## ADR-003: Frontend Framework — Next.js / React with TypeScript
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: The analyst dashboard requires rich interactive components: dynamic tables with filtering/pagination, data visualization charts, side-by-side case comparison, and responsive layouts.
- **Decision**: Use Next.js / React with TypeScript and Tailwind CSS.
- **Consequences**:
  - *Positive*: Strong typing for API responses, modern UI components, fast local developer experience with hot reloading.
  - *Negative*: Requires Node.js build step alongside Python backend.
  - *Mitigation*: Clearly document start commands for both frontend and backend in project README.

---

## ADR-004: Persistence Engine — SQLite for Local Development
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: The application must persist case data, investigator notes, and audit logs across sessions without requiring external database servers (like PostgreSQL or MySQL) to run locally.
- **Decision**: Use SQLite with SQLAlchemy / SQLModel as the default database for local development, maintaining clean schema definitions that can easily point to PostgreSQL via connection string configuration.
- **Consequences**:
  - *Positive*: Zero installation/container overhead, single file persistence, ideal for academic demos and automated integration tests.
  - *Negative*: Concurrency limitations under high write loads. Acceptable for decision-support analyst workloads.

---

## ADR-005: Decoupled ML Modeling — Anomaly Detection vs. Supervised Classification
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: Real-world fraud analysis involves both novel anomalous behaviors (unlabeled) and known fraud patterns (labeled). Confusing an anomaly score with a fraud probability misleads investigators.
- **Decision**: Strictly separate unsupervised anomaly detection (e.g. Isolation Forest) and supervised fraud classification (e.g. Logistic Regression / Random Forest). The system will report both as independent metrics and never describe an anomaly score as a probability of guilt.
- **Consequences**:
  - *Positive*: Methodologically sound, reduces false accusations, maintains analyst trust.
  - *Negative*: UI must display and explain two distinct scoring dimensions.

---

## ADR-006: LLM & RAG Provider-Agnostic Design with Offline Fallback
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: Relying strictly on a paid cloud LLM (e.g., OpenAI, Anthropic, Gemini) creates cost risks, privacy concerns, and blocks local execution when API keys are absent.
- **Decision**: Build a provider-agnostic adapter layer with a full local template-based fallback. If no API key is provided, report drafting and RAG query formatting execute deterministically using local structured templates and extracted database facts.
- **Consequences**:
  - *Positive*: Application remains 100% operational offline and without external API costs.
  - *Negative*: Offline template summaries lack the freeform narrative nuance of large language models.

---

## ADR-007: Privacy, Masking & Decision-Support Boundaries
- **Date**: 2026-10-31
- **Status**: Accepted
- **Context**: Financial records contain sensitive PII and financial transaction data. Furthermore, automated systems making punitive financial decisions pose severe legal, regulatory, and ethical hazards.
- **Decision**: Enforce strict logging filters that mask sensitive fields (card numbers, names, SSNs/IDs). Maintain an absolute architectural boundary: AuditLens will produce recommendations, evidence, and risk indicators, but will never execute automated external actions (e.g. freezing accounts or auto-declining).
- **Consequences**:
  - *Positive*: Prevents data leaks in logs, complies with financial regulations, ensures human-in-the-loop review.
  - *Negative*: Analysts must manually approve actions in downstream external banking systems.

