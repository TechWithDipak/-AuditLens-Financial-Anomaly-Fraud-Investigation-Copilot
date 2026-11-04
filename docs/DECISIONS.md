# AuditLens — Architectural Decision Records (ADR)

**Project Name:** AuditLens  
**Date:** 2026-11-04  
**Status:** Approved for Implementation

---

## ADR-001: Technology Stack Selection & Runtime Support

### Status
Accepted

### Context
AuditLens requires a high-performance, locally executable, and maintainable stack supporting data science, statistical machine learning, RESTful web services, and an interactive dark-themed dashboard. The local runtime environment is Windows 11 with:
- Python 3.13.0
- Node.js v24.16.0
- npm 11.13.0
- git version 2.54.0

### Decision
1. **Backend:** Python 3.13 with FastAPI, Pydantic v2, and Uvicorn. Data science libraries will utilize pre-compiled wheels for `win_amd64` (NumPy >=2.1.0, Pandas >=2.2.3, Scikit-learn >=1.5.2, XGBoost >=2.1.2).
2. **Persistence:** SQLite for relational state and case management; Alembic for database migrations; joblib and JSON for model artifact registries.
3. **Frontend:** React 18 with TypeScript, bundled with Vite for fast local hot-module-replacement. Styling will use Tailwind CSS with a customized graphite palette. Data visualizations will utilize Recharts.

### Consequences
- Standardized, reproducible local development without heavy external infrastructure dependencies (such as Docker or external PostgreSQL servers).
- Ensures seamless cross-platform execution on Windows while maintaining full compatibility with modern Python 3.13.

---

## ADR-002: Primary Tabular Dataset Selection

### Status
Accepted

### Context
The project requires a representative financial fraud dataset for Unit 3 (K-Means, GMM, PCA) and Unit 5 (CART, XGBoost, Ensemble). Candidate datasets evaluated:
1. **Kaggle Credit Card Fraud Detection (ULB):** 284,807 transactions, 31 features (`Time`, `V1`-`V28` PCA features, `Amount`, `Class`), 492 fraud cases (0.172% imbalance).
2. **IEEE-CIS Fraud Detection:** Massive multi-table dataset (>1.5 GB), complex joins, high memory footprint unsuitable for standard local academic runtimes.
3. **Synthetic PaySim:** 6.3M rows, synthetic financial transfer data with account numbers, but large file size (~500 MB).

### Decision
Adopt the **Kaggle Credit Card Fraud Detection dataset** as the primary tabular dataset for Units 3 and 5.
- The raw dataset will be downloaded into `data/raw/` and strictly gitignored.
- A small, deterministic, synthetic fixture (`data/fixtures/creditcard_sample_fixture.csv`) containing 1,000 transactions (with positive and negative instances matching the feature schema) will be checked into Git for automated testing and CI verification.

### Consequences
- Real-world fraud evaluation characteristics with genuine class imbalance (0.17%).
- Fast ingestion, low memory overhead, and instant evaluation on standard workstations.
- Raw personal or benchmark data is kept out of Git.

---

## ADR-003: Sequential Data Strategy for Unit 4 (HMM, Forward, Viterbi)

### Status
Accepted

### Context
Unit 4 requires sequence modeling across chronological transaction histories for individual customer accounts:
- Hidden Markov Models (HMM)
- Forward Algorithm (sequence likelihood)
- Viterbi Algorithm (decoding optimal latent state paths)

However, the Kaggle Credit Card Fraud dataset is deliberately anonymized: it contains only elapsed `Time` (seconds from first transaction), without customer IDs or multi-transaction account trajectories. Inventing fake customer IDs for anonymized credit card data would violate academic integrity and dataset provenance rules.

### Decision
1. Explicitly document this limitation in `docs/DATASET_CARD.md` and `docs/ALGORITHM_GUIDE.md`.
2. For Unit 4, provide a dedicated, clearly labeled **Synthetic Financial Account Sequence Dataset** (`data/fixtures/sequence_account_fixture.csv`).
3. The sequence dataset will represent realistic multi-step customer behavioral states:
   - State 0: Normal Day-to-Day Spending
   - State 1: Travel / Elevated Activity
   - State 2: Rapid High-Risk Burst / Account Takeover
4. In addition, provide educational hand-calculated toy sequence matrices to rigorously verify Forward and Viterbi dynamic programming calculations.

### Consequences
- 100% adherence to academic integrity: no fabricated claims that the Kaggle dataset possesses account histories.
- Flawless demonstration of HMM transitions, Forward sequence scoring, and Viterbi path decoding on legitimate sequential data.

---

## ADR-004: Direct Vectorized Implementation of HMM Forward and Viterbi

### Status
Accepted

### Context
External HMM libraries (such as `hmmlearn`) rely on compiled C/Cython extensions that frequently encounter build-tooling bottlenecks and compiler mismatches on Windows with new Python versions (Python 3.13). Furthermore, the academic specification explicitly requires hand-calculated verification and demonstration of numerical underflow protection.

### Decision
Implement the HMM inference engine (`hmm.py`, `hmm_forward.py`, `hmm_viterbi.py`) as clean, vectorized, pure NumPy modules:
1. **Forward Algorithm:** Implement with numerical scaling factors ($c_t = \frac{1}{\sum \alpha_t}$) and log-likelihood accumulation to prevent float underflow on long sequences.
2. **Viterbi Algorithm:** Formulate entirely in log-space ($\max [\log v_{t-1} + \log a_{ij}] + \log b_j$) to guarantee numerical stability.
3. Include unit tests asserting exact equivalence against small hand-calculated analytical matrices.

### Consequences
- Zero external C-compiler dependencies; guaranteed installation on any OS running Python 3.13.
- Full inspection capability and academic clarity for reviewers and students.
- Immune to external library deprecations or version drift.

---

## ADR-005: Strict Score Taxonomy and Output Semantics

### Status
Accepted

### Context
In multi-model fraud analytics systems, analysts are frequently confused when different model outputs (posterior probabilities, log-likelihood densities, distance metrics, and cluster IDs) are conflated or averaged into an arbitrary "Risk Score".

### Decision
Enforce strict typing and distinct semantic labeling across all APIs, database schemas, and frontend views:
1. `fraud_probability`: Strictly for supervised classifiers (CART, XGBoost, Ensembles) representing $P(Y=1 \mid X) \in [0, 1]$.
2. `gmm_log_likelihood`: Strictly for GMM density representing log-likelihood under the mixture distribution $\in (-\infty, 0]$.
3. `unusualness_percentile`: Percentile rank of low log-likelihood observations.
4. `kmeans_cluster_id`: Discrete cluster index $\{0, \dots, k-1\}$ denoting behavioral peer grouping.
5. `hmm_sequence_log_prob`: Observation sequence likelihood under Markovian transition dynamics.
6. `hmm_inferred_state`: Decoded discrete behavioral state from the Viterbi path.

No composite single "Risk Score" will be computed unless a formal, calibrated weighting model is documented and validated.

### Consequences
- Prevents false-positive accusations based on statistical density anomalies alone.
- Maintains high regulatory and compliance standards.

---

## ADR-006: Local Persistence via SQLite and Alembic

### Status
Accepted

### Context
AuditLens needs transactional persistence for datasets, model runs, predictions, analyst cases, case notes, and immutable audit logs, while remaining lightweight and zero-setup for local academic evaluation.

### Decision
- Use SQLite with WAL (Write-Ahead Logging) mode enabled for concurrency.
- Structure all schema changes through Alembic migrations in `backend/alembic/`.
- Provide a reset script and documented commands to safely wipe local demo data when requested.

### Consequences
- Zero external database services to configure or start.
- Seamless local portability across developer workstations.

---

## ADR-007: Pluggable RAG and Dual-Mode LLM Assistant

### Status
Accepted

### Context
The application includes a RAG knowledge base of regulatory references (FATF, FinCEN, Wolfsberg) and an optional LLM investigation copilot. However:
1. Reviewers may not possess an active OpenAI or Gemini API key.
2. Core application features (upload, preprocessing, ML models, case management) must function 100% without external API keys.
3. Sensitive financial data must never be transmitted to external endpoints without analyst consent.

### Decision
1. Implement a pluggable Assistant Engine:
   - **Offline / Mock Mode (Default):** Deterministic, rule-based report generator that extracts verified citations from the local RAG store and populates a structured case investigation draft without any API key or network request.
   - **External LLM Mode (Optional):** Active only if `OPENAI_API_KEY` or `GEMINI_API_KEY` is provided in `.env`.
2. Enforce strict JSON schema validation on all generated reports.
3. Every draft generated must display a prominent badge: *"AI-Generated Draft — Requires Human Analyst Adjudication"*.

### Consequences
- The application is 100% testable and runnable offline out of the box.
- Zero risk of evaluation failure due to missing third-party credentials or rate limits.

