# AuditLens REST API Documentation

This document describes the AuditLens REST API endpoints provided by FastAPI, including request and response schemas, error responses, and database persistence models.

---

## 1. Overview & Base URLs

- **Base URL:** `http://localhost:8000`
- **Interactive Documentation:** `http://localhost:8000/docs` (OpenAPI Swagger UI)
- **Alternative Docs:** `http://localhost:8000/redoc` (ReDoc)
- **Version Prefix:** `/api/v1`

---

## 2. Health & System Endpoints

### `GET /health`
Verifies backend service operational health.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "app_name": "AuditLens Backend",
  "version": "0.1.0"
}
```

---

## 3. Dataset Management Endpoints (`/api/v1/datasets`)

### `GET /api/v1/datasets/sample`
Returns paginated transactions from the canonical European credit card dataset fixture.
- **Query Parameters**:
  - `limit` (int, default 100, max 1000): Number of rows to return.
  - `offset` (int, default 0): Row index offset.
- **Response `200 OK`**: Returns array of transaction records with columns `Time`, `V1`..`V28`, `Amount`, `Class`.

### `GET /api/v1/datasets/sequence-sample`
Returns temporal account sequences designed for Unit 4 HMM sequence evaluation.
- **Query Parameters**:
  - `account_id` (optional str): Filter by account ID (e.g. `ACC_0001`).
- **Response `200 OK`**: Returns array of records with `account_id`, `step`, `amount`, `merchant_category`, `is_anomaly`.

### `POST /api/v1/datasets/validate-upload`
Uploads and validates a CSV dataset against strict schema and integrity checks.
- **Form Data**: `file` (multipart/form-data CSV file, max 100MB).
- **Response `200 OK`**:
```json
{
  "filename": "transactions.csv",
  "row_count": 1000,
  "column_count": 31,
  "columns": ["Time", "V1", "...", "Amount", "Class"],
  "missing_values": {},
  "duplicate_rows": 0,
  "class_balance": {"0": 980, "1": 20},
  "is_valid": true,
  "warnings": []
}
```

---

## 4. Preprocessing & Splits (`/api/v1/preprocessing`)

### `POST /api/v1/preprocessing/eda`
Computes non-destructive exploratory data distributions, statistical moments, skewness, and velocity.
- **Request Body**:
```json
{
  "dataset_path": "data/fixtures/creditcard_sample_fixture.csv"
}
```
- **Response `200 OK`**: Summary statistics per column (mean, median, std, skewness, min, max, missing count).

### `POST /api/v1/preprocessing/split`
Partitions data into Train, Validation, and Test sets without lookahead or data leakage.
- **Request Body**:
```json
{
  "dataset_path": "data/fixtures/creditcard_sample_fixture.csv",
  "split_type": "stratified",
  "test_size": 0.2,
  "val_size": 0.1,
  "random_state": 42
}
```
- **Response `200 OK`**: Class distributions and row counts for train, validation, and test splits.

---

## 5. Unsupervised Algorithms - Unit 3 (`/api/v1/algorithms`)

### `POST /api/v1/algorithms/kmeans`
Clusters transactions to identify normal spending peer groups and compute an elbow curve.
- **Request Body**:
```json
{
  "n_clusters": 5,
  "max_iter": 300,
  "random_state": 42,
  "compute_elbow": true,
  "k_min": 2,
  "k_max": 8
}
```
- **Response `200 OK`**: Cluster centroids, cluster sizes, inertia, silhouette score, and optional elbow curve points.

### `POST /api/v1/algorithms/gmm`
Fits Gaussian Mixture Models via Expectation-Maximization to calculate sample responsibilities and density-based anomaly ranking.
- **Request Body**:
```json
{
  "n_components": 3,
  "covariance_type": "full",
  "max_iter": 100,
  "random_state": 42,
  "top_anomalous_n": 10
}
```
- **Response `200 OK`**: Component weights, convergence status, lower-bound log-likelihood, and top ranked rare transactions.

### `POST /api/v1/algorithms/pca`
Projects multi-dimensional transaction features to 2D for visual inspection and computes explained variance ratios.
- **Request Body**:
```json
{
  "n_components": 2,
  "scale_data": true
}
```
- **Response `200 OK`**: `explained_variance_ratio`, `cumulative_variance`, and 2D coordinate projections.

---

## 6. Sequence Analysis - Unit 4 (`/api/v1/algorithms/hmm`)

### `POST /api/v1/algorithms/hmm/forward`
Executes the scaled Forward algorithm to compute sequence observation probability $P(O|\lambda)$ without floating-point underflow.
- **Request Body**:
```json
{
  "observations": [0, 1, 0, 2, 1],
  "initial_probs": [0.6, 0.3, 0.1],
  "transition_matrix": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
  "emission_matrix": [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
}
```
- **Response `200 OK`**: `log_likelihood`, `total_probability`, and forward lattice dimensions.

### `POST /api/v1/algorithms/hmm/viterbi`
Decodes the optimal hidden state sequence $\arg\max_Q P(Q|O,\lambda)$ using max-product dynamic programming in log-space.
- **Request Body**: (Identical to Forward schema above)
- **Response `200 OK`**: `best_path` (array of hidden state indices) and `best_path_log_prob`.

### `GET /api/v1/algorithms/hmm/accounts/{account_id}/timeline`
Generates a temporal state timeline for a given account across historical sequence steps.
- **Response `200 OK`**: Step-by-step timeline including amount, observed category, decoded hidden state, and anomaly indicator.

---

## 7. Supervised Models - Unit 5 (`/api/v1/algorithms/supervised`)

### `POST /api/v1/algorithms/supervised/cart/train`
Trains a balanced CART decision tree and extracts human-auditable if-then rules.
- **Request Body**: `max_depth` (int, default 5), `min_samples_split` (int, default 10), `class_weight` ("balanced").
- **Response `200 OK`**: Precision, Recall, F1, PR-AUC, ROC-AUC, feature importances, and extracted text decision rules.

### `POST /api/v1/algorithms/supervised/xgboost/train`
Trains an XGBoost gradient boosted tree with `scale_pos_weight` and `aucpr` evaluation.
- **Request Body**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`.
- **Response `200 OK`**: Full standardized metrics, training duration, and top feature importances.

### `POST /api/v1/algorithms/supervised/ensemble/train`
Constructs Soft Voting or Stacking ensembles combining CART, XGBoost, and Logistic Regression meta-learners.
- **Request Body**: `ensemble_type` ("voting" or "stacking"), `weights` (for voting).
- **Response `200 OK`**: Combined ensemble evaluation metrics on validation and test sets.

### `POST /api/v1/algorithms/supervised/compare`
Runs an apples-to-apples fair comparison of CART, XGBoost, and Ensemble on the identical test split.
- **Response `200 OK`**: Metric comparison table and execution time benchmark.

---

## 8. Explainability & Evaluation (`/api/v1/evaluation`)

### `POST /api/v1/evaluation/explain-transaction`
Calculates local SHAP feature attributions using `shap.TreeExplainer` for a specific transaction row.
- **Request Body**:
```json
{
  "transaction_index": 42,
  "model_name": "xgboost"
}
```
- **Response `200 OK`**: `base_value`, `predicted_probability`, and top feature contributions with sign and magnitude.

### `POST /api/v1/evaluation/thresholds`
Evaluates business cost tradeoff curves across 12 decision thresholds (from 0.05 to 0.95).
- **Request Body**: Cost matrix weights (`cost_fp`, `cost_fn`).
- **Response `200 OK`**: Threshold curve points with Precision, Recall, Expected Cost, and optimal cost threshold.

### `POST /api/v1/evaluation/isolation-forest`
Runs unsupervised Isolation Forest anomaly benchmark scoring.
- **Response `200 OK`**: Anomaly score distribution and top outliers.

---

## 9. Investigation Cases & Audit Trails (`/api/v1/cases`)

### `POST /api/v1/cases/`
Creates a new investigation case.
- **Request Body**:
```json
{
  "title": "Suspected Rapid Inflow Anomaly",
  "description": "High velocity burst flagged by HMM sequence monitor.",
  "priority": "HIGH",
  "assigned_analyst": "analyst_sarah",
  "initial_transactions": [
    {
      "transaction_index": 105,
      "amount": 2400.00,
      "fraud_probability": 0.88,
      "anomaly_score": -0.32,
      "added_reason": "Top SHAP contributor V14"
    }
  ]
}
```
- **Response `201 Created`**: Returns created `CaseResponse` with generated `case_number` (e.g. `CASE-20261104-A1B2C3`).

### `GET /api/v1/cases/`
Lists all cases with optional filtering and pagination.
- **Query Parameters**:
  - `status`: Filter by `NEW`, `UNDER_REVIEW`, `ESCALATED`, `CLOSED_BENIGN`, `CLOSED_SUSPICIOUS`.
  - `priority`: Filter by `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
  - `skip` (int, default 0).
  - `limit` (int, default 50).
- **Response `200 OK`**: List of cases with `notes_count` and `transactions_count`.

### `GET /api/v1/cases/{case_id}`
Returns complete case record including all notes history and linked transactions.
- **Response `200 OK`**: `CaseDetailResponse`.
- **Response `404 Not Found`**: When case ID does not exist.

### `PATCH /api/v1/cases/{case_id}/status`
Transitions a case to a new lifecycle state.
- **Request Body**:
```json
{
  "status": "ESCALATED",
  "actor": "analyst_sarah",
  "note": "Escalated due to cross-border card cloning indicators."
}
```
- **Response `200 OK`**: Updated case object. Automatically appends transition note and records an immutable audit log entry.

### `POST /api/v1/cases/{case_id}/notes`
Adds an investigation note to a case.
- **Request Body**:
```json
{
  "content": "Cardholder contacted; confirmed transaction was unauthorized.",
  "author": "analyst_sarah"
}
```
- **Response `201 Created`**: `CaseNoteResponse`.

### `POST /api/v1/cases/{case_id}/transactions`
Links an additional transaction and model outputs to an existing case.
- **Request Body**:
```json
{
  "transaction_index": 112,
  "amount": 890.50,
  "fraud_probability": 0.74,
  "anomaly_score": -0.18,
  "added_reason": "Follow-up charge within 10 minutes"
}
```
- **Response `201 Created`**: `CaseTransactionResponse`.

### `GET /api/v1/cases/audit/events`
Retrieves immutable governance audit log entries.
- **Query Parameters**:
  - `entity_type`: Filter by entity (e.g. `case`).
  - `entity_id`: Filter by entity identifier (e.g. `CASE-20261104-A1B2C3`).
  - `skip`: Pagination offset.
  - `limit`: Number of events (max 500).
- **Response `200 OK`**: List of `AuditEventResponse` items ordered newest first.

---

## 10. Regulatory Knowledge Base & RAG (`/api/v1/rag`)

### `GET /api/v1/rag/sources`
Returns all indexed public regulatory reference documents and metadata.
- **Response `200 OK`**: Array of `RagSourceDocument` objects (`doc_id`, `title`, `issuing_body`, `publication_date`, `version`, `source_url`, `chunk_count`, `sections`).

### `POST /api/v1/rag/index`
Triggers chunking and vector index rebuild of documents in `data/reference_docs/`.
- **Response `200 OK`**: Array of indexed documents with chunk statistics.

### `POST /api/v1/rag/query`
Queries the regulatory corpus using deterministic TF-IDF and cosine similarity with weak-similarity cutoffs and injection defenses.
- **Request Body**:
```json
{
  "query": "structuring cash deposits below mandatory currency reporting thresholds",
  "top_k": 3,
  "min_similarity": 0.12,
  "issuing_body": "FinCEN"
}
```
- **Response `200 OK`**:
```json
{
  "query": "structuring cash deposits below mandatory currency reporting thresholds",
  "status": "success",
  "citations": [
    {
      "chunk_id": "fincen_sar_advisory-c2",
      "doc_id": "fincen_sar_advisory",
      "title": "FinCEN Advisory on Structuring and Suspicious Activity Reporting (SAR)",
      "issuing_body": "Financial Crimes Enforcement Network (FinCEN)",
      "section": "Section 2: Red Flags for Structuring and Smurfing",
      "source_url": "https://www.fincen.gov/resources/advisories",
      "similarity_score": 0.4281,
      "text": "Structuring occurs when an individual conducts or attempts to conduct...",
      "has_injection_risk": false
    }
  ],
  "advisory_notice": "Retrieved verified regulatory citations from approved public AML/CFT reference documents."
}
```
- **Safe Fallback Response**: If similarity falls below `min_similarity`, returns:
```json
{
  "query": "photosynthesis cellular biology",
  "status": "no_relevant_source_found",
  "citations": [],
  "advisory_notice": "No relevant regulatory source found meeting the similarity threshold. AuditLens strictly adheres to evidence-based retrieval and will not invent regulatory obligations without verified citations."
}
```

---

## 11. AI Investigation Assistant (`/api/v1/assistant`)

### `POST /api/v1/assistant/generate-draft`
Generates a structured, evidence-grounded investigation report draft with claim verification.
- **Request Body**:
```json
{
  "transaction_index": 12,
  "case_id": 1,
  "analyst_hypothesis": "Observed rapid card-not-present outflow burst",
  "provider": "offline"
}
```
- **Response `200 OK`**: `InvestigationReportDraft` object with fields:
  - `report_id`: Unique identifier (e.g. `REP-A1B2C3D4`).
  - `case_id`: Optional associated case.
  - `report_title`: Descriptive header.
  - `case_summary`: High-level summary of transaction behavior and risk drivers.
  - `observed_facts`: Concrete verified transaction facts (amount, timestamp, context).
  - `model_scores`: Strict taxonomy separation (`fraud_probability`, `gmm_log_likelihood`, `hmm_decoded_state`, `kmeans_cluster_id`, `isolation_forest_score`).
  - `explanation_signals`: Top local SHAP feature attributions with direction and value.
  - `regulatory_citations`: Grounded citations retrieved from verified AML/CFT sources.
  - `missing_evidence_and_uncertainty`: Explicit checklist of unresolved operational telemetry.
  - `suggested_follow_up_questions`: Actionable follow-up inquiries for the human investigator.
  - `is_ai_generated`: `true`.
  - `mandatory_advisory_label`: Prominent human-in-the-loop review disclaimer.
  - `verification_passed`: Boolean indicating mathematical claim verification result.
  - `verification_warnings`: Array of any discrepancy warnings.
  - `generation_provider`: Provider mode (default `offline_rule_engine`).

### `POST /api/v1/assistant/verify-evidence`
Cross-checks an investigation draft's claims against ground-truth transaction telemetry and model probabilities.
- **Request Body**:
```json
{
  "draft": { ... },
  "actual_amount": 1450.50,
  "actual_fraud_probability": 0.88
}
```
- **Response `200 OK`**:
```json
{
  "verification_passed": true,
  "discrepancies_detected": 0,
  "warnings": []
}
```



