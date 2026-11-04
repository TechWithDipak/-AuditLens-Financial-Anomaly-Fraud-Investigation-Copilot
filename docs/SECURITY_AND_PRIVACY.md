# AuditLens Security, Privacy, and Failure-Mode Architecture

## 1. Executive Summary & Security Principles
AuditLens is designed as a locally runnable, privacy-preserving financial anomaly detection and regulatory copilot. Financial investigation software handles highly sensitive data—including transaction timestamps, monetary volumes, anonymized behavioral vectors ($V_1$ through $V_{28}$), and investigative notes.

To guarantee zero data exfiltration, AuditLens enforces five core security principles:
1. **Local-First / Zero Mandatory Cloud Dependencies:** Core feature extraction, unsupervised clustering (K-Means, GMM, PCA), sequential modeling (Forward/Viterbi HMM), supervised prediction (CART, XGBoost, Ensembles), local SHAP explainability, and regulatory RAG retrieval run entirely on local offline CPU/memory without contacting third-party APIs.
2. **Strict Score Taxonomy Separation:** Model outputs are strictly partitioned into distinct semantic domains (`fraud_probability`, `gmm_log_likelihood`, `kmeans_cluster_id`, `isolation_forest_score`, `hmm_decoded_state`). Uncalibrated anomaly scores are never conflated with supervised fraud probabilities.
3. **Defense-in-Depth Against Adversarial Payloads:** Multi-layer regex pattern sanitization detects and neutralizes prompt injection, jailbreak attempts, and path traversal across user notes, hypotheses, and regulatory documents.
4. **Least-Privilege Role-Based Access Control (RBAC):** API endpoints enforce explicit investigator permissions (`VIEWER`, `ANALYST`, `LEAD_INVESTIGATOR`, `ADMIN`).
5. **Zero PII/PAN Logging:** Sensitive payment instrument identifiers and raw transaction arrays are excluded from standard logging pipelines.

---

## 2. Threat Model (STRIDE Assessment)

| Threat Category | Potential Attack Vector | AuditLens Countermeasure & Mitigation |
| :--- | :--- | :--- |
| **Spoofing** | Unauthorized user assumes lead investigator privileges to dismiss SAR alerts. | Strict `X-Analyst-Role` header validation with role hierarchy enforcement; actions logged to immutable `audit_events` ledger with actor identity. |
| **Tampering** | Modifying closed investigation cases or tampering with training fixtures. | Append-only database schemas for notes and audit events; SHA-256 hash checks and fixture read-only safeguards. |
| **Repudiation** | Analyst denies filing or escalating a suspicious transaction case. | `AuditEvent` model captures immutable record of all state transitions, including `timestamp`, `actor`, `entity_type`, and `details`. |
| **Information Disclosure** | Memory dumps or logs exposing cardholder Primary Account Numbers (PAN) or transaction amounts. | `mask_sensitive_identifier` masks strings revealing only last 4 digits; logging pipelines record only aggregate counts and execution statuses. |
| **Denial of Service (DoS)** | Uploading multi-gigabyte files to exhaust backend server RAM. | Bounded streaming reads (`MAX_UPLOAD_BYTES = 100MB`); immediate HTTP 413 rejection; safe in-memory data profiling. |
| **Elevation of Privilege** | Read-only viewer invoking model retrain or case creation endpoints. | Endpoint level FastAPI `require_role(min_role)` dependency enforcing HTTP 403 Forbidden for unauthorized tiers. |

---

## 3. Role-Based Access Control (RBAC) Architecture

AuditLens defines four hierarchical investigator roles:

```
[VIEWER (1)]  --> Read-only access to datasets, model labs, case queues, and audit trails.
     |
[ANALYST (2)] --> Default developer/investigator mode: create cases, link flagged transactions, add notes, transition to UNDER_REVIEW or CLOSED_BENIGN.
     |
[LEAD_INVESTIGATOR (3)] --> Authorized to escalate cases or file CLOSED_SUSPICIOUS (SAR filing recommendation).
     |
[ADMIN (4)]   --> System administrator: trigger bulk RAG vector re-indexing, schema maintenance, and demo database reset.
```

### Authorization Enforcement
Endpoints in `app.api.v1.endpoints.cases` and `app.api.v1.endpoints.rag` inject the dependency:
```python
current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST))
```
If an HTTP client provides an insufficient role (e.g., `X-Analyst-Role: viewer` on a `POST /cases` request), the server immediately returns:
```json
{
  "detail": "Access forbidden: Operation requires at least 'analyst' role (current caller role: 'viewer')."
}
```
If no role header is supplied, AuditLens defaults to `AnalystRole.ANALYST` to maintain seamless local developer and desktop usage.

---

## 4. Input Sanitization & Path Traversal Prevention

1. **Filename Sanitization:**
   `sanitize_filename(filename)` validates all incoming file paths:
   - Strips directory components using `Path(filename).name`.
   - Rejects poison null bytes (`\0`) with HTTP 400.
   - Enforces regex `^[a-zA-Z0-9_.\-]+$` to block shell injection characters (`;`, `&`, `|`, `<`, `>`).
2. **File Upload Bounds:**
   - Chunked reading streams at most `MAX_UPLOAD_BYTES + 1024` bytes.
   - Files exceeding 100MB trigger HTTP 413 Content Too Large before memory exhaustion.
   - Strict extension check (`.csv`) prevents executable uploads.
3. **Prompt Injection & Adversarial Defense:**
   - Evaluated in `scan_for_injection_risk` and `sanitize_text`.
   - Neutralizes phrases such as `system override`, `disregard prior rules`, `ignore all instructions`, replacing offending substrings with `[BLOCKED_INJECTION_PAYLOAD]`.

---

## 5. Failure Modes & Graceful Degradation

| Failure Mode | Root Cause | System Behavior & Graceful Fallback |
| :--- | :--- | :--- |
| **Corrupted RAG Index** | Missing or corrupted `rag_index.joblib`. | `VectorIndex.load()` catches deserialization exceptions, sets `is_indexed = False`, and query returns `"no_relevant_source_found"` without crashing the server. |
| **External LLM Outage** | Network timeout or invalid API keys. | `ExternalLLMAdapter` transparently falls back to `OfflineRuleBasedAdapter` with zero user disruption. |
| **Database Disconnect** | SQLite lock contention or file deletion. | Live health check (`GET /api/v1/health`) flags `"database": "degraded_unreachable"` and overall status `"degraded"`. |
| **Out-of-Bounds Row Query** | Analyst requests nonexistent transaction ID. | Handled with HTTP 404 / `IndexError` rather than unhandled server exception. |

---

## 6. Data Retention & Local Reset

- **SQLite WAL Database:** Stored at `backend/data/auditlens.db`.
- **Resetting Demo State:** To reset the application database to clean state, run:
  ```bash
  rm backend/data/auditlens.db
  .venv/Scripts/alembic -c backend/alembic.ini upgrade head
  ```
- **RAG Index Regeneration:** Vector cache at `artifacts/rag_index.joblib` can be rebuilt on-demand via `POST /api/v1/rag/index`.

