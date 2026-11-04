# AuditLens RAG Reference Sources Registry

This catalog documents the approved, public reference documents indexed in the AuditLens Regulatory Knowledge Base (RAG). Every retrieved chunk is tied back to this registry.

---

## 1. Approved Document Registry

| Document ID | Title | Issuing Body | Publication Date | Version | Permitted Use | File Path |
|---|---|---|---|---|---|---|
| `fatf_rba_2014` | FATF Guidance on the Risk-Based Approach for the Banking Sector | Financial Action Task Force (FATF) | 2014-10-27 | 1.0 | Public regulatory guidance / Educational fair use | `data/reference_docs/fatf_risk_based_guidance.md` |
| `fincen_sar_2021` | FinCEN Advisory on Structuring and Suspicious Activity Reporting (SAR) | Financial Crimes Enforcement Network (FinCEN) | 2021-03-12 | 2.1 | Public government advisory | `data/reference_docs/fincen_sar_advisory.md` |
| `eba_psd2_2020` | European Banking Authority Guidelines on Fraud Reporting under PSD2 | European Banking Authority (EBA) | 2020-01-22 | 1.2 | Public regulatory standard | `data/reference_docs/eba_guidelines_fraud_reporting.md` |

---

## 2. Ingestion & Indexing Policy

1. **Approved Sources Only:** The knowledge base indexes only verified public AML, CFT, and payment fraud standards. Unverified or proprietary third-party text is rejected.
2. **Deterministic Chunking:** Documents are split at markdown section headers (`## Section X`) and further chunked with maximum 1,200 character sliding windows with 200 character overlap to ensure semantic coherence.
3. **Structured Metadata:** Each indexed chunk carries:
   - `doc_id`: Unique document identifier
   - `title`: Formal publication title
   - `issuing_body`: Regulatory authority
   - `section`: Specific section name/number
   - `source_url`: Public reference URL
   - `chunk_id`: Unique chunk locator
4. **Citation Grounding:** The retrieval engine returns verbatim citations with similarity scores. If cosine similarity between the query and top candidate chunk is below the minimum threshold ($\theta_{sim} < 0.15$), the system returns `"no_relevant_source_found"`.
5. **Adversarial Untrusted Text Handling:** Retrieved chunks are treated as untrusted data. Any instructions embedded within chunks attempting to override system behavior (e.g., `"ignore previous instructions"`, `"system override"`, jailbreaks) are scrubbed and flagged with an injection risk alert.

