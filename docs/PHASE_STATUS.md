# Phase 9 — Reference Document Ingestion and RAG

## Changes made
- **Regulatory Reference Documents Curation (`data/policies/`)**:
  - `manifest.json`: Document catalog registering metadata, issuing body, jurisdiction, publication date, source URL, license, and document description across 4 approved regulatory policies.
  - `FIN-2021-A003_elder_financial_exploitation.md`: FinCEN guidance on Elder Financial Exploitation (EFE), sudden asset depletions, account turnover, and SAR filing instructions.
  - `FATF_red_flag_indicators_money_laundering.md`: FATF international red flag indicators for structuring, smurfing, pass-through accounts, and velocity anomalies.
  - `FFIEC_bsa_aml_examination_manual_sars.md`: FFIEC examination standards for SAR filing timing (30 calendar days threshold), statutory non-disclosure / tipping-off prohibition (31 U.S.C. 5318(g)(2)), and 5-year record retention.
  - `FIN-2020-A001_ransomware_cyber_extortion.md`: FinCEN advisory on ransomware indicators, convertible virtual currency (CVC) wallet hops, mixing services, and OFAC sanctions nexus.
- **RAG Architecture & Document Processing Package (`rag/`)**:
  - `rag/schemas.py`: Defined Pydantic models for `DocumentMetadata`, `DocumentChunk`, `RetrievedPassage`, `PolicySearchRequest`, `PolicySearchResponse`, and `DocumentCatalogResponse`.
  - `rag/sanitizer.py`: Security layer treating document text as untrusted passive data; defangs prompt-injection patterns (`ignore previous instructions`, `system:`, `[INST]`, `developer mode`), strips control characters, and blocks executable `<script>`/`<iframe>` tags.
  - `rag/chunker.py`: Document parser splitting Markdown/plain-text by section headings and paragraphs into bounded passages (target 400 chars, ~60 chars overlap) with metadata inheritance and injection sanitization.
  - `rag/retriever.py`: Implemented `AuditLensRetriever` utilizing Scikit-Learn `TfidfVectorizer` (sublinear term frequency, (1, 2) n-grams) and Cosine Similarity, disk persistence in `data/vector_store/`, configurable `top_k`, issuing body filtering, and explicit "no relevant source found" fallback handling.
- **Backend API Integration (`backend/app/api/v1/policy.py`)**:
  - `POST /api/v1/policy/search`: Semantic passage retrieval with query validation, optional authority filtering, similarity threshold cutoff, latency calculation, and mandatory non-determination disclaimer.
  - `GET /api/v1/policy/documents`: Approved public reference document catalog endpoint.
  - `POST /api/v1/policy/reindex`: Administrative re-indexing endpoint to rebuild vector store from `data/policies/`.
  - Service abstraction in `backend/app/services/policy_service.py`.
  - Registered router in `backend/app/main.py`.
- **Frontend Policy References View (`frontend/src/components/PolicyReferencesTab.tsx`)**:
  - Interactive search bar with search inputs, issuing body filter dropdown (`All`, `FinCEN`, `FATF`, `FFIEC`), and top-k selector.
  - Quick query suggestion chips for common AML patterns (Elder Financial Exploitation, Structuring & Smurfing, SAR 30-Day Deadline, Ransomware & Crypto).
  - Traceable passage cards displaying Document Title, Issuing Agency badge, Section name, Publication date, Relevance score meter (% match), Sanitized snippet, and Authoritative external URL link.
  - Mandatory regulatory decision-support advisory disclaimer banner.
  - Clean "No Relevant Regulatory Source Found" empty state preventing fabricated citations.
  - Collapsible reference document library catalog showing all 4 indexed source documents with "Re-index Documents" action.
  - Connected tab in `frontend/src/App.tsx` and updated brand badge in `frontend/src/components/Navbar.tsx`.
- **Testing & Quality Assurance**:
  - 21 new automated Python tests across `tests/rag/test_sanitizer.py`, `tests/rag/test_chunker.py`, `tests/rag/test_retriever.py`, and `tests/backend/test_policy_api.py`.
  - 2 new frontend tests in `frontend/test/workflow.test.js` validating RAG traceability contracts, prompt-injection safety, and no-hallucination empty states.
  - Full suite: 114 backend & ML tests passing; 8 frontend tests passing; 0 lint errors; Vite production build succeeds cleanly.

## Files created or modified
- `data/policies/manifest.json`
- `data/policies/FIN-2021-A003_elder_financial_exploitation.md`
- `data/policies/FATF_red_flag_indicators_money_laundering.md`
- `data/policies/FFIEC_bsa_aml_examination_manual_sars.md`
- `data/policies/FIN-2020-A001_ransomware_cyber_extortion.md`
- `rag/__init__.py`
- `rag/schemas.py`
- `rag/sanitizer.py`
- `rag/chunker.py`
- `rag/retriever.py`
- `backend/app/services/policy_service.py`
- `backend/app/api/v1/policy.py`
- `backend/app/main.py`
- `frontend/src/services/api.ts`
- `frontend/src/components/PolicyReferencesTab.tsx`
- `frontend/src/components/OverviewTab.tsx`
- `frontend/src/components/Navbar.tsx`
- `frontend/src/App.tsx`
- `frontend/test/workflow.test.js`
- `tests/rag/test_sanitizer.py`
- `tests/rag/test_chunker.py`
- `tests/rag/test_retriever.py`
- `tests/backend/test_policy_api.py`
- `docs/PHASE_STATUS.md`

## Verification performed
| Check | Command / Method | Result |
|---|---|---|
| Full Backend, ML & RAG Test Suite | `.venv/bin/pytest -v` | **PASS** (114 passed in 3.02s) |
| Backend & RAG Linting | `.venv/bin/ruff check .` | **PASS** (All checks passed, 0 errors) |
| Frontend Test Suite | `cd frontend && npm test` | **PASS** (8 passed in 75ms) |
| Frontend TypeScript Verification | `cd frontend && npm run lint` | **PASS** (`tsc --noEmit` clean, 0 errors) |
| Frontend Production Build | `cd frontend && npm run build` | **PASS** (Vite built in 270ms) |
| Prompt-Injection Neutralization | `tests/rag/test_sanitizer.py` | **PASS** (Defangs instructions & scripts) |
| Citation Traceability & Provenance | `tests/rag/test_retriever.py` | **PASS** (Every passage traces to valid document, section, and URL) |
| Non-Hallucination on Irrelevant Query | `test_retrieval_irrelevant_query_returns_no_results` | **PASS** (Returns `has_results=False`, empty list) |
| Policy REST Endpoints | `tests/backend/test_policy_api.py` | **PASS** (Search, documents, reindex verified) |

## Issues and limitations
- RAG uses deterministic local TF-IDF + Cosine Similarity (per ADR-006) which requires zero external API keys and runs completely offline. Dense neural embeddings can be plugged in when configured.
- Investigation Report Drafting (LLM) is scheduled for Phase 10.

## Exit gate
**PASS**

## Next phase
Phase 10 — LLM Investigation Assistant and Report Generation.
Do not begin until the user explicitly approves.
