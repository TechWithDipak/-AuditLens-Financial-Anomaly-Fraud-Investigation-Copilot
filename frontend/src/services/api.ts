/**
 * AuditLens Frontend API Service Client
 * Fully typed integration with FastAPI backend endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface Transaction {
  id: number;
  transaction_id: string;
  time: number;
  amount: number;
  features: Record<string, number>;
  anomaly_score: number | null;
  fraud_probability: number | null;
  is_flagged: boolean;
  score_type: string;
  label: number | null;
  created_at: string;
}

export interface TransactionListResponse {
  items: Transaction[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface TransactionFilters {
  page?: number;
  page_size?: number;
  min_amount?: number;
  max_amount?: number;
  is_flagged?: boolean;
  min_score?: number;
  max_score?: number;
  score_type?: string;
}

export interface FeatureContribution {
  feature_name: string;
  transformed_value: number;
  contribution: number;
  direction: 'increases_risk' | 'decreases_risk' | 'neutral';
  raw_value?: number | null;
  relative_magnitude: number;
  plain_text_description: string;
}

export interface TransactionExplanation {
  transaction_id: string;
  model_name: string;
  model_version: string;
  explanation_method: string;
  score_type: string;
  prediction_score: number;
  decision_threshold: number;
  is_flagged: boolean;
  base_value: number;
  top_contributions: FeatureContribution[];
  all_contributions: Record<string, number>;
  summary_text: string;
  warnings: string[];
  limitations: string[];
  timestamp: string;
}

export interface CaseNote {
  id: number;
  case_id: number;
  author: string;
  content: string;
  created_at: string;
}

export type CaseStatus =
  | 'NEW'
  | 'IN_REVIEW'
  | 'ESCALATED'
  | 'CLOSED_FALSE_POSITIVE'
  | 'CLOSED_CONFIRMED_FRAUD';

export type CasePriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Case {
  id: number;
  case_id: string;
  title: string;
  description: string;
  status: CaseStatus;
  priority: CasePriority;
  assigned_to: string | null;
  transaction_ids: string[];
  notes: CaseNote[];
  created_at: string;
  updated_at: string;
}

export interface CaseListResponse {
  items: Case[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CaseCreatePayload {
  title: string;
  description?: string;
  priority: CasePriority;
  transaction_ids?: string[];
  assigned_to?: string;
}

export interface CaseUpdatePayload {
  title?: string;
  description?: string;
  status?: CaseStatus;
  priority?: CasePriority;
  assigned_to?: string;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  model_type: string;
  description: string;
  features: string[];
  hyperparameters: Record<string, unknown>;
}

export interface DocumentMetadata {
  id: string;
  title: string;
  issuing_body: string;
  jurisdiction: string;
  publication_date: string;
  filename: string;
  source_url: string;
  license: string;
  description: string;
}

export interface RetrievedPassage {
  chunk_id: string;
  document_id: string;
  document_title: string;
  issuing_body: string;
  section: string;
  source_url: string;
  publication_date: string;
  text: string;
  similarity_score: number;
}

export interface PolicySearchRequest {
  query: string;
  issuing_body?: string;
  top_k?: number;
  min_similarity?: number;
}

export interface PolicySearchResponse {
  query: string;
  total_retrieved: number;
  passages: RetrievedPassage[];
  has_results: boolean;
  execution_time_ms: number;
  disclaimer: string;
}

export interface DocumentCatalogResponse {
  version: string;
  total_documents: number;
  documents: DocumentMetadata[];
}

export interface ModelMetadata {
  inference_engine_ready: boolean;
  operational_threshold: number;
  feature_count: number;
  models: Record<string, ModelInfo>;
  notice: string;
}

export interface ModelEvaluation {
  generated_at?: string;
  best_model_name?: string;
  operational_threshold?: number;
  models_evaluated: Array<{
    model_name: string;
    model_version?: string;
    metrics: Record<string, number>;
  }>;
  dataset_summary: Record<string, unknown>;
  limitations: string[];
}

export interface UploadResult {
  filename: string;
  size_bytes: number;
  row_count: number;
  column_count: number;
  duplicate_count: number;
  ingested_count: number;
  summary: Record<string, unknown>;
  warnings: string[];
}

// ----------------------------------------------------------------------
// API Methods
// ----------------------------------------------------------------------

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errMsg = `Request failed (${res.status} ${res.statusText})`;
    try {
      const errorJson = await res.json();
      if (errorJson.error?.message) {
        errMsg = errorJson.error.message;
      } else if (errorJson.detail) {
        errMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errMsg);
  }
  return res.json();
}

/**
 * Fetch health status from backend
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  return handleResponse<HealthResponse>(res);
}

/**
 * Fetch paginated & filtered transactions
 */
export async function fetchTransactions(filters: TransactionFilters = {}): Promise<TransactionListResponse> {
  const params = new URLSearchParams();
  if (filters.page) params.set('page', String(filters.page));
  if (filters.page_size) params.set('page_size', String(filters.page_size));
  if (filters.min_amount !== undefined) params.set('min_amount', String(filters.min_amount));
  if (filters.max_amount !== undefined) params.set('max_amount', String(filters.max_amount));
  if (filters.is_flagged !== undefined) params.set('is_flagged', String(filters.is_flagged));
  if (filters.min_score !== undefined) params.set('min_score', String(filters.min_score));
  if (filters.max_score !== undefined) params.set('max_score', String(filters.max_score));
  if (filters.score_type) params.set('score_type', filters.score_type);

  const res = await fetch(`${API_BASE_URL}/api/v1/transactions?${params.toString()}`);
  return handleResponse<TransactionListResponse>(res);
}

/**
 * Fetch single transaction
 */
export async function fetchTransaction(idOrTxId: string | number): Promise<Transaction> {
  const res = await fetch(`${API_BASE_URL}/api/v1/transactions/${idOrTxId}`);
  return handleResponse<Transaction>(res);
}

/**
 * Fetch transaction SHAP/fallback explanation
 */
export async function fetchTransactionExplanation(idOrTxId: string | number): Promise<TransactionExplanation> {
  const res = await fetch(`${API_BASE_URL}/api/v1/transactions/${idOrTxId}/explanation`);
  return handleResponse<TransactionExplanation>(res);
}

/**
 * Upload CSV file with optional DB ingestion
 */
export async function uploadDataset(
  file: File,
  isTraining: boolean = false,
  ingestToDb: boolean = true
): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('is_training', String(isTraining));
  formData.append('ingest_to_db', String(ingestToDb));

  const res = await fetch(`${API_BASE_URL}/api/v1/data/upload`, {
    method: 'POST',
    body: formData,
  });

  const body = await handleResponse<{ success: boolean; data: UploadResult }>(res);
  return body.data;
}

/**
 * Fetch investigation cases
 */
export async function fetchCases(
  page: number = 1,
  pageSize: number = 20,
  status?: CaseStatus,
  priority?: CasePriority
): Promise<CaseListResponse> {
  const params = new URLSearchParams();
  params.set('page', String(page));
  params.set('page_size', String(pageSize));
  if (status) params.set('status', status);
  if (priority) params.set('priority', priority);

  const res = await fetch(`${API_BASE_URL}/api/v1/cases?${params.toString()}`);
  return handleResponse<CaseListResponse>(res);
}

/**
 * Fetch single case detail
 */
export async function fetchCase(caseIdentifier: string): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cases/${caseIdentifier}`);
  return handleResponse<Case>(res);
}

/**
 * Create a new case
 */
export async function createCase(payload: CaseCreatePayload): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<Case>(res);
}

/**
 * Update case status or metadata
 */
export async function updateCase(caseIdentifier: string, payload: CaseUpdatePayload): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cases/${caseIdentifier}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<Case>(res);
}

/**
 * Add note to case
 */
export async function addCaseNote(caseIdentifier: string, author: string, content: string): Promise<CaseNote> {
  const res = await fetch(`${API_BASE_URL}/api/v1/cases/${caseIdentifier}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ author, content }),
  });
  return handleResponse<CaseNote>(res);
}

/**
 * Fetch model metadata & operational thresholds
 */
export async function fetchModelMetadata(): Promise<ModelMetadata> {
  const res = await fetch(`${API_BASE_URL}/api/v1/models/metadata`);
  return handleResponse<ModelMetadata>(res);
}

/**
 * Fetch comparative model evaluation report
 */
export async function fetchModelEvaluation(): Promise<ModelEvaluation> {
  const res = await fetch(`${API_BASE_URL}/api/v1/models/evaluation`);
  return handleResponse<ModelEvaluation>(res);
}

/**
 * Perform semantic search on regulatory policies
 */
export async function searchPolicies(payload: PolicySearchRequest): Promise<PolicySearchResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/policy/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<PolicySearchResponse>(res);
}

/**
 * Fetch catalog of approved regulatory reference documents
 */
export async function fetchPolicyCatalog(): Promise<DocumentCatalogResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/policy/documents`);
  return handleResponse<DocumentCatalogResponse>(res);
}

/**
 * Trigger re-indexing of policy documents
 */
export async function reindexPolicies(): Promise<{ status: string; message: string; indexed_chunks: number }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/policy/reindex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  return handleResponse<{ status: string; message: string; indexed_chunks: number }>(res);
}
