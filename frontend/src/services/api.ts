import axios from 'axios';
import {
  AccountTimelineResponse,
  AuditEventItem,
  CartResult,
  CaseDetail,
  CaseItem,
  CaseNote,
  CaseTransaction,
  DatasetSampleResponse,
  EdaSummary,
  EnsembleResult,
  GmmResult,
  HealthResponse,
  HmmForwardResult,
  HmmViterbiResult,
  InvestigationReportDraft,
  IsolationForestResult,
  KMeansResult,
  ModelComparisonResult,
  PcaResult,
  RagQueryResponse,
  RagSourceDocument,
  SequenceStep,
  ShapExplanationResult,
  SplitResult,
  ThresholdAnalysisResult,
  ValidationReport,
  XgboostResult,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 60000, // 60s for ML model training
  headers: {
    'Content-Type': 'application/json',
  },
});

// System Health
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

// Datasets
export const getSampleTransactions = async (
  limit = 100,
  offset = 0
): Promise<DatasetSampleResponse> => {
  const response = await apiClient.get<DatasetSampleResponse>('/datasets/sample', {
    params: { limit, offset },
  });
  return response.data;
};

export const getSequenceSample = async (accountId?: string): Promise<SequenceStep[]> => {
  const response = await apiClient.get<SequenceStep[]>('/datasets/sequence-sample', {
    params: accountId ? { account_id: accountId } : {},
  });
  return response.data;
};

export const validateDatasetUpload = async (file: File): Promise<ValidationReport> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<ValidationReport>('/datasets/validate-upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

// Preprocessing
export const runEda = async (datasetPath?: string): Promise<EdaSummary> => {
  const response = await apiClient.post<EdaSummary>('/preprocessing/eda', {
    dataset_path: datasetPath || 'data/fixtures/creditcard_sample_fixture.csv',
  });
  return response.data;
};

export const splitDataset = async (
  splitType = 'stratified',
  testSize = 0.2,
  valSize = 0.1
): Promise<SplitResult> => {
  const response = await apiClient.post<SplitResult>('/preprocessing/split', {
    dataset_path: 'data/fixtures/creditcard_sample_fixture.csv',
    split_type: splitType,
    test_size: testSize,
    val_size: valSize,
  });
  return response.data;
};

// Unit 3 Algorithms
export const runKMeans = async (
  nClusters = 5,
  computeElbow = true,
  kMin = 2,
  kMax = 8
): Promise<KMeansResult> => {
  const response = await apiClient.post<KMeansResult>('/algorithms/kmeans', {
    n_clusters: nClusters,
    max_iter: 300,
    random_state: 42,
    compute_elbow: computeElbow,
    k_min: kMin,
    k_max: kMax,
  });
  return response.data;
};

export const runGmm = async (
  nComponents = 3,
  covarianceType = 'full',
  topAnomalous = 10
): Promise<GmmResult> => {
  const response = await apiClient.post<GmmResult>('/algorithms/gmm', {
    n_components: nComponents,
    covariance_type: covarianceType,
    max_iter: 100,
    random_state: 42,
    top_anomalous_n: topAnomalous,
  });
  return response.data;
};

export const runPca = async (nComponents = 2): Promise<PcaResult> => {
  const response = await apiClient.post<PcaResult>('/algorithms/pca', {
    n_components: nComponents,
    scale_data: true,
  });
  return response.data;
};

// Unit 4 HMM
export const runHmmForward = async (observations: number[]): Promise<HmmForwardResult> => {
  const response = await apiClient.post<HmmForwardResult>('/algorithms/hmm/forward', {
    observations,
  });
  return response.data;
};

export const runHmmViterbi = async (observations: number[]): Promise<HmmViterbiResult> => {
  const response = await apiClient.post<HmmViterbiResult>('/algorithms/hmm/viterbi', {
    observations,
  });
  return response.data;
};

export const getAccountTimeline = async (accountId: string): Promise<AccountTimelineResponse> => {
  const response = await apiClient.get<AccountTimelineResponse>(
    `/algorithms/hmm/accounts/${accountId}/timeline`
  );
  return response.data;
};

// Unit 5 Supervised Models
export const trainCart = async (
  maxDepth = 5,
  minSamplesSplit = 10,
  classWeight = 'balanced'
): Promise<CartResult> => {
  const response = await apiClient.post<CartResult>('/algorithms/supervised/cart/train', {
    max_depth: maxDepth,
    min_samples_split: minSamplesSplit,
    class_weight: classWeight,
  });
  return response.data;
};

export const trainXgboost = async (
  nEstimators = 50,
  maxDepth = 4,
  learningRate = 0.1
): Promise<XgboostResult> => {
  const response = await apiClient.post<XgboostResult>('/algorithms/supervised/xgboost/train', {
    n_estimators: nEstimators,
    max_depth: maxDepth,
    learning_rate: learningRate,
    subsample: 0.8,
  });
  return response.data;
};

export const trainEnsemble = async (ensembleType = 'voting'): Promise<EnsembleResult> => {
  const response = await apiClient.post<EnsembleResult>('/algorithms/supervised/ensemble/train', {
    ensemble_type: ensembleType,
  });
  return response.data;
};

export const compareModels = async (): Promise<ModelComparisonResult> => {
  const response = await apiClient.post<ModelComparisonResult>('/algorithms/supervised/compare', {});
  return response.data;
};

// Evaluation & Explainability
export const explainTransaction = async (
  transactionIndex: number,
  modelName = 'xgboost'
): Promise<ShapExplanationResult> => {
  const response = await apiClient.post<ShapExplanationResult>('/evaluation/explain-transaction', {
    transaction_index: transactionIndex,
    model_name: modelName,
  });
  return response.data;
};

export const analyzeThresholds = async (
  costFp = 10.0,
  costFn = 500.0
): Promise<ThresholdAnalysisResult> => {
  const response = await apiClient.post<ThresholdAnalysisResult>('/evaluation/thresholds', {
    cost_fp: costFp,
    cost_fn: costFn,
  });
  return response.data;
};

export const runIsolationForest = async (contamination = 0.02): Promise<IsolationForestResult> => {
  const response = await apiClient.post<IsolationForestResult>('/evaluation/isolation-forest', {
    contamination,
  });
  return response.data;
};

// Cases & Audit
export const listCases = async (status?: string, priority?: string): Promise<CaseItem[]> => {
  const response = await apiClient.get<CaseItem[]>('/cases/', {
    params: {
      status: status || undefined,
      priority: priority || undefined,
    },
  });
  return response.data;
};

export const getCaseDetail = async (caseId: number): Promise<CaseDetail> => {
  const response = await apiClient.get<CaseDetail>(`/cases/${caseId}`);
  return response.data;
};

export interface CreateCasePayload {
  title: string;
  description?: string;
  priority?: string;
  assigned_analyst?: string;
  initial_transactions?: {
    transaction_index: number;
    amount: number;
    fraud_probability?: number;
    anomaly_score?: number;
    added_reason?: string;
  }[];
}

export const createCase = async (payload: CreateCasePayload): Promise<CaseItem> => {
  const response = await apiClient.post<CaseItem>('/cases/', payload);
  return response.data;
};

export const updateCaseStatus = async (
  caseId: number,
  status: string,
  note?: string,
  actor = 'analyst'
): Promise<CaseItem> => {
  const response = await apiClient.patch<CaseItem>(`/cases/${caseId}/status`, {
    status,
    note,
    actor,
  });
  return response.data;
};

export const addCaseNote = async (
  caseId: number,
  content: string,
  author = 'analyst'
): Promise<CaseNote> => {
  const response = await apiClient.post<CaseNote>(`/cases/${caseId}/notes`, {
    content,
    author,
  });
  return response.data;
};

export const linkCaseTransaction = async (
  caseId: number,
  transaction: {
    transaction_index: number;
    amount: number;
    fraud_probability?: number;
    anomaly_score?: number;
    added_reason?: string;
  },
  actor = 'analyst'
): Promise<CaseTransaction> => {
  const response = await apiClient.post<CaseTransaction>(
    `/cases/${caseId}/transactions`,
    transaction,
    { params: { actor } }
  );
  return response.data;
};

export const listAuditEvents = async (
  entityType?: string,
  entityId?: string
): Promise<AuditEventItem[]> => {
  const response = await apiClient.get<AuditEventItem[]>('/cases/audit/events', {
    params: {
      entity_type: entityType || undefined,
      entity_id: entityId || undefined,
    },
  });
  return response.data;
};

// Regulatory Knowledge Base (RAG)
export const listRagSources = async (): Promise<RagSourceDocument[]> => {
  const response = await apiClient.get<RagSourceDocument[]>('/rag/sources');
  return response.data;
};

export const rebuildRagIndex = async (): Promise<RagSourceDocument[]> => {
  const response = await apiClient.post<RagSourceDocument[]>('/rag/index');
  return response.data;
};

export const queryRag = async (
  query: string,
  topK = 3,
  minSimilarity = 0.15,
  issuingBody?: string
): Promise<RagQueryResponse> => {
  const response = await apiClient.post<RagQueryResponse>('/rag/query', {
    query,
    top_k: topK,
    min_similarity: minSimilarity,
    issuing_body: issuingBody || undefined,
  });
  return response.data;
};

// AI Investigation Assistant
export const generateInvestigationDraft = async (
  transactionIndex: number,
  caseId?: number,
  hypothesis?: string,
  provider = 'offline'
): Promise<InvestigationReportDraft> => {
  const response = await apiClient.post<InvestigationReportDraft>('/assistant/generate-draft', {
    transaction_index: transactionIndex,
    case_id: caseId || undefined,
    analyst_hypothesis: hypothesis || undefined,
    provider,
  });
  return response.data;
};

export const verifyReportEvidence = async (
  draft: InvestigationReportDraft,
  actualAmount: number,
  actualProbability: number
): Promise<{ verification_passed: boolean; discrepancies_detected: number; warnings: string[] }> => {
  const response = await apiClient.post('/assistant/verify-evidence', {
    draft,
    actual_amount: actualAmount,
    actual_fraud_probability: actualProbability,
  });
  return response.data;
};


