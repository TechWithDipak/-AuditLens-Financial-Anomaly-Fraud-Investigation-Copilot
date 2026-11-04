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

const API_BASE_URL =
  import.meta.env.VITE_API_URL !== undefined
    ? import.meta.env.VITE_API_URL
    : (typeof window !== 'undefined' ? '' : 'http://127.0.0.1:8000');

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
  const response = await apiClient.get<any>('/datasets/sample', {
    params: { limit, offset },
  });
  const data = response.data;
  const records = data.records || data.preview_rows || [];
  const total_records = data.total_records ?? data.quality?.row_count ?? records.length;
  const columns =
    data.columns ||
    (data.quality?.columns ? data.quality.columns.map((c: any) => c.name) : []);
  return {
    total_records,
    limit,
    offset,
    columns,
    records,
  };
};

export const getSequenceSample = async (accountId?: string): Promise<SequenceStep[]> => {
  const acc = accountId || 'ACC_0001';
  try {
    const res = await apiClient.get<any>(`/algorithms/hmm/accounts/${acc}/timeline`);
    const timeline = res.data.timeline || [];
    return timeline.map((t: any) => ({
      account_id: acc,
      step: t.step ?? 0,
      amount: t.amount ?? 0,
      merchant_category:
        t.merchant_category ?? (t.tx_type === 'PAYMENT' ? 0 : t.tx_type === 'DEBIT' ? 1 : 2),
      is_anomaly: t.is_anomaly ?? (t.is_fraud === 1 || t.inferred_state === 2 ? 1 : 0),
    }));
  } catch {
    const response = await apiClient.get<any>('/datasets/sequence-sample', {
      params: { account_id: acc },
    });
    const items = response.data.records || response.data.preview_rows || [];
    return items.map((r: any) => ({
      account_id: r.customer_id || acc,
      step: r.step ?? 0,
      amount: r.amount ?? 0,
      merchant_category:
        r.merchant_category ?? (r.tx_type === 'PAYMENT' ? 0 : r.tx_type === 'DEBIT' ? 1 : 2),
      is_anomaly: r.is_anomaly ?? (r.is_fraud === 1 ? 1 : 0),
    }));
  }
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
  const safeName = datasetPath ? datasetPath.split('/').pop() || datasetPath : 'creditcard_sample_fixture.csv';
  const response = await apiClient.post<any>('/preprocessing/eda', {
    dataset_path: datasetPath || 'data/fixtures/creditcard_sample_fixture.csv',
    dataset_name: safeName,
  });
  const data = response.data;
  let columns: Record<string, any> = data.columns || {};
  if (!data.columns && data.feature_distributions) {
    columns = {};
    for (const d of data.feature_distributions) {
      columns[d.feature] = {
        mean: d.mean,
        median: d.median,
        std: d.std,
        skewness: d.skewness,
        min: d.min,
        max: d.max,
        missing_count: 0,
      };
    }
  }
  return {
    total_rows: data.total_rows ?? data.total_records ?? 0,
    total_columns: data.total_columns ?? data.total_features ?? 0,
    columns,
    duration_hours: data.duration_hours ?? (data.time_analysis?.duration_hours ?? 0),
    hourly_transaction_rate: data.hourly_transaction_rate ?? (data.time_analysis?.hourly_rate_avg ?? 0),
    fraud_rate: data.fraud_rate ?? ((data.fraud_rate_pct ?? 0) / 100),
  };
};

export const splitDataset = async (
  splitType = 'stratified',
  testSize = 0.2,
  valSize = 0.1
): Promise<SplitResult> => {
  const response = await apiClient.post<any>('/preprocessing/split', {
    dataset_name: 'creditcard_sample_fixture.csv',
    dataset_path: 'data/fixtures/creditcard_sample_fixture.csv',
    strategy: splitType,
    split_type: splitType,
    test_size: testSize,
    val_size: valSize,
  });
  const d = response.data;
  return {
    split_type: d.strategy || d.split_type || splitType,
    train_count: d.train?.sample_count ?? d.train_count ?? 0,
    val_count: d.val?.sample_count ?? d.val_count ?? 0,
    test_count: d.test?.sample_count ?? d.test_count ?? 0,
    train_fraud_rate: d.train ? d.train.fraud_rate_pct / 100 : (d.train_fraud_rate ?? 0),
    val_fraud_rate: d.val ? d.val.fraud_rate_pct / 100 : (d.val_fraud_rate ?? 0),
    test_fraud_rate: d.test ? d.test.fraud_rate_pct / 100 : (d.test_fraud_rate ?? 0),
  };
};

// Unit 3 Algorithms
export const runKMeans = async (
  nClusters = 5,
  computeElbow = true,
  kMin = 2,
  kMax = 8
): Promise<KMeansResult> => {
  const response = await apiClient.post<any>('/algorithms/kmeans', {
    k: nClusters,
    n_clusters: nClusters,
    max_iter: 300,
    random_state: 42,
    compute_elbow: computeElbow,
    k_min: kMin,
    k_max: kMax,
  });
  const d = response.data;
  const cluster_sizes: Record<string, number> = d.cluster_sizes || {};
  if (!d.cluster_sizes && d.cluster_summaries) {
    for (const cs of d.cluster_summaries) {
      cluster_sizes[String(cs.cluster_id)] = cs.sample_count;
    }
  }
  return {
    n_clusters: d.n_clusters ?? d.k ?? nClusters,
    inertia: d.inertia ?? 0,
    silhouette_score: d.silhouette_score ?? 0,
    cluster_sizes,
    elbow_curve: d.elbow_curve || [],
  };
};

export const runGmm = async (
  nComponents = 3,
  covarianceType = 'full',
  topAnomalous = 10
): Promise<GmmResult> => {
  const response = await apiClient.post<any>('/algorithms/gmm', {
    n_components: nComponents,
    covariance_type: covarianceType,
    max_iter: 100,
    random_state: 42,
    top_anomalous_n: topAnomalous,
    unusual_percentile: 1.0,
  });
  const d = response.data;
  return {
    n_components: d.n_components ?? nComponents,
    covariance_type: d.covariance_type ?? covarianceType,
    converged: d.converged ?? true,
    lower_bound_log_likelihood: d.lower_bound_log_likelihood ?? d.unusual_threshold_log_likelihood ?? 0,
    weights: d.weights ?? Array(nComponents).fill(1 / nComponents),
    top_anomalous_indices: d.top_anomalous_indices ?? (d.unusual_indices_preview || []).map((idx: number, i: number) => ({
      index: idx,
      log_likelihood: d.log_likelihoods_preview?.[i] ?? 0,
    })),
  };
};

export const runPca = async (nComponents = 2): Promise<PcaResult> => {
  const response = await apiClient.post<any>('/algorithms/pca', {
    n_components: nComponents,
    scale_data: true,
  });
  const d = response.data;
  return {
    n_components: d.n_components ?? nComponents,
    explained_variance_ratio: d.explained_variance_ratio ?? [],
    cumulative_variance: d.cumulative_variance ?? d.cumulative_variance_ratio ?? [],
    projection_samples: d.projection_samples ?? (d.projection_preview || []).map((p: number[]) => ({
      x: p[0] ?? 0,
      y: p[1] ?? 0,
      class_label: 0,
    })),
  };
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
  const response = await apiClient.post<any>('/algorithms/supervised/compare', {});
  const d = response.data;
  let models = d.models;
  if (!models && d.comparison) {
    models = d.comparison.map((item: any) => ({
      name: item.model_name,
      metrics: {
        pr_auc: item.pr_auc,
        f1: item.f1_score,
        precision: item.precision,
        recall: item.recall,
        roc_auc: item.roc_auc,
        confusion_matrix: [[0, 0], [0, 0]],
      },
    }));
  }
  return {
    models: models || [],
  };
};

// Evaluation & Explainability
export const explainTransaction = async (
  transactionIndex: number,
  modelName = 'xgboost'
): Promise<ShapExplanationResult> => {
  const response = await apiClient.post<any>('/evaluation/explain-transaction', {
    transaction_index: transactionIndex,
    model_name: modelName,
  });
  const d = response.data;
  let features = d.features;
  if (!features && d.top_attributions) {
    features = d.top_attributions.map((attr: any) => ({
      feature: attr.feature_name,
      value: attr.feature_value,
      shap_value: attr.shap_value,
      impact: attr.contribution,
    }));
  }
  return {
    transaction_index: d.transaction_index ?? transactionIndex,
    model_name: d.model_name || modelName,
    predicted_probability: d.predicted_probability ?? 0,
    base_value: d.base_value ?? 0,
    features: features || [],
    advisory_notice: d.advisory_notice || d.caveat || '',
  };
};

export const analyzeThresholds = async (
  costFp = 10.0,
  costFn = 500.0
): Promise<ThresholdAnalysisResult> => {
  const response = await apiClient.post<any>('/evaluation/thresholds', {
    cost_fp: costFp,
    cost_fn: costFn,
  });
  const d = response.data;
  const rawPoints = d.thresholds || d.points || [];
  const thresholds = rawPoints.map((pt: any) => ({
    threshold: pt.threshold,
    precision: pt.precision,
    recall: pt.recall,
    f1: pt.f1 ?? pt.f1_score ?? 0,
    expected_cost: pt.expected_cost ?? (pt.false_positive * costFp + pt.false_negative * costFn),
    flagged_count: pt.flagged_count ?? (pt.true_positive + pt.false_positive),
  }));
  return {
    thresholds,
    optimal_f1_threshold: d.optimal_f1_threshold ?? 0.5,
    min_cost_threshold: d.min_cost_threshold ?? 0.5,
  };
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


