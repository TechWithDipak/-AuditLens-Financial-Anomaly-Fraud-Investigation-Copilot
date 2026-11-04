export interface HealthResponse {
  status: string;
  app_name?: string;
  app?: string;
  version: string;
  environment: string;
  timestamp: string;
  python_version: string;
  services?: {
    api: string;
    database: string;
    ml_engine: string;
    llm_mode: string;
  };
}

export type ActiveTab =
  | 'overview'
  | 'workspace'
  | 'algorithms'
  | 'models'
  | 'explorer'
  | 'cases'
  | 'rag'
  | 'assistant'
  | 'settings';

// Dataset & Preprocessing Types
export interface TransactionRow {
  Time: number;
  Amount: number;
  Class: number;
  [key: string]: number;
}

export interface DatasetSampleResponse {
  total_records: number;
  limit: number;
  offset: number;
  columns: string[];
  records: TransactionRow[];
}

export interface SequenceStep {
  account_id: string;
  step: number;
  amount: number;
  merchant_category: number;
  is_anomaly: number;
}

export interface ValidationReport {
  filename: string;
  row_count: number;
  column_count: number;
  columns: string[];
  missing_values: Record<string, number>;
  duplicate_rows: number;
  class_balance: Record<string, number>;
  is_valid: boolean;
  warnings: string[];
}

export interface EdaColumnSummary {
  mean: number;
  median: number;
  std: number;
  skewness: number;
  min: number;
  max: number;
  missing_count: number;
}

export interface EdaSummary {
  total_rows: number;
  total_columns: number;
  columns: Record<string, EdaColumnSummary>;
  duration_hours: number;
  hourly_transaction_rate: number;
  fraud_rate: number;
}

export interface SplitResult {
  split_type: string;
  train_count: number;
  val_count: number;
  test_count: number;
  train_fraud_rate: number;
  val_fraud_rate: number;
  test_fraud_rate: number;
}

// Unit 3 Algorithms
export interface KMeansResult {
  n_clusters: number;
  inertia: number;
  silhouette_score: number;
  cluster_sizes: Record<string, number>;
  elbow_curve?: { k: number; inertia: number }[];
}

export interface GmmResult {
  n_components: number;
  covariance_type: string;
  converged: boolean;
  lower_bound_log_likelihood: number;
  weights: number[];
  top_anomalous_indices: { index: number; log_likelihood: number }[];
}

export interface PcaResult {
  n_components: number;
  explained_variance_ratio: number[];
  cumulative_variance: number[];
  projection_samples: { x: number; y: number; class_label: number }[];
}

// Unit 4 HMM
export interface HmmForwardResult {
  log_likelihood: number;
  total_probability: number;
  sequence_length: number;
  n_states: number;
}

export interface HmmViterbiResult {
  best_path: number[];
  best_path_log_prob: number;
  sequence_length: number;
  state_labels: string[];
}

export interface AccountTimelineEvent {
  step: number;
  amount: number;
  merchant_category: number;
  is_anomaly: number;
  decoded_state: number;
  decoded_state_name: string;
}

export interface AccountTimelineResponse {
  account_id: string;
  total_steps: number;
  anomaly_count: number;
  timeline: AccountTimelineEvent[];
}

// Unit 5 Supervised Models
export interface ModelMetrics {
  precision: number;
  recall: number;
  f1: number;
  pr_auc: number;
  roc_auc: number;
  confusion_matrix: number[][];
  train_time_sec?: number;
}

export interface CartResult {
  model_name: string;
  metrics: ModelMetrics;
  feature_importances: Record<string, number>;
  rules_text: string;
}

export interface XgboostResult {
  model_name: string;
  metrics: ModelMetrics;
  feature_importances: Record<string, number>;
}

export interface EnsembleResult {
  ensemble_type: string;
  metrics: ModelMetrics;
  weights?: Record<string, number>;
}

export interface ModelComparisonResult {
  models: {
    name: string;
    metrics: ModelMetrics;
  }[];
}

// Explainability & Thresholds
export interface ShapFeatureContribution {
  feature: string;
  value: number;
  shap_value: number;
  impact: 'increases_risk' | 'decreases_risk';
}

export interface ShapExplanationResult {
  transaction_index: number;
  model_name: string;
  predicted_probability: number;
  base_value: number;
  features: ShapFeatureContribution[];
  advisory_notice: string;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  expected_cost: number;
  flagged_count: number;
}

export interface ThresholdAnalysisResult {
  thresholds: ThresholdPoint[];
  optimal_f1_threshold: number;
  min_cost_threshold: number;
}

export interface IsolationForestResult {
  contamination: number;
  anomaly_count: number;
  anomaly_indices: number[];
  average_score: number;
}

// Cases & Audit
export interface CaseTransaction {
  id: number;
  case_id: number;
  transaction_index: number;
  amount: number;
  fraud_probability: number | null;
  anomaly_score: number | null;
  added_reason: string | null;
  created_at: string;
}

export interface CaseNote {
  id: number;
  case_id: number;
  author: string;
  content: string;
  created_at: string;
}

export interface CaseItem {
  id: number;
  case_number: string;
  title: string;
  description: string | null;
  status: 'NEW' | 'UNDER_REVIEW' | 'ESCALATED' | 'CLOSED_BENIGN' | 'CLOSED_SUSPICIOUS';
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  assigned_analyst: string;
  created_at: string;
  updated_at: string;
  notes_count: number;
  transactions_count: number;
}

export interface CaseDetail extends CaseItem {
  notes: CaseNote[];
  transactions: CaseTransaction[];
}

export interface AuditEventItem {
  id: number;
  entity_type: string;
  entity_id: string;
  action: string;
  actor: string;
  details: string | null;
  timestamp: string;
}

// Regulatory Knowledge Base (RAG)
export interface RagSourceDocument {
  doc_id: string;
  title: string;
  issuing_body: string;
  publication_date: string;
  version: string;
  source_url: string;
  file_path: string;
  chunk_count: number;
  sections: string[];
}

export interface RagCitation {
  chunk_id: string;
  doc_id: string;
  title: string;
  issuing_body: string;
  section: string;
  source_url: string;
  similarity_score: number;
  text: string;
  has_injection_risk: boolean;
}

export interface RagQueryResponse {
  query: string;
  status: 'success' | 'no_relevant_source_found';
  citations: RagCitation[];
  advisory_notice: string;
}

// AI Investigation Assistant Types
export interface ObservedFact {
  transaction_index: number;
  amount: number;
  time_seconds?: number;
  merchant_category?: string;
  context?: string;
}

export interface ModelScoreBreakdown {
  supervised_model_name: string;
  fraud_probability: number;
  gmm_log_likelihood?: number;
  kmeans_cluster_id?: number;
  hmm_decoded_state?: string;
  isolation_forest_score?: number;
}

export interface ExplanationSignal {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
}

export interface CitedPassage {
  doc_id: string;
  title: string;
  issuing_body: string;
  section: string;
  source_url: string;
  similarity_score: number;
  excerpt: string;
}

export interface InvestigationReportDraft {
  report_id: string;
  case_id?: number;
  report_title: string;
  case_summary: string;
  observed_facts: ObservedFact[];
  model_scores: ModelScoreBreakdown;
  explanation_signals: ExplanationSignal[];
  regulatory_citations: CitedPassage[];
  missing_evidence_and_uncertainty: string[];
  suggested_follow_up_questions: string[];
  is_ai_generated: boolean;
  mandatory_advisory_label: string;
  verification_passed: boolean;
  verification_warnings: string[];
  generation_provider: string;
  timestamp: string;
}


