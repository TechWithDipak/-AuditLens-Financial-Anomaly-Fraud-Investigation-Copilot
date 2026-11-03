import React, { useState, useEffect } from 'react';
import {
  HealthResponse,
  ModelEvaluation,
  ModelMetadata,
  fetchCases,
  fetchModelEvaluation,
  fetchModelMetadata,
  fetchTransactions,
} from '../services/api';
import { UploadModal } from './UploadModal';

interface OverviewTabProps {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  onNavigateTab: (tab: string) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  health,
  loading: healthLoading,
  error: healthError,
  onNavigateTab,
}) => {
  const [totalTransactions, setTotalTransactions] = useState<number | null>(null);
  const [flaggedTransactions, setFlaggedTransactions] = useState<number | null>(null);
  const [totalCases, setTotalCases] = useState<number | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMetadata | null>(null);
  const [modelEval, setModelEval] = useState<ModelEvaluation | null>(null);
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);

  const loadMetrics = async () => {
    try {
      // 1. Transactions count
      const allTx = await fetchTransactions({ page: 1, page_size: 1 });
      setTotalTransactions(allTx.total);

      // 2. Flagged count
      const flaggedTx = await fetchTransactions({ is_flagged: true, page: 1, page_size: 1 });
      setFlaggedTransactions(flaggedTx.total);

      // 3. Cases count
      const casesData = await fetchCases(1, 1);
      setTotalCases(casesData.total);

      // 4. Model metadata
      const meta = await fetchModelMetadata();
      setModelMeta(meta);

      // 5. Evaluation benchmark
      const ev = await fetchModelEvaluation();
      setModelEval(ev);
    } catch {
      // Non-blocking for offline or starting states
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  return (
    <div>
      {/* High-Level Triage KPI Cards */}
      <div className="grid-cards">
        <div className="card">
          <div className="card-title">Backend Status</div>
          <div className="card-value">
            {healthLoading ? 'Checking...' : health ? 'ONLINE' : 'OFFLINE'}
          </div>
          <div className="card-description">
            {health ? `${health.service} (${health.environment})` : healthError || 'FastAPI backend unreachable'}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Ingested Transactions</div>
          <div className="card-value" style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            <span>{totalTransactions !== null ? totalTransactions.toLocaleString() : '...'}</span>
            {flaggedTransactions !== null && totalTransactions !== null && totalTransactions > 0 && (
              <span style={{ fontSize: '0.875rem', color: '#f87171', fontWeight: 600 }}>
                ({((flaggedTransactions / totalTransactions) * 100).toFixed(1)}% flagged)
              </span>
            )}
          </div>
          <div className="card-description">
            <button
              type="button"
              className="btn btn-outline btn-sm"
              style={{ marginTop: '0.4rem' }}
              onClick={() => onNavigateTab('transactions')}
            >
              View All Transactions &rarr;
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Investigation Cases</div>
          <div className="card-value">{totalCases !== null ? totalCases : '...'}</div>
          <div className="card-description">
            <button
              type="button"
              className="btn btn-outline btn-sm"
              style={{ marginTop: '0.4rem' }}
              onClick={() => onNavigateTab('cases')}
            >
              Manage Cases &bull; Notes &rarr;
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Active Scoring Models</div>
          <div className="card-value">
            {modelMeta?.inference_engine_ready ? 'Dual-Scoring' : 'Unloaded'}
          </div>
          <div className="card-description">
            Cutoff: <strong>50.0%</strong> &bull; {modelMeta?.feature_count || 30} features
          </div>
        </div>
      </div>

      {/* Dataset Actions Banner */}
      <div
        className="card"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '2rem',
          padding: '1.25rem 1.5rem',
          background: 'linear-gradient(90deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%)',
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.25rem' }}>
            Transaction Data Ingestion &bull; Automated Scoring
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Upload raw CSV datasets to validate schemas, calculate outlier anomaly scores, and estimate supervised fraud probabilities.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setShowUploadModal(true)}
          style={{ whiteSpace: 'nowrap' }}
        >
          &#128196; Upload Dataset (.csv)
        </button>
      </div>

      {/* Comparative Evaluation Benchmark Table */}
      {modelEval && modelEval.models_evaluated.length > 0 && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <div className="card-title" style={{ marginBottom: '0.25rem' }}>Model Evaluation &amp; Benchmark Comparison</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Evaluated on held-out test split using imbalanced PR-AUC (Average Precision) and operational review metrics.
              </div>
            </div>
            {modelEval.best_model_name && (
              <span className="status-badge pass">
                Top Candidate: {modelEval.best_model_name}
              </span>
            )}
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '0.6rem 0' }}>Model Architecture</th>
                <th style={{ padding: '0.6rem 0' }}>PR-AUC (Avg Precision)</th>
                <th style={{ padding: '0.6rem 0' }}>ROC-AUC</th>
                <th style={{ padding: '0.6rem 0' }}>F1-Score</th>
                <th style={{ padding: '0.6rem 0' }}>Precision</th>
                <th style={{ padding: '0.6rem 0' }}>Recall</th>
              </tr>
            </thead>
            <tbody>
              {modelEval.models_evaluated.map((m, idx) => {
                const metrics = m.metrics || {};
                return (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <td style={{ padding: '0.75rem 0', fontWeight: 600 }}>{m.model_name}</td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace' }}>
                      {metrics.pr_auc !== undefined ? metrics.pr_auc.toFixed(4) : 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace' }}>
                      {metrics.roc_auc !== undefined ? metrics.roc_auc.toFixed(4) : 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace' }}>
                      {metrics.f1_score !== undefined
                        ? metrics.f1_score.toFixed(4)
                        : metrics.f1_at_0_5 !== undefined
                        ? metrics.f1_at_0_5.toFixed(4)
                        : 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace' }}>
                      {metrics.precision !== undefined
                        ? metrics.precision.toFixed(4)
                        : metrics.precision_at_0_5 !== undefined
                        ? metrics.precision_at_0_5.toFixed(4)
                        : 'N/A'}
                    </td>
                    <td style={{ padding: '0.75rem 0', fontFamily: 'monospace' }}>
                      {metrics.recall !== undefined
                        ? metrics.recall.toFixed(4)
                        : metrics.recall_at_0_5 !== undefined
                        ? metrics.recall_at_0_5.toFixed(4)
                        : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Architecture Pipeline Status */}
      <div className="card">
        <div className="card-title">Architecture Pipeline Status</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem', fontSize: '0.875rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
              <th style={{ padding: '0.5rem 0' }}>Module</th>
              <th style={{ padding: '0.5rem 0' }}>Implementation Phase</th>
              <th style={{ padding: '0.5rem 0' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Project Foundation &amp; Local Dev Environment</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 1</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Dataset Acquisition &amp; Validation</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 2</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Exploratory Data Analysis &amp; Preprocessing</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 3</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Baseline Machine Learning Models</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 4</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Model Evaluation &amp; Threshold Optimization</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 5</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Explainability &amp; Analyst-Facing Evidence (SHAP)</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 6</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Backend API &amp; Case Management Persistence</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 7</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Dashboard &amp; Analyst Workflow UI</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 8</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
              <td style={{ padding: '0.75rem 0' }}>Reference Document Ingestion &amp; RAG</td>
              <td style={{ padding: '0.75rem 0' }}>Phase 9</td>
              <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onUploadSuccess={() => {
          loadMetrics();
        }}
      />
    </div>
  );
};

