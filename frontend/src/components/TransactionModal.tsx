import React, { useEffect, useState } from 'react';
import { Transaction, TransactionExplanation, fetchTransactionExplanation } from '../services/api';

interface TransactionModalProps {
  transaction: Transaction | null;
  onClose: () => void;
  onCreateCase?: (transactionId: string) => void;
}

export const TransactionModal: React.FC<TransactionModalProps> = ({
  transaction,
  onClose,
  onCreateCase,
}) => {
  const [explanation, setExplanation] = useState<TransactionExplanation | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!transaction) {
      setExplanation(null);
      return;
    }

    let isMounted = true;
    async function loadExplanation() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchTransactionExplanation(transaction!.transaction_id);
        if (isMounted) setExplanation(data);
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load explanation');
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadExplanation();
    return () => {
      isMounted = false;
    };
  }, [transaction]);

  if (!transaction) return null;

  const fraudProb = transaction.fraud_probability ?? 0.0;
  const anomalyScore = transaction.anomaly_score ?? 0.0;
  const isFlagged = transaction.is_flagged;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="tx-detail-title">
      <div className="modal-content" style={{ maxWidth: '800px' }}>
        <div className="modal-header">
          <div>
            <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
              Transaction Inspection &bull; {transaction.transaction_id}
            </div>
            <h2 id="tx-detail-title" className="modal-title" style={{ marginTop: '0.25rem' }}>
              ${transaction.amount.toFixed(2)} &bull; {new Date(transaction.created_at).toLocaleString()}
            </h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">&times;</button>
        </div>

        <div className="modal-body">
          {/* Dual Scoring Section */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
            <div className="score-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-title">Supervised Fraud Probability</span>
                <span className={`status-badge ${isFlagged ? 'badge-flagged' : 'badge-normal'}`}>
                  {isFlagged ? 'FLAGGED' : 'NORMAL'}
                </span>
              </div>
              <div style={{ fontSize: '1.75rem', fontWeight: 700, margin: '0.25rem 0' }}>
                {(fraudProb * 100).toFixed(1)}%
              </div>
              <div className="score-bar-bg">
                <div
                  className="score-bar-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, fraudProb * 100))}%`,
                    backgroundColor: fraudProb >= 0.5 ? '#ef4444' : '#10b981',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                <span>Cutoff: 50.0%</span>
                <span>Calibrated Model: {explanation?.model_name || 'Logistic Regression'}</span>
              </div>
            </div>

            <div className="score-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-title">Unsupervised Anomaly Distance</span>
                <span className="status-badge" style={{ backgroundColor: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa' }}>
                  OUTLIER SCORE
                </span>
              </div>
              <div style={{ fontSize: '1.75rem', fontWeight: 700, margin: '0.25rem 0' }}>
                {(anomalyScore * 100).toFixed(1)}%
              </div>
              <div className="score-bar-bg">
                <div
                  className="score-bar-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, anomalyScore * 100))}%`,
                    backgroundColor: '#3b82f6',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                <span>Type: Isolation Forest</span>
                <span>Domain: Multi-dimensional distance</span>
              </div>
            </div>
          </div>

          {/* Explainability Section */}
          <div className="card" style={{ padding: '1.25rem', marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                Explainable Feature Attribution (SHAP Waterfall)
              </div>
              {explanation && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Method: <code>{explanation.explanation_method}</code> &bull; v{explanation.model_version}
                </div>
              )}
            </div>

            {loading && <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Calculating feature attributions...</div>}

            {error && (
              <div style={{ color: '#f87171', fontSize: '0.875rem' }}>
                Explanation unavailable: {error}
              </div>
            )}

            {explanation && (
              <div>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  {explanation.summary_text}
                </p>

                <div style={{ marginBottom: '1rem' }}>
                  {explanation.top_contributions.map((contrib, idx) => {
                    const isPositive = contrib.direction === 'increases_risk';
                    const barPct = Math.min(100, Math.max(10, contrib.relative_magnitude * 100));
                    return (
                      <div key={idx} className="waterfall-row">
                        <div className="waterfall-label" title={contrib.plain_text_description}>
                          {contrib.feature_name}
                        </div>
                        <div className="waterfall-bar-track">
                          <div
                            className="waterfall-bar-fill"
                            style={{
                              width: `${barPct}%`,
                              backgroundColor: isPositive ? '#ef4444' : '#10b981',
                            }}
                          />
                        </div>
                        <div className="waterfall-val" style={{ color: isPositive ? '#f87171' : '#34d399' }}>
                          {contrib.contribution >= 0 ? `+${contrib.contribution.toFixed(3)}` : contrib.contribution.toFixed(3)}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Regulatory disclaimers */}
                {explanation.limitations.length > 0 && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', backgroundColor: 'rgba(15, 23, 42, 0.4)', padding: '0.75rem', borderRadius: '6px' }}>
                    <strong>Regulatory Notice:</strong> {explanation.limitations[0]}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-outline" onClick={onClose}>
            Close
          </button>
          {onCreateCase && (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                onCreateCase(transaction.transaction_id);
                onClose();
              }}
            >
              Open Investigation Case
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

