import React, { useState, useEffect, useCallback } from 'react';
import { Transaction, fetchTransactions } from '../services/api';
import { TransactionModal } from './TransactionModal';

interface TransactionsTabProps {
  onOpenCreateCase?: (transactionId: string) => void;
}

export const TransactionsTab: React.FC<TransactionsTabProps> = ({ onOpenCreateCase }) => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(50);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'FLAGGED' | 'NORMAL'>('ALL');
  const [minAmount, setMinAmount] = useState<string>('');
  const [maxAmount, setMaxAmount] = useState<string>('');
  const [minScore, setMinScore] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Selected Transaction for Inspection Modal
  const [selectedTx, setSelectedTx] = useState<Transaction | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const filters: Record<string, unknown> = {
        page,
        page_size: pageSize,
      };

      if (statusFilter === 'FLAGGED') filters.is_flagged = true;
      if (statusFilter === 'NORMAL') filters.is_flagged = false;
      if (minAmount) filters.min_amount = parseFloat(minAmount);
      if (maxAmount) filters.max_amount = parseFloat(maxAmount);
      if (minScore) filters.min_score = parseFloat(minScore) / 100;

      const data = await fetchTransactions(filters);
      setTransactions(data.items);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load transactions');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, statusFilter, minAmount, maxAmount, minScore]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Client-side quick filter for transaction ID search
  const displayedTransactions = searchTerm
    ? transactions.filter((t) => t.transaction_id.toLowerCase().includes(searchTerm.toLowerCase()))
    : transactions;

  return (
    <div>
      {/* Filter Toolbar */}
      <div className="filter-toolbar">
        <div className="filter-group">
          <input
            type="text"
            className="input-field"
            placeholder="Search Transaction ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: '220px' }}
          />

          <select
            className="select-field"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value as 'ALL' | 'FLAGGED' | 'NORMAL');
              setPage(1);
            }}
          >
            <option value="ALL">All Statuses</option>
            <option value="FLAGGED">Flagged Only (&gt;= 50%)</option>
            <option value="NORMAL">Normal Only (&lt; 50%)</option>
          </select>

          <input
            type="number"
            className="input-field"
            placeholder="Min Amount ($)"
            value={minAmount}
            onChange={(e) => {
              setMinAmount(e.target.value);
              setPage(1);
            }}
            style={{ width: '130px' }}
          />

          <input
            type="number"
            className="input-field"
            placeholder="Max Amount ($)"
            value={maxAmount}
            onChange={(e) => {
              setMaxAmount(e.target.value);
              setPage(1);
            }}
            style={{ width: '130px' }}
          />

          <input
            type="number"
            className="input-field"
            placeholder="Min Fraud Prob %"
            value={minScore}
            onChange={(e) => {
              setMinScore(e.target.value);
              setPage(1);
            }}
            style={{ width: '150px' }}
          />
        </div>

        <div className="filter-group">
          <button className="btn btn-outline btn-sm" onClick={() => loadData()}>
            &#x21bb; Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="banner" style={{ borderLeftColor: 'var(--status-danger)', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
          <div className="banner-title" style={{ color: '#f87171' }}>Failed to retrieve transactions</div>
          <div className="banner-body">{error}</div>
        </div>
      )}

      {/* Transactions Data Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Time (s)</th>
              <th>Amount ($)</th>
              <th>Anomaly Score</th>
              <th>Fraud Probability</th>
              <th>Review Status</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  Loading transactions from database...
                </td>
              </tr>
            ) : displayedTransactions.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  {total === 0 ? (
                    <div>
                      <p style={{ marginBottom: '0.5rem', fontWeight: 600 }}>No transactions stored yet.</p>
                      <p style={{ fontSize: '0.8125rem' }}>Upload a dataset CSV in the Overview tab or use the API to ingest records.</p>
                    </div>
                  ) : (
                    'No transactions match the selected filters.'
                  )}
                </td>
              </tr>
            ) : (
              displayedTransactions.map((tx) => {
                const fraudProb = tx.fraud_probability ?? 0.0;
                const anomalyScore = tx.anomaly_score ?? 0.0;
                const isFlagged = tx.is_flagged;

                return (
                  <tr key={tx.id}>
                    <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{tx.transaction_id}</td>
                    <td style={{ color: 'var(--text-secondary)' }}>{tx.time.toFixed(0)}s</td>
                    <td style={{ fontWeight: 600 }}>${tx.amount.toFixed(2)}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.8125rem', fontFamily: 'monospace', width: '45px' }}>
                          {(anomalyScore * 100).toFixed(1)}%
                        </span>
                        <div style={{ width: '60px', height: '6px', backgroundColor: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${anomalyScore * 100}%`, height: '100%', backgroundColor: '#3b82f6' }} />
                        </div>
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.8125rem', fontFamily: 'monospace', width: '45px', color: isFlagged ? '#f87171' : 'var(--text-primary)' }}>
                          {(fraudProb * 100).toFixed(1)}%
                        </span>
                        <div style={{ width: '60px', height: '6px', backgroundColor: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${fraudProb * 100}%`,
                              height: '100%',
                              backgroundColor: isFlagged ? '#ef4444' : '#10b981',
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`status-badge ${isFlagged ? 'badge-flagged' : 'badge-normal'}`}>
                        {isFlagged ? 'FLAGGED' : 'NORMAL'}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '0.5rem' }}>
                        <button
                          type="button"
                          className="btn btn-outline btn-sm"
                          onClick={() => setSelectedTx(tx)}
                        >
                          Inspect &bull; SHAP
                        </button>
                        {onOpenCreateCase && (
                          <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            onClick={() => onOpenCreateCase(tx.transaction_id)}
                          >
                            + Case
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>

        {/* Pagination Bar */}
        <div className="pagination">
          <div>
            Showing <strong>{transactions.length > 0 ? (page - 1) * pageSize + 1 : 0}</strong> to{' '}
            <strong>{Math.min(total, page * pageSize)}</strong> of <strong>{total}</strong> transactions
          </div>

          <div className="pagination-controls">
            <select
              className="select-field"
              style={{ padding: '0.25rem 0.5rem', fontSize: '0.8125rem' }}
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
            >
              <option value={25}>25 / page</option>
              <option value={50}>50 / page</option>
              <option value={100}>100 / page</option>
            </select>

            <button
              type="button"
              className="btn btn-outline btn-sm"
              disabled={page <= 1}
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
            >
              &larr; Prev
            </button>

            <span style={{ fontSize: '0.8125rem', padding: '0 0.5rem' }}>
              Page {page} of {totalPages}
            </span>

            <button
              type="button"
              className="btn btn-outline btn-sm"
              disabled={page >= totalPages}
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
            >
              Next &rarr;
            </button>
          </div>
        </div>
      </div>

      {/* Transaction Inspection & Explanation Modal */}
      {selectedTx && (
        <TransactionModal
          transaction={selectedTx}
          onClose={() => setSelectedTx(null)}
          onCreateCase={onOpenCreateCase}
        />
      )}
    </div>
  );
};

