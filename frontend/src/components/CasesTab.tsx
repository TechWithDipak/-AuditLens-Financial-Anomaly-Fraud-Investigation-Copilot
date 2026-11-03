import React, { useState, useEffect, useCallback } from 'react';
import {
  Case,
  CaseCreatePayload,
  CasePriority,
  CaseStatus,
  addCaseNote,
  createCase,
  fetchCase,
  fetchCases,
  updateCase,
} from '../services/api';

interface CasesTabProps {
  initialTxIdForCase?: string | null;
  onClearInitialTx?: () => void;
}

export const CasesTab: React.FC<CasesTabProps> = ({ initialTxIdForCase, onClearInitialTx }) => {
  const [cases, setCases] = useState<Case[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(20);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [priorityFilter, setPriorityFilter] = useState<string>('ALL');

  // Create Case Modal
  const [showCreateModal, setShowCreateModal] = useState<boolean>(false);
  const [newTitle, setNewTitle] = useState<string>('');
  const [newDescription, setNewDescription] = useState<string>('');
  const [newPriority, setNewPriority] = useState<CasePriority>('MEDIUM');
  const [newAssignee, setNewAssignee] = useState<string>('');
  const [newTxIds, setNewTxIds] = useState<string>('');
  const [creating, setCreating] = useState<boolean>(false);

  // Selected Case Detail
  const [selectedCase, setSelectedCase] = useState<Case | null>(null);
  const [newNoteContent, setNewNoteContent] = useState<string>('');
  const [noteAuthor, setNoteAuthor] = useState<string>('analyst_primary');
  const [submittingNote, setSubmittingNote] = useState<boolean>(false);

  // Open create modal if initialTxIdForCase is passed from transaction table
  useEffect(() => {
    if (initialTxIdForCase) {
      setNewTxIds(initialTxIdForCase);
      setNewTitle(`Investigation for Transaction ${initialTxIdForCase}`);
      setShowCreateModal(true);
      if (onClearInitialTx) onClearInitialTx();
    }
  }, [initialTxIdForCase, onClearInitialTx]);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const st = statusFilter !== 'ALL' ? (statusFilter as CaseStatus) : undefined;
      const pr = priorityFilter !== 'ALL' ? (priorityFilter as CasePriority) : undefined;

      const data = await fetchCases(page, pageSize, st, pr);
      setCases(data.items);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load cases');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, statusFilter, priorityFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;

    try {
      setCreating(true);
      const txList = newTxIds
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      const payload: CaseCreatePayload = {
        title: newTitle,
        description: newDescription,
        priority: newPriority,
        transaction_ids: txList,
        assigned_to: newAssignee || undefined,
      };

      const created = await createCase(payload);
      setShowCreateModal(false);
      setNewTitle('');
      setNewDescription('');
      setNewTxIds('');
      await loadData();
      setSelectedCase(created);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to create case');
    } finally {
      setCreating(false);
    }
  };

  const handleStatusChange = async (newStatus: CaseStatus) => {
    if (!selectedCase) return;
    try {
      const updated = await updateCase(selectedCase.case_id, { status: newStatus });
      setSelectedCase(updated);
      await loadData();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to update status');
    }
  };

  const handlePriorityChange = async (newPriorityVal: CasePriority) => {
    if (!selectedCase) return;
    try {
      const updated = await updateCase(selectedCase.case_id, { priority: newPriorityVal });
      setSelectedCase(updated);
      await loadData();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to update priority');
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCase || !newNoteContent.trim()) return;

    try {
      setSubmittingNote(true);
      await addCaseNote(selectedCase.case_id, noteAuthor, newNoteContent);
      setNewNoteContent('');
      // Reload case detail
      const refreshed = await fetchCase(selectedCase.case_id);
      setSelectedCase(refreshed);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to add note');
    } finally {
      setSubmittingNote(false);
    }
  };

  const getPriorityBadgeClass = (p: CasePriority) => {
    switch (p) {
      case 'CRITICAL':
        return 'badge-critical';
      case 'HIGH':
        return 'badge-high';
      case 'MEDIUM':
        return 'badge-medium';
      case 'LOW':
        return 'badge-low';
    }
  };

  const getStatusBadgeClass = (s: CaseStatus) => {
    switch (s) {
      case 'NEW':
        return 'badge-medium';
      case 'IN_REVIEW':
        return 'badge-low';
      case 'ESCALATED':
        return 'badge-critical';
      case 'CLOSED_CONFIRMED_FRAUD':
        return 'badge-flagged';
      case 'CLOSED_FALSE_POSITIVE':
        return 'badge-normal';
    }
  };

  return (
    <div>
      {/* Action Toolbar */}
      <div className="filter-toolbar">
        <div className="filter-group">
          <select
            className="select-field"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
          >
            <option value="ALL">All Statuses</option>
            <option value="NEW">New</option>
            <option value="IN_REVIEW">In Review</option>
            <option value="ESCALATED">Escalated</option>
            <option value="CLOSED_CONFIRMED_FRAUD">Closed: Confirmed Fraud</option>
            <option value="CLOSED_FALSE_POSITIVE">Closed: False Positive</option>
          </select>

          <select
            className="select-field"
            value={priorityFilter}
            onChange={(e) => {
              setPriorityFilter(e.target.value);
              setPage(1);
            }}
          >
            <option value="ALL">All Priorities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>

        <div className="filter-group">
          <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
            + New Case
          </button>
        </div>
      </div>

      {error && (
        <div className="banner" style={{ borderLeftColor: 'var(--status-danger)', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
          <div className="banner-title" style={{ color: '#f87171' }}>Failed to load investigation cases</div>
          <div className="banner-body">{error}</div>
        </div>
      )}

      {/* Cases Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Investigation Title</th>
              <th>Priority</th>
              <th>Status</th>
              <th>Assignee</th>
              <th>Transactions</th>
              <th>Last Updated</th>
              <th style={{ textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  Loading cases...
                </td>
              </tr>
            ) : cases.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                  No investigation cases found. Click <strong>+ New Case</strong> to open an investigation.
                </td>
              </tr>
            ) : (
              cases.map((c) => (
                <tr key={c.id}>
                  <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{c.case_id}</td>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{c.title}</td>
                  <td>
                    <span className={`status-badge ${getPriorityBadgeClass(c.priority)}`}>
                      {c.priority}
                    </span>
                  </td>
                  <td>
                    <span className={`status-badge ${getStatusBadgeClass(c.status)}`}>
                      {c.status.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{c.assigned_to || 'Unassigned'}</td>
                  <td>
                    <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                      {c.transaction_ids.length} linked
                    </span>
                  </td>
                  <td style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    {new Date(c.updated_at).toLocaleDateString()}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      type="button"
                      className="btn btn-outline btn-sm"
                      onClick={() => setSelectedCase(c)}
                    >
                      Manage &bull; Notes
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* Pagination */}
        <div className="pagination">
          <div>
            Showing <strong>{cases.length > 0 ? (page - 1) * pageSize + 1 : 0}</strong> to{' '}
            <strong>{Math.min(total, page * pageSize)}</strong> of <strong>{total}</strong> cases
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
              <option value={10}>10 / page</option>
              <option value={20}>20 / page</option>
              <option value={50}>50 / page</option>
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

      {/* Create Case Modal */}
      {showCreateModal && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="create-case-title">
          <div className="modal-content">
            <div className="modal-header">
              <h2 id="create-case-title" className="modal-title">Open Fraud Investigation Case</h2>
              <button className="modal-close-btn" onClick={() => setShowCreateModal(false)} aria-label="Close modal">&times;</button>
            </div>

            <form onSubmit={handleCreateCase}>
              <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                    Case Title *
                  </label>
                  <input
                    type="text"
                    className="input-field"
                    style={{ width: '100%' }}
                    placeholder="e.g. Card velocity spike across atypical merchant MCCs"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    required
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                    Description &amp; Preliminary Evidence
                  </label>
                  <textarea
                    className="input-field"
                    style={{ width: '100%', minHeight: '80px', fontFamily: 'inherit' }}
                    placeholder="Provide rationale, observed behavioral shifts, and investigation context..."
                    value={newDescription}
                    onChange={(e) => setNewDescription(e.target.value)}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                      Initial Priority
                    </label>
                    <select
                      className="select-field"
                      style={{ width: '100%' }}
                      value={newPriority}
                      onChange={(e) => setNewPriority(e.target.value as CasePriority)}
                    >
                      <option value="LOW">Low</option>
                      <option value="MEDIUM">Medium</option>
                      <option value="HIGH">High</option>
                      <option value="CRITICAL">Critical</option>
                    </select>
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                      Assignee Analyst ID
                    </label>
                    <input
                      type="text"
                      className="input-field"
                      style={{ width: '100%' }}
                      placeholder="e.g. analyst_sarah"
                      value={newAssignee}
                      onChange={(e) => setNewAssignee(e.target.value)}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                    Linked Transaction IDs (comma-separated)
                  </label>
                  <input
                    type="text"
                    className="input-field"
                    style={{ width: '100%', fontFamily: 'monospace' }}
                    placeholder="e.g. TX_00000000_0, TX_00000001_1"
                    value={newTxIds}
                    onChange={(e) => setNewTxIds(e.target.value)}
                  />
                </div>
              </div>

              <div className="modal-footer">
                <button type="button" className="btn btn-outline" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={creating || !newTitle.trim()}>
                  {creating ? 'Creating...' : 'Open Case'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Case Detail & Management Modal */}
      {selectedCase && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="manage-case-title">
          <div className="modal-content" style={{ maxWidth: '800px' }}>
            <div className="modal-header">
              <div>
                <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                  Investigation File &bull; {selectedCase.case_id}
                </div>
                <h2 id="manage-case-title" className="modal-title" style={{ marginTop: '0.25rem' }}>
                  {selectedCase.title}
                </h2>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedCase(null)} aria-label="Close modal">&times;</button>
            </div>

            <div className="modal-body">
              {/* Status and Priority Controls */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
                <div className="card" style={{ padding: '1rem' }}>
                  <label style={{ display: 'block', fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.4rem' }}>
                    Investigation Lifecycle Status
                  </label>
                  <select
                    className="select-field"
                    style={{ width: '100%' }}
                    value={selectedCase.status}
                    onChange={(e) => handleStatusChange(e.target.value as CaseStatus)}
                  >
                    <option value="NEW">NEW</option>
                    <option value="IN_REVIEW">IN REVIEW</option>
                    <option value="ESCALATED">ESCALATED</option>
                    <option value="CLOSED_CONFIRMED_FRAUD">CLOSED: CONFIRMED FRAUD</option>
                    <option value="CLOSED_FALSE_POSITIVE">CLOSED: FALSE POSITIVE</option>
                  </select>
                </div>

                <div className="card" style={{ padding: '1rem' }}>
                  <label style={{ display: 'block', fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.4rem' }}>
                    Triage Priority
                  </label>
                  <select
                    className="select-field"
                    style={{ width: '100%' }}
                    value={selectedCase.priority}
                    onChange={(e) => handlePriorityChange(e.target.value as CasePriority)}
                  >
                    <option value="LOW">LOW</option>
                    <option value="MEDIUM">MEDIUM</option>
                    <option value="HIGH">HIGH</option>
                    <option value="CRITICAL">CRITICAL</option>
                  </select>
                </div>
              </div>

              {selectedCase.description && (
                <div style={{ marginBottom: '1.25rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  <strong>Context:</strong> {selectedCase.description}
                </div>
              )}

              {/* Linked Transactions */}
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '0.5rem' }}>
                  Linked Flagged Transactions ({selectedCase.transaction_ids.length})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {selectedCase.transaction_ids.length === 0 ? (
                    <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>No transactions linked yet.</span>
                  ) : (
                    selectedCase.transaction_ids.map((txId) => (
                      <span
                        key={txId}
                        style={{
                          backgroundColor: 'rgba(15, 23, 42, 0.7)',
                          border: '1px solid var(--border-color)',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '4px',
                          fontSize: '0.8125rem',
                          fontFamily: 'monospace',
                        }}
                      >
                        {txId}
                      </span>
                    ))
                  )}
                </div>
              </div>

              {/* Analyst Notes History */}
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                  Analyst Audit Notes ({selectedCase.notes.length})
                </div>

                {selectedCase.notes.length === 0 ? (
                  <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                    No notes recorded yet. Add initial investigation findings below.
                  </div>
                ) : (
                  selectedCase.notes.map((note) => (
                    <div key={note.id} className="note-item">
                      <div className="note-header">
                        <span className="note-author">{note.author}</span>
                        <span>{new Date(note.created_at).toLocaleString()}</span>
                      </div>
                      <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                        {note.content}
                      </div>
                    </div>
                  ))
                )}

                {/* Add Note Form */}
                <form onSubmit={handleAddNote} style={{ marginTop: '1rem' }}>
                  <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.5rem' }}>
                    <input
                      type="text"
                      className="input-field"
                      style={{ width: '160px' }}
                      placeholder="Analyst ID"
                      value={noteAuthor}
                      onChange={(e) => setNoteAuthor(e.target.value)}
                      required
                    />
                    <input
                      type="text"
                      className="input-field"
                      style={{ flex: 1 }}
                      placeholder="Enter investigation finding, customer response, or audit note..."
                      value={newNoteContent}
                      onChange={(e) => setNewNoteContent(e.target.value)}
                      required
                    />
                    <button type="submit" className="btn btn-primary" disabled={submittingNote || !newNoteContent.trim()}>
                      {submittingNote ? 'Adding...' : 'Add Note'}
                    </button>
                  </div>
                </form>
              </div>
            </div>

            <div className="modal-footer">
              <button type="button" className="btn btn-outline" onClick={() => setSelectedCase(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
