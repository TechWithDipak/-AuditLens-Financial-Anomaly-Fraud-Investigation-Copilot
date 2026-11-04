import React, { useState, useEffect } from 'react';
import {
  Briefcase,
  Plus,
  RefreshCw,
  X,
  History,
  Send,
} from 'lucide-react';
import { AuditEventItem, CaseDetail, CaseItem } from '../../types';
import {
  addCaseNote,
  createCase,
  getCaseDetail,
  listAuditEvents,
  listCases,
  updateCaseStatus,
} from '../../services/api';

export const CaseManagementTab: React.FC = () => {
  const [view, setView] = useState<'cases' | 'audit'>('cases');

  // Case List State
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [isLoadingCases, setIsLoadingCases] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [priorityFilter, setPriorityFilter] = useState<string>('');

  // Selected Case Detail
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  // Status transition state
  const [newStatus, setNewStatus] = useState<string>('');
  const [statusNote, setStatusNote] = useState<string>('');
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false);

  // New Note state
  const [noteContent, setNoteContent] = useState<string>('');
  const [isAddingNote, setIsAddingNote] = useState(false);

  // New Case Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newPriority, setNewPriority] = useState('MEDIUM');
  const [newAnalyst, setNewAnalyst] = useState('analyst_smith');
  const [isCreating, setIsCreating] = useState(false);

  // Audit Events State
  const [auditEvents, setAuditEvents] = useState<AuditEventItem[]>([]);
  const [isLoadingAudit, setIsLoadingAudit] = useState(false);

  useEffect(() => {
    loadCases();
  }, [statusFilter, priorityFilter]);

  const loadCases = async () => {
    setIsLoadingCases(true);
    try {
      const data = await listCases(statusFilter, priorityFilter);
      setCases(data);
    } finally {
      setIsLoadingCases(false);
    }
  };

  const handleSelectCase = async (id: number) => {
    setSelectedCaseId(id);
    setIsLoadingDetail(true);
    try {
      const data = await getCaseDetail(id);
      setCaseDetail(data);
      setNewStatus(data.status);
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const handleStatusUpdate = async () => {
    if (!caseDetail || !newStatus) return;
    setIsUpdatingStatus(true);
    try {
      await updateCaseStatus(caseDetail.id, newStatus, statusNote, 'analyst_active');
      setStatusNote('');
      await handleSelectCase(caseDetail.id);
      loadCases();
    } finally {
      setIsUpdatingStatus(false);
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!caseDetail || !noteContent.trim()) return;
    setIsAddingNote(true);
    try {
      await addCaseNote(caseDetail.id, noteContent.trim(), 'analyst_active');
      setNoteContent('');
      await handleSelectCase(caseDetail.id);
    } finally {
      setIsAddingNote(false);
    }
  };

  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    setIsCreating(true);
    try {
      await createCase({
        title: newTitle.trim(),
        description: newDesc.trim() || undefined,
        priority: newPriority,
        assigned_analyst: newAnalyst,
      });
      setShowCreateModal(false);
      setNewTitle('');
      setNewDesc('');
      loadCases();
    } finally {
      setIsCreating(false);
    }
  };

  const loadAuditEvents = async () => {
    setIsLoadingAudit(true);
    try {
      const data = await listAuditEvents();
      setAuditEvents(data);
    } finally {
      setIsLoadingAudit(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-graphite-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Briefcase className="w-5 h-5 text-cyan-400" />
            <span>Investigation Cases & Immutable Governance Trail</span>
          </h2>
          <p className="text-xs text-graphite-400 mt-0.5">
            Manage fraud investigation lifecycles, analyst rationales, and audit histories.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-graphite-900 p-1 rounded-lg border border-graphite-800 text-xs">
            <button
              onClick={() => setView('cases')}
              className={`px-3 py-1.5 rounded-md font-medium transition ${
                view === 'cases'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-graphite-400 hover:text-white'
              }`}
            >
              Cases Queue
            </button>
            <button
              onClick={() => {
                setView('audit');
                loadAuditEvents();
              }}
              className={`px-3 py-1.5 rounded-md font-medium transition ${
                view === 'audit'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-graphite-400 hover:text-white'
              }`}
            >
              Audit Trail
            </button>
          </div>

          {view === 'cases' && (
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Create Case</span>
            </button>
          )}
        </div>
      </div>

      {/* CASES VIEW */}
      {view === 'cases' && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex items-center justify-between bg-graphite-900 p-3 rounded-xl border border-graphite-800 text-xs">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-1.5">
                <span className="text-graphite-400">Status:</span>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-graphite-950 border border-graphite-800 rounded px-2.5 py-1 text-white text-xs"
                >
                  <option value="">All Statuses</option>
                  <option value="NEW">NEW</option>
                  <option value="UNDER_REVIEW">UNDER_REVIEW</option>
                  <option value="ESCALATED">ESCALATED</option>
                  <option value="CLOSED_BENIGN">CLOSED_BENIGN</option>
                  <option value="CLOSED_SUSPICIOUS">CLOSED_SUSPICIOUS</option>
                </select>
              </div>

              <div className="flex items-center space-x-1.5">
                <span className="text-graphite-400">Priority:</span>
                <select
                  value={priorityFilter}
                  onChange={(e) => setPriorityFilter(e.target.value)}
                  className="bg-graphite-950 border border-graphite-800 rounded px-2.5 py-1 text-white text-xs"
                >
                  <option value="">All Priorities</option>
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </div>
            </div>

            <button
              onClick={loadCases}
              disabled={isLoadingCases}
              className="text-graphite-400 hover:text-white flex items-center space-x-1 text-xs"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingCases ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>

          {/* Master-Detail Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Case List */}
            <div className={`${selectedCaseId ? 'lg:col-span-1' : 'lg:col-span-3'} space-y-2`}>
              {cases.length === 0 ? (
                <div className="p-8 text-center rounded-xl bg-graphite-900 border border-graphite-800 text-graphite-400 text-xs">
                  No cases found. Click "Create Case" or flag a transaction in Explorer.
                </div>
              ) : (
                cases.map((c) => {
                  const isSelected = selectedCaseId === c.id;
                  return (
                    <div
                      key={c.id}
                      onClick={() => handleSelectCase(c.id)}
                      className={`p-4 rounded-xl border bg-graphite-900 hover:bg-graphite-850 cursor-pointer transition ${
                        isSelected
                          ? 'border-cyan-500/50 bg-graphite-850'
                          : 'border-graphite-800'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <span className="font-mono text-xs font-bold text-cyan-400">
                          {c.case_number}
                        </span>
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                            c.priority === 'HIGH' || c.priority === 'CRITICAL'
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                              : 'bg-graphite-800 text-graphite-300'
                          }`}
                        >
                          {c.priority}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-white mt-1">{c.title}</div>
                      <div className="flex items-center justify-between text-[11px] text-graphite-400 mt-2 pt-2 border-t border-graphite-800/80">
                        <span className="font-mono">{c.status}</span>
                        <span>
                          {c.notes_count} notes • {c.transactions_count} txs
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Case Detail View */}
            {selectedCaseId && isLoadingDetail && (
              <div className="lg:col-span-2 rounded-xl border border-graphite-800 bg-graphite-900 p-12 text-center text-graphite-400 text-xs">
                Loading case investigation file...
              </div>
            )}
            {selectedCaseId && !isLoadingDetail && caseDetail && (
              <div className="lg:col-span-2 rounded-xl border border-graphite-800 bg-graphite-900 p-5 space-y-5 shadow-xl">
                <div className="flex justify-between items-start border-b border-graphite-800 pb-3">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-xs text-cyan-400 font-bold">
                        {caseDetail.case_number}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded bg-graphite-800 font-mono text-graphite-300">
                        {caseDetail.status}
                      </span>
                    </div>
                    <h3 className="text-base font-bold text-white mt-1">{caseDetail.title}</h3>
                    {caseDetail.description && (
                      <p className="text-xs text-graphite-400 mt-1 leading-relaxed">
                        {caseDetail.description}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={() => {
                      setSelectedCaseId(null);
                      setCaseDetail(null);
                    }}
                    className="text-graphite-400 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Status Transition Control */}
                <div className="p-3.5 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-white">Status Transition:</div>
                  <div className="flex flex-col md:flex-row gap-2">
                    <select
                      value={newStatus}
                      onChange={(e) => setNewStatus(e.target.value)}
                      className="bg-graphite-900 border border-graphite-800 rounded px-2.5 py-1.5 text-xs text-white"
                    >
                      <option value="NEW">NEW</option>
                      <option value="UNDER_REVIEW">UNDER_REVIEW</option>
                      <option value="ESCALATED">ESCALATED</option>
                      <option value="CLOSED_BENIGN">CLOSED_BENIGN</option>
                      <option value="CLOSED_SUSPICIOUS">CLOSED_SUSPICIOUS</option>
                    </select>
                    <input
                      type="text"
                      placeholder="Transition rationale note..."
                      value={statusNote}
                      onChange={(e) => setStatusNote(e.target.value)}
                      className="flex-1 bg-graphite-900 border border-graphite-800 rounded px-2.5 py-1.5 text-xs text-white"
                    />
                    <button
                      onClick={handleStatusUpdate}
                      disabled={isUpdatingStatus || newStatus === caseDetail.status}
                      className="px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs transition"
                    >
                      {isUpdatingStatus ? 'Updating...' : 'Update Status'}
                    </button>
                  </div>
                </div>

                {/* Linked Transactions */}
                <div className="space-y-2">
                  <div className="text-xs font-semibold text-white">
                    Linked Flagged Transactions ({caseDetail.transactions.length})
                  </div>
                  {caseDetail.transactions.length === 0 ? (
                    <p className="text-xs text-graphite-500">No transactions linked to this case.</p>
                  ) : (
                    <div className="divide-y divide-graphite-800/80 rounded border border-graphite-800 bg-graphite-950 overflow-hidden font-mono text-xs">
                      {caseDetail.transactions.map((t) => (
                        <div key={t.id} className="p-2.5 flex justify-between items-center">
                          <div>
                            <span className="font-semibold text-white">
                              Tx #{t.transaction_index}
                            </span>
                            <span className="text-emerald-400 ml-2">${t.amount.toFixed(2)}</span>
                            {t.added_reason && (
                              <p className="text-[10px] text-graphite-400 font-sans mt-0.5">
                                {t.added_reason}
                              </p>
                            )}
                          </div>
                          <div className="text-right">
                            {t.fraud_probability !== null && (
                              <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 text-[10px]">
                                Prob: {(t.fraud_probability * 100).toFixed(1)}%
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Notes Thread */}
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-white">
                    Analyst Notes & Activity History ({caseDetail.notes.length})
                  </div>

                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {caseDetail.notes.map((n) => (
                      <div
                        key={n.id}
                        className="p-3 rounded bg-graphite-950 border border-graphite-800 space-y-1 text-xs"
                      >
                        <div className="flex justify-between items-center text-graphite-400 font-mono text-[10px]">
                          <span className="font-semibold text-white">{n.author}</span>
                          <span>{new Date(n.created_at).toLocaleString()}</span>
                        </div>
                        <p className="text-graphite-300 leading-relaxed">{n.content}</p>
                      </div>
                    ))}
                  </div>

                  <form onSubmit={handleAddNote} className="flex space-x-2 pt-2">
                    <input
                      type="text"
                      placeholder="Add investigation finding or note..."
                      value={noteContent}
                      onChange={(e) => setNoteContent(e.target.value)}
                      className="flex-1 bg-graphite-950 border border-graphite-800 rounded px-3 py-1.5 text-xs text-white"
                    />
                    <button
                      type="submit"
                      disabled={isAddingNote || !noteContent.trim()}
                      className="px-3.5 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs flex items-center space-x-1"
                    >
                      <Send className="w-3 h-3" />
                      <span>Post</span>
                    </button>
                  </form>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* AUDIT VIEW */}
      {view === 'audit' && (
        <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                <History className="w-4 h-4 text-cyan-400" />
                <span>Immutable Governance Audit Events</span>
              </h3>
              <p className="text-xs text-graphite-400 mt-0.5">
                Append-only log of case creation, status changes, transaction links, and analyst notes.
              </p>
            </div>

            <button
              onClick={loadAuditEvents}
              disabled={isLoadingAudit}
              className="text-graphite-400 hover:text-white flex items-center space-x-1 text-xs"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingAudit ? 'animate-spin' : ''}`} />
              <span>Refresh Log</span>
            </button>
          </div>

          <div className="rounded-xl border border-graphite-800 bg-graphite-950 overflow-hidden font-mono text-xs">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-graphite-300">
                <thead className="bg-graphite-900 text-graphite-400 text-[10px] uppercase sticky top-0">
                  <tr>
                    <th className="p-3">Timestamp</th>
                    <th className="p-3">Action</th>
                    <th className="p-3">Entity Type</th>
                    <th className="p-3">Entity ID</th>
                    <th className="p-3">Actor</th>
                    <th className="p-3">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-graphite-800/60 text-[11px]">
                  {auditEvents.map((ev) => (
                    <tr key={ev.id} className="hover:bg-graphite-850">
                      <td className="p-3 text-graphite-500 whitespace-nowrap">
                        {new Date(ev.timestamp).toLocaleString()}
                      </td>
                      <td className="p-3 text-cyan-400 font-semibold">{ev.action}</td>
                      <td className="p-3 text-graphite-400">{ev.entity_type}</td>
                      <td className="p-3 text-white">{ev.entity_id}</td>
                      <td className="p-3 text-emerald-400">{ev.actor}</td>
                      <td className="p-3 text-graphite-400 max-w-xs truncate">
                        {ev.details || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* CREATE CASE MODAL */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-6 max-w-md w-full space-y-4 shadow-2xl">
            <div className="flex justify-between items-center border-b border-graphite-800 pb-3">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Briefcase className="w-4 h-4 text-cyan-400" />
                <span>Create Investigation Case</span>
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-graphite-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateCase} className="space-y-3 text-xs">
              <div>
                <label className="text-graphite-400 font-semibold block mb-1">Case Title *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Card-Not-Present Velocity Burst"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-2 text-white"
                />
              </div>

              <div>
                <label className="text-graphite-400 font-semibold block mb-1">Description</label>
                <textarea
                  rows={3}
                  placeholder="Hypothesis, observed pattern, or context..."
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-2 text-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-graphite-400 font-semibold block mb-1">Priority</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value)}
                    className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white"
                  >
                    <option value="LOW">LOW</option>
                    <option value="MEDIUM">MEDIUM</option>
                    <option value="HIGH">HIGH</option>
                    <option value="CRITICAL">CRITICAL</option>
                  </select>
                </div>

                <div>
                  <label className="text-graphite-400 font-semibold block mb-1">
                    Assigned Analyst
                  </label>
                  <input
                    type="text"
                    value={newAnalyst}
                    onChange={(e) => setNewAnalyst(e.target.value)}
                    className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white font-mono"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-3 border-t border-graphite-800">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-3 py-1.5 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-300 font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreating}
                  className="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium"
                >
                  {isCreating ? 'Creating...' : 'Create Case'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
