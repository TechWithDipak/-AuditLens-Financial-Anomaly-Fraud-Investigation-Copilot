import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Search,
  Briefcase,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Server,
  RefreshCw,
} from 'lucide-react';
import { ActiveTab, CaseItem, HealthResponse } from '../../types';
import { listCases } from '../../services/api';

interface OverviewTabProps {
  health: HealthResponse | null;
  isLoadingHealth: boolean;
  onRefreshHealth: () => void;
  onSelectTab: (tab: ActiveTab) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  health,
  isLoadingHealth,
  onRefreshHealth,
  onSelectTab,
}) => {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [isLoadingCases, setIsLoadingCases] = useState(false);

  useEffect(() => {
    const fetchCases = async () => {
      setIsLoadingCases(true);
      try {
        const data = await listCases();
        setCases(data.slice(0, 5));
      } catch (e) {
        // Fallback or ignore if backend has no cases yet
      } finally {
        setIsLoadingCases(false);
      }
    };
    fetchCases();
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header Hero */}
      <div className="p-6 rounded-xl bg-gradient-to-r from-graphite-900 via-graphite-850 to-graphite-900 border border-graphite-800 shadow-lg">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Phase 9 Operational
              </span>
              <span className="text-xs font-mono text-graphite-400">
                FastAPI 0.1.0 • Python 3.13 • React 18
              </span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white mt-1.5">
              AuditLens Financial Investigation Copilot
            </h1>
            <p className="text-sm text-graphite-400 mt-1 max-w-2xl leading-relaxed">
              Unified machine learning analytics system combining unsupervised anomaly detection,
              Markovian temporal state modeling, supervised tree ensembles, and local SHAP attributions.
            </p>
          </div>

          <div className="flex items-center space-x-3 shrink-0">
            <button
              onClick={() => onSelectTab('explorer')}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-md transition"
            >
              <Search className="w-3.5 h-3.5" />
              <span>Explore Transactions</span>
            </button>
            <button
              onClick={() => onSelectTab('algorithms')}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-200 font-medium text-xs border border-graphite-700 transition"
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Algorithm Lab</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-graphite-400 text-xs font-semibold uppercase tracking-wider">
            <span>Backend Health</span>
            <Server className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-3">
            <div className="text-xl font-bold text-white flex items-center space-x-2">
              {health ? (
                <>
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span>Online</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-5 h-5 text-rose-400" />
                  <span>Offline</span>
                </>
              )}
            </div>
            <p className="text-xs text-graphite-400 mt-0.5">
              {health ? `Host: 127.0.0.1:8000` : 'Cannot connect to API'}
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-graphite-800 text-[11px] text-graphite-400 flex justify-between items-center">
            <span>Uptime Check</span>
            <button
              onClick={onRefreshHealth}
              disabled={isLoadingHealth}
              className="hover:text-emerald-400 transition"
            >
              <RefreshCw className={`w-3 h-3 ${isLoadingHealth ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-graphite-400 text-xs font-semibold uppercase tracking-wider">
            <span>Syllabus Algorithms</span>
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-3">
            <div className="text-xl font-bold text-white font-mono">9 / 9</div>
            <p className="text-xs text-graphite-400 mt-0.5">Units 3, 4, and 5 Implemented</p>
          </div>
          <div className="mt-3 pt-2 border-t border-graphite-800 text-[11px] text-indigo-400 flex justify-between">
            <span>All Pure ML / NumPy</span>
            <span className="text-graphite-500">100% Pass</span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-graphite-400 text-xs font-semibold uppercase tracking-wider">
            <span>Investigation Cases</span>
            <Briefcase className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-3">
            <div className="text-xl font-bold text-white font-mono">
              {isLoadingCases ? '...' : cases.length}
            </div>
            <p className="text-xs text-graphite-400 mt-0.5">Tracked in SQLite WAL</p>
          </div>
          <div className="mt-3 pt-2 border-t border-graphite-800 text-[11px] text-cyan-400 flex justify-between">
            <button onClick={() => onSelectTab('cases')} className="hover:underline flex items-center space-x-1">
              <span>View Audit Logs</span>
              <ArrowRight className="w-2.5 h-2.5" />
            </button>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-graphite-400 text-xs font-semibold uppercase tracking-wider">
            <span>Advisory Guardrail</span>
            <ShieldCheck className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-3">
            <div className="text-xl font-bold text-amber-400 font-mono">Strict</div>
            <p className="text-xs text-graphite-400 mt-0.5">Analyst-in-the-Loop Required</p>
          </div>
          <div className="mt-3 pt-2 border-t border-graphite-800 text-[11px] text-graphite-400">
            <span>Zero automated accusations</span>
          </div>
        </div>
      </div>

      {/* Syllabus Matrix & Quick Navigation */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Unit 3 Card */}
        <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-indigo-400 uppercase tracking-wider">
              Unit 3: Mixture Models & EM
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
              Unsupervised
            </span>
          </div>
          <p className="text-xs text-graphite-400">
            Clustering normal spending behaviors and discovering anomalous peer outliers.
          </p>
          <div className="space-y-2 pt-1 font-mono text-xs">
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">K-Means Clustering</span>
              <span className="text-[10px] text-graphite-400">Elbow & Silhouette</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">GMM via EM</span>
              <span className="text-[10px] text-graphite-400">Log-Likelihood Rarity</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">PCA Projection</span>
              <span className="text-[10px] text-graphite-400">2D Coordinate Map</span>
            </div>
          </div>
          <button
            onClick={() => onSelectTab('algorithms')}
            className="w-full mt-2 text-center text-xs text-indigo-400 hover:text-indigo-300 py-1.5 rounded bg-indigo-500/5 hover:bg-indigo-500/10 border border-indigo-500/20 transition flex items-center justify-center space-x-1"
          >
            <span>Launch Unit 3 Lab</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {/* Unit 4 Card */}
        <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
              Unit 4: Hidden Markov Models
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
              Sequence Analysis
            </span>
          </div>
          <p className="text-xs text-graphite-400">
            Temporal account behavior transitions and high-risk sequence state decoding.
          </p>
          <div className="space-y-2 pt-1 font-mono text-xs">
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">Scaled Forward Algo</span>
              <span className="text-[10px] text-graphite-400">Underflow-Free P(O|λ)</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">Viterbi Max-Product</span>
              <span className="text-[10px] text-graphite-400">Log-Space Decoding</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">Account Timelines</span>
              <span className="text-[10px] text-graphite-400">Temporal Anomaly Flow</span>
            </div>
          </div>
          <button
            onClick={() => onSelectTab('algorithms')}
            className="w-full mt-2 text-center text-xs text-emerald-400 hover:text-emerald-300 py-1.5 rounded bg-emerald-500/5 hover:bg-emerald-500/10 border border-emerald-500/20 transition flex items-center justify-center space-x-1"
          >
            <span>Launch Unit 4 Lab</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {/* Unit 5 Card */}
        <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-amber-400 uppercase tracking-wider">
              Unit 5: Combining Models
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
              Supervised Ensembles
            </span>
          </div>
          <p className="text-xs text-graphite-400">
            Discriminative fraud probability estimation with fair identical-split benchmarking.
          </p>
          <div className="space-y-2 pt-1 font-mono text-xs">
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">Balanced CART Tree</span>
              <span className="text-[10px] text-graphite-400">Auditable Decision Rules</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">XGBoost Classifier</span>
              <span className="text-[10px] text-graphite-400">PR-AUC Optimized</span>
            </div>
            <div className="p-2 rounded bg-graphite-950 border border-graphite-800/80 flex justify-between items-center">
              <span className="text-graphite-300">Voting & Stacking</span>
              <span className="text-[10px] text-graphite-400">Meta-Learner Combos</span>
            </div>
          </div>
          <button
            onClick={() => onSelectTab('models')}
            className="w-full mt-2 text-center text-xs text-amber-400 hover:text-amber-300 py-1.5 rounded bg-amber-500/5 hover:bg-amber-500/10 border border-amber-500/20 transition flex items-center justify-center space-x-1"
          >
            <span>Launch Model Comparison</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Active Cases Snapshot */}
      <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Briefcase className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Active Investigation Cases</h3>
          </div>
          <button
            onClick={() => onSelectTab('cases')}
            className="text-xs text-emerald-400 hover:underline flex items-center space-x-1"
          >
            <span>Open Case Manager</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {cases.length === 0 ? (
          <div className="p-6 text-center rounded-lg bg-graphite-950 border border-graphite-800 text-graphite-400 text-xs">
            <p>No active investigation cases recorded yet.</p>
            <p className="mt-1 text-graphite-500">
              Flag suspicious transactions in the Explorer or create a case in the Case Management tab.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-graphite-800/80">
            {cases.map((c) => (
              <div
                key={c.id}
                className="py-2.5 flex items-center justify-between hover:bg-graphite-850 px-2 rounded cursor-pointer transition"
                onClick={() => onSelectTab('cases')}
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-semibold text-emerald-400">
                      {c.case_number}
                    </span>
                    <span className="text-xs text-white font-medium">{c.title}</span>
                  </div>
                  <div className="text-[11px] text-graphite-400 mt-0.5">
                    Assigned: {c.assigned_analyst} • Updated: {new Date(c.updated_at).toLocaleDateString()}
                  </div>
                </div>
                <div className="flex items-center space-x-3 text-xs">
                  <span
                    className={`px-2 py-0.5 rounded font-mono text-[10px] ${
                      c.priority === 'HIGH' || c.priority === 'CRITICAL'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : 'bg-graphite-800 text-graphite-300'
                    }`}
                  >
                    {c.priority}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-graphite-800 text-graphite-300 font-mono text-[10px]">
                    {c.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
