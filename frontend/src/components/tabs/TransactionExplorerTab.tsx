import React, { useState, useEffect } from 'react';
import {
  Search,
  Eye,
  Briefcase,
  RefreshCw,
  X,
  ShieldCheck,
} from 'lucide-react';
import {
  DatasetSampleResponse,
  ShapExplanationResult,
  TransactionRow,
} from '../../types';
import {
  createCase,
  explainTransaction,
  getSampleTransactions,
} from '../../services/api';

export const TransactionExplorerTab: React.FC = () => {
  const [data, setData] = useState<DatasetSampleResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [filterClass, setFilterClass] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Selected Transaction for Drawer / Inspection
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedTx, setSelectedTx] = useState<TransactionRow | null>(null);
  const [shapData, setShapData] = useState<ShapExplanationResult | null>(null);
  const [isLoadingShap, setIsLoadingShap] = useState(false);
  const [shapError, setShapError] = useState<string | null>(null);

  // Case creation feedback
  const [caseCreatedMsg, setCaseCreatedMsg] = useState<string | null>(null);
  const [isCreatingCase, setIsCreatingCase] = useState(false);

  useEffect(() => {
    loadTransactions();
  }, []);

  const loadTransactions = async () => {
    setIsLoading(true);
    try {
      const res = await getSampleTransactions(100, 0);
      setData(res);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectTransaction = async (idx: number, tx: TransactionRow) => {
    setSelectedIndex(idx);
    setSelectedTx(tx);
    setCaseCreatedMsg(null);
    setShapError(null);
    setIsLoadingShap(true);
    try {
      const res = await explainTransaction(idx, 'xgboost');
      setShapData(res);
    } catch (e: any) {
      setShapError(e.response?.data?.detail || e.message || 'Failed to compute SHAP');
    } finally {
      setIsLoadingShap(false);
    }
  };

  const handleCreateCaseFromTx = async () => {
    if (selectedIndex === null || !selectedTx) return;
    setIsCreatingCase(true);
    try {
      const caseItem = await createCase({
        title: `Investigation on Tx #${selectedIndex} ($${selectedTx.Amount?.toFixed(2)})`,
        description: `Flagged from Transaction Explorer. Supervised fraud probability: ${
          shapData ? shapData.predicted_probability.toFixed(3) : 'Pending'
        }.`,
        priority: selectedTx.Class === 1 ? 'HIGH' : 'MEDIUM',
        assigned_analyst: 'analyst_active',
        initial_transactions: [
          {
            transaction_index: selectedIndex,
            amount: selectedTx.Amount,
            fraud_probability: shapData?.predicted_probability ?? undefined,
            anomaly_score: undefined,
            added_reason: 'Flagged via SHAP attribution drawer',
          },
        ],
      });
      setCaseCreatedMsg(`Created case ${caseItem.case_number} successfully!`);
    } catch (e: any) {
      setCaseCreatedMsg(`Error creating case: ${e.message}`);
    } finally {
      setIsCreatingCase(false);
    }
  };

  const filteredRecords = data?.records.filter((r, idx) => {
    if (filterClass === 'fraud' && r.Class !== 1) return false;
    if (filterClass === 'benign' && r.Class !== 0) return false;
    if (searchTerm) {
      const strIdx = String(idx);
      const strAmt = String(r.Amount);
      return strIdx.includes(searchTerm) || strAmt.includes(searchTerm);
    }
    return true;
  });

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-graphite-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Search className="w-5 h-5 text-emerald-400" />
            <span>Transaction Explorer & Local SHAP Explainability</span>
          </h2>
          <p className="text-xs text-graphite-400 mt-0.5">
            Inspect individual financial transactions, compute game-theoretic SHAP attributions,
            and create linked investigation cases.
          </p>
        </div>

        <button
          onClick={loadTransactions}
          disabled={isLoading}
          className="px-3.5 py-1.5 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-200 border border-graphite-700 text-xs font-medium flex items-center space-x-1.5 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-col md:flex-row gap-3 items-center justify-between bg-graphite-900 p-3 rounded-xl border border-graphite-800 text-xs">
        <div className="flex items-center space-x-2 w-full md:w-auto">
          <Search className="w-4 h-4 text-graphite-400" />
          <input
            type="text"
            placeholder="Search by Row Index or Amount..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white text-xs w-64"
          />
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1">
            <span className="text-graphite-400">Filter:</span>
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              className="bg-graphite-950 border border-graphite-800 rounded-lg px-2.5 py-1 text-white text-xs"
            >
              <option value="all">All Transactions</option>
              <option value="fraud">Fraud (Class 1) Only</option>
              <option value="benign">Benign (Class 0) Only</option>
            </select>
          </div>

          <span className="text-graphite-500 font-mono">
            Showing {filteredRecords?.length || 0} of {data?.total_records || 0}
          </span>
        </div>
      </div>

      {/* Main Grid: Table & Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Transaction Table */}
        <div
          className={`${
            selectedTx ? 'lg:col-span-2' : 'lg:col-span-3'
          } rounded-xl border border-graphite-800 bg-graphite-900 overflow-hidden shadow-lg transition-all`}
        >
          <div className="overflow-x-auto max-h-[550px]">
            <table className="w-full text-left text-xs text-graphite-300 font-mono">
              <thead className="bg-graphite-950 text-graphite-400 uppercase text-[10px] sticky top-0 border-b border-graphite-800">
                <tr>
                  <th className="p-3">Index</th>
                  <th className="p-3">Time</th>
                  <th className="p-3">Amount ($)</th>
                  <th className="p-3">Ground Truth</th>
                  <th className="p-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-graphite-800/60 text-xs">
                {filteredRecords?.map((r, idx) => {
                  const isSelected = selectedIndex === idx;
                  return (
                    <tr
                      key={idx}
                      className={`hover:bg-graphite-850 cursor-pointer transition ${
                        isSelected ? 'bg-graphite-800/80 border-l-2 border-emerald-400' : ''
                      }`}
                      onClick={() => handleSelectTransaction(idx, r)}
                    >
                      <td className="p-3 text-graphite-400">#{idx}</td>
                      <td className="p-3 text-white">{r.Time}</td>
                      <td className="p-3 text-emerald-400 font-semibold">${r.Amount?.toFixed(2)}</td>
                      <td className="p-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] ${
                            r.Class === 1
                              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                              : 'bg-emerald-500/10 text-emerald-400'
                          }`}
                        >
                          {r.Class === 1 ? 'FRAUD (1)' : 'BENIGN (0)'}
                        </span>
                      </td>
                      <td className="p-3 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectTransaction(idx, r);
                          }}
                          className="px-2.5 py-1 rounded bg-graphite-800 hover:bg-graphite-700 text-graphite-300 hover:text-white text-[11px] inline-flex items-center space-x-1"
                        >
                          <Eye className="w-3 h-3" />
                          <span>Inspect</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* SHAP & Case Action Drawer */}
        {selectedTx && (
          <div className="rounded-xl border border-graphite-800 bg-graphite-900 p-5 space-y-4 shadow-xl flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-graphite-800 pb-3">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-sm font-bold text-white">
                    Transaction #{selectedIndex}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                    ${selectedTx.Amount?.toFixed(2)}
                  </span>
                </div>
                <button
                  onClick={() => {
                    setSelectedTx(null);
                    setSelectedIndex(null);
                  }}
                  className="text-graphite-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {isLoadingShap && (
                <div className="py-12 text-center text-graphite-400 text-xs flex flex-col items-center space-x-2">
                  <RefreshCw className="w-5 h-5 animate-spin mb-2 text-emerald-400" />
                  <span>Computing tree Shapley value attributions...</span>
                </div>
              )}

              {shapError && (
                <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
                  {shapError}
                </div>
              )}

              {shapData && !isLoadingShap && (
                <div className="space-y-4 font-mono text-xs">
                  {/* Probability Banner */}
                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800 space-y-1">
                    <div className="flex justify-between items-center text-graphite-400">
                      <span>Predicted Fraud Risk:</span>
                      <span className="text-white font-bold">
                        {(shapData.predicted_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-graphite-900 h-2 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          shapData.predicted_probability > 0.5 ? 'bg-rose-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, shapData.predicted_probability * 100)}%` }}
                      ></div>
                    </div>
                    <div className="text-[10px] text-graphite-500 flex justify-between pt-1">
                      <span>Base Expected Value: {(shapData.base_value * 100).toFixed(1)}%</span>
                      <span>Model: {shapData.model_name.toUpperCase()}</span>
                    </div>
                  </div>

                  {/* Top SHAP Attributions */}
                  <div className="space-y-2">
                    <div className="text-white font-semibold flex items-center justify-between">
                      <span>Top Feature Attributions (SHAP)</span>
                      <span className="text-[10px] text-graphite-400 font-normal">Impact on Score</span>
                    </div>

                    <div className="space-y-1.5 max-h-52 overflow-y-auto">
                      {shapData.features.slice(0, 6).map((feat, i) => (
                        <div
                          key={i}
                          className="p-2 rounded bg-graphite-950 border border-graphite-800 flex justify-between items-center text-xs"
                        >
                          <div>
                            <span className="text-white font-bold">{feat.feature}</span>
                            <span className="text-graphite-500 text-[10px] ml-1.5">
                              ={feat.value.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center space-x-1.5">
                            <span
                              className={`font-bold ${
                                feat.impact === 'increases_risk'
                                  ? 'text-rose-400'
                                  : 'text-cyan-400'
                              }`}
                            >
                              {feat.shap_value > 0 ? '+' : ''}
                              {feat.shap_value.toFixed(4)}
                            </span>
                            <span className="text-[10px] text-graphite-500">
                              {feat.impact === 'increases_risk' ? '▲ Risk' : '▼ Risk'}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Legal Disclaimer Alert */}
                  <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/20 text-[10px] text-amber-300 leading-snug flex items-start space-x-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-400" />
                    <span>
                      Advisory Notice: SHAP attributions reflect mathematical feature contribution to
                      model scores, not legal proof of fraud or causality.
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Create Case Button */}
            <div className="pt-3 border-t border-graphite-800 space-y-2">
              <button
                onClick={handleCreateCaseFromTx}
                disabled={isCreatingCase}
                className="w-full py-2 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center justify-center space-x-2 shadow transition"
              >
                <Briefcase className="w-4 h-4" />
                <span>{isCreatingCase ? 'Creating Case...' : 'Open Investigation Case'}</span>
              </button>

              {caseCreatedMsg && (
                <div className="text-[11px] text-center text-emerald-400 font-mono">
                  {caseCreatedMsg}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
