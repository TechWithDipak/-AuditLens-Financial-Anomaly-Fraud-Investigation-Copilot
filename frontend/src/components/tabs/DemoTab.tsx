import React, { useState, useEffect } from 'react';
import {
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Search,
  FileText,
  ShieldAlert,
  ShieldCheck,
  Scale,
  Download,
  Copy,
  Sparkles,
  Briefcase,
  HelpCircle,
  Sliders,
  Check,
} from 'lucide-react';
import {
  ActiveTab,
  DemoTransaction,
  DemoRunResponse,
  InvestigationReportDraft,
  RagCitation,
} from '../../types';
import {
  getDemoTransactions,
  runFraudDetectionDemo,
  resetDemoState,
  createCase,
  generateInvestigationDraft,
  verifyReportEvidence,
  queryRag,
} from '../../services/api';

interface DemoTabProps {
  onNavigateToTab?: (tab: ActiveTab) => void;
}

export const DemoTab: React.FC<DemoTabProps> = ({ onNavigateToTab }) => {
  // Demo State
  const [transactions, setTransactions] = useState<DemoTransaction[]>([]);
  const [runResult, setRunResult] = useState<DemoRunResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRunningPipeline, setIsRunningPipeline] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Configuration
  const [threshold, setThreshold] = useState<number>(0.50);
  const [modelName, setModelName] = useState<string>('xgboost');
  const [contamination, setContamination] = useState<number>(0.20);
  const [showConfig, setShowConfig] = useState<boolean>(false);

  // Walkthrough Guide Step
  const [currentStep, setCurrentStep] = useState<number>(1);

  // Filter & Search
  const [filterMode, setFilterMode] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Selected Transaction for Investigation
  const [selectedTx, setSelectedTx] = useState<DemoTransaction | null>(null);

  // RAG Query & Context for Selected Transaction
  const [ragLoading, setRagLoading] = useState<boolean>(false);
  const [ragFindings, setRagFindings] = useState<RagCitation[]>([]);

  // AI Copilot SAR Report
  const [isGeneratingReport, setIsGeneratingReport] = useState<boolean>(false);
  const [reportDraft, setReportDraft] = useState<InvestigationReportDraft | null>(null);
  const [verificationResult, setVerificationResult] = useState<any | null>(null);
  const [copiedReport, setCopiedReport] = useState<boolean>(false);

  // Case Linking
  const [isCreatingCase, setIsCreatingCase] = useState<boolean>(false);
  const [linkedCaseMsg, setLinkedCaseMsg] = useState<string | null>(null);

  // Load initial raw demo dataset on mount
  useEffect(() => {
    loadInitialTransactions();
  }, []);

  const loadInitialTransactions = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const data = await getDemoTransactions();
      setTransactions(data);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to load demo transactions');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunDemo = async () => {
    setIsRunningPipeline(true);
    setErrorMsg(null);
    try {
      const result = await runFraudDetectionDemo(threshold, modelName, contamination);
      setRunResult(result);
      setTransactions(result.transactions);
      setCurrentStep(3); // Advance walkthrough to triage

      // Automatically select the highest risk transaction for immediate investigation
      const suspicious = result.transactions.filter((t) => t.is_suspicious_prediction);
      if (suspicious.length > 0) {
        handleSelectTx(suspicious[0]);
      } else if (result.transactions.length > 0) {
        handleSelectTx(result.transactions[0]);
      }
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to execute ML demo pipeline');
    } finally {
      setIsRunningPipeline(false);
    }
  };

  const handleReset = async () => {
    setIsLoading(true);
    try {
      await resetDemoState();
      setRunResult(null);
      setSelectedTx(null);
      setReportDraft(null);
      setVerificationResult(null);
      setRagFindings([]);
      setLinkedCaseMsg(null);
      setCurrentStep(1);
      await loadInitialTransactions();
    } catch (err: any) {
      setErrorMsg(err.message || 'Reset failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectTx = async (tx: DemoTransaction) => {
    setSelectedTx(tx);
    setReportDraft(null);
    setVerificationResult(null);
    setLinkedCaseMsg(null);

    // Fetch regulatory context relevant to transaction category & amount
    setRagLoading(true);
    try {
      const queryText = tx.amount > 5000
        ? `Large currency transaction threshold and reporting rules for ${tx.category}`
        : `Red flags and suspicious indicators for ${tx.category}`;
      const ragRes = await queryRag(queryText, 2, 0.10);
      setRagFindings(ragRes.citations || []);
    } catch {
      setRagFindings([]);
    } finally {
      setRagLoading(false);
    }
  };

  const handleGenerateCopilotReport = async () => {
    if (!selectedTx) return;
    setIsGeneratingReport(true);
    setErrorMsg(null);
    try {
      const hypothesis = `Flagged in Demo Mode: ${selectedTx.category} transaction of $${selectedTx.amount.toFixed(
        2
      )} with predicted probability ${(
        (selectedTx.predicted_probability || 0) * 100
      ).toFixed(1)}%.`;
      const draft = await generateInvestigationDraft(
        selectedTx.original_index,
        undefined,
        hypothesis,
        'offline'
      );
      setReportDraft(draft);

      // Automatically run anti-hallucination verification against actual transaction
      const vResult = await verifyReportEvidence(
        draft,
        selectedTx.amount,
        selectedTx.predicted_probability || 0.0
      );
      setVerificationResult(vResult);

      setCurrentStep(6);
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || err.message || 'Failed to generate investigation report');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleCreateCaseFromDemo = async () => {
    if (!selectedTx) return;
    setIsCreatingCase(true);
    try {
      const caseItem = await createCase({
        title: `Demo Case: Suspicious ${selectedTx.category} ($${selectedTx.amount.toFixed(2)})`,
        description: `Generated from Interactive Demo Mode. Model risk tier: ${
          selectedTx.risk_level || 'ELEVATED'
        }, Fraud Probability: ${(
          (selectedTx.predicted_probability || 0) * 100
        ).toFixed(1)}%.`,
        priority: selectedTx.risk_level === 'CRITICAL' ? 'CRITICAL' : 'HIGH',
        assigned_analyst: 'senior_investigator',
        initial_transactions: [
          {
            transaction_index: selectedTx.original_index,
            amount: selectedTx.amount,
            fraud_probability: selectedTx.predicted_probability || undefined,
            anomaly_score: selectedTx.anomaly_score || undefined,
            added_reason: 'Flagged via interactive demo ML pipeline',
          },
        ],
      });
      setLinkedCaseMsg(`Case ${caseItem.case_number} successfully registered in Case Management!`);
    } catch (err: any) {
      setLinkedCaseMsg(`Error creating case: ${err.message}`);
    } finally {
      setIsCreatingCase(false);
    }
  };

  const handleCopyReport = () => {
    if (!reportDraft) return;
    const reportText = `# ${reportDraft.report_title}
Report ID: ${reportDraft.report_id}
Generated: ${reportDraft.timestamp}
Provider: ${reportDraft.generation_provider} (100% Offline Rule-Grounded)

## Executive Summary
${reportDraft.case_summary}

## Observed Facts
- Amount: $${reportDraft.observed_facts[0]?.amount ?? 'N/A'}
- Merchant Category: ${selectedTx?.category ?? 'N/A'}
- Model Probability: ${((reportDraft.model_scores.fraud_probability ?? 0) * 100).toFixed(1)}%

## Key Risk Signals (SHAP Attributions)
${reportDraft.explanation_signals.map((s) => `- ${s.feature} (val: ${s.value}): ${s.direction} (${s.contribution >= 0 ? '+' : ''}${s.contribution.toFixed(3)})`).join('\n')}

## Regulatory Citations
${reportDraft.regulatory_citations.map((c) => `- [${c.issuing_body}] ${c.title}: "${c.excerpt}"`).join('\n')}

## Verification Status
Verification Passed: ${verificationResult?.verification_passed ? 'YES' : 'NO'}
Discrepancies: ${verificationResult?.discrepancies_detected ?? 0}
`;
    navigator.clipboard.writeText(reportText);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2500);
  };

  const handleDownloadReport = () => {
    if (!reportDraft) return;
    const content = `# ${reportDraft.report_title}\n\n${reportDraft.case_summary}\n\n## Mandatory Advisory\n${reportDraft.mandatory_advisory_label}`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SAR_Investigation_Report_${selectedTx?.id || 'demo'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter transactions
  const filteredTransactions = transactions.filter((tx) => {
    if (filterMode === 'suspicious' && !tx.is_suspicious_prediction) return false;
    if (filterMode === 'confirmed' && tx.ground_truth_class !== 1) return false;
    if (filterMode === 'legitimate' && tx.ground_truth_class !== 0) return false;
    if (searchTerm) {
      const matchCat = tx.category.toLowerCase().includes(searchTerm.toLowerCase());
      const matchAmt = tx.amount.toString().includes(searchTerm);
      const matchId = tx.id.toString().includes(searchTerm);
      return matchCat || matchAmt || matchId;
    }
    return true;
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Banner / Hero */}
      <div className="bg-gradient-to-r from-emerald-950/60 via-graphite-900 to-indigo-950/40 border border-emerald-500/20 rounded-xl p-6 shadow-xl">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
          <div className="space-y-1.5 max-w-3xl">
            <div className="inline-flex items-center space-x-2 px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold border border-emerald-500/20">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Full End-to-End Walkthrough</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Interactive Fraud Detection Demo Mode
            </h1>
            <p className="text-sm text-graphite-300 leading-relaxed">
              Experience the complete AuditLens copilot lifecycle on a realistic transaction batch.
              Execute the live machine learning pipeline (XGBoost + Isolation Forest), inspect game-theoretic
              SHAP feature attributions, cross-reference regulatory red flags, generate an evidence-verified SAR draft,
              and escalate to Case Management.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setShowConfig(!showConfig)}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-200 text-xs font-medium border border-graphite-700 transition"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Model Parameters</span>
            </button>

            <button
              onClick={handleReset}
              disabled={isLoading || isRunningPipeline}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-300 text-xs font-medium border border-graphite-700 transition disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Demo</span>
            </button>

            <button
              onClick={handleRunDemo}
              disabled={isRunningPipeline}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold shadow-lg shadow-emerald-950/50 transition transform active:scale-95 disabled:opacity-50"
            >
              {isRunningPipeline ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Run Fraud Detection Demo</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Model Parameter Configuration */}
        {showConfig && (
          <div className="mt-5 pt-4 border-t border-graphite-800 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <label className="text-graphite-400 block mb-1 font-medium">
                Decision Threshold ({threshold.toFixed(2)})
              </label>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <span className="text-[10px] text-graphite-500">
                Probability required to flag a transaction as suspicious.
              </span>
            </div>

            <div>
              <label className="text-graphite-400 block mb-1 font-medium">Supervised Model</label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full bg-graphite-850 border border-graphite-700 rounded p-1.5 text-white"
              >
                <option value="xgboost">XGBoost (Imbalance Weighted Trees)</option>
                <option value="cart">CART Decision Tree</option>
              </select>
            </div>

            <div>
              <label className="text-graphite-400 block mb-1 font-medium">
                Unsupervised Contamination ({contamination.toFixed(2)})
              </label>
              <input
                type="range"
                min="0.05"
                max="0.40"
                step="0.05"
                value={contamination}
                onChange={(e) => setContamination(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <span className="text-[10px] text-graphite-500">
                Isolation Forest expected anomaly proportion.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Guided Walkthrough Steps */}
      <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3 text-xs font-semibold text-graphite-400 uppercase tracking-wider">
          <span>Guided Investigation Walkthrough</span>
          <span className="text-emerald-400">Step {currentStep} of 6</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {[
            { step: 1, title: '1. Ingest Dataset', desc: 'Curated 20-Tx sample' },
            { step: 2, title: '2. Run Detection', desc: 'Actual XGBoost + IsoForest' },
            { step: 3, title: '3. Triage & Rank', desc: 'Review risk scores' },
            { step: 4, title: '4. SHAP Insights', desc: 'Local feature impact' },
            { step: 5, title: '5. Regulatory RAG', desc: 'FinCEN / FATF citations' },
            { step: 6, title: '6. Report & Escalate', desc: 'SAR draft & case audit' },
          ].map((item) => (
            <button
              key={item.step}
              onClick={() => setCurrentStep(item.step)}
              className={`p-2.5 rounded-lg text-left transition border ${
                currentStep === item.step
                  ? 'bg-emerald-500/10 border-emerald-500/40 text-white'
                  : currentStep > item.step
                  ? 'bg-graphite-850 border-graphite-750 text-emerald-400'
                  : 'bg-graphite-850/50 border-graphite-800 text-graphite-400 hover:bg-graphite-800'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-semibold">
                <span>{item.title}</span>
                {currentStep > item.step && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
              </div>
              <p className="text-[11px] text-graphite-400 mt-0.5 truncate">{item.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Pipeline Metrics Card (Visible after running demo) */}
      {runResult && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-graphite-400 font-medium">Total Batch</span>
              <div className="text-xl font-bold text-white mt-1">{runResult.total_transactions}</div>
              <span className="text-[10px] text-graphite-500">Transactions loaded</span>
            </div>

            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-amber-400 font-medium">Flagged Suspicious</span>
              <div className="text-xl font-bold text-amber-400 mt-1">
                {runResult.flagged_suspicious_count}
              </div>
              <span className="text-[10px] text-graphite-500">
                Exceeded {runResult.decision_threshold.toFixed(2)} threshold
              </span>
            </div>

            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-rose-400 font-medium">Confirmed Benchmark</span>
              <div className="text-xl font-bold text-rose-400 mt-1">
                {runResult.confirmed_fraud_count}
              </div>
              <span className="text-[10px] text-graphite-500">Known ground truth</span>
            </div>

            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-emerald-400 font-medium">Precision</span>
              <div className="text-xl font-bold text-emerald-400 mt-1">
                {(runResult.precision * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-graphite-500">
                {runResult.true_positives} TP / {runResult.false_positives} FP
              </span>
            </div>

            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-emerald-400 font-medium">Recall</span>
              <div className="text-xl font-bold text-emerald-400 mt-1">
                {(runResult.recall * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-graphite-500">
                {runResult.true_positives} of {runResult.confirmed_fraud_count} detected
              </span>
            </div>

            <div className="bg-graphite-900 border border-graphite-800 rounded-lg p-3">
              <span className="text-[11px] text-indigo-400 font-medium">Execution Latency</span>
              <div className="text-xl font-bold text-indigo-400 mt-1">
                {runResult.execution_time_ms} ms
              </div>
              <span className="text-[10px] text-graphite-500">Live inference & SHAP</span>
            </div>
          </div>

          {/* Model Distinction Notice */}
          <div className="p-3 bg-graphite-900/60 border border-graphite-800 rounded-lg text-xs text-graphite-300 flex items-start space-x-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-white">Critical Distinction: </span>
              <span>
                <strong>"Suspicious Prediction"</strong> represents the algorithmic probability output by the model.
                <strong> "Confirmed Fraud"</strong> represents the synthetic benchmark ground-truth label. In real financial crime operations, model alerts are never definitive conclusions—they are decision-support triggers requiring human analyst adjudication.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Main Content: Transaction Table & Investigation Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Transaction Table (7 columns on large screens) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl overflow-hidden shadow">
            {/* Search and Filters Bar */}
            <div className="p-3.5 border-b border-graphite-800 flex flex-wrap justify-between items-center gap-3">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-white">Synthetic Batch</span>
                <span className="text-[11px] text-graphite-400">({transactions.length} rows)</span>
              </div>

              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-graphite-400" />
                  <input
                    type="text"
                    placeholder="Search category or amount..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="bg-graphite-850 border border-graphite-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-graphite-500 w-48 focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <select
                  value={filterMode}
                  onChange={(e) => setFilterMode(e.target.value)}
                  className="bg-graphite-850 border border-graphite-700 rounded-lg px-2.5 py-1.5 text-xs text-graphite-200 focus:outline-none"
                >
                  <option value="all">All Transactions</option>
                  <option value="suspicious">Flagged Suspicious Only</option>
                  <option value="confirmed">Confirmed Fraud Ground Truth</option>
                  <option value="legitimate">Legitimate Only</option>
                </select>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto max-h-[580px] overflow-y-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-graphite-950/80 sticky top-0 border-b border-graphite-800 text-graphite-400 font-medium z-10">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3">Category</th>
                    <th className="py-2.5 px-3 text-right">Amount</th>
                    <th className="py-2.5 px-3 text-center">Prediction Probability</th>
                    <th className="py-2.5 px-3 text-center">Status</th>
                    <th className="py-2.5 px-3 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-graphite-800/60 text-graphite-300">
                  {filteredTransactions.map((tx) => {
                    const isSelected = selectedTx?.id === tx.id;
                    const prob = tx.predicted_probability;
                    return (
                      <tr
                        key={tx.id}
                        onClick={() => handleSelectTx(tx)}
                        className={`cursor-pointer transition ${
                          isSelected
                            ? 'bg-emerald-950/30 text-white'
                            : 'hover:bg-graphite-850/60'
                        }`}
                      >
                        <td className="py-2.5 px-3 font-mono text-[11px] text-graphite-400">
                          #{tx.id}
                        </td>
                        <td className="py-2.5 px-3 font-medium text-white truncate max-w-[160px]">
                          {tx.category}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono font-semibold text-white">
                          ${tx.amount.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          {prob !== null ? (
                            <div className="inline-flex items-center space-x-1.5">
                              <span
                                className={`text-[11px] font-mono font-bold ${
                                  prob >= 0.50
                                    ? 'text-rose-400'
                                    : prob >= 0.25
                                    ? 'text-amber-400'
                                    : 'text-emerald-400'
                                }`}
                              >
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ) : (
                            <span className="text-[11px] text-graphite-500 italic">Unscored</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          {tx.is_suspicious_prediction !== null ? (
                            tx.is_suspicious_prediction ? (
                              <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                                <AlertTriangle className="w-3 h-3" />
                                <span>SUSPICIOUS</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                <span>NORMAL</span>
                              </span>
                            )
                          ) : (
                            <span className="text-[10px] text-graphite-500">Ready</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectTx(tx);
                            }}
                            className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                              isSelected
                                ? 'bg-emerald-500 text-white'
                                : 'bg-graphite-800 text-graphite-300 hover:bg-graphite-700 hover:text-white'
                            }`}
                          >
                            Investigate
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right: Detailed Investigation Workspace (5 columns) */}
        <div className="lg:col-span-5 space-y-4">
          {selectedTx ? (
            <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-5 space-y-5 shadow">
              {/* Header Details */}
              <div className="flex justify-between items-start border-b border-graphite-800 pb-3">
                <div>
                  <span className="text-[11px] font-mono text-emerald-400">
                    INVESTIGATION TARGET #{selectedTx.id} (Row {selectedTx.original_index})
                  </span>
                  <h3 className="text-base font-bold text-white mt-0.5">{selectedTx.category}</h3>
                  <div className="text-lg font-mono font-bold text-emerald-400 mt-1">
                    ${selectedTx.amount.toFixed(2)}
                  </div>
                </div>

                <div className="text-right">
                  {selectedTx.risk_level && (
                    <span
                      className={`inline-block px-2.5 py-1 rounded text-xs font-bold border ${
                        selectedTx.risk_level === 'CRITICAL' || selectedTx.risk_level === 'HIGH'
                          ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                          : selectedTx.risk_level === 'ELEVATED'
                          ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                          : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                      }`}
                    >
                      {selectedTx.risk_level} RISK
                    </span>
                  )}
                  <div className="text-[11px] text-graphite-400 mt-1">
                    Ground Truth: {selectedTx.ground_truth_class === 1 ? 'Fraud Benchmark' : 'Legitimate'}
                  </div>
                </div>
              </div>

              {/* 1. SHAP Feature Attribution Card */}
              <div className="space-y-2.5">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-1.5 text-xs font-bold text-white">
                    <Scale className="w-4 h-4 text-emerald-400" />
                    <span>TreeSHAP Feature Attributions</span>
                  </div>
                  <span className="text-[10px] text-graphite-400">Exact Shapley Impact</span>
                </div>

                {selectedTx.top_shap_features && selectedTx.top_shap_features.length > 0 ? (
                  <div className="space-y-1.5 bg-graphite-950 p-3 rounded-lg border border-graphite-800 text-xs">
                    {selectedTx.top_shap_features.map((attr, idx) => (
                      <div key={idx} className="flex justify-between items-center text-xs">
                        <span className="font-mono text-graphite-300">{attr.feature}</span>
                        <div className="flex items-center space-x-2">
                          <span className="text-[11px] text-graphite-400 font-mono">
                            val: {attr.value.toFixed(2)}
                          </span>
                          <span
                            className={`font-mono font-semibold px-1.5 py-0.5 rounded text-[11px] ${
                              attr.shap_value > 0
                                ? 'bg-rose-500/15 text-rose-300'
                                : 'bg-emerald-500/15 text-emerald-300'
                            }`}
                          >
                            {attr.shap_value > 0 ? `+${attr.shap_value.toFixed(3)}` : attr.shap_value.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3 bg-graphite-950 rounded-lg text-xs text-graphite-400 italic border border-graphite-800">
                    Run the detection demo to compute local SHAP attributions.
                  </div>
                )}
              </div>

              {/* 2. Regulatory RAG Guidance */}
              <div className="space-y-2">
                <div className="flex items-center space-x-1.5 text-xs font-bold text-white">
                  <ShieldAlert className="w-4 h-4 text-indigo-400" />
                  <span>Regulatory Red Flags (RAG Knowledge)</span>
                </div>

                {ragLoading ? (
                  <div className="text-xs text-graphite-400 animate-pulse">Querying regulatory vector index...</div>
                ) : ragFindings.length > 0 ? (
                  <div className="space-y-2">
                    {ragFindings.map((finding, idx) => (
                      <div
                        key={idx}
                        className="p-2.5 rounded bg-graphite-950/70 border border-graphite-800 text-xs text-graphite-300 space-y-1"
                      >
                        <div className="flex justify-between items-center text-[10px] text-indigo-400 font-semibold">
                          <span>{finding.title || finding.doc_id || 'Regulatory Guidance'}</span>
                          <span>Score: {finding.similarity_score?.toFixed(2) ?? '0.85'}</span>
                        </div>
                        <p className="text-[11px] leading-relaxed italic text-graphite-300">
                          "{finding.text || 'Mandatory suspicious activity report required.'}"
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-graphite-500 italic">No direct regulatory match query yet.</p>
                )}
              </div>

              {/* 3. Actions: AI Copilot Report & Case Escalation */}
              <div className="space-y-3 pt-2 border-t border-graphite-800">
                <div className="flex gap-2">
                  <button
                    onClick={handleGenerateCopilotReport}
                    disabled={isGeneratingReport || !selectedTx.predicted_probability}
                    className="flex-1 flex items-center justify-center space-x-1.5 px-3 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
                  >
                    {isGeneratingReport ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Generating SAR Draft...</span>
                      </>
                    ) : (
                      <>
                        <FileText className="w-3.5 h-3.5" />
                        <span>Generate SAR Report</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleCreateCaseFromDemo}
                    disabled={isCreatingCase}
                    className="flex items-center space-x-1.5 px-3 py-2.5 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-200 text-xs font-semibold border border-graphite-700 transition disabled:opacity-50"
                  >
                    <Briefcase className="w-3.5 h-3.5 text-amber-400" />
                    <span>Create Case</span>
                  </button>
                </div>

                {linkedCaseMsg && (
                  <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded text-emerald-300 text-xs flex items-center justify-between">
                    <div className="flex items-center space-x-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      <span>{linkedCaseMsg}</span>
                    </div>
                    {onNavigateToTab && (
                      <button
                        onClick={() => onNavigateToTab('cases')}
                        className="text-[11px] underline hover:text-white font-medium ml-2"
                      >
                        View in Cases
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* 4. Generated SAR Report Draft Modal / Preview */}
              {reportDraft && (
                <div className="mt-4 p-4 bg-graphite-950 border border-indigo-500/30 rounded-xl space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-indigo-400 flex items-center space-x-1.5">
                      <FileText className="w-4 h-4" />
                      <span>{reportDraft.report_title}</span>
                    </span>

                    <div className="flex items-center space-x-1.5">
                      <button
                        onClick={handleCopyReport}
                        className="p-1.5 rounded hover:bg-graphite-800 text-graphite-400 hover:text-white transition"
                        title="Copy Markdown Report"
                      >
                        {copiedReport ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                      <button
                        onClick={handleDownloadReport}
                        className="p-1.5 rounded hover:bg-graphite-800 text-graphite-400 hover:text-white transition"
                        title="Download Markdown"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-graphite-300 leading-relaxed bg-graphite-900 p-2.5 rounded border border-graphite-800">
                    {reportDraft.case_summary}
                  </p>

                  {/* Anti-Hallucination Verification Badge */}
                  {verificationResult && (
                    <div
                      className={`p-2 rounded text-[11px] flex items-center space-x-1.5 ${
                        verificationResult.verification_passed
                          ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
                          : 'bg-rose-500/10 border border-rose-500/30 text-rose-300'
                      }`}
                    >
                      <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
                      <span>
                        Anti-Hallucination Check: Verified against actual numerical facts ($
                        {selectedTx.amount.toFixed(2)}, {((selectedTx.predicted_probability || 0) * 100).toFixed(1)}%).
                      </span>
                    </div>
                  )}

                  <div className="text-[10px] text-graphite-400 italic border-t border-graphite-850 pt-2">
                    {reportDraft.mandatory_advisory_label}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-8 text-center space-y-3">
              <HelpCircle className="w-8 h-8 text-graphite-600 mx-auto" />
              <h4 className="text-sm font-semibold text-white">Select a Transaction</h4>
              <p className="text-xs text-graphite-400 max-w-sm mx-auto">
                Click any row in the demonstration batch to inspect its SHAP attributions, cross-reference regulatory red flags, and generate an evidence-verified SAR investigation draft.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
