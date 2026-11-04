import React, { useState } from 'react';
import {
  FileText,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  HelpCircle,
  BookOpen,
  Briefcase,
} from 'lucide-react';
import { InvestigationReportDraft } from '../../types';
import {
  addCaseNote,
  generateInvestigationDraft,
  verifyReportEvidence,
} from '../../services/api';

export const AssistantTab: React.FC = () => {
  const [transactionIndex, setTransactionIndex] = useState<number>(12);
  const [caseId, setCaseId] = useState<string>('');
  const [hypothesis, setHypothesis] = useState<string>('Sudden velocity burst flagged by HMM sequence monitor.');
  const [provider, setProvider] = useState<string>('offline');

  const [draft, setDraft] = useState<InvestigationReportDraft | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);

  // Verification state
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState<string | null>(null);

  // Attach state
  const [isAttaching, setIsAttaching] = useState(false);
  const [attachMsg, setAttachMsg] = useState<string | null>(null);

  const handleGenerate = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsGenerating(true);
    setGenError(null);
    setVerifyStatus(null);
    setAttachMsg(null);
    try {
      const res = await generateInvestigationDraft(
        transactionIndex,
        caseId ? parseInt(caseId, 10) : undefined,
        hypothesis.trim() || undefined,
        provider
      );
      setDraft(res);
    } catch (err: any) {
      setGenError(err.response?.data?.detail || err.message || 'Draft generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunVerify = async () => {
    if (!draft || !draft.observed_facts.length) return;
    setIsVerifying(true);
    try {
      const res = await verifyReportEvidence(
        draft,
        draft.observed_facts[0].amount,
        draft.model_scores.fraud_probability
      );
      if (res.verification_passed) {
        setVerifyStatus('All claims and numerical citations strictly verified against ground truth.');
      } else {
        setVerifyStatus(`Discrepancies detected: ${res.warnings.join('; ')}`);
      }
    } catch (err: any) {
      setVerifyStatus(`Verification check failed: ${err.message}`);
    } finally {
      setIsVerifying(false);
    }
  };

  const handleAttachToCase = async () => {
    if (!draft || !draft.case_id) {
      setAttachMsg('Please specify a valid Case ID to attach this draft.');
      return;
    }
    setIsAttaching(true);
    try {
      await addCaseNote(
        draft.case_id,
        `[AI Investigation Report #${draft.report_id}]\n${draft.case_summary}\n\nTop Risk Drivers: ${draft.explanation_signals.map(s => s.feature).join(', ')}`,
        'assistant_copilot'
      );
      setAttachMsg(`Successfully attached to Case #${draft.case_id} notes!`);
    } catch (err: any) {
      setAttachMsg(`Failed to attach: ${err.message}`);
    } finally {
      setIsAttaching(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="p-6 rounded-xl bg-gradient-to-r from-graphite-900 via-graphite-850 to-graphite-900 border border-graphite-800 shadow-lg">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-semibold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                Phase 11 Operational
              </span>
              <span className="text-xs font-mono text-graphite-400">
                100% Offline Rule Engine (Zero API Keys) • Grounded Citations • Human Review Enforced
              </span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white mt-1.5 flex items-center space-x-2">
              <FileText className="w-6 h-6 text-cyan-400" />
              <span>AI Investigation Assistant (Evidence-Grounded Drafter)</span>
            </h2>
            <p className="text-xs text-graphite-400 mt-1 max-w-2xl leading-relaxed">
              Synthesizes observed transaction telemetry, local SHAP attribution values, and verified RAG
              regulatory passages into a structured investigation draft. Every draft is strictly labeled for human review.
            </p>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
        <div className="text-xs font-semibold text-white">Generate Structured Investigation Draft</div>

        <form onSubmit={handleGenerate} className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <div>
            <label className="text-graphite-400 font-semibold block mb-1">
              Transaction Index *
            </label>
            <input
              type="number"
              min="0"
              required
              value={transactionIndex}
              onChange={(e) => setTransactionIndex(parseInt(e.target.value, 10))}
              className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-graphite-400 font-semibold block mb-1">
              Case ID (Optional)
            </label>
            <input
              type="number"
              placeholder="e.g. 1"
              value={caseId}
              onChange={(e) => setCaseId(e.target.value)}
              className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-graphite-400 font-semibold block mb-1">
              Provider Mode
            </label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white"
            >
              <option value="offline">Offline Rule Engine (0 API Keys)</option>
              <option value="gemini">Gemini (Offline Fallback)</option>
              <option value="openai">OpenAI (Offline Fallback)</option>
            </select>
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              disabled={isGenerating}
              className="w-full px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-1.5 shadow transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
              <span>{isGenerating ? 'Synthesizing...' : 'Draft Investigation'}</span>
            </button>
          </div>

          <div className="md:col-span-4">
            <label className="text-graphite-400 font-semibold block mb-1">
              Analyst Preliminary Hypothesis (Sanitized against Prompt Injection)
            </label>
            <input
              type="text"
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="e.g. Card-Not-Present anomaly observed after high velocity sequence..."
              className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-white"
            />
          </div>
        </form>

        {genError && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
            {genError}
          </div>
        )}
      </div>

      {/* Generated Report Display */}
      {draft && (
        <div className="space-y-6">
          {/* Mandatory Advisory Notice Banner */}
          <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 space-y-1.5">
            <div className="flex items-center space-x-2 font-mono font-bold text-xs uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
              <span>Mandatory Advisory Notice: Human Review Required</span>
            </div>
            <p className="text-xs leading-relaxed text-amber-200">
              {draft.mandatory_advisory_label}
            </p>
          </div>

          {/* Action Bar */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 bg-graphite-900 p-3 rounded-xl border border-graphite-800 text-xs">
            <div className="flex items-center space-x-2 font-mono">
              <span className="font-bold text-white">{draft.report_title}</span>
              <span className="text-[10px] text-graphite-400">({draft.report_id})</span>
              <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px]">
                {draft.generation_provider}
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleRunVerify}
                disabled={isVerifying}
                className="px-3 py-1.5 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-graphite-200 border border-graphite-700 font-medium flex items-center space-x-1 transition"
              >
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Verify Claims</span>
              </button>

              <button
                onClick={handleAttachToCase}
                disabled={isAttaching}
                className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium flex items-center space-x-1 shadow transition"
              >
                <Briefcase className="w-3.5 h-3.5" />
                <span>Attach to Case</span>
              </button>
            </div>
          </div>

          {verifyStatus && (
            <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800 text-xs text-graphite-300 font-mono">
              {verifyStatus}
            </div>
          )}

          {attachMsg && (
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300 font-mono">
              {attachMsg}
            </div>
          )}

          {/* Report Sections */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left 2 Cols: Summary, Facts, Scores, Signals */}
            <div className="lg:col-span-2 space-y-6">
              {/* 1. Case Summary */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-graphite-400">
                  1. Executive Case Summary
                </h4>
                <p className="text-xs text-graphite-200 font-sans leading-relaxed bg-graphite-950 p-3.5 rounded-lg border border-graphite-800/80">
                  {draft.case_summary}
                </p>
              </div>

              {/* 2. Observed Facts */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-graphite-400">
                  2. Observed Transaction Facts
                </h4>
                <div className="divide-y divide-graphite-800/80 rounded border border-graphite-800 bg-graphite-950 overflow-hidden font-mono text-xs">
                  {draft.observed_facts.map((f, i) => (
                    <div key={i} className="p-3 flex justify-between items-center">
                      <div>
                        <span className="text-white font-bold">Tx #{f.transaction_index}</span>
                        <span className="text-emerald-400 ml-2 font-bold">${f.amount.toFixed(2)}</span>
                        {f.context && (
                          <div className="text-[10px] text-graphite-400 font-sans mt-0.5">
                            {f.context}
                          </div>
                        )}
                      </div>
                      <div className="text-[10px] text-graphite-400 font-sans">
                        Category: {f.merchant_category}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 3. Model Scores Breakdown */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <div className="flex justify-between items-center">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-graphite-400">
                    3. Exact Model Score Breakdown
                  </h4>
                  <span className="text-[10px] text-graphite-500 font-mono">
                    Strict Taxonomy Separation
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 font-mono text-xs">
                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-graphite-400 text-[10px]">Supervised Fraud Prob</div>
                    <div className="text-base font-bold text-rose-400 mt-0.5">
                      {(draft.model_scores.fraud_probability * 100).toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-graphite-500 mt-1">
                      {draft.model_scores.supervised_model_name}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-graphite-400 text-[10px]">GMM Log-Likelihood</div>
                    <div className="text-base font-bold text-indigo-400 mt-0.5">
                      {draft.model_scores.gmm_log_likelihood?.toFixed(2)}
                    </div>
                    <div className="text-[9px] text-graphite-500 mt-1">Density rarity metric</div>
                  </div>

                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-graphite-400 text-[10px]">HMM Behavioral State</div>
                    <div className="text-xs font-bold text-white mt-1">
                      {draft.model_scores.hmm_decoded_state}
                    </div>
                    <div className="text-[9px] text-graphite-500 mt-1">Markovian transition</div>
                  </div>

                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-graphite-400 text-[10px]">K-Means Cluster ID</div>
                    <div className="text-base font-bold text-cyan-400 mt-0.5">
                      Cluster {draft.model_scores.kmeans_cluster_id}
                    </div>
                    <div className="text-[9px] text-graphite-500 mt-1">Peer group baseline</div>
                  </div>

                  <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-graphite-400 text-[10px]">Isolation Forest Score</div>
                    <div className="text-base font-bold text-amber-400 mt-0.5">
                      {draft.model_scores.isolation_forest_score?.toFixed(3)}
                    </div>
                    <div className="text-[9px] text-graphite-500 mt-1">Unsupervised tree depth</div>
                  </div>
                </div>
              </div>

              {/* 4. Top SHAP Explanation Signals */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-graphite-400">
                  4. Top SHAP Feature Attribution Signals
                </h4>
                <div className="space-y-1.5 font-mono text-xs">
                  {draft.explanation_signals.map((sig, i) => (
                    <div
                      key={i}
                      className="p-2.5 rounded bg-graphite-950 border border-graphite-800 flex justify-between items-center"
                    >
                      <div>
                        <span className="font-bold text-white">{sig.feature}</span>
                        <span className="text-graphite-500 text-[10px] ml-2">
                          Value = {sig.value.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span
                          className={`font-bold ${
                            sig.direction === 'increases_risk' ? 'text-rose-400' : 'text-cyan-400'
                          }`}
                        >
                          {sig.contribution > 0 ? '+' : ''}
                          {sig.contribution.toFixed(4)}
                        </span>
                        <span className="text-[10px] text-graphite-500">
                          {sig.direction === 'increases_risk' ? '▲ Risk' : '▼ Mitigator'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Column: Citations, Missing Evidence, Questions */}
            <div className="space-y-6">
              {/* 5. Regulatory Citations */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <div className="flex items-center space-x-2 text-white font-semibold text-xs uppercase tracking-wider">
                  <BookOpen className="w-4 h-4 text-indigo-400" />
                  <span>5. Regulatory Citations</span>
                </div>

                {draft.regulatory_citations.map((c, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-lg bg-graphite-950 border border-graphite-800 space-y-1.5 text-xs"
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono font-bold text-indigo-300">
                        {c.issuing_body}
                      </span>
                      <span className="text-[10px] font-mono text-emerald-400 font-bold">
                        {(c.similarity_score * 100).toFixed(0)}% match
                      </span>
                    </div>
                    <div className="font-semibold text-white text-[11px]">{c.title}</div>
                    <div className="text-[10px] text-cyan-400 font-mono">{c.section}</div>
                    <p className="text-[11px] text-graphite-300 font-sans italic bg-graphite-900/60 p-2 rounded border border-graphite-800/80">
                      "{c.excerpt}"
                    </p>
                  </div>
                ))}
              </div>

              {/* 6. Missing Evidence */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <div className="flex items-center space-x-2 text-white font-semibold text-xs uppercase tracking-wider">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <span>6. Missing Evidence & Uncertainty</span>
                </div>
                <ul className="space-y-2 text-xs text-graphite-300 list-disc list-inside">
                  {draft.missing_evidence_and_uncertainty.map((item, i) => (
                    <li key={i} className="leading-relaxed">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>

              {/* 7. Suggested Questions */}
              <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
                <div className="flex items-center space-x-2 text-white font-semibold text-xs uppercase tracking-wider">
                  <HelpCircle className="w-4 h-4 text-cyan-400" />
                  <span>7. Investigator Follow-Up Questions</span>
                </div>
                <div className="space-y-2 text-xs">
                  {draft.suggested_follow_up_questions.map((q, i) => (
                    <div
                      key={i}
                      className="p-2.5 rounded bg-graphite-950 border border-graphite-800 text-graphite-200"
                    >
                      <span className="font-mono text-cyan-400 font-bold mr-1.5">Q{i + 1}:</span>
                      <span>{q}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
