import React, { useState, useEffect } from 'react';
import {
  Database,
  Upload,
  GitFork,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react';
import {
  DatasetSampleResponse,
  EdaSummary,
  SequenceStep,
  SplitResult,
  ValidationReport,
} from '../../types';
import {
  getSampleTransactions,
  getSequenceSample,
  runEda,
  splitDataset,
  validateDatasetUpload,
} from '../../services/api';

export const DataWorkspaceTab: React.FC = () => {
  const [subTab, setSubTab] = useState<'sample' | 'upload' | 'eda' | 'split' | 'sequence'>('sample');

  // Sample State
  const [sampleData, setSampleData] = useState<DatasetSampleResponse | null>(null);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);

  // Upload State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // EDA State
  const [edaData, setEdaData] = useState<EdaSummary | null>(null);
  const [isLoadingEda, setIsLoadingEda] = useState(false);
  const [edaError, setEdaError] = useState<string | null>(null);

  // Split State
  const [splitResult, setSplitResult] = useState<SplitResult | null>(null);
  const [splitType, setSplitType] = useState<'stratified' | 'chronological'>('stratified');
  const [testSize, setTestSize] = useState<number>(0.2);
  const [isLoadingSplit, setIsLoadingSplit] = useState(false);

  // Sequence State
  const [sequenceData, setSequenceData] = useState<SequenceStep[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>('ACC_0001');
  const [isLoadingSequence, setIsLoadingSequence] = useState(false);

  // Load sample on mount
  useEffect(() => {
    loadSample();
  }, []);

  const loadSample = async () => {
    setIsLoadingSample(true);
    setSampleError(null);
    try {
      const data = await getSampleTransactions(20, 0);
      setSampleData(data);
    } catch (e: any) {
      setSampleError(e.message || 'Failed to fetch sample dataset');
    } finally {
      setIsLoadingSample(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploadFile(file);
    setIsUploading(true);
    setUploadError(null);
    try {
      const report = await validateDatasetUpload(file);
      setValidationReport(report);
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || err.message || 'Validation failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleRunEda = async () => {
    setIsLoadingEda(true);
    setEdaError(null);
    try {
      const data = await runEda();
      setEdaData(data);
    } catch (e: any) {
      setEdaError(e.message || 'EDA execution failed');
    } finally {
      setIsLoadingEda(false);
    }
  };

  const handleRunSplit = async () => {
    setIsLoadingSplit(true);
    try {
      const data = await splitDataset(splitType, testSize, 0.1);
      setSplitResult(data);
    } catch (e) {
      // Split error
    } finally {
      setIsLoadingSplit(false);
    }
  };

  const loadSequence = async (accId: string) => {
    setIsLoadingSequence(true);
    try {
      const data = await getSequenceSample(accId);
      setSequenceData(data);
    } catch (e) {
      // Sequence error
    } finally {
      setIsLoadingSequence(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Title & Sub-tabs */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-graphite-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Database className="w-5 h-5 text-emerald-400" />
            <span>Data Workspace & Integrity</span>
          </h2>
          <p className="text-xs text-graphite-400 mt-0.5">
            Ingestion, schema validation, non-leaking splits, and exploratory distribution profiles.
          </p>
        </div>

        <div className="flex items-center space-x-1 bg-graphite-900 p-1 rounded-lg border border-graphite-800 text-xs">
          <button
            onClick={() => setSubTab('sample')}
            className={`px-3 py-1.5 rounded-md font-medium transition ${
              subTab === 'sample'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Sample Fixture
          </button>
          <button
            onClick={() => setSubTab('upload')}
            className={`px-3 py-1.5 rounded-md font-medium transition ${
              subTab === 'upload'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Upload & Validate
          </button>
          <button
            onClick={() => {
              setSubTab('eda');
              if (!edaData) handleRunEda();
            }}
            className={`px-3 py-1.5 rounded-md font-medium transition ${
              subTab === 'eda'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            EDA Profiler
          </button>
          <button
            onClick={() => {
              setSubTab('split');
              if (!splitResult) handleRunSplit();
            }}
            className={`px-3 py-1.5 rounded-md font-medium transition ${
              subTab === 'split'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Partitioning (Zero Leakage)
          </button>
          <button
            onClick={() => {
              setSubTab('sequence');
              if (sequenceData.length === 0) loadSequence(selectedAccount);
            }}
            className={`px-3 py-1.5 rounded-md font-medium transition ${
              subTab === 'sequence'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Sequence Fixtures (HMM)
          </button>
        </div>
      </div>

      {/* SUBTAB 1: SAMPLE TABLE */}
      {subTab === 'sample' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center text-xs">
            <span className="text-graphite-400">
              Showing canonical 1,000-row sample fixture (`creditcard_sample_fixture.csv`).
            </span>
            <button
              onClick={loadSample}
              disabled={isLoadingSample}
              className="flex items-center space-x-1.5 text-emerald-400 hover:text-emerald-300 font-medium"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingSample ? 'animate-spin' : ''}`} />
              <span>Refresh Sample</span>
            </button>
          </div>

          {isLoadingSample && (
            <div className="p-12 text-center text-graphite-400 text-xs">Loading transaction table...</div>
          )}

          {sampleError && (
            <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{sampleError}</span>
            </div>
          )}

          {sampleData && !isLoadingSample && (
            <div className="rounded-xl border border-graphite-800 bg-graphite-900 overflow-hidden shadow-lg">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-graphite-300">
                  <thead className="bg-graphite-950 text-graphite-400 uppercase font-mono text-[10px] border-b border-graphite-800">
                    <tr>
                      <th className="px-4 py-2.5">Row</th>
                      <th className="px-4 py-2.5">Time (s)</th>
                      <th className="px-4 py-2.5">Amount ($)</th>
                      <th className="px-4 py-2.5">Class</th>
                      <th className="px-4 py-2.5">V1</th>
                      <th className="px-4 py-2.5">V2</th>
                      <th className="px-4 py-2.5">V3</th>
                      <th className="px-4 py-2.5">V4</th>
                      <th className="px-4 py-2.5">V5</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-graphite-800/60 font-mono">
                    {sampleData.records.map((r, idx) => (
                      <tr key={idx} className="hover:bg-graphite-850/60 transition">
                        <td className="px-4 py-2 text-graphite-500">{idx}</td>
                        <td className="px-4 py-2 text-white">{r.Time}</td>
                        <td className="px-4 py-2 text-emerald-400 font-semibold">
                          ${r.Amount?.toFixed(2)}
                        </td>
                        <td className="px-4 py-2">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${
                              r.Class === 1
                                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                : 'bg-emerald-500/10 text-emerald-400'
                            }`}
                          >
                            {r.Class === 1 ? 'FRAUD (1)' : 'BENIGN (0)'}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-graphite-400">{r.V1?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-graphite-400">{r.V2?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-graphite-400">{r.V3?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-graphite-400">{r.V4?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-graphite-400">{r.V5?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="p-3 bg-graphite-950 border-t border-graphite-800 text-[11px] text-graphite-400 flex justify-between">
                <span>Displaying first 20 records</span>
                <span>Total Fixture Rows: {sampleData.total_records}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* SUBTAB 2: UPLOAD & VALIDATE */}
      {subTab === 'upload' && (
        <div className="space-y-6">
          <div className="p-6 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <h3 className="text-sm font-semibold text-white">Upload Financial CSV for Validation</h3>
            <p className="text-xs text-graphite-400">
              Validates schema constraints, null checks, duplicate row detection, negative amounts,
              and calculates class imbalance proportions.
            </p>

            <div className="border-2 border-dashed border-graphite-700 hover:border-emerald-500/50 rounded-xl p-8 text-center bg-graphite-950/60 transition">
              <Upload className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
              <label className="cursor-pointer">
                <span className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-md transition inline-block">
                  Select CSV File
                </span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <p className="text-xs text-graphite-400 mt-2">
                {uploadFile ? uploadFile.name : 'Drag & drop or browse a .csv file (Max 100MB)'}
              </p>
            </div>

            {isUploading && (
              <div className="text-xs text-graphite-400 text-center py-2">
                Validating CSV integrity against dataset schema...
              </div>
            )}

            {uploadError && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
                {uploadError}
              </div>
            )}
          </div>

          {validationReport && (
            <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h4 className="text-sm font-semibold text-white">
                    Validation Result: {validationReport.filename}
                  </h4>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {validationReport.is_valid ? 'SCHEMA VALID' : 'SCHEMA REJECTED'}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-graphite-400">Total Rows</div>
                  <div className="text-base font-bold text-white mt-1">
                    {validationReport.row_count.toLocaleString()}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-graphite-400">Total Columns</div>
                  <div className="text-base font-bold text-white mt-1">
                    {validationReport.column_count}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-graphite-400">Duplicate Rows</div>
                  <div className="text-base font-bold text-white mt-1">
                    {validationReport.duplicate_rows}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-graphite-400">Class Balance</div>
                  <div className="text-xs text-white mt-1">
                    Legit: {validationReport.class_balance['0'] || 0} | Fraud:{' '}
                    {validationReport.class_balance['1'] || 0}
                  </div>
                </div>
              </div>

              {validationReport.warnings.length > 0 && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 space-y-1">
                  <div className="text-xs font-semibold text-amber-400">Warnings Detected:</div>
                  <ul className="text-xs text-amber-300 list-disc list-inside">
                    {validationReport.warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* SUBTAB 3: EDA PROFILER */}
      {subTab === 'eda' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <span className="text-xs text-graphite-400">
              Non-destructive exploratory statistical moments and distributions.
            </span>
            <button
              onClick={handleRunEda}
              disabled={isLoadingEda}
              className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center space-x-1.5 transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingEda ? 'animate-spin' : ''}`} />
              <span>Compute EDA Moments</span>
            </button>
          </div>

          {isLoadingEda && (
            <div className="p-12 text-center text-graphite-400 text-xs">Computing statistical moments...</div>
          )}

          {edaError && (
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
              {edaError}
            </div>
          )}

          {edaData && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-graphite-900 border border-graphite-800">
                  <div className="text-graphite-400">Dataset Span</div>
                  <div className="text-base font-bold text-white mt-1 font-mono">
                    {edaData.duration_hours?.toFixed(1)} hrs
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-900 border border-graphite-800">
                  <div className="text-graphite-400">Velocity</div>
                  <div className="text-base font-bold text-white mt-1 font-mono">
                    {edaData.hourly_transaction_rate?.toFixed(1)} tx/hr
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-900 border border-graphite-800">
                  <div className="text-graphite-400">Fraud Prevalence</div>
                  <div className="text-base font-bold text-rose-400 mt-1 font-mono">
                    {(edaData.fraud_rate * 100)?.toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-900 border border-graphite-800">
                  <div className="text-graphite-400">Total Analyzed</div>
                  <div className="text-base font-bold text-white mt-1 font-mono">
                    {edaData.total_rows} rows × {edaData.total_columns} cols
                  </div>
                </div>
              </div>

              {/* Column Summary Table */}
              <div className="rounded-xl border border-graphite-800 bg-graphite-900 overflow-hidden">
                <div className="p-3 bg-graphite-950 border-b border-graphite-800 text-xs font-semibold text-white">
                  Feature Distribution Moments (Mean, Std, Skewness)
                </div>
                <div className="overflow-x-auto max-h-96">
                  <table className="w-full text-left text-xs text-graphite-300">
                    <thead className="bg-graphite-950 text-graphite-400 uppercase font-mono text-[10px] sticky top-0">
                      <tr>
                        <th className="px-4 py-2">Column</th>
                        <th className="px-4 py-2">Mean</th>
                        <th className="px-4 py-2">Median</th>
                        <th className="px-4 py-2">Std Dev</th>
                        <th className="px-4 py-2">Skewness</th>
                        <th className="px-4 py-2">Min</th>
                        <th className="px-4 py-2">Max</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite-800/60 font-mono text-[11px]">
                      {Object.entries(edaData.columns).map(([col, stats]) => (
                        <tr key={col} className="hover:bg-graphite-850">
                          <td className="px-4 py-1.5 font-semibold text-white">{col}</td>
                          <td className="px-4 py-1.5">{stats.mean?.toFixed(4)}</td>
                          <td className="px-4 py-1.5">{stats.median?.toFixed(4)}</td>
                          <td className="px-4 py-1.5">{stats.std?.toFixed(4)}</td>
                          <td
                            className={`px-4 py-1.5 ${
                              Math.abs(stats.skewness) > 2 ? 'text-amber-400 font-semibold' : ''
                            }`}
                          >
                            {stats.skewness?.toFixed(3)}
                          </td>
                          <td className="px-4 py-1.5 text-graphite-400">{stats.min?.toFixed(2)}</td>
                          <td className="px-4 py-1.5 text-graphite-400">{stats.max?.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* SUBTAB 4: PARTITIONING */}
      {subTab === 'split' && (
        <div className="space-y-6">
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <h3 className="text-sm font-semibold text-white">
              Data Partitioning & Leakage Prevention Controls
            </h3>
            <p className="text-xs text-graphite-400 leading-relaxed">
              To prevent data leakage, scalers and transformers are fit strictly on training data
              and applied onto test data. Chronological splits prevent temporal lookahead bias.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-xs text-graphite-400 font-semibold block mb-1">
                  Split Strategy
                </label>
                <select
                  value={splitType}
                  onChange={(e: any) => setSplitType(e.target.value)}
                  className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-xs text-white"
                >
                  <option value="stratified">Stratified (Preserve Class Ratio)</option>
                  <option value="chronological">Chronological (Temporal Time-Order)</option>
                </select>
              </div>

              <div>
                <label className="text-xs text-graphite-400 font-semibold block mb-1">
                  Test Split Ratio
                </label>
                <select
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  className="w-full bg-graphite-950 border border-graphite-800 rounded-lg px-3 py-1.5 text-xs text-white"
                >
                  <option value="0.2">20% Test (0.20)</option>
                  <option value="0.25">25% Test (0.25)</option>
                  <option value="0.3">30% Test (0.30)</option>
                </select>
              </div>

              <div className="flex items-end">
                <button
                  onClick={handleRunSplit}
                  disabled={isLoadingSplit}
                  className="w-full px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs transition"
                >
                  {isLoadingSplit ? 'Computing Splits...' : 'Execute Partitioning'}
                </button>
              </div>
            </div>
          </div>

          {splitResult && (
            <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
              <div className="flex items-center space-x-2">
                <GitFork className="w-5 h-5 text-emerald-400" />
                <h4 className="text-sm font-semibold text-white capitalize">
                  {splitResult.split_type} Split Results
                </h4>
              </div>

              <div className="grid grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-1">
                  <div className="text-graphite-400 font-semibold">Training Set (70%)</div>
                  <div className="text-lg font-bold text-white">{splitResult.train_count} rows</div>
                  <div className="text-[11px] text-emerald-400">
                    Fraud Rate: {(splitResult.train_fraud_rate * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-1">
                  <div className="text-graphite-400 font-semibold">Validation Set (10%)</div>
                  <div className="text-lg font-bold text-white">{splitResult.val_count} rows</div>
                  <div className="text-[11px] text-cyan-400">
                    Fraud Rate: {(splitResult.val_fraud_rate * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-1">
                  <div className="text-graphite-400 font-semibold">Test Set (20%)</div>
                  <div className="text-lg font-bold text-white">{splitResult.test_count} rows</div>
                  <div className="text-[11px] text-amber-400">
                    Fraud Rate: {(splitResult.test_fraud_rate * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 shrink-0" />
                <span>
                  Leakage Guard Verified: Target labels and test split observations were excluded
                  during scaler parameter estimation.
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* SUBTAB 5: SEQUENCE FIXTURES */}
      {subTab === 'sequence' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Account Sequence Behavior Fixture (`sequence_account_fixture.csv`)
              </h3>
              <p className="text-xs text-graphite-400">
                10 accounts × 50 sequence steps designed for Unit 4 Hidden Markov Model analysis.
              </p>
            </div>

            <div className="flex items-center space-x-2 text-xs">
              <span className="text-graphite-400">Account:</span>
              <select
                value={selectedAccount}
                onChange={(e) => {
                  setSelectedAccount(e.target.value);
                  loadSequence(e.target.value);
                }}
                className="bg-graphite-950 border border-graphite-800 rounded px-2.5 py-1 text-white font-mono"
              >
                {Array.from({ length: 10 }).map((_, i) => {
                  const id = `ACC_${String(i + 1).padStart(4, '0')}`;
                  return (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  );
                })}
              </select>
            </div>
          </div>

          {isLoadingSequence && (
            <div className="p-8 text-center text-graphite-400 text-xs">Loading account sequences...</div>
          )}

          <div className="rounded-xl border border-graphite-800 bg-graphite-900 overflow-hidden">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs text-graphite-300 font-mono">
                <thead className="bg-graphite-950 text-graphite-400 uppercase text-[10px] sticky top-0">
                  <tr>
                    <th className="px-4 py-2">Step</th>
                    <th className="px-4 py-2">Account ID</th>
                    <th className="px-4 py-2">Amount ($)</th>
                    <th className="px-4 py-2">Merchant Category</th>
                    <th className="px-4 py-2">Anomaly Indicator</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-graphite-800/60">
                  {sequenceData.map((s, idx) => (
                    <tr key={idx} className="hover:bg-graphite-850">
                      <td className="px-4 py-1.5 text-graphite-500">{s.step}</td>
                      <td className="px-4 py-1.5 text-white font-semibold">{s.account_id}</td>
                      <td className="px-4 py-1.5 text-emerald-400">${s.amount.toFixed(2)}</td>
                      <td className="px-4 py-1.5">
                        {s.merchant_category === 0
                          ? '0 (Low Risk / Grocery)'
                          : s.merchant_category === 1
                          ? '1 (Medium / Retail)'
                          : '2 (High Risk / Crypto)'}
                      </td>
                      <td className="px-4 py-1.5">
                        {s.is_anomaly === 1 ? (
                          <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 text-[10px]">
                            ANOMALOUS BURST
                          </span>
                        ) : (
                          <span className="text-graphite-500 text-[10px]">Normal</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
