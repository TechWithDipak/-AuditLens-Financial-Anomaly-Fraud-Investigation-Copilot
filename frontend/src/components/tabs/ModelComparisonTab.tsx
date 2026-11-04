import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Sliders,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  LineChart,
  Line,
} from 'recharts';
import { ModelComparisonResult, ThresholdAnalysisResult } from '../../types';
import { analyzeThresholds, compareModels } from '../../services/api';

export const ModelComparisonTab: React.FC = () => {
  const [comparison, setComparison] = useState<ModelComparisonResult | null>(null);
  const [isLoadingComparison, setIsLoadingComparison] = useState(false);

  // Threshold State
  const [costFp, setCostFp] = useState<number>(10.0);
  const [costFn, setCostFn] = useState<number>(500.0);
  const [thresholds, setThresholds] = useState<ThresholdAnalysisResult | null>(null);
  const [isLoadingThresholds, setIsLoadingThresholds] = useState(false);

  useEffect(() => {
    handleRunComparison();
    handleAnalyzeThresholds();
  }, []);

  const handleRunComparison = async () => {
    setIsLoadingComparison(true);
    try {
      const res = await compareModels();
      setComparison(res);
    } finally {
      setIsLoadingComparison(false);
    }
  };

  const handleAnalyzeThresholds = async () => {
    setIsLoadingThresholds(true);
    try {
      const res = await analyzeThresholds(costFp, costFn);
      setThresholds(res);
    } finally {
      setIsLoadingThresholds(false);
    }
  };

  const chartData = comparison?.models.map((m) => ({
    name: m.name.toUpperCase(),
    'PR-AUC': parseFloat(m.metrics.pr_auc.toFixed(3)),
    'ROC-AUC': parseFloat(m.metrics.roc_auc.toFixed(3)),
    F1: parseFloat(m.metrics.f1.toFixed(3)),
    Recall: parseFloat(m.metrics.recall.toFixed(3)),
  }));

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-graphite-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            <span>Fair Model Comparison & Threshold Tuning</span>
          </h2>
          <p className="text-xs text-graphite-400 mt-0.5">
            Strict identical-test-split evaluation comparing CART, XGBoost, and Ensembles.
          </p>
        </div>

        <button
          onClick={handleRunComparison}
          disabled={isLoadingComparison}
          className="px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoadingComparison ? 'animate-spin' : ''}`} />
          <span>{isLoadingComparison ? 'Evaluating...' : 'Re-Run Comparison'}</span>
        </button>
      </div>

      {/* Model Comparison Table & Chart */}
      {comparison && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Table */}
            <div className="lg:col-span-2 rounded-xl border border-graphite-800 bg-graphite-900 overflow-hidden shadow-lg">
              <div className="p-3 bg-graphite-950 border-b border-graphite-800 text-xs font-semibold text-white">
                Identical Split Performance Matrix
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-graphite-300 font-mono">
                  <thead className="bg-graphite-950 text-graphite-400 uppercase text-[10px]">
                    <tr>
                      <th className="p-3">Model</th>
                      <th className="p-3">PR-AUC</th>
                      <th className="p-3">ROC-AUC</th>
                      <th className="p-3">F1</th>
                      <th className="p-3">Recall</th>
                      <th className="p-3">Precision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-graphite-800/60 text-xs">
                    {comparison.models.map((m) => (
                      <tr key={m.name} className="hover:bg-graphite-850">
                        <td className="p-3 font-bold text-white uppercase">{m.name}</td>
                        <td className="p-3 text-amber-400 font-bold">{m.metrics.pr_auc.toFixed(4)}</td>
                        <td className="p-3">{m.metrics.roc_auc.toFixed(4)}</td>
                        <td className="p-3 text-emerald-400 font-semibold">{m.metrics.f1.toFixed(4)}</td>
                        <td className="p-3">{m.metrics.recall.toFixed(4)}</td>
                        <td className="p-3">{m.metrics.precision.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Metric Chart */}
            <div className="rounded-xl border border-graphite-800 bg-graphite-900 p-4 flex flex-col justify-between">
              <div className="text-xs font-semibold text-white mb-2">Metrics Comparison</div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 9 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 9 }} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        fontSize: 10,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Bar dataKey="PR-AUC" fill="#f59e0b" />
                    <Bar dataKey="F1" fill="#10b981" />
                    <Bar dataKey="Recall" fill="#38bdf8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Threshold Analysis Section */}
      <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              <span>Operational Decision Threshold & Cost Analysis</span>
            </h3>
            <p className="text-xs text-graphite-400">
              Evaluates trade-offs across 12 decision thresholds ($\theta \in [0.05, 0.95]$) given financial penalty costs.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-1 font-mono">
              <span className="text-graphite-400">FP Cost:</span>
              <input
                type="number"
                value={costFp}
                onChange={(e) => setCostFp(parseFloat(e.target.value))}
                className="w-16 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white text-xs"
              />
            </div>
            <div className="flex items-center space-x-1 font-mono">
              <span className="text-graphite-400">FN Cost:</span>
              <input
                type="number"
                value={costFn}
                onChange={(e) => setCostFn(parseFloat(e.target.value))}
                className="w-16 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white text-xs"
              />
            </div>
            <button
              onClick={handleAnalyzeThresholds}
              disabled={isLoadingThresholds}
              className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium"
            >
              Recalculate
            </button>
          </div>
        </div>

        {thresholds && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-3 font-mono text-xs">
              <div className="text-white font-semibold">Recommended Operating Points</div>
              <div className="p-2.5 rounded bg-graphite-900 border border-graphite-800">
                <div className="text-graphite-400">Optimal Cost Threshold:</div>
                <div className="text-base font-bold text-emerald-400 mt-0.5">
                  θ* = {thresholds.min_cost_threshold.toFixed(2)}
                </div>
                <p className="text-[10px] text-graphite-500 mt-1">
                  Minimizes total operational false-positive friction and fraud losses.
                </p>
              </div>

              <div className="p-2.5 rounded bg-graphite-900 border border-graphite-800">
                <div className="text-graphite-400">Max F1 Threshold:</div>
                <div className="text-base font-bold text-amber-400 mt-0.5">
                  θ = {thresholds.optimal_f1_threshold.toFixed(2)}
                </div>
                <p className="text-[10px] text-graphite-500 mt-1">
                  Maximizes balanced precision-recall harmonic mean.
                </p>
              </div>
            </div>

            <div className="md:col-span-2 p-4 rounded-lg bg-graphite-950 border border-graphite-800">
              <div className="text-xs font-semibold text-white mb-2">
                Expected Financial Cost vs Threshold Curve
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={thresholds.thresholds}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="threshold" stroke="#64748b" tick={{ fontSize: 9 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 9 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        fontSize: 10,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="expected_cost"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      name="Expected Loss ($)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
