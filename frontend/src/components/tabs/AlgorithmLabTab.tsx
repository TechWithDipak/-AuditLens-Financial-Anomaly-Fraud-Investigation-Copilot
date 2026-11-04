import React, { useState } from 'react';
import {
  Cpu,
  Play,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import {
  CartResult,
  EnsembleResult,
  GmmResult,
  HmmForwardResult,
  HmmViterbiResult,
  KMeansResult,
  PcaResult,
  XgboostResult,
  AccountTimelineResponse,
} from '../../types';
import {
  getAccountTimeline,
  runGmm,
  runHmmForward,
  runHmmViterbi,
  runKMeans,
  runPca,
  trainCart,
  trainEnsemble,
  trainXgboost,
} from '../../services/api';

export const AlgorithmLabTab: React.FC = () => {
  const [activeUnit, setActiveUnit] = useState<3 | 4 | 5>(3);

  // Unit 3 State
  const [kClusters, setKClusters] = useState<number>(5);
  const [kmeansResult, setKmeansResult] = useState<KMeansResult | null>(null);
  const [isLoadingKmeans, setIsLoadingKmeans] = useState(false);

  const [gmmComponents, setGmmComponents] = useState<number>(3);
  const [gmmResult, setGmmResult] = useState<GmmResult | null>(null);
  const [isLoadingGmm, setIsLoadingGmm] = useState(false);

  const [pcaResult, setPcaResult] = useState<PcaResult | null>(null);
  const [isLoadingPca, setIsLoadingPca] = useState(false);

  // Unit 4 State
  const [obsInput, setObsInput] = useState<string>('0, 1, 0, 2, 1, 0, 0, 2, 2, 1');
  const [forwardResult, setForwardResult] = useState<HmmForwardResult | null>(null);
  const [viterbiResult, setViterbiResult] = useState<HmmViterbiResult | null>(null);
  const [isLoadingHmm, setIsLoadingHmm] = useState(false);
  const [selectedAcc, setSelectedAcc] = useState<string>('ACC_0001');
  const [timelineData, setTimelineData] = useState<AccountTimelineResponse | null>(null);
  const [isLoadingTimeline, setIsLoadingTimeline] = useState(false);

  // Unit 5 State
  const [cartDepth, setCartDepth] = useState<number>(5);
  const [cartResult, setCartResult] = useState<CartResult | null>(null);
  const [isLoadingCart, setIsLoadingCart] = useState(false);

  const [xgbEstimators, setXgbEstimators] = useState<number>(50);
  const [xgbResult, setXgbResult] = useState<XgboostResult | null>(null);
  const [isLoadingXgb, setIsLoadingXgb] = useState(false);

  const [ensembleType, setEnsembleType] = useState<string>('voting');
  const [ensembleResult, setEnsembleResult] = useState<EnsembleResult | null>(null);
  const [isLoadingEnsemble, setIsLoadingEnsemble] = useState(false);

  // Unit 3 Handlers
  const handleRunKMeans = async () => {
    setIsLoadingKmeans(true);
    try {
      const res = await runKMeans(kClusters, true, 2, 8);
      setKmeansResult(res);
    } finally {
      setIsLoadingKmeans(false);
    }
  };

  const handleRunGmm = async () => {
    setIsLoadingGmm(true);
    try {
      const res = await runGmm(gmmComponents, 'full', 10);
      setGmmResult(res);
    } finally {
      setIsLoadingGmm(false);
    }
  };

  const handleRunPca = async () => {
    setIsLoadingPca(true);
    try {
      const res = await runPca(2);
      setPcaResult(res);
    } finally {
      setIsLoadingPca(false);
    }
  };

  // Unit 4 Handlers
  const handleRunHmm = async () => {
    setIsLoadingHmm(true);
    try {
      const obsArray = obsInput.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      const [fwd, vit] = await Promise.all([runHmmForward(obsArray), runHmmViterbi(obsArray)]);
      setForwardResult(fwd);
      setViterbiResult(vit);
    } finally {
      setIsLoadingHmm(false);
    }
  };

  const handleFetchTimeline = async () => {
    setIsLoadingTimeline(true);
    try {
      const res = await getAccountTimeline(selectedAcc);
      setTimelineData(res);
    } finally {
      setIsLoadingTimeline(false);
    }
  };

  // Unit 5 Handlers
  const handleTrainCart = async () => {
    setIsLoadingCart(true);
    try {
      const res = await trainCart(cartDepth, 10, 'balanced');
      setCartResult(res);
    } finally {
      setIsLoadingCart(false);
    }
  };

  const handleTrainXgb = async () => {
    setIsLoadingXgb(true);
    try {
      const res = await trainXgboost(xgbEstimators, 4, 0.1);
      setXgbResult(res);
    } finally {
      setIsLoadingXgb(false);
    }
  };

  const handleTrainEnsemble = async () => {
    setIsLoadingEnsemble(true);
    try {
      const res = await trainEnsemble(ensembleType);
      setEnsembleResult(res);
    } finally {
      setIsLoadingEnsemble(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Unit Selector Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-graphite-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <span>Algorithm Laboratory (9 Mandatory Syllabus Algorithms)</span>
          </h2>
          <p className="text-xs text-graphite-400 mt-0.5">
            Execute real model fits, inspect analytical math, and visualize clusters, sequence states, and trees.
          </p>
        </div>

        <div className="flex items-center space-x-1 bg-graphite-900 p-1 rounded-lg border border-graphite-800 text-xs font-medium">
          <button
            onClick={() => setActiveUnit(3)}
            className={`px-3 py-1.5 rounded-md transition ${
              activeUnit === 3
                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-semibold'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Unit 3: Mixture & EM (3)
          </button>
          <button
            onClick={() => setActiveUnit(4)}
            className={`px-3 py-1.5 rounded-md transition ${
              activeUnit === 4
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Unit 4: HMM & Sequences (3)
          </button>
          <button
            onClick={() => setActiveUnit(5)}
            className={`px-3 py-1.5 rounded-md transition ${
              activeUnit === 5
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 font-semibold'
                : 'text-graphite-400 hover:text-white'
            }`}
          >
            Unit 5: Ensembles & Trees (3)
          </button>
        </div>
      </div>

      {/* UNIT 3 PANEL */}
      {activeUnit === 3 && (
        <div className="space-y-6">
          {/* 1. K-Means */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  ALGORITHM 1: K-MEANS
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  K-Means Clustering & Elbow Analysis
                </h3>
                <p className="text-xs text-graphite-400">
                  Partitions transactions into $K$ clusters minimizing within-cluster sum of squares (WCSS).
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1.5 text-xs text-graphite-300">
                  <span>Clusters (K):</span>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={kClusters}
                    onChange={(e) => setKClusters(parseInt(e.target.value, 10))}
                    className="w-14 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white font-mono text-xs"
                  />
                </div>
                <button
                  onClick={handleRunKMeans}
                  disabled={isLoadingKmeans}
                  className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingKmeans ? 'animate-spin' : ''}`} />
                  <span>{isLoadingKmeans ? 'Clustering...' : 'Run K-Means'}</span>
                </button>
              </div>
            </div>

            {kmeansResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-3">
                  <div className="text-xs font-semibold text-white">Cluster Metrics</div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">Inertia (WCSS)</div>
                      <div className="text-sm font-bold text-indigo-400 mt-0.5">
                        {kmeansResult.inertia.toFixed(2)}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">Silhouette Score</div>
                      <div className="text-sm font-bold text-emerald-400 mt-0.5">
                        {kmeansResult.silhouette_score.toFixed(4)}
                      </div>
                    </div>
                  </div>

                  <div className="text-xs font-semibold text-white pt-2">Cluster Distribution</div>
                  <div className="space-y-1 font-mono text-xs">
                    {Object.entries(kmeansResult.cluster_sizes).map(([cid, size]) => (
                      <div key={cid} className="flex justify-between items-center text-graphite-300">
                        <span>Cluster {cid}:</span>
                        <span className="text-white font-bold">{size} transactions</span>
                      </div>
                    ))}
                  </div>
                </div>

                {kmeansResult.elbow_curve && (
                  <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800">
                    <div className="text-xs font-semibold text-white mb-2">
                      Elbow Curve (Inertia vs K)
                    </div>
                    <div className="h-44">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={kmeansResult.elbow_curve}>
                          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                          <XAxis dataKey="k" stroke="#64748b" tick={{ fontSize: 10 }} />
                          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#0f172a',
                              borderColor: '#334155',
                              fontSize: 11,
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="inertia"
                            stroke="#818cf8"
                            strokeWidth={2}
                            dot={{ fill: '#818cf8', r: 3 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. GMM via EM */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  ALGORITHM 2: GMM VIA EM
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  Gaussian Mixture Models & EM Anomaly Ranking
                </h3>
                <p className="text-xs text-graphite-400">
                  Fits multi-modal Gaussian distributions using Expectation-Maximization to calculate sample density rarity.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1.5 text-xs text-graphite-300">
                  <span>Components (G):</span>
                  <input
                    type="number"
                    min="2"
                    max="6"
                    value={gmmComponents}
                    onChange={(e) => setGmmComponents(parseInt(e.target.value, 10))}
                    className="w-14 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white font-mono text-xs"
                  />
                </div>
                <button
                  onClick={handleRunGmm}
                  disabled={isLoadingGmm}
                  className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingGmm ? 'animate-spin' : ''}`} />
                  <span>{isLoadingGmm ? 'Optimizing EM...' : 'Run GMM via EM'}</span>
                </button>
              </div>
            </div>

            {gmmResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-3">
                  <div className="text-xs font-semibold text-white">EM Optimization Status</div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">Convergence</div>
                      <div className="text-sm font-bold text-emerald-400 mt-0.5">
                        {gmmResult.converged ? 'Converged' : 'Max Iter'}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">Lower Bound Log-Likelihood</div>
                      <div className="text-sm font-bold text-indigo-400 mt-0.5">
                        {gmmResult.lower_bound_log_likelihood.toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <div className="text-xs font-semibold text-white pt-2">Component Mixing Weights</div>
                  <div className="space-y-1 font-mono text-xs">
                    {gmmResult.weights.map((w, idx) => (
                      <div key={idx} className="flex justify-between items-center text-graphite-300">
                        <span>Component {idx}:</span>
                        <span className="text-white font-bold">{(w * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-white">
                    Top Rare Transactions (Lowest Density under GMM)
                  </div>
                  <div className="overflow-x-auto max-h-48">
                    <table className="w-full text-left text-xs text-graphite-300 font-mono">
                      <thead className="bg-graphite-900 text-graphite-400 text-[10px]">
                        <tr>
                          <th className="p-1.5">Row Index</th>
                          <th className="p-1.5">Log-Likelihood</th>
                          <th className="p-1.5">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-800/60">
                        {gmmResult.top_anomalous_indices.map((item, idx) => (
                          <tr key={idx} className="hover:bg-graphite-900">
                            <td className="p-1.5 text-white font-semibold">{item.index}</td>
                            <td className="p-1.5 text-rose-400 font-bold">
                              {item.log_likelihood.toFixed(3)}
                            </td>
                            <td className="p-1.5 text-[10px] text-amber-400">Rare Density</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* 3. PCA */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  ALGORITHM 3: PCA
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  Principal Component Analysis (2D Projection)
                </h3>
                <p className="text-xs text-graphite-400">
                  Orthogonal linear transformation maximizing variance onto primary eigenvector directions.
                </p>
              </div>

              <button
                onClick={handleRunPca}
                disabled={isLoadingPca}
                className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
              >
                <Play className={`w-3.5 h-3.5 ${isLoadingPca ? 'animate-spin' : ''}`} />
                <span>{isLoadingPca ? 'Projecting...' : 'Compute PCA'}</span>
              </button>
            </div>

            {pcaResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-white">Variance Explained</div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">PC1 Ratio</div>
                      <div className="text-sm font-bold text-indigo-400 mt-0.5">
                        {(pcaResult.explained_variance_ratio[0] * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-2 rounded bg-graphite-900 border border-graphite-800">
                      <div className="text-graphite-400">PC2 Ratio</div>
                      <div className="text-sm font-bold text-indigo-400 mt-0.5">
                        {(pcaResult.explained_variance_ratio[1] * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  <p className="text-[11px] text-graphite-400 pt-2">
                    Cumulative 2D Explained Variance:{' '}
                    <span className="text-white font-mono font-bold">
                      {(
                        pcaResult.cumulative_variance[pcaResult.cumulative_variance.length - 1] *
                        100
                      ).toFixed(2)}
                      %
                    </span>
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-xs font-semibold text-white mb-2">2D Coordinate Scatter</div>
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                        <CartesianGrid stroke="#1e293b" />
                        <XAxis dataKey="x" stroke="#64748b" tick={{ fontSize: 9 }} name="PC1" />
                        <YAxis dataKey="y" stroke="#64748b" tick={{ fontSize: 9 }} name="PC2" />
                        <ZAxis range={[20, 40]} />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{
                            backgroundColor: '#0f172a',
                            borderColor: '#334155',
                            fontSize: 10,
                          }}
                        />
                        <Scatter
                          data={pcaResult.projection_samples.filter((s) => s.class_label === 0)}
                          fill="#38bdf8"
                          name="Normal"
                        />
                        <Scatter
                          data={pcaResult.projection_samples.filter((s) => s.class_label === 1)}
                          fill="#f43f5e"
                          name="Fraud"
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* UNIT 4 PANEL */}
      {activeUnit === 4 && (
        <div className="space-y-6">
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                ALGORITHMS 4 & 5: SCALED FORWARD & LOG-SPACE VITERBI
              </span>
              <h3 className="text-base font-bold text-white mt-1">
                Hidden Markov Models: Evaluation & State Decoding
              </h3>
              <p className="text-xs text-graphite-400">
                Pure NumPy analytical implementation of Forward algorithm (with coefficient scaling)
                and Viterbi max-product in log-space.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-3">
              <label className="text-xs text-graphite-300 font-semibold block">
                Observation Sequence (0: Low Risk, 1: Medium, 2: High Risk):
              </label>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={obsInput}
                  onChange={(e) => setObsInput(e.target.value)}
                  className="flex-1 bg-graphite-900 border border-graphite-800 rounded px-3 py-1.5 text-xs text-white font-mono"
                  placeholder="0, 1, 0, 2, 1"
                />
                <button
                  onClick={handleRunHmm}
                  disabled={isLoadingHmm}
                  className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center space-x-1.5 transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingHmm ? 'animate-spin' : ''}`} />
                  <span>Execute Forward & Viterbi</span>
                </button>
              </div>
            </div>

            {forwardResult && viterbiResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-emerald-400">
                    Scaled Forward Evaluation: P(O|λ)
                  </div>
                  <div className="text-xs font-mono space-y-1 text-graphite-300">
                    <div>
                      Log-Likelihood:{' '}
                      <span className="text-white font-bold">
                        {forwardResult.log_likelihood.toFixed(5)}
                      </span>
                    </div>
                    <div>
                      Total Probability:{' '}
                      <span className="text-emerald-400 font-bold">
                        {forwardResult.total_probability.toExponential(4)}
                      </span>
                    </div>
                    <div>Sequence Length: {forwardResult.sequence_length} steps</div>
                    <div className="text-[10px] text-graphite-500 pt-1">
                      Computed with scaling factor $c_t$ preventing underflow.
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-cyan-400">
                    Log-Space Viterbi Decoded State Path
                  </div>
                  <div className="text-xs font-mono space-y-1 text-graphite-300">
                    <div>
                      Best Path Log-Prob:{' '}
                      <span className="text-white font-bold">
                        {viterbiResult.best_path_log_prob.toFixed(5)}
                      </span>
                    </div>
                    <div className="pt-1">
                      <span className="text-graphite-400">Optimal States:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {viterbiResult.best_path.map((state, idx) => (
                          <span
                            key={idx}
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              state === 2
                                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                : state === 1
                                ? 'bg-amber-500/20 text-amber-300'
                                : 'bg-emerald-500/10 text-emerald-400'
                            }`}
                          >
                            S{state}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Account Timeline */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  ALGORITHM 6: TEMPORAL ACCOUNT TIMELINE VISUALIZER
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  Sequential Account Behavioral Decoding
                </h3>
              </div>

              <div className="flex items-center space-x-2 text-xs">
                <select
                  value={selectedAcc}
                  onChange={(e) => setSelectedAcc(e.target.value)}
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
                <button
                  onClick={handleFetchTimeline}
                  disabled={isLoadingTimeline}
                  className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium"
                >
                  {isLoadingTimeline ? 'Decoding...' : 'Inspect Timeline'}
                </button>
              </div>
            </div>

            {timelineData && (
              <div className="space-y-3">
                <div className="flex space-x-4 text-xs font-mono text-graphite-400">
                  <span>Account: {timelineData.account_id}</span>
                  <span>Steps: {timelineData.total_steps}</span>
                  <span className="text-rose-400 font-bold">
                    Anomalous Steps: {timelineData.anomaly_count}
                  </span>
                </div>

                <div className="overflow-x-auto max-h-60 rounded border border-graphite-800">
                  <table className="w-full text-left text-xs text-graphite-300 font-mono">
                    <thead className="bg-graphite-950 text-graphite-400 text-[10px] sticky top-0">
                      <tr>
                        <th className="p-2">Step</th>
                        <th className="p-2">Amount ($)</th>
                        <th className="p-2">Merchant Category</th>
                        <th className="p-2">Decoded State</th>
                        <th className="p-2">Anomaly Flag</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite-800/60">
                      {timelineData.timeline.map((item, idx) => (
                        <tr key={idx} className="hover:bg-graphite-850">
                          <td className="p-2 text-graphite-500">{item.step}</td>
                          <td className="p-2 text-emerald-400 font-semibold">
                            ${item.amount.toFixed(2)}
                          </td>
                          <td className="p-2">Category {item.merchant_category}</td>
                          <td className="p-2 font-bold text-white">
                            {item.decoded_state_name} ({item.decoded_state})
                          </td>
                          <td className="p-2">
                            {item.is_anomaly === 1 ? (
                              <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 text-[10px]">
                                ANOMALY BURST
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
            )}
          </div>
        </div>
      )}

      {/* UNIT 5 PANEL */}
      {activeUnit === 5 && (
        <div className="space-y-6">
          {/* CART */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  ALGORITHM 7: BALANCED CART
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  CART Decision Tree & Auditable Rules
                </h3>
                <p className="text-xs text-graphite-400">
                  Classification tree trained with inverse class weighting and human-auditable rule extraction.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1.5 text-xs text-graphite-300">
                  <span>Max Depth:</span>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={cartDepth}
                    onChange={(e) => setCartDepth(parseInt(e.target.value, 10))}
                    className="w-14 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white font-mono text-xs"
                  />
                </div>
                <button
                  onClick={handleTrainCart}
                  disabled={isLoadingCart}
                  className="px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingCart ? 'animate-spin' : ''}`} />
                  <span>{isLoadingCart ? 'Fitting Tree...' : 'Train CART'}</span>
                </button>
              </div>
            </div>

            {cartResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2 font-mono text-xs">
                  <div className="text-white font-semibold mb-2">Standardized Test Metrics</div>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">PR-AUC</div>
                      <div className="text-amber-400 font-bold text-sm">
                        {cartResult.metrics.pr_auc.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">ROC-AUC</div>
                      <div className="text-white font-bold text-sm">
                        {cartResult.metrics.roc_auc.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">F1 Score</div>
                      <div className="text-emerald-400 font-bold text-sm">
                        {cartResult.metrics.f1.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800">
                  <div className="text-xs font-semibold text-white mb-2">
                    Extracted Auditable Decision Rules (Text)
                  </div>
                  <pre className="text-[10px] font-mono text-graphite-300 max-h-40 overflow-y-auto bg-graphite-900 p-2 rounded border border-graphite-800">
                    {cartResult.rules_text}
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* XGBoost */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  ALGORITHM 8: XGBOOST
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  XGBoost Gradient Boosting Classifier
                </h3>
                <p className="text-xs text-graphite-400">
                  Gradient boosted trees trained with `scale_pos_weight` and evaluated against PR-AUC.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1.5 text-xs text-graphite-300">
                  <span>Estimators:</span>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={xgbEstimators}
                    onChange={(e) => setXgbEstimators(parseInt(e.target.value, 10))}
                    className="w-14 bg-graphite-950 border border-graphite-800 rounded px-2 py-1 text-white font-mono text-xs"
                  />
                </div>
                <button
                  onClick={handleTrainXgb}
                  disabled={isLoadingXgb}
                  className="px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingXgb ? 'animate-spin' : ''}`} />
                  <span>{isLoadingXgb ? 'Boosting...' : 'Train XGBoost'}</span>
                </button>
              </div>
            </div>

            {xgbResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2 font-mono text-xs">
                  <div className="text-white font-semibold mb-2">Standardized Test Metrics</div>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">PR-AUC</div>
                      <div className="text-amber-400 font-bold text-sm">
                        {xgbResult.metrics.pr_auc.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">ROC-AUC</div>
                      <div className="text-white font-bold text-sm">
                        {xgbResult.metrics.roc_auc.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                      <div className="text-graphite-400">F1 Score</div>
                      <div className="text-emerald-400 font-bold text-sm">
                        {xgbResult.metrics.f1.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2">
                  <div className="text-xs font-semibold text-white">Top Feature Importances</div>
                  <div className="space-y-1 font-mono text-xs max-h-40 overflow-y-auto">
                    {Object.entries(xgbResult.feature_importances)
                      .slice(0, 5)
                      .map(([feat, val]) => (
                        <div key={feat} className="flex justify-between items-center text-graphite-300">
                          <span>{feat}:</span>
                          <span className="text-amber-400 font-bold">{val.toFixed(4)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Ensembles */}
          <div className="p-5 rounded-xl bg-graphite-900 border border-graphite-800 space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
              <div>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  ALGORITHM 9: ENSEMBLE COMBINATION
                </span>
                <h3 className="text-base font-bold text-white mt-1">
                  Ensemble: Soft Voting & Out-of-Fold Stacking
                </h3>
                <p className="text-xs text-graphite-400">
                  Combines diverse base learners with probability aggregation or meta-learning.
                </p>
              </div>

              <div className="flex items-center space-x-3">
                <select
                  value={ensembleType}
                  onChange={(e) => setEnsembleType(e.target.value)}
                  className="bg-graphite-950 border border-graphite-800 rounded px-2.5 py-1 text-white font-mono text-xs"
                >
                  <option value="voting">Soft Voting Ensemble</option>
                  <option value="stacking">Stacking (Logistic Regression Meta)</option>
                </select>
                <button
                  onClick={handleTrainEnsemble}
                  disabled={isLoadingEnsemble}
                  className="px-3.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-medium text-xs flex items-center space-x-1.5 shadow transition"
                >
                  <Play className={`w-3.5 h-3.5 ${isLoadingEnsemble ? 'animate-spin' : ''}`} />
                  <span>{isLoadingEnsemble ? 'Ensembling...' : 'Train Ensemble'}</span>
                </button>
              </div>
            </div>

            {ensembleResult && (
              <div className="p-4 rounded-lg bg-graphite-950 border border-graphite-800 space-y-2 font-mono text-xs">
                <div className="text-white font-semibold mb-2">Ensemble Performance</div>
                <div className="grid grid-cols-4 gap-2">
                  <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                    <div className="text-graphite-400">Precision</div>
                    <div className="text-white font-bold text-sm">
                      {ensembleResult.metrics.precision.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                    <div className="text-graphite-400">Recall</div>
                    <div className="text-white font-bold text-sm">
                      {ensembleResult.metrics.recall.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                    <div className="text-graphite-400">F1 Score</div>
                    <div className="text-emerald-400 font-bold text-sm">
                      {ensembleResult.metrics.f1.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 bg-graphite-900 rounded border border-graphite-800">
                    <div className="text-graphite-400">PR-AUC</div>
                    <div className="text-amber-400 font-bold text-sm">
                      {ensembleResult.metrics.pr_auc.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
