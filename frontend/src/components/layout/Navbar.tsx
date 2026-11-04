import React from 'react';
import { ShieldAlert, Activity, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { HealthResponse } from '../../types';

interface NavbarProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onRefreshHealth: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, isLoading, onRefreshHealth }) => {
  return (
    <header className="h-14 border-b border-graphite-800 bg-graphite-900/90 backdrop-blur px-5 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold tracking-tight text-white">AuditLens</h1>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Copilot v1.0
            </span>
          </div>
          <p className="text-[11px] text-graphite-400 hidden sm:block">
            Financial Anomaly & Fraud Investigation Decision Support
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* Decision Support Advisory Pill */}
        <div className="hidden lg:flex items-center space-x-1.5 text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
          <span>Advisory Decision Support — Human Adjudication Required</span>
        </div>

        {/* Backend Connectivity Status */}
        <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-graphite-850 border border-graphite-800 text-xs">
          <Activity className="w-3.5 h-3.5 text-graphite-400" />
          <span className="text-graphite-400">API:</span>
          {isLoading ? (
            <span className="flex items-center text-graphite-400 space-x-1">
              <RefreshCw className="w-3 h-3 animate-spin text-emerald-400" />
              <span>Connecting...</span>
            </span>
          ) : health && health.status === 'ok' ? (
            <span className="flex items-center text-emerald-400 space-x-1 font-mono font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Connected (Py {health.python_version})</span>
            </span>
          ) : (
            <span className="flex items-center text-rose-400 space-x-1 font-mono font-medium">
              <XCircle className="w-3.5 h-3.5 text-rose-400" />
              <span>Offline</span>
            </span>
          )}

          <button
            onClick={onRefreshHealth}
            title="Refresh API Connection"
            className="p-1 hover:text-emerald-400 text-graphite-400 transition"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};

