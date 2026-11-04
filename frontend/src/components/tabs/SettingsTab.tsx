import React from 'react';
import { Settings, Server, Database, ShieldCheck } from 'lucide-react';
import { HealthResponse } from '../../types';

interface SettingsTabProps {
  health: HealthResponse | null;
}

export const SettingsTab: React.FC<SettingsTabProps> = ({ health }) => {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="p-6 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-emerald-400" />
          <h2 className="text-xl font-bold text-white">System Settings & Infrastructure</h2>
        </div>
        <p className="text-xs text-graphite-400 leading-relaxed">
          Configuration parameters, data persistence directories, and runtime execution flags.
        </p>
      </div>

      <div className="space-y-4 font-mono text-xs">
        {/* Backend & API */}
        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center space-x-2 text-white font-sans font-semibold text-sm">
            <Server className="w-4 h-4 text-emerald-400" />
            <span>API & Compute Runtime</span>
          </div>
          <div className="space-y-1.5 text-graphite-300">
            <div className="flex justify-between py-1 border-b border-graphite-800">
              <span className="text-graphite-400">API Host</span>
              <span className="text-white">127.0.0.1:8000</span>
            </div>
            <div className="flex justify-between py-1 border-b border-graphite-800">
              <span className="text-graphite-400">Environment</span>
              <span className="text-emerald-400">{health?.environment || 'development'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-graphite-800">
              <span className="text-graphite-400">Python Version</span>
              <span className="text-white">{health?.python_version || 'Python 3.13'}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-graphite-400">App Name</span>
              <span className="text-white">{health?.app_name || 'AuditLens Backend'}</span>
            </div>
          </div>
        </div>

        {/* Persistence */}
        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center space-x-2 text-white font-sans font-semibold text-sm">
            <Database className="w-4 h-4 text-cyan-400" />
            <span>Database & Storage</span>
          </div>
          <div className="space-y-1.5 text-graphite-300">
            <div className="flex justify-between py-1 border-b border-graphite-800">
              <span className="text-graphite-400">Database Engine</span>
              <span className="text-white">SQLite (WAL Mode)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-graphite-800">
              <span className="text-graphite-400">Target Database File</span>
              <span className="text-cyan-400">backend/data/auditlens.db</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-graphite-400">Model Artifacts Storage</span>
              <span className="text-white">artifacts/</span>
            </div>
          </div>
        </div>

        {/* Offline & Security */}
        <div className="p-4 rounded-xl bg-graphite-900 border border-graphite-800 space-y-3">
          <div className="flex items-center space-x-2 text-white font-sans font-semibold text-sm">
            <ShieldCheck className="w-4 h-4 text-amber-400" />
            <span>Security & Zero API Key Requirements</span>
          </div>
          <p className="text-xs text-graphite-400 font-sans leading-relaxed">
            AuditLens is designed to run 100% locally with zero external API key requirements.
            All 9 syllabus ML algorithms (Units 3, 4, and 5) run completely on local CPU.
          </p>
        </div>
      </div>
    </div>
  );
};
