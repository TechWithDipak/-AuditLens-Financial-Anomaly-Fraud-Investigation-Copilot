import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  Database,
  Cpu,
  BarChart3,
  Search,
  Briefcase,
  BookOpen,
  FileText,
  Settings as SettingsIcon,
} from 'lucide-react';
import { ActiveTab } from '../../types';

interface SidebarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  const navItems = [
    { id: 'overview' as ActiveTab, label: 'Overview', icon: LayoutDashboard },
    { id: 'demo' as ActiveTab, label: 'Interactive Demo', icon: Sparkles, badge: 'Guided' },
    { id: 'workspace' as ActiveTab, label: 'Data Workspace', icon: Database },
    { id: 'algorithms' as ActiveTab, label: 'Algorithm Lab', icon: Cpu, badge: '9 Algos' },
    { id: 'models' as ActiveTab, label: 'Model Comparison', icon: BarChart3 },
    { id: 'explorer' as ActiveTab, label: 'Transactions & SHAP', icon: Search },
    { id: 'cases' as ActiveTab, label: 'Case Management', icon: Briefcase },
    { id: 'rag' as ActiveTab, label: 'Regulatory Knowledge', icon: BookOpen },
    { id: 'assistant' as ActiveTab, label: 'Investigation Assistant', icon: FileText, badge: 'Copilot' },
    { id: 'settings' as ActiveTab, label: 'System Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="w-64 border-r border-graphite-800 bg-graphite-900 flex flex-col justify-between shrink-0">
      <div className="p-3 space-y-1">
        <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-graphite-400">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition ${
                isActive
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'text-graphite-300 hover:bg-graphite-850 hover:text-white'
              }`}
            >
              <div className="flex items-center space-x-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-graphite-400'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-graphite-800 text-graphite-300 border border-graphite-700">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-graphite-800 bg-graphite-950/40 text-[11px] text-graphite-400 space-y-1">
        <div className="flex justify-between items-center text-graphite-300 font-mono text-[10px]">
          <span>UNITS 3, 4, 5</span>
          <span className="text-emerald-400">9 ALGORITHMS</span>
        </div>
        <p className="text-[10px] leading-tight text-graphite-500">
          K-Means, GMM/EM, PCA, HMM, Forward, Viterbi, CART, XGBoost, Ensemble
        </p>
      </div>
    </aside>
  );
};

