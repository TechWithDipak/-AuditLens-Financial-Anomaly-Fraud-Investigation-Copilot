import React from 'react';
import { HealthResponse } from '../services/api';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  loading,
  error,
}) => {
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'transactions', label: 'Transactions' },
    { id: 'cases', label: 'Cases' },
    { id: 'rag', label: 'Policy References' },
    { id: 'reports', label: 'Investigation Reports' },
  ];

  return (
    <header className="navbar">
      <div className="brand">
        <span className="brand-icon">🔍</span>
        <div>
          <span className="brand-title">AuditLens</span>
          <span className="brand-badge" style={{ marginLeft: '8px' }}>
            Phase 9 Active
          </span>
        </div>
      </div>

      <nav className="nav-links">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="nav-status">
        <div
          className={`status-dot ${health?.status === 'ok' ? 'online' : 'offline'}`}
        />
        <span>
          {loading
            ? 'Checking API...'
            : health?.status === 'ok'
            ? `API Online (v${health.version})`
            : error
            ? 'API Disconnected'
            : 'API Offline'}
        </span>
      </div>
    </header>
  );
};

