import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewTab } from './components/OverviewTab';
import { TransactionsTab } from './components/TransactionsTab';
import { CasesTab } from './components/CasesTab';
import { PolicyReferencesTab } from './components/PolicyReferencesTab';
import { fetchHealth, HealthResponse } from './services/api';
import './styles.css';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [initialTxForCase, setInitialTxForCase] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function checkBackend() {
      try {
        setLoading(true);
        const data = await fetchHealth();
        if (isMounted) {
          setHealth(data);
          setError(null);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to connect');
          setHealth(null);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    checkBackend();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        loading={loading}
        error={error}
      />

      <main className="main-content">
        <div className="page-header">
          <h1 className="page-title">Financial Fraud Investigation Copilot</h1>
          <p className="page-subtitle">
            Decision-support copilot for transaction triage, explainable ML scoring, regulatory citation, and audit documentation.
          </p>
        </div>

        <div className="banner">
          <div className="banner-title">
            Phase 9 Active &bull; Human-in-the-Loop Decision Support
          </div>
          <div className="banner-body">
            AuditLens is strictly an advisory decision-support system. It will not execute automated account freezes, decline transactions autonomously, or file suspicious activity reports without an analyst's review.
          </div>
        </div>

        {activeTab === 'overview' && (
          <OverviewTab
            health={health}
            loading={loading}
            error={error}
            onNavigateTab={setActiveTab}
          />
        )}

        {activeTab === 'transactions' && (
          <TransactionsTab
            onOpenCreateCase={(txId) => {
              setInitialTxForCase(txId);
              setActiveTab('cases');
            }}
          />
        )}

        {activeTab === 'cases' && (
          <CasesTab
            initialTxIdForCase={initialTxForCase}
            onClearInitialTx={() => setInitialTxForCase(null)}
          />
        )}

        {activeTab === 'rag' && <PolicyReferencesTab />}

        {activeTab === 'reports' && (
          <div className="card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>&#128221;</div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', fontWeight: 700 }}>
              Investigation Report Drafting &amp; LLM Copilot
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '600px', margin: '0 auto 1.5rem' }}>
              Automated narrative generation, case report exports (Markdown/JSON/PDF), and LLM-assisted investigation summaries are scheduled for <strong>Phase 10</strong>.
            </p>
            <div style={{ display: 'inline-block' }}>
              <span className="status-badge pending">SCHEDULED FOR PHASE 10</span>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        AuditLens &bull; Financial Anomaly &amp; Fraud Investigation Copilot &bull; Phase 9 Verified
      </footer>
    </div>
  );
};
