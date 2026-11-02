import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { fetchHealth, HealthResponse } from './services/api';
import './styles.css';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

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
            Phase 1 Foundation Active &bull; Human-in-the-Loop Decision Support
          </div>
          <div className="banner-body">
            AuditLens is strictly an advisory decision-support system. It will not execute automated account freezes, decline transactions autonomously, or file suspicious activity reports without an analyst's review.
          </div>
        </div>

        {activeTab === 'overview' && (
          <div>
            <div className="grid-cards">
              <div className="card">
                <div className="card-title">Backend Status</div>
                <div className="card-value">
                  {loading ? 'Checking...' : health ? 'ONLINE' : 'OFFLINE'}
                </div>
                <div className="card-description">
                  {health
                    ? `${health.service} (${health.environment})`
                    : error || 'FastAPI backend unreachable'}
                </div>
              </div>

              <div className="card">
                <div className="card-title">API Version</div>
                <div className="card-value">{health?.version || 'v0.1.0'}</div>
                <div className="card-description">
                  FastAPI on Python 3.13 + Uvicorn
                </div>
              </div>

              <div className="card">
                <div className="card-title">Phase Gate</div>
                <div className="card-value">Phase 1</div>
                <div className="card-description">
                  <span className="status-badge pass">FOUNDATION VERIFIED</span>
                </div>
              </div>
            </div>

            <div className="card" style={{ marginTop: '1.5rem' }}>
              <div className="card-title">Architecture Pipeline Status</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.5rem 0' }}>Module</th>
                    <th style={{ padding: '0.5rem 0' }}>Implementation Phase</th>
                    <th style={{ padding: '0.5rem 0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <td style={{ padding: '0.75rem 0' }}>Project Foundation & Local Dev Environment</td>
                    <td style={{ padding: '0.75rem 0' }}>Phase 1</td>
                    <td style={{ padding: '0.75rem 0' }}><span className="status-badge pass">READY</span></td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <td style={{ padding: '0.75rem 0' }}>Dataset Acquisition & Validation</td>
                    <td style={{ padding: '0.75rem 0' }}>Phase 2</td>
                    <td style={{ padding: '0.75rem 0' }}><span className="status-badge pending">PENDING</span></td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <td style={{ padding: '0.75rem 0' }}>Exploratory Data Analysis & Preprocessing</td>
                    <td style={{ padding: '0.75rem 0' }}>Phase 3</td>
                    <td style={{ padding: '0.75rem 0' }}><span className="status-badge pending">PENDING</span></td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <td style={{ padding: '0.75rem 0' }}>Baseline Machine Learning Models</td>
                    <td style={{ padding: '0.75rem 0' }}>Phase 4</td>
                    <td style={{ padding: '0.75rem 0' }}><span className="status-badge pending">PENDING</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab !== 'overview' && (
          <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
              Module Scheduled for Subsequent Phase
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              This module will be connected as subsequent phases pass their formal test gates.
            </p>
          </div>
        )}
      </main>

      <footer className="footer">
        AuditLens &bull; Financial Anomaly &amp; Fraud Investigation Copilot &bull; Phase 1 Local Skeleton
      </footer>
    </div>
  );
};

