import React, { useState, useEffect } from 'react';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { checkHealth } from './services/api';
import { HealthResponse, ActiveTab } from './types';
import { OverviewTab } from './components/tabs/OverviewTab';
import { DemoTab } from './components/tabs/DemoTab';
import { DataWorkspaceTab } from './components/tabs/DataWorkspaceTab';
import { AlgorithmLabTab } from './components/tabs/AlgorithmLabTab';
import { ModelComparisonTab } from './components/tabs/ModelComparisonTab';
import { TransactionExplorerTab } from './components/tabs/TransactionExplorerTab';
import { CaseManagementTab } from './components/tabs/CaseManagementTab';
import { RagTab } from './components/tabs/RagTab';
import { AssistantTab } from './components/tabs/AssistantTab';
import { SettingsTab } from './components/tabs/SettingsTab';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  const fetchHealth = async () => {
    setIsLoadingHealth(true);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setIsLoadingHealth(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-graphite-950 text-graphite-100 font-sans">
      <Navbar health={health} isLoading={isLoadingHealth} onRefreshHealth={fetchHealth} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

        <main className="flex-1 overflow-y-auto p-6 bg-graphite-950">
          {activeTab === 'overview' && (
            <OverviewTab
              health={health}
              isLoadingHealth={isLoadingHealth}
              onRefreshHealth={fetchHealth}
              onSelectTab={setActiveTab}
            />
          )}

          {activeTab === 'demo' && <DemoTab onNavigateToTab={setActiveTab} />}

          {activeTab === 'workspace' && <DataWorkspaceTab />}

          {activeTab === 'algorithms' && <AlgorithmLabTab />}

          {activeTab === 'models' && <ModelComparisonTab />}

          {activeTab === 'explorer' && <TransactionExplorerTab />}

          {activeTab === 'cases' && <CaseManagementTab />}

          {activeTab === 'rag' && <RagTab />}

          {activeTab === 'assistant' && <AssistantTab />}

          {activeTab === 'settings' && <SettingsTab health={health} />}
        </main>
      </div>
    </div>
  );
};
