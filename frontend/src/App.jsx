import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { HomePage } from './pages/HomePage';
import { UploadPage } from './pages/UploadPage';
import { DipDashboardPage } from './pages/DipDashboardPage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { OptimizationPage } from './pages/OptimizationPage';
import { ResultsPage } from './pages/ResultsPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { AssistantPage } from './pages/AssistantPage';
import { HistoryPage } from './pages/HistoryPage';
import { AuditPage } from './pages/AuditPage';
import { apiService } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');

  // Persistent React State backed by SessionStorage
  const [dataset, setDatasetState] = useState(() => {
    try {
      const saved = sessionStorage.getItem('genesis_dataset');
      return saved ? JSON.parse(saved) : null;
    } catch (e) { return null; }
  });

  const [targetColumn, setTargetColumnState] = useState(() => {
    try {
      return sessionStorage.getItem('genesis_target') || '';
    } catch (e) { return ''; }
  });

  const [dipProfile, setDipProfileState] = useState(() => {
    try {
      const saved = sessionStorage.getItem('genesis_dip');
      return saved ? JSON.parse(saved) : null;
    } catch (e) { return null; }
  });

  const [experiment, setExperimentState] = useState(() => {
    try {
      const saved = sessionStorage.getItem('genesis_experiment');
      return saved ? JSON.parse(saved) : null;
    } catch (e) { return null; }
  });

  const [completedExp, setCompletedExpState] = useState(() => {
    try {
      const saved = sessionStorage.getItem('genesis_completed_exp');
      return saved ? JSON.parse(saved) : null;
    } catch (e) { return null; }
  });

  const setDataset = (data) => {
    setDatasetState(data);
    try {
      if (data) sessionStorage.setItem('genesis_dataset', JSON.stringify(data));
      else sessionStorage.removeItem('genesis_dataset');
    } catch (e) {}
  };

  const setTargetColumn = (target) => {
    setTargetColumnState(target);
    try {
      if (target) sessionStorage.setItem('genesis_target', target);
      else sessionStorage.removeItem('genesis_target');
    } catch (e) {}
  };

  const setDipProfile = (profile) => {
    setDipProfileState(profile);
    try {
      if (profile) sessionStorage.setItem('genesis_dip', JSON.stringify(profile));
      else sessionStorage.removeItem('genesis_dip');
    } catch (e) {}
  };

  const setExperiment = (exp) => {
    setExperimentState(exp);
    try {
      if (exp) sessionStorage.setItem('genesis_experiment', JSON.stringify(exp));
      else sessionStorage.removeItem('genesis_experiment');
    } catch (e) {}
  };

  const setCompletedExp = (exp) => {
    setCompletedExpState(exp);
    try {
      if (exp) sessionStorage.setItem('genesis_completed_exp', JSON.stringify(exp));
      else sessionStorage.removeItem('genesis_completed_exp');
    } catch (e) {}
  };

  // Explicit helper to wipe previous active and completed experiment state
  const resetExperimentContext = () => {
    setExperimentState(null);
    setCompletedExpState(null);
    try {
      sessionStorage.removeItem('genesis_experiment');
      sessionStorage.removeItem('genesis_completed_exp');
    } catch (e) {}
  };

  // Callback when dataset is uploaded and DIP profile is generated
  const handleDatasetUploaded = (dsData, target, profile) => {
    resetExperimentContext();
    setDataset(dsData);
    setTargetColumn(target);
    setDipProfile(profile);
    setActiveTab('dip');
  };

  // Callback when user completes or selects an experiment from History Inspect
  const handleSelectExperiment = async (exp) => {
    try {
      const fullExp = await apiService.getExperiment(exp.id);
      const dsId = fullExp.dataset_id;
      if (dsId) {
        const ds = await apiService.getDataset(dsId);
        setDataset(ds);
        setTargetColumn(fullExp.target_column);
        const profile = await apiService.getDatasetProfile(dsId, fullExp.target_column);
        setDipProfile(profile);
      }
      setExperiment(fullExp);
      if (fullExp.status === 'COMPLETED') {
        setCompletedExp(fullExp);
        setActiveTab('results');
      } else {
        setCompletedExp(null);
        setActiveTab('optimization');
      }
    } catch (e) {
      console.warn('Could not fetch selected experiment:', e);
      // Fallback
      setDataset(null);
      setTargetColumn(exp.target_column);
      setExperiment(exp);
      if (exp.status === 'COMPLETED') {
        setCompletedExp(exp);
        setActiveTab('results');
      } else {
        setCompletedExp(null);
        setActiveTab('optimization');
      }
    }
  };

  // Dataset/Experiment context validation invariants
  const currentDatasetId = dataset?.id || dataset?.dataset_id;

  const isExperimentForContext = (exp, datasetId, targetCol) => {
    if (!exp || !datasetId || !targetCol) return false;
    return exp.dataset_id === datasetId && exp.target_column === targetCol;
  };

  const activeExperiment = isExperimentForContext(experiment, currentDatasetId, targetColumn) ? experiment : null;

  let activeCompletedExp = null;
  if (isExperimentForContext(completedExp, currentDatasetId, targetColumn) && completedExp.status === 'COMPLETED') {
    activeCompletedExp = completedExp;
  } else if (activeExperiment && activeExperiment.status === 'COMPLETED') {
    activeCompletedExp = activeExperiment;
  }

  return (
    <div className="min-h-screen flex flex-col justify-between">
      
      {/* Global Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasDataset={!!dataset}
        hasExperiment={!!activeExperiment}
        hasCompletedExperiment={!!activeCompletedExp}
      />

      {/* Main Screen Content */}
      <main className="flex-1">
        {activeTab === 'home' && <HomePage onNavigate={(tab) => setActiveTab(tab)} />}

        {activeTab === 'upload' && (
          <UploadPage onDatasetUploaded={handleDatasetUploaded} />
        )}

        {activeTab === 'dip' && (
          <DipDashboardPage
            dipProfile={dipProfile}
            onNext={() => setActiveTab('recommendations')}
          />
        )}

        {activeTab === 'recommendations' && (
          <RecommendationsPage
            dataset={dataset}
            targetColumn={targetColumn}
            experiment={activeExperiment}
            onStartOptimization={() => setActiveTab('optimization')}
          />
        )}

        {activeTab === 'optimization' && (
          <OptimizationPage
            dataset={dataset}
            targetColumn={targetColumn}
            currentExperiment={activeExperiment}
            onOptimizationComplete={(exp) => {
              setCompletedExp(exp);
              setExperiment(exp);
              setActiveTab('results');
            }}
          />
        )}

        {activeTab === 'results' && (
          <ResultsPage
            dataset={dataset}
            experiment={activeCompletedExp || activeExperiment}
            onNavigateExplainability={() => setActiveTab('explainability')}
          />
        )}

        {activeTab === 'explainability' && (
          <ExplainabilityPage
            dataset={dataset}
            experiment={activeCompletedExp || activeExperiment}
            onNavigateAssistant={() => setActiveTab('assistant')}
          />
        )}

        {activeTab === 'assistant' && (
          <AssistantPage
            dataset={dataset}
            experiment={activeCompletedExp || activeExperiment}
          />
        )}

        {activeTab === 'history' && (
          <HistoryPage
            activeTab={activeTab}
            onSelectExperiment={handleSelectExperiment}
          />
        )}

        {activeTab === 'audit' && (
          <AuditPage
            dataset={dataset}
            experiment={activeCompletedExp || activeExperiment}
            dipProfile={dipProfile}
          />
        )}
      </main>

      {/* Footer */}
      <Footer />

    </div>
  );
}
