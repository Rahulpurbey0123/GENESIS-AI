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

  // Callback when dataset is uploaded and DIP profile is generated
  const handleDatasetUploaded = (dsData, target, profile) => {
    setDataset(dsData);
    setTargetColumn(target);
    setDipProfile(profile);
    setActiveTab('dip');
  };

  // Callback when user completes or selects an experiment
  const handleSelectExperiment = async (exp) => {
    setExperiment(exp);
    const dsId = exp.dataset_id;
    if (dsId && (!dataset || (dataset.id !== dsId && dataset.dataset_id !== dsId))) {
      try {
        const ds = await apiService.getDataset(dsId);
        setDataset(ds);
        setTargetColumn(exp.target_column);
        const profile = await apiService.getDatasetProfile(dsId);
        setDipProfile(profile);
      } catch (e) {
        console.warn('Could not auto-fetch dataset metadata for selected experiment:', e);
      }
    }
    if (exp.status === 'COMPLETED') {
      setCompletedExp(exp);
      setActiveTab('results');
    } else {
      setActiveTab('optimization');
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between">
      
      {/* Global Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasDataset={!!dataset}
        hasExperiment={!!experiment}
        hasCompletedExperiment={!!completedExp}
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
            onStartOptimization={() => setActiveTab('optimization')}
          />
        )}

        {activeTab === 'optimization' && (
          <OptimizationPage
            dataset={dataset}
            targetColumn={targetColumn}
            currentExperiment={experiment}
            onOptimizationComplete={(exp) => {
              setCompletedExp(exp);
              setExperiment(exp);
              setActiveTab('results');
            }}
          />
        )}

        {activeTab === 'results' && (
          <ResultsPage
            experiment={completedExp || experiment}
            onNavigateExplainability={() => setActiveTab('explainability')}
          />
        )}

        {activeTab === 'explainability' && (
          <ExplainabilityPage
            experiment={completedExp || experiment}
            onNavigateAssistant={() => setActiveTab('assistant')}
          />
        )}

        {activeTab === 'assistant' && (
          <AssistantPage experiment={completedExp || experiment} />
        )}

        {activeTab === 'history' && (
          <HistoryPage onSelectExperiment={handleSelectExperiment} />
        )}
      </main>

      {/* Footer */}
      <Footer />

    </div>
  );
}
