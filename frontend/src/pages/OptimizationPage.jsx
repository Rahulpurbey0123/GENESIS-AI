import React, { useState, useEffect } from 'react';
import { Cpu, Play, CheckCircle2, AlertTriangle, RefreshCw, Layers, ShieldCheck, Zap } from 'lucide-react';
import { apiService } from '../services/api';
import { PlotViewer } from '../components/PlotViewer';

export function OptimizationPage({ dataset, targetColumn, currentExperiment, onOptimizationComplete }) {
  const [mode, setMode] = useState('genesis');
  const [generations, setGenerations] = useState(10);
  const [populationSize, setPopulationSize] = useState(20);
  const [maxEvaluations, setMaxEvaluations] = useState(200);

  const [activeExp, setActiveExp] = useState(currentExperiment);
  const [statusData, setStatusData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [fitnessHistory, setFitnessHistory] = useState([]);

  // Start Optimization Experiment
  const handleStart = async () => {
    const dsId = dataset?.id || dataset?.dataset_id;
    if (!dsId || !targetColumn) return;
    setError(null);
    setIsRunning(true);
    setFitnessHistory([]);

    try {
      const newExp = await apiService.createExperiment({
        datasetId: dsId,
        targetColumn: targetColumn,
        mode: mode,
        populationSize: populationSize,
        generations: generations,
        maxEvaluations: maxEvaluations
      });
      setActiveExp(newExp);
    } catch (err) {
      setError(err.message || 'Failed to start optimization experiment.');
      setIsRunning(false);
    }
  };

  // Poll experiment status if running
  useEffect(() => {
    const expId = activeExp?.id || activeExp?.experiment_id;
    if (!expId) return;

    let intervalId = null;
    async function pollStatus() {
      try {
        const expData = await apiService.getExperiment(expId);
        setStatusData(expData);

        if (expData.progress?.history && Array.isArray(expData.progress.history)) {
          setFitnessHistory(expData.progress.history.map(h => ({ gen: h.gen, fitness: h.best_score })));
        } else if (expData.progress && expData.progress.best_score !== undefined && expData.progress.best_score !== null) {
          setFitnessHistory((prev) => {
            const currentGen = expData.progress.current_generation || 1;
            const existingIdx = prev.findIndex((p) => p.gen === currentGen);
            if (existingIdx >= 0) return prev;
            return [...prev, { gen: currentGen, fitness: expData.progress.best_score }];
          });
        }

        if (expData.status === 'COMPLETED') {
          setIsRunning(false);
          clearInterval(intervalId);
          onOptimizationComplete(expData);
        } else if (expData.status === 'FAILED') {
          setIsRunning(false);
          setError(expData.error_message || 'Optimization experiment failed.');
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error('Error polling experiment status:', err);
      }
    }

    pollStatus();
    intervalId = setInterval(pollStatus, 1500);

    return () => clearInterval(intervalId);
  }, [activeExp]);

  const progress = statusData?.progress || {};
  const currentGen = progress.current_generation ?? null;
  const maxGen = progress.max_generations ?? generations;
  const evaluatedCount = progress.evaluated_pipelines ?? null;
  const bestScore = progress.best_score ?? null;
  const runtime = progress.runtime ?? null;
  const reductionPct = progress.search_space_reduction !== undefined && progress.search_space_reduction !== null
    ? `${Math.round(progress.search_space_reduction * 100)}%`
    : 'N/A';

  // Convergence Chart
  const chartData = [
    {
      x: fitnessHistory.map((h) => `Gen ${h.gen}`),
      y: fitnessHistory.map((h) => h.fitness),
      type: 'scatter',
      mode: 'lines+markers',
      marker: { color: '#6366f1', size: 8 },
      line: { color: '#818cf8', width: 3 }
    }
  ];

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
          Week 4 Evolutionary Engine
        </span>
        <h2 className="text-3xl font-extrabold text-white mt-1">Multi-Objective Optimization</h2>
      </div>

      {!activeExp ? (
        /* Setup & Launch Form */
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white">Configure GA Search Space</h3>
            <p className="text-xs text-slate-400">
              Set search constraints for candidate pipeline generation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Optimization Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs font-semibold outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="genesis">GENESIS (Guided Search)</option>
                <option value="baseline">Baseline GA</option>
                <option value="random">Random Search</option>
              </select>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Generations</label>
              <input
                type="number"
                min="1"
                max="50"
                value={generations}
                onChange={(e) => setGenerations(parseInt(e.target.value) || 10)}
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs font-semibold outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1">Population Size</label>
              <input
                type="number"
                min="5"
                max="100"
                value={populationSize}
                onChange={(e) => setPopulationSize(parseInt(e.target.value) || 20)}
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs font-semibold outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

          </div>

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}

          <button
            onClick={handleStart}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-sm shadow-xl shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all"
          >
            <Play className="w-4 h-4 fill-white" />
            <span>Start Optimization Run</span>
          </button>
        </div>
      ) : (
        /* Progress & Live Dashboard */
        <div className="space-y-6">
          
          {/* Status Tracker Banner */}
          <div className={`glass-panel p-6 rounded-2xl space-y-4 ${
            statusData?.status === 'FAILED' ? 'border-red-500/40' : 'border-indigo-500/30'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {statusData?.status === 'FAILED' ? (
                  <AlertTriangle className="w-6 h-6 text-red-400" />
                ) : isRunning ? (
                  <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
                ) : (
                  <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                )}
                <div>
                  <h3 className="text-lg font-bold text-white">
                    {statusData?.status === 'FAILED'
                      ? 'Optimization Failed'
                      : isRunning
                      ? 'Optimization in Progress...'
                      : 'Optimization Complete!'}
                  </h3>
                  <p className="text-xs text-slate-400">Experiment ID: {activeExp.id}</p>
                </div>
              </div>

              <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                statusData?.status === 'FAILED'
                  ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                  : isRunning
                  ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}>
                {statusData?.status || 'RUNNING'}
              </span>
            </div>

            {/* Error Message Box for FAILED status */}
            {(statusData?.status === 'FAILED' || error) && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                <div>
                  <p className="font-bold text-red-400 mb-0.5">Execution Error</p>
                  <p className="text-red-300 font-mono">{statusData?.error_message || error || 'Optimization experiment encountered an unrecoverable failure.'}</p>
                </div>
              </div>
            )}

            {/* Progress Bar */}
            <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden border border-slate-800">
              <div
                className={`h-full transition-all duration-500 ${
                  statusData?.status === 'FAILED'
                    ? 'bg-red-500'
                    : 'bg-gradient-to-r from-indigo-500 to-purple-500'
                }`}
                style={{ width: `${currentGen !== null && maxGen ? Math.min(100, (currentGen / maxGen) * 100) : 0}%` }}
              ></div>
            </div>

            {/* Stats Metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pt-2">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-xs text-slate-400 block font-medium">Generation</span>
                <span className="text-xl font-black text-white">{currentGen !== null ? `${currentGen} / ${maxGen}` : 'N/A'}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-xs text-slate-400 block font-medium">Evaluated</span>
                <span className="text-xl font-black text-indigo-400">{evaluatedCount !== null ? evaluatedCount : 'N/A'}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-xs text-slate-400 block font-medium">Best Score</span>
                <span className={`text-xl font-black ${statusData?.status === 'FAILED' ? 'text-slate-500' : 'text-emerald-400'}`}>
                  {statusData?.status === 'FAILED'
                    ? 'N/A'
                    : bestScore !== null ? (typeof bestScore === 'number' ? bestScore.toFixed(4) : bestScore) : 'N/A'}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-xs text-slate-400 block font-medium">Runtime</span>
                <span className="text-xl font-black text-purple-400">
                  {runtime !== null ? `${runtime}s` : 'N/A'}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-xs text-slate-400 block font-medium">Space Reduction</span>
                <span className="text-xl font-black text-sky-400">{reductionPct}</span>
              </div>
            </div>
          </div>

          {/* Convergence Visualization */}
          <div className="glass-panel p-6 rounded-2xl space-y-4">
            <h3 className="text-sm font-bold text-white">Fitness Convergence History</h3>
            {fitnessHistory.length > 0 ? (
              <PlotViewer
                data={chartData}
                layout={{
                  height: 300,
                  xaxis: { title: 'Generations' },
                  yaxis: { title: 'Best Fitness Score' }
                }}
              />
            ) : (
              <div className="h-48 flex items-center justify-center text-xs font-semibold text-slate-500 bg-slate-900/40 rounded-xl border border-slate-800/60">
                Waiting for optimization history...
              </div>
            )}
          </div>

        </div>
      )}

    </div>
  );
}
