import React, { useState, useEffect } from 'react';
import { CheckCircle2, Award, Cpu, Zap, Activity, AlertCircle, ArrowRight } from 'lucide-react';
import { apiService } from '../services/api';

export function ResultsPage({ experiment, onNavigateExplainability }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const expId = experiment?.id || experiment?.experiment_id;
    if (!expId) {
      setLoading(false);
      return;
    }
    async function fetchResults() {
      try {
        setLoading(true);
        const data = await apiService.getResults(expId);
        setResults(data);
      } catch (err) {
        setError(err.message || 'Failed to load experiment results.');
      } finally {
        setLoading(false);
      }
    }
    fetchResults();
  }, [experiment]);

  if (!experiment) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center text-slate-400">
        No experiment results selected. Please complete or select an optimization experiment first.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center text-slate-400">
        Loading evaluation results...
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center p-6 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400">
        <AlertCircle className="w-8 h-8 mx-auto mb-2" />
        <p className="font-bold">{error}</p>
      </div>
    );
  }

  const { best_pipeline, metrics, efficiency } = results || {};
  const isClassification = best_pipeline?.task_type?.toLowerCase() === 'classification' || metrics?.f1 !== undefined;

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            Optimization Complete
          </span>
          <h2 className="text-3xl font-extrabold text-white mt-1">Best Pipeline Evaluation Results</h2>
        </div>

        <button
          onClick={onNavigateExplainability}
          className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center gap-2 transition-all"
        >
          <span>View Explainability</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Best Pipeline Architecture Card */}
      <div className="glass-panel p-6 rounded-2xl border-l-4 border-emerald-500 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Award className="w-6 h-6 text-emerald-400" />
            <div>
              <h3 className="text-lg font-bold text-white">{best_pipeline?.model_name || 'Best Pipeline Estimator'}</h3>
              <p className="text-xs text-slate-400 font-mono">Pipeline ID: {best_pipeline?.id}</p>
            </div>
          </div>

          <div className="text-right">
            <span className="text-xs text-slate-400 block font-medium">Fitness Score</span>
            <span className="text-3xl font-black text-emerald-400">
              {best_pipeline?.fitness ? best_pipeline.fitness.toFixed(4) : 'N/A'}
            </span>
          </div>
        </div>

        {/* Hyperparameters / Preprocessing */}
        <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-xs space-y-2">
          <span className="font-bold text-slate-300 block">Optimal Hyperparameters & Configuration:</span>
          <pre className="text-indigo-300 font-mono overflow-x-auto p-2 rounded bg-slate-950/60">
            {JSON.stringify(best_pipeline?.hyperparameters || {}, null, 2)}
          </pre>
        </div>
      </div>

      {/* Metric Grid */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-indigo-400" />
          <span>Evaluation Metrics ({isClassification ? 'Classification' : 'Regression'})</span>
        </h3>

        {isClassification ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">Accuracy</span>
              <p className="text-2xl font-black text-white">{metrics?.accuracy !== undefined ? metrics.accuracy : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">F1 Score (Macro)</span>
              <p className="text-2xl font-black text-indigo-400">{metrics?.f1 !== undefined ? metrics.f1 : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">Precision</span>
              <p className="text-2xl font-black text-purple-400">{metrics?.precision !== undefined ? metrics.precision : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">Recall</span>
              <p className="text-2xl font-black text-emerald-400">{metrics?.recall !== undefined ? metrics.recall : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">ROC-AUC</span>
              <p className="text-2xl font-black text-sky-400">{metrics?.roc_auc !== undefined ? metrics.roc_auc : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">PR-AUC</span>
              <p className="text-2xl font-black text-pink-400">{metrics?.pr_auc !== undefined ? metrics.pr_auc : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1 col-span-2">
              <span className="text-xs text-slate-400 font-medium">Balanced Accuracy</span>
              <p className="text-2xl font-black text-amber-400">{metrics?.balanced_accuracy !== undefined ? metrics.balanced_accuracy : 'N/A'}</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">MAE</span>
              <p className="text-2xl font-black text-indigo-400">{metrics?.mae !== undefined ? metrics.mae : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">RMSE</span>
              <p className="text-2xl font-black text-purple-400">{metrics?.rmse !== undefined ? metrics.rmse : 'N/A'}</p>
            </div>

            <div className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-xs text-slate-400 font-medium">R² Score</span>
              <p className="text-2xl font-black text-emerald-400">{metrics?.r2 !== undefined ? metrics.r2 : 'N/A'}</p>
            </div>
          </div>
        )}
      </div>

      {/* Execution Efficiency Banner */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <span>Execution Efficiency</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Optimization Runtime</span>
            <span className="text-white font-bold text-sm">
              {efficiency?.runtime_seconds !== undefined && efficiency?.runtime_seconds !== null ? `${efficiency.runtime_seconds} seconds` : 'N/A'}
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Total Pipelines Evaluated</span>
            <span className="text-indigo-400 font-bold text-sm">{efficiency?.pipelines_evaluated ?? 'N/A'}</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Generations Run</span>
            <span className="text-emerald-400 font-bold text-sm">{efficiency?.generations ?? 'N/A'}</span>
          </div>
        </div>
      </div>

    </div>
  );
}
