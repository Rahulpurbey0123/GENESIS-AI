import React, { useState, useEffect, useRef } from 'react';
import { History, Eye, CheckCircle2, RefreshCw, AlertTriangle, ArrowRight } from 'lucide-react';
import { apiService } from '../services/api';

export function HistoryPage({ activeTab, onSelectExperiment }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fetching, setFetching] = useState(false);
  const fetchingRef = useRef(false);

  async function fetchHistory(showLoading = true) {
    if (fetchingRef.current) return;
    try {
      fetchingRef.current = true;
      setFetching(true);
      if (showLoading) setLoading(true);
      setError(null);
      const data = await apiService.listExperiments();
      const sortedData = [...data].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setExperiments(sortedData);
    } catch (err) {
      setError(err.message || 'Failed to load experiment history.');
    } finally {
      setLoading(false);
      fetchingRef.current = false;
      setFetching(false);
    }
  }

  // Fetch on initial mount
  useEffect(() => {
    fetchHistory(true);
  }, []);

  // Fetch whenever History page becomes active
  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory(experiments.length === 0);
    }
  }, [activeTab]);

  // Refresh when browser/tab becomes visible again
  useEffect(() => {
    function handleVisibilityChange() {
      if (document.visibilityState === 'visible' && activeTab === 'history') {
        fetchHistory(false);
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [activeTab, experiments.length]);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400 animate-spin">
          <RefreshCw className="w-6 h-6" />
        </div>
        <p className="text-sm font-semibold text-slate-300">Loading experiment history from database...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center p-6 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400">
        <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
        <p className="font-bold">{error}</p>
        <button
          onClick={() => fetchHistory(true)}
          className="mt-4 px-4 py-2 rounded-xl bg-red-500 text-white text-xs font-semibold hover:bg-red-600 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex items-center justify-between">
        <div>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            SQLite Experiment Store
          </span>
          <h2 className="text-3xl font-extrabold text-white mt-1">Experiment History</h2>
        </div>

        <button
          disabled={fetching}
          onClick={() => fetchHistory(true)}
          className="px-4 py-2 rounded-xl glass-panel glass-panel-hover text-slate-300 text-xs font-semibold flex items-center gap-2 disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${fetching ? 'animate-spin' : ''}`} />
          <span>{fetching ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>

      {/* History Table */}
      {experiments.length === 0 ? (
        <div className="glass-panel p-12 rounded-2xl text-center text-slate-400 space-y-2">
          <History className="w-10 h-10 mx-auto text-slate-600 mb-2" />
          <p className="font-bold text-white">No Experiments Saved Yet</p>
          <p className="text-xs">Upload a dataset and launch an optimization run to populate experiment history.</p>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 bg-slate-900/60">
                  <th className="py-3 px-4 font-semibold">Experiment ID</th>
                  <th className="py-3 px-4 font-semibold">Dataset</th>
                  <th className="py-3 px-4 font-semibold">Target Column</th>
                  <th className="py-3 px-4 font-semibold">Strategy</th>
                  <th className="py-3 px-4 font-semibold">Status</th>
                  <th className="py-3 px-4 font-semibold">Best Score</th>
                  <th className="py-3 px-4 font-semibold">Runtime</th>
                  <th className="py-3 px-4 font-semibold">Created At</th>
                  <th className="py-3 px-4 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {experiments.map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-3 px-4 font-mono font-bold text-indigo-400">{exp.id}</td>
                    <td className="py-3 px-4 font-bold text-white">{exp.dataset_name}</td>
                    <td className="py-3 px-4 font-semibold text-indigo-300 font-mono text-[11px]">{exp.target_column || 'N/A'}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-semibold uppercase text-[10px]">
                        {exp.mode}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                          exp.status === 'COMPLETED'
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : exp.status === 'RUNNING'
                            ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                            : 'bg-red-500/10 text-red-400 border border-red-500/20'
                        }`}
                      >
                        {exp.status}
                      </span>
                    </td>
                    <td className={`py-3 px-4 font-bold ${exp.status === 'FAILED' ? 'text-slate-500' : 'text-emerald-400'}`}>
                      {exp.status === 'FAILED' || exp.best_score === null || exp.best_score === undefined
                        ? 'N/A'
                        : typeof exp.best_score === 'number'
                        ? exp.best_score.toFixed(4)
                        : exp.best_score}
                    </td>

                    <td className="py-3 px-4 text-slate-400">{exp.runtime}s</td>
                    <td className="py-3 px-4 text-slate-500">{new Date(exp.created_at).toLocaleString()}</td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => onSelectExperiment(exp)}
                        className="px-3 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white font-semibold text-[11px] transition-all flex items-center gap-1.5 ml-auto"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}
