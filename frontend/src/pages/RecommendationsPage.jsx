import React, { useState, useEffect } from 'react';
import { Award, Layers, CheckCircle2, AlertCircle, ArrowRight, ShieldCheck, Cpu } from 'lucide-react';
import { apiService } from '../services/api';

export function RecommendationsPage({ dataset, targetColumn, onStartOptimization }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const dsId = dataset?.id || dataset?.dataset_id;

  useEffect(() => {
    if (!dsId || !targetColumn) {
      setLoading(false);
      return;
    }
    async function fetchRecs() {
      try {
        setLoading(true);
        // Read-only recommendation call — does NOT create an experiment
        const recData = await apiService.getDatasetRecommendations(dsId, targetColumn);
        setRecommendations(recData);
      } catch (err) {
        setError(err.message || 'Failed to fetch model recommendations.');
      } finally {
        setLoading(false);
      }
    }
    fetchRecs();
  }, [dataset, targetColumn, dsId]);

  if (!dsId) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center text-slate-400">
        No dataset selected. Please upload a dataset first.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400 animate-spin">
          <Cpu className="w-6 h-6" />
        </div>
        <p className="text-sm font-semibold text-slate-300">Generating Pipeline Recommendations & Search-Space Reduction...</p>
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

  const candidateBefore = recommendations?.candidate_count_before ?? 'N/A';
  const candidateAfter = recommendations?.candidate_count_after_filtering ?? 'N/A';
  const reductionPct = recommendations?.search_space_reduction !== undefined && recommendations?.search_space_reduction !== null
    ? `${Math.round(recommendations.search_space_reduction * 100)}% Reduction`
    : 'N/A';

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
            DIP-Guided Search Space Reduction
          </span>
          <h2 className="text-3xl font-extrabold text-white mt-1">Pipeline Candidate Prioritization</h2>
        </div>

        <button
          onClick={onStartOptimization}
          className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-xl shadow-indigo-600/30 flex items-center gap-2 transition-all hover:scale-105"
        >
          <Cpu className="w-4 h-4" />
          <span>Launch Evolutionary Optimization</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Search-Space Reduction Statistics Banner */}
      <div className="glass-panel p-6 rounded-2xl bg-gradient-to-r from-indigo-950/40 to-purple-950/40 border border-indigo-500/30 space-y-4">
        <div className="flex items-center gap-2 text-indigo-300 font-bold text-sm">
          <Layers className="w-5 h-5 text-indigo-400" />
          <span>Search-Space Reduction Analysis (Real Calculation)</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
            <span className="text-xs text-slate-400 font-medium block">Original Candidate Pool</span>
            <span className="text-3xl font-black text-slate-200">{candidateBefore} {typeof candidateBefore === 'number' ? 'Models' : ''}</span>
            <span className="text-[10px] text-slate-500 block mt-1">Full scikit-learn search space</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
            <span className="text-xs text-slate-400 font-medium block">DIP Filtered Candidates</span>
            <span className="text-3xl font-black text-indigo-400">{candidateAfter} {typeof candidateAfter === 'number' ? 'Candidates' : ''}</span>
            <span className="text-[10px] text-indigo-400/80 block mt-1">Compatible model family pool</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/70 border border-indigo-500/40">
            <span className="text-xs text-slate-400 font-medium block">Search Space Reduction</span>
            <span className="text-3xl font-black text-emerald-400">{reductionPct}</span>
            <span className="text-[10px] text-emerald-400/80 block mt-1">Fewer GA evaluations required</span>
          </div>
        </div>
      </div>

      {/* Recommendations Cards List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-purple-400" />
            <span>Recommended Model Architectures</span>
          </h3>
          <span className="text-xs text-slate-400 italic">
            "Higher-priority candidates based on dataset profiling rules"
          </span>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {(recommendations?.recommendations || []).map((rec, index) => (
            <div
              key={rec.pipeline_id || index}
              className="glass-panel p-5 rounded-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 hover:border-purple-500/40 transition-colors"
            >
              <div className="space-y-2 max-w-2xl">
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-300 font-extrabold text-xs">
                    #{index + 1}
                  </span>
                  <h4 className="text-base font-bold text-white">{rec.display_name || rec.pipeline_id}</h4>
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-semibold">
                    Higher-priority candidate
                  </span>
                </div>

                {/* Reasons List */}
                <div className="flex flex-wrap gap-2 pt-1">
                  {(rec.explanations || []).map((rule, rIdx) => (
                    <span
                      key={rIdx}
                      className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 text-xs flex items-center gap-1.5"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />
                      <span>{typeof rule === 'string' ? rule : rule.description || rule.rule_code}</span>
                    </span>
                  ))}
                </div>
              </div>

              <div className="text-right shrink-0">
                <span className="text-xs text-slate-400 font-medium block">Priority Score</span>
                <span className="text-2xl font-black text-purple-400">
                  {rec.score !== undefined && rec.score !== null
                    ? (rec.score * 1.0).toFixed(2)
                    : rec.suitability_score !== undefined && rec.suitability_score !== null
                    ? (rec.suitability_score * 1.0).toFixed(2)
                    : rec.priority !== undefined && rec.priority !== null
                    ? (rec.priority * 1.0).toFixed(2)
                    : 'Recommended'}
                </span>
              </div>

            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
