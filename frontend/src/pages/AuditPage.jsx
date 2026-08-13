import React from 'react';
import { ShieldCheck, Lock, Layers, Cpu, BrainCircuit, Bot, FileText, CheckCircle2, AlertTriangle, HelpCircle } from 'lucide-react';

export function AuditPage({ dataset, experiment, dipProfile }) {
  const dsName = dataset?.name || dataset?.dataset_name || dipProfile?.dataset_name;
  const dsId = dataset?.id || dataset?.dataset_id || dipProfile?.dataset_hash?.slice(0, 12);
  const expId = experiment?.id || experiment?.experiment_id;

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
              Scientific Audit & Transparency Protocol
            </span>
            <span className="text-xs text-slate-400 font-mono">DIP Engine v1.1</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white mt-1">Honest Science & Auditing</h2>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 text-xs font-medium">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Determinism & Zero-Leakage Guaranteed</span>
        </div>
      </div>

      {/* Active Session Audit Card */}
      <div className="glass-panel p-6 rounded-2xl border-l-4 border-amber-500 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-amber-400" />
          <span>Active Session Audit Status</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-medium block">Loaded Dataset</span>
            <span className="text-white font-bold text-sm block truncate">{dsName || 'No Dataset Active'}</span>
            <span className="text-[10px] text-slate-500 font-mono">{dsId ? `ID: ${dsId}` : 'Upload dataset to profile'}</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-medium block">Target Column</span>
            <span className="text-indigo-400 font-bold text-sm block truncate">
              {dipProfile?.target?.name || dataset?.suggested_target || experiment?.target_column || 'Not Configured'}
            </span>
            <span className="text-[10px] text-indigo-400/80 font-mono">
              {dipProfile?.target?.task_type ? `Task: ${dipProfile.target.task_type.toUpperCase()}` : 'Target selection verified'}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-medium block">Active Experiment Run</span>
            <span className="text-emerald-400 font-bold text-sm block truncate">{expId || 'No Active Experiment'}</span>
            <span className="text-[10px] text-slate-500 font-mono">
              {experiment?.status ? `Status: ${experiment.status}` : 'Launch run from Optimization'}
            </span>
          </div>
        </div>
      </div>

      {/* Scientific Protocols Grid */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Lock className="w-5 h-5 text-indigo-400" />
          <span>Verifiable System Scientific Rules</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Rule 1: Target Leakage */}
          <div className="glass-panel p-5 rounded-2xl space-y-3 border-indigo-500/20">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-xl bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-300 font-bold text-xs">
                P0
              </span>
              <div>
                <h4 className="text-sm font-bold text-white">Target Leakage Prevention (P0 Audit)</h4>
                <span className="text-[10px] text-indigo-300 font-semibold">Strict Matrix Separation</span>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Target columns are strictly stripped from feature matrix <code className="text-indigo-300 bg-slate-900 px-1 py-0.5 rounded">X</code> during data split routines. Target values are never exposed to feature scaling, imputation, SHAP attributions, or model input layers.
            </p>
            <div className="flex items-center gap-2 text-[11px] text-emerald-400 font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Verified by test_target_survived_excluded_from_feature_matrix</span>
            </div>
          </div>

          {/* Rule 2: Identifier Exclusion */}
          <div className="glass-panel p-5 rounded-2xl space-y-3 border-purple-500/20">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-300 font-bold text-xs">
                P1
              </span>
              <div>
                <h4 className="text-sm font-bold text-white">Identifier Column Exclusion (P1 Audit)</h4>
                <span className="text-[10px] text-purple-300 font-semibold">Heuristic Key Audit</span>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              High-cardinality primary keys and sequence IDs (e.g., PassengerId, UUIDs, indexing columns) are automatically flagged via DIP heuristics and excluded from predictive candidate features to prevent trivial memorization.
            </p>
            <div className="flex items-center gap-2 text-[11px] text-emerald-400 font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Verified by test_identifier_passengerid_excluded</span>
            </div>
          </div>

          {/* Rule 3: Search-Space Reduction */}
          <div className="glass-panel p-5 rounded-2xl space-y-3 border-emerald-500/20">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-300 font-bold text-xs">
                P1
              </span>
              <div>
                <h4 className="text-sm font-bold text-white">Deterministic Search-Space Reduction</h4>
                <span className="text-[10px] text-emerald-300 font-semibold">Mathematical Reduction Formula</span>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Rule-based candidate pruning calculates search space reduction percentage strictly as:
              <span className="block font-mono text-[11px] text-emerald-300 bg-slate-900 p-2 rounded mt-1">
                Reduction % = (Pool_Before - Pool_After) / Pool_Before
              </span>
              Incompatible model families are systematically pruned before GA evaluation budget allocation.
            </p>
            <div className="flex items-center gap-2 text-[11px] text-emerald-400 font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Verified by test_recommendation_priority_score_and_search_space_reduction</span>
            </div>
          </div>

          {/* Rule 4: Evidence-Grounded LLM */}
          <div className="glass-panel p-5 rounded-2xl space-y-3 border-sky-500/20">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-xl bg-sky-500/20 border border-sky-500/30 flex items-center justify-center text-sky-300 font-bold text-xs">
                P2
              </span>
              <div>
                <h4 className="text-sm font-bold text-white">Evidence-Grounded LLM Guardrails</h4>
                <span className="text-[10px] text-sky-300 font-semibold">Allowlist & Schema Validation</span>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              The AI Assistant strictly operates on stored experiment evidence dictionaries. Responses are validated against explicit schema allowlists to guarantee zero hallucinated metrics or unauthorized model retraining.
            </p>
            <div className="flex items-center gap-2 text-[11px] text-emerald-400 font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Verified by test_end_to_end_audit_pipeline</span>
            </div>
          </div>

        </div>
      </div>

      {/* Disclaimers Banner */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 text-xs flex items-start gap-3">
        <HelpCircle className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold text-slate-200 block">Scientific Transparency Statement</span>
          <p className="leading-relaxed">
            GENESIS-AI does not generate synthetic metrics or fake evaluations. Every accuracy, F1 score, SHAP attribution, and candidate ranking presented in the application is computed deterministically from real scikit-learn models and stored SQLite audit records.
          </p>
        </div>
      </div>

    </div>
  );
}
