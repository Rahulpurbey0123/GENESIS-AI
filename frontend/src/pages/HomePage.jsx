import React from 'react';
import { Upload, BarChart3, Cpu, BrainCircuit, Bot, ShieldCheck, ArrowRight, Layers, Sparkles } from 'lucide-react';

export function HomePage({ onNavigate }) {
  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-12">
      
      {/* Hero Section */}
      <div className="text-center space-y-6 max-w-3xl mx-auto py-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Dataset Intelligence Profile (DIP)-Guided AutoML</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-tight">
          Understand your dataset. <br />
          <span className="glow-gradient-text">Search smarter.</span> <br />
          Explain your model.
        </h1>

        <p className="text-base sm:text-lg text-slate-400 leading-relaxed font-normal">
          GENESIS-AI leverages an interpretable Dataset Intelligence Profile to prune scikit-learn AutoML search spaces prior to evolutionary optimization, paired with post-hoc SHAP and evidence-grounded LLM explainability.
        </p>

        <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            onClick={() => onNavigate('upload')}
            className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-sm shadow-xl shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all duration-200 hover:scale-105"
          >
            <Upload className="w-4 h-4" />
            <span>Upload Dataset</span>
            <ArrowRight className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => onNavigate('history')}
            className="w-full sm:w-auto px-6 py-3.5 rounded-xl glass-panel glass-panel-hover text-slate-300 font-semibold text-sm flex items-center justify-center gap-2"
          >
            <span>View History</span>
          </button>
        </div>
      </div>

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
        
        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-indigo-500/20">
          <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
            <BarChart3 className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Dataset Intelligence Profile (DIP)</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Deterministic extraction of feature types, target missingness, imbalance ratio, Pearson correlations, and DIP complexity score heuristic.
          </p>
        </div>

        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-purple-500/20">
          <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
            <Layers className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Search-Space Reduction</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Rule-based filtering prunes incompatible pipelines and prioritizes high-suitability candidates, reducing GA search evaluation budget.
          </p>
        </div>

        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-emerald-500/20">
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Cpu className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Evolutionary Optimization</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Genetic algorithm optimizes pipeline selection, preprocessing steps, and model hyperparameters across generations.
          </p>
        </div>

        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-pink-500/20">
          <div className="w-12 h-12 rounded-xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center text-pink-400">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Post-Hoc Explainability</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Permutation and native SHAP feature importance summary, local instance contributions, and classification diagnostic metrics.
          </p>
        </div>

        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-sky-500/20">
          <div className="w-12 h-12 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
            <Bot className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Grounded LLM Assistant</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Evidence-grounded explanations powered strictly by stored experiment facts. No hallucinated metrics or unauthorized retraining.
          </p>
        </div>

        <div className="glass-panel glass-panel-hover p-6 rounded-2xl space-y-3 border-amber-500/20">
          <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-white">Honest Science & Auditing</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Strict statistical rules, deterministic recommendation weights, transparent heuristics, and full experiment history preservation.
          </p>
        </div>

      </div>

    </div>
  );
}
