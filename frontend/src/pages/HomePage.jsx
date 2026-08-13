import React from 'react';
import { Upload, BarChart3, Cpu, BrainCircuit, Bot, ShieldCheck, ArrowRight, Layers, Sparkles } from 'lucide-react';

export function HomePage({ onNavigate }) {
  const featureCards = [
    {
      id: 'dip',
      title: 'Dataset Intelligence Profile (DIP)',
      description: 'Deterministic extraction of feature types, target missingness, imbalance ratio, Pearson correlations, and DIP complexity score heuristic.',
      icon: BarChart3,
      borderClass: 'border-indigo-500/20 hover:border-indigo-500/50 hover:shadow-indigo-500/10',
      iconContainer: 'bg-indigo-500/10 border-indigo-500/20 text-indigo-400',
      tab: 'dip',
      ctaText: 'Explore DIP Dashboard',
      accessibleLabel: 'Open Dataset Intelligence Profile Dashboard',
    },
    {
      id: 'recommendations',
      title: 'Search-Space Reduction',
      description: 'Rule-based filtering prunes incompatible pipelines and prioritizes high-suitability candidates, reducing GA search evaluation budget.',
      icon: Layers,
      borderClass: 'border-purple-500/20 hover:border-purple-500/50 hover:shadow-purple-500/10',
      iconContainer: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
      tab: 'recommendations',
      ctaText: 'View Candidate Recommendations',
      accessibleLabel: 'Open Search-Space Reduction and Recommendations',
    },
    {
      id: 'optimization',
      title: 'Evolutionary Optimization',
      description: 'Genetic algorithm optimizes pipeline selection, preprocessing steps, and model hyperparameters across generations.',
      icon: Cpu,
      borderClass: 'border-emerald-500/20 hover:border-emerald-500/50 hover:shadow-emerald-500/10',
      iconContainer: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
      tab: 'optimization',
      ctaText: 'Configure Optimization',
      accessibleLabel: 'Open Evolutionary Optimization Setup',
    },
    {
      id: 'explainability',
      title: 'Post-Hoc Explainability',
      description: 'Permutation and native SHAP feature importance summary, local instance contributions, and classification diagnostic metrics.',
      icon: BrainCircuit,
      borderClass: 'border-pink-500/20 hover:border-pink-500/50 hover:shadow-pink-500/10',
      iconContainer: 'bg-pink-500/10 border-pink-500/20 text-pink-400',
      tab: 'explainability',
      ctaText: 'View Model Interpretability',
      accessibleLabel: 'Open Post-Hoc Explainability Page',
    },
    {
      id: 'assistant',
      title: 'Grounded LLM Assistant',
      description: 'Evidence-grounded explanations powered strictly by stored experiment facts. No hallucinated metrics or unauthorized retraining.',
      icon: Bot,
      borderClass: 'border-sky-500/20 hover:border-sky-500/50 hover:shadow-sky-500/10',
      iconContainer: 'bg-sky-500/10 border-sky-500/20 text-sky-400',
      tab: 'assistant',
      ctaText: 'Ask AI Assistant',
      accessibleLabel: 'Open Evidence-Grounded LLM Assistant',
    },
    {
      id: 'audit',
      title: 'Honest Science & Auditing',
      description: 'Strict statistical rules, deterministic recommendation weights, transparent heuristics, and full experiment history preservation.',
      icon: ShieldCheck,
      borderClass: 'border-amber-500/20 hover:border-amber-500/50 hover:shadow-amber-500/10',
      iconContainer: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
      tab: 'audit',
      ctaText: 'Inspect Scientific Audit',
      accessibleLabel: 'Open Honest Science and Audit Dashboard',
    },
  ];

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
        {featureCards.map((card) => {
          const Icon = card.icon;
          return (
            <button
              key={card.id}
              type="button"
              onClick={() => onNavigate(card.tab)}
              aria-label={card.accessibleLabel}
              className={`group text-left glass-panel p-6 rounded-2xl space-y-4 border ${card.borderClass} transition-all duration-200 hover:scale-[1.02] hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-950 active:scale-[0.98] cursor-pointer flex flex-col justify-between`}
            >
              <div className="space-y-3">
                <div className={`w-12 h-12 rounded-xl border flex items-center justify-center ${card.iconContainer}`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-white group-hover:text-indigo-200 transition-colors">
                  {card.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {card.description}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs font-semibold text-slate-400 group-hover:text-indigo-400 transition-colors">
                <span>{card.ctaText}</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-200" />
              </div>
            </button>
          );
        })}
      </div>

    </div>
  );
}
