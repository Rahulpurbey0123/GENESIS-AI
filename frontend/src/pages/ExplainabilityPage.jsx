import React, { useState, useEffect } from 'react';
import { BrainCircuit, Info, AlertCircle, BarChart2, ShieldAlert, ArrowRight, Activity } from 'lucide-react';
import { apiService } from '../services/api';
import { PlotViewer } from '../components/PlotViewer';

export function ExplainabilityPage({ experiment, onNavigateAssistant }) {
  const [explanations, setExplanations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const expId = experiment?.id || experiment?.experiment_id;
    if (!expId) {
      setLoading(false);
      return;
    }
    async function fetchExplanations() {
      try {
        setLoading(true);
        const data = await apiService.getExplanations(expId);
        setExplanations(data);
      } catch (err) {
        setError(err.message || 'Failed to load explainability output.');
      } finally {
        setLoading(false);
      }
    }
    fetchExplanations();
  }, [experiment]);

  if (!experiment) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center text-slate-400">
        No experiment selected. Please complete or select an optimization experiment first.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center text-slate-400">
        Generating model interpretability and feature importance explanations...
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

  const { shap, global_importance, evaluation_plots } = explanations || {};

  const method = shap?.method || global_importance?.method || 'unsupported';

  let methodTitle = 'Model Interpretability & Feature Importance';
  let methodBadge = 'Feature Importance';

  if (method === 'shap_tree') {
    methodTitle = 'Model Interpretability & SHAP Insights';
    methodBadge = 'SHAP (Tree Explainer)';
  } else if (method === 'permutation_importance') {
    methodTitle = 'Model Interpretability & Permutation Importance';
    methodBadge = 'Permutation Importance';
  } else if (method === 'linear_coefficients') {
    methodTitle = 'Model Interpretability & Linear Coefficients';
    methodBadge = 'Linear Coefficients';
  } else if (method === 'native_tree') {
    methodTitle = 'Model Interpretability & Native Feature Importance';
    methodBadge = 'Native Feature Importance';
  }

  // Global Feature Importance Chart Data
  const globalFeatures = (global_importance?.features || []).slice(0, 10);
  const globalChartData = [
    {
      x: globalFeatures.map((f) => f.importance),
      y: globalFeatures.map((f) => f.feature),
      type: 'bar',
      orientation: 'h',
      marker: { color: '#818cf8' }
    }
  ];

  // Confusion Matrix Plot
  const confMatrix = evaluation_plots?.confusion_matrix;
  const confChartData = confMatrix ? [
    {
      z: confMatrix,
      x: ['Pred Negative', 'Pred Positive'],
      y: ['Actual Negative', 'Actual Positive'],
      type: 'heatmap',
      colorscale: 'Viridis'
    }
  ] : null;

  // Real ROC Curve Plot
  const rocData = evaluation_plots?.roc_curve;
  const rocChartData = rocData ? [
    {
      x: rocData.fpr,
      y: rocData.tpr,
      type: 'scatter',
      mode: 'lines+markers',
      marker: { color: '#ec4899', size: 5 },
      line: { color: '#f472b6', width: 3 },
      name: 'ROC'
    },
    {
      x: [0, 1],
      y: [0, 1],
      type: 'scatter',
      mode: 'lines',
      line: { color: '#64748b', dash: 'dash' },
      name: 'Baseline'
    }
  ] : null;

  // Real Regression Actual vs Predicted Plot
  const residualsData = evaluation_plots?.residuals;
  const regressionChartData = residualsData ? [
    {
      x: residualsData.y_true,
      y: residualsData.y_pred,
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#38bdf8', size: 7 },
      name: 'Test Samples'
    }
  ] : null;

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-pink-500/20 text-pink-300 border border-pink-500/30">
              Week 5 Post-Hoc Explainability Engine
            </span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Method: {methodBadge}
            </span>
          </div>
          <h2 className="text-3xl font-extrabold text-white mt-1">{methodTitle}</h2>
        </div>

        <button
          onClick={onNavigateAssistant}
          className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center gap-2 transition-all"
        >
          <span>Ask Grounded AI Assistant</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Mandatory Scientific Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-3">
        <Info className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
        <div>
          <span className="font-bold block">Scientific Causality Disclaimer</span>
          <p className="opacity-90 leading-relaxed">
            Feature importance indicates predictive contribution/association in the model via {methodBadge}. It does not establish causality.
          </p>
        </div>
      </div>

      {/* Visualizations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Global Feature Importance Chart */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold text-white">Global Feature Importance Ranking</h3>
          </div>
          <PlotViewer
            data={globalChartData}
            layout={{
              height: 300,
              xaxis: { title: 'Importance Score' },
              yaxis: { autorange: 'reversed' }
            }}
          />
        </div>

        {/* Diagnostic Evaluation Plot */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-bold text-white">
              {rocChartData ? 'ROC Curve (Test Set)' : regressionChartData ? 'Actual vs Predicted (Regression)' : 'Confusion Matrix'}
            </h3>
          </div>

          {rocChartData ? (
            <PlotViewer
              data={rocChartData}
              layout={{
                height: 300,
                xaxis: { title: 'False Positive Rate (FPR)', range: [0, 1] },
                yaxis: { title: 'True Positive Rate (TPR)', range: [0, 1] }
              }}
            />
          ) : regressionChartData ? (
            <PlotViewer
              data={regressionChartData}
              layout={{
                height: 300,
                xaxis: { title: 'Actual Values (y_true)' },
                yaxis: { title: 'Predicted Values (y_pred)' }
              }}
            />
          ) : confChartData ? (
            <PlotViewer
              data={confChartData}
              layout={{
                height: 300
              }}
            />
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-slate-500 italic border border-dashed border-slate-800 rounded-xl">
              Diagnostic evaluation plot unavailable for this dataset/model.
            </div>
          )}
        </div>

      </div>

      {/* Feature Attribution Summary */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h3 className="text-base font-bold text-white">Feature Attribution Summary</h3>
        <div className="grid grid-cols-1 gap-3">
          {(shap?.summary || []).map((item, idx) => (
            <div
              key={idx}
              className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between gap-4 text-xs"
            >
              <div className="space-y-0.5">
                <span className="font-bold text-white block">{item.feature}</span>
                <span className="text-slate-400 text-[11px] block">{item.summary}</span>
              </div>

              <div className="text-right shrink-0 font-mono font-bold text-indigo-400 text-sm">
                {item.mean_shap_value !== undefined && item.mean_shap_value !== null
                  ? item.mean_shap_value.toFixed(4)
                  : item.importance !== undefined && item.importance !== null
                  ? item.importance.toFixed(4)
                  : 'N/A'}
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
