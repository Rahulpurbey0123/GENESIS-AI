import React from 'react';
import { BarChart3, AlertCircle, Database, Layers, Hash, PieChart, ShieldAlert } from 'lucide-react';
import { PlotViewer } from '../components/PlotViewer';

export function DipDashboardPage({ dipProfile, onNext }) {
  if (!dipProfile) {
    return (
      <div className="max-w-4xl mx-auto py-12 text-center text-slate-400">
        No dataset profile available. Please upload a dataset first.
      </div>
    );
  }

  const { dataset, schema, quality, target, complexity_score, complexity_detail, complexity_level, dataset_name } = dipProfile;

  // Helper function to extract count whether backend returns number or list
  const getCount = (val) => {
    if (typeof val === 'number') return val;
    if (Array.isArray(val)) return val.length;
    return 0;
  };

  const numCount = getCount(schema?.numeric_features ?? schema?.continuous_numerical);
  const catCount = getCount(schema?.categorical_features ?? schema?.categorical_multiclass);
  const binCount = getCount(schema?.binary_features);
  const boolCount = getCount(schema?.boolean_features);
  const dtCount = getCount(schema?.datetime_features);

  const rawValues = [numCount, catCount, binCount, boolCount, dtCount];
  const rawLabels = ['Numerical', 'Categorical', 'Binary', 'Boolean', 'Datetime'];
  const colors = ['#6366f1', '#a855f7', '#ec4899', '#38bdf8', '#10b981'];

  // Filter labels and values for pie chart
  const activeIndices = rawValues.map((v, i) => (v > 0 || i < 2 ? i : -1)).filter((i) => i !== -1);
  const chartValues = activeIndices.map((i) => rawValues[i]);
  const chartLabels = activeIndices.map((i) => rawLabels[i]);
  const chartColors = activeIndices.map((i) => colors[i]);

  const featureTypeData = [
    {
      values: chartValues,
      labels: chartLabels,
      type: 'pie',
      hole: 0.4,
      marker: {
        colors: chartColors
      },
      textinfo: 'label+value',
      textposition: 'outside',
      hoverinfo: 'label+value'
    }
  ];

  // Missingness Profile Chart Data
  // Access dictionary per_feature_missing_rates or feature_missingness object
  const rawMissingness = quality?.feature_missingness;
  let missingDict = {};
  if (rawMissingness && typeof rawMissingness === 'object') {
    if (rawMissingness.per_feature_missing_rates && typeof rawMissingness.per_feature_missing_rates === 'object') {
      missingDict = rawMissingness.per_feature_missing_rates;
    } else {
      // Filter out internal metadata keys if present
      const internalKeys = new Set([
        'total_missing', 'missing_rate', 'columns_with_missing',
        'max_column_missing_rate', 'per_feature_missing_rates', 'target_missingness'
      ]);
      Object.keys(rawMissingness).forEach((k) => {
        if (!internalKeys.has(k) && typeof rawMissingness[k] === 'number') {
          missingDict[k] = rawMissingness[k];
        }
      });
    }
  }

  const missingnessFeatures = Object.keys(missingDict).filter((f) => typeof missingDict[f] === 'number').slice(0, 15);
  const missingnessValues = missingnessFeatures.map((f) => {
    const val = missingDict[f];
    return val <= 1.0 ? val * 100 : val;
  });

  const missingnessChartData = [
    {
      x: missingnessFeatures,
      y: missingnessValues,
      type: 'bar',
      marker: {
        color: missingnessValues.map((v) => (v > 20 ? '#ef4444' : v > 5 ? '#f59e0b' : '#10b981'))
      }
    }
  ];

  // Complexity label extraction
  const displayComplexityLabel = complexity_detail?.label || complexity_level || 'N/A';

  // Target Missingness extraction
  let targetMissingRateStr = 'N/A';
  const tMiss = quality?.target_missingness;
  if (typeof tMiss === 'object' && tMiss !== null && tMiss.missing_rate !== undefined && tMiss.missing_rate !== null) {
    targetMissingRateStr = `${(tMiss.missing_rate <= 1.0 ? tMiss.missing_rate * 100 : tMiss.missing_rate).toFixed(1)}%`;
  } else if (typeof tMiss === 'number') {
    targetMissingRateStr = `${(tMiss <= 1.0 ? tMiss * 100 : tMiss).toFixed(1)}%`;
  } else if (target?.missing_rate !== undefined && target?.missing_rate !== null) {
    targetMissingRateStr = `${(target.missing_rate <= 1.0 ? target.missing_rate * 100 : target.missing_rate).toFixed(1)}%`;
  }

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              DIP v1.1 Profile
            </span>
            <span className="text-xs text-slate-400 font-mono">ID: {dipProfile.dataset_hash?.slice(0, 12)}</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white mt-1">{dataset_name} Profile</h2>
        </div>

        <button
          onClick={onNext}
          className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 transition-all"
        >
          View Recommendations →
        </button>
      </div>

      {/* Primary Complexity Score Banner */}
      <div className="glass-panel p-6 rounded-2xl border-l-4 border-indigo-500 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-white">Complexity Score (heuristic)</h3>
            <span className="text-[10px] text-slate-400 font-medium px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
              Engineering Metric
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Weighted 0–10 heuristic based on feature dimensionality, missingness rate, imbalance ratio, and sample size.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-center">
            <span className="text-4xl font-black text-indigo-400">{complexity_score ?? 'N/A'}</span>
            <span className="text-xs text-slate-500 font-bold"> / 10</span>
          </div>
          <span className="px-4 py-1.5 rounded-xl bg-indigo-500/20 text-indigo-300 text-xs font-bold border border-indigo-500/30 uppercase tracking-wider">
            {displayComplexityLabel} Complexity
          </span>
        </div>
      </div>

      {/* Metrics Stat Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium">Rows</span>
            <Database className="w-4 h-4 text-indigo-400" />
          </div>
          <p className="text-2xl font-black text-white">{dataset?.rows?.toLocaleString() ?? 'N/A'}</p>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium">Features</span>
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-black text-white">{dataset?.feature_count ?? dataset?.columns ?? 'N/A'}</p>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium">Missing Rate</span>
            <AlertCircle className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-black text-white">
            {quality?.missing_rate !== undefined && quality?.missing_rate !== null
              ? `${(quality.missing_rate * 100).toFixed(1)}%`
              : quality?.overall_missing_rate !== undefined
              ? `${(quality.overall_missing_rate * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium">Duplicate Rate</span>
            <Hash className="w-4 h-4 text-sky-400" />
          </div>
          <p className="text-2xl font-black text-white">
            {quality?.duplicate_rate !== undefined && quality?.duplicate_rate !== null
              ? `${(quality.duplicate_rate * 100).toFixed(1)}%`
              : quality?.duplicate_rows_ratio !== undefined
              ? `${(quality.duplicate_rows_ratio * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>

        <div className="glass-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium">Imbalance Ratio</span>
            <ShieldAlert className="w-4 h-4 text-pink-400" />
          </div>
          <p className="text-2xl font-black text-white">
            {target?.imbalance_ratio !== undefined && target?.imbalance_ratio !== null
              ? target.imbalance_ratio.toFixed(2)
              : 'N/A'}
          </p>
        </div>

      </div>

      {/* Visualizations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Feature Types Chart */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center gap-2">
            <PieChart className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold text-white">Feature Type Breakdown</h3>
          </div>
          <PlotViewer
            data={featureTypeData}
            layout={{
              height: 280,
              showlegend: true,
              legend: { orientation: 'h', y: -0.1 }
            }}
          />
        </div>

        {/* Missingness Profile Chart */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">Feature Missingness Rate (%)</h3>
          </div>
          <PlotViewer
            data={missingnessChartData}
            layout={{
              height: 280,
              yaxis: { title: 'Missing (%)', range: [0, 100] },
              xaxis: { tickangle: -30 }
            }}
          />
        </div>

      </div>

      {/* Target Details Card */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h3 className="text-base font-bold text-white">Target Column Signal Summary</h3>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs">
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Target Name</span>
            <span className="text-white font-bold text-sm">{target?.name || 'N/A'}</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Detected Task Type</span>
            <span className="text-indigo-400 font-bold text-sm uppercase">{target?.task_type || 'N/A'}</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Unique Classes / Distinct Values</span>
            <span className="text-white font-bold text-sm">{target?.class_count ?? target?.distinct_values ?? 'N/A'}</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 block font-medium">Target Missingness</span>
            <span className="text-emerald-400 font-bold text-sm">{targetMissingRateStr}</span>
          </div>
        </div>
      </div>

    </div>
  );
}
