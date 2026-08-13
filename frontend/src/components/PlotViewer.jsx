import React, { useState, useEffect } from 'react';

export function PlotViewer({ data, layout, style, title }) {
  const [Plot, setPlot] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    import('react-plotly.js')
      .then((module) => {
        setPlot(() => module.default);
      })
      .catch((err) => {
        console.error('Failed to load Plotly component:', err);
        setError(true);
      });
  }, []);

  const defaultLayout = {
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
    font: { color: '#94a3b8', family: 'sans-serif', size: 12 },
    margin: { t: 40, r: 20, l: 40, b: 40 },
    xaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' },
    yaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' },
    ...layout
  };

  if (error) {
    return (
      <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/10 text-red-400 text-xs">
        Failed to render chart.
      </div>
    );
  }

  if (!Plot) {
    return (
      <div className="h-64 flex items-center justify-center rounded-xl bg-slate-900/50 border border-slate-800 animate-pulse">
        <span className="text-xs text-slate-500 font-medium">Loading visualization...</span>
      </div>
    );
  }

  return (
    <div className="w-full rounded-xl bg-slate-900/60 border border-slate-800 p-3 overflow-hidden">
      {title && <h4 className="text-xs font-semibold text-slate-300 mb-2 px-1">{title}</h4>}
      <Plot
        data={data}
        layout={defaultLayout}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%', minHeight: '280px', ...style }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  );
}
