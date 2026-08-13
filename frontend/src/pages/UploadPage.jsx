import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertTriangle, ArrowRight, Loader2, Target } from 'lucide-react';
import { apiService } from '../services/api';

export function UploadPage({ onDatasetUploaded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [profiling, setProfiling] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      if (!selected.name.toLowerCase().endsWith('.csv')) {
        setError('Please select a valid CSV dataset file (.csv).');
        setFile(null);
        return;
      }
      setError(null);
      setFile(selected);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);

    try {
      const data = await apiService.uploadDataset(file);
      setUploadedData(data);
      if (data.suggested_target) {
        setSelectedTarget(data.suggested_target);
      } else if (data.features && data.features.length > 0) {
        setSelectedTarget(data.features[data.features.length - 1]);
      }
    } catch (err) {
      setError(err.message || 'Unable to upload dataset. Please check that the file is a valid CSV.');
    } finally {
      setUploading(false);
    }
  };

  const handleConfirmTarget = async () => {
    if (!uploadedData || !selectedTarget) return;
    setProfiling(true);
    setError(null);

    try {
      const dipProfile = await apiService.profileDataset(uploadedData.id, selectedTarget);
      onDatasetUploaded(uploadedData, selectedTarget, dipProfile);
    } catch (err) {
      setError(err.message || 'Dataset profiling failed.');
    } finally {
      setProfiling(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 space-y-8">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-extrabold text-white">Dataset Ingestion & Validation</h2>
        <p className="text-sm text-slate-400">
          Upload your tabular CSV dataset to initiate Dataset Intelligence Profiling.
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Upload Failure</p>
            <p className="text-xs opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* File Upload Box */}
      {!uploadedData ? (
        <div className="glass-panel p-8 rounded-2xl border-2 border-dashed border-slate-700 hover:border-indigo-500/50 transition-colors text-center space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto text-indigo-400">
            <Upload className="w-8 h-8" />
          </div>

          <div className="space-y-2">
            <h3 className="text-base font-bold text-white">Select a CSV File</h3>
            <p className="text-xs text-slate-400">Supported format: .csv (Max 50 MB)</p>
          </div>

          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            id="csv-file-input"
          />

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
            <label
              htmlFor="csv-file-input"
              className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs cursor-pointer transition-colors"
            >
              Browse Files
            </label>

            {file && (
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center gap-2 transition-all"
              >
                {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                <span>{uploading ? 'Validating CSV...' : 'Upload & Inspect'}</span>
              </button>
            )}
          </div>

          {file && (
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-slate-300 inline-flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              <span className="font-medium">{file.name}</span>
              <span className="text-slate-500">({(file.size / 1024).toFixed(1)} KB)</span>
            </div>
          )}
        </div>
      ) : (
        /* Uploaded Summary & Target Selection */
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-6 h-6 text-emerald-400" />
              <div>
                <h3 className="text-base font-bold text-white">{uploadedData.name}</h3>
                <p className="text-xs text-slate-400">Dataset Hash: {uploadedData.content_hash.slice(0, 16)}...</p>
              </div>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold border border-emerald-500/20">
              Valid CSV
            </span>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Rows</p>
              <p className="text-2xl font-black text-white">{uploadedData.rows.toLocaleString()}</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Columns</p>
              <p className="text-2xl font-black text-white">{uploadedData.columns}</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Features</p>
              <p className="text-2xl font-black text-white">{uploadedData.columns - 1}</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <p className="text-xs text-slate-400 font-medium">Status</p>
              <p className="text-xs font-bold text-emerald-400 mt-2">Ready for Profiling</p>
            </div>
          </div>

          {/* Target Selection Form */}
          <div className="p-5 rounded-xl bg-indigo-950/30 border border-indigo-500/20 space-y-4">
            <div className="flex items-center gap-2 text-indigo-300 font-bold text-sm">
              <Target className="w-4 h-4" />
              <span>Target Column Selection (Required)</span>
            </div>

            <p className="text-xs text-slate-400">
              Select and confirm the target column to predict. Auto-suggestion provided based on standard column conventions.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4">
              <select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                className="w-full sm:w-auto flex-1 px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-white font-medium text-xs focus:ring-2 focus:ring-indigo-500 outline-none"
              >
                {uploadedData.features.map((col) => (
                  <option key={col} value={col}>
                    {col} {col === uploadedData.suggested_target ? '(Suggested Target)' : ''}
                  </option>
                ))}
              </select>

              <button
                onClick={handleConfirmTarget}
                disabled={profiling || !selectedTarget}
                className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all"
              >
                {profiling ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                <span>{profiling ? 'Generating Profile...' : 'Confirm Target & Generate DIP'}</span>
              </button>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
