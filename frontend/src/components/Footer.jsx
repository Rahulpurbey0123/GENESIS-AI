import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

export function Footer() {
  return (
    <footer className="mt-16 border-t border-slate-900 bg-slate-950/60 py-8 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>GENESIS-AI &copy; 2026 — Dataset Intelligence Profile-Guided AutoML Framework</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <Info className="w-4 h-4 text-indigo-400" />
          <span>Feature importance indicates predictive contribution. It does not establish causality.</span>
        </div>
      </div>
    </footer>
  );
}
