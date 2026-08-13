import React from 'react';
import { 
  Activity, 
  Upload, 
  BarChart3, 
  Award, 
  Cpu, 
  CheckCircle2, 
  BrainCircuit, 
  Bot, 
  History,
  Sparkles
} from 'lucide-react';

export function Navbar({ activeTab, setActiveTab, hasDataset, hasExperiment, hasCompletedExperiment }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Sparkles, enabled: true },
    { id: 'upload', label: 'Upload Dataset', icon: Upload, enabled: true },
    { id: 'dip', label: 'DIP Dashboard', icon: BarChart3, enabled: hasDataset },
    { id: 'recommendations', label: 'Recommendations', icon: Award, enabled: hasDataset },
    { id: 'optimization', label: 'Optimization', icon: Cpu, enabled: hasDataset },
    { id: 'results', label: 'Results', icon: CheckCircle2, enabled: hasCompletedExperiment },
    { id: 'explainability', label: 'Explainability', icon: BrainCircuit, enabled: hasCompletedExperiment },
    { id: 'assistant', label: 'AI Assistant', icon: Bot, enabled: hasCompletedExperiment },
    { id: 'history', label: 'History', icon: History, enabled: true },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800 bg-slate-950/80 backdrop-blur-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('home')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-wider text-white">GENESIS-AI</span>
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  DIP v1.1
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Dataset Intelligence AutoML</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  disabled={!item.enabled}
                  onClick={() => item.enabled && setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400'
                      : item.enabled
                      ? 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                      : 'text-slate-600 cursor-not-allowed opacity-50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

        </div>
      </div>
    </header>
  );
}
