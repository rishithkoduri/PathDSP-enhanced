import { useState } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { useGpuStatus } from './hooks/useGpuStatus';
import Header from './components/Header';
import AboutSection from './components/AboutSection';
import ApproachSection from './components/ApproachSection';
import InferenceWorkspace from './components/InferenceWorkspace';
import { MODEL_METRICS } from './data/mockData';

function MainApp() {
  const [activeTab, setActiveTab] = useState('workspace');
  const { gpuState, setComputing } = useGpuStatus(4000);

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-[#0f172a] transition-colors">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        gpuState={gpuState}
      />

      <main className="flex-1">
        {activeTab === 'about' && (
          <AboutSection onNavigateToWorkspace={() => setActiveTab('workspace')} />
        )}
        {activeTab === 'approach' && (
          <ApproachSection metrics={MODEL_METRICS} />
        )}
        {activeTab === 'workspace' && (
          <InferenceWorkspace
            gpuState={gpuState}
            setComputing={setComputing}
            metrics={MODEL_METRICS}
          />
        )}
      </main>

      <footer className="border-t border-slate-200/60 dark:border-slate-800/60">
        <div className="max-w-6xl mx-auto py-4 px-6 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>GNN-PathDSP · Graph-Based Drug Sensitivity Prediction</span>
          <span>A100 MIG 20GB · Checkpoint best_cv_fold8.pt · GDSC1 48.3k pairs</span>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <MainApp />
    </ThemeProvider>
  );
}
