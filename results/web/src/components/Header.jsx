import { Dna } from 'lucide-react';
import GpuPill from './GpuPill';
import ThemeToggle from './ThemeToggle';

const tabs = [
  { key: 'about', label: 'About' },
  { key: 'approach', label: 'Approach' },
  { key: 'workspace', label: 'Workspace' },
];

export default function Header({ activeTab, setActiveTab, gpuState }) {
  return (
    <header className="sticky top-0 z-40 w-full h-14 bg-white/90 dark:bg-[#0f172a]/90 backdrop-blur-md border-b border-slate-200/60 dark:border-slate-800/60">
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between h-full">
        {/* Branding */}
        <div className="flex items-center gap-2">
          <Dna className="w-4 h-4 text-emerald-500" />
          <span className="text-sm font-semibold font-mono text-slate-900 dark:text-slate-100">
            GNN-PathDSP
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-1">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-4 py-1.5 rounded-full text-[13px] font-medium transition-all duration-150 ${
                activeTab === key
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100/80 dark:hover:bg-slate-800/60'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        {/* Status & Theme */}
        <div className="flex items-center gap-3">
          <GpuPill gpuState={gpuState} />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
