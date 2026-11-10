import React from 'react';
import { Download } from 'lucide-react';
import { exportToCsv } from '../services/api';

export default function PathwayBarChart({ pathways = [], compoundName, cellName }) {
  const maxAttention = Math.max(...pathways.map((p) => p.attention || 0), 0.01);

  const handleExportCsv = () => {
    exportToCsv(
      `pathway_attention_${compoundName}_${cellName}.csv`,
      pathways.map((p, idx) => ({
        rank: idx + 1,
        pathway: p.name,
        category: p.category,
        attention: p.attention,
      })),
      ['rank', 'pathway', 'category', 'attention']
    );
  };

  return (
    <div className="w-full rounded-2xl bg-white dark:bg-slate-900/60 p-6 text-left border border-slate-200/60 dark:border-slate-800/60 shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Pathway Attention Distribution
          </h4>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            Normalized cross-attention weights across 196 MSigDB PID pathway tokens
          </p>
        </div>

        <button
          onClick={handleExportCsv}
          className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-[11px] font-mono text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors"
          title="Export CSV"
        >
          <Download className="w-3 h-3" />
          <span>CSV</span>
        </button>
      </div>

      {/* Bars */}
      <div className="space-y-3">
        {pathways.map((p, idx) => {
          const pct = Math.round(((p.attention || 0) / maxAttention) * 100);
          return (
            <div key={idx} className="group">
              <div className="flex items-center justify-between text-[11px] mb-1">
                <span className="font-mono text-slate-700 dark:text-slate-300 truncate max-w-[280px] sm:max-w-md">
                  {p.name}
                </span>
                <span className="font-mono text-slate-400 ml-2">
                  {((p.attention || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-emerald-500 dark:bg-emerald-400 transition-all duration-500 group-hover:brightness-110"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
        <span>Query: Drug Target + 256-bit ECFP</span>
        <span>Keys/Values: PID Pathways</span>
      </div>
    </div>
  );
}
