import React, { useState } from 'react';
import { Download, FileSpreadsheet, Network, BarChart3, Activity } from 'lucide-react';
import TopologicalGraph from './TopologicalGraph';
import PathwayBarChart from './PathwayBarChart';
import ExplainabilityTable from './ExplainabilityTable';
import { exportToCsv } from '../services/api';

export default function ResultsPanel({ result, isLoading }) {
  const [visTab, setVisTab] = useState('network'); // 'network' | 'pathway'

  if (isLoading) {
    return (
      <div className="w-full h-full min-h-[480px] rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800/60 p-8 flex flex-col items-center justify-center space-y-3 text-center">
        <div className="w-8 h-8 rounded-full border-2 border-slate-300 dark:border-slate-700 border-t-emerald-500 animate-spin" />
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Forward Pass In Progress
          </h4>
          <p className="text-[11px] font-mono text-slate-400">
            A100 GPU: Batched PPI Message Passing &amp; Cross-Attention...
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="w-full h-full min-h-[480px] rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-dashed border-slate-200 dark:border-slate-800 p-8 flex flex-col items-center justify-center space-y-3 text-center">
        <div className="w-10 h-10 rounded-full bg-white dark:bg-slate-900 flex items-center justify-center text-slate-400 shadow-xs">
          <Activity className="w-4 h-4" />
        </div>
        <div className="space-y-1 max-w-xs">
          <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            Awaiting Parameters
          </h4>
          <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
            Configure cell line profile and small molecule on the left, then click <strong>Run Inference</strong>.
          </p>
        </div>
      </div>
    );
  }

  const { metrics, topDriverGenes, topPpiEdges, pathwayAttention, compoundName, cellName } = result;

  const handleExportSummaryCsv = () => {
    exportToCsv(
      `gnn_pathdsp_${compoundName}_${cellName}_summary.csv`,
      [{
        cell: cellName,
        compound: compoundName,
        log10_ic50: metrics.log10Ic50,
        ln_ic50: metrics.lnIc50,
        ic50_um: metrics.ic50Micromolar,
        ci95_lower: metrics.ci95Lower,
        ci95_upper: metrics.ci95Upper,
        category: metrics.category,
        runtime_ms: result.runtimeMs,
        device: result.inferenceDevice,
        timestamp: result.timestamp,
      }],
      ['cell', 'compound', 'log10_ic50', 'ln_ic50', 'ic50_um', 'ci95_lower', 'ci95_upper', 'category', 'runtime_ms', 'device', 'timestamp']
    );
  };

  const categoryStyles = {
    Sensitive: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
    Intermediate: 'bg-amber-500/10 text-amber-600 dark:text-amber-400',
    Resistant: 'bg-rose-500/10 text-rose-600 dark:text-rose-400',
  }[metrics.category] || 'bg-slate-100 text-slate-600';

  return (
    <div className="w-full space-y-6 text-left">
      
      {/* ── 1. PRIMARY METRIC & CONFIDENCE INTERVALS ── */}
      <div className="w-full rounded-2xl bg-white dark:bg-slate-900/60 p-6 text-left border border-slate-200/60 dark:border-slate-800/60 shadow-xs space-y-5">
        
        {/* Header row */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/80 gap-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono uppercase text-slate-400">Response Evaluation</span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-semibold ${categoryStyles}`}>
                {metrics.category}
              </span>
            </div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mt-1">
              {compoundName} &times; {cellName}
            </h3>
          </div>

          <button
            onClick={handleExportSummaryCsv}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100 dark:bg-slate-800 text-[11px] font-mono text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors self-start sm:self-auto"
            title="Download CSV Summary"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-500" />
            <span>Export CSV</span>
          </button>
        </div>

        {/* Primary Metric Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          
          {/* Card 1: IC50 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800/60">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
              Predicted IC₅₀
            </span>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="text-3xl font-mono font-bold text-slate-900 dark:text-slate-50">
                {metrics.ic50Micromolar}
              </span>
              <span className="text-xs font-mono text-slate-400">μM</span>
            </div>
            <div className="text-[11px] font-mono text-slate-400 mt-1">
              ln(IC₅₀) = <strong className="text-slate-700 dark:text-slate-300">{metrics.lnIc50}</strong>
            </div>
          </div>

          {/* Card 2: Normalized log10 */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800/60">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
              Normalized log₁₀(IC₅₀)
            </span>
            <div className="mt-1">
              <span className="text-3xl font-mono font-bold text-slate-900 dark:text-slate-50">
                {metrics.log10Ic50}
              </span>
            </div>
            <div className="text-[11px] font-mono text-slate-400 mt-1">
              Z-score: <strong className="text-slate-700 dark:text-slate-300">{metrics.zScore} σ</strong>
            </div>
          </div>

          {/* Card 3: Confidence Interval */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800/60">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
              95% Confidence Interval
            </span>
            <div className="mt-1">
              <span className="text-2xl font-mono font-bold text-indigo-600 dark:text-indigo-400">
                [{metrics.ci95Lower}, {metrics.ci95Upper}]
              </span>
            </div>
            <div className="text-[11px] font-mono text-slate-400 mt-1">
              Margin: &plusmn;{metrics.ciHalfWidth}
            </div>
          </div>

        </div>

        {/* Visual Confidence Interval Range */}
        <div className="pt-2">
          <div className="flex justify-between items-center text-[11px] font-mono text-slate-400 mb-1.5">
            <span>Visual CI Range (log₁₀ IC₅₀)</span>
            <span>[{metrics.ci95Lower} &larr; {metrics.log10Ic50} &rarr; {metrics.ci95Upper}]</span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
            {(() => {
              const minVal = -1.5;
              const maxVal = 2.5;
              const range = maxVal - minVal;
              const leftPct = Math.max(0, Math.min(100, ((metrics.ci95Lower - minVal) / range) * 100));
              const rightPct = Math.max(0, Math.min(100, ((metrics.ci95Upper - minVal) / range) * 100));
              const widthPct = Math.max(4, rightPct - leftPct);
              const centerPct = Math.max(0, Math.min(100, ((metrics.log10Ic50 - minVal) / range) * 100));

              return (
                <>
                  <div
                    className="absolute top-0 bottom-0 bg-indigo-200 dark:bg-indigo-900/60 rounded-full"
                    style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  />
                  <div
                    className="absolute top-0 bottom-0 w-2 bg-indigo-600 dark:bg-indigo-400 rounded-full -ml-1"
                    style={{ left: `${centerPct}%` }}
                  />
                </>
              );
            })()}
          </div>
          <div className="flex justify-between text-[10px] font-mono text-slate-400 mt-1">
            <span>-1.5 (Sensitive)</span>
            <span>0.5 (Intermediate)</span>
            <span>2.5 (Resistant)</span>
          </div>
        </div>

      </div>

      {/* ── 2. VISUALIZATION PLACEHOLDER CONTAINER ── */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
            Topological &amp; Functional Visualizations
          </span>

          <div className="inline-flex rounded-full bg-slate-100 dark:bg-slate-800 p-0.5 text-[11px] font-mono">
            <button
              onClick={() => setVisTab('network')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full transition-colors ${
                visTab === 'network'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              <Network className="w-3 h-3 text-emerald-500" />
              <span>Topological Graph</span>
            </button>
            <button
              onClick={() => setVisTab('pathway')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full transition-colors ${
                visTab === 'pathway'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              <BarChart3 className="w-3 h-3 text-purple-500" />
              <span>Pathway Distribution</span>
            </button>
          </div>
        </div>

        <div>
          {visTab === 'network' ? (
            <TopologicalGraph
              genes={topDriverGenes}
              edges={topPpiEdges}
              compoundName={compoundName}
              cellName={cellName}
            />
          ) : (
            <PathwayBarChart
              pathways={pathwayAttention}
              compoundName={compoundName}
              cellName={cellName}
            />
          )}
        </div>
      </div>

      {/* ── 3. MECHANISTIC EXPLAINABILITY DATA TABLES ── */}
      <ExplainabilityTable
        genes={topDriverGenes}
        edges={topPpiEdges}
        compoundName={compoundName}
        cellName={cellName}
      />

    </div>
  );
}
