import React, { useState } from 'react';
import { Download, ExternalLink } from 'lucide-react';
import { exportToCsv } from '../services/api';

export default function ExplainabilityTable({ genes = [], edges = [], compoundName, cellName }) {
  const [activeTab, setActiveTab] = useState('genes'); // 'genes' | 'edges'

  const exportGenes = () => {
    exportToCsv(
      `top15_driver_genes_${compoundName}_${cellName}.csv`,
      genes.map((g) => ({
        rank: g.rank,
        gene: g.gene,
        saliency_score: g.saliency,
        biological_role: g.role,
      })),
      ['rank', 'gene', 'saliency_score', 'biological_role']
    );
  };

  const exportEdges = () => {
    exportToCsv(
      `top15_ppi_edges_${compoundName}_${cellName}.csv`,
      edges.map((e) => ({
        rank: e.rank,
        source: e.source,
        target: e.target,
        gatv2_attention: e.attention,
        string_score: e.score,
        pathway: e.pathway,
      })),
      ['rank', 'source', 'target', 'gatv2_attention', 'string_score', 'pathway']
    );
  };

  return (
    <div className="w-full rounded-2xl bg-white dark:bg-slate-900/60 p-6 text-left border border-slate-200/60 dark:border-slate-800/60 shadow-xs">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
        <div>
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Mechanistic Explainability
          </h4>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            Topological saliency |∂y/∂x|·|x| and GATv2 attention weights on STRING PPI
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Pill Toggle */}
          <div className="inline-flex rounded-full bg-slate-100 dark:bg-slate-800 p-0.5 text-[11px] font-mono">
            <button
              onClick={() => setActiveTab('genes')}
              className={`px-3 py-1 rounded-full transition-colors ${
                activeTab === 'genes'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              Top 15 Driver Genes
            </button>
            <button
              onClick={() => setActiveTab('edges')}
              className={`px-3 py-1 rounded-full transition-colors ${
                activeTab === 'edges'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-medium shadow-xs'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
              }`}
            >
              Top PPI Edges
            </button>
          </div>

          <button
            onClick={activeTab === 'genes' ? exportGenes : exportEdges}
            className="p-1.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
            title="Export CSV"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto">
        {activeTab === 'genes' ? (
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-400">
                <th className="pb-2.5 font-medium w-12">#</th>
                <th className="pb-2.5 font-medium w-24">Gene</th>
                <th className="pb-2.5 font-medium w-36">Saliency Score</th>
                <th className="pb-2.5 font-medium">Biological Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100/80 dark:divide-slate-800/50">
              {genes.slice(0, 15).map((g) => (
                <tr
                  key={g.rank}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                >
                  <td className="py-2.5 font-mono text-slate-400">#{g.rank}</td>
                  <td className="py-2.5 font-mono font-semibold text-slate-900 dark:text-slate-100">
                    <a
                      href={`https://www.genecards.org/cgi-bin/carddisp.pl?gene=${g.gene}`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 hover:text-emerald-500 transition-colors"
                    >
                      <span>{g.gene}</span>
                      <ExternalLink className="w-2.5 h-2.5 text-slate-400" />
                    </a>
                  </td>
                  <td className="py-2.5 font-mono">
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                        {(g.saliency || 0).toFixed(4)}
                      </span>
                      <div className="w-12 h-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{ width: `${Math.min(100, (g.saliency || 0) * 1000)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400 font-sans text-xs">
                    {g.role}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-400">
                <th className="pb-2.5 font-medium w-12">#</th>
                <th className="pb-2.5 font-medium w-36">Interaction</th>
                <th className="pb-2.5 font-medium w-32">GATv2 Attention α</th>
                <th className="pb-2.5 font-medium w-24">STRING</th>
                <th className="pb-2.5 font-medium">Pathway</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100/80 dark:divide-slate-800/50">
              {edges.slice(0, 15).map((e) => (
                <tr
                  key={e.rank}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                >
                  <td className="py-2.5 font-mono text-slate-400">#{e.rank}</td>
                  <td className="py-2.5 font-mono font-medium text-slate-900 dark:text-slate-100">
                    <span className="text-indigo-600 dark:text-indigo-400">{e.source}</span>
                    <span className="text-slate-400 mx-1.5">→</span>
                    <span className="text-indigo-600 dark:text-indigo-400">{e.target}</span>
                  </td>
                  <td className="py-2.5 font-mono">
                    <div className="flex items-center gap-2">
                      <span className="text-indigo-600 dark:text-indigo-400 font-medium">
                        {(e.attention || 0).toFixed(4)}
                      </span>
                      <div className="w-10 h-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-indigo-500"
                          style={{ width: `${Math.min(100, (e.attention || 0) * 2000)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="py-2.5 font-mono text-slate-500">{e.score}</td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400 font-sans text-xs">
                    <span className="px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-[10px] font-mono">
                      {e.pathway}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
