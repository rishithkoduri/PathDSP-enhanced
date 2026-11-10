import React from 'react';

export default function ArchitectureDiagram() {
  const stages = [
    {
      num: '01',
      title: 'Inputs',
      color: 'text-emerald-500 dark:text-emerald-400',
      badge: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
      items: [
        'Cell Multi-Omics [B, 2468, 3]',
        'Drug Target Affinity [B, 2468, 2]',
        'Morgan Fingerprint [B, 256 bits]',
      ],
      desc: 'Shared STRING PPI Topology (score ≥ 750)',
    },
    {
      num: '02',
      title: 'GATv2 Encoders',
      color: 'text-indigo-500 dark:text-indigo-400',
      badge: 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400',
      items: [
        '2 Layers · 4 Heads · 99.5k Edges',
        'Dynamic α_ij Attention Weights',
        'Cell Hc & Drug Hd [B, N, 64]',
      ],
      desc: 'LayerNorm & Residual Message Passing',
    },
    {
      num: '03',
      title: 'Pathway Readout & Fusion',
      color: 'text-purple-500 dark:text-purple-400',
      badge: 'bg-purple-500/10 text-purple-600 dark:text-purple-400',
      items: [
        '196 PID Tokens · einsum tensor',
        'Global Mean & Max Tokens [B, 198, d]',
        'Cross-Modal Attention (Q: Drug, K/V: Cell)',
      ],
      desc: 'Dynamic Multi-Scale Pathway Cross-Attention',
    },
    {
      num: '04',
      title: 'Prediction & Saliency',
      color: 'text-rose-500 dark:text-rose-400',
      badge: 'bg-rose-500/10 text-rose-600 dark:text-rose-400',
      items: [
        'MLP Head [384 → 512 → 64 → 1]',
        'Target: ln(IC₅₀) / log₁₀(IC₅₀)',
        'Saliency: |∂y/∂x| · |x| Top Drivers',
      ],
      desc: 'Topological Interpretability & Sensitivity',
    },
  ];

  return (
    <div className="w-full rounded-2xl bg-slate-50 dark:bg-slate-800/40 p-6 md:p-8 text-left border border-slate-200/60 dark:border-slate-800/60">
      <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-200/60 dark:border-slate-800/60">
        <div>
          <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 uppercase tracking-widest">
            A100 Tensor Flow
          </span>
          <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mt-0.5">
            End-to-End Computational Pipeline
          </h3>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Dual GATv2 → PID Einsum → Cross-Attn
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stages.map((stage) => (
          <div
            key={stage.num}
            className="rounded-xl bg-white dark:bg-slate-900/80 p-5 flex flex-col justify-between border border-slate-200/50 dark:border-slate-800/50 shadow-xs"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono font-bold text-slate-400">
                  {stage.num}
                </span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-medium ${stage.badge}`}
                >
                  {stage.title}
                </span>
              </div>

              <ul className="space-y-1.5 text-[11px] font-mono text-slate-600 dark:text-slate-300">
                {stage.items.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-1.5">
                    <span className="text-slate-400 select-none">›</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <p className="text-[10px] text-slate-400 dark:text-slate-500 font-sans mt-4 pt-3 border-t border-slate-100 dark:border-slate-800">
              {stage.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
