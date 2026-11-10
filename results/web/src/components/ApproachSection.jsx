import React from 'react';
import { Network, Layers, Zap } from 'lucide-react';
import ArchitectureDiagram from './ArchitectureDiagram';

const methodologyCards = [
  {
    icon: Network,
    iconColor: 'text-indigo-500',
    title: 'Multi-head GATv2 PPI Encoder',
    subtitle: 'Dynamic Topological Message Passing',
    subtitleColor: 'text-indigo-500 dark:text-indigo-400',
    paragraphs: [
      'Two-layer GATv2 with 4 attention heads learns dynamic edge-wise attention coefficients over the STRING PPI graph. Each node aggregates neighbour features weighted by learned biological relevance rather than fixed topology.',
      'The attention mechanism computes α_ij = softmax(a^T LeakyReLU(W[h_i || h_j])), enabling the network to re-weight protein interactions per sample based on expression context.',
    ],
    formula: 'h_i^(l+1) = σ( Σ_j α_ij · W · h_j^(l) )',
    footer: '2 Layers · 4 Heads · 99.5k edges',
  },
  {
    icon: Layers,
    iconColor: 'text-emerald-500',
    title: 'Multi-Scale Pathway Readout',
    subtitle: 'MSigDB PID Structural Aggregation',
    subtitleColor: 'text-emerald-500 dark:text-emerald-400',
    paragraphs: [
      'Gene-level GATv2 embeddings are aggregated into 196 pathway-level tokens via masked mean pooling. Each pathway token captures the coordinated activity of its constituent genes within the PPI context.',
      'Two additional global tokens (mean and max over all nodes) are prepended, yielding a [B, 198, 128] pathway tensor that preserves both local pathway specificity and global transcriptomic state.',
    ],
    formula: 'p_k = (1/|S_k|) Σ_{i∈S_k} h_i^(L)  →  [B, 198, 128]',
    footer: '196 Pathways · [B, 198, 128]',
  },
  {
    icon: Zap,
    iconColor: 'text-purple-500',
    title: 'Cross-Modal Attention Fusion',
    subtitle: 'Drug–Pathway Interaction Modeling',
    subtitleColor: 'text-purple-500 dark:text-purple-400',
    paragraphs: [
      'The drug fingerprint vector is projected into a query token that attends over the pathway tensor via multi-head cross-attention. This learns which biological pathways are most relevant to each compound\'s mechanism of action.',
      'The attended representation is concatenated with the drug embedding and passed through a 3-layer MLP with residual connections for final ln(IC₅₀) regression.',
    ],
    formula: 'Attn(Q_drug, K_path, V_path) → fusion → MLP → ŷ',
    footer: '4 Heads · d_model = 128',
  },
];

const ablationRows = [
  {
    model: 'GNN-Concat',
    split: '10-fold CV',
    rmse: '0.5765 ± 0.0172',
    r2: '0.6834 ± 0.0181',
    pcc: '0.8305 ± 0.0109',
    highlight: '',
  },
  {
    model: 'FNN-PathDSP',
    split: '10-fold CV',
    rmse: '0.5501 ± 0.0148',
    r2: '0.7116 ± 0.0156',
    pcc: '0.8463 ± 0.0094',
    highlight: '',
  },
  {
    model: 'GNN-PathDSP Full',
    split: '10-fold CV',
    rmse: '0.5340 ± 0.0161',
    r2: '0.7367 ± 0.0153',
    pcc: '0.8605 ± 0.0089',
    highlight: 'bg-emerald-50/50 dark:bg-emerald-950/20',
  },
  {
    model: 'Best Fold 8',
    split: 'Single Fold',
    rmse: '0.5098',
    r2: '0.7596',
    pcc: '0.8737',
    highlight: 'bg-indigo-50/50 dark:bg-indigo-950/20',
  },
];

export default function ApproachSection() {
  return (
    <div className="max-w-4xl mx-auto py-16 px-6 space-y-16 text-left">
      {/* Block 1: Header */}
      <div>
        <p className="text-[11px] font-mono text-indigo-500 dark:text-indigo-400 tracking-widest uppercase mb-2">
          Methodology
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">
          Architecture &amp; Methodology
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed mt-3 max-w-2xl">
          Combining dynamic graph attention with hierarchical pathway readout
          and cross-modal drug–biology fusion for interpretable drug sensitivity
          prediction.
        </p>
      </div>

      {/* Block 2: Architectural Diagram */}
      <div>
        <p className="text-xs font-mono text-slate-400 mb-3">
          Figure 1 — Computational Dataflow
        </p>
        <ArchitectureDiagram />
      </div>

      {/* Block 3: Methodology Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {methodologyCards.map((card) => {
          const Icon = card.icon;
          return (
            <div
              key={card.title}
              className="rounded-2xl bg-slate-50 dark:bg-slate-800/40 p-6 flex flex-col"
            >
              <div className="w-10 h-10 rounded-xl bg-white dark:bg-slate-900 flex items-center justify-center mb-4 shadow-xs">
                <Icon size={20} className={card.iconColor} />
              </div>

              <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                {card.title}
              </p>
              <p
                className={`text-[11px] font-mono ${card.subtitleColor} mt-1`}
              >
                {card.subtitle}
              </p>

              <div className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed mt-3 space-y-2 flex-1">
                {card.paragraphs.map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
                <div className="rounded-lg bg-white dark:bg-slate-900 p-2.5 font-mono text-[11px] text-slate-600 dark:text-slate-300">
                  {card.formula}
                </div>
              </div>

              <p className="text-[11px] font-mono text-slate-400 pt-3 mt-auto border-t border-slate-200/60 dark:border-slate-800/60">
                {card.footer}
              </p>
            </div>
          );
        })}
      </div>

      {/* Block 4: Ablation Table */}
      <div className="rounded-2xl bg-slate-50 dark:bg-slate-800/40 p-6">
        <div className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center justify-between">
          <span>Ablation Study</span>
          <span className="text-[11px] font-mono text-slate-400 font-normal">
            Identical Folds
          </span>
        </div>

        <table className="w-full text-left text-xs mt-4">
          <thead>
            <tr className="text-[11px] font-mono text-slate-400 uppercase tracking-wider border-b border-slate-200/60 dark:border-slate-800/60">
              <th className="pb-2 font-medium">Model</th>
              <th className="pb-2 font-medium">Split</th>
              <th className="pb-2 font-medium">RMSE↓</th>
              <th className="pb-2 font-medium">R²↑</th>
              <th className="pb-2 font-medium">PCC↑</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/40 dark:divide-slate-800/40">
            {ablationRows.map((row) => (
              <tr key={row.model} className={row.highlight}>
                <td className="py-2.5 font-medium text-slate-900 dark:text-slate-100">
                  {row.model}
                </td>
                <td className="py-2.5 text-slate-500 dark:text-slate-400">
                  {row.split}
                </td>
                <td className="py-2.5 font-mono text-slate-600 dark:text-slate-300">
                  {row.rmse}
                </td>
                <td className="py-2.5 font-mono text-slate-600 dark:text-slate-300">
                  {row.r2}
                </td>
                <td className="py-2.5 font-mono text-slate-600 dark:text-slate-300">
                  {row.pcc}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
