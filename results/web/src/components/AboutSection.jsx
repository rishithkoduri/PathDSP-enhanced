import React from 'react';
import { ArrowRight } from 'lucide-react';
import { MODEL_METRICS } from '../data/mockData';

const biologicalCards = [
  {
    title: 'Growth Signaling',
    body: 'Oncogenic MAPK/ERK and PI3K/AKT cascades drive uncontrolled proliferation and are primary targets for kinase inhibitors.',
  },
  {
    title: 'Survival & Metabolism',
    body: 'Metabolic reprogramming through mTOR and HIF-1α sustains tumor bioenergetics under hypoxic microenvironments.',
  },
  {
    title: 'DNA Repair',
    body: 'Defective homologous recombination and mismatch repair create synthetic lethal vulnerabilities exploited by PARP inhibitors.',
  },
  {
    title: 'Apoptosis & Epigenetics',
    body: 'Dysregulated BCL-2 family signaling and aberrant histone modification silence pro-apoptotic tumor suppressor programs.',
  },
];

const datasetCards = [
  {
    label: 'GDSC1',
    labelColor: 'text-emerald-600 dark:text-emerald-400',
    title: 'Genomics of Drug Sensitivity in Cancer',
    description:
      '48,373 drug–cell line pairs spanning 319 cell lines and 153 compounds with ln(IC₅₀) response values from high-throughput screening.',
  },
  {
    label: 'STRING PPI v11',
    labelColor: 'text-indigo-600 dark:text-indigo-400',
    title: 'Protein–Protein Interaction Network',
    description:
      '99,542 high-confidence edges (combined score ≥ 750) connecting 2,468 genes used as the GATv2 message-passing substrate.',
  },
  {
    label: 'MSigDB PID',
    labelColor: 'text-purple-600 dark:text-purple-400',
    title: 'Curated Pathway Collection',
    description:
      '196 Pathway Interaction Database gene sets providing biologically interpretable hierarchical readout structure.',
  },
];

export default function AboutSection({ onNavigateToWorkspace }) {
  return (
    <div className="max-w-3xl mx-auto py-16 px-6 space-y-16 text-left">
      {/* Block 1: Article Header */}
      <div className="space-y-4">
        <p className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 tracking-widest uppercase mb-2">
          Research Overview
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-50 leading-snug font-sans">
          GNN-PathDSP: Graph Neural Network for Drug Sensitivity Prediction
          Using Cancer Pathway Enrichment
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed mt-3">
          An end-to-end deep learning framework combining protein–protein
          interaction topology with pathway-level cross-modal attention for
          interpretable pharmacogenomic response modeling.
        </p>
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-6 text-[11px] font-mono text-slate-400">
          <span>Architecture: Multi-head GATv2 + Cross-Attention</span>
          <span>Hardware: A100 MIG 20GB</span>
          <span>Checkpoint: best_cv_fold8.pt</span>
        </div>
      </div>

      {/* Block 2: Abstract */}
      <div className="rounded-2xl bg-slate-50 dark:bg-slate-800/40 p-8">
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-5">
          Abstract
        </h2>
        <div className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed space-y-4 font-sans">
          <p>
            Predicting anti-cancer drug sensitivity from multi-omics molecular
            profiles and chemical structures remains a foundational challenge in
            computational pharmacogenomics. Traditional machine learning
            approaches treat gene expression measurements as flat feature
            vectors, discarding the rich biological network topology that
            governs cellular signaling and drug response mechanisms.
          </p>
          <p>
            We introduce <strong>GNN-PathDSP</strong>, a dual-branch graph
            attention architecture that operates directly on the STRING
            protein–protein interaction network (
            <strong>99,542 edges</strong>, <strong>2,468 genes</strong>) to
            learn topologically-aware gene representations. These node
            embeddings are hierarchically aggregated through{' '}
            <strong>196 MSigDB PID pathway tokens</strong> via multi-scale
            readout, then fused with <strong>256-bit Morgan Fingerprints</strong>{' '}
            through a cross-modal attention mechanism that captures
            drug–pathway interaction patterns.
          </p>
          <p>
            Evaluated on the GDSC1 benchmark (
            <strong>48,373 drug–cell line pairs</strong>), GNN-PathDSP achieves
            state-of-the-art 10-fold cross-validation performance:{' '}
            <strong>R² 0.7367 ± 0.0153</strong>,{' '}
            <strong>PCC 0.8605 ± 0.0089</strong>,{' '}
            <strong>RMSE 0.5340 ± 0.0161</strong>. The best checkpoint (
            <code className="text-[11px] bg-slate-100 dark:bg-slate-700/60 px-1.5 py-0.5 rounded">
              best_cv_fold8.pt
            </code>
            ) achieves <strong>RMSE 0.5098</strong> and{' '}
            <strong>PCC 0.8737</strong>, demonstrating robust generalization
            across pharmacogenomic contexts.
          </p>
        </div>
      </div>

      {/* Block 3: Research Motivation */}
      <div>
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-4">
          Research Motivation
        </h2>
        <div className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed space-y-3">
          <p>
            High-throughput drug screening campaigns generate inherently noisy
            dose–response measurements. Classical machine learning baselines —
            random forests, elastic nets, shallow neural networks — tend to
            overfit to transcriptomic hub genes without learning the causal
            signaling cascades that mechanistically determine cellular drug
            response.
          </p>
          <p>
            GNN-PathDSP addresses this by embedding validated biological priors
            directly into the computational graph: STRING PPI edge topology
            constrains message passing to biologically plausible interactions,
            while curated MSigDB pathway representations enforce hierarchical
            aggregation that mirrors real cellular signaling architecture.
          </p>
        </div>
      </div>

      {/* Block 4: Biological Context */}
      <div>
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-4">
          Biological Context
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          Cancer drug efficacy is determined not by individual gene
          perturbations but by coordinated pathway deregulation across
          interconnected signaling modules. Understanding these pathway-level
          interactions is essential for accurate sensitivity prediction.
        </p>
        <div className="grid grid-cols-2 gap-3 mt-4">
          {biologicalCards.map((card) => (
            <div
              key={card.title}
              className="rounded-xl bg-slate-50 dark:bg-slate-800/40 p-4"
            >
              <p className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                {card.title}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                {card.body}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Block 5: Dataset Provenance */}
      <div>
        <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-4">
          Dataset Provenance
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {datasetCards.map((card) => (
            <div
              key={card.label}
              className="rounded-xl bg-slate-50 dark:bg-slate-800/40 p-5"
            >
              <p
                className={`text-[10px] font-mono uppercase tracking-wider ${card.labelColor}`}
              >
                {card.label}
              </p>
              <p className="text-sm font-semibold text-slate-900 dark:text-slate-100 mt-1">
                {card.title}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
                {card.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Block 6: Footer Action */}
      <div className="flex items-center justify-between pt-8">
        <span className="text-[11px] font-mono text-slate-400">
          Checkpoint: best_cv_fold8.pt · RMSE 0.5098
        </span>
        <button
          onClick={onNavigateToWorkspace}
          className="px-5 py-2 rounded-full bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 text-xs font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
        >
          Open Workspace
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
