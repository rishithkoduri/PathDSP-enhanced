# GNN-PathDSP: Cancer Drug Sensitivity Prediction via Topological Graph Attention and Cross-Modal Pathway Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-3C2179.svg?style=flat-square&logo=pytorch)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Domain: Bioinformatics](https://img.shields.io/badge/Domain-Computational_Oncology-2ea44f.svg?style=flat-square)]()

---

## 1. Overview & Research Motivation

Accurate prediction of the half-maximal inhibitory concentration ($\text{IC}_{50}$) of anti-cancer therapeutics across diverse patient-derived cancer cell lines is a foundational challenge in precision oncology. High-throughput screening initiatives such as the **Genomics of Drug Sensitivity in Cancer (GDSC)** and the **Cancer Cell Line Encyclopedia (CCLE)** provide expansive multi-omic profiles alongside pharmacological response curves. However, raw transcriptomic and genomic feature spaces suffer from the curse of dimensionality ($>20,000$ genes vs. hundreds of cell lines), leading to severe overfitting when modeled directly.

To mitigate dimensionality while preserving biological interpretability, pathway-level frameworks such as **PathDSP** (Tang & Gottlieb, _Briefings in Bioinformatics_, 2021) transform gene-level multi-omics into condensed cancer signaling pathway enrichment scores (e.g., via ssGSEA and network-based pathway enrichment analysis) alongside molecular fingerprints.

While pathway-enrichment models establish a strong benchmark on random cross-validation splits, they rely on **Feed-Forward Neural Networks (FNNs)** operating on pre-computed, static pathway scalars. This architectural design introduces critical bottlenecks when generalizing to unseen biological contexts. **GNN-PathDSP** resolves these bottlenecks by replacing static enrichment scalars with dynamic, message-passing **Multi-Head Graph Attention Networks (GATv2)** operating directly over the human Protein-Protein Interaction (PPI) topology, coupled with multi-scale pathway readouts and cross-modal attention fusion.

---

## 2. Limitations of the Baseline FNN Architecture (PathDSP)

A rigorous analysis of the baseline FNN-based PathDSP paradigm reveals three fundamental structural limitations:

1. **Topological Blindness (Static Compression):**  
   By collapsing gene expression, somatic mutation, and copy number variation (CNV) profiles into 196 static pathway scalars _prior_ to neural network ingestion, the model discards the underlying signaling topology of the Protein-Protein Interaction (PPI) network. The network cannot dynamically weight cascade interactions or adapt pathway representations to specific chemical perturbations.
2. **Severe In-Distribution Memorization vs. Out-of-Distribution (OOD) Degradation:**  
   Because static pathway vectors act as rigid lookup signatures for each cell line, dense FNN layers rapidly memorize training pair combinations. In the baseline study, this manifests as near-perfect in-distribution 10-Fold Cross-Validation ($R^2 = 0.92$, $\text{PCC} = 0.96$), followed by a precipitous collapse in predictive accuracy under **Leave-One-Cell-Out (LOCO)** and **Leave-One-Drug-Out (LODO)** out-of-distribution evaluations.
3. **Coarse-Grained Explainability:**  
   Feature attribution techniques (e.g., SHAP, Integrated Gradients) applied to the baseline FNN can only rank broad, pre-aggregated pathway names (e.g., _PID_PI3KCI_PATHWAY_). They are mathematically incapable of isolating the exact **driver genes** or **signaling edges** (protein-protein interactions) responsible for drug resistance or sensitivity.

---

## 3. Our Approach: The GNN-PathDSP Architecture

**GNN-PathDSP** reformulates pathway-guided drug sensitivity prediction as a multimodal graph representation learning problem. Rather than feeding static scalars into an MLP, our architecture processes raw multi-omic node states across the biological interactome before projecting into pathway space.

```text
[Cell Multi-Omics: X_cell (N x 3)] ──► [Multi-Head GATv2 Encoder] ──► [Node Embeddings H_c (N x h)]
                                              ▲                                  │
                                      (STRING PPI Graph)                         ▼
                                              ▼                       [Pathway Membership Projection]
[Drug Target Mask: X_drug (N x 2)] ──► [Multi-Head GATv2 Encoder]                │
                                              │                                  ▼
[Morgan Fingerprint (256-bit)] ──────► [Drug Projection MLP] ───────► [Cross-Modal Attention Fusion]
                                                                                 │
                                                                                 ▼
                                                                     [Residual Regression Head]
                                                                                 │
                                                                                 ▼
                                                                     Predicted log10(IC50) + Saliency
```
