# GNN-PathDSP: Pathway-Informed Graph Neural Network for Cancer Drug Sensitivity Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![A100 GPU](https://img.shields.io/badge/NVIDIA-A100--20GB-76b900.svg)](https://nvidia.com)

A production-ready research portal and real-time deep learning inference engine for **GNN-PathDSP**, coupling dual-branch Multi-head Graph Attention Networks (GATv2) with Cross-Modal Attention Fusion for interpretable oncology sensitivity modeling.

---

## 1. Repository Analysis & Optimal Checkpoint

Experimental evaluations were executed on NVIDIA A100 (20 GB MIG slice) across 48,373 drug-cell pairs in GDSC1, filtered against STRING PPI v11.5 (interaction score $\ge 750$, 99,542 edges across 2,468 genes):

| Metric | 10-Fold CV Mean &plusmn; SD | LOCO (Cell-Blind) | LODO (Drug-Blind) | **Leading Checkpoint: Fold 8** |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE** &darr; | $0.5340 \pm 0.0161$ | $0.6054 \pm 0.0220$ | $1.0850 \pm 0.3230$ | **`0.5098`** (Lowest error) |
| **$R^2$** &uarr; | $0.7367 \pm 0.0153$ | $0.6609 \pm 0.0234$ | $-0.2083 \pm 0.3233$ | **`0.7590`** (Highest variance explained) |
| **PCC** &uarr; | $0.8605 \pm 0.0089$ | $0.8216 \pm 0.0109$ | $0.1745 \pm 0.2186$ | **`0.8737`** (Leading linear correlation) |
| **MAE** &darr; | $0.3915 \pm 0.0124$ | $0.4499 \pm 0.0163$ | $0.8550 \pm 0.2335$ | **`0.3744`** |

* **Best Overall Checkpoint:** [`checkpoints/best_cv_fold8.pt`](checkpoints/best_cv_fold8.pt)
  * File size: ~3.2 MB
  * Trained epochs: 38 (Cosine Annealing LR schedule, Huber loss $\delta=1.0$)
  * External Validation: CCLE shared pair RMSE: 0.6854 (beating 0.7190 noise floor).

---

## 2. Web Application Frontend Architecture

The frontend is housed in [`web/`](web/) and built with React, Vite, and Tailwind CSS:

* **Persistent Minimalist Navigation:**
  * Clean academic branding with version tag (`A100-v2`).
  * Seamless view navigation: **About**, **Approach**, and **Inference Workspace**.
  * **GPU Status Indicator:** Pulsing hardware pill (`GPU: Online / Idle` in green, `GPU: Computing` in yellow, `GPU: Offline` in red/gray) with telemetry popover (VRAM usage, tensor utilization, precision BF16/TF32, Kubernetes pod identifier).
  * **Theme Toggle:** Switch between academic paper style (Light) and terminal/developer style (Dark).
* **About View:**
  * Academic abstract with formal typesetting.
  * Research motivation and cancer biological context (PID canonical signaling pathways).
  * Dataset provenance (GDSC1, STRING PPI v11.5, CCLE).
* **Approach View:**
  * Methodology cards describing:
    1. **Multi-Head GATv2 Architecture:** Batched shared topology with LayerNorm and residual message passing.
    2. **Multi-Scale Pathway Readout:** 196 PID pathway tokens + global mean/max tokens via einsum aggregation.
    3. **Cross-Modal Attention Fusion:** Small molecule acts as Query attending to 198 cell/pathway tokens.
  * Interactive architectural diagram schema.
  * Controlled ablation study comparison table.
* **Inference Workspace (The Core Tool):**
  * **Inputs (Left):** Select pre-loaded CCLE/GDSC cell lines (with mutation annotations: BRAF V600E, EGFR ex19del, PIK3CA, etc.) or raw transcriptomic/gene expression arrays (JSON/CSV upload). Input SMILES string (Olaparib, RVX-208, Voxtalisib, etc.) or upload 256-bit Morgan Fingerprints.
  * **Outputs (Right):**
    * Prominent Predicted $\text{IC}_{50}$ ($\mu\text{M}$, $\ln(\text{IC}_{50})$, $\log_{10}$) with 95% Confidence Intervals, Z-score, and Sensitive/Resistant classification.
    * Interactive **Topological PPI Network** (HTML5 Canvas with node dragging, edge attention weights $\alpha_{ij}$, and saliency glow).
    * **Pathway Attention Distribution** bar chart across canonical oncogenic pathways.
    * Mechanistic explainability tables (**Top 15 Driver Genes** with saliency $|\partial y / \partial x| \times |x|$ and **Top Topological PPI Edges**).
    * One-click export buttons (CSV summary, high-res 300-DPI publication PNG).

---

## 3. Quick Start Guide

### Launch FastAPI Backend
```bash
# From workspace root
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Launch React Web Application
```bash
cd web
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your web browser.
*(Note: If the FastAPI backend is not running, the frontend automatically falls back to calibrated standalone simulation, guaranteeing uninterrupted interactive demos.)*
