# PathDSP-PIU: Drug-Conditioned Pathway Interaction Learning with Uncertainty-Aware Evaluation

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3%2BCUDA-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB)-76b900.svg)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, scientifically rigorous deep learning framework for anticancer drug sensitivity prediction.

This project implements and extends the cancer pathway enrichment concepts of **Tang & Gottlieb (2021)** (*Scientific Reports*, [doi:10.1038/s41598-021-82612-7](https://doi.org/10.1038/s41598-021-82612-7)), introducing **drug-conditioned pathway gating**, **multiplicative interaction learning**, and **deep ensemble conformal uncertainty calibration**.

---

## 1. Key Architectural Innovations

1. **Drug-Conditioned Pathway Gating**: Dynamically computes pathway-specific modulation factors $g(d, p) \in [0, 1]$ based on joint drug chemical embeddings, drug-target enrichment, and cell pathway state.
2. **Low-Rank Multiplicative Interaction**: Explicitly models drug–pathway synergy $z(d, p) = (U e_d) \odot (V h'_{c, p})$ aggregated via attention scoring weights $\alpha(d, p)$.
3. **Rigorous Generalization Benchmarking**: Evaluates out-of-distribution cold-start performance across:
   - **Leave-One-Drug-Out (LODO)** (Unseen chemical compounds)
   - **Leave-One-Cell-Line-Out (LOCO)** (Unseen cancer patients / models)
   - **Double-Cold-Start** (Simultaneously unseen drugs and cell lines)
4. **Uncertainty Quantification**: 3-model deep ensemble with split conformal prediction providing exact 90% and 95% confidence intervals.

---

## 2. Directory Structure

```text
pathdsp-piu/
├── configs/
│   ├── base.yaml               # Training & evaluation hyperparameters
│   └── model.yaml              # Model architecture parameters
├── data/
│   ├── raw/input.txt           # Primary GDSC dataset (Zenodo 7532963)
│   ├── processed/              # Cached NPZ matrices and pathway catalog
│   ├── manifests/              # Cell line, SMILES, and pathway mappings
│   └── splits/                 # Disjoint splits (Random, LODO, LOCO, Double-Cold-Start)
├── src/
│   ├── data.py                 # Leakage-safe preprocessor & PyTorch DataLoaders
│   ├── features.py             # Feature registry & 196-pathway alignment
│   ├── model.py                # PathDSP-PIU modular architecture
│   ├── baseline.py             # PathDSP FNN, XGBoost, ElasticNet, Mean Predictors
│   ├── splits.py               # Disjoint split generator with leakage assertions
│   ├── train.py                # Training engine with early stopping & checkpointing
│   ├── evaluate.py             # Evaluation metrics & paired bootstrap tests
│   ├── uncertainty.py          # Deep ensemble & conformal prediction calibrator
│   └── explain.py              # Drug-pathway attribution extraction
├── scripts/
│   ├── prepare.py              # Data caching & split generation
│   ├── run.py                  # CLI experiment orchestrator
│   └── report.py               # Automated publication figure & table generator
├── tests/
│   ├── test_data.py            # Unit tests for preprocessing & datasets
│   ├── test_model.py           # Unit tests for model forward & backward passes
│   └── test_splits.py          # Unit tests for split leakage prevention
├── results/
│   ├── metrics/                # JSON metrics for baselines, proposed, ablations
│   ├── predictions/            # Out-of-fold and ensemble test predictions
│   ├── figures/                # 12 Publication-quality figures (300 DPI)
│   └── reports/                # Formatted markdown tables for manuscript
├── checkpoints/                # Model weights (.pt)
├── docs/                       # Comprehensive research paper documentation
│   ├── methodology.md          # Theoretical equations & architecture
│   ├── data.md                 # Dataset provenance & feature schema
│   ├── experiments.md          # Experimental protocols & split designs
│   ├── results.md              # Full empirical results & findings
│   ├── limitations.md          # Scientific caveats & assay boundaries
│   └── references.md           # Bibliography
├── requirements.txt
└── README.md
```

---

## 3. Quick Start & CLI Usage

### System Audit & Dependency Check
```bash
python scripts/run.py --check
```

### Inspect Dataset Characteristics
```bash
python scripts/run.py --audit
```

### Prepare Dataset & Generate Splits
```bash
python scripts/prepare.py
```

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Run Lightweight Smoke Test
```bash
python scripts/run.py --mode smoke
```

### Train Baseline Suite (FNN, XGBoost, ElasticNet, Mean Baselines)
```bash
python scripts/run.py --mode baseline
```

### Train PathDSP-PIU 3-Seed Ensemble
```bash
python scripts/run.py --mode train
```

### Run Generalization & Ablation Evaluation
```bash
python scripts/run.py --mode evaluate
```

### Full End-to-End Experiment Pipeline
```bash
python scripts/run.py --mode full --device cuda
```

### Generate Publication Figures and Tables
```bash
python scripts/report.py
```

---

## 4. Hardware and Computational Efficiency

The framework is optimized for local GPU acceleration on an **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)**:
- Batch size: 128 (configurable)
- Peak GPU VRAM: < 1.5 GB
- Epoch training duration: ~1.2 seconds on RTX 4060
- Total full experiment pipeline runtime: ~12 minutes
