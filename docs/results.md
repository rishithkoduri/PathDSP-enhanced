# Experimental Results & Manuscript Findings

## 1. Dataset Overview
# Table 1: Dataset Characteristics (GDSC Primary Benchmark)

| Characteristic | Measured Value | Unit / Format | Description |
|---|---|---|---|
| **Response Observations (Pairs)** | **37,732** | Pairs | Screened drug–cell line pairs |
| **Unique Compounds / Drugs** | **144** | Chemical Entities | Audited FDA-approved & investigational compounds |
| **Unique Cancer Cell Lines** | **319** | Cell Lines | Sanger / COSMIC SIDM models |
| **Chemical Fingerprint (CHEM)** | **256** | Binary Bits | Radius-2 Morgan fingerprints |
| **Drug-Gene Pathways (DG-Net)** | **196** | Continuous Scores | NetPEA target-network enrichment |
| **Expression Pathways (EXP)** | **187** | Continuous Scores | ssGSEA enrichment (aligned to 196 reference pathways) |
| **Mutation Pathways (MUT-Net)** | **196** | Continuous Scores | NetPEA somatic mutation network enrichment |
| **Copy Number Pathways (CNV-Net)** | **196** | Continuous Scores | NetPEA copy number variation enrichment |
| **Total Flat Features** | **1,031** | Features | Combined multi-omics representation |
| **Target Variable (`resp`)** | **$-\log_10(\mathrm{IC50})$** | $\mu\mathrm{M}$ scale | Mean: -1.0622, Std: 1.2038, Range: [-4.07, 3.74] |
| **Missing Values / NaNs** | **0** | Counts | 100% complete feature and response matrix |
| **Matrix Density** | **82.14%** | Percentage | 37732 / (144 × 319) |


## 2. Benchmark Comparison on In-Distribution Random Split
# Table 2: Random-Split Performance Comparison (In-Distribution Reference)

| Model | MAE (lower is better) | RMSE (lower is better) | R2 (higher is better) | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| DrugMean | 0.4999 | 0.6590 | 0.7024 | 0.8382 | 0.7648 |
| CellMean | 0.8759 | 1.1454 | 0.1010 | 0.3199 | 0.3571 |
| AdditiveMean | 0.3877 | 0.5209 | 0.8141 | 0.9023 | 0.8670 |
| ElasticNet | 0.4125 | 0.5476 | 0.7945 | 0.8924 | 0.8510 |
| RandomForest | 0.3938 | 0.5265 | 0.8100 | 0.9003 | 0.8612 |
| XGBoost | 0.3620 | 0.4834 | 0.8399 | 0.9173 | 0.8829 |
| PathDSP_FNN | **0.3028** | **0.4056** | **0.8873** | 0.9421 | 0.9122 |
| PathDSP-PIU (Proposed Ensemble) | 0.3084 | 0.4078 | 0.8860 | **0.9437** | **0.9168** |

*Note: Bold denotes the best performance per metric. On in-distribution random splits where both drugs and cell lines are observed in training, the baseline PathDSP FNN achieves slightly lower MAE/RMSE, while the proposed PathDSP-PIU ensemble achieves higher correlation metrics.*


## 3. Generalization Performance across Cold-Start Regimes
# Table 3: Rigorous Generalization Evaluation across Cold-Start Regimes

| Regime | Model | MAE (Mean +/- Std) | RMSE (Mean +/- Std) | R2 | Pearson r | Spearman rho | Evaluated Entities | Notes |
|---|---|---|---|---|---|---|---|---|
| Leave-One-Drug-Out (LODO) | PathDSP FNN | 1.2428 +/- 0.8519 | 1.3844 +/- 0.8311 | - | - | - | 10 Held-out Drugs | Prioritized 10-drug subset |
| Leave-One-Drug-Out (LODO) | **PathDSP-PIU (Proposed)** | **1.1347 +/- 0.7633** | **1.2506 +/- 0.7319** | - | - | - | 10 Held-out Drugs | Lower error on 10 held-out drugs |
| Leave-One-Cell-Out (LOCO) | PathDSP FNN | **0.5837 +/- 0.1806** | 0.7536 +/- 0.1941 | - | - | - | 10 Cancer Cell Lines | Prioritized 10-cell line subset |
| Leave-One-Cell-Out (LOCO) | PathDSP-PIU (Proposed) | 0.5905 +/- 0.2070 | **0.7491 +/- 0.2155** | - | - | - | 10 Cancer Cell Lines | Competitive performance across lineages |
| Double-Cold-Start (Pooled) | PathDSP FNN | 1.0285 | 1.4891 | -0.5972 | 0.0773 | 0.2186 | 734 pairs (17 drugs x 47 cells) | Unseen drug AND cell line (negative R2) |
| Double-Cold-Start (Pooled) | **PathDSP-PIU (Proposed)** | **0.9500** | **1.3711** | **-0.3540** | **0.1089** | **0.2333** | 734 pairs (17 drugs x 47 cells) | Delta MAE: +0.0785 (relative improvement) |
| Double-Cold-Start (Macro-Drug) | PathDSP FNN | 1.0123 +/- 0.8994 | 1.1683 +/- 0.8796 | - | 0.3106 | 0.2931 | 17 test drugs | Macro-averaged by drug |
| Double-Cold-Start (Macro-Drug) | **PathDSP-PIU** | **0.9335 +/- 0.8235** | **1.0766 +/- 0.8061** | - | **0.3308** | **0.3229** | 17 test drugs | Macro-averaged by drug |
| Double-Cold-Start (Macro-Cell) | PathDSP FNN | 1.0375 +/- 0.1896 | 1.4918 +/- 0.1943 | - | -0.0196 | **0.1755** | 47 test cells | Macro-averaged by cell line |
| Double-Cold-Start (Macro-Cell) | **PathDSP-PIU** | **0.9540 +/- 0.1879** | **1.3649 +/- 0.1987** | - | **0.0058** | 0.1607 | 47 test cells | Macro-averaged by cell line |

### Double-Cold-Start Bootstrap Significance Analysis

| Resampling Method | Resampling Unit | Observed Delta MAE | Bootstrap Mean Delta MAE | 95% Confidence Interval | Statistically Significant |
|---|---|---|---|---|---|
| Observation-level Bootstrap | Observation pair ($N=734$) | 0.0785 | 0.0788 | [0.0466, 0.1115] | Yes (CI strictly > 0) |
| Drug-Clustered Bootstrap | Unique drug ($K=17$) | 0.0785 | 0.0785 | [0.0148, 0.1419] | Yes (CI strictly > 0) |
| Cell-Clustered Bootstrap | Unique cell line ($K=47$) | 0.0785 | 0.0784 | [0.0456, 0.1080] | Yes (CI strictly > 0) |
| Two-Way Clustered Bootstrap | Drug x Cell cross-product | 0.0785 | 0.0789 | [0.0073, 0.1492] | Yes (CI strictly > 0) |

*Note: Positive Delta MAE indicates PathDSP-PIU has lower error than PathDSP FNN. In double cold start, both models exhibit negative R2 (-0.5972 for FNN, -0.3540 for PIU), reflecting the extreme difficulty of unseen drug/cell extrapolation without anchor entities, although PathDSP-PIU significantly reduces error relative to FNN.*


## 4. Architectural Ablation Analysis
# Table 4: Architectural Ablation Study (Variants A through G)

| Variant | Architectural Configuration | MAE | RMSE | R2 | Pearson r | Spearman rho |
|---|---|---|---|---|---|---|
| Variant_A_Original_FNN | Original PathDSP FNN baseline (Tang & Gottlieb, 2021) | 0.3045 | 0.4141 | 0.8825 | 0.9401 | 0.9083 |
| Variant_B_FNN_plus_Gating | FNN + Drug-Conditioned Pathway Gating | 0.3172 | 0.4247 | 0.8764 | 0.9367 | 0.9066 |
| Variant_C_FNN_plus_Interaction | FNN + Multiplicative Pathway Interaction | 0.3172 | 0.4218 | 0.8781 | 0.9389 | 0.9094 |
| Variant_D_Full_PathDSP_PIU | **Full PathDSP-PIU (3-Model Ensemble)** | 0.3084 | 0.4078 | 0.8860 | 0.9437 | 0.9168 |
| Variant_E_No_DGNet | No DG-Net (CHEM + cell pathways only, masked in residual) | 0.3155 | 0.4176 | 0.8805 | 0.9396 | 0.9099 |
| Variant_F_No_MUT_CNV | No MUT/CNV (Expression-only omics, masked in residual) | 0.3224 | 0.4304 | 0.8731 | 0.9348 | 0.9025 |
| Variant_G_Single_Model_No_Ensemble | Single-Model PathDSP-PIU (Independent Seed 42) | 0.3249 | 0.4271 | 0.8750 | 0.9382 | 0.9087 |

*Ensemble Contribution: Variant G (Single Model) vs. Variant D (3-Model Ensemble) yields Delta MAE = +0.0165 and Delta RMSE = +0.0193, demonstrating a 5.1% error reduction from ensembling on identical splits and features.*


## 5. Predictive Uncertainty Quantification
# Table 5: Uncertainty Quantification and Conformal Prediction Calibration

| Metric | Target Level | Measured Value | Standard / Note |
|---|---|---|---|
| **Ensemble Model Members** | - | **3 Models** | Independently initialized seeds (42, 1042, 2042) |
| **90% Conformal Empirical Coverage** | 90.00% | **90.04%** | Gap: +0.04% |
| **90% Mean Prediction Interval Width** | - | **1.9805** | Log IC50 scale |
| **95% Conformal Empirical Coverage** | 95.00% | **95.63%** | Gap: +0.63% |
| **95% Mean Prediction Interval Width** | - | **3.1369** | Log IC50 scale |
| **Error–Uncertainty Spearman Correlation** | Positive | **0.1287** | Rank correlation between ensemble std and absolute error |
| **Error–Uncertainty Pearson Correlation** | Positive | **0.1733** | Linear correlation between ensemble std and absolute error |


## 6. External Cross-Dataset Validation
# Table 6: Cross-Dataset Transfer Protocol & Assay Harmonization Audit (GDSC to CCLE)

| Parameter | Primary Training Dataset (GDSC2) | External Transfer Target (CCLE) | Harmonization Specification & Impact |
|---|---|---|---|
| **Data Release / Source** | Sanger Cell Model Passports (GDSC2) | Broad Institute DepMap / CCLE (20Q1) | Matched via Sanger SIDM and Broad DepMap IDs |
| **Overlapping Compounds** | 144 compounds (curated Zenodo 7532963) | 24 common compounds (6 directly shared) | Matched via canonical InChIKey and PubChem CID |
| **Overlapping Cell Lines** | 319 cancer cell lines | 478 cancer cell lines (35 shared lines) | Matched via COSMIC SIDM identifiers |
| **Assay Platform** | 72-hour Syto60 / Resazurin viability | 84-hour CellTiter-Glo ATP luminescent | Screening assay kinetics and ATP vs DNA dye differences |
| **Dose Titration** | Extended 9-point dilution series | Capped maximum concentration (8.0 μM) | Upper concentration ceiling causes artificial distribution shift |
| **Target Metric** | Continuous $-\log_{10}(\mathrm{IC50})$ [$\mu$M] | $-\log_{10}(\mathrm{IC50})$ truncated at 8.0 μM | Ceiling truncation restricts dynamic range in CCLE |
| **Gene Expression (EXP)** | ssGSEA on 196 PID pathways | ssGSEA on 196 PID pathways | Identical PID pathway projection pipeline |
| **Mutation (MUT-Net)** | NetPEA on STRING v11 PPI network | NetPEA on STRING v11 PPI network | Network propagation enrichment harmonized |
| **Copy Number (CNV-Net)** | NetPEA on STRING v11 PPI network | NetPEA on STRING v11 PPI network | Network propagation enrichment harmonized |
| **External Validation Status** | Benchmarked & Verified | **Planned Future Work** | Full external deployment requires batch-effect adjustment |


## 7. Biological Pathway Explainability Case Study
In line with Tang & Gottlieb (2021), PathDSP-PIU identifies key interacting cancer signaling pathways:
- **BET Inhibitor Case Study (RVX-208 / Apabetalone)**:
  - Top identified interaction: **PID_ALK1_PATHWAY** (Activin receptor-like kinase 1) and **PID_BMP_PATHWAY**.
  - ALK1 mediates TGF-beta superfamily signaling crucial in tumor angiogenesis and is known to be suppressed by BET inhibition through MAPK downregulation.
- **Top Global Modulators**:
  - **PID_HDAC_III_PATHWAY** (Histone Deacetylase III / Sirtuins)
  - **PID_NFKB_CANONICAL_PATHWAY**
  - **PID_FANCONI_PATHWAY**
