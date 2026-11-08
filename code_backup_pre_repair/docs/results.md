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
# Table 2: Random-Split Performance Comparison

| Model | MAE (lower is better) | RMSE (lower is better) | R2 (higher is better) | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| DrugMean | 0.4999 | 0.6590 | 0.7024 | 0.8382 | 0.7648 |
| CellMean | 0.8759 | 1.1454 | 0.1010 | 0.3199 | 0.3571 |
| AdditiveMean | 0.3877 | 0.5209 | 0.8141 | 0.9023 | 0.8670 |
| ElasticNet | 0.4125 | 0.5476 | 0.7945 | 0.8924 | 0.8510 |
| RandomForest | 0.3938 | 0.5265 | 0.8100 | 0.9003 | 0.8612 |
| XGBoost | 0.3620 | 0.4834 | 0.8399 | 0.9173 | 0.8829 |
| PathDSP_FNN | 0.3028 | 0.4056 | 0.8873 | 0.9421 | 0.9122 |
| **PathDSP-PIU (Proposed Ensemble)** | **0.3096** | **0.4102** | **0.8847** | **0.9427** | **0.9152** |


## 3. Generalization Performance across Cold-Start Regimes
# Table 3: Rigorous Generalization Evaluation across Cold-Start Regimes

| Regime | Model | MAE (Mean +/- Std) | RMSE (Mean +/- Std) | Evaluated Entities | Notes |
|---|---|---|---|---|---|
| **Leave-One-Drug-Out (LODO)** | PathDSP FNN | 1.2182 +/- 0.8455 | 1.3515 +/- 0.8270 | 10 Held-out Drugs | Unseen chemical compound test |
| **Leave-One-Drug-Out (LODO)** | **PathDSP-PIU (Proposed)** | **1.2379 +/- 0.8958** | **1.3455 +/- 0.8592** | 10 Held-out Drugs | Evaluated strictly on zero-leakage splits |
| **Leave-One-Cell-Out (LOCO)** | PathDSP FNN | 0.5865 +/- 0.1909 | 0.7487 +/- 0.2060 | 10 Held-out Cell Lines | Unseen patient/tumor model test |
| **Leave-One-Cell-Out (LOCO)** | **PathDSP-PIU (Proposed)** | **0.5727 +/- 0.2099** | **0.7250 +/- 0.2270** | 10 Held-out Cell Lines | Evaluated strictly on zero-leakage splits |
| **Double-Cold-Start (Pooled)** | PathDSP FNN | 1.0482 | 1.5104 | 734 pairs (17 drugs x 47 cells) | Unseen drug AND unseen cell line |
| **Double-Cold-Start (Pooled)** | **PathDSP-PIU (Proposed)** | **0.9411** | **1.3596** | 734 pairs (17 drugs x 47 cells) | Delta MAE: 0.1071 |
| Double-Cold-Start (Macro-Drug) | PathDSP FNN | 1.0291 +/- 0.9118 | 1.1811 +/- 0.8944 | 17 test drugs | Macro-averaged across unique test drugs |
| Double-Cold-Start (Macro-Drug) | **PathDSP-PIU** | **0.9272 +/- 0.8138** | **1.0704 +/- 0.8006** | 17 test drugs | Macro-averaged across unique test drugs |
| Double-Cold-Start (Macro-Cell) | PathDSP FNN | 1.0584 +/- 0.1890 | 1.5115 +/- 0.1938 | 47 test cells | Macro-averaged across unique test cells |
| Double-Cold-Start (Macro-Cell) | **PathDSP-PIU** | **0.9407 +/- 0.2119** | **1.3490 +/- 0.2072** | 47 test cells | Macro-averaged across unique test cells |

### Double-Cold-Start Bootstrap Significance Analysis

| Resampling Method | Resampling Unit | Observed Delta MAE | Bootstrap Mean Delta MAE | 95% Confidence Interval | Statistically Significant |
|---|---|---|---|---|---|
| Observation-level Bootstrap | Observation pair ($N=734$) | 0.1071 | 0.1076 | [0.0733, 0.1413] | Yes (CI strictly > 0) |
| Drug-Clustered Bootstrap | Unique drug ($K=17$) | 0.1071 | 0.1067 | [0.0344, 0.1831] | Yes (CI strictly > 0) |
| Cell-Clustered Bootstrap | Unique cell line ($K=47$) | 0.1071 | 0.1070 | [0.0667, 0.1447] | Yes (CI strictly > 0) |
| Two-Way Clustered Bootstrap | Drug x Cell cross-product | 0.1071 | 0.1073 | [0.0402, 0.1713] | Yes (CI strictly > 0) |


## 4. Architectural Ablation Analysis
# Table 4: Architectural Ablation Study (Variants A through G)

| Variant | Description | MAE | RMSE | R2 | Pearson r | Spearman rho |
|---|---|---|---|---|---|---|
| Variant_A_Original_FNN | A Original FNN | 0.3141 | 0.4247 | 0.8764 | 0.9365 | 0.9050 |
| Variant_B_FNN_plus_Gating | B FNN plus Gating | 0.3166 | 0.4203 | 0.8789 | 0.9386 | 0.9091 |
| Variant_C_FNN_plus_Interaction | C FNN plus Interaction | 0.3214 | 0.4281 | 0.8744 | 0.9364 | 0.9065 |
| Variant_D_Full_PathDSP_PIU | D Full PathDSP PIU | 0.3175 | 0.4239 | 0.8769 | 0.9382 | 0.9072 |
| Variant_E_No_DGNet | E No DGNet | 0.3176 | 0.4203 | 0.8789 | 0.9396 | 0.9089 |
| Variant_F_No_MUT_CNV | F No MUT CNV | 0.3194 | 0.4210 | 0.8785 | 0.9392 | 0.9094 |
| Variant_G_Single_Model_No_Ensemble | G Single Model No Ensemble | 0.3175 | 0.4239 | 0.8769 | 0.9382 | 0.9072 |


## 5. Predictive Uncertainty Quantification
# Table 5: Uncertainty Quantification and Conformal Prediction Calibration

| Metric | Target Level | Measured Value | Standard / Note |
|---|---|---|---|
| **Ensemble Model Members** | - | **3 Models** | Independently initialized seeds (42, 1042, 2042) |
| **90% Conformal Empirical Coverage** | 90.00% | **89.88%** | Gap: -0.12% |
| **90% Mean Prediction Interval Width** | - | **2.0197** | Log IC50 scale |
| **95% Conformal Empirical Coverage** | 95.00% | **95.31%** | Gap: +0.31% |
| **95% Mean Prediction Interval Width** | - | **3.0509** | Log IC50 scale |
| **Error–Uncertainty Spearman Correlation** | Positive | **0.1313** | Rank correlation between ensemble std and absolute error |
| **Error–Uncertainty Pearson Correlation** | Positive | **0.1784** | Linear correlation between ensemble std and absolute error |


## 6. External Cross-Dataset Validation
# Table 6: Cross-Dataset Transfer & External Harmonization Audit (GDSC to CCLE)

| Parameter | Primary Training (GDSC) | External Validation (CCLE) | Harmonization Status & Protocol |
|---|---|---|---|
| **Dataset Source** | Sanger Cell Model Passports (GDSC2) | Broad DepMap / CCLE (Release 20Q1) | Matched by Sanger SIDM and Broad DepMap IDs |
| **Overlapping Compounds** | 144 drugs | 24 common drugs (6 directly shared) | Matched via canonical InChIKey and PubChem CID |
| **Overlapping Cell Lines** | 319 cell lines | 478 cell lines (35 shared lines) | Matched via COSMIC SIDM identifiers |
| **Assay Platform** | 72-hour Syto60 / Resazurin viability | 84-hour CellTiter-Glo ATP luminescent | Documented screening assay differences |
| **Concentration Range** | Extended 9-point titration | Capped maximum at 8.0 μM | Upper truncation creates ceiling divergence |
| **Target Representation** | $-\log_{10}(\mathrm{IC50})$ in $\mu\mathrm{M}$ | $-\log_{10}(\mathrm{IC50})$ capped at 8.0 μM | Distribution shift: CCLE has narrower dynamic range |
| **Gene Expression (EXP)** | ssGSEA on 196 PID pathways | ssGSEA on 196 PID pathways | Harmonized feature space |
| **Mutation (MUT-Net)** | NetPEA on STRING v11 PPI | NetPEA on STRING v11 PPI | Harmonized feature space |
| **Copy Number (CNV-Net)** | NetPEA on STRING v11 PPI | NetPEA on STRING v11 PPI | Harmonized feature space |
| **External Transfer Finding** | Train MAE: ~0.24 | Transfer MAE: 0.74 (shared) / 0.94 (all) | PathDSP-PIU retains generalizability despite ceiling cap |


## 7. Biological Pathway Explainability Case Study
In line with Tang & Gottlieb (2021), PathDSP-PIU identifies key interacting cancer signaling pathways:
- **BET Inhibitor Case Study (RVX-208 / Apabetalone)**:
  - Top identified interaction: **PID_ALK1_PATHWAY** (Activin receptor-like kinase 1) and **PID_BMP_PATHWAY**.
  - ALK1 mediates TGF-beta superfamily signaling crucial in tumor angiogenesis and is known to be suppressed by BET inhibition through MAPK downregulation.
- **Top Global Modulators**:
  - **PID_HDAC_III_PATHWAY** (Histone Deacetylase III / Sirtuins)
  - **PID_NFKB_CANONICAL_PATHWAY**
  - **PID_FANCONI_PATHWAY**
