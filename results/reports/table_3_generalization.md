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
