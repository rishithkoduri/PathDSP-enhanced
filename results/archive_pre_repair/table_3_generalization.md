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
