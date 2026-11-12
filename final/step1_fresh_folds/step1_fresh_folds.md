# Step 1 — the recipe on fresh folds (seed 43)

The recipe (C·D×3·E·F·G·Z) was chosen while looking at the test folds of seed 42. Here it is tested on new folds (seed 43) that played no part in choosing it. Units: −log10 IC50; mixed 10-fold.

| Run | RMSE | MAE | R² | PCC | Per-drug PCC | Within-drug AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Paper exact (plain) | 0.4015 ± 0.0057 | 0.2961 | 0.889 | 0.943 | 0.753 | 0.865 |
| Paper exact + C·D×3·E·F·G·Z | 0.3828 ± 0.0042 | 0.2829 | 0.899 | 0.948 | 0.766 | 0.873 |

**Paired test (recipe − plain, per fold):** mean ΔRMSE -0.0187, recipe better on 10 of 10 folds; sign p = 0.002, Wilcoxon p = 0.002, paired t p = 6.0e-09.

**Recipe without the ensemble (seed 1 of 3) vs plain:** mean ΔRMSE -0.0044, better on 9 of 10 folds; Wilcoxon p = 0.010, t p = 0.003.

**Same comparison on the original seed-42 folds:** mean ΔRMSE -0.0150, recipe better on 10 of 10 folds (Wilcoxon p = 0.002).

**Verdict:** CONFIRMED — the recipe beats plain Paper exact on fresh folds (≥ 8 of 10 folds, Wilcoxon p < 0.05).
