# Step 4 — unseen cell lines and unseen drugs (grouped 10-fold, seed 42)

Plain Paper exact vs the recipe, on the same folds. Leave-drug-out uses the recipe without the drug-wise z target (C·D×3·E·F·G): a held-out drug has no training responses to z-score with. RMSE paper style = mean ± sd over the held-out cell lines / drugs; pooled metrics over all held-out pairs. Units: −log10 IC50.

## Unseen cell lines (LOCO)

| Model | RMSE (paper style) | RMSE pooled | MAE | R² | PCC | Per-drug PCC | Within-drug AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper exact (plain) | 0.5852 ± 0.1724 | 0.6087 | 0.4510 | 0.744 | 0.869 | 0.380 | 0.702 |
| Paper exact + recipe | 0.5612 ± 0.1661 | 0.5841 | 0.4353 | 0.765 | 0.876 | 0.431 | 0.720 |
| Baseline: drug's mean over training cell lines | 0.6299 ± 0.1991 | 0.6573 | 0.4989 | 0.702 | 0.838 | n/a | n/a |
| Baseline: global training mean | 1.1986 ± 0.1369 | 1.2043 | 0.9230 | -7.7e-04 | n/a | n/a | n/a |

**Recipe vs plain (per fold):** mean ΔRMSE -0.0243, recipe better on 10 of 10 folds; Wilcoxon p = 0.002, t p = 3.3e-05.

- **Paper exact (plain) vs the drug's mean over training cell lines:** better on 185 of 319 held-out cell lines (mean per-cell line ΔRMSE -0.0447, Wilcoxon p = 1.4e-05).
- **Paper exact + recipe vs the drug's mean over training cell lines:** better on 202 of 319 held-out cell lines (mean per-cell line ΔRMSE -0.0687, Wilcoxon p = 1.4e-11).

## Unseen drugs (LODO)

| Model | RMSE (paper style) | RMSE pooled | MAE | R² | PCC | Per-drug PCC | Within-drug AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper exact (plain) | 0.9822 ± 0.5637 | 1.1436 | 0.8608 | 0.097 | 0.460 | 0.629 | 0.796 |
| Paper exact + recipe | 0.9867 ± 0.5849 | 1.1600 | 0.8785 | 0.071 | 0.421 | 0.641 | 0.804 |
| Baseline: cell line's mean over training drugs | 0.9502 ± 0.5937 | 1.1354 | 0.8693 | 0.110 | 0.333 | 0.655 | 0.812 |
| Baseline: global training mean | 1.0525 ± 0.5733 | 1.2119 | 0.9290 | -0.013 | n/a | n/a | n/a |

**Recipe vs plain (per fold):** mean ΔRMSE 0.0110, recipe better on 3 of 10 folds; Wilcoxon p = 0.492, t p = 0.411.

- **Paper exact (plain) vs the cell line's mean over training drugs:** better on 62 of 144 held-out drugs (mean per-drug ΔRMSE 0.0319, Wilcoxon p = 0.097).
- **Paper exact + recipe vs the cell line's mean over training drugs:** better on 66 of 144 held-out drugs (mean per-drug ΔRMSE 0.0365, Wilcoxon p = 0.157).
