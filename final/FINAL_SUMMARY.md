# PathDSP — final validation (v6.0)

Written 2026-11-10 19:59. Runs: 1-plain = 20261109-175325_PAPER_input_1-plain, 1-recipe = 20261109-184612_PAPER_input_1-recipe, 4-plain = 20261109-203618_PAPER_input_4-plain, 4-recipe-LOCO = 20261109-221456_PAPER_input_4-recipe-LOCO, 4-recipe-LODO = 20261109-235843_PAPER_input_4-recipe-LODO, F1-ic50 = 20261110-013147_PAPER_processed_F1-ic50, F2-aac-A = 20261110-020257_GRAPH_csa-ALL_F2-aac-A, F3-aac-B = 20261110-081934_GRAPH_csa-ALL_F3-aac-B

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


# Step 2 — leakage audit

## A. Checks on our pipeline (step 1's recipe run)

| check | how it was verified | result | detail |
| --- | --- | --- | --- |
| Train / inner-validation / test rows are disjoint in every fold | fold jobs rebuilt from config.json (core.engine.make_jobs) | PASS | 10 folds; inner validation ⊂ training rows; test ∩ training = ∅ |
| Every test prediction comes from a model that never saw that pair | preds.csv vs the rebuilt test folds | PASS | duplicates 0; each mixed repeat covers all 37,732 pairs once: True |
| Feature z-scoring is fitted on the fold's training rows only | mean / sd stored with the best fold vs recomputed | PASS | max |μ − μ(train rows)| = 6.1e-08; max |μ − μ(all rows)| = 4.9e-03 |
| Drug-wise z target uses each drug's training responses only | stored drug means / sds vs recomputed | PASS | max |μ − μ(train rows)| = 0.0e+00; vs all rows 1.3e-01 |
| Conformal interval of a fold is calibrated on the other folds only | fold 1's predictions shifted by +5; its interval widths recomputed | PASS | widths unchanged |
| Early stopping and the best epoch use the inner validation rows, never the test rows | code: engine.Pool scores job.va; test rows are predicted once, after training (runner._fold_done); demo L4 shows what test-based stopping would do | PASS | by construction |
| Response labels are not used to build features | the authors' ssGSEA / NetPEA scores and Morgan bits are computed per cell line / per drug without any drug response | PASS | label-free (computed once for all cell lines: transductive, but no labels) |

## B. What each leak would do (same mixed 10-fold split, seed 42; clean vs leaky)

| leak | model | RMSE clean | RMSE leaky | Δ (leaky − clean) | relative | folds leaky looks better | folds | Wilcoxon p | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 · z-scoring fitted on all rows (incl. test) | Ridge (α = 10) | 0.5080 | 0.5080 | 4.6e-09 | +0.0 % | 3 | 10 | 0.625 | our runs fit it per fold |
| L2 · top-200 features chosen on all rows | Ridge (α = 10) | 0.6476 | 0.6479 | 0.0002 | +0.0 % | 4 | 10 | 1.000 | our runs select no features |
| L3 · drug means / sds of the z target from all rows | Ridge (α = 10) | 0.5000 | 0.4991 | -0.0010 | -0.2 % | 10 | 10 | 0.002 | our runs use training rows only |
| L4 · early stopping / best round chosen on the test fold | LightGBM (app defaults, drug-wise z) | 0.4372 | 0.4370 | -0.0002 | -0.0 % | 7 | 10 | 0.492 | best round 1572 (clean) vs 1599 (test-stopped), median |

A negative Δ means the leak makes the model look better than it is. The demos use fast models (Ridge, LightGBM); the size of a leak depends on the model, the direction does not.

## C. Model-selection leak (choosing the recipe on the reported folds)

The recipe was tuned on the seed-42 folds. On fresh seed-43 folds (step 1) it is CONFIRMED: mean ΔRMSE -0.0187, better on 10 of 10 folds (Wilcoxon p = 0.002).

## D. Known limitation (not a leak)

PathDSP's 10 % inner validation set is a random sample of training pairs. In leave-cell-out / leave-drug-out it therefore contains the same cell lines and drugs as training, so early stopping cannot see the unseen-cell / unseen-drug difficulty (v4 repeat 1: validation RMSE at the best epoch 0.380 / 0.367 / 0.366 for mixed / LOCO / LODO, test 0.394 / 0.576 / 0.957). It uses no test information, so it is not leakage, but group-aware inner validation would match the task better.

# Step 3 — do the explanations point to the drug's targets?

Model: step 1's recipe run (20261109-184612_PAPER_input_1-recipe), its best fold: 3,773 held-out pairs, 3 seed(s), SHAP DeepExplainer. A pathway's score for a drug = mean |SHAP| of its EXP, MUT and CNV features over that drug's pairs (cell-line side only; DG is built from the targets, so it is reported only as a circular control). 128 drugs with ≥ 8 pairs; 97 have annotated targets in PID pathways.

## A. Target recovery

| Metric | Model | Random |
| --- | --- | --- |
| AUROC (target pathways vs others) | 0.499 | 0.500 |
| Precision@5 | 0.052 | 0.103 |
| Hit@5 | 0.216 | — |
| Precision@10 | 0.065 | 0.103 |
| Hit@10 | 0.464 | — |
| Precision@20 | 0.072 | 0.103 |
| Hit@20 | 0.680 | — |
| AUROC of the DG attribution (circular control) | 0.630 | 0.500 |

**Permutation control** (target sets shuffled between drugs, 1000×): mean AUROC 0.499 vs 0.498 under the null (p = 0.470); P@10 0.065 vs 0.054 (p = 0.019).

## B. Stability across seeds

Per drug, the pathway scores of two seeds correlate at Spearman 0.735; their top-10 pathways overlap with Jaccard 0.410.

## C. Deletion test (faithfulness)

A pathway is deleted by replacing its EXP / MUT / CNV values with the training mean; the change is measured on the drug's held-out pairs (mean |Δ prediction| in z units, and the within-drug PCC with the measured response).

| Comparison | First | Second | Drugs where first > second | Wilcoxon p |
| --- | --- | --- | --- | --- |
| top-5 vs 5 random pathways: mean |Δ prediction| | 0.0819 | 0.0436 | 128 / 128 | 9.5e-23 |
| top-10 vs 10 random: mean |Δ prediction| | 0.1225 | 0.0663 | 128 / 128 | 9.5e-23 |
| 10 random vs top-10: within-drug PCC after deletion | 0.8009 | 0.7919 | 80 / 128 | 3.3e-05 |
| target pathways vs as many random: mean |Δ prediction| | 0.1106 | 0.1020 | 53 / 97 | 0.003 |

Within-drug PCC before deletion: 0.805.

## D. Top cell-line pathways (all drugs)

| pathway | score |
| --- | --- |
| PID_GLYPICAN_1PATHWAY | 0.03494 |
| PID_ARF_3PATHWAY | 0.03394 |
| PID_MAPK_TRK_PATHWAY | 0.02921 |
| PID_WNT_CANONICAL_PATHWAY | 0.02858 |
| PID_RAS_PATHWAY | 0.02823 |
| PID_P73PATHWAY | 0.02776 |
| PID_TGFBR_PATHWAY | 0.02776 |
| PID_RAC1_REG_PATHWAY | 0.02771 |
| PID_NFAT_TFPATHWAY | 0.02753 |
| PID_S1P_META_PATHWAY | 0.02742 |
| PID_DNA_PK_PATHWAY | 0.02724 |
| PID_RHODOPSIN_PATHWAY | 0.02654 |
| PID_ILK_PATHWAY | 0.02572 |
| PID_ERBB_NETWORK_PATHWAY | 0.02566 |
| PID_CD8_TCR_DOWNSTREAM_PATHWAY | 0.02366 |


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


# Final decision

Step 1: the recipe must beat plain Paper exact on fresh folds on ≥ 8 of 10 folds with Wilcoxon p < 0.05. If it does, the final IC50 model uses it; otherwise plain Paper exact.

- Step 1 confirmed the recipe: **yes**
- Final IC50 model: **Paper exact + C·D×3·E·F·G·Z**, trained on all of processed_data · GDSC
- Final AAC models: Run A recipe and Run B recipe, each trained on all 587,709 CSA pairs


# Train-on-everything models (used by Predict)

- **IC50 · processed_data:** run 20261110-013147_PAPER_processed_F1-ic50, models/20261110-013147_PAPER_processed_F1-ic50.pt (final, 40,365 training pairs, upgrades C·D×3·E·F·G·Z, target drug_z); trained in 0.5 h; CCLE external test RMSE 0.929, PCC 0.776 (2,611 pairs)
- **AAC · CSA all 5 · Run A recipe:** run 20261110-020257_GRAPH_csa-ALL_F2-aac-A, models/20261110-020257_GRAPH_csa-ALL_F2-aac-A.pt (final, 587,709 training pairs, upgrades C·E·F·G, target raw); trained in 6.3 h
- **AAC · CSA all 5 · Run B recipe:** run 20261110-081934_GRAPH_csa-ALL_F3-aac-B, models/20261110-081934_GRAPH_csa-ALL_F3-aac-B.pt (final, 587,709 training pairs, upgrades C·E·F·G·M·R·S, target study_z); trained in 2.5 h
