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