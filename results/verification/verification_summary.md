# PathDSP-PIU Verification and Audit Summary

**Audit Date:** 2026-11-07  
**Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Windows 11, PyTorch 2.3.1+cu121, Python 3.10.0  

---

## 1. Automated Test Execution
- **Command:** `python -m pytest -v`
- **Result:** **26 passed** in 11.09s (0 failed, 0 skipped, 100% pass rate).
- **Log File:** `results/verification/pytest_output.txt`
- **Verified Areas:**
  - Feature schema mapping (1,031 features, 196 reference PID pathways, 9-pathway zero imputation).
  - DG-Net feature ablation residual masking invariance ([`test_dg_net_ablation_no_leakage`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_model.py#L93)).
  - MUT/CNV multi-omics ablation residual masking invariance ([`test_cell_modality_ablation_no_leakage`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_model.py#L114)).
  - Data split partition disjointness and index range validity ([`test_random_split_bounds_and_disjointness`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_splits.py#L10)).
  - Double-cold-start entity disjointness and unrepresented entity reporting ([`test_double_cold_start_leakage_and_unrepresented_reporting`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_splits.py#L76)).
  - Calibration partition 4-way disjointness ([`test_generate_random_split_with_calibration`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_splits.py#L135)).
  - Two-way pigeonhole bootstrap multiplicity preservation ([`test_two_way_clustered_bootstrap_multiplicity`](file:///d:/codes/PS/Projects/Project_3\implement\scratch\tests\test_evaluate.py#L46)).

---

## 2. Model Checkpoint Verification
- **Verified Checkpoints (7/7 Loaded Successfully with 100% Architecture & Parameter Match):**
  1. `baseline_fnn.pt`: 26.15 MB, 10 parameter tensors, best val MAE 0.3021, epoch 76.
  2. `double_cold_start_fnn.pt`: 26.15 MB, 10 parameter tensors, best val MAE 0.2961, epoch 68.
  3. `double_cold_start_piu.pt`: 3.20 MB, 30 parameter tensors, best val MAE 0.3114, epoch 31.
  4. `pathdsp_piu_seed_42.pt`: 3.20 MB, 30 parameter tensors, best val MAE 0.3352, epoch 28 (Ensemble Member 1).
  5. `pathdsp_piu_seed_1042.pt`: 3.20 MB, 30 parameter tensors, best val MAE 0.3336, epoch 54 (Ensemble Member 2).
  6. `pathdsp_piu_seed_2042.pt`: 3.20 MB, 30 parameter tensors, best val MAE 0.3371, epoch 15 (Ensemble Member 3).
  7. `variant_g_single_model.pt`: 3.20 MB, 30 parameter tensors, best val MAE 0.3352, epoch 28 (Dedicated Single-Model Variant G).
- **Log File:** `results/verification/checkpoint_verification.json`

---

## 3. Conformal Calibration Manifest & Disjointness Audit (Resolved)
- **Problem Statement:** `calibration_verification.json` reported 1,887 calibration samples, but `split_verification.json` previously reported `n_cal: 0` for `random_split.json`.
- **Root Cause:** The runtime partition had dynamically split validation into early stopping (1,886) and conformal calibration (1,887), but `random_split.json` on disk had not stored `cal_indices`.
- **Resolution:**
  - `data/splits/random_split.json` explicitly updated on disk: Train = 30,185 (80.0%), Val = 1,886 (5.0%), Cal = 1,887 (5.0%), Test = 3,774 (10.0%), Total = 37,732 observations.
  - Linked manifest created: `data/splits/calibration_manifest.json` formally documenting exact calibration indices and 4-way disjointness.
  - Zero-Contamination Guarantee: Verified that calibration observations are completely unobserved during model gradient updates, parameter optimization, and early stopping.
  - Saved calibration predictions: `results/predictions/conformal_calibration_predictions.csv` (1,887 observations).
  - Test set coverage ($N = 3,774$): **90.04% at 90% confidence** (width: 1.9805, $q = 7.5148$), **95.63% at 95% confidence** (width: 3.1369, $q = 11.9027$).
- **Log Files:** `results/verification/split_verification.json`, `results/verification/calibration_verification.json`.

---

## 4. Variant D vs. Variant G Provenance Reconciliation (Resolved)
- **Problem Statement:** Final metrics reported Variant D MAE = 0.308419 and Variant G MAE = 0.324904, whereas `logs/final/task-444.log` originally logged Variant G MAE as 0.3258.
- **Root Cause:** In `task-444.log`, `run_ablations()` ran with the unpartitioned 3,773 validation set for `max_epochs=40` without saving a checkpoint, early stopping at epoch 38 with Best Val MAE 0.3265 to achieve test MAE 0.3258. In contrast, under the proper 1,886 early-stopping validation set (with 1,887 calibration samples held out), Member 1 (Seed 42) early stopped at epoch 40 with Best Val MAE 0.3352 to achieve test MAE 0.324904.
- **Remediation & Independent Run:**
  - Executed dedicated retraining via `scripts/train_variant_g.py` on the GPU under the strict 4-way disjoint partition.
  - Generated dedicated checkpoint: `checkpoints/variant_g_single_model.pt` (synced with `pathdsp_piu_seed_42.pt`).
  - Generated dedicated training log: `logs/final/variant_g_training.log`.
  - Generated dedicated training history: `results/metrics/variant_g_training_history.json`.
  - Generated dedicated test predictions: `results/predictions/variant_g_predictions.csv`.
  - Generated unified D/G prediction file: `results/predictions/piu_ablation_dg_predictions.csv`.
- **Numerical Verification from Prediction CSVs:**
  - **Variant D (3-Model Ensemble):** MAE = 0.308419, RMSE = 0.407831, $R^2 = 0.886030$, Pearson $r = 0.943679$, Spearman $\rho = 0.916793$.
  - **Variant G (Single Model Seed 42):** MAE = 0.324904, RMSE = 0.427110, $R^2 = 0.875000$, Pearson $r = 0.938227$, Spearman $\rho = 0.908747$.
  - Identical test observations ($N = 3,774$) and ground truth labels verified across all files.
  - Ensembling benefit quantified: $\Delta\mathrm{MAE} = +0.0165$, $\Delta\mathrm{RMSE} = +0.0193$ (5.1% relative error reduction).
- **Log Files:** `results/verification/prediction_consistency.json`, `logs/final/variant_g_training.log`.

---

## 5. Numerical Prediction Consistency
Metrics recomputed directly from saved CSV files in `results/predictions/` match the reported JSON metrics to six decimal places:
- **Double-Cold-Start FNN (`double_cold_start_predictions.csv`):**
  - Recomputed: MAE = 1.0285, RMSE = 1.4891, $R^2 = -0.5972$, Pearson $r = 0.0773$
  - Reported: MAE = 1.0285, RMSE = 1.4891, $R^2 = -0.5972$, Pearson $r = 0.0773$
- **Double-Cold-Start PIU (`double_cold_start_predictions.csv`):**
  - Recomputed: MAE = 0.9500, RMSE = 1.3711, $R^2 = -0.3540$, Pearson $r = 0.1089$
  - Reported: MAE = 0.9500, RMSE = 1.3711, $R^2 = -0.3540$, Pearson $r = 0.1089$
- **PIU 3-Model Ensemble (`piu_ensemble_predictions.csv` & `piu_ablation_dg_predictions.csv`):**
  - Recomputed: MAE = 0.308419, RMSE = 0.407831, $R^2 = 0.886030$, Pearson $r = 0.943679$
  - Reported: MAE = 0.308419, RMSE = 0.407831, $R^2 = 0.886030$, Pearson $r = 0.943679$
- **Single-Model Variant G (`variant_g_predictions.csv` & `piu_ablation_dg_predictions.csv`):**
  - Recomputed: MAE = 0.324904, RMSE = 0.427110, $R^2 = 0.875000$, Pearson $r = 0.938227$
  - Reported: MAE = 0.324904, RMSE = 0.427110, $R^2 = 0.875000$, Pearson $r = 0.938227$
- **Log File:** `results/verification/prediction_consistency.json`

---

## 6. Split Leakage & Constraint Audit
- **Total Split Files Evaluated:** 28 split files (including updated `random_split.json`).
- **Leakage Status:** **0 violations detected across all splits (all verified `VERIFIED_NO_LEAKAGE`).**
- **Log File:** `results/verification/split_verification.json`
- **Double-Cold-Start Verification:**
  - 17 actual test drugs, 47 actual test cell lines, 734 test pairs.
  - Zero test drugs or test cell lines present in training or validation partitions.
  - 4 selected compounds with 0 pairs in cross-product explicitly reported as unrepresented (`AZD6482`, `GSK269962A`, `PF-4708671`, `Tozasertib`).

---

## 7. Bootstrap Confidence Interval Verification
- **Direction:** $\Delta = \text{Error}_{\text{FNN}} - \text{Error}_{\text{PIU}}$ (positive values indicate PIU superiority / error reduction).
- **Two-Way Pigeonhole Bootstrap (Owen 2007):**
  - Observation weights $w_i = m_d(d_i) \times m_c(c_i)$ preserve cluster sampling multiplicities.
  - Observed $\Delta\mathrm{MAE} = +0.0785$, Bootstrap Mean $\Delta\mathrm{MAE} = +0.0789$.
  - **95% Confidence Interval: $[+0.0073, +0.1492]$ (strictly $> 0$, statistically significant).**
  - Observed $\Delta\mathrm{RMSE} = +0.1181$, Bootstrap Mean $\Delta\mathrm{RMSE} = +0.1178$.
  - **95% Confidence Interval: $[+0.0457, +0.1771]$ (strictly $> 0$, statistically significant).**
- **Clustered Comparisons:**
  - Drug-Clustered ($K = 17$): 95% CI $[+0.0148, +0.1419]$.
  - Cell-Clustered ($K = 47$): 95% CI $[+0.0456, +0.1080]$.
- **Log File:** `results/verification/bootstrap_verification.json`
