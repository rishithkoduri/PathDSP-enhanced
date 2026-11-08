# PathDSP-PIU Repository Repair Log

**Document Version:** 1.0.0  
**Timestamp:** 2026-11-07  
**Project:** Drug-Conditioned Pathway Interaction Learning with Uncertainty-Aware Evaluation for Generalizable Cancer Drug Sensitivity Prediction (**PathDSP-PIU**)  
**Development Platform:** Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), PyTorch 2.3.1+cu121, Python 3.10.0  

---

## 1. Summary of Repository Audit

An exhaustive audit of the PathDSP-PIU codebase, configurations, experiment outputs, and documentation was conducted. The baseline repository established the fundamental architecture extending Tang & Gottlieb (2021) *Scientific Reports* ([10.1038/s41598-021-82612-7](https://doi.org/10.1038/s41598-021-82612-7)). However, several critical methodological flaws, data leakage pathways, and unverified claims were identified that compromised scientific rigor.

Prior to modifying any code or results, a complete archive backup of all initial experiment results, checkpoints, scripts, and documentation was created in:
- `results/archive_pre_repair/`
- `checkpoints/archive_pre_repair/`
- `code_backup_pre_repair/`

---

## 2. Identified Critical Issues and Technical Remedies

### Issue 1: Residual Pathway Feature Leakage in Ablations (Phase 2.1)
- **Location:** `src/model.py` (`PathDSP_PIU.forward()`, lines 224-276).
- **Severity:** High (Methodological Invalidity of Ablations).
- **Root Cause:** The model accepted structured inputs (`chem`, `dg`, `cell_pathways`) and a flat feature vector `flat` (1031-dim). In `forward()`, while `dg` was zeroed out when `use_dg_net=False`, and `cell_pathways` slices were masked when `active_modalities != ("exp", "mut", "cnv")`, the 1031-dim vector passed to `self.residual_proj(flat)` was **never masked**. Consequently, ablated modalities (DG-Net columns 256:452, MUT-Net columns 639:835, CNV-Net columns 835:1031) leaked into the regression head via the residual path.
- **Remedy:** In `PathDSP_PIU.forward()`, clone `flat` and explicitly zero out the corresponding column ranges whenever modalities are ablated. Verify via unit tests that outputs are invariant and gradients are strictly zero with respect to ablated slices.
- **Impact / Retraining Needed:** Variants E (No DG-Net) and F (No MUT/CNV) must be completely retrained from scratch; prior results are invalidated and discarded.

### Issue 2: Copied / Fabricated Variant G Single-Model Result (Phase 2.2)
- **Location:** `scripts/run.py` (`run_ablations()`, line 616).
- **Severity:** High (Scientific Misrepresentation).
- **Root Cause:** In `run_ablations()`, `Variant_G_Single_Model_No_Ensemble` was created by calling `copy.deepcopy(ablation_metrics["Variant_D_Full_PathDSP_PIU"])` rather than training an independent single model.
- **Remedy:** Implement genuine independent single-model training. Instantiate a fresh `PathDSP_PIU` model with an independent seed, train it under the exact same protocol as the ensemble members, evaluate its test predictions independently, and compute metrics.
- **Impact / Retraining Needed:** Variant G must be trained independently and reported with genuine empirical metrics.

### Issue 3: Set-Collapsed Two-Way Clustered Bootstrap (Phase 2.3)
- **Location:** `src/evaluate.py` (`compute_two_way_clustered_bootstrap()`, lines 218-225).
- **Severity:** High (Statistical Bias in Cold-Start Evaluation).
- **Root Cause:** When resampling drugs and cell lines with replacement, the function converted sampled cluster arrays into Python `set()` objects (`set(rng.choice(unique_drugs, ...))`). This discarded sampling multiplicities (e.g. if a drug was resampled 3 times, its count became 1), invalidating the empirical likelihood / cluster bootstrap distribution.
- **Remedy:** Replace set-filtering with the statistically sound Two-Way Pigeonhole Bootstrap (Owen 2007). Calculate cluster draw frequencies $m_d = \text{bincount}(\text{boot\_drugs})$ and $m_c = \text{bincount}(\text{boot\_cells})$. Assign observation weights $w_i = m_d(d_i) \times m_c(c_i)$ and compute weighted MAE and RMSE across resampled replicates.
- **Impact / Recomputation Needed:** Recompute all Double-Cold-Start bootstrap statistics and confidence intervals.

### Issue 4: Dual-Use of Validation Set for Conformal Calibration (Phase 2.4)
- **Location:** `scripts/run.py` (`train_piu_ensemble()`, lines 285-289) and `src/uncertainty.py`.
- **Severity:** Medium-High (Optimization Bias / Calibration Exchangeability Violation).
- **Root Cause:** The validation set was used for early stopping model checkpoint selection, and then the same validation set predictions were fed into `ConformalCalibrator.calibrate()`. Because checkpoints were selected to minimize validation loss, validation residuals were artificially small, leading to under-conservative nonconformity quantiles and coverage degradation.
- **Remedy:** Implement a strictly partitioned train / val / cal / test scheme (e.g., 70% train, 10% validation for early stopping, 10% independent calibration for conformal prediction, 10% test). Calibrate strictly on `cal_loader`. Update `ConformalCalibrator.calibrate()` to use the exact finite-sample quantile formula:
  $$q_{1-\alpha} = \text{np.quantile}\left(\text{scores}, \min\left(1.0, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right), \text{method='higher'}\right)$$
- **Impact / Retraining Needed:** Train on train split, stop early on val split, calibrate on cal split, and evaluate coverage on test split.

### Issue 5: Dataset Count Discrepancy & Unverified CCLE Transfer Claims (Phase 1 & Phase 7)
- **Location:** `docs/data.md`, `scripts/report.py` (Table 6), and `results/reports/table_6_cross_dataset.md`.
- **Severity:** Medium (Documentation Ambiguity & Unverified Generalization Claim).
- **Root Cause:** Zenodo record 7532963 contains 144 drugs and 319 cell lines (37,732 rows) after applying GDSC quality filters, while the text of Tang & Gottlieb (2021) mentions 153 drugs. Furthermore, Table 6 displayed hardcoded transfer MAE numbers from the paper rather than results from an actual executed CCLE pipeline.
- **Remedy:**
  1. Clearly document in `docs/data.md` and `docs/repair_log.md` the exact 144-drug vs 153-drug distinction (144 compounds have complete Morgan fingerprints and NetPEA target enrichment scores in the official Zenodo release).
  2. Remove numerical claims from Table 6. Designate CCLE external validation as "Planned Future Work" and provide a rigorous protocol audit detailing screening assay differences (72h Syto60 vs 84h CellTiter-Glo ATP luminescence, 8.0 $\mu$M concentration capping, and distribution shifts).

---

## 3. Centralized Reproducibility Protocol (Phase 3.1)

To guarantee absolute experimental reproducibility across the hardware environment:
- Create `src/utils.py` implementing `set_seed(seed: int, deterministic: bool = True)`.
- Enforce seeds for Python `random`, NumPy `np.random`, and PyTorch:
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- Save exact split manifests (seed, observation indices, entity lists) for all experiments.

---

## 4. Retraining and Execution Roadmap

| Step | Script / Command | Target Output | Status |
|---|---|---|---|
| 1. Unit Tests | `pytest tests/` | Verify feature masking & bootstrap math | Pending |
| 2. Smoke Test | `python scripts/run.py --mode smoke` | Pipeline smoke test on RTX 4060 | Pending |
| 3. Full Retraining | `python scripts/run.py --mode all` | Baselines, Ensemble, DCS, Ablations A-G | Pending |
| 4. Reports & Figures | `python scripts/report.py` | 12 figures, 6 tables, updated docs | Pending |
