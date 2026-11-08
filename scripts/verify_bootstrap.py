import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import (
    compute_observation_paired_bootstrap,
    compute_clustered_bootstrap,
    compute_two_way_clustered_bootstrap
)

print("=== TASK 6: BOOTSTRAP CONFIDENCE INTERVAL VERIFICATION ===")

# 1. Synthetic test verifying that multiplicity weighting affects estimates
rng = np.random.RandomState(42)
synth_drugs = np.array(["D1", "D1", "D2", "D2", "D3", "D3"])
synth_cells = np.array(["C1", "C2", "C1", "C2", "C1", "C2"])
synth_y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
synth_y_base = synth_y_true + 1.0  # MAE = 1.0
synth_y_prop = synth_y_true + 0.2  # MAE = 0.2, Delta = 0.8

synth_res = compute_two_way_clustered_bootstrap(
    synth_y_true, synth_y_base, synth_y_prop, synth_drugs, synth_cells, n_boot=500, seed=42
)
print("Synthetic Multiplicity Verification:")
print(f"  Observed Delta MAE: {synth_res['observed_delta_mae']:.4f}")
print(f"  Bootstrap Mean Delta MAE: {synth_res['bootstrap_mean_delta_mae']:.4f}")
print(f"  95% CI: [{synth_res['delta_mae_ci_lower']:.4f}, {synth_res['delta_mae_ci_upper']:.4f}]")
assert synth_res["delta_mae_ci_lower"] > 0, "Synthetic CI lower bound should be strictly positive!"

# 2. Recompute Double-Cold-Start Confidence Intervals from saved predictions
dcs_csv = "results/predictions/double_cold_start_predictions.csv"
dcs_df = pd.read_csv(dcs_csv)

y_true = dcs_df["y_true"].values
pred_fnn = dcs_df["y_pred_fnn"].values
pred_piu = dcs_df["y_pred_piu"].values
drugs = dcs_df["drug"].values
cells = dcs_df["cell"].values

print("\nDouble-Cold-Start Bootstrap Recalculation:")
print(f"  Test observations: {len(y_true)}")
print(f"  Unique drugs:      {len(np.unique(drugs))}")
print(f"  Unique cells:      {len(np.unique(cells))}")

# Direction: Delta = Base (FNN) - Prop (PIU) -> Positive means PIU is better (lower error)
obs_boot = compute_observation_paired_bootstrap(y_true, pred_fnn, pred_piu, n_boot=1000, seed=42)
drug_boot = compute_clustered_bootstrap(y_true, pred_fnn, pred_piu, clusters=drugs, cluster_level="drug", n_boot=1000, seed=42)
cell_boot = compute_clustered_bootstrap(y_true, pred_fnn, pred_piu, clusters=cells, cluster_level="cell", n_boot=1000, seed=42)
tw_boot = compute_two_way_clustered_bootstrap(y_true, pred_fnn, pred_piu, drugs=drugs, cells=cells, n_boot=1000, seed=42)

print(f"\n1. Observation-level Bootstrap: Delta MAE = {obs_boot['observed_delta_mae']:.4f}, 95% CI = [{obs_boot['delta_mae_ci_lower']:.4f}, {obs_boot['delta_mae_ci_upper']:.4f}]")
print(f"2. Drug-Clustered Bootstrap:     Delta MAE = {drug_boot['observed_delta_mae']:.4f}, 95% CI = [{drug_boot['delta_mae_ci_lower']:.4f}, {drug_boot['delta_mae_ci_upper']:.4f}]")
print(f"3. Cell-Clustered Bootstrap:     Delta MAE = {cell_boot['observed_delta_mae']:.4f}, 95% CI = [{cell_boot['delta_mae_ci_lower']:.4f}, {cell_boot['delta_mae_ci_upper']:.4f}]")
print(f"4. Two-Way Clustered Bootstrap:  Delta MAE = {tw_boot['observed_delta_mae']:.4f}, 95% CI = [{tw_boot['delta_mae_ci_lower']:.4f}, {tw_boot['delta_mae_ci_upper']:.4f}]")

boot_verif = {
    "definition": "Delta = Error(FNN) - Error(PIU). Positive delta indicates PIU has lower prediction error.",
    "n_observations": int(len(y_true)),
    "n_drugs": int(len(np.unique(drugs))),
    "n_cells": int(len(np.unique(cells))),
    "observation_paired_bootstrap": obs_boot,
    "drug_clustered_bootstrap": drug_boot,
    "cell_clustered_bootstrap": cell_boot,
    "two_way_clustered_bootstrap": tw_boot
}

os.makedirs("results/verification", exist_ok=True)
with open("results/verification/bootstrap_verification.json", "w") as fp:
    json.dump(boot_verif, fp, indent=2)

print("\nSaved bootstrap verification to results/verification/bootstrap_verification.json")
