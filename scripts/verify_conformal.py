import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset, DataPreprocessor, DrugResponseDataset
from torch.utils.data import DataLoader
from src.splits import load_split
from src.model import PathDSP_PIU
from src.uncertainty import PathDSPEnsemble, ConformalCalibrator

data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
split = load_split("data/splits/random_split.json")

tr_idx = np.array(split["train_indices"])
va_idx = np.array(split["val_indices"])
ca_idx = np.array(split["cal_indices"])
te_idx = np.array(split["test_indices"])

assert len(ca_idx) == 1887, f"Expected 1887 calibration indices, got {len(ca_idx)}"
assert len(va_idx) == 1886, f"Expected 1886 validation indices, got {len(va_idx)}"
assert len(te_idx) == 3774, f"Expected 3774 test indices, got {len(te_idx)}"
assert len(tr_idx) == 30185, f"Expected 30185 training indices, got {len(tr_idx)}"

# Verify pairwise disjointness
s_tr, s_va, s_ca, s_te = set(tr_idx), set(va_idx), set(ca_idx), set(te_idx)
assert len(s_tr & s_va) == 0, "Train-Val overlap!"
assert len(s_tr & s_ca) == 0, "Train-Cal overlap!"
assert len(s_tr & s_te) == 0, "Train-Test overlap!"
assert len(s_va & s_ca) == 0, "Val-Cal overlap!"
assert len(s_va & s_te) == 0, "Val-Test overlap!"
assert len(s_ca & s_te) == 0, "Cal-Test overlap!"
assert len(s_tr | s_va | s_ca | s_te) == 37732, "Incomplete coverage!"

preprocessor = DataPreprocessor().fit(data, tr_idx)
ca_set = preprocessor.transform(data, ca_idx)
te_set = preprocessor.transform(data, te_idx)

ca_ds = DrugResponseDataset(
    chem=ca_set["chem"], dg=ca_set["dg"], cell_pathways=ca_set["cell_pathways"],
    flat=ca_set["flat_features"], targets=ca_set["target"], drugs=ca_set["drugs"], cells=ca_set["cells"]
)
te_ds = DrugResponseDataset(
    chem=te_set["chem"], dg=te_set["dg"], cell_pathways=te_set["cell_pathways"],
    flat=te_set["flat_features"], targets=te_set["target"], drugs=te_set["drugs"], cells=te_set["cells"]
)
ca_ld = DataLoader(ca_ds, batch_size=128, shuffle=False)
te_ld = DataLoader(te_ds, batch_size=128, shuffle=False)

# Load ensemble models
models = []
for s in [42, 1042, 2042]:
    m = PathDSP_PIU()
    chk = torch.load(f"checkpoints/pathdsp_piu_seed_{s}.pt", map_location="cpu")
    m.load_state_dict(chk["model_state_dict"])
    m.eval()
    models.append(m)

ensemble = PathDSPEnsemble(models)
cal_res = ensemble.predict(ca_ld, device=torch.device("cpu"))
test_res = ensemble.predict(te_ld, device=torch.device("cpu"))

# Non-conformity scores on calibration set
cal_scores = np.abs(cal_res["targets"] - cal_res["mean"]) / np.maximum(cal_res["std"], 1e-6)

# Save calibration predictions CSV
os.makedirs("results/predictions", exist_ok=True)
cal_pred_df = pd.DataFrame({
    "drug": cal_res["drugs"],
    "cell": cal_res["cells"],
    "y_true": cal_res["targets"],
    "pred_seed_42": cal_res["individual_predictions"][:, 0],
    "pred_seed_1042": cal_res["individual_predictions"][:, 1],
    "pred_seed_2042": cal_res["individual_predictions"][:, 2],
    "pred_ensemble_mean": cal_res["mean"],
    "ensemble_std": cal_res["std"],
    "nonconformity_score": cal_scores
})
cal_pred_csv_path = "results/predictions/conformal_calibration_predictions.csv"
cal_pred_df.to_csv(cal_pred_csv_path, index=False)
print(f"Saved calibration predictions to {cal_pred_csv_path}")

calibrator = ConformalCalibrator(adaptive=True)
calibrator.calibrate(cal_res["targets"], cal_res["mean"], cal_res["std"], alphas=[0.10, 0.05])
cov_90 = calibrator.evaluate_coverage(test_res["targets"], test_res["mean"], test_res["std"], alpha=0.10)
cov_95 = calibrator.evaluate_coverage(test_res["targets"], test_res["mean"], test_res["std"], alpha=0.05)

print("\n=== CONFORMAL CALIBRATION VERIFICATION ===")
print(f"Calibration samples (n): {len(cal_res['targets'])}")
print(f"Test samples (n):        {len(test_res['targets'])}")
print(f"90% Target Coverage:     {cov_90['empirical_coverage']*100:.2f}% (Mean Interval Width: {cov_90['mean_interval_width']:.4f})")
print(f"95% Target Coverage:     {cov_95['empirical_coverage']*100:.2f}% (Mean Interval Width: {cov_95['mean_interval_width']:.4f})")
print(f"Quantile q_0.90:         {calibrator.quantiles[0.10]:.4f}")
print(f"Quantile q_0.95:         {calibrator.quantiles[0.05]:.4f}")

cal_verif = {
    "protocol": "Strict split conformal prediction with held-out calibration partition",
    "calibration_manifest": "data/splits/calibration_manifest.json",
    "random_split_file": "data/splits/random_split.json",
    "calibration_predictions_csv": cal_pred_csv_path,
    "n_train_samples": int(len(tr_idx)),
    "n_val_samples": int(len(va_idx)),
    "n_calibration_samples": int(len(cal_res["targets"])),
    "n_test_samples": int(len(test_res["targets"])),
    "disjointness_verified": True,
    "zero_contamination_verified": True,
    "quantile_formula": "np.quantile(scores, min(1.0, ceil((n+1)*(1-alpha))/n), method='higher')",
    "conformal_90": {
        "target": 0.90,
        "empirical_coverage": float(cov_90["empirical_coverage"]),
        "mean_interval_width": float(cov_90["mean_interval_width"]),
        "coverage_gap": float(cov_90["coverage_gap"]),
        "calibrated_quantile": float(calibrator.quantiles[0.10])
    },
    "conformal_95": {
        "target": 0.95,
        "empirical_coverage": float(cov_95["empirical_coverage"]),
        "mean_interval_width": float(cov_95["mean_interval_width"]),
        "coverage_gap": float(cov_95["coverage_gap"]),
        "calibrated_quantile": float(calibrator.quantiles[0.05])
    }
}

os.makedirs("results/verification", exist_ok=True)
with open("results/verification/calibration_verification.json", "w") as fp:
    json.dump(cal_verif, fp, indent=2)
print("Saved updated calibration verification to results/verification/calibration_verification.json")
