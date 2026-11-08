import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import compute_metrics
from src.model import PathDSP_PIU
from src.baseline import PathDSP_FNN

results = {}

print("=== CHECKPOINT VERIFICATION ===")
chk_files = [f for f in sorted(os.listdir("checkpoints")) if f.endswith(".pt")]
for f in chk_files:
    p = os.path.join("checkpoints", f)
    size_mb = os.path.getsize(p) / (1024 * 1024)
    chk = torch.load(p, map_location="cpu")
    epoch = chk.get("epoch", "N/A")
    val_mae = chk.get("best_val_mae", "N/A")
    state_keys = list(chk.get("model_state_dict", {}).keys())
    
    # Model architecture match
    if "fnn" in f:
        model = PathDSP_FNN()
    else:
        model = PathDSP_PIU()
    try:
        model.load_state_dict(chk["model_state_dict"])
        load_status = "SUCCESS"
    except Exception as e:
        load_status = f"FAILED: {e}"

    info = {
        "file": f,
        "size_mb": round(size_mb, 2),
        "keys": list(chk.keys()),
        "epoch": epoch,
        "best_val_mae": round(val_mae, 4) if isinstance(val_mae, (float, np.floating)) else val_mae,
        "n_params": len(state_keys),
        "load_status": load_status
    }
    results[f] = info
    print(f"[{load_status}] {f}: size={size_mb:.2f}MB, epoch={epoch}, val_mae={info['best_val_mae']}, params={len(state_keys)}")

print("\n=== PREDICTION VERIFICATION ===")
pred_files = [f for f in sorted(os.listdir("results/predictions")) if f.endswith(".csv")]
pred_results = {}
for f in pred_files:
    p = os.path.join("results/predictions", f)
    df = pd.read_csv(p)
    cols = list(df.columns)
    print(f"{f}: rows={len(df)}, cols={cols}")
    pred_results[f] = {
        "rows": len(df),
        "columns": cols
    }

print("\n=== PREDICTION NUMERICAL CONSISTENCY CHECK ===")
# 1. Check double_cold_start_predictions.csv
dcs_df = pd.read_csv("results/predictions/double_cold_start_predictions.csv")
fnn_m = compute_metrics(dcs_df["y_true"], dcs_df["y_pred_fnn"])
piu_m = compute_metrics(dcs_df["y_true"], dcs_df["y_pred_piu"])
print(f"DCS FNN metrics from CSV: MAE={fnn_m['mae']:.4f}, RMSE={fnn_m['rmse']:.4f}, R2={fnn_m['r2']:.4f}")
print(f"DCS PIU metrics from CSV: MAE={piu_m['mae']:.4f}, RMSE={piu_m['rmse']:.4f}, R2={piu_m['r2']:.4f}")

# 2. Check piu_ensemble_predictions.csv (Variant D)
ens_df = pd.read_csv("results/predictions/piu_ensemble_predictions.csv")
ens_m = compute_metrics(ens_df["y_true"], ens_df["y_pred_mean"])
print(f"Variant D (PIU Ensemble) from piu_ensemble_predictions.csv: MAE={ens_m['mae']:.6f}, RMSE={ens_m['rmse']:.6f}, R2={ens_m['r2']:.6f}")

# 3. Check piu_ablation_dg_predictions.csv (Variant D and Variant G)
dg_df = pd.read_csv("results/predictions/piu_ablation_dg_predictions.csv")
dg_ens_m = compute_metrics(dg_df["y_true"], dg_df["pred_ensemble_mean"])
dg_g_m = compute_metrics(dg_df["y_true"], dg_df["pred_seed_42"])
print(f"Variant D from piu_ablation_dg_predictions.csv: MAE={dg_ens_m['mae']:.6f}, RMSE={dg_ens_m['rmse']:.6f}, R2={dg_ens_m['r2']:.6f}")
print(f"Variant G (Single Model Seed 42) from piu_ablation_dg_predictions.csv: MAE={dg_g_m['mae']:.6f}, RMSE={dg_g_m['rmse']:.6f}, R2={dg_g_m['r2']:.6f}")

# 4. Check variant_g_predictions.csv
vg_df = pd.read_csv("results/predictions/variant_g_predictions.csv")
vg_m = compute_metrics(vg_df["y_true"], vg_df["y_pred"])
print(f"Variant G from variant_g_predictions.csv: MAE={vg_m['mae']:.6f}, RMSE={vg_m['rmse']:.6f}, R2={vg_m['r2']:.6f}")

# 5. Check conformal_calibration_predictions.csv
cal_df = pd.read_csv("results/predictions/conformal_calibration_predictions.csv")
cal_ens_m = compute_metrics(cal_df["y_true"], cal_df["pred_ensemble_mean"])
print(f"Conformal Calibration set (n={len(cal_df)}): MAE={cal_ens_m['mae']:.6f}, RMSE={cal_ens_m['rmse']:.6f}")

# 6. Check baseline_predictions.csv
base_df = pd.read_csv("results/predictions/baseline_predictions.csv")
base_metrics = {}
for col in base_df.columns:
    if col != "y_true":
        bm = compute_metrics(base_df["y_true"], base_df[col])
        base_metrics[col] = bm
        print(f"Baseline {col:15s} from CSV: MAE={bm['mae']:.4f}, RMSE={bm['rmse']:.4f}, R2={bm['r2']:.4f}")

# Verify exact equivalence
assert np.isclose(ens_m["mae"], dg_ens_m["mae"], atol=1e-5), "Variant D MAE mismatch between prediction CSVs!"
assert np.isclose(vg_m["mae"], dg_g_m["mae"], atol=1e-5), "Variant G MAE mismatch between prediction CSVs!"

consistency = {
    "variant_d_ensemble": {
        "piu_ensemble_predictions_csv": ens_m,
        "piu_ablation_dg_predictions_csv": dg_ens_m,
        "reconciled_mae": float(ens_m["mae"]),
        "reconciled_rmse": float(ens_m["rmse"]),
        "status": "VERIFIED_EXACT_MATCH"
    },
    "variant_g_single_model": {
        "variant_g_predictions_csv": vg_m,
        "piu_ablation_dg_predictions_csv": dg_g_m,
        "reconciled_mae": float(vg_m["mae"]),
        "reconciled_rmse": float(vg_m["rmse"]),
        "status": "VERIFIED_EXACT_MATCH"
    },
    "double_cold_start": {
        "fnn": fnn_m,
        "piu": piu_m
    },
    "conformal_calibration": {
        "n_samples": len(cal_df),
        "cal_metrics": cal_ens_m
    },
    "baselines": base_metrics
}

os.makedirs("results/verification", exist_ok=True)
with open("results/verification/prediction_consistency.json", "w") as fp:
    json.dump(consistency, fp, indent=2)

with open("results/verification/checkpoint_verification.json", "w") as fp:
    json.dump({"checkpoints": results, "predictions": pred_results}, fp, indent=2)

print("\nSuccessfully saved verified checkpoint and prediction consistency evidence.")
