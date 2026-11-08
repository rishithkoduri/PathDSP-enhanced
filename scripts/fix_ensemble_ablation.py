import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset, DataPreprocessor, create_dataloaders
from src.splits import load_split
from src.model import PathDSP_PIU
from src.evaluate import compute_metrics
from src.uncertainty import PathDSPEnsemble

data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
split = load_split("data/splits/random_split.json")
tr_idx = np.array(split["train_indices"])
va_idx = np.array(split["val_indices"])
te_idx = np.array(split["test_indices"])

preprocessor = DataPreprocessor().fit(data, tr_idx)
tr_set = preprocessor.transform(data, tr_idx)
va_set = preprocessor.transform(data, va_idx)
te_set = preprocessor.transform(data, te_idx)

tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=128)

models = []
seeds = [42, 1042, 2042]
for s in seeds:
    m = PathDSP_PIU()
    chk_path = f"checkpoints/pathdsp_piu_seed_{s}.pt"
    chk = torch.load(chk_path, map_location="cpu")
    m.load_state_dict(chk["model_state_dict"])
    m.eval()
    models.append(m)

ens = PathDSPEnsemble(models)
res = ens.predict(te_ld, device=torch.device("cpu"))

y_true = res["targets"]
preds_single_42 = res["individual_predictions"][:, 0]
preds_single_1042 = res["individual_predictions"][:, 1]
preds_single_2042 = res["individual_predictions"][:, 2]
preds_ens = res["mean"]
ens_std = res["std"]

# Save individual and ensemble predictions
pred_df = pd.DataFrame({
    "drug": res["drugs"],
    "cell": res["cells"],
    "y_true": y_true,
    "pred_seed_42": preds_single_42,
    "pred_seed_1042": preds_single_1042,
    "pred_seed_2042": preds_single_2042,
    "pred_ensemble_mean": preds_ens,
    "ensemble_std": ens_std
})
pred_df.to_csv("results/predictions/piu_ablation_dg_predictions.csv", index=False)

m_single_42 = compute_metrics(y_true, preds_single_42)
m_single_1042 = compute_metrics(y_true, preds_single_1042)
m_single_2042 = compute_metrics(y_true, preds_single_2042)
m_ens = compute_metrics(y_true, preds_ens)

print("=== TASK 3: D/G ENSEMBLE ABLATION RESULTS ===")
print(f"Variant G (Single Model Seed 42):   MAE={m_single_42['mae']:.4f}, RMSE={m_single_42['rmse']:.4f}, R2={m_single_42['r2']:.4f}, PCC={m_single_42['pearson']:.4f}")
print(f"Member 2 (Single Model Seed 1042): MAE={m_single_1042['mae']:.4f}, RMSE={m_single_1042['rmse']:.4f}, R2={m_single_1042['r2']:.4f}, PCC={m_single_1042['pearson']:.4f}")
print(f"Member 3 (Single Model Seed 2042): MAE={m_single_2042['mae']:.4f}, RMSE={m_single_2042['rmse']:.4f}, R2={m_single_2042['r2']:.4f}, PCC={m_single_2042['pearson']:.4f}")
print(f"Variant D (3-Model PIU Ensemble):  MAE={m_ens['mae']:.4f}, RMSE={m_ens['rmse']:.4f}, R2={m_ens['r2']:.4f}, PCC={m_ens['pearson']:.4f}")

delta_mae = m_single_42['mae'] - m_ens['mae']
delta_rmse = m_single_42['rmse'] - m_ens['rmse']
print(f"\nDelta (Single Seed 42 - Ensemble): Delta MAE={delta_mae:+.4f}, Delta RMSE={delta_rmse:+.4f}")

# Update results/metrics/ablation_metrics.json
with open("results/metrics/ablation_metrics.json", "r") as fp:
    abl_data = json.load(fp)

abl_data["Variant_D_Full_PathDSP_PIU"] = m_ens
abl_data["Variant_G_Single_Model_No_Ensemble"] = m_single_42

with open("results/metrics/ablation_metrics.json", "w") as fp:
    json.dump(abl_data, fp, indent=2)

print("Successfully updated ablation_metrics.json with verified D/G comparison.")
