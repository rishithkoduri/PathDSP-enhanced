import os
import sys
import json
import logging
import copy
import yaml
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset, DataPreprocessor, DrugResponseDataset, create_dataloaders
from src.splits import load_split
from src.model import PathDSP_PIU
from src.train import train_model, evaluate_loader
from src.evaluate import compute_metrics
from src.utils import set_seed

def get_device(requested: str = "auto") -> torch.device:
    if requested.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
from src.uncertainty import PathDSPEnsemble

# Configure logger to write to both stdout and log file
log_dir = "logs/final"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "variant_g_training.log")

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Variant_G_Logger")
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(log_file, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Stream handler
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)

def main():
    logger.info("=" * 60)
    logger.info("STARTING DEDICATED TRAINING FOR ABLATION VARIANT G (Single Model No Ensemble)")
    logger.info("=" * 60)

    # 1. Load config and device
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_device("auto")
    logger.info(f"Compute device: {device}")

    # 2. Load dataset and split
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    split = load_split("data/splits/random_split.json")

    tr_idx = np.array(split["train_indices"])
    va_idx = np.array(split["val_indices"])
    ca_idx = np.array(split["cal_indices"])
    te_idx = np.array(split["test_indices"])

    logger.info(f"Dataset partitions: Train={len(tr_idx)}, Val={len(va_idx)}, Cal={len(ca_idx)}, Test={len(te_idx)}")
    assert len(tr_idx) == 30185
    assert len(va_idx) == 1886
    assert len(ca_idx) == 1887
    assert len(te_idx) == 3774

    # 3. Fit train-only preprocessor
    preprocessor = DataPreprocessor().fit(data, tr_idx)
    tr_set = preprocessor.transform(data, tr_idx)
    va_set = preprocessor.transform(data, va_idx)
    te_set = preprocessor.transform(data, te_idx)

    tr_ld, va_ld, te_ld = create_dataloaders(
        tr_set, va_set, te_set,
        batch_size=config["training"]["batch_size"]
    )

    # 4. Initialize model with deterministic seed 42
    seed = 42
    set_seed(seed, deterministic=True)
    model = PathDSP_PIU(use_gating=True, use_interaction=True, use_dg_net=True)

    # 5. Train model with checkpointing
    chk_path = "checkpoints/variant_g_single_model.pt"
    os.makedirs("checkpoints", exist_ok=True)
    logger.info(f"Training Variant G (Seed: {seed}) with checkpoint saving to {chk_path}...")
    
    # Train using standard training config (max_epochs=80, patience=12)
    trained_model, history = train_model(
        model, tr_ld, va_ld, config, device,
        checkpoint_path=chk_path
    )

    # Save training history
    os.makedirs("results/metrics", exist_ok=True)
    hist_path = "results/metrics/variant_g_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {hist_path}")

    # 6. Evaluate on test set
    eval_res = evaluate_loader(trained_model, te_ld, device)
    y_test = te_set["target"]
    y_pred = eval_res["predictions"]

    metrics_g = compute_metrics(y_test, y_pred)
    logger.info("=" * 60)
    logger.info("VARIANT G (Single Model No Ensemble) TEST RESULTS:")
    logger.info(f"  MAE:      {metrics_g['mae']:.6f}")
    logger.info(f"  RMSE:     {metrics_g['rmse']:.6f}")
    logger.info(f"  R2:       {metrics_g['r2']:.6f}")
    logger.info(f"  Pearson:  {metrics_g['pearson']:.6f}")
    logger.info(f"  Spearman: {metrics_g['spearman']:.6f}")
    logger.info("=" * 60)

    # 7. Save standalone Variant G predictions
    os.makedirs("results/predictions", exist_ok=True)
    var_g_pred_df = pd.DataFrame({
        "drug": te_set["drugs"],
        "cell": te_set["cells"],
        "y_true": y_test,
        "y_pred": y_pred,
        "abs_error": np.abs(y_test - y_pred)
    })
    var_g_pred_path = "results/predictions/variant_g_predictions.csv"
    var_g_pred_df.to_csv(var_g_pred_path, index=False)
    logger.info(f"Saved Variant G test predictions to {var_g_pred_path}")

    # 8. Load ensemble members 2 and 3 to re-generate unified piu_ablation_dg_predictions.csv
    logger.info("Loading ensemble members for unified ablation prediction file...")
    m2 = PathDSP_PIU()
    chk2 = torch.load("checkpoints/pathdsp_piu_seed_1042.pt", map_location=device)
    m2.load_state_dict(chk2["model_state_dict"])
    m2.eval()

    m3 = PathDSP_PIU()
    chk3 = torch.load("checkpoints/pathdsp_piu_seed_2042.pt", map_location=device)
    m3.load_state_dict(chk3["model_state_dict"])
    m3.eval()

    trained_model.eval()
    ens = PathDSPEnsemble([trained_model, m2, m3])
    ens_res = ens.predict(te_ld, device=device)

    preds_single_42 = ens_res["individual_predictions"][:, 0]
    preds_single_1042 = ens_res["individual_predictions"][:, 1]
    preds_single_2042 = ens_res["individual_predictions"][:, 2]
    preds_ens_mean = ens_res["mean"]
    ens_std = ens_res["std"]

    # Save unified D/G prediction file
    dg_df = pd.DataFrame({
        "drug": ens_res["drugs"],
        "cell": ens_res["cells"],
        "y_true": ens_res["targets"],
        "pred_seed_42": preds_single_42,
        "pred_seed_1042": preds_single_1042,
        "pred_seed_2042": preds_single_2042,
        "pred_ensemble_mean": preds_ens_mean,
        "ensemble_std": ens_std
    })
    dg_path = "results/predictions/piu_ablation_dg_predictions.csv"
    dg_df.to_csv(dg_path, index=False)
    logger.info(f"Updated unified ablation prediction file at {dg_path}")

    # Also update piu_ensemble_predictions.csv
    ens_df = pd.DataFrame({
        "drug": ens_res["drugs"],
        "cell": ens_res["cells"],
        "y_true": ens_res["targets"],
        "y_pred_mean": preds_ens_mean,
        "ensemble_std": ens_std,
        "abs_error": np.abs(ens_res["targets"] - preds_ens_mean)
    })
    ens_path = "results/predictions/piu_ensemble_predictions.csv"
    ens_df.to_csv(ens_path, index=False)
    logger.info(f"Updated ensemble predictions at {ens_path}")

    # Compute and verify metrics
    metrics_ens = compute_metrics(ens_res["targets"], preds_ens_mean)
    logger.info("VARIANT D (3-Model PIU Ensemble) TEST RESULTS:")
    logger.info(f"  MAE:      {metrics_ens['mae']:.6f}")
    logger.info(f"  RMSE:     {metrics_ens['rmse']:.6f}")
    logger.info(f"  R2:       {metrics_ens['r2']:.6f}")
    logger.info(f"  Pearson:  {metrics_ens['pearson']:.6f}")
    logger.info(f"  Spearman: {metrics_ens['spearman']:.6f}")

    # Update results/metrics/ablation_metrics.json
    with open("results/metrics/ablation_metrics.json", "r") as fp:
        abl_metrics = json.load(fp)

    abl_metrics["Variant_D_Full_PathDSP_PIU"] = metrics_ens
    abl_metrics["Variant_G_Single_Model_No_Ensemble"] = metrics_g

    with open("results/metrics/ablation_metrics.json", "w") as fp:
        json.dump(abl_metrics, fp, indent=2)
    logger.info("Updated results/metrics/ablation_metrics.json with verified Variant D and Variant G metrics.")

    # Copy checkpoint to checkpoints/pathdsp_piu_seed_42.pt to ensure 100% equivalence
    torch.save(torch.load(chk_path, map_location="cpu"), "checkpoints/pathdsp_piu_seed_42.pt")
    logger.info("Synced checkpoints/pathdsp_piu_seed_42.pt with variant_g_single_model.pt.")

if __name__ == "__main__":
    main()
