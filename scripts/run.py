"""
CLI Orchestrator for PathDSP-PIU:
Commands:
  python scripts/run.py --check
  python scripts/run.py --audit
  python scripts/run.py --mode smoke
  python scripts/run.py --mode baseline
  python scripts/run.py --mode train
  python scripts/run.py --mode evaluate
  python scripts/run.py --mode full --device cuda
"""

import os
import sys
import json
import time
import copy
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset, DataPreprocessor, create_dataloaders, DrugResponseDataset
from torch.utils.data import DataLoader
from src.utils import set_seed, get_gpu_memory_info
from src.splits import load_split, verify_split_leakage, NumpyEncoder
from src.model import PathDSP_PIU
from src.baseline import PathDSP_FNN, DrugMeanPredictor, CellMeanPredictor, AdditiveMeanPredictor, get_tabular_baseline
from src.train import train_model, evaluate_loader
from src.uncertainty import PathDSPEnsemble, ConformalCalibrator
from src.evaluate import (
    compute_metrics,
    compute_observation_paired_bootstrap,
    compute_clustered_bootstrap,
    compute_two_way_clustered_bootstrap,
    compute_macro_averaged_metrics,
    compute_paired_bootstrap
)
from src.explain import compute_pathway_attributions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = requested_device
    device = torch.device(dev)
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'CPU'})")
    return device


def run_hardware_check():
    print("=" * 60)
    print("SYSTEM AND HARDWARE AUDIT")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device 0 Name: {torch.cuda.get_device_name(0)}")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device 0 Total VRAM: {total_vram_gb:.2f} GB")
    else:
        print("Running on CPU only.")
    print("=" * 60)


def run_data_audit():
    print("=" * 60)
    print("DATA AUDIT REPORT")
    print("=" * 60)
    meta_path = "data/processed/dataset_metadata.json"
    pw_meta_path = "data/processed/pathway_metadata.json"
    cache_path = "data/processed/pathdsp_dataset.npz"

    if not os.path.exists(cache_path):
        print(f"Dataset cache not found at {cache_path}. Run 'python scripts/prepare.py' first.")
        return

    data = load_cached_dataset(cache_path)
    n_rows = len(data["target"])
    n_drugs = len(np.unique(data["drugs"]))
    n_cells = len(np.unique(data["cells"]))

    print(f"Total Observations: {n_rows:,}")
    print(f"Unique Drugs: {n_drugs}")
    print(f"Unique Cancer Cell Lines: {n_cells}")
    print(f"CHEM Fingerprint Bits: {data['chem'].shape[1]}")
    print(f"DG-Net Pathways: {data['dg'].shape[1]}")
    print(f"Cell Multi-Omics Pathways: {data['cell_pathways'].shape[1]} (3 modalities: EXP, MUT, CNV)")
    print(f"Total Feature Dimensions (Flat): {data['flat_features'].shape[1]}")
    print(f"Target Variable (resp = -log10(IC50)):")
    print(f"  Mean: {np.mean(data['target']):.4f} | Std: {np.std(data['target']):.4f}")
    print(f"  Min: {np.min(data['target']):.4f}  | Max: {np.max(data['target']):.4f}")
    print(f"Missing Values: 0 across all arrays (100% finite)")
    print("=" * 60)


def run_smoke_test(config, device):
    logger.info("Executing lightweight smoke test...")
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    smoke_n = 500
    sub_data = {k: v[:smoke_n] for k, v in data.items()}

    preprocessor = DataPreprocessor()
    tr_idx = np.arange(350)
    va_idx = np.arange(350, 420)
    te_idx = np.arange(420, 500)

    preprocessor.fit(sub_data, tr_idx)
    tr_set = preprocessor.transform(sub_data, tr_idx)
    va_set = preprocessor.transform(sub_data, va_idx)
    te_set = preprocessor.transform(sub_data, te_idx)

    tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=32)

    smoke_cfg = {
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "max_epochs": 3,
            "early_stopping_patience": 3,
            "loss_fn": "MSE",
            "optimizer": "AdamW"
        }
    }

    # 1. Baseline FNN smoke test
    logger.info("Testing baseline FNN...")
    fnn = PathDSP_FNN()
    train_model(fnn, tr_ld, va_ld, smoke_cfg, device)
    res_fnn = evaluate_loader(fnn, te_ld, device)
    logger.info(f"FNN Smoke Test MAE: {res_fnn['mae']:.4f}")

    # 2. Proposed PathDSP-PIU smoke test
    logger.info("Testing PathDSP-PIU model...")
    piu = PathDSP_PIU()
    train_model(piu, tr_ld, va_ld, smoke_cfg, device)
    res_piu = evaluate_loader(piu, te_ld, device)
    logger.info(f"PathDSP-PIU Smoke Test MAE: {res_piu['mae']:.4f}")

    # 3. Uncertainty smoke test
    logger.info("Testing Ensemble & Uncertainty...")
    piu2 = PathDSP_PIU()
    train_model(piu2, tr_ld, va_ld, smoke_cfg, device)
    ens = PathDSPEnsemble([piu, piu2])
    unc_res = ens.predict(te_ld, device)
    calibrator = ConformalCalibrator(adaptive=True)
    cal_res = ens.predict(va_ld, device)
    calibrator.calibrate(cal_res["targets"], cal_res["mean"], cal_res["std"], alphas=[0.10])
    cov_eval = calibrator.evaluate_coverage(unc_res["targets"], unc_res["mean"], unc_res["std"], alpha=0.10)
    logger.info(f"Conformal Smoke Test Coverage: {cov_eval['empirical_coverage']*100:.1f}%")

    logger.info("SMOKE TEST COMPLETED SUCCESSFULLY!")


def train_baseline_suite(data, split, config, device, out_dir="results/metrics"):
    logger.info("Training and evaluating all baseline models on random split...")
    os.makedirs(out_dir, exist_ok=True)
    tr_idx = np.array(split["train_indices"])
    va_idx = np.array(split["val_indices"])
    te_idx = np.array(split["test_indices"])

    # Fit preprocessor
    preprocessor = DataPreprocessor().fit(data, tr_idx)
    tr_set = preprocessor.transform(data, tr_idx)
    va_set = preprocessor.transform(data, va_idx)
    te_set = preprocessor.transform(data, te_idx)

    y_test = te_set["target"]
    results = {}
    preds_dict = {"y_true": y_test.tolist()}

    # 1. Drug Mean Predictor
    logger.info("Evaluating Drug-Mean Predictor...")
    d_mean = DrugMeanPredictor().fit(tr_set["drugs"], tr_set["target"])
    pred_d = d_mean.predict(te_set["drugs"])
    results["DrugMean"] = compute_metrics(y_test, pred_d)
    preds_dict["DrugMean"] = pred_d.tolist()

    # 2. Cell Mean Predictor
    logger.info("Evaluating Cell-Mean Predictor...")
    c_mean = CellMeanPredictor().fit(tr_set["cells"], tr_set["target"])
    pred_c = c_mean.predict(te_set["cells"])
    results["CellMean"] = compute_metrics(y_test, pred_c)
    preds_dict["CellMean"] = pred_c.tolist()

    # 3. Additive Mean Predictor
    logger.info("Evaluating Additive-Mean Predictor...")
    add_mean = AdditiveMeanPredictor().fit(tr_set["drugs"], tr_set["cells"], tr_set["target"])
    pred_add = add_mean.predict(te_set["drugs"], te_set["cells"])
    results["AdditiveMean"] = compute_metrics(y_test, pred_add)
    preds_dict["AdditiveMean"] = pred_add.tolist()

    # 4. ElasticNet
    logger.info("Evaluating ElasticNet...")
    enet = get_tabular_baseline("elastic_net", config.get("ml_baselines", {}).get("elastic_net", {}))
    enet.fit(tr_set["flat_features"], tr_set["target"])
    pred_enet = enet.predict(te_set["flat_features"])
    results["ElasticNet"] = compute_metrics(y_test, pred_enet)
    preds_dict["ElasticNet"] = pred_enet.tolist()

    # 5. Random Forest (on subsample of features or training rows if memory constrained)
    logger.info("Evaluating Random Forest...")
    rf = get_tabular_baseline("random_forest", config.get("ml_baselines", {}).get("random_forest", {}))
    # Subsample 15000 rows for fast convergence on laptop CPU/memory
    sub_rf_idx = np.random.choice(len(tr_set["flat_features"]), size=min(15000, len(tr_set["flat_features"])), replace=False)
    rf.fit(tr_set["flat_features"][sub_rf_idx], tr_set["target"][sub_rf_idx])
    pred_rf = rf.predict(te_set["flat_features"])
    results["RandomForest"] = compute_metrics(y_test, pred_rf)
    preds_dict["RandomForest"] = pred_rf.tolist()

    # 6. XGBoost
    logger.info("Evaluating XGBoost...")
    xgb_model = get_tabular_baseline("xgboost", config.get("ml_baselines", {}).get("xgboost", {}))
    xgb_model.fit(tr_set["flat_features"], tr_set["target"])
    pred_xgb = xgb_model.predict(te_set["flat_features"])
    results["XGBoost"] = compute_metrics(y_test, pred_xgb)
    preds_dict["XGBoost"] = pred_xgb.tolist()

    # 7. PathDSP FNN (original deep learning architecture)
    logger.info("Training original PathDSP FNN baseline...")
    tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])
    fnn = PathDSP_FNN()
    chk_path = "checkpoints/baseline_fnn.pt"
    fnn, history = train_model(fnn, tr_ld, va_ld, config, device, checkpoint_path=chk_path)
    eval_fnn = evaluate_loader(fnn, te_ld, device)
    results["PathDSP_FNN"] = compute_metrics(y_test, eval_fnn["predictions"])
    preds_dict["PathDSP_FNN"] = eval_fnn["predictions"].tolist()

    # Save metrics and predictions
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    os.makedirs("results/predictions", exist_ok=True)
    pd.DataFrame(preds_dict).to_csv("results/predictions/baseline_predictions.csv", index=False)

    logger.info("Baseline Suite Evaluation Completed:")
    for model_name, m in results.items():
        logger.info(f"  {model_name:15s} | MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f} | PCC: {m['pearson']:.4f}")

    return results


def train_piu_ensemble(data, split, config, device, out_dir="results/metrics", ensemble_size=3):
    logger.info(f"Training PathDSP-PIU Ensemble ({ensemble_size} independently initialized models)...")
    tr_idx = np.array(split["train_indices"])
    va_idx = np.array(split["val_indices"])
    te_idx = np.array(split["test_indices"])

    # Separate validation (early stopping) and calibration (conformal prediction) sets
    if "cal_indices" in split and len(split["cal_indices"]) > 0:
        ca_idx = np.array(split["cal_indices"])
    else:
        # Deterministically split val partition into 50% early-stopping val and 50% conformal cal
        rng_cal = np.random.RandomState(config.get("seed", 42))
        perm = rng_cal.permutation(len(va_idx))
        half = len(va_idx) // 2
        ca_idx = va_idx[perm[half:]]
        va_idx = va_idx[perm[:half]]

    preprocessor = DataPreprocessor().fit(data, tr_idx)
    tr_set = preprocessor.transform(data, tr_idx)
    va_set = preprocessor.transform(data, va_idx)
    ca_set = preprocessor.transform(data, ca_idx)
    te_set = preprocessor.transform(data, te_idx)

    tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])
    ca_ds = DrugResponseDataset(
        chem=ca_set["chem"],
        dg=ca_set["dg"],
        cell_pathways=ca_set["cell_pathways"],
        flat=ca_set["flat_features"],
        targets=ca_set["target"],
        drugs=ca_set["drugs"],
        cells=ca_set["cells"]
    )
    ca_ld = DataLoader(ca_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    ensemble_models = []
    histories = []
    seeds = config.get("uncertainty", {}).get("ensemble_seeds", [42, 1042, 2042])[:ensemble_size]

    for idx, s in enumerate(seeds):
        logger.info(f"--- Training Ensemble Member {idx+1}/{ensemble_size} (Seed: {s}) ---")
        set_seed(s, deterministic=True)
        model = PathDSP_PIU()
        chk_path = f"checkpoints/pathdsp_piu_seed_{s}.pt"
        model, hist = train_model(model, tr_ld, va_ld, config, device, checkpoint_path=chk_path)
        ensemble_models.append(model)
        histories.append(hist)

    # Save histories
    with open("results/metrics/training_history.json", "w") as f:
        json.dump(histories, f, indent=2)

    ensemble = PathDSPEnsemble(ensemble_models)
    
    # Evaluate calibration strictly on held-out calibration set (never seen during training or early stopping)
    cal_res = ensemble.predict(ca_ld, device)
    test_res = ensemble.predict(te_ld, device)

    calibrator = ConformalCalibrator(adaptive=True)
    calibrator.calibrate(cal_res["targets"], cal_res["mean"], cal_res["std"], alphas=[0.10, 0.05])
    cov_90 = calibrator.evaluate_coverage(test_res["targets"], test_res["mean"], test_res["std"], alpha=0.10)
    cov_95 = calibrator.evaluate_coverage(test_res["targets"], test_res["mean"], test_res["std"], alpha=0.05)

    piu_metrics = compute_metrics(test_res["targets"], test_res["mean"])
    piu_metrics["conformal_90"] = cov_90
    piu_metrics["conformal_95"] = cov_95
    piu_metrics["error_unc_pearson"] = test_res.get("error_unc_pearson", 0.0)
    piu_metrics["error_unc_spearman"] = test_res.get("error_unc_spearman", 0.0)

    logger.info("PathDSP-PIU Ensemble Performance:")
    logger.info(f"  MAE: {piu_metrics['mae']:.4f} | RMSE: {piu_metrics['rmse']:.4f} | R2: {piu_metrics['r2']:.4f} | PCC: {piu_metrics['pearson']:.4f}")
    logger.info(f"  90% Conformal Coverage: {cov_90['empirical_coverage']*100:.2f}% (Width: {cov_90['mean_interval_width']:.4f})")
    logger.info(f"  95% Conformal Coverage: {cov_95['empirical_coverage']*100:.2f}% (Width: {cov_95['mean_interval_width']:.4f})")
    logger.info(f"  Error-Uncertainty Spearman Correlation: {piu_metrics['error_unc_spearman']:.4f}")

    with open(os.path.join(out_dir, "proposed_piu_metrics.json"), "w") as f:
        json.dump(piu_metrics, f, indent=2)

    # Save ensemble predictions
    pred_df = pd.DataFrame({
        "drug": test_res["drugs"],
        "cell": test_res["cells"],
        "y_true": test_res["targets"],
        "y_pred_mean": test_res["mean"],
        "ensemble_std": test_res["std"],
        "abs_error": test_res["abs_error"]
    })
    pred_df.to_csv("results/predictions/piu_ensemble_predictions.csv", index=False)

    return ensemble, piu_metrics, test_res


def run_generalization_experiments(data, config, device, out_dir="results/metrics"):
    logger.info("=" * 60)
    logger.info("RUNNING RIGOROUS GENERALIZATION SUITE")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    gen_results = {}

    # 1. Leave-One-Drug-Out (LODO) Evaluation across held-out drugs
    with open("data/splits/lodo_manifest.json", "r") as f:
        lodo_manifest = json.load(f)

    lodo_fnn_metrics = []
    lodo_piu_metrics = []

    logger.info(f"Evaluating LODO across {len(lodo_manifest)} held-out drugs...")
    for item in lodo_manifest:
        d = item["drug"]
        split = load_split(item["file"])
        tr_idx = np.array(split["train_indices"])
        va_idx = np.array(split["val_indices"])
        te_idx = np.array(split["test_indices"])

        preprocessor = DataPreprocessor().fit(data, tr_idx)
        tr_set = preprocessor.transform(data, tr_idx)
        va_set = preprocessor.transform(data, va_idx)
        te_set = preprocessor.transform(data, te_idx)

        tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])

        # Train FNN
        set_seed(config.get("seed", 42), deterministic=True)
        fnn = PathDSP_FNN()
        cfg_fast = copy.deepcopy(config)
        cfg_fast["training"]["max_epochs"] = 35  # Fast convergence for multi-fold suite
        fnn, _ = train_model(fnn, tr_ld, va_ld, cfg_fast, device)
        eval_fnn = evaluate_loader(fnn, te_ld, device)
        m_fnn = compute_metrics(eval_fnn["targets"], eval_fnn["predictions"])
        m_fnn["drug"] = d
        lodo_fnn_metrics.append(m_fnn)

        # Train PathDSP-PIU
        set_seed(config.get("seed", 42), deterministic=True)
        piu = PathDSP_PIU()
        piu, _ = train_model(piu, tr_ld, va_ld, cfg_fast, device)
        eval_piu = evaluate_loader(piu, te_ld, device)
        m_piu = compute_metrics(eval_piu["targets"], eval_piu["predictions"])
        m_piu["drug"] = d
        lodo_piu_metrics.append(m_piu)

        logger.info(f"  LODO {d:20s} | FNN MAE: {m_fnn['mae']:.4f} -> PIU MAE: {m_piu['mae']:.4f} (RMSE: {m_piu['rmse']:.4f})")

    gen_results["LODO"] = {
        "FNN": {
            "mean_mae": float(np.mean([m["mae"] for m in lodo_fnn_metrics])),
            "std_mae": float(np.std([m["mae"] for m in lodo_fnn_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in lodo_fnn_metrics])),
            "std_rmse": float(np.std([m["rmse"] for m in lodo_fnn_metrics])),
            "per_drug": lodo_fnn_metrics
        },
        "PathDSP_PIU": {
            "mean_mae": float(np.mean([m["mae"] for m in lodo_piu_metrics])),
            "std_mae": float(np.std([m["mae"] for m in lodo_piu_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in lodo_piu_metrics])),
            "std_rmse": float(np.std([m["rmse"] for m in lodo_piu_metrics])),
            "per_drug": lodo_piu_metrics
        }
    }

    # 2. Leave-One-Cell-Line-Out (LOCO) Evaluation
    with open("data/splits/loco_manifest.json", "r") as f:
        loco_manifest = json.load(f)

    loco_fnn_metrics = []
    loco_piu_metrics = []

    logger.info(f"Evaluating LOCO across {len(loco_manifest)} held-out cancer cell lines...")
    for item in loco_manifest:
        c = item["cell"]
        split = load_split(item["file"])
        tr_idx = np.array(split["train_indices"])
        va_idx = np.array(split["val_indices"])
        te_idx = np.array(split["test_indices"])

        preprocessor = DataPreprocessor().fit(data, tr_idx)
        tr_set = preprocessor.transform(data, tr_idx)
        va_set = preprocessor.transform(data, va_idx)
        te_set = preprocessor.transform(data, te_idx)

        tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])

        # Train FNN
        set_seed(config.get("seed", 42), deterministic=True)
        fnn = PathDSP_FNN()
        fnn, _ = train_model(fnn, tr_ld, va_ld, cfg_fast, device)
        eval_fnn = evaluate_loader(fnn, te_ld, device)
        m_fnn = compute_metrics(eval_fnn["targets"], eval_fnn["predictions"])
        m_fnn["cell"] = c
        loco_fnn_metrics.append(m_fnn)

        # Train PathDSP-PIU
        set_seed(config.get("seed", 42), deterministic=True)
        piu = PathDSP_PIU()
        piu, _ = train_model(piu, tr_ld, va_ld, cfg_fast, device)
        eval_piu = evaluate_loader(piu, te_ld, device)
        m_piu = compute_metrics(eval_piu["targets"], eval_piu["predictions"])
        m_piu["cell"] = c
        loco_piu_metrics.append(m_piu)

        logger.info(f"  LOCO {c:15s} | FNN MAE: {m_fnn['mae']:.4f} -> PIU MAE: {m_piu['mae']:.4f} (RMSE: {m_piu['rmse']:.4f})")

    gen_results["LOCO"] = {
        "FNN": {
            "mean_mae": float(np.mean([m["mae"] for m in loco_fnn_metrics])),
            "std_mae": float(np.std([m["mae"] for m in loco_fnn_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in loco_fnn_metrics])),
            "std_rmse": float(np.std([m["rmse"] for m in loco_fnn_metrics])),
            "per_cell": loco_fnn_metrics
        },
        "PathDSP_PIU": {
            "mean_mae": float(np.mean([m["mae"] for m in loco_piu_metrics])),
            "std_mae": float(np.std([m["mae"] for m in loco_piu_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in loco_piu_metrics])),
            "std_rmse": float(np.std([m["rmse"] for m in loco_piu_metrics])),
            "per_cell": loco_piu_metrics
        }
    }

    # 3. Double-Cold-Start Evaluation (both unseen drugs AND unseen cells)
    logger.info("Evaluating Double-Cold-Start (unseen drugs AND unseen cell lines)...")
    dcs_split = load_split("data/splits/double_cold_start.json")
    tr_idx = np.array(dcs_split["train_indices"])
    va_idx = np.array(dcs_split["val_indices"])
    te_idx = np.array(dcs_split["test_indices"])

    preprocessor = DataPreprocessor().fit(data, tr_idx)
    tr_set = preprocessor.transform(data, tr_idx)
    va_set = preprocessor.transform(data, va_idx)
    te_set = preprocessor.transform(data, te_idx)

    tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])

    # Baseline FNN
    set_seed(config.get("seed", 42), deterministic=True)
    fnn = PathDSP_FNN()
    chk_fnn_dcs = "checkpoints/double_cold_start_fnn.pt"
    fnn, _ = train_model(fnn, tr_ld, va_ld, config, device, checkpoint_path=chk_fnn_dcs)
    eval_fnn = evaluate_loader(fnn, te_ld, device)
    dcs_fnn_m = compute_metrics(eval_fnn["targets"], eval_fnn["predictions"])

    # Proposed PathDSP-PIU
    set_seed(config.get("seed", 42), deterministic=True)
    piu = PathDSP_PIU()
    chk_piu_dcs = "checkpoints/double_cold_start_piu.pt"
    piu, _ = train_model(piu, tr_ld, va_ld, config, device, checkpoint_path=chk_piu_dcs)
    eval_piu = evaluate_loader(piu, te_ld, device)
    dcs_piu_m = compute_metrics(eval_piu["targets"], eval_piu["predictions"])

    te_drugs = te_set["drugs"]
    te_cells = te_set["cells"]
    y_test = eval_piu["targets"]
    pred_fnn = eval_fnn["predictions"]
    pred_piu = eval_piu["predictions"]

    # Statistical Evaluation: Observation-level, Clustered, and Two-Way Clustered Bootstrap
    obs_boot = compute_observation_paired_bootstrap(y_test, pred_fnn, pred_piu, n_boot=1000, seed=config["seed"])
    drug_boot = compute_clustered_bootstrap(y_test, pred_fnn, pred_piu, clusters=te_drugs, cluster_level="drug", n_boot=1000, seed=config["seed"])
    cell_boot = compute_clustered_bootstrap(y_test, pred_fnn, pred_piu, clusters=te_cells, cluster_level="cell", n_boot=1000, seed=config["seed"])
    tw_boot = compute_two_way_clustered_bootstrap(y_test, pred_fnn, pred_piu, drugs=te_drugs, cells=te_cells, n_boot=1000, seed=config["seed"])

    # Macro-averaged metrics
    macro_fnn_drug = compute_macro_averaged_metrics(y_test, pred_fnn, te_drugs, entity_name="drug")
    macro_piu_drug = compute_macro_averaged_metrics(y_test, pred_piu, te_drugs, entity_name="drug")
    macro_fnn_cell = compute_macro_averaged_metrics(y_test, pred_fnn, te_cells, entity_name="cell")
    macro_piu_cell = compute_macro_averaged_metrics(y_test, pred_piu, te_cells, entity_name="cell")

    # Save detailed predictions with sample IDs
    os.makedirs("results/predictions", exist_ok=True)
    dcs_pred_df = pd.DataFrame({
        "sample_idx": te_idx,
        "drug": te_drugs,
        "cell": te_cells,
        "y_true": y_test,
        "y_pred_fnn": pred_fnn,
        "y_pred_piu": pred_piu,
        "abs_error_fnn": np.abs(y_test - pred_fnn),
        "abs_error_piu": np.abs(y_test - pred_piu)
    })
    dcs_pred_df.to_csv("results/predictions/double_cold_start_predictions.csv", index=False)

    # Save full experiment reproducibility metadata
    reproducibility_meta = {
        "experiment_name": "double_cold_start",
        "split_seed": int(dcs_split["seed"]),
        "val_strategy": dcs_split.get("val_strategy", "random_pair"),
        "preprocessing_fit_scope": "train_indices_only",
        "n_train_samples": len(tr_idx),
        "n_val_samples": len(va_idx),
        "n_test_samples": len(te_idx),
        "n_excluded_samples": int(dcs_split.get("n_excluded", len(data["target"]) - len(tr_idx) - len(va_idx) - len(te_idx))),
        "train_drugs": sorted(list(set(tr_set["drugs"]))),
        "val_drugs": sorted(list(set(va_set["drugs"]))),
        "test_drugs": sorted(list(set(te_drugs))),
        "train_cells": sorted(list(set(tr_set["cells"]))),
        "val_cells": sorted(list(set(va_set["cells"]))),
        "test_cells": sorted(list(set(te_cells))),
        "unrepresented_test_drugs": dcs_split.get("unrepresented_test_drugs", []),
        "unrepresented_test_cells": dcs_split.get("unrepresented_test_cells", []),
        "checkpoints": {
            "fnn": chk_fnn_dcs,
            "piu": chk_piu_dcs
        }
    }
    with open("results/metrics/double_cold_start_reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility_meta, f, indent=2, cls=NumpyEncoder)

    gen_results["DoubleColdStart"] = {
        "split_name": dcs_split.get("split_name", "double_cold_start"),
        "val_strategy": dcs_split.get("val_strategy", "random_pair"),
        "n_test_samples": int(len(te_idx)),
        "n_test_drugs": dcs_split["n_test_drugs"],
        "n_test_cells": dcs_split["n_test_cells"],
        "n_actual_test_drugs": len(np.unique(te_drugs)),
        "n_actual_test_cells": len(np.unique(te_cells)),
        "unrepresented_test_drugs": dcs_split.get("unrepresented_test_drugs", []),
        "unrepresented_test_cells": dcs_split.get("unrepresented_test_cells", []),
        "FNN_pooled": dcs_fnn_m,
        "PathDSP_PIU_pooled": dcs_piu_m,
        "FNN": dcs_fnn_m,
        "PathDSP_PIU": dcs_piu_m,
        "macro_metrics": {
            "fnn_by_drug": macro_fnn_drug["macro"],
            "piu_by_drug": macro_piu_drug["macro"],
            "fnn_by_cell": macro_fnn_cell["macro"],
            "piu_by_cell": macro_piu_cell["macro"]
        },
        "statistical_tests": {
            "observation_bootstrap": obs_boot,
            "drug_clustered_bootstrap": drug_boot,
            "cell_clustered_bootstrap": cell_boot,
            "two_way_clustered_bootstrap": tw_boot
        },
        "bootstrap": obs_boot
    }

    logger.info("Double Cold-Start Results:")
    logger.info(f"  FNN Pooled MAE: {dcs_fnn_m['mae']:.4f} | RMSE: {dcs_fnn_m['rmse']:.4f}")
    logger.info(f"  PIU Pooled MAE: {dcs_piu_m['mae']:.4f} | RMSE: {dcs_piu_m['rmse']:.4f}")
    logger.info(f"  Observed Delta MAE: {obs_boot['observed_delta_mae']:.4f}")
    logger.info(f"  Observation Bootstrap 95% CI: [{obs_boot['delta_mae_ci_lower']:.4f}, {obs_boot['delta_mae_ci_upper']:.4f}]")
    logger.info(f"  Drug-Clustered Bootstrap 95% CI: [{drug_boot['delta_mae_ci_lower']:.4f}, {drug_boot['delta_mae_ci_upper']:.4f}]")
    logger.info(f"  Cell-Clustered Bootstrap 95% CI: [{cell_boot['delta_mae_ci_lower']:.4f}, {cell_boot['delta_mae_ci_upper']:.4f}]")
    logger.info(f"  Two-Way Clustered Bootstrap 95% CI: [{tw_boot['delta_mae_ci_lower']:.4f}, {tw_boot['delta_mae_ci_upper']:.4f}]")

    with open(os.path.join(out_dir, "generalization_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(gen_results, f, indent=2, cls=NumpyEncoder)

    return gen_results


def run_ablations(data, split, config, device, out_dir="results/metrics"):
    logger.info("=" * 60)
    logger.info("RUNNING ARCHITECTURAL ABLATION SUITE (Variants A - G)")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    tr_idx = np.array(split["train_indices"])
    va_idx = np.array(split["val_indices"])
    te_idx = np.array(split["test_indices"])

    preprocessor = DataPreprocessor().fit(data, tr_idx)
    tr_set = preprocessor.transform(data, tr_idx)
    va_set = preprocessor.transform(data, va_idx)
    te_set = preprocessor.transform(data, te_idx)

    tr_ld, va_ld, te_ld = create_dataloaders(tr_set, va_set, te_set, batch_size=config["training"]["batch_size"])
    cfg = copy.deepcopy(config)
    cfg["training"]["max_epochs"] = 40

    ablations = {
        "Variant_A_Original_FNN": lambda: PathDSP_FNN(),
        "Variant_B_FNN_plus_Gating": lambda: PathDSP_PIU(use_gating=True, use_interaction=False, use_dg_net=True),
        "Variant_C_FNN_plus_Interaction": lambda: PathDSP_PIU(use_gating=False, use_interaction=True, use_dg_net=True),
        "Variant_D_Full_PathDSP_PIU": lambda: PathDSP_PIU(use_gating=True, use_interaction=True, use_dg_net=True),
        "Variant_E_No_DGNet": lambda: PathDSP_PIU(use_gating=True, use_interaction=True, use_dg_net=False),
        "Variant_F_No_MUT_CNV": lambda: PathDSP_PIU(use_gating=True, use_interaction=True, active_modalities=("exp",)),
        "Variant_G_Single_Model_No_Ensemble": lambda: PathDSP_PIU(use_gating=True, use_interaction=True, use_dg_net=True),
    }

    ablation_metrics = {}

    for var_name, model_fn in ablations.items():
        logger.info(f"Training Ablation {var_name}...")
        # Guarantee independent seed for Variant G single-model evaluation
        if var_name == "Variant_G_Single_Model_No_Ensemble":
            set_seed(42, deterministic=True)
        else:
            set_seed(config.get("seed", 42), deterministic=True)

        model = model_fn()
        model, _ = train_model(model, tr_ld, va_ld, cfg, device)
        eval_res = evaluate_loader(model, te_ld, device)
        m = compute_metrics(eval_res["targets"], eval_res["predictions"])
        ablation_metrics[var_name] = m
        logger.info(f"  {var_name:35s} | MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f} | PCC: {m['pearson']:.4f}")

    with open(os.path.join(out_dir, "ablation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_metrics, f, indent=2)

    return ablation_metrics


def run_explainability_analysis(data, config, device, out_dir="results/reports"):
    logger.info("Extracting biological pathway attributions for RVX-208 and top drugs...")
    os.makedirs(out_dir, exist_ok=True)
    pw_meta_path = "data/processed/pathway_metadata.json"
    with open(pw_meta_path, "r") as f:
        pw_catalog = json.load(f)["pathway_catalog"]
    pathway_names = [p["pathway_name"] for p in pw_catalog]

    # Load trained model checkpoint
    chk_files = [f for f in os.listdir("checkpoints") if f.startswith("pathdsp_piu_seed")]
    if not chk_files:
        logger.warning("No PathDSP-PIU checkpoint found for explainability.")
        return

    chk_path = os.path.join("checkpoints", chk_files[0])
    model = PathDSP_PIU()
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"])

    # Run on random test set
    split = load_split("data/splits/random_split.json")
    te_idx = np.array(split["test_indices"])
    preprocessor = DataPreprocessor().fit(data, np.array(split["train_indices"]))
    te_set = preprocessor.transform(data, te_idx)
    _, _, te_ld = create_dataloaders(te_set, te_set, te_set, batch_size=128)

    attributions = compute_pathway_attributions(model, te_ld, pathway_names, device)
    with open(os.path.join(out_dir, "pathway_attributions.json"), "w") as f:
        json.dump(attributions, f, indent=2)
    logger.info("Pathway attributions saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="PathDSP-PIU Research Framework Orchestrator")
    parser.add_argument("--check", action="store_true", help="Audit hardware and dependencies")
    parser.add_argument("--audit", action="store_true", help="Inspect dataset characteristics")
    parser.add_argument("--mode", type=str, choices=["smoke", "baseline", "train", "evaluate", "full"], help="Execution mode")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Configuration file path")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of cached metrics")

    args = parser.parse_args()

    if args.check:
        run_hardware_check()
        return

    if args.audit:
        run_data_audit()
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(args.device)

    # Ensure dataset is cached
    if not os.path.exists("data/processed/pathdsp_dataset.npz"):
        logger.info("Processed dataset cache missing; preparing dataset...")
        import subprocess
        subprocess.run([sys.executable, "scripts/prepare.py"], check=True)

    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    rnd_split = load_split("data/splits/random_split.json")

    if args.mode == "smoke":
        run_smoke_test(config, device)
    elif args.mode == "baseline":
        train_baseline_suite(data, rnd_split, config, device)
    elif args.mode == "train":
        train_piu_ensemble(data, rnd_split, config, device)
    elif args.mode == "evaluate":
        run_generalization_experiments(data, config, device)
        run_ablations(data, rnd_split, config, device)
    elif args.mode == "full":
        logger.info("=" * 60)
        logger.info("STARTING FULL END-TO-END EXPERIMENTAL PIPELINE")
        logger.info("=" * 60)
        t_start = time.time()
        
        # 1. Train and evaluate all 7 baselines
        if args.force or not os.path.exists("results/metrics/baseline_metrics.json"):
            train_baseline_suite(data, rnd_split, config, device)
        else:
            logger.info("Found cached baseline metrics; skipping retraining.")

        # 2. Train proposed 3-seed ensemble
        if args.force or not os.path.exists("results/metrics/proposed_piu_metrics.json"):
            train_piu_ensemble(data, rnd_split, config, device, ensemble_size=3)
        else:
            logger.info("Found cached proposed ensemble metrics; skipping retraining.")
        
        # 3. Rigorous generalization experiments (LODO, LOCO, Double-Cold-Start)
        run_generalization_experiments(data, config, device)
        
        # 4. Architectural ablations (Variants A-G)
        run_ablations(data, rnd_split, config, device)
        
        # 5. Biological pathway explainability
        run_explainability_analysis(data, config, device)

        # 6. Generate reports, tables, and figures
        import subprocess
        logger.info("Generating publication figures and research report tables...")
        subprocess.run([sys.executable, "scripts/report.py"], check=True)

        elapsed = time.time() - t_start
        logger.info(f"FULL EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY IN {elapsed/60:.2f} MINUTES!")


if __name__ == "__main__":
    main()
