import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset
from src.splits import verify_split_leakage, save_split

data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
drugs = data["drugs"]
cells = data["cells"]
n_samples = len(drugs)

split_path = "data/splits/random_split.json"
with open(split_path, "r") as fp:
    split = json.load(fp)

tr_idx = [int(i) for i in split["train_indices"]]
va_all = np.array(split["val_indices"])
te_idx = [int(i) for i in split["test_indices"]]

# If random_split.json already has cal_indices of length 1887, reuse or verify
if "cal_indices" in split and len(split["cal_indices"]) == 1887 and len(split["val_indices"]) == 1886:
    ca_idx = [int(i) for i in split["cal_indices"]]
    va_idx = [int(i) for i in split["val_indices"]]
    print("random_split.json already contains separated val (1886) and cal (1887).")
else:
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(va_all))
    half = len(va_all) // 2
    va_idx = [int(i) for i in va_all[perm[:half]]]
    ca_idx = [int(i) for i in va_all[perm[half:]]]

new_split = {
    "split_name": "random_split",
    "split_type": "random",
    "val_strategy": "random_pair",
    "seed": 42,
    "train_indices": tr_idx,
    "val_indices": va_idx,
    "cal_indices": ca_idx,
    "test_indices": te_idx,
    "n_train": len(tr_idx),
    "n_val": len(va_idx),
    "n_cal": len(ca_idx),
    "n_test": len(te_idx),
    "n_total": len(tr_idx) + len(va_idx) + len(ca_idx) + len(te_idx)
}

# Verify leakage and 4-way disjointness
verify_split_leakage(new_split, drugs, cells, split_type="random", n_samples=n_samples)
save_split(new_split, split_path)
print(f"Updated {split_path}: Train={new_split['n_train']}, Val={new_split['n_val']}, Cal={new_split['n_cal']}, Test={new_split['n_test']}, Total={new_split['n_total']}")

# Create calibration manifest
cal_manifest = {
    "manifest_name": "conformal_calibration_manifest",
    "associated_split": "data/splits/random_split.json",
    "split_type": "random_pair",
    "total_dataset_samples": n_samples,
    "partition_breakdown": {
        "train": {
            "n_samples": len(tr_idx),
            "ratio": len(tr_idx) / n_samples,
            "role": "Model parameter training via backpropagation / gradient descent"
        },
        "val": {
            "n_samples": len(va_idx),
            "ratio": len(va_idx) / n_samples,
            "role": "Validation monitoring and early stopping (patience=12)"
        },
        "cal": {
            "n_samples": len(ca_idx),
            "ratio": len(ca_idx) / n_samples,
            "role": "Conformal calibration (unobserved during training/early stopping/model selection)"
        },
        "test": {
            "n_samples": len(te_idx),
            "ratio": len(te_idx) / n_samples,
            "role": "Final held-out empirical evaluation and interval coverage assessment"
        }
    },
    "disjointness_verification": {
        "train_val_overlap": len(set(tr_idx).intersection(set(va_idx))),
        "train_cal_overlap": len(set(tr_idx).intersection(set(ca_idx))),
        "train_test_overlap": len(set(tr_idx).intersection(set(te_idx))),
        "val_cal_overlap": len(set(va_idx).intersection(set(ca_idx))),
        "val_test_overlap": len(set(va_idx).intersection(set(te_idx))),
        "cal_test_overlap": len(set(ca_idx).intersection(set(te_idx))),
        "total_unique_indices": len(set(tr_idx) | set(va_idx) | set(ca_idx) | set(te_idx)),
        "all_disjoint": True
    },
    "protocol_details": {
        "framework": "Strict split-conformal prediction (adaptive)",
        "nonconformity_score": "s_i = |y_i - mu_hat(x_i)| / sigma_hat(x_i)",
        "finite_sample_coverage_guarantee": "P(y_test in C(x_test)) >= 1 - alpha",
        "quantile_formula": "np.quantile(scores, min(1.0, ceil((n+1)*(1-alpha))/n), method='higher')",
        "contamination_status": "NONE. Calibration observations strictly excluded from gradient updates, hyperparameter selection, and early stopping."
    },
    "calibration_indices": ca_idx
}

manifest_path = "data/splits/calibration_manifest.json"
with open(manifest_path, "w") as fp:
    json.dump(cal_manifest, fp, indent=2)

print(f"Saved linked calibration manifest to {manifest_path}")
