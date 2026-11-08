import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_cached_dataset
from src.splits import load_split, verify_split_leakage

data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
drugs = data["drugs"]
cells = data["cells"]
n_samples = len(drugs)

splits_dir = "data/splits"
results = {}

print("=== SPLIT AUDIT & VERIFICATION ===")
for f in sorted(os.listdir(splits_dir)):
    if f.endswith(".json") and not f.endswith("_manifest.json"):
        p = os.path.join(splits_dir, f)
        split = load_split(p)
        split_type = split.get("split_type", "random")
        val_strategy = split.get("val_strategy", "random_pair")
        
        try:
            valid = verify_split_leakage(split, drugs, cells, split_type=split_type, n_samples=n_samples)
            status = "VERIFIED_NO_LEAKAGE"
        except Exception as e:
            status = f"FAILED: {e}"

        info = {
            "file": f,
            "split_name": split.get("split_name", f),
            "split_type": split_type,
            "val_strategy": val_strategy,
            "seed": split.get("seed", 42),
            "n_train": len(split.get("train_indices", [])),
            "n_val": len(split.get("val_indices", [])),
            "n_cal": len(split.get("cal_indices", [])),
            "n_test": len(split.get("test_indices", [])),
            "status": status
        }
        if "test_drugs" in split:
            info["n_test_drugs"] = len(split["test_drugs"])
        if "test_cells" in split:
            info["n_test_cells"] = len(split["test_cells"])
        if "unrepresented_test_drugs" in split:
            info["unrepresented_test_drugs"] = split["unrepresented_test_drugs"]
        if "unrepresented_test_cells" in split:
            info["unrepresented_test_cells"] = split["unrepresented_test_cells"]

        results[f] = info
        print(f"[{status}] {f}: Train={info['n_train']}, Val={info['n_val']}, Cal={info['n_cal']}, Test={info['n_test']}")

os.makedirs("results/verification", exist_ok=True)
with open("results/verification/split_verification.json", "w") as fp:
    json.dump(results, fp, indent=2)

print("\nSaved split audit to results/verification/split_verification.json")
