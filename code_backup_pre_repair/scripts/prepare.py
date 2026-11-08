"""
Data Preparation and Split Generation Script:
1. Loads raw dataset (data/raw/input.txt)
2. Validates schema, dimensions, and pathway alignments
3. Saves binary dataset cache (data/processed/pathdsp_dataset.npz)
4. Saves pathway metadata catalog (data/processed/pathway_metadata.json)
5. Generates and verifies disjoint splits:
   - Random Split (80/10/10)
   - LODO splits (for prioritized top drugs)
   - LOCO splits (for prioritized top cell lines)
   - Double-Cold-Start splits across multiple predefined seeds (e.g. 42, 1042, 2042, 3042, 4042)
   - Entity-disjoint validation splits
6. Outputs Data Audit Report and Split Manifests
"""

import os
import sys
import json
import logging
import yaml
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FeatureRegistry
from src.data import prepare_and_cache_dataset, load_cached_dataset
from src.splits import (
    generate_random_split,
    generate_lodo_split,
    generate_loco_split,
    generate_double_cold_start_split,
    verify_split_leakage,
    save_split,
    NumpyEncoder
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = f.read()
    config = yaml.safe_load(cfg)

    raw_path = config["data"]["raw_file"]
    cache_path = config["data"]["cache_file"]
    meta_path = config["data"]["metadata_file"]
    pw_meta_path = config["data"]["pathway_metadata_file"]
    splits_dir = config["data"]["splits_dir"]

    logger.info("Step 1: Preparing and caching structured dataset...")
    if not os.path.exists(cache_path):
        structured, metadata = prepare_and_cache_dataset(
            raw_path=raw_path,
            cache_path=cache_path,
            metadata_path=meta_path,
            pathway_meta_path=pw_meta_path
        )
    else:
        structured = load_cached_dataset(cache_path)

    drugs = structured["drugs"]
    cells = structured["cells"]
    n_samples = len(drugs)

    logger.info(f"Step 2: Generating and verifying leakage-safe splits for {n_samples} response pairs...")
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Random split (80/10/10)
    rnd_split = generate_random_split(
        n_samples=n_samples,
        train_ratio=config["evaluation"]["random_split"]["train_ratio"],
        val_ratio=config["evaluation"]["random_split"]["val_ratio"],
        seed=config["seed"]
    )
    verify_split_leakage(rnd_split, drugs, cells, split_type="random", n_samples=n_samples)
    save_split(rnd_split, os.path.join(splits_dir, "random_split.json"))

    # 2. LODO splits (select top drugs with high representation)
    unique_drugs, drug_counts = np.unique(drugs, return_counts=True)
    sorted_drugs = unique_drugs[np.argsort(-drug_counts)]
    n_lodo = config["evaluation"]["lodo"]["n_drugs_eval"]
    lodo_eval_drugs = sorted_drugs[:n_lodo]

    lodo_manifest = []
    for d in lodo_eval_drugs:
        split = generate_lodo_split(drugs, target_drug=d, val_strategy="random_pair", seed=config["seed"])
        verify_split_leakage(split, drugs, cells, split_type="lodo", n_samples=n_samples)
        path = os.path.join(splits_dir, f"lodo_{d}.json")
        save_split(split, path)
        lodo_manifest.append({
            "drug": str(d),
            "n_test": split["n_test"],
            "val_strategy": split["val_strategy"],
            "file": path
        })

    with open(os.path.join(splits_dir, "lodo_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(lodo_manifest, f, indent=2, cls=NumpyEncoder)

    # 3. LOCO splits (select top cell lines with high representation)
    unique_cells, cell_counts = np.unique(cells, return_counts=True)
    sorted_cells = unique_cells[np.argsort(-cell_counts)]
    n_loco = config["evaluation"]["loco"]["n_cells_eval"]
    loco_eval_cells = sorted_cells[:n_loco]

    loco_manifest = []
    for c in loco_eval_cells:
        split = generate_loco_split(cells, target_cell=c, val_strategy="random_pair", seed=config["seed"])
        verify_split_leakage(split, drugs, cells, split_type="loco", n_samples=n_samples)
        path = os.path.join(splits_dir, f"loco_{c}.json")
        save_split(split, path)
        loco_manifest.append({
            "cell": str(c),
            "n_test": split["n_test"],
            "val_strategy": split["val_strategy"],
            "file": path
        })

    with open(os.path.join(splits_dir, "loco_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(loco_manifest, f, indent=2, cls=NumpyEncoder)

    # 4. Multi-Seed Double Cold-Start splits
    dcs_cfg = config["evaluation"]["double_cold_start"]
    dcs_seeds = [42, 1042, 2042, 3042, 4042]
    dcs_manifest = []

    for s in dcs_seeds:
        dcs_split = generate_double_cold_start_split(
            drugs=drugs,
            cells=cells,
            drug_test_ratio=dcs_cfg["drug_test_ratio"],
            cell_test_ratio=dcs_cfg["cell_test_ratio"],
            val_ratio=dcs_cfg["val_ratio"],
            val_strategy="random_pair",
            min_test_samples=50,
            seed=s
        )
        verify_split_leakage(dcs_split, drugs, cells, split_type="double_cold_start", n_samples=n_samples)
        path = os.path.join(splits_dir, f"double_cold_start_seed_{s}.json")
        save_split(dcs_split, path)

        manifest_entry = {
            "seed": int(s),
            "status": dcs_split["status"],
            "val_strategy": dcs_split["val_strategy"],
            "n_test": dcs_split["n_test"],
            "n_test_drugs": dcs_split["n_test_drugs"],
            "n_test_cells": dcs_split["n_test_cells"],
            "n_actual_test_drugs": dcs_split["n_actual_test_drugs"],
            "n_actual_test_cells": dcs_split["n_actual_test_cells"],
            "unrepresented_test_drugs": dcs_split["unrepresented_test_drugs"],
            "unrepresented_test_cells": dcs_split["unrepresented_test_cells"],
            "file": path
        }
        if dcs_split["status"] != "valid":
            manifest_entry["exclusion_reason"] = f"Insufficient test observations ({dcs_split['n_test']} < threshold 50)."
        dcs_manifest.append(manifest_entry)

        # Primary benchmark split alias for seed 42
        if s == 42:
            save_split(dcs_split, os.path.join(splits_dir, "double_cold_start.json"))

    # Also generate entity-disjoint validation double-cold-start split for seed 42
    dcs_disjoint = generate_double_cold_start_split(
        drugs=drugs,
        cells=cells,
        drug_test_ratio=dcs_cfg["drug_test_ratio"],
        cell_test_ratio=dcs_cfg["cell_test_ratio"],
        val_ratio=dcs_cfg["val_ratio"],
        val_strategy="entity_disjoint",
        min_test_samples=50,
        seed=42
    )
    verify_split_leakage(dcs_disjoint, drugs, cells, split_type="double_cold_start", n_samples=n_samples)
    save_split(dcs_disjoint, os.path.join(splits_dir, "double_cold_start_entity_disjoint.json"))

    with open(os.path.join(splits_dir, "double_cold_start_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dcs_manifest, f, indent=2, cls=NumpyEncoder)

    logger.info("All splits successfully generated, verified, and logged in split manifests!")


if __name__ == "__main__":
    main()
