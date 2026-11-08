"""
Data splitting and validation module for PathDSP-PIU:
- Random Pair Split (80/10/10) with exact partition disjointness
- Leave-One-Drug-Out (LODO) with random-pair or entity-disjoint validation
- Leave-One-Cell-Line-Out (LOCO) with random-pair or entity-disjoint validation
- Double-Cold-Start with multiple predefined seeds and entity-disjoint validation
- Strict split leakage verification, index range verification, entity representation verification
- Robust JSON serialization for NumPy scalar types
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    JSON Encoder robust to NumPy data types (integers, floats, arrays, booleans, strings).
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.str_, np.bytes_)):
            return str(obj)
        elif isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        return super(NumpyEncoder, self).default(obj)


def verify_split_leakage(split: dict, drugs: np.ndarray, cells: np.ndarray, split_type: str = "random", n_samples: int = None):
    """
    Formally verifies complete partition disjointness, index range validity,
    and entity-level cold-start constraints.
    
    Raises:
        ValueError: On index range violations, internal duplicates, or invalid configuration.
        AssertionError: On data leakage or constraint violation.
    """
    drugs = np.asarray(drugs)
    cells = np.asarray(cells)
    if n_samples is None:
        n_samples = len(drugs)

    tr_list = list(split.get("train_indices", []))
    va_list = list(split.get("val_indices", []))
    te_list = list(split.get("test_indices", []))

    tr_idx = set(tr_list)
    va_idx = set(va_list)
    te_idx = set(te_list)

    # 1. Index range and duplicate checks
    for partition_name, p_list, p_set in [("Train", tr_list, tr_idx), ("Val", va_list, va_idx), ("Test", te_list, te_idx)]:
        if len(p_list) != len(p_set):
            raise ValueError(f"{partition_name} partition contains duplicate indices (list length: {len(p_list)}, unique: {len(p_set)})!")
        for idx in p_list:
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(f"Index {idx} in {partition_name} is not an integer.")
            if idx < 0 or idx >= n_samples:
                raise ValueError(f"Index {idx} in {partition_name} is out of bounds [0, {n_samples - 1}].")

    # 2. Complete partition-disjointness checks
    tr_va_overlap = tr_idx.intersection(va_idx)
    tr_te_overlap = tr_idx.intersection(te_idx)
    va_te_overlap = va_idx.intersection(te_idx)

    assert len(tr_va_overlap) == 0, f"Partition Overlap: Train and Val share {len(tr_va_overlap)} indices!"
    assert len(tr_te_overlap) == 0, f"Partition Overlap: Train and Test share {len(tr_te_overlap)} indices!"
    assert len(va_te_overlap) == 0, f"Partition Overlap: Val and Test share {len(va_te_overlap)} indices!"

    # 3. Partition coverage assessment
    assigned_count = len(tr_idx) + len(va_idx) + len(te_idx)
    if split_type == "random":
        if assigned_count != n_samples:
            logger.warning(f"Random split assigned {assigned_count}/{n_samples} total observations.")
    else:
        n_excluded = n_samples - assigned_count
        logger.debug(f"{split_type} split: {assigned_count} assigned, {n_excluded} excluded buffer observations.")

    # 4. Entity-level extraction
    tr_drugs = set(drugs[list(tr_idx)]) if tr_idx else set()
    va_drugs = set(drugs[list(va_idx)]) if va_idx else set()
    te_drugs = set(drugs[list(te_idx)]) if te_idx else set()

    tr_cells = set(cells[list(tr_idx)]) if tr_idx else set()
    va_cells = set(cells[list(va_idx)]) if va_idx else set()
    te_cells = set(cells[list(te_idx)]) if te_idx else set()

    val_strategy = split.get("val_strategy", "random_pair")

    # 5. Generalization constraint checks
    if split_type == "lodo":
        drug_overlap = (tr_drugs | va_drugs).intersection(te_drugs)
        assert len(drug_overlap) == 0, f"LODO Leakage: Test drugs {drug_overlap} found in training/val!"
        if val_strategy == "entity_disjoint":
            va_tr_drug_overlap = tr_drugs.intersection(va_drugs)
            assert len(va_tr_drug_overlap) == 0, f"LODO Entity-Disjoint Leakage: Val drugs {va_tr_drug_overlap} found in training!"

    elif split_type == "loco":
        cell_overlap = (tr_cells | va_cells).intersection(te_cells)
        assert len(cell_overlap) == 0, f"LOCO Leakage: Test cells {cell_overlap} found in training/val!"
        if val_strategy == "entity_disjoint":
            va_tr_cell_overlap = tr_cells.intersection(va_cells)
            assert len(va_tr_cell_overlap) == 0, f"LOCO Entity-Disjoint Leakage: Val cells {va_tr_cell_overlap} found in training!"

    elif split_type == "double_cold_start":
        selected_test_drugs = set(split.get("test_drugs", []))
        selected_test_cells = set(split.get("test_cells", []))

        # A. Every test pair must contain a selected held-out drug AND held-out cell line
        for idx in te_idx:
            d = drugs[idx]
            c = cells[idx]
            assert d in selected_test_drugs, f"Double-Cold-Start Violation: Test observation {idx} contains drug '{d}' not in test_drugs!"
            assert c in selected_test_cells, f"Double-Cold-Start Violation: Test observation {idx} contains cell '{c}' not in test_cells!"

        # B. No selected held-out drug or cell line may occur in training or validation
        for idx in (tr_idx | va_idx):
            d = drugs[idx]
            c = cells[idx]
            assert d not in selected_test_drugs, f"Double-Cold-Start Leakage: Train/Val observation {idx} contains held-out test drug '{d}'!"
            assert c not in selected_test_cells, f"Double-Cold-Start Leakage: Train/Val observation {idx} contains held-out test cell '{c}'!"

        # C. Entity representation checks
        actual_test_drugs = set(drugs[list(te_idx)]) if te_idx else set()
        actual_test_cells = set(cells[list(te_idx)]) if te_idx else set()

        unrep_drugs = selected_test_drugs - actual_test_drugs
        unrep_cells = selected_test_cells - actual_test_cells

        if unrep_drugs:
            logger.info(f"Double-Cold-Start: {len(unrep_drugs)} selected test drugs have zero test pair observations: {sorted(list(unrep_drugs))[:5]}...")
        if unrep_cells:
            logger.info(f"Double-Cold-Start: {len(unrep_cells)} selected test cells have zero test pair observations: {sorted(list(unrep_cells))[:5]}...")

        # D. If validation is entity-disjoint, verify validation sets
        if val_strategy == "entity_disjoint":
            selected_val_drugs = set(split.get("val_drugs", []))
            selected_val_cells = set(split.get("val_cells", []))

            # Val entities must be disjoint from test entities
            assert len(selected_val_drugs.intersection(selected_test_drugs)) == 0, "Val drugs overlap with Test drugs!"
            assert len(selected_val_cells.intersection(selected_test_cells)) == 0, "Val cells overlap with Test cells!"

            # Every val pair must be drawn from selected val entities
            for idx in va_idx:
                d = drugs[idx]
                c = cells[idx]
                assert d in selected_val_drugs, f"Entity-Disjoint Val Violation: Val observation {idx} has drug '{d}' not in val_drugs!"
                assert c in selected_val_cells, f"Entity-Disjoint Val Violation: Val observation {idx} has cell '{c}' not in val_cells!"

            # No val entities may appear in training
            for idx in tr_idx:
                d = drugs[idx]
                c = cells[idx]
                assert d not in selected_val_drugs, f"Entity-Disjoint Val Leakage: Train observation {idx} contains val drug '{d}'!"
                assert c not in selected_val_cells, f"Entity-Disjoint Val Leakage: Train observation {idx} contains val cell '{c}'!"

    return True


def generate_random_split(n_samples: int, train_ratio: float = 0.80, val_ratio: float = 0.10, seed: int = 42):
    """
    Generates standard train / val / test index partition with ratio validation.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = [int(i) for i in indices[:n_train]]
    val_indices = [int(i) for i in indices[n_train:n_train + n_val]]
    test_indices = [int(i) for i in indices[n_train + n_val:]]

    return {
        "split_name": "random_split",
        "split_type": "random",
        "val_strategy": "random_pair",
        "seed": int(seed),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "n_total": n_samples
    }


def generate_lodo_split(drugs: np.ndarray,
                        target_drug: str,
                        val_ratio: float = 0.10,
                        val_strategy: str = "random_pair",
                        n_val_drugs: int = 5,
                        seed: int = 42):
    """
    Leave-One-Drug-Out (LODO):
    - Test set: All observations where drug == target_drug.
    - If val_strategy == 'random_pair': remaining drug pairs partitioned randomly into train/val.
    - If val_strategy == 'entity_disjoint': validation compounds are held out separately from test and train!
    """
    drugs = np.asarray(drugs)
    unique_drugs = np.unique(drugs)
    if target_drug not in unique_drugs:
        raise ValueError(f"Target drug '{target_drug}' not found in dataset drugs.")

    test_mask = (drugs == target_drug)
    test_indices = [int(i) for i in np.where(test_mask)[0]]
    remaining_indices = np.where(~test_mask)[0]

    rng = np.random.RandomState(seed)

    if val_strategy == "entity_disjoint":
        candidate_val_drugs = [d for d in unique_drugs if d != target_drug]
        n_val_drugs = max(1, min(n_val_drugs, int(len(candidate_val_drugs) * val_ratio) or 1))
        shuffled_val_drugs = rng.permutation(candidate_val_drugs)
        val_drugs = set(shuffled_val_drugs[:n_val_drugs])
        train_drugs = set(shuffled_val_drugs[n_val_drugs:])

        val_mask = np.isin(drugs, list(val_drugs))
        train_mask = np.isin(drugs, list(train_drugs))

        val_indices = [int(i) for i in np.where(val_mask)[0]]
        train_indices = [int(i) for i in np.where(train_mask)[0]]
    else:
        # Default: random_pair validation from non-test observations
        shuffled_remaining = rng.permutation(remaining_indices)
        n_val = max(1, int(len(shuffled_remaining) * val_ratio))
        val_indices = [int(i) for i in shuffled_remaining[:n_val]]
        train_indices = [int(i) for i in shuffled_remaining[n_val:]]
        val_drugs = sorted(list(set(drugs[val_indices])))
        train_drugs = sorted(list(set(drugs[train_indices])))

    return {
        "split_name": f"lodo_{target_drug}",
        "split_type": "lodo",
        "val_strategy": val_strategy,
        "target_drug": str(target_drug),
        "test_drugs": [str(target_drug)],
        "val_drugs": sorted(list(val_drugs)),
        "train_drugs": sorted(list(train_drugs)),
        "seed": int(seed),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "n_excluded": len(drugs) - (len(train_indices) + len(val_indices) + len(test_indices))
    }


def generate_loco_split(cells: np.ndarray,
                        target_cell: str,
                        val_ratio: float = 0.10,
                        val_strategy: str = "random_pair",
                        n_val_cells: int = 10,
                        seed: int = 42):
    """
    Leave-One-Cell-Line-Out (LOCO):
    - Test set: All observations where cell == target_cell.
    - If val_strategy == 'random_pair': remaining cell pairs partitioned randomly into train/val.
    - If val_strategy == 'entity_disjoint': validation cell lines held out separately.
    """
    cells = np.asarray(cells)
    unique_cells = np.unique(cells)
    if target_cell not in unique_cells:
        raise ValueError(f"Target cell line '{target_cell}' not found in dataset cell lines.")

    test_mask = (cells == target_cell)
    test_indices = [int(i) for i in np.where(test_mask)[0]]
    remaining_indices = np.where(~test_mask)[0]

    rng = np.random.RandomState(seed)

    if val_strategy == "entity_disjoint":
        candidate_val_cells = [c for c in unique_cells if c != target_cell]
        n_val_cells = max(1, min(n_val_cells, int(len(candidate_val_cells) * val_ratio) or 1))
        shuffled_val_cells = rng.permutation(candidate_val_cells)
        val_cells = set(shuffled_val_cells[:n_val_cells])
        train_cells = set(shuffled_val_cells[n_val_cells:])

        val_mask = np.isin(cells, list(val_cells))
        train_mask = np.isin(cells, list(train_cells))

        val_indices = [int(i) for i in np.where(val_mask)[0]]
        train_indices = [int(i) for i in np.where(train_mask)[0]]
    else:
        shuffled_remaining = rng.permutation(remaining_indices)
        n_val = max(1, int(len(shuffled_remaining) * val_ratio))
        val_indices = [int(i) for i in shuffled_remaining[:n_val]]
        train_indices = [int(i) for i in shuffled_remaining[n_val:]]
        val_cells = sorted(list(set(cells[val_indices])))
        train_cells = sorted(list(set(cells[train_indices])))

    return {
        "split_name": f"loco_{target_cell}",
        "split_type": "loco",
        "val_strategy": val_strategy,
        "target_cell": str(target_cell),
        "test_cells": [str(target_cell)],
        "val_cells": sorted(list(val_cells)),
        "train_cells": sorted(list(train_cells)),
        "seed": int(seed),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "n_excluded": len(cells) - (len(train_indices) + len(val_indices) + len(test_indices))
    }


def generate_double_cold_start_split(drugs: np.ndarray,
                                    cells: np.ndarray,
                                    drug_test_ratio: float = 0.15,
                                    cell_test_ratio: float = 0.15,
                                    val_ratio: float = 0.10,
                                    val_strategy: str = "random_pair",
                                    min_test_samples: int = 50,
                                    seed: int = 42):
    """
    Double Cold-Start:
    - Test set consists STRICTLY of pairs where (d in D_test) AND (c in C_test).
    - If val_strategy == 'entity_disjoint':
        Separate D_val and C_val are sampled disjoint from D_test, D_train and C_test, C_train.
        Validation observations are drawn STRICTLY from (d in D_val) AND (c in C_val).
    - If val_strategy == 'random_pair':
        Validation observations are drawn randomly from candidate training pool (d not in D_test) AND (c not in C_test).
    - Intermediate pairs (d in D_test, c not in C_test) or (d not in D_test, c in C_test) are excluded.
    """
    drugs = np.asarray(drugs)
    cells = np.asarray(cells)
    unique_drugs = np.unique(drugs)
    unique_cells = np.unique(cells)

    if not (0.0 < drug_test_ratio < 0.5):
        raise ValueError(f"drug_test_ratio must be in (0, 0.5), got {drug_test_ratio}")
    if not (0.0 < cell_test_ratio < 0.5):
        raise ValueError(f"cell_test_ratio must be in (0, 0.5), got {cell_test_ratio}")
    if not (0.0 < val_ratio < 0.5):
        raise ValueError(f"val_ratio must be in (0, 0.5), got {val_ratio}")

    rng = np.random.RandomState(seed)
    n_test_drugs = max(1, int(len(unique_drugs) * drug_test_ratio))
    n_test_cells = max(1, int(len(unique_cells) * cell_test_ratio))

    shuffled_drugs = rng.permutation(unique_drugs)
    shuffled_cells = rng.permutation(unique_cells)

    test_drugs = set(shuffled_drugs[:n_test_drugs])
    test_cells = set(shuffled_cells[:n_test_cells])

    # Identify test observations
    is_test_drug = np.isin(drugs, list(test_drugs))
    is_test_cell = np.isin(cells, list(test_cells))
    test_indices = [int(i) for i in np.where(is_test_drug & is_test_cell)[0]]

    # Check for unrepresented entities
    actual_test_drugs = set(drugs[test_indices]) if test_indices else set()
    actual_test_cells = set(cells[test_indices]) if test_indices else set()
    unrepresented_drugs = sorted(list(test_drugs - actual_test_drugs))
    unrepresented_cells = sorted(list(test_cells - actual_test_cells))

    if len(test_indices) < min_test_samples:
        logger.warning(f"Double-Cold-Start (seed {seed}): Sparse split warning. Only {len(test_indices)} test pairs found (threshold: {min_test_samples}).")

    if val_strategy == "entity_disjoint":
        remaining_drugs = shuffled_drugs[n_test_drugs:]
        remaining_cells = shuffled_cells[n_test_cells:]

        n_val_drugs = max(1, int(len(remaining_drugs) * val_ratio))
        n_val_cells = max(1, int(len(remaining_cells) * val_ratio))

        val_drugs = set(remaining_drugs[:n_val_drugs])
        val_cells = set(remaining_cells[:n_val_cells])

        train_drugs = set(remaining_drugs[n_val_drugs:])
        train_cells = set(remaining_cells[n_val_cells:])

        # Val observations: STRICTLY (d in D_val) AND (c in C_val)
        is_val_drug = np.isin(drugs, list(val_drugs))
        is_val_cell = np.isin(cells, list(val_cells))
        val_indices = [int(i) for i in np.where(is_val_drug & is_val_cell)[0]]

        # Train observations: STRICTLY (d in D_train) AND (c in C_train)
        is_tr_drug = np.isin(drugs, list(train_drugs))
        is_tr_cell = np.isin(cells, list(train_cells))
        train_indices = [int(i) for i in np.where(is_tr_drug & is_tr_cell)[0]]
    else:
        # Default: random_pair from candidate training pool
        train_pool_indices = np.where((~is_test_drug) & (~is_test_cell))[0]
        shuffled_pool = rng.permutation(train_pool_indices)
        n_val = max(1, int(len(shuffled_pool) * val_ratio))

        val_indices = [int(i) for i in shuffled_pool[:n_val]]
        train_indices = [int(i) for i in shuffled_pool[n_val:]]

        val_drugs = set(drugs[val_indices])
        val_cells = set(cells[val_indices])
        train_drugs = set(drugs[train_indices])
        train_cells = set(cells[train_indices])

    n_assigned = len(train_indices) + len(val_indices) + len(test_indices)
    n_excluded = len(drugs) - n_assigned

    status = "valid" if len(test_indices) >= min_test_samples else "insufficient_test_samples"

    split = {
        "split_name": f"double_cold_start_seed_{seed}",
        "split_type": "double_cold_start",
        "val_strategy": val_strategy,
        "seed": int(seed),
        "status": status,
        "min_test_samples_threshold": int(min_test_samples),
        "n_test_drugs": len(test_drugs),
        "n_test_cells": len(test_cells),
        "n_actual_test_drugs": len(actual_test_drugs),
        "n_actual_test_cells": len(actual_test_cells),
        "test_drugs": sorted(list(test_drugs)),
        "test_cells": sorted(list(test_cells)),
        "unrepresented_test_drugs": unrepresented_drugs,
        "unrepresented_test_cells": unrepresented_cells,
        "val_drugs": sorted(list(val_drugs)),
        "val_cells": sorted(list(val_cells)),
        "train_drugs": sorted(list(train_drugs)),
        "train_cells": sorted(list(train_cells)),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "n_excluded": int(n_excluded),
        "n_total": len(drugs)
    }
    return split


def save_split(split: dict, output_path: str):
    """
    Saves split dictionary to JSON with safe directory creation
    and robust NumPy scalar encoding.
    """
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved split '{split.get('split_name')}' to {output_path}")


def load_split(split_path: str) -> dict:
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)
