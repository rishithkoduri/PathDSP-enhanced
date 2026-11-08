"""
Comprehensive unit tests for data splits, zero-leakage assertions,
index validation, entity representation, and serialization.
"""

import os
import json
import pytest
import numpy as np
from src.splits import (
    generate_random_split,
    generate_lodo_split,
    generate_loco_split,
    generate_double_cold_start_split,
    verify_split_leakage,
    save_split,
    load_split,
    NumpyEncoder
)


@pytest.fixture
def mock_dataset():
    # 100 observations across 5 drugs and 4 cell lines
    drugs = np.array(["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4", "D5", "D5"] * 10)
    cells = np.array(["C1", "C2", "C1", "C2", "C1", "C2", "C3", "C4", "C3", "C4"] * 10)
    return drugs, cells


def test_random_split_bounds_and_disjointness(mock_dataset):
    drugs, cells = mock_dataset
    n_samples = len(drugs)
    split = generate_random_split(n_samples, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert verify_split_leakage(split, drugs, cells, split_type="random", n_samples=n_samples)


def test_invalid_split_ratios(mock_dataset):
    drugs, cells = mock_dataset
    n_samples = len(drugs)
    with pytest.raises(ValueError):
        generate_random_split(n_samples, train_ratio=0.8, val_ratio=0.3)  # sum > 1.0

    with pytest.raises(ValueError):
        generate_double_cold_start_split(drugs, cells, drug_test_ratio=0.6)  # ratio >= 0.5


def test_out_of_bounds_index_rejection(mock_dataset):
    drugs, cells = mock_dataset
    n_samples = len(drugs)
    split = generate_random_split(n_samples, train_ratio=0.7, val_ratio=0.15, seed=42)
    split["test_indices"].append(n_samples + 5)  # Out of bounds!
    with pytest.raises(ValueError, match="out of bounds"):
        verify_split_leakage(split, drugs, cells, split_type="random", n_samples=n_samples)


def test_duplicate_index_rejection(mock_dataset):
    drugs, cells = mock_dataset
    n_samples = len(drugs)
    split = generate_random_split(n_samples, train_ratio=0.7, val_ratio=0.15, seed=42)
    split["train_indices"].append(split["train_indices"][0])  # Duplicate!
    with pytest.raises(ValueError, match="duplicate indices"):
        verify_split_leakage(split, drugs, cells, split_type="random", n_samples=n_samples)


def test_lodo_random_and_entity_disjoint(mock_dataset):
    drugs, cells = mock_dataset
    # Random-pair val
    split_rand = generate_lodo_split(drugs, target_drug="D1", val_strategy="random_pair", seed=42)
    assert verify_split_leakage(split_rand, drugs, cells, split_type="lodo", n_samples=len(drugs))

    # Entity-disjoint val
    split_disjoint = generate_lodo_split(drugs, target_drug="D1", val_strategy="entity_disjoint", n_val_drugs=1, seed=42)
    assert verify_split_leakage(split_disjoint, drugs, cells, split_type="lodo", n_samples=len(drugs))
    assert len(set(split_disjoint["val_drugs"]).intersection(set(split_disjoint["train_drugs"]))) == 0


def test_loco_random_and_entity_disjoint(mock_dataset):
    drugs, cells = mock_dataset
    split_rand = generate_loco_split(cells, target_cell="C1", val_strategy="random_pair", seed=42)
    assert verify_split_leakage(split_rand, drugs, cells, split_type="loco", n_samples=len(cells))

    split_disjoint = generate_loco_split(cells, target_cell="C1", val_strategy="entity_disjoint", n_val_cells=1, seed=42)
    assert verify_split_leakage(split_disjoint, drugs, cells, split_type="loco", n_samples=len(cells))
    assert len(set(split_disjoint["val_cells"]).intersection(set(split_disjoint["train_cells"]))) == 0


def test_double_cold_start_leakage_and_unrepresented_reporting(mock_dataset):
    drugs, cells = mock_dataset
    split = generate_double_cold_start_split(drugs, cells, drug_test_ratio=0.2, cell_test_ratio=0.2, seed=42)
    assert verify_split_leakage(split, drugs, cells, split_type="double_cold_start", n_samples=len(drugs))

    test_drugs = set(drugs[split["test_indices"]])
    test_cells = set(cells[split["test_indices"]])
    train_drugs = set(drugs[split["train_indices"]])
    train_cells = set(cells[split["train_indices"]])

    assert len(test_drugs.intersection(train_drugs)) == 0
    assert len(test_cells.intersection(train_cells)) == 0
    assert "unrepresented_test_drugs" in split
    assert "unrepresented_test_cells" in split


def test_double_cold_start_entity_disjoint(mock_dataset):
    drugs, cells = mock_dataset
    split = generate_double_cold_start_split(drugs, cells, drug_test_ratio=0.2, cell_test_ratio=0.2, val_strategy="entity_disjoint", seed=42)
    assert verify_split_leakage(split, drugs, cells, split_type="double_cold_start", n_samples=len(drugs))
    assert set(split["val_drugs"]).isdisjoint(set(split["test_drugs"]))
    assert set(split["val_drugs"]).isdisjoint(set(split["train_drugs"]))


def test_save_split_without_parent_directory(tmp_path):
    split_data = {
        "split_name": "test_no_parent",
        "scalar_int": np.int64(42),
        "scalar_float": np.float32(3.14),
        "array": np.array([1, 2, 3]),
        "train_indices": [0, 1, 2],
        "val_indices": [3],
        "test_indices": [4]
    }
    # Path without directory component
    current_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        filename = "standalone_split.json"
        save_split(split_data, filename)
        assert os.path.exists(filename)
        loaded = load_split(filename)
        assert loaded["scalar_int"] == 42
        assert loaded["scalar_float"] == pytest.approx(3.14, rel=1e-3)
    finally:
        os.chdir(current_dir)
