"""
Tests for Data Loading, Feature Preprocessing, and PyTorch Datasets.
"""

import os
import pytest
import numpy as np
import torch
from src.data import load_cached_dataset, DataPreprocessor, DrugResponseDataset, create_dataloaders


def test_cached_data_shapes():
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    assert data["chem"].shape == (37732, 256)
    assert data["dg"].shape == (37732, 196)
    assert data["cell_pathways"].shape == (37732, 196, 3)
    assert data["flat_features"].shape == (37732, 1031)
    assert data["target"].shape == (37732,)
    assert len(data["drugs"]) == 37732
    assert len(data["cells"]) == 37732


def test_no_nans_or_infs():
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    for k in ["chem", "dg", "cell_pathways", "flat_features", "target"]:
        assert not np.isnan(data[k]).any(), f"NaN found in {k}"
        assert not np.isinf(data[k]).any(), f"Inf found in {k}"


def test_data_preprocessor_leakage_safety():
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    preprocessor = DataPreprocessor()
    
    # Train indices on first 1000 samples
    train_idx = np.arange(1000)
    test_idx = np.arange(1000, 1500)
    
    preprocessor.fit(data, train_idx)
    tr_trans = preprocessor.transform(data, train_idx)
    te_trans = preprocessor.transform(data, test_idx)
    
    # Verify train transformed flat features have mean ~ 0 and std ~ 1
    assert np.allclose(np.mean(tr_trans["flat_features"], axis=0), 0.0, atol=1e-2)
    assert tr_trans["cell_pathways"].shape == (1000, 196, 3)
    assert te_trans["cell_pathways"].shape == (500, 196, 3)
    assert tr_trans["chem"].shape == (1000, 256)


def test_drug_response_dataset():
    data = load_cached_dataset("data/processed/pathdsp_dataset.npz")
    ds = DrugResponseDataset(
        chem=data["chem"][:10],
        dg=data["dg"][:10],
        cell_pathways=data["cell_pathways"][:10],
        flat=data["flat_features"][:10],
        targets=data["target"][:10],
        drugs=data["drugs"][:10],
        cells=data["cells"][:10]
    )
    assert len(ds) == 10
    item = ds[0]
    assert item["chem"].shape == torch.Size([256])
    assert item["dg"].shape == torch.Size([196])
    assert item["cell_pathways"].shape == torch.Size([196, 3])
    assert item["flat"].shape == torch.Size([1031])
    assert item["target"].shape == torch.Size([1])
