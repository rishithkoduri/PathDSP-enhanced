"""
Dataset loader, leakage-free preprocessing, and PyTorch Dataset/DataLoader for PathDSP-PIU.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.features import FeatureRegistry

logger = logging.getLogger(__name__)


class DrugResponseDataset(Dataset):
    """
    PyTorch Dataset providing structured inputs for PathDSP-PIU:
    - chem: (256,) Morgan fingerprint bits
    - dg: (196,) Drug target network pathway enrichment
    - cell_pathways: (196, 3) Cell multi-omics pathway matrix [EXP, MUT, CNV]
    - flat: (1031,) Full feature vector for baseline FNN and residual connection
    - target: (1,) Response target (-log10(IC50))
    - drug: Drug name string
    - cell: Cell line ID string
    """
    def __init__(self, chem, dg, cell_pathways, flat, targets=None, drugs=None, cells=None):
        self.chem = torch.from_numpy(chem).float()
        self.dg = torch.from_numpy(dg).float()
        self.cell_pathways = torch.from_numpy(cell_pathways).float()
        self.flat = torch.from_numpy(flat).float()
        self.targets = torch.from_numpy(targets).float().unsqueeze(1) if targets is not None else None
        self.drugs = drugs
        self.cells = cells

    def __len__(self):
        return len(self.chem)

    def __getitem__(self, idx):
        item = {
            "chem": self.chem[idx],
            "dg": self.dg[idx],
            "cell_pathways": self.cell_pathways[idx],
            "flat": self.flat[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.drugs is not None:
            item["drug"] = self.drugs[idx]
        if self.cells is not None:
            item["cell"] = self.cells[idx]
        return item


class DataPreprocessor:
    """
    Leakage-safe preprocessor. Fits standardizers strictly on training data
    and transforms validation, test, or external evaluation data.
    """
    def __init__(self):
        self.flat_scaler = StandardScaler()
        self.dg_scaler = StandardScaler()
        # Modality-specific scalers for cell pathways
        self.exp_scaler = StandardScaler()
        self.mut_scaler = StandardScaler()
        self.cnv_scaler = StandardScaler()
        self.fitted = False

    def fit(self, structured_data: dict, train_indices: np.ndarray):
        """
        Fit scalers strictly on training indices.
        """
        flat_tr = structured_data["flat_features"][train_indices]
        self.flat_scaler.fit(flat_tr)

        dg_tr = structured_data["dg"][train_indices]
        self.dg_scaler.fit(dg_tr)

        # Modality 0: EXP (only fit on non-zero / observed entries if desired, or standard fit)
        exp_tr = structured_data["cell_pathways"][train_indices, :, 0]
        self.exp_scaler.fit(exp_tr)

        # Modality 1: MUT
        mut_tr = structured_data["cell_pathways"][train_indices, :, 1]
        self.mut_scaler.fit(mut_tr)

        # Modality 2: CNV
        cnv_tr = structured_data["cell_pathways"][train_indices, :, 2]
        self.cnv_scaler.fit(cnv_tr)

        self.fitted = True
        return self

    def transform(self, structured_data: dict, indices: np.ndarray = None):
        """
        Transform a subset or all data using training-fitted scalers.
        """
        if not self.fitted:
            raise RuntimeError("DataPreprocessor must be fitted on training set before transforming.")

        if indices is not None:
            chem = structured_data["chem"][indices].copy()
            dg = structured_data["dg"][indices].copy()
            cp = structured_data["cell_pathways"][indices].copy()
            flat = structured_data["flat_features"][indices].copy()
            target = structured_data["target"][indices].copy() if structured_data["target"] is not None else None
            drugs = structured_data["drugs"][indices] if structured_data["drugs"] is not None else None
            cells = structured_data["cells"][indices] if structured_data["cells"] is not None else None
        else:
            chem = structured_data["chem"].copy()
            dg = structured_data["dg"].copy()
            cp = structured_data["cell_pathways"].copy()
            flat = structured_data["flat_features"].copy()
            target = structured_data["target"].copy() if structured_data["target"] is not None else None
            drugs = structured_data["drugs"]
            cells = structured_data["cells"]

        # Transform flat and dg
        flat_scaled = self.flat_scaler.transform(flat).astype(np.float32)
        dg_scaled = self.dg_scaler.transform(dg).astype(np.float32)

        # Transform cell pathways per modality
        cp_scaled = cp.copy()
        cp_scaled[:, :, 0] = self.exp_scaler.transform(cp[:, :, 0]).astype(np.float32)
        cp_scaled[:, :, 1] = self.mut_scaler.transform(cp[:, :, 1]).astype(np.float32)
        cp_scaled[:, :, 2] = self.cnv_scaler.transform(cp[:, :, 2]).astype(np.float32)

        return {
            "chem": chem.astype(np.float32),  # Keep binary bits intact
            "dg": dg_scaled,
            "cell_pathways": cp_scaled,
            "flat_features": flat_scaled,
            "target": target,
            "drugs": drugs,
            "cells": cells
        }


def load_raw_dataset(raw_path: str = "data/raw/input.txt") -> pd.DataFrame:
    """
    Loads raw PathDSP dataset with validation.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, sep="\t")
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def prepare_and_cache_dataset(raw_path: str = "data/raw/input.txt",
                              cache_path: str = "data/processed/pathdsp_dataset.npz",
                              metadata_path: str = "data/processed/dataset_metadata.json",
                              pathway_meta_path: str = "data/processed/pathway_metadata.json"):
    """
    Parses raw TSV, builds pathway registry, and caches structured numpy arrays.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df = load_raw_dataset(raw_path)

    registry = FeatureRegistry(manifest_dir=os.path.dirname(metadata_path))
    registry.inspect_and_build(df, save_metadata_path=pathway_meta_path)
    structured = registry.extract_structured_arrays(df)

    logger.info(f"Saving structured dataset cache to {cache_path}...")
    np.savez_compressed(
        cache_path,
        chem=structured["chem"],
        dg=structured["dg"],
        cell_pathways=structured["cell_pathways"],
        flat_features=structured["flat_features"],
        target=structured["target"],
        drugs=np.array(structured["drugs"], dtype=str),
        cells=np.array(structured["cells"], dtype=str)
    )

    metadata = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_drugs": int(len(np.unique(structured["drugs"]))),
        "n_cells": int(len(np.unique(structured["cells"]))),
        "target_col": "resp",
        "target_mean": float(np.mean(structured["target"])),
        "target_std": float(np.std(structured["target"])),
        "target_min": float(np.min(structured["target"])),
        "target_max": float(np.max(structured["target"])),
        "cache_file": cache_path
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Cache preparation complete. Metadata saved to {metadata_path}")
    return structured, metadata


def load_cached_dataset(cache_path: str = "data/processed/pathdsp_dataset.npz"):
    """
    Fast loader for cached structured dataset.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found at: {cache_path}. Run scripts/prepare.py first.")
    
    data = np.load(cache_path, allow_pickle=True)
    return {
        "chem": data["chem"],
        "dg": data["dg"],
        "cell_pathways": data["cell_pathways"],
        "flat_features": data["flat_features"],
        "target": data["target"],
        "drugs": data["drugs"],
        "cells": data["cells"]
    }


def create_dataloaders(train_data: dict,
                       val_data: dict,
                       test_data: dict = None,
                       batch_size: int = 128,
                       num_workers: int = 0):
    """
    Helper to construct training, validation, and testing DataLoaders.
    """
    train_ds = DrugResponseDataset(
        chem=train_data["chem"],
        dg=train_data["dg"],
        cell_pathways=train_data["cell_pathways"],
        flat=train_data["flat_features"],
        targets=train_data["target"],
        drugs=train_data["drugs"],
        cells=train_data["cells"]
    )
    val_ds = DrugResponseDataset(
        chem=val_data["chem"],
        dg=val_data["dg"],
        cell_pathways=val_data["cell_pathways"],
        flat=val_data["flat_features"],
        targets=val_data["target"],
        drugs=val_data["drugs"],
        cells=val_data["cells"]
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = None
    if test_data is not None:
        test_ds = DrugResponseDataset(
            chem=test_data["chem"],
            dg=test_data["dg"],
            cell_pathways=test_data["cell_pathways"],
            flat=test_data["flat_features"],
            targets=test_data["target"],
            drugs=test_data["drugs"],
            cells=test_data["cells"]
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader, test_loader
