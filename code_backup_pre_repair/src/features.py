"""
Feature schema, pathway catalog alignment, and tensor construction for PathDSP-PIU.
Handles:
- 256 CHEM bits (Morgan fingerprints)
- 196 DG-Net pathways (TARGETnNEIGHBOR_)
- 187 EXP pathways (ssGSEA, aligned to 196 reference pathways with zero-padding)
- 196 MUT-Net pathways (MUT_)
- 196 CNV-Net pathways (CNV_)
- Extraction of cell pathway matrix X_c in R^(P x M) with P=196, M=3
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reference 9 pathways missing in ssGSEA EXP features in Zenodo record 7532963
MISSING_EXP_PATHWAYS = [
    "PID_ALK2_PATHWAY",
    "PID_IL5_PATHWAY",
    "PID_INTEGRIN4_PATHWAY",
    "PID_P38_GAMMA_DELTA_PATHWAY",
    "PID_RANBP2_PATHWAY",
    "PID_S1P_S1P4_PATHWAY",
    "PID_TCR_JNK_PATHWAY",
    "PID_TCR_RAS_PATHWAY",
    "PID_VEGF_VEGFR_PATHWAY"
]


class FeatureRegistry:
    """
    Manages column indices, pathway alignment catalog, and feature group partitioning.
    """
    def __init__(self, manifest_dir: str = "data/manifests"):
        self.manifest_dir = manifest_dir
        self.alignment_file = os.path.join(manifest_dir, "pathway_alignment.json")
        self.pathway_catalog = []
        self.pathway_names = []
        self.chem_cols = []
        self.dg_cols = []
        self.exp_cols = []
        self.mut_cols = []
        self.cnv_cols = []
        self.all_feature_cols = []
        
        self._load_or_build_alignment()

    def _load_or_build_alignment(self):
        if os.path.exists(self.alignment_file):
            with open(self.alignment_file, "r") as f:
                data = json.load(f)
            self.pathway_catalog = data.get("pathway_catalog", [])
            self.pathway_names = [p["pathway_name"] for p in self.pathway_catalog]
            logger.info(f"Loaded pathway catalog with {len(self.pathway_catalog)} pathways from {self.alignment_file}")
        else:
            logger.warning(f"Alignment file {self.alignment_file} not found; will build dynamically on data inspection.")

    def inspect_and_build(self, df: pd.DataFrame, save_metadata_path: str = None):
        """
        Inspect DataFrame columns and construct deterministic feature indices.
        """
        cols = list(df.columns)
        self.chem_cols = [c for c in cols if c.startswith("CHEM_mBit_")]
        self.dg_cols = [c for c in cols if c.startswith("TARGETnNEIGHBOR_")]
        self.exp_cols = [c for c in cols if c.startswith("EXP_")]
        self.mut_cols = [c for c in cols if c.startswith("MUT_")]
        self.cnv_cols = [c for c in cols if c.startswith("CNV_")]

        # Determine reference 196 pathways from DG-Net (or MUT-Net / CNV-Net)
        ref_pathways = [c.replace("TARGETnNEIGHBOR_", "") for c in self.dg_cols]
        self.pathway_names = ref_pathways
        self.pathway_catalog = []

        exp_set = {c.replace("EXP_", "") for c in self.exp_cols}

        for idx, pw in enumerate(ref_pathways):
            in_exp = pw in exp_set
            col_exp = f"EXP_{pw}" if in_exp else None
            idx_exp = self.exp_cols.index(col_exp) if in_exp else -1

            entry = {
                "pathway_name": pw,
                "index_in_196": idx,
                "col_dg": f"TARGETnNEIGHBOR_{pw}",
                "col_mut": f"MUT_{pw}",
                "col_cnv": f"CNV_{pw}",
                "in_exp": in_exp,
                "col_exp": col_exp,
                "index_in_exp": idx_exp
            }
            self.pathway_catalog.append(entry)

        self.all_feature_cols = self.chem_cols + self.dg_cols + self.exp_cols + self.mut_cols + self.cnv_cols
        
        if save_metadata_path:
            os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)
            with open(save_metadata_path, "w") as f:
                json.dump({
                    "n_total_features": len(self.all_feature_cols),
                    "n_chem": len(self.chem_cols),
                    "n_dg": len(self.dg_cols),
                    "n_exp": len(self.exp_cols),
                    "n_mut": len(self.mut_cols),
                    "n_cnv": len(self.cnv_cols),
                    "n_reference_pathways": len(self.pathway_catalog),
                    "pathway_catalog": self.pathway_catalog
                }, f, indent=2)
            logger.info(f"Saved pathway metadata to {save_metadata_path}")

    def extract_structured_arrays(self, df: pd.DataFrame):
        """
        Extracts structured numpy arrays:
        - chem: (N, 256) float32
        - dg: (N, 196) float32
        - cell_pathways: (N, 196, 3) float32 with modalities [EXP, MUT, CNV]
        - flat_features: (N, 1031) float32
        - target: (N,) float32
        """
        N = len(df)
        P = len(self.pathway_catalog)
        M = 3  # EXP, MUT, CNV

        chem = df[self.chem_cols].values.astype(np.float32)
        dg = df[self.dg_cols].values.astype(np.float32)
        mut = df[self.mut_cols].values.astype(np.float32)
        cnv = df[self.cnv_cols].values.astype(np.float32)

        # Assemble aligned cell_pathways matrix (N, P, 3)
        cell_pathways = np.zeros((N, P, M), dtype=np.float32)

        for p_idx, entry in enumerate(self.pathway_catalog):
            # Modality 0: EXP (fill 0 if missing in EXP)
            if entry["in_exp"]:
                cell_pathways[:, p_idx, 0] = df[entry["col_exp"]].values.astype(np.float32)
            else:
                cell_pathways[:, p_idx, 0] = 0.0

            # Modality 1: MUT
            cell_pathways[:, p_idx, 1] = df[entry["col_mut"]].values.astype(np.float32)

            # Modality 2: CNV
            cell_pathways[:, p_idx, 2] = df[entry["col_cnv"]].values.astype(np.float32)

        flat_features = df[self.all_feature_cols].values.astype(np.float32)
        target = df["resp"].values.astype(np.float32) if "resp" in df.columns else None

        return {
            "chem": chem,
            "dg": dg,
            "cell_pathways": cell_pathways,
            "flat_features": flat_features,
            "target": target,
            "drugs": df["drug"].values,
            "cells": df["cell"].values
        }
