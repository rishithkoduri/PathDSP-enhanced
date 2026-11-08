"""
Baseline models for benchmark comparison:
1. PathDSP_FNN: Faithful original 5-layer feedforward architecture
2. DrugMeanPredictor: Mean response per drug
3. CellMeanPredictor: Mean response per cell line
4. AdditiveMeanPredictor: Additive two-way main effects (global + drug + cell)
5. ElasticNet
6. RandomForest
7. XGBoost
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class PathDSP_FNN(nn.Module):
    """
    Original PathDSP architecture (Tang & Gottlieb 2021 / Li et al. 2019):
    5-layer fully-connected neural network with ELU activations and Dropout(0.1).
    Input dimension: 1,031 features.
    """
    def __init__(self, input_dim: int = 1031, dropout: float = 0.1):
        super(PathDSP_FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 800),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(800, 500),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(500, 100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        if isinstance(x, dict):
            # If passed batch dict, extract flat features
            x = x["flat"]
        return self.net(x)


class DrugMeanPredictor:
    """Predicts historical mean response of each drug; falls back to global mean for unseen drugs."""
    def __init__(self):
        self.global_mean = 0.0
        self.drug_means = {}

    def fit(self, drugs: np.ndarray, y: np.ndarray):
        self.global_mean = float(np.mean(y))
        for d in np.unique(drugs):
            self.drug_means[d] = float(np.mean(y[drugs == d]))
        return self

    def predict(self, drugs: np.ndarray) -> np.ndarray:
        return np.array([self.drug_means.get(d, self.global_mean) for d in drugs], dtype=np.float32)


class CellMeanPredictor:
    """Predicts historical mean response of each cell line; falls back to global mean for unseen cell lines."""
    def __init__(self):
        self.global_mean = 0.0
        self.cell_means = {}

    def fit(self, cells: np.ndarray, y: np.ndarray):
        self.global_mean = float(np.mean(y))
        for c in np.unique(cells):
            self.cell_means[c] = float(np.mean(y[cells == c]))
        return self

    def predict(self, cells: np.ndarray) -> np.ndarray:
        return np.array([self.cell_means.get(c, self.global_mean) for c in cells], dtype=np.float32)


class AdditiveMeanPredictor:
    """
    Two-way additive main-effects predictor:
    y_pred = global_mean + (drug_mean - global_mean) + (cell_mean - global_mean)
    """
    def __init__(self):
        self.global_mean = 0.0
        self.drug_effects = {}
        self.cell_effects = {}

    def fit(self, drugs: np.ndarray, cells: np.ndarray, y: np.ndarray):
        self.global_mean = float(np.mean(y))
        for d in np.unique(drugs):
            self.drug_effects[d] = float(np.mean(y[drugs == d])) - self.global_mean
        for c in np.unique(cells):
            self.cell_effects[c] = float(np.mean(y[cells == c])) - self.global_mean
        return self

    def predict(self, drugs: np.ndarray, cells: np.ndarray) -> np.ndarray:
        preds = []
        for d, c in zip(drugs, cells):
            d_eff = self.drug_effects.get(d, 0.0)
            c_eff = self.cell_effects.get(c, 0.0)
            preds.append(self.global_mean + d_eff + c_eff)
        return np.array(preds, dtype=np.float32)


def get_tabular_baseline(name: str, config: dict = None):
    """
    Factory for tabular machine learning baselines.
    """
    config = config or {}
    if name == "elastic_net":
        return ElasticNet(
            alpha=config.get("alpha", 0.01),
            l1_ratio=config.get("l1_ratio", 0.5),
            max_iter=config.get("max_iter", 1000),
            random_state=config.get("random_state", 42)
        )
    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 16),
            n_jobs=config.get("n_jobs", -1),
            random_state=config.get("random_state", 42)
        )
    elif name == "xgboost":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost library is not installed in the current environment.")
        return xgb.XGBRegressor(
            n_estimators=config.get("n_estimators", 150),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            n_jobs=config.get("n_jobs", -1),
            tree_method=config.get("tree_method", "hist"),
            random_state=config.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unknown baseline name: {name}")
