"""
Uncertainty Estimation and Conformal Calibration Module for PathDSP-PIU:
- Deep Ensemble aggregation (Mean, Variance, Std)
- Split Conformal Prediction (standard & locally-adaptive prediction intervals)
- Uncertainty vs error correlation and calibration assessment
"""

import logging
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Split conformal prediction calibrator.
    Fits nonconformity quantiles on calibration (validation) set and evaluates coverage on test set.
    """
    def __init__(self, adaptive: bool = True):
        self.adaptive = adaptive
        self.quantiles = {}
        self.eps = 1e-4

    def calibrate(self, y_cal: np.ndarray, mu_cal: np.ndarray, sigma_cal: np.ndarray = None, alphas: list = [0.10, 0.05]):
        """
        Calibrates on held-out validation data.
        alphas: list of miscoverage rates (e.g. 0.10 -> 90% confidence, 0.05 -> 95% confidence).
        """
        n = len(y_cal)
        abs_res = np.abs(y_cal - mu_cal)

        if self.adaptive and sigma_cal is not None:
            scores = abs_res / (sigma_cal + self.eps)
        else:
            scores = abs_res

        self.quantiles = {}
        for alpha in alphas:
            q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
            q_val = float(np.quantile(scores, q_level))
            self.quantiles[alpha] = q_val
            logger.info(f"Conformal calibrated at alpha={alpha:.2f} ({(1-alpha)*100:.0f}% target): q={q_val:.4f}")

    def predict_intervals(self, mu_test: np.ndarray, sigma_test: np.ndarray = None, alpha: float = 0.10):
        if alpha not in self.quantiles:
            raise ValueError(f"Alpha {alpha} not calibrated.")
        q = self.quantiles[alpha]

        if self.adaptive and sigma_test is not None:
            half_width = q * (sigma_test + self.eps)
        else:
            half_width = q

        lower = mu_test - half_width
        upper = mu_test + half_width
        return lower, upper, (upper - lower)

    def evaluate_coverage(self, y_true: np.ndarray, mu_test: np.ndarray, sigma_test: np.ndarray = None, alpha: float = 0.10):
        lower, upper, width = self.predict_intervals(mu_test, sigma_test, alpha)
        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(np.mean(covered))
        mean_width = float(np.mean(width))
        return {
            "target_coverage": 1.0 - alpha,
            "empirical_coverage": empirical_coverage,
            "mean_interval_width": mean_width,
            "coverage_gap": empirical_coverage - (1.0 - alpha)
        }


class PathDSPEnsemble:
    """
    Manages an ensemble of PathDSP-PIU models for uncertainty estimation.
    """
    def __init__(self, models: list):
        self.models = models
        self.k = len(models)

    def predict(self, dataloader, device: torch.device):
        for m in self.models:
            m.to(device)
            m.eval()

        all_preds = [[] for _ in range(self.k)]
        all_targets = []
        all_drugs = []
        all_cells = []

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                for idx, m in enumerate(self.models):
                    pred = m(batch_dev).squeeze(-1).cpu().numpy()
                    all_preds[idx].append(pred)

                if "target" in batch:
                    all_targets.append(batch["target"].squeeze(-1).numpy())
                if "drug" in batch:
                    all_drugs.extend(batch["drug"])
                if "cell" in batch:
                    all_cells.extend(batch["cell"])

        # Stack into (N, K)
        individual_preds = np.column_stack([np.concatenate(p) for p in all_preds])
        mean_pred = np.mean(individual_preds, axis=1)
        var_pred = np.var(individual_preds, axis=1, ddof=1) if self.k > 1 else np.zeros_like(mean_pred)
        std_pred = np.sqrt(var_pred)

        results = {
            "individual_predictions": individual_preds,
            "mean": mean_pred,
            "variance": var_pred,
            "std": std_pred,
            "drugs": np.array(all_drugs) if all_drugs else None,
            "cells": np.array(all_cells) if all_cells else None
        }
        if all_targets:
            results["targets"] = np.concatenate(all_targets)
            results["abs_error"] = np.abs(results["targets"] - mean_pred)

            # Error vs uncertainty correlation
            if self.k > 1:
                p_corr, _ = pearsonr(std_pred, results["abs_error"])
                s_corr, _ = spearmanr(std_pred, results["abs_error"])
                results["error_unc_pearson"] = float(p_corr)
                results["error_unc_spearman"] = float(s_corr)
            else:
                results["error_unc_pearson"] = 0.0
                results["error_unc_spearman"] = 0.0

        return results
