"""
Evaluation Engine for PathDSP-PIU:
- Standard regression metrics (MAE, RMSE, R2, Pearson r, Spearman rho)
- Observation-level paired bootstrap analysis
- Clustered / Hierarchical bootstrap analysis (drug-clustered, cell-clustered, two-way clustered)
- Pooled and macro-averaged metrics per entity (per drug and per cell line)
"""

import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes all standard regression metrics on identical response scale.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Guard against zero variance in degenerate constant predictions
    if np.std(y_pred) < 1e-7 or np.std(y_true) < 1e-7:
        p_val, s_val = 0.0, 0.0
    else:
        try:
            p_val, _ = pearsonr(y_true, y_pred)
            p_val = float(p_val) if not np.isnan(p_val) else 0.0
        except Exception:
            p_val = 0.0

        try:
            s_val, _ = spearmanr(y_true, y_pred)
            s_val = float(s_val) if not np.isnan(s_val) else 0.0
        except Exception:
            s_val = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": p_val,
        "spearman": s_val,
        "n_samples": int(len(y_true))
    }


def compute_observation_paired_bootstrap(y_true: np.ndarray,
                                         y_pred_base: np.ndarray,
                                         y_pred_prop: np.ndarray,
                                         n_boot: int = 1000,
                                         alpha: float = 0.05,
                                         seed: int = 42) -> dict:
    """
    Observation-level paired bootstrap:
    Resampling unit: Individual response observations (i.i.d. observation pairs).
    Evaluates:
      Delta MAE = MAE_base - MAE_prop (positive indicates proposed model improvement)
      Delta RMSE = RMSE_base - RMSE_prop
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred_base = np.asarray(y_pred_base)
    y_pred_prop = np.asarray(y_pred_prop)

    # Observed point estimates
    obs_mae_base = np.mean(np.abs(y_true - y_pred_base))
    obs_mae_prop = np.mean(np.abs(y_true - y_pred_prop))
    observed_delta_mae = float(obs_mae_base - obs_mae_prop)

    obs_rmse_base = np.sqrt(np.mean((y_true - y_pred_base) ** 2))
    obs_rmse_prop = np.sqrt(np.mean((y_true - y_pred_prop) ** 2))
    observed_delta_rmse = float(obs_rmse_base - obs_rmse_prop)

    delta_maes = []
    delta_rmses = []

    for _ in range(n_boot):
        sample_idx = rng.randint(0, n, size=n)
        yt = y_true[sample_idx]
        pb = y_pred_base[sample_idx]
        pp = y_pred_prop[sample_idx]

        delta_maes.append(np.mean(np.abs(yt - pb)) - np.mean(np.abs(yt - pp)))
        delta_rmses.append(np.sqrt(np.mean((yt - pb) ** 2)) - np.sqrt(np.mean((yt - pp) ** 2)))

    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    return {
        "analysis_type": "observation_level_bootstrap",
        "resampling_unit": "observation_pair",
        "n_boot": n_boot,
        "n_samples": n,
        "observed_delta_mae": observed_delta_mae,
        "bootstrap_mean_delta_mae": float(np.mean(delta_maes)),
        "delta_mae_ci_lower": float(np.percentile(delta_maes, lower_pct)),
        "delta_mae_ci_upper": float(np.percentile(delta_maes, upper_pct)),
        "observed_delta_rmse": observed_delta_rmse,
        "bootstrap_mean_delta_rmse": float(np.mean(delta_rmses)),
        "delta_rmse_ci_lower": float(np.percentile(delta_rmses, lower_pct)),
        "delta_rmse_ci_upper": float(np.percentile(delta_rmses, upper_pct)),
        # Deprecated key aliases for backward compatibility
        "delta_mae_mean": float(np.mean(delta_maes)),
        "delta_rmse_mean": float(np.mean(delta_rmses))
    }


# Alias for backward compatibility
compute_paired_bootstrap = compute_observation_paired_bootstrap


def compute_clustered_bootstrap(y_true: np.ndarray,
                                y_pred_base: np.ndarray,
                                y_pred_prop: np.ndarray,
                                clusters: np.ndarray,
                                cluster_level: str = "drug",
                                n_boot: int = 1000,
                                alpha: float = 0.05,
                                seed: int = 42) -> dict:
    """
    Cluster-level bootstrap:
    Resampling unit: Entire unique entities (e.g. unique test drugs or cell lines).
    For each bootstrap replicate, unique clusters are resampled with replacement,
    and all observations belonging to the selected clusters are pooled.
    This accounts for correlated residuals within drugs or cell lines.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred_base = np.asarray(y_pred_base)
    y_pred_prop = np.asarray(y_pred_prop)
    clusters = np.asarray(clusters)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError(f"Need at least 2 unique clusters for clustered bootstrap, got {n_clusters}")

    # Map cluster -> observation indices
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    # Observed point estimates
    observed_delta_mae = float(np.mean(np.abs(y_true - y_pred_base)) - np.mean(np.abs(y_true - y_pred_prop)))
    observed_delta_rmse = float(np.sqrt(np.mean((y_true - y_pred_base) ** 2)) - np.sqrt(np.mean((y_true - y_pred_prop) ** 2)))

    delta_maes = []
    delta_rmses = []

    for _ in range(n_boot):
        boot_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_indices = np.concatenate([cluster_indices[c] for c in boot_clusters])

        yt = y_true[sample_indices]
        pb = y_pred_base[sample_indices]
        pp = y_pred_prop[sample_indices]

        delta_maes.append(np.mean(np.abs(yt - pb)) - np.mean(np.abs(yt - pp)))
        delta_rmses.append(np.sqrt(np.mean((yt - pb) ** 2)) - np.sqrt(np.mean((yt - pp) ** 2)))

    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    return {
        "analysis_type": "clustered_bootstrap",
        "resampling_unit": str(cluster_level),
        "n_clusters": int(n_clusters),
        "n_observations": int(len(y_true)),
        "n_boot": n_boot,
        "observed_delta_mae": observed_delta_mae,
        "bootstrap_mean_delta_mae": float(np.mean(delta_maes)),
        "delta_mae_ci_lower": float(np.percentile(delta_maes, lower_pct)),
        "delta_mae_ci_upper": float(np.percentile(delta_maes, upper_pct)),
        "observed_delta_rmse": observed_delta_rmse,
        "bootstrap_mean_delta_rmse": float(np.mean(delta_rmses)),
        "delta_rmse_ci_lower": float(np.percentile(delta_rmses, lower_pct)),
        "delta_rmse_ci_upper": float(np.percentile(delta_rmses, upper_pct))
    }


def compute_two_way_clustered_bootstrap(y_true: np.ndarray,
                                        y_pred_base: np.ndarray,
                                        y_pred_prop: np.ndarray,
                                        drugs: np.ndarray,
                                        cells: np.ndarray,
                                        n_boot: int = 1000,
                                        alpha: float = 0.05,
                                        seed: int = 42) -> dict:
    """
    Two-way clustered bootstrap for Double-Cold-Start comparisons:
    Resamples unique test drugs AND unique test cell lines with replacement.
    Evaluates pairs in the cross-product of resampled entities.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred_base = np.asarray(y_pred_base)
    y_pred_prop = np.asarray(y_pred_prop)
    drugs = np.asarray(drugs)
    cells = np.asarray(cells)

    unique_drugs = np.unique(drugs)
    unique_cells = np.unique(cells)

    observed_delta_mae = float(np.mean(np.abs(y_true - y_pred_base)) - np.mean(np.abs(y_true - y_pred_prop)))
    observed_delta_rmse = float(np.sqrt(np.mean((y_true - y_pred_base) ** 2)) - np.sqrt(np.mean((y_true - y_pred_prop) ** 2)))

    delta_maes = []
    delta_rmses = []

    for _ in range(n_boot):
        boot_d = set(rng.choice(unique_drugs, size=len(unique_drugs), replace=True))
        boot_c = set(rng.choice(unique_cells, size=len(unique_cells), replace=True))

        mask = np.isin(drugs, list(boot_d)) & np.isin(cells, list(boot_c))
        if np.sum(mask) < 5:
            continue

        yt = y_true[mask]
        pb = y_pred_base[mask]
        pp = y_pred_prop[mask]

        delta_maes.append(np.mean(np.abs(yt - pb)) - np.mean(np.abs(yt - pp)))
        delta_rmses.append(np.sqrt(np.mean((yt - pb) ** 2)) - np.sqrt(np.mean((yt - pp) ** 2)))

    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    return {
        "analysis_type": "two_way_clustered_bootstrap",
        "resampling_units": ["drug", "cell"],
        "n_unique_drugs": int(len(unique_drugs)),
        "n_unique_cells": int(len(unique_cells)),
        "n_observations": int(len(y_true)),
        "n_boot_valid": int(len(delta_maes)),
        "observed_delta_mae": observed_delta_mae,
        "bootstrap_mean_delta_mae": float(np.mean(delta_maes)) if delta_maes else 0.0,
        "delta_mae_ci_lower": float(np.percentile(delta_maes, lower_pct)) if delta_maes else 0.0,
        "delta_mae_ci_upper": float(np.percentile(delta_maes, upper_pct)) if delta_maes else 0.0,
        "observed_delta_rmse": observed_delta_rmse,
        "bootstrap_mean_delta_rmse": float(np.mean(delta_rmses)) if delta_rmses else 0.0,
        "delta_rmse_ci_lower": float(np.percentile(delta_rmses, lower_pct)) if delta_rmses else 0.0,
        "delta_rmse_ci_upper": float(np.percentile(delta_rmses, upper_pct)) if delta_rmses else 0.0
    }


def compute_macro_averaged_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   entities: np.ndarray,
                                   entity_name: str = "drug",
                                   min_samples: int = 3) -> dict:
    """
    Computes both pooled metrics and macro-averaged metrics stratified across entities.
    Macro-averaging calculates metrics independently per entity and computes mean/std across entities.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    entities = np.asarray(entities)

    pooled_metrics = compute_metrics(y_true, y_pred)

    unique_ents = np.unique(entities)
    per_entity_records = []

    entity_maes = []
    entity_rmses = []
    entity_pearsons = []
    entity_spearmans = []

    for ent in unique_ents:
        mask = (entities == ent)
        if np.sum(mask) >= min_samples:
            yt = y_true[mask]
            yp = y_pred[mask]
            m = compute_metrics(yt, yp)
            m["entity"] = str(ent)
            per_entity_records.append(m)

            entity_maes.append(m["mae"])
            entity_rmses.append(m["rmse"])
            entity_pearsons.append(m["pearson"])
            entity_spearmans.append(m["spearman"])

    return {
        "pooled": pooled_metrics,
        "macro": {
            "entity_type": entity_name,
            "n_entities_evaluated": len(entity_maes),
            "n_entities_total": len(unique_ents),
            "macro_mae_mean": float(np.mean(entity_maes)) if entity_maes else 0.0,
            "macro_mae_std": float(np.std(entity_maes)) if entity_maes else 0.0,
            "macro_rmse_mean": float(np.mean(entity_rmses)) if entity_rmses else 0.0,
            "macro_rmse_std": float(np.std(entity_rmses)) if entity_rmses else 0.0,
            "macro_pearson_mean": float(np.mean(entity_pearsons)) if entity_pearsons else 0.0,
            "macro_pearson_std": float(np.std(entity_pearsons)) if entity_pearsons else 0.0,
            "macro_spearman_mean": float(np.mean(entity_spearmans)) if entity_spearmans else 0.0,
            "macro_spearman_std": float(np.std(entity_spearmans)) if entity_spearmans else 0.0
        },
        "per_entity_details": per_entity_records
    }
