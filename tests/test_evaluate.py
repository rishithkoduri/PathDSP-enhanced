"""
Tests for Evaluation Metrics and Statistical Bootstrap Engines.
"""

import pytest
import numpy as np
from src.evaluate import (
    compute_metrics,
    compute_observation_paired_bootstrap,
    compute_clustered_bootstrap,
    compute_two_way_clustered_bootstrap,
    compute_macro_averaged_metrics
)


def test_compute_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    m = compute_metrics(y_true, y_pred)
    assert "mae" in m and "rmse" in m and "pearson" in m and "spearman" in m
    assert m["mae"] < 0.2
    assert m["pearson"] > 0.95


def test_observation_paired_bootstrap():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_base = y_true + rng.randn(100) * 0.5
    y_prop = y_true + rng.randn(100) * 0.2  # Proposed is better

    res = compute_observation_paired_bootstrap(y_true, y_base, y_prop, n_boot=200, seed=42)
    assert res["observed_delta_mae"] > 0
    assert res["delta_mae_ci_lower"] < res["observed_delta_mae"] < res["delta_mae_ci_upper"]


def test_clustered_bootstrap():
    rng = np.random.RandomState(42)
    clusters = np.array(["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25)
    y_true = rng.randn(100)
    y_base = y_true + 0.5
    y_prop = y_true + 0.1

    res = compute_clustered_bootstrap(y_true, y_base, y_prop, clusters, cluster_level="drug", n_boot=100, seed=42)
    assert res["n_clusters"] == 4
    assert res["observed_delta_mae"] > 0
    assert "delta_mae_ci_lower" in res


def test_two_way_clustered_bootstrap_multiplicity():
    """
    Verify two-way pigeonhole bootstrap preserves multiplicity:
    Construct synthetic dataset with 3 drugs and 3 cells.
    """
    drugs = np.array(["D1", "D1", "D2", "D2", "D3", "D3"])
    cells = np.array(["C1", "C2", "C1", "C2", "C1", "C2"])
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_base = y_true + 1.0  # MAE = 1.0
    y_prop = y_true + 0.2  # MAE = 0.2, delta MAE = 0.8

    res = compute_two_way_clustered_bootstrap(y_true, y_base, y_prop, drugs, cells, n_boot=200, seed=42)
    assert res["analysis_type"] == "two_way_clustered_bootstrap"
    assert res["resampling_method"] == "two_way_pigeonhole_bootstrap"
    assert res["n_unique_drugs"] == 3
    assert res["n_unique_cells"] == 2
    assert np.isclose(res["observed_delta_mae"], 0.8, atol=1e-5)
    assert res["n_boot_valid"] > 0
    # Proposed model is strictly better, CI lower bound should be > 0
    assert res["delta_mae_ci_lower"] > 0.5


def test_macro_averaged_metrics():
    drugs = np.array(["D1", "D1", "D1", "D2", "D2", "D2"])
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.2, 5.2, 6.2])

    res = compute_macro_averaged_metrics(y_true, y_pred, drugs, entity_name="drug", min_samples=3)
    assert "pooled" in res and "macro" in res
    assert res["macro"]["n_entities_evaluated"] == 2
    assert np.isclose(res["macro"]["macro_mae_mean"], 0.15, atol=1e-4)
