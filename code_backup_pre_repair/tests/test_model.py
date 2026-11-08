"""
Tests for Model Architectures, Loss Convergence, and Explainability.
"""

import pytest
import torch
import numpy as np
from src.model import PathDSP_PIU, DrugEncoder, CellPathwayEncoder, DrugPathwayGating, MultiplicativePathwayInteraction
from src.baseline import PathDSP_FNN


def get_dummy_batch(batch_size=8, device="cpu"):
    return {
        "chem": torch.randn(batch_size, 256, device=device),
        "dg": torch.randn(batch_size, 196, device=device),
        "cell_pathways": torch.randn(batch_size, 196, 3, device=device),
        "flat": torch.randn(batch_size, 1031, device=device),
        "target": torch.randn(batch_size, 1, device=device)
    }


def test_submodules_shapes():
    batch = get_dummy_batch(4)
    # DrugEncoder
    drug_enc = DrugEncoder()
    e_d = drug_enc(batch["chem"], batch["dg"])
    assert e_d.shape == (4, 128)

    # CellPathwayEncoder
    cell_enc = CellPathwayEncoder()
    h_cp, h_c = cell_enc(batch["cell_pathways"])
    assert h_cp.shape == (4, 196, 32)
    assert h_c.shape == (4, 32)

    # Gating
    gating = DrugPathwayGating()
    h_prime, gates = gating(e_d, h_cp, batch["dg"])
    assert h_prime.shape == (4, 196, 32)
    assert gates.shape == (4, 196)

    # Interaction
    interact = MultiplicativePathwayInteraction()
    z_d, alpha = interact(e_d, h_prime)
    assert z_d.shape == (4, 32)
    assert alpha.shape == (4, 196)


def test_full_model_forward():
    model = PathDSP_PIU()
    batch = get_dummy_batch(16)
    y_pred = model(batch)
    assert y_pred.shape == (16, 1)
    assert not torch.isnan(y_pred).any()


def test_baseline_fnn_forward():
    fnn = PathDSP_FNN()
    batch = get_dummy_batch(16)
    y_pred = fnn(batch)
    assert y_pred.shape == (16, 1)
    assert not torch.isnan(y_pred).any()


def test_gradient_step():
    model = PathDSP_PIU()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()
    batch = get_dummy_batch(8)

    # Step 1
    opt.zero_grad()
    loss1 = crit(model(batch), batch["target"])
    loss1.backward()
    opt.step()

    # Step 2
    loss2 = crit(model(batch), batch["target"])
    assert loss2.item() < loss1.item(), "Single batch optimization step failed to decrease loss."


def test_explainability():
    model = PathDSP_PIU()
    batch = get_dummy_batch(4)
    expl = model.get_explainability(batch)
    assert "gating_scores" in expl
    assert "pathway_weights" in expl
    assert expl["gating_scores"].shape == (4, 196)
    assert expl["pathway_weights"].shape == (4, 196)
