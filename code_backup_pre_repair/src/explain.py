"""
Explainability and Pathway Attribution for PathDSP-PIU:
- Extracts drug-conditioned gating scores g(d, p)
- Extracts pathway interaction attention weights alpha(d, p)
- Computes global and drug-specific pathway importance rankings
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def compute_pathway_attributions(model, dataloader, pathway_names: list, device: torch.device):
    """
    Computes average gating and attention importance per drug across all cell lines.
    """
    model.to(device)
    model.eval()

    drug_gates = {}
    drug_attns = {}

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            expl = model.get_explainability(batch_dev)
            gates = expl["gating_scores"]  # (B, P)
            attns = expl["pathway_weights"]  # (B, P)
            drugs = batch["drug"]

            for i, d in enumerate(drugs):
                if d not in drug_gates:
                    drug_gates[d] = []
                    drug_attns[d] = []
                drug_gates[d].append(gates[i])
                drug_attns[d].append(attns[i])

    # Aggregate per drug
    drug_importance = {}
    for d in drug_gates:
        avg_gate = np.mean(drug_gates[d], axis=0)
        avg_attn = np.mean(drug_attns[d], axis=0)
        combined_score = avg_gate * avg_attn

        top_indices = np.argsort(combined_score)[::-1]
        ranked_pathways = [
            {
                "rank": rank + 1,
                "pathway": pathway_names[idx],
                "score": float(combined_score[idx]),
                "gate": float(avg_gate[idx]),
                "attention": float(avg_attn[idx])
            }
            for rank, idx in enumerate(top_indices[:20])
        ]
        drug_importance[d] = ranked_pathways

    # Also compute global pathway importance across all drugs
    all_gates = np.concatenate([np.array(v) for v in drug_gates.values()], axis=0)
    all_attns = np.concatenate([np.array(v) for v in drug_attns.values()], axis=0)
    global_score = np.mean(all_gates * all_attns, axis=0)
    global_top = np.argsort(global_score)[::-1]

    global_ranked = [
        {
            "rank": rank + 1,
            "pathway": pathway_names[idx],
            "score": float(global_score[idx]),
            "gate": float(np.mean(all_gates[:, idx])),
            "attention": float(np.mean(all_attns[:, idx]))
        }
        for rank, idx in enumerate(global_top[:30])
    ]

    return {
        "global_importance": global_ranked,
        "drug_specific_importance": drug_importance
    }
