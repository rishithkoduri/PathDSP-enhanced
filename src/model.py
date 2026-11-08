"""
PathDSP-PIU: Drug-Conditioned Pathway Interaction Learning with Uncertainty-Aware Evaluation
Architecture Modules:
1. DrugEncoder: [CHEM, DG-Net] -> e_d
2. CellPathwayEncoder: X_c in R^(P x M) -> h(c, p) and h_c
3. DrugPathwayGating: g(d, p) = sigmoid(MLP([e_d, h(c, p), DGNet(d, p)]))
4. MultiplicativePathwayInteraction: z(d, p) = (U e_d) * (V h'(d, p)), aggregated via attention weights alpha(d, p)
5. RegressionHead: [e_d, h_c, z_d, residual] -> y_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEncoder(nn.Module):
    """Encodes 256-bit CHEM fingerprint and 196-dim DG-Net target enrichment into embedding e_d."""
    def __init__(self, chem_dim: int = 256, dg_dim: int = 196, hidden_dim: int = 256, embed_dim: int = 128, dropout: float = 0.15):
        super(DrugEncoder, self).__init__()
        in_dim = chem_dim + dg_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, chem: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([chem, dg], dim=-1)
        return self.net(x)


class CellPathwayEncoder(nn.Module):
    """
    Encodes cell-line pathway multi-omics matrix X_c in R^(B x P x M)
    where P=196 pathways and M=3 modalities (EXP, MUT-Net, CNV-Net).
    Outputs per-pathway representation h(c, p) in R^(B x P x d_h) and global cell vector h_c.
    """
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 64, embed_dim: int = 32, dropout: float = 0.10):
        super(CellPathwayEncoder, self).__init__()
        self.pathway_mlp = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, cell_pathways: torch.Tensor):
        # cell_pathways: (B, P, M)
        h_cp = self.pathway_mlp(cell_pathways)  # (B, P, embed_dim)
        h_c = torch.mean(h_cp, dim=1)           # (B, embed_dim)
        return h_cp, h_c


class DrugPathwayGating(nn.Module):
    """
    Drug-conditioned gating:
    g(d, p) = sigmoid(MLP([e_d, h(c, p), DGNet(d, p)]))
    h'(d, p) = g(d, p) * h(c, p)
    """
    def __init__(self, drug_dim: int = 128, pathway_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.10):
        super(DrugPathwayGating, self).__init__()
        in_dim = drug_dim + pathway_dim + 1  # 1 for specific DGNet(d, p) score
        self.gating_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, e_d: torch.Tensor, h_cp: torch.Tensor, dg: torch.Tensor):
        # e_d: (B, d_e)
        # h_cp: (B, P, d_h)
        # dg: (B, P)
        B, P, d_h = h_cp.shape
        e_d_exp = e_d.unsqueeze(1).expand(-1, P, -1)     # (B, P, d_e)
        dg_exp = dg.unsqueeze(-1)                        # (B, P, 1)

        gating_input = torch.cat([e_d_exp, h_cp, dg_exp], dim=-1) # (B, P, d_e + d_h + 1)
        gates = self.gating_mlp(gating_input)                      # (B, P, 1)
        h_prime = gates * h_cp                                     # (B, P, d_h)
        return h_prime, gates.squeeze(-1)                          # h_prime, (B, P)


class MultiplicativePathwayInteraction(nn.Module):
    """
    Low-rank multiplicative pathway interaction:
    z(d, p) = (U e_d) * (V h'(d, p))
    Aggregated interaction vector:
    z(d) = sum_p alpha(d, p) * z(d, p)
    """
    def __init__(self, drug_dim: int = 128, pathway_dim: int = 32, interaction_dim: int = 32, dropout: float = 0.10):
        super(MultiplicativePathwayInteraction, self).__init__()
        self.proj_u = nn.Linear(drug_dim, interaction_dim, bias=False)
        self.proj_v = nn.Linear(pathway_dim, interaction_dim, bias=False)
        
        # Attention scoring network for pathway weighting
        self.attn_net = nn.Sequential(
            nn.Linear(interaction_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_d: torch.Tensor, h_prime: torch.Tensor):
        # e_d: (B, d_e)
        # h_prime: (B, P, d_h)
        B, P, _ = h_prime.shape
        u = self.proj_u(e_d).unsqueeze(1)    # (B, 1, d_i)
        v = self.proj_v(h_prime)             # (B, P, d_i)

        z_dp = self.dropout(u * v)           # (B, P, d_i)

        # Compute normalized pathway attention weights
        u_expanded = u.expand(-1, P, -1)     # (B, P, d_i)
        attn_in = torch.cat([u_expanded, v], dim=-1) # (B, P, 2 * d_i)
        scores = self.attn_net(attn_in)      # (B, P, 1)
        alpha = F.softmax(scores, dim=1)     # (B, P, 1)

        z_d = torch.sum(alpha * z_dp, dim=1) # (B, d_i)
        return z_d, alpha.squeeze(-1)        # z_d, (B, P)


class PathDSP_PIU(nn.Module):
    """
    Full PathDSP-PIU Model Architecture.
    Combines:
    - DrugEncoder
    - CellPathwayEncoder
    - DrugPathwayGating
    - MultiplicativePathwayInteraction
    - Residual Connection
    - RegressionHead
    Supports controlled ablation flags:
    - use_gating: bool
    - use_interaction: bool
    - use_dg_net: bool
    - use_cell_modalities: tuple ('exp', 'mut', 'cnv')
    - use_residual: bool
    """
    def __init__(self,
                 chem_dim: int = 256,
                 dg_dim: int = 196,
                 drug_embed_dim: int = 128,
                 num_pathways: int = 196,
                 num_modalities: int = 3,
                 pathway_embed_dim: int = 32,
                 interaction_dim: int = 32,
                 dropout: float = 0.15,
                 use_gating: bool = True,
                 use_interaction: bool = True,
                 use_dg_net: bool = True,
                 use_residual: bool = True,
                 active_modalities: tuple = ("exp", "mut", "cnv")):
        super(PathDSP_PIU, self).__init__()
        self.use_gating = use_gating
        self.use_interaction = use_interaction
        self.use_dg_net = use_dg_net
        self.use_residual = use_residual
        self.active_modalities = active_modalities

        # Drug Encoder
        self.drug_encoder = DrugEncoder(
            chem_dim=chem_dim,
            dg_dim=dg_dim,
            hidden_dim=256,
            embed_dim=drug_embed_dim,
            dropout=dropout
        )

        # Cell Pathway Encoder
        self.cell_encoder = CellPathwayEncoder(
            num_modalities=num_modalities,
            hidden_dim=64,
            embed_dim=pathway_embed_dim,
            dropout=dropout
        )

        # Drug-Conditioned Pathway Gating
        if self.use_gating:
            self.gating = DrugPathwayGating(
                drug_dim=drug_embed_dim,
                pathway_dim=pathway_embed_dim,
                hidden_dim=64,
                dropout=dropout
            )

        # Multiplicative Pathway Interaction
        if self.use_interaction:
            self.interaction = MultiplicativePathwayInteraction(
                drug_dim=drug_embed_dim,
                pathway_dim=pathway_embed_dim,
                interaction_dim=interaction_dim,
                dropout=dropout
            )

        # Residual projection from flat input (1031)
        if self.use_residual:
            self.residual_proj = nn.Sequential(
                nn.Linear(1031, 64),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            res_dim = 64
        else:
            res_dim = 0

        # Regression Head
        head_in_dim = drug_embed_dim + pathway_embed_dim + (interaction_dim if self.use_interaction else 0) + res_dim
        self.regression_head = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        chem = batch["chem"]
        dg = batch["dg"]
        cell_pathways = batch["cell_pathways"]
        flat = batch["flat"]

        # Ablation check: DG-Net ablation
        if not self.use_dg_net:
            dg = torch.zeros_like(dg)

        # Ablation check: Modality ablation
        if self.active_modalities != ("exp", "mut", "cnv"):
            cp_masked = torch.zeros_like(cell_pathways)
            if "exp" in self.active_modalities:
                cp_masked[:, :, 0] = cell_pathways[:, :, 0]
            if "mut" in self.active_modalities:
                cp_masked[:, :, 1] = cell_pathways[:, :, 1]
            if "cnv" in self.active_modalities:
                cp_masked[:, :, 2] = cell_pathways[:, :, 2]
            cell_pathways = cp_masked

        # 1. Encode drug
        e_d = self.drug_encoder(chem, dg)  # (B, d_e)

        # 2. Encode cell pathways
        h_cp, h_c = self.cell_encoder(cell_pathways)  # (B, P, d_h), (B, d_h)

        # 3. Drug-conditioned gating
        if self.use_gating:
            h_prime, gate_scores = self.gating(e_d, h_cp, dg)
        else:
            h_prime = h_cp
            gate_scores = torch.ones(h_cp.shape[0], h_cp.shape[1], device=h_cp.device)

        # 4. Multiplicative interaction
        if self.use_interaction:
            z_d, alpha_weights = self.interaction(e_d, h_prime)
        else:
            z_d = None
            alpha_weights = torch.ones(h_cp.shape[0], h_cp.shape[1], device=h_cp.device) / h_cp.shape[1]

        # 5. Assemble representations
        feats = [e_d, h_c]
        if self.use_interaction:
            feats.append(z_d)
        if self.use_residual:
            # Ensure residual projection strictly does not leak ablated features
            needs_flat_mask = (not self.use_dg_net) or (self.active_modalities != ("exp", "mut", "cnv"))
            if needs_flat_mask:
                flat_masked = flat.clone()
                if not self.use_dg_net:
                    flat_masked[:, 256:452] = 0.0
                if "exp" not in self.active_modalities:
                    flat_masked[:, 452:639] = 0.0
                if "mut" not in self.active_modalities:
                    flat_masked[:, 639:835] = 0.0
                if "cnv" not in self.active_modalities:
                    flat_masked[:, 835:1031] = 0.0
                res = self.residual_proj(flat_masked)
            else:
                res = self.residual_proj(flat)
            feats.append(res)

        joint_repr = torch.cat(feats, dim=-1)
        y_pred = self.regression_head(joint_repr)

        return y_pred

    def get_explainability(self, batch: dict):
        """
        Extracts gating and attention weights for model explainability.
        """
        chem = batch["chem"]
        dg = batch["dg"]
        cell_pathways = batch["cell_pathways"]
        
        if not self.use_dg_net:
            dg = torch.zeros_like(dg)

        if self.active_modalities != ("exp", "mut", "cnv"):
            cp_masked = torch.zeros_like(cell_pathways)
            if "exp" in self.active_modalities:
                cp_masked[:, :, 0] = cell_pathways[:, :, 0]
            if "mut" in self.active_modalities:
                cp_masked[:, :, 1] = cell_pathways[:, :, 1]
            if "cnv" in self.active_modalities:
                cp_masked[:, :, 2] = cell_pathways[:, :, 2]
            cell_pathways = cp_masked

        e_d = self.drug_encoder(chem, dg)
        h_cp, _ = self.cell_encoder(cell_pathways)

        if self.use_gating:
            h_prime, gates = self.gating(e_d, h_cp, dg)
        else:
            h_prime = h_cp
            gates = torch.ones(h_cp.shape[0], h_cp.shape[1], device=h_cp.device)

        if self.use_interaction:
            _, alpha = self.interaction(e_d, h_prime)
        else:
            alpha = torch.ones(h_cp.shape[0], h_cp.shape[1], device=h_cp.device) / h_cp.shape[1]

        return {
            "gating_scores": gates.detach().cpu().numpy(),
            "pathway_weights": alpha.detach().cpu().numpy()
        }
