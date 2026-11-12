"""
(M) Learned drug encoder — a small graph neural network over each drug's molecule, trained end-to-end with the rest
of the model (Phase 4). The fixed 256-bit fingerprint only says which substructures a drug has; the encoder learns
which parts of a molecule matter for the response, so an unseen drug is placed by its chemistry, not by bit overlap.

  atoms      42 features (element, degree, charge, H count, hybridisation, aromatic, in ring) — data/csa/molecules.npz
  embed      H0 = ELU(A W_a + b_a)                                   (d = 64)
  message    H_{l+1} = H_l + ELU((H_l + Σ_neighbours H_l) W_l + b_l)  l = 1..3   (GIN-style sum aggregation, residual)
  readout    e = dropout(ELU(mean_atoms(H_L) W_o + b_o))              (E = 64, per drug)
  model      the base network (Paper / Repo / Proposed FNN or the pathway graph) gets [x ; e] — 64 extra inputs
             MOL_0 … MOL_63 (the graph model's drug query uses them together with CHEM)

The molecule table follows the CSA package's drug order, which is also the drug index of every CSA row (PairX.di),
so a row's molecule needs no lookup. Only for the CSA datasets.
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MOL_D = 64            # atom state size
MOL_E = 64            # drug embedding size (extra inputs of the base network)
MOL_L = 3             # message-passing rounds
PREFIX = "MOL_"


def mol_features(n=MOL_E):
    return [f"{PREFIX}{i}" for i in range(n)]


def aug_features(features, spec):
    return list(features) + mol_features(int(spec.get("mol_dim", MOL_E)))


def settings(spec):
    return int(spec.get("mol_hidden", MOL_D)), int(spec.get("mol_dim", MOL_E)), int(spec.get("mol_layers", MOL_L))


@lru_cache(maxsize=2)
def _load(path, mtime):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def molecules():
    """the molecule table of the CSA package (None if absent)"""
    from . import csa as CSA
    p = CSA.DIR / "molecules.npz"
    if not p.exists():
        return None
    return _load(str(p), p.stat().st_mtime_ns)


def available():
    return molecules() is not None


def tensors(dev):
    m = molecules()
    if m is None:
        raise FileNotFoundError("The drug encoder (M) needs data/csa/molecules.npz (tools/build_csa.py --molecules-only).")
    n_mols = int(m["n_mols"])
    mol = torch.as_tensor(m["mol"].astype(np.int64), device=dev)
    cnt = torch.bincount(mol, minlength=n_mols).clamp(min=1).to(torch.float32)
    return {"x": torch.as_tensor(m["x"].astype(np.float32), device=dev),
            "src": torch.as_tensor(m["src"].astype(np.int64), device=dev),
            "dst": torch.as_tensor(m["dst"].astype(np.int64), device=dev),
            "mol": mol, "cnt": cnt, "n_mols": n_mols, "n_atoms": int(len(m["mol"])), "n_feat": int(m["x"].shape[1]),
            "drug_ids": [str(d) for d in m["drug_ids"]]}


def entries(spec, n_feat):
    d, E, L = settings(spec)
    return ([("atom", "lin", (n_feat, d))] + [(f"conv{l + 1}", "lin", (d, d)) for l in range(L)] +
            [("out", "lin", (d, E))])


def encode(W, mt, p_drop, training):
    """batched encoder: W = flat [W, b] list per entry with a leading models axis A -> e [A, n_mols, E]"""
    L = (len(W) - 4) // 2
    x = mt["x"]
    H = F.elu(torch.matmul(x.unsqueeze(0), W[0]) + W[1].unsqueeze(1))                     # [A, atoms, d]
    for l in range(L):
        k = 2 + 2 * l
        M = torch.zeros_like(H).index_add(1, mt["dst"], H.index_select(1, mt["src"]))
        H = H + F.elu(torch.matmul(H + M, W[k]) + W[k + 1].unsqueeze(1))
    P = torch.zeros(H.shape[0], mt["n_mols"], H.shape[2], dtype=H.dtype, device=H.device).index_add(1, mt["mol"], H)
    P = P / mt["cnt"].view(1, -1, 1)
    e = F.elu(torch.matmul(P, W[-2]) + W[-1].unsqueeze(1))
    return F.dropout(e, p_drop, training)


# ======================================================================================================================
# single model (prediction, SHAP)
# ======================================================================================================================
class MolEncoder(nn.Module):
    def __init__(self, spec, n_feat):
        super().__init__()
        d, E, L = settings(spec)
        self.atom = nn.Linear(n_feat, d)
        self.L = L
        for l in range(L):
            setattr(self, f"conv{l + 1}", nn.Linear(d, d))
        self.out = nn.Linear(d, E)

    def weights(self):
        """the batched layout (models axis of 1) of this module's weights"""
        W = []
        for nm in ["atom"] + [f"conv{l + 1}" for l in range(self.L)] + ["out"]:
            lin = getattr(self, nm)
            W += [lin.weight.t().unsqueeze(0), lin.bias.unsqueeze(0)]
        return W


class MolNet(nn.Module):
    """base network on [x ; e(drug)]: forward(x, mol_idx) — mol_idx = the CSA drug index of every row"""

    def __init__(self, base, spec, dropout=0.1):
        super().__init__()
        self.base = base
        self.mt = tensors(torch.device("cpu"))
        self.enc = MolEncoder(spec, self.mt["n_feat"])
        self.p_drop = float(dropout)
        self.drug_ids = self.mt["drug_ids"]

    def _move(self, dev):
        if self.mt["x"].device != dev:
            self.mt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in self.mt.items()}

    def embeddings(self):
        self._move(self.enc.atom.weight.device)
        return encode(self.enc.weights(), self.mt, self.p_drop, self.training)[0]          # [n_mols, E]

    def augment(self, x, mol_idx):
        return torch.cat([x, self.embeddings().index_select(0, mol_idx.to(x.device).long())], 1)

    def forward(self, x, mol_idx):
        return self.base(self.augment(x, mol_idx))


# ======================================================================================================================
# batched training form
# ======================================================================================================================
class MolArch:
    """base arch on the augmented features + the encoder entries; params = [base params …, encoder params …]"""
    kind = "mol"

    def __init__(self, spec, base, dev=None):
        self.base = base
        self.mt = tensors(dev) if dev is not None else None
        n_feat = len(molecules()["features"]) if molecules() is not None else 42
        self.enc_entries = entries(spec, n_feat)
        self.n_base = base.n_tensors
        self.entries = list(base.entries) + [("enc." + nm, k, s) for nm, k, s in self.enc_entries]
        self.n_tensors = self.n_base + 2 * len(self.enc_entries)
        self.d, self.E, self.L = settings(spec)
        self.n_in = base.n_in
        self.d_int = getattr(base, "d_int", 0)
        self.interaction = getattr(base, "interaction", False)
        self.gate_init = getattr(base, "gate_init", 0.0)       # (B) gates of a Proposed base start here
        self.n_atoms = self.mt["n_atoms"] if self.mt else int(len(molecules()["mol"]))
        self.n_mols = self.mt["n_mols"] if self.mt else int(molecules()["n_mols"])

    @property
    def n_params(self):
        n = self.base.n_params
        for _, _, (fi, fo) in self.enc_entries:
            n += fi * fo + fo
        return n

    @property
    def act_width(self):
        return self.base.act_width

    @property
    def row_floats(self):
        return self.base.row_floats + self.E

    @property
    def feat_width(self):
        return self.base.feat_width

    @property
    def fixed_bytes(self):
        """per-model memory that does not grow with the batch: every molecule is encoded at every step"""
        return int(self.n_atoms * self.d * (4 * self.L + 4) * 4 * 3 + self.n_mols * self.E * 4 * 4)


def subset(mt, sel):
    """the molecule table restricted to the molecules sel (sorted, unique): their atoms and bonds, re-indexed.
    Every molecule is encoded on its own (bonds never cross molecules), so encoding a subset gives the same
    embeddings as encoding the full table and picking those rows."""
    n = int(mt["n_mols"])
    keep = torch.zeros(n, dtype=torch.bool, device=sel.device)
    keep[sel] = True
    am = keep[mt["mol"]]                                                # atoms of the selected molecules
    pos = torch.cumsum(am.to(torch.int64), 0) - 1                       # their new atom index (order kept)
    em = am[mt["src"]]                                                  # bonds inside the selected molecules
    newm = torch.full((n,), -1, dtype=torch.int64, device=sel.device)
    newm[sel] = torch.arange(sel.numel(), device=sel.device)
    return {"x": mt["x"][am], "src": pos[mt["src"][em]], "dst": pos[mt["dst"][em]], "mol": newm[mt["mol"][am]],
            "cnt": mt["cnt"][sel], "n_mols": int(sel.numel())}


def forward_batched(params, x, arch, p_drop, training, return_feat, mol):
    from . import nets
    mol = mol.long()
    sel, inv = torch.unique(mol, return_inverse=True)
    if 2 * sel.numel() < arch.n_mols:
        # a batch touches few drugs: encode only those (same embeddings, a fraction of the time and memory)
        e_all = encode(params[arch.n_base:], subset(arch.mt, sel), p_drop, training)       # [A, n_sel, E]
        idx = inv
    else:
        e_all = encode(params[arch.n_base:], arch.mt, p_drop, training)                   # [A, n_mols, E]
        idx = mol
    e = torch.gather(e_all, 1, idx.unsqueeze(-1).expand(-1, -1, e_all.shape[2]))          # [A, b, E]
    return nets.forward_batched(params[:arch.n_base], torch.cat([x, e], -1), arch.base, p_drop, training, return_feat)
