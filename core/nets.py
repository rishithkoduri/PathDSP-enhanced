"""
Networks.

Single-model modules (used for prediction and SHAP):
  PathDSPNet   — identical to the authors' myModel.FNN (same layer names, so their state_dicts load as-is)
  ProposedNet  — PathDSPNet + optional parts of the proposed system:
                   interaction module   drug-pathway × cell-pathway products → Linear d_int → ELU → dropout
                   (B) gate             a learnable per-unit gate on the interaction output, starting at 0
                   (A) factorization    drug tower (CHEM + DG → 256 → r) · cell tower (EXP/MUT/CNV → 256 → r),
                                        dot product / r added to the network's output (rank r = spec 'fac_dim';
                                        runs made before the setting existed: 64)

Batched form (used for training many fold-models at once on one GPU): every linear layer's weight has shape
[models, in, out]; one torch.baddbmm runs every model's own mini-batch. The maths per model is exactly the
single-model network; `to_state_dict` converts model i of a batch into the single-model state_dict.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DRUG_PREFIXES

HIDDEN = (1000, 800, 500, 100)
FAC_HIDDEN, FAC_DIM = 256, 64          # (A) tower hidden size; FAC_DIM = rank of runs made before 'fac_dim' existed


def fac_dim(spec):
    return int(spec.get("fac_dim", FAC_DIM))


def fac_index(features):
    """(drug feature indices, cell feature indices) for the factorization towers: CHEM_ + DG_ = drug, rest = cell"""
    drug = np.array([str(f).startswith(DRUG_PREFIXES + ("MOL_",)) for f in features], dtype=bool)
    return np.where(drug)[0].astype(np.int64), np.where(~drug)[0].astype(np.int64)


# ======================================================================================================================
# single models
# ======================================================================================================================
class PathDSPNet(nn.Module):
    def __init__(self, n_in, dropout=0.1):
        super().__init__()
        self.hidden1 = nn.Linear(n_in, 1000)
        self.hidden2 = nn.Linear(1000, 800)
        self.hidden3 = nn.Linear(800, 500)
        self.hidden4 = nn.Linear(500, 100)
        self.output = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fnn = nn.Sequential(self.hidden1, nn.ELU(), self.dropout,
                                 self.hidden2, nn.ELU(), self.dropout,
                                 self.hidden3, nn.ELU(), self.dropout,
                                 self.hidden4, nn.ELU(), self.dropout,
                                 self.output)

    def forward(self, x):
        return self.fnn(x)                                   # [N, 1]


class ProposedNet(nn.Module):
    """x (1,031 PathDSP features)
         interaction:  z_k = DG(p_k) · CELL(p_k) for the aligned pathway pairs → h = dropout(ELU(W_int z + b)) (d_int)
                       (B) h ← h ⊙ gate        (gate starts at 0)
         FNN:          PathDSP FNN on [x ; h]  (on x alone without the interaction module)
         (A)           + (u · v) / r,  u = drug tower(x_drug), v = cell tower(x_cell), rank r"""

    def __init__(self, n_in, dg_idx=None, cell_idx=None, d_int=128, dropout=0.1, interaction=True, gate=False,
                 fac_idx=None, fac_rank=FAC_DIM):
        super().__init__()
        self.use_int = bool(interaction)
        self.use_gate = self.use_int and bool(gate)
        self.use_fac = fac_idx is not None
        self.fac_rank = int(fac_rank)
        first = n_in
        if self.use_int:
            self.register_buffer("dg_idx", torch.as_tensor(np.asarray(dg_idx), dtype=torch.long))
            self.register_buffer("cell_idx", torch.as_tensor(np.asarray(cell_idx), dtype=torch.long))
            self.interaction = nn.Linear(len(dg_idx), d_int)
            first = n_in + d_int
            if self.use_gate:
                self.gate = nn.Parameter(torch.zeros(d_int))
        self.hidden1 = nn.Linear(first, 1000)
        self.hidden2 = nn.Linear(1000, 800)
        self.hidden3 = nn.Linear(800, 500)
        self.hidden4 = nn.Linear(500, 100)
        self.output = nn.Linear(100, 1)
        if self.use_fac:
            fd, fc = fac_idx
            self.register_buffer("fac_drug", torch.as_tensor(np.asarray(fd), dtype=torch.long))
            self.register_buffer("fac_cell", torch.as_tensor(np.asarray(fc), dtype=torch.long))
            self.drug1 = nn.Linear(len(fd), FAC_HIDDEN)
            self.drug2 = nn.Linear(FAC_HIDDEN, self.fac_rank)
            self.cell1 = nn.Linear(len(fc), FAC_HIDDEN)
            self.cell2 = nn.Linear(FAC_HIDDEN, self.fac_rank)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ELU()

    def forward(self, x):
        a = x
        if self.use_int:
            z = x.index_select(1, self.dg_idx) * x.index_select(1, self.cell_idx)
            h = self.dropout(self.act(self.interaction(z)))
            if self.use_gate:
                h = h * self.gate
            a = torch.cat([x, h], 1)
        for lin in (self.hidden1, self.hidden2, self.hidden3, self.hidden4):
            a = self.dropout(self.act(lin(a)))
        out = self.output(a)                                 # [N, 1]
        if self.use_fac:
            u = self.drug2(self.dropout(self.act(self.drug1(x.index_select(1, self.fac_drug)))))
            v = self.cell2(self.dropout(self.act(self.cell1(x.index_select(1, self.fac_cell)))))
            out = out + (u * v).sum(1, keepdim=True) / self.fac_rank
        return out


def is_graph(spec):
    return spec.get("arch", "fnn") == "graph"


def is_mol(spec):
    """(M) learned drug encoder in front of the network"""
    return bool(spec.get("mol"))


def _no_mol(spec):
    s = dict(spec)
    s["mol"] = False
    return s


def build(spec, n_in, pairs=None, features=None):
    """spec: model settings dict ('interaction', 'gate', 'fac', 'dropout', optional 'd_int', 'arch').
    pairs = (dg_idx, cell_idx) for the interaction module; features = feature names (factorization towers, graph)."""
    if is_mol(spec):                                         # (M) base network on [x ; drug embedding]
        from . import molenc as M
        if features is None:
            raise ValueError("the drug encoder needs the feature names")
        aug = M.aug_features(features, spec)
        return M.MolNet(build(_no_mol(spec), len(aug), pairs, aug), spec, float(spec.get("dropout", 0.1)))
    if is_graph(spec):
        from .graph import GraphNet
        if features is None:
            raise ValueError("the pathway-graph model needs the feature names")
        return GraphNet(features, spec)
    inter, fac = bool(spec.get("interaction")), bool(spec.get("fac"))
    drop = float(spec.get("dropout", 0.1))
    if not inter and not fac:
        return PathDSPNet(n_in, drop)
    if fac and features is None:
        raise ValueError("the factorization term needs the feature names")
    return ProposedNet(n_in, pairs[0] if inter else None, pairs[1] if inter else None, int(spec.get("d_int", 128)),
                       drop, inter, bool(spec.get("gate")), fac_index(features) if fac else None, fac_dim(spec))


# ======================================================================================================================
# batched training form
# ======================================================================================================================
def make_arch(spec, features, pairs=None, fac_idx=None, dev=None):
    """batched layout for a model definition: Arch (the PathDSP FNN family), graph.GraphArch, or molenc.MolArch
    (M: either of them on [x ; drug embedding])"""
    if is_mol(spec):
        from . import molenc as M
        aug = M.aug_features(features, spec)
        base = make_arch(_no_mol(spec), aug, pairs, fac_index(aug) if spec.get("fac") else None, dev)
        return M.MolArch(spec, base, dev)
    if is_graph(spec):
        from .graph import GraphArch
        return GraphArch(spec, features, dev)
    return Arch(spec, len(features), pairs, fac_idx, dev)


class Arch:
    """Layout of the batched parameter list for one model definition.
    entries: (name, kind, shape) in parameter order — kind 'lin' (fan_in, fan_out) gives two tensors W [A,in,out],
    b [A,out]; kind 'vec' (size,) gives one tensor [A,size]. Names are the single-model module attributes."""
    kind = "fnn"

    def __init__(self, spec, n_in, pairs=None, fac_idx=None, dev=None):
        self.n_in = int(n_in)
        self.interaction = bool(spec.get("interaction"))
        self.gate = self.interaction and bool(spec.get("gate"))
        self.gate_init = float(spec.get("gate_init", 0.0))   # (B) runs made before 'gate_init' existed: 0
        self.fac = bool(spec.get("fac"))
        self.fac_rank = fac_dim(spec)
        d_int = int(spec.get("d_int", 128))
        n_pairs = len(pairs[0]) if (self.interaction and pairs is not None) else 0
        n_drug = len(fac_idx[0]) if (self.fac and fac_idx is not None) else 0
        n_cell = len(fac_idx[1]) if (self.fac and fac_idx is not None) else 0
        ent = []
        if self.interaction:
            ent.append(("interaction", "lin", (n_pairs, d_int)))
            if self.gate:
                ent.append(("gate", "vec", (d_int,)))
        chain = (self.n_in + (d_int if self.interaction else 0),) + HIDDEN + (1,)
        for k, nm in enumerate(("hidden1", "hidden2", "hidden3", "hidden4", "output")):
            ent.append((nm, "lin", (chain[k], chain[k + 1])))
        if self.fac:
            ent += [("drug1", "lin", (n_drug, FAC_HIDDEN)), ("drug2", "lin", (FAC_HIDDEN, self.fac_rank)),
                    ("cell1", "lin", (n_cell, FAC_HIDDEN)), ("cell2", "lin", (FAC_HIDDEN, self.fac_rank))]
        self.entries = ent
        self.pos, i = {}, 0
        for nm, kind, _ in ent:
            self.pos[nm] = i
            i += 2 if kind == "lin" else 1
        self.n_tensors = i
        self.n_pairs, self.n_drug, self.n_cell, self.d_int = n_pairs, n_drug, n_cell, d_int
        t = (lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)) if dev is not None else None
        self.dg = t(pairs[0]) if (t and self.interaction) else None
        self.cl = t(pairs[1]) if (t and self.interaction) else None
        self.fd = t(fac_idx[0]) if (t and self.fac) else None
        self.fc = t(fac_idx[1]) if (t and self.fac) else None

    @property
    def n_params(self):
        n = 0
        for _, kind, shp in self.entries:
            n += shp[0] * shp[1] + shp[1] if kind == "lin" else shp[0]
        return n

    @property
    def row_floats(self):
        """values per row that size the validation / test chunks (the input width, as before)"""
        return self.n_in

    @property
    def feat_width(self):
        """(H) width of the representation the study discriminator sees: the last hidden layer"""
        return HIDDEN[-1]

    @property
    def act_width(self):
        """activation values per training row (for the memory estimate)"""
        w = sum(HIDDEN)
        if self.interaction:
            w += self.d_int + 2 * self.n_pairs
        if self.fac:
            w += 2 * (FAC_HIDDEN + self.fac_rank) + self.n_drug + self.n_cell
        return w


def init_batched(A, arch, init, gen, dev):
    """He-normal (kaiming_normal_, fan_in, gain √2) or Kaiming-uniform (kaiming_uniform_) weights, bias 0.01 —
    the authors' init_weights; gates start at arch.gate_init; 'mat' (pathway embeddings) N(0, 0.1²).
    Returns a flat list of leaf tensors in arch order."""
    params = []
    for _, kind, shp in arch.entries:
        if kind == "vec":
            params.append(torch.full((A,) + tuple(shp), arch.gate_init, device=dev).requires_grad_())
            continue
        if kind == "mat":
            from .graph import EMB_STD
            params.append((torch.randn((A,) + tuple(shp), generator=gen, device=dev) * EMB_STD).requires_grad_())
            continue
        fan_in, fan_out = shp
        if init == "he_uniform":
            bound = math.sqrt(6.0 / fan_in)
            w = (torch.rand(A, fan_in, fan_out, generator=gen, device=dev) * 2.0 - 1.0) * bound
        else:
            w = torch.randn(A, fan_in, fan_out, generator=gen, device=dev) * math.sqrt(2.0 / fan_in)
        params += [w.requires_grad_(), torch.full((A, fan_out), 0.01, device=dev).requires_grad_()]
    return params


def _lin(params, k, a):
    return torch.baddbmm(params[k + 1].unsqueeze(1), a, params[k])


def forward_batched(params, x, arch, p_drop, training, return_feat=False, mol=None):
    """params: flat list from init_batched; x: [A, b, F] (already scaled) -> [A, b]
    (and the last hidden layer [A, b, feat_width] with return_feat, for the study discriminator H);
    mol [A, b] = each row's drug index for the drug encoder (M)"""
    if arch.kind == "mol":
        from . import molenc as M
        return M.forward_batched(params, x, arch, p_drop, training, return_feat, mol)
    if arch.kind == "graph":
        from .graph import forward_batched as graph_forward
        return graph_forward(params, x, arch, p_drop, training, return_feat)
    P = arch.pos
    k1 = P["hidden1"]
    if arch.interaction:
        z = x.index_select(2, arch.dg) * x.index_select(2, arch.cl)
        h = F.dropout(F.elu(_lin(params, P["interaction"], z)), p_drop, training)
        if arch.gate:
            h = h * params[P["gate"]].unsqueeze(1)
        W1, B1 = params[k1], params[k1 + 1]
        n_in = x.shape[2]
        a = torch.baddbmm(B1.unsqueeze(1), x, W1[:, :n_in]).baddbmm_(h, W1[:, n_in:])
    else:
        a = _lin(params, k1, x)
    a = F.dropout(F.elu(a), p_drop, training)
    for nm in ("hidden2", "hidden3", "hidden4"):
        a = F.dropout(F.elu(_lin(params, P[nm], a)), p_drop, training)
    out = _lin(params, P["output"], a).squeeze(-1)
    if arch.fac:
        u = _lin(params, P["drug2"], F.dropout(F.elu(_lin(params, P["drug1"], x.index_select(2, arch.fd))), p_drop, training))
        v = _lin(params, P["cell2"], F.dropout(F.elu(_lin(params, P["cell1"], x.index_select(2, arch.fc))), p_drop, training))
        out = out + (u * v).sum(-1) / arch.fac_rank
    return (out, a) if return_feat else out


def to_state_dict(params, i, spec, features, pairs=None):
    """model i of a batch -> state_dict of the single-model module (CPU)."""
    net = build(spec, len(features), pairs, features)
    arch = make_arch(spec, features, pairs, fac_index(features) if spec.get("fac") else None)
    if arch.kind == "mol":                                   # (M) base entries -> net.base, encoder -> net.enc
        layout = [(net.base, nm, kind, arch.base.pos[nm]) for nm, kind, _ in arch.base.entries]
        k = arch.n_base
        for nm, kind, _ in arch.enc_entries:
            layout.append((net.enc, nm, kind, k))
            k += 2
    else:
        layout = [(net, nm, kind, arch.pos[nm]) for nm, kind, _ in arch.entries]
    with torch.no_grad():
        for mod, nm, kind, k in layout:
            if kind in ("vec", "mat"):
                getattr(mod, nm).copy_(params[k][i].detach().cpu())
            else:
                lin = getattr(mod, nm)
                lin.weight.copy_(params[k][i].detach().t().cpu())
                lin.bias.copy_(params[k + 1][i].detach().cpu())
    return cpu_state(net)


def cpu_state(net):
    """state_dict on CPU; PathDSPNet registers every layer twice (hidden1 == fnn.0 …) — the aliases are kept so
    torch.save stores each tensor once."""
    out, seen = {}, {}
    for k, v in net.state_dict().items():
        key = (v.data_ptr(), tuple(v.shape))
        if key not in seen:
            seen[key] = v.detach().to("cpu", copy=True)
        out[k] = seen[key]
    return out
