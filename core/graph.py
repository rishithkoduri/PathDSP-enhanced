"""
Pathway-graph model (PathDSP-X), as a single module (prediction, SHAP, attention) and in the batched form the GPU
pool trains (weights [models, ...]). Both compute exactly the same function (checked in the tests).

  node features   X[p] = (DG(p), EXP(p), MUT(p), CNV(p))                       per pathway p, from the 1,031 inputs
  node embedding  H0 = dropout(ELU(X W_in + b_in + E[p]))                      E = learned identity of each pathway
  message passing H_{l+1} = H_l + dropout(ELU((Â H_l) W_l + b_l))  l = 1..L    Â = normalised gene-overlap graph
  drug query      u = dropout(ELU(CHEM W_d + b_d)),  q = u W_q + b_q
  attention       α = softmax_p(H_L[p] · q / √d),  g = Σ_p α_p H_L[p],  m = mean_p H_L[p]
  readout         pure:   ŷ = W_o dropout(ELU(W_h [g ; m ; u] + b_h)) + b_o
                  hybrid: ŷ = PathDSP FNN on [x ; g ; m]   (the authors' 1000-800-500-100-1 network)
α is the drug-conditioned weight of every pathway for this drug–cell pair (saved for explanation).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pathways as PW

HIDDEN = (1000, 800, 500, 100)
HEAD = 128                     # pure readout hidden size
EMB_STD = 0.1                  # initial scale of the pathway identity embeddings


def settings(spec):
    return (int(spec.get("g_dim", 32)), int(spec.get("g_layers", 2)), str(spec.get("g_mode", "hybrid")),
            float(spec.get("g_tau", 0.05)))


def shuffled(spec):
    """control experiment: the same model on a randomly rewired pathway graph (degree-preserving edge swaps)"""
    return spec.get("g_graph", "pid") == "random"


def entries(spec, n_in, n_nodes, n_chem):
    """(name, kind, shape) in batched-parameter order; kinds: 'lin' (in, out) -> W, b; 'mat' (rows, cols) -> one tensor"""
    d, L, mode, _ = settings(spec)
    ent = [("node_in", "lin", (4, d)), ("node_emb", "mat", (n_nodes, d))]
    ent += [(f"gconv{l + 1}", "lin", (d, d)) for l in range(L)]
    ent += [("drug", "lin", (n_chem, d)), ("query", "lin", (d, d))]
    if mode == "pure":
        ent += [("head1", "lin", (3 * d, HEAD)), ("output", "lin", (HEAD, 1))]
    else:
        chain = (n_in + 2 * d,) + HIDDEN + (1,)
        ent += [(nm, "lin", (chain[k], chain[k + 1])) for k, nm in enumerate(("hidden1", "hidden2", "hidden3", "hidden4", "output"))]
    return ent


# ======================================================================================================================
# single model
# ======================================================================================================================
class GraphNet(nn.Module):
    def __init__(self, features, spec):
        super().__init__()
        d, L, mode, tau = settings(spec)
        gs = PW.graph_spec(features, tau, need_adj=False, shuffle=shuffled(spec))
        P, n_in = len(gs["nodes"]), len(features)
        self.d, self.L, self.mode, self.nodes = d, L, mode, list(gs["nodes"])
        idx = torch.as_tensor(gs["idx"])
        self.register_buffer("node_idx", idx.clamp(min=0))
        self.register_buffer("node_mask", (idx >= 0).float())
        adj = gs["adj"] if gs["adj"] is not None else np.zeros((P, P), dtype="float32")   # filled by load_state_dict
        self.register_buffer("adj", torch.as_tensor(adj))
        self.register_buffer("chem_idx", torch.as_tensor(gs["chem"]))
        self.node_in = nn.Linear(4, d)
        self.node_emb = nn.Parameter(torch.zeros(P, d))
        for l in range(L):
            setattr(self, f"gconv{l + 1}", nn.Linear(d, d))
        self.drug = nn.Linear(len(gs["chem"]), d)
        self.query = nn.Linear(d, d)
        if mode == "pure":
            self.head1 = nn.Linear(3 * d, HEAD)
            self.output = nn.Linear(HEAD, 1)
        else:
            self.hidden1 = nn.Linear(n_in + 2 * d, 1000)
            self.hidden2 = nn.Linear(1000, 800)
            self.hidden3 = nn.Linear(800, 500)
            self.hidden4 = nn.Linear(500, 100)
            self.output = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=float(spec.get("dropout", 0.1)))
        self.act = nn.ELU()

    def encode(self, x):
        """x [N, F] (scaled) -> (H [N, P, d], u [N, d], α [N, P], g [N, d], m [N, d])"""
        Xn = torch.stack([x.index_select(1, self.node_idx[k]) * self.node_mask[k] for k in range(4)], -1)
        H = self.dropout(self.act(self.node_in(Xn) + self.node_emb))
        for l in range(self.L):
            H = H + self.dropout(self.act(getattr(self, f"gconv{l + 1}")(torch.matmul(self.adj, H))))
        u = self.dropout(self.act(self.drug(x.index_select(1, self.chem_idx))))
        q = self.query(u)
        alpha = torch.softmax(torch.einsum("npd,nd->np", H, q) / math.sqrt(self.d), dim=1)
        g = torch.einsum("np,npd->nd", alpha, H)
        return H, u, alpha, g, H.mean(1)

    def forward(self, x):
        H, u, alpha, g, m = self.encode(x)
        if self.mode == "pure":
            return self.output(self.dropout(self.act(self.head1(torch.cat([g, m, u], 1)))))
        a = torch.cat([x, g, m], 1)
        for lin in (self.hidden1, self.hidden2, self.hidden3, self.hidden4):
            a = self.dropout(self.act(lin(a)))
        return self.output(a)

    @torch.no_grad()
    def attention(self, x):
        return self.encode(x)[2]


# ======================================================================================================================
# batched training form
# ======================================================================================================================
class GraphArch:
    kind = "graph"
    interaction = gate = fac = False
    d_int = 0
    gate_init = 0.0

    def __init__(self, spec, features, dev=None):
        d, L, mode, tau = settings(spec)
        gs = PW.graph_spec(features, tau, need_adj=True, shuffle=shuffled(spec))
        self.n_in, self.d, self.L, self.mode = len(features), d, L, mode
        self.n_nodes, self.n_chem = len(gs["nodes"]), len(gs["chem"])
        self.entries = entries(spec, self.n_in, self.n_nodes, self.n_chem)
        self.pos, i = {}, 0
        for nm, kind, _ in self.entries:
            self.pos[nm] = i
            i += 2 if kind == "lin" else 1
        self.n_tensors = i
        if dev is not None:
            idx = torch.as_tensor(gs["idx"], device=dev)
            self.idx = idx.clamp(min=0)
            self.mask = (idx >= 0).to(torch.float32)
            self.adj = torch.as_tensor(gs["adj"], device=dev)
            self.chem = torch.as_tensor(gs["chem"], device=dev)

    @property
    def n_params(self):
        n = 0
        for _, kind, shp in self.entries:
            n += shp[0] * shp[1] + (shp[1] if kind == "lin" else 0)
        return n

    @property
    def act_width(self):
        """activation values per training row (memory estimate)"""
        P, d, L = self.n_nodes, self.d, self.L
        w = P * 4 + P * d * (3 * L + 4) + 4 * d
        return w + (HEAD + 3 * d if self.mode == "pure" else self.n_in + 2 * d + sum(HIDDEN))

    @property
    def feat_width(self):
        """(H) width of the representation the study discriminator sees"""
        return HEAD if self.mode == "pure" else HIDDEN[-1]

    @property
    def row_floats(self):
        """activation values per row in eval mode (sizes the validation / test chunks)"""
        return self.n_nodes * (4 + self.d * (self.L + 3)) + self.n_in + sum(HIDDEN)


def _lin3(params, k, a):
    """linear layer on [A, b, P, in] with weights [A, in, out]"""
    return torch.einsum("abpi,aio->abpo", a, params[k]) + params[k + 1][:, None, None, :]


def _lin(params, k, a):
    return torch.baddbmm(params[k + 1].unsqueeze(1), a, params[k])


def forward_batched(params, x, arch, p_drop, training, return_feat=False):
    """x [A, b, F] (scaled) -> [A, b] (+ the last hidden layer with return_feat)"""
    P = arch.pos
    Xn = torch.stack([x.index_select(2, arch.idx[k]) * arch.mask[k] for k in range(4)], -1)      # [A, b, P, 4]
    H = F.dropout(F.elu(_lin3(params, P["node_in"], Xn) + params[P["node_emb"]][:, None]), p_drop, training)
    for l in range(arch.L):
        H = H + F.dropout(F.elu(_lin3(params, P[f"gconv{l + 1}"], torch.matmul(arch.adj, H))), p_drop, training)
    u = F.dropout(F.elu(_lin(params, P["drug"], x.index_select(2, arch.chem))), p_drop, training)
    q = _lin(params, P["query"], u)
    alpha = torch.softmax(torch.einsum("abpd,abd->abp", H, q) / math.sqrt(arch.d), dim=2)
    g = torch.einsum("abp,abpd->abd", alpha, H)
    m = H.mean(2)
    if arch.mode == "pure":
        h = F.dropout(F.elu(_lin(params, P["head1"], torch.cat([g, m, u], 2))), p_drop, training)
        out = _lin(params, P["output"], h).squeeze(-1)
        return (out, h) if return_feat else out
    a = torch.cat([x, g, m], 2)
    for nm in ("hidden1", "hidden2", "hidden3", "hidden4"):
        a = F.dropout(F.elu(_lin(params, P[nm], a)), p_drop, training)
    out = _lin(params, P["output"], a).squeeze(-1)
    return (out, a) if return_feat else out
