"""
The pathway graph used by the Pathway-graph model (PathDSP-X).

Nodes  = the pathways that have a DG_<pathway> feature (196 PID pathways in input.txt / processed_data), in the
         order of the DG_ features. Each node carries the pair's 4 scores for that pathway: DG, EXP, MUT, CNV
         (a block the dataset lacks for a pathway -> 0, i.e. the training mean after z-scoring).
Edges  = pathways that share genes: Jaccard(genes p, genes q) >= tau, weighted by the Jaccard value, plus a
         self-loop of weight 1; symmetric normalisation  D^-1/2 (A + I) D^-1/2.
Genes  = MSigDB C2 CP:PID gene sets, a .gmt file in refs/ (e.g. c2.cp.pid.v2024.1.Hs.symbols.gmt).
"""
from functools import lru_cache

import numpy as np

from . import config as C

BLOCKS = ("DG_", "EXP_", "MUT_", "CNV_")


def gmt_path():
    """the PID gene-set file in refs/ (None if absent)"""
    if not C.REFS.exists():
        return None
    for p in sorted(C.REFS.glob("*.gmt")):
        if "pid" in p.name.lower():
            return p
    return None


@lru_cache(maxsize=4)
def load_gmt(path):
    sets = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) >= 3 and t[0]:
                sets[t[0]] = frozenset(g for g in t[2:] if g)
    return sets


def nodes_of(features):
    return [f[len("DG_"):] for f in features if f.startswith("DG_")]


def graph_spec(features, tau=0.05, need_adj=True, shuffle=False):
    """{'nodes', 'idx' [4, P] (-1 = missing), 'adj' [P, P] float32 (None if need_adj is False and no .gmt),
        'chem' (CHEM_ feature indices), 'edges', 'matched'} — cached; callers must not modify it.
    shuffle: the control graph — the same edges and weights, rewired by degree-preserving swaps (seed 0)"""
    return _graph_spec(tuple(features), round(float(tau), 6), bool(need_adj), str(gmt_path()), bool(shuffle))


SHUFFLE_SEED = 0
SWAPS_PER_EDGE = 10


def rewire(A, seed=SHUFFLE_SEED, swaps_per_edge=SWAPS_PER_EDGE):
    """degree-preserving randomisation of a weighted symmetric graph (no self-loops): repeated double-edge swaps
    (a–b, c–d) -> (a–d, c–b) that create no self-loop or duplicate edge; every edge keeps its weight. Each node keeps
    its number of neighbours, but WHICH pathways are linked becomes random."""
    rng = np.random.RandomState(seed)
    iu = np.triu_indices(len(A), 1)
    m = A[iu] > 0
    E = [[int(a), int(b)] for a, b in zip(iu[0][m], iu[1][m])]
    W = A[iu][m].tolist()
    present = {(min(a, b), max(a, b)) for a, b in E}
    for _ in range(swaps_per_edge * len(E)):
        i, j = rng.randint(len(E)), rng.randint(len(E))
        if i == j:
            continue
        (a, b), (c, d) = E[i], E[j]
        if rng.rand() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present -= {(min(a, b), max(a, b)), (min(c, d), max(c, d))}
        present |= {e1, e2}
        E[i], E[j] = [a, d], [c, b]
    R = np.zeros_like(A)
    for (a, b), w in zip(E, W):
        R[a, b] = R[b, a] = w
    return R


@lru_cache(maxsize=8)
def _graph_spec(features, tau, need_adj, _gmt, shuffle=False):
    features = list(features)
    pos = {f: i for i, f in enumerate(features)}
    nodes = nodes_of(features)
    P = len(nodes)
    idx = np.full((4, P), -1, dtype=np.int64)
    for k, b in enumerate(BLOCKS):
        for p, name in enumerate(nodes):
            idx[k, p] = pos.get(b + name, -1)
    chem = np.array([i for i, f in enumerate(features) if f.startswith(("CHEM_", "MOL_"))], dtype=np.int64)
    out = {"nodes": nodes, "idx": idx, "chem": chem, "adj": None, "edges": 0, "matched": 0, "tau": float(tau)}
    path = gmt_path()
    if path is None:
        if need_adj:
            raise FileNotFoundError("The pathway-graph model needs the PID gene sets (MSigDB C2 CP:PID .gmt) in the "
                                    f"refs folder ({C.REFS}).")
        return out
    sets = load_gmt(str(path))
    genes = [sets.get(n) for n in nodes]
    out["matched"] = int(sum(g is not None for g in genes))
    A = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        if genes[i] is None:
            continue
        for j in range(i + 1, P):
            if genes[j] is None:
                continue
            u = len(genes[i] | genes[j])
            jac = len(genes[i] & genes[j]) / u if u else 0.0
            if jac >= tau and jac > 0:
                A[i, j] = A[j, i] = jac
    if shuffle:
        A = rewire(A)
    out["edges"] = int((A > 0).sum() // 2)
    out["shuffled"] = bool(shuffle)
    A += np.eye(P)
    d = 1.0 / np.sqrt(A.sum(1))
    out["adj"] = (A * d[:, None] * d[None, :]).astype("float32")
    return out


def check(features, tau=0.05):
    """None if the pathway-graph model can run on these features, else the reason"""
    if gmt_path() is None:
        return (f"The pathway-graph model needs the PID gene sets: put c2.cp.pid.*.symbols.gmt (MSigDB C2 CP:PID) into "
                f"{C.REFS}.")
    gs = graph_spec(features, tau)
    if len(gs["nodes"]) == 0:
        return "This dataset has no DG_ pathway features, so there are no pathway nodes."
    if gs["matched"] == 0:
        return "None of this dataset's pathways are in the .gmt file in refs/."
    if len(gs["chem"]) == 0:
        return "This dataset has no CHEM_ (drug fingerprint) features for the drug query."
    return None
