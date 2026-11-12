"""
Training engine: fold-parallel training of the three PathDSP models on one GPU.

All fold-models of a run (k-fold × repeats, every leave-one-out fold, and the all-data model) are trained in a POOL
of GPU slots: as many models as fit in free VRAM run at the same time as one batched network (one baddbmm per layer
for every model's own mini-batch), and the moment a model early-stops its slot is refilled with the next waiting
model, so the GPU stays full until the queue is empty. Per model the maths is identical to training it alone:

  * its own training rows, inner 10 % validation split (train_test_split, random_state = seed [+ repeat]),
    its own feature z-scoring (fitted on its training rows) when the model uses it
  * authors' init (He-normal or Kaiming-uniform, bias 0.01), ELU, dropout 0.1, RMSE loss sqrt(MSE + 1e-6)
  * Adam / AdaMax with weight decay 1e-5 and gradient-value clipping ±5 — element-wise, with a step counter (bias
    correction) PER MODEL, so a model that joins the pool later is optimized exactly as if it trained alone
  * validation loss = mean of per-batch RMSE over consecutive batches (authors' fit()), early stopping
    (val ≤ best counts as improvement, patience 30), best-epoch weights restored; a model that stops leaves the pool
Only difference: when the models of the pool have training sets of different sizes, the smaller ones wrap around to
the start of their own shuffled order so every mini-batch is full (affects < 1 % of rows).

Optional upgrades (off by default, see config.UPGRADES):
  (C) learning rate halved for a model after `plateau_patience` epochs without a validation loss at least
      `plateau_delta` below its best so far (floor LR_FLOOR×); runs made before these settings existed: 10 epochs, 0
  (F) MSE training loss: the optimizer minimises the mini-batch MSE instead of the authors' mini-batch RMSE
      (sqrt(MSE) weights every mini-batch by 1 / its own RMSE; the MSE gradient is the unbiased one for the RMSE metric)
  (G) global validation RMSE: early stopping / best epoch use the RMSE over the whole validation set instead of the
      authors' mean of per-batch RMSEs
  (E) weight averaging: an exponential moving average of each model's weights over about the last epoch of steps;
      validation, early stopping, the restored best weights and the test predictions all use the averaged weights
  (D) seed ensemble: every fold is trained by `ensemble` models with different initial weights / batch order / dropout
      masks; their test predictions are averaged (the runner combines them when the last one finishes)
"""
import dataclasses
import importlib.util
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle as sk_shuffle

from . import config as C
from . import nets


class Stopped(Exception):
    """the user pressed Stop"""


class TPairX:
    """GPU form of data.PairX: X[rows] (any index tensor shape) = cat(D[di[rows]], C[ci[rows]]) without materialising
    the [N, F] matrix"""

    def __init__(self, px, dev):
        self.D = torch.as_tensor(px.D, device=dev)
        self.C = torch.as_tensor(px.C, device=dev)
        self.di = torch.as_tensor(px.di, device=dev)
        self.ci = torch.as_tensor(px.ci, device=dev)
        self.shape = (int(px.shape[0]), int(px.shape[1]))

    def __getitem__(self, rows):
        return torch.cat([self.D[self.di[rows]], self.C[self.ci[rows]]], -1)

    def pairs(self, di, ci):
        """feature rows of arbitrary (drug index, cell index) pairs of the same tables"""
        return torch.cat([self.D[di], self.C[ci]], -1)

    def stats(self, rows=None):
        """(mean, population sd) per feature over the given rows (all when None) — float32 on the device"""
        out_m, out_s = [], []
        for T_, ix in ((self.D, self.di), (self.C, self.ci)):
            ix = ix if rows is None else ix[rows]
            w = torch.bincount(ix, minlength=T_.shape[0]).double()
            n = w.sum().clamp(min=1.0)
            T64 = T_.double()
            m = (w @ T64) / n
            v = (w @ (T64 - m) ** 2) / n
            out_m.append(m.float()); out_s.append(v.clamp(min=0).sqrt().float())
        return torch.cat(out_m), torch.cat(out_s)

    def mean(self, dim=0):
        return self.stats()[0]

    def std(self, dim=0):
        return self.stats()[1]


def device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def set_exact_math():
    """full fp32 matmuls (no TF32) — the PathDSP models are trained in exact fp32"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


# ======================================================================================================================
# jobs (one job = one model to train)
# ======================================================================================================================
@dataclass
class Job:
    scheme: str              # mixed | LOCO | LODO | final
    rep: int
    fold: int
    tr: np.ndarray           # all training rows (feature scaling / drug stats are fitted on these)
    tr_in: np.ndarray        # rows the optimizer sees
    va: np.ndarray           # inner validation rows (early stopping)
    te: np.ndarray           # held-out test rows (empty for the all-data model)
    group: str = ""          # held-out cell line / drug for leave-one-out folds
    extra: dict = field(default_factory=dict)
    member: int = 0          # (D) seed-ensemble member 1..n (0 = no ensemble)

    @property
    def key(self):
        """the fold (shared by all ensemble members)"""
        return f"{self.scheme}|{self.rep}|{self.fold}"

    @property
    def mkey(self):
        """this model"""
        return self.key + (f"|s{self.member}" if self.member else "")

    @property
    def final(self):
        return self.scheme == "final"


def _group_kfold(groups, k, seed):
    """balanced grouped k-fold (largest groups first, random tie order)"""
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k); fold_of_group = {}
    for gi in order:
        f = int(np.argmin(load)); fold_of_group[uniq[gi]] = f; load[f] += counts[gi]
    fold_of = np.array([fold_of_group[g] for g in groups])
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]


def make_jobs(ds, cfg):
    """cfg: schemes, grouping ('logo' | 'kfold'), folds, seed, final, spec(val_frac, repeats)"""
    n, seed = ds.n, int(cfg["seed"])
    k_folds = max(2, int(cfg.get("folds", C.N_FOLDS)))
    vf = float(cfg["spec"].get("val_frac", 0.1))
    reps = max(1, int(cfg["spec"].get("repeats", 1)))
    jobs = []
    if cfg.get("final", True):
        tr = np.arange(n)
        tr_in, va = train_test_split(tr, test_size=vf, random_state=seed)
        jobs.append(Job("final", 1, 0, tr, tr_in, va, np.zeros(0, dtype=np.int64)))
    for s in cfg["schemes"]:
        if s == "mixed":
            for r in range(reps):
                sd = seed + r
                perm = sk_shuffle(np.arange(n), random_state=sd)            # = sklearn.utils.shuffle(df, seed) in FNN.py
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=sd)
                for k, (a, b) in enumerate(kf.split(perm)):
                    tr, te = perm[a], perm[b]
                    tr_in, va = train_test_split(tr, test_size=vf, random_state=sd)
                    jobs.append(Job("mixed", r + 1, k + 1, tr, tr_in, va, te))
            continue
        if s == C.CSA_SCHEME:                                             # the benchmark's own splits
            from . import csa as CSA
            for k, (tr_in, va, te) in enumerate(CSA.splits(ds.study)):
                tr = np.sort(np.concatenate([tr_in, va]))
                jobs.append(Job(s, 1, k + 1, tr, tr_in, va, te, group=f"split {k}"))
            continue
        if s == C.LOSO_SCHEME:                                            # train on 4 studies, test on the 5th
            sc = ds.scodes
            for k, name in enumerate(ds.study_names):
                tr, te = np.where(sc != k)[0], np.where(sc == k)[0]
                tr_in, va = train_test_split(tr, test_size=vf, random_state=seed)
                jobs.append(Job(s, 1, k + 1, tr, tr_in, va, te, group=str(name)))
            continue
        groups = ds.cells if C.SCHEMES[s]["group"] == "cells" else ds.drugs
        if cfg.get("grouping", "logo") == "logo":                         # = sklearn LeaveOneGroupOut (sorted groups)
            for k, g in enumerate(np.unique(groups)):
                tr = np.where(groups != g)[0]; te = np.where(groups == g)[0]
                tr_in, va = train_test_split(tr, test_size=vf, random_state=seed)
                jobs.append(Job(s, 1, k + 1, tr, tr_in, va, te, group=str(g)))
        else:
            for r in range(reps):
                sd = seed + r
                for k, (tr, te) in enumerate(_group_kfold(groups, min(k_folds, len(np.unique(groups))), sd)):
                    tr_in, va = train_test_split(tr, test_size=vf, random_state=sd)
                    jobs.append(Job(s, r + 1, k + 1, tr, tr_in, va, te))
    n_ens = ensemble_size(cfg["spec"])
    if n_ens > 1:                                                         # (D) members of a fold stay next to each other
        jobs = [dataclasses.replace(j, member=m, extra={}) for j in jobs for m in range(1, n_ens + 1)]
    return jobs


def ensemble_size(spec):
    return max(1, int(spec.get("ensemble", 1) or 1))


# ======================================================================================================================
# targets
# ======================================================================================================================
def drug_stats(ds, idx):
    """per-drug mean / sd (population) of the response over rows idx -> (mu, sd) arrays over ds.drug_list (NaN if absent)"""
    codes = ds.drug_codes[idx]; y = ds.y[idx].astype("float64")
    D = len(ds.drug_list)
    cnt = np.bincount(codes, minlength=D).astype("float64")
    s1 = np.bincount(codes, weights=y, minlength=D)
    s2 = np.bincount(codes, weights=y * y, minlength=D)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 0.0))
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    mu[cnt == 0] = np.nan
    return mu, sd


def target_of(ds, job, kind):
    """(training target for every row [N] float32, inverse dict, drug-mean baseline for the test rows)"""
    mu, sd = drug_stats(ds, job.tr)
    base = mu[ds.drug_codes[job.te]] if len(job.te) else np.zeros(0)
    if kind == "drug_z":
        c = ds.drug_codes
        t = (ds.y - mu[c]) / sd[c]
        return t.astype("float32"), {"kind": "drug_z", "mu": mu, "sd": sd}, base
    if kind == "study_z":
        smu, ssd, fmu, fsd = study_stats(ds, job.tr)
        sc = ds.scodes
        t = (ds.y - smu[sc]) / ssd[sc]
        return t.astype("float32"), {"kind": "study_z", "mu": smu, "sd": ssd, "fallback": [fmu, fsd],
                                     "studies": list(ds.study_names)}, base
    return ds.y.astype("float32"), {"kind": "raw"}, base


def study_stats(ds, idx):
    """(S) per-study mean / sd (population) of the response over rows idx. A study without training rows (the
    held-out study of leave-study-out) gets the AVERAGE source study: mean of the studies' means and sds."""
    S = ds.n_studies
    sc = ds.scodes[idx]; y = ds.y[idx].astype("float64")
    cnt = np.bincount(sc, minlength=S).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.bincount(sc, weights=y, minlength=S) / cnt
        sd = np.sqrt(np.maximum(np.bincount(sc, weights=y * y, minlength=S) / cnt - mu * mu, 0.0))
    seen = cnt > 0
    sd[seen & (sd < 1e-6)] = 1.0
    fmu, fsd = float(mu[seen].mean()), float(sd[seen].mean())
    mu[~seen], sd[~seen] = fmu, fsd
    return mu, sd, fmu, fsd


def inverse(pred, inv, codes, scodes=None):
    """model output -> response. study_z: per-study scale of each row's study (scodes); without scodes (a new pair
    with no study, or a cross-study prediction) the average source study"""
    if inv["kind"] == "drug_z":
        return pred * inv["sd"][codes] + inv["mu"][codes]
    if inv["kind"] == "study_z":
        if scodes is None:
            fmu, fsd = inv["fallback"]
            return pred * fsd + fmu
        return pred * np.asarray(inv["sd"])[scodes] + np.asarray(inv["mu"])[scodes]
    return pred


def _chunked_order(TR, ntr, gkey, C, gen):
    """(R) epoch order in which rows of the same drug (and study) come in runs of C: rows grouped by key in a random
    order, cut into chunks of C, chunks shuffled. A mini-batch of b rows then holds ~b/C drugs × C cell lines, so the
    ranking loss has within-drug pairs. Same layout as the plain order: [A, L] positions, padding at the end."""
    A, L = TR.shape
    dev = TR.device
    ar = torch.arange(L, device=dev)
    order = ar.unsqueeze(0).repeat(A, 1)
    for a in range(A):
        n = int(ntr[a])
        if n == 0:
            continue
        g = gkey[TR[a, :n]]
        i1 = torch.argsort(torch.rand(n, generator=gen, device=dev))
        p = i1[torch.sort(g[i1], stable=True).indices]               # grouped by key, random inside a group
        gs = g[p]
        new_grp = torch.ones(n, dtype=torch.bool, device=dev)
        new_grp[1:] = gs[1:] != gs[:-1]
        start = torch.cummax(torch.where(new_grp, ar[:n], torch.zeros_like(ar[:n])), 0).values
        new_chunk = new_grp | ((ar[:n] - start) % C == 0)
        cid = torch.cumsum(new_chunk.long(), 0) - 1
        ck = torch.rand(int(cid[-1]) + 1, generator=gen, device=dev)
        order[a, :n] = p[torch.sort(ck[cid], stable=True).indices]
    return order


def rank_loss(out, yb, gb, tsd):
    """(R) RankNet loss within drug and study: for every pair of the mini-batch from the same drug and study with
    y_i > y_j, softplus(−(ŷ_i − ŷ_j) / sd); mean over such pairs per model ([A]). sd = the model's training-target sd,
    so the ordering term itself is unit-free (its weight against the main loss: see loss_scale)"""
    same = (gb.unsqueeze(2) == gb.unsqueeze(1)) & (yb.unsqueeze(2) > yb.unsqueeze(1))
    z = (out.unsqueeze(2) - out.unsqueeze(1)) / tsd.view(-1, 1, 1)
    sf = same.to(out.dtype)
    return (F.softplus(-z) * sf).sum((1, 2)) / sf.sum((1, 2)).clamp(min=1.0)


def loss_scale(tsd, mse_loss):
    """(R) / (H) weight of the extra terms against the main loss: the main loss of a model whose training target has
    sd tsd is of order tsd² (MSE) or tsd (RMSE), the ranking / adversarial terms are unit-free; scaling them by the
    same factor keeps rank_weight / adv_weight meaning the same for a raw target (AAC sd ≈ 0.2) as for a z-scored one
    (sd = 1, factor 1). [A]"""
    return tsd * tsd if mse_loss else tsd


AUX_HIDDEN = 64


def init_aux(A, feat_w, n_cls, gen, dev):
    """(H) study discriminator per model: feat -> 64 -> ELU -> n studies (He-normal, bias 0.01)"""
    out = []
    for fi, fo in ((feat_w, AUX_HIDDEN), (AUX_HIDDEN, n_cls)):
        out += [(torch.randn(A, fi, fo, generator=gen, device=dev) * math.sqrt(2.0 / fi)).requires_grad_(),
                torch.full((A, fo), 0.01, device=dev).requires_grad_()]
    return out


def adv_loss(aux, feat, coef, labels):
    """(H) cross-entropy of the study discriminator on gradient-REVERSED features: the discriminator learns to tell
    the studies apart, the network (scaled by coef per model) learns features that hide the study. [A]"""
    c = coef.view(-1, 1, 1).to(feat.dtype)
    fr = feat.detach() + (feat - feat.detach()) * (-c)              # forward = feat, backward = −coef · grad
    h = F.elu(torch.baddbmm(aux[1].unsqueeze(1), fr, aux[0]))
    logits = torch.baddbmm(aux[3].unsqueeze(1), h, aux[2])
    A, b, S = logits.shape
    return F.cross_entropy(logits.reshape(A * b, S), labels.reshape(-1), reduction="none").view(A, b).mean(1)


# ======================================================================================================================
# memory / pool size
# ======================================================================================================================
def per_model_bytes(arch, spec, n_rows):
    P = arch.n_params
    bs = int(spec["batch"])
    state = P * 4 * (7 if spec.get("ema") else 6)           # weights, grads, 2 moments, best copy, temporaries (+ EMA)
    state += int(getattr(arch, "fixed_bytes", 0))           # (M) every molecule is encoded at every step
    act = bs * arch.act_width * 4 * 6                       # activations, ELU / dropout buffers and their grads
    inp = bs * (arch.n_in + arch.d_int) * 4 * 4
    return int(state + act + inp + n_rows * 4 * 2)          # + the model's target row


def auto_parallel(n_jobs, arch, spec, n_rows, free_bytes=None, cap=512, max_perf=True):
    """how many fold-models fit in free VRAM at once (CPU: 8). max_perf uses 95 % of the free memory (else 85 %)."""
    if free_bytes is None:
        if not torch.cuda.is_available():
            return max(1, min(n_jobs, 8))
        free_bytes, _ = torch.cuda.mem_get_info()
    frac, reserve = (0.95, 600) if max_perf else (0.85, 900)
    budget = free_bytes * frac - reserve * 2 ** 20          # room for validation chunks, CUDA workspace
    return int(max(1, min(n_jobs, cap, budget // per_model_bytes(arch, spec, n_rows))))


def gpu_info():
    """device-wide numbers plus what THIS process holds (PyTorch's allocator) — the latter is what Task Manager
    shows as the process's dedicated GPU memory."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {"name": torch.cuda.get_device_name(0), "cuda": True, "free_mb": free / 2 ** 20, "total_mb": total / 2 ** 20,
            "used_mb": (total - free) / 2 ** 20, "proc_mb": torch.cuda.memory_reserved() / 2 ** 20,
            "alloc_mb": torch.cuda.memory_allocated() / 2 ** 20}


# ======================================================================================================================
# optimizer: Adam / AdaMax with a step counter per model
# ======================================================================================================================
def _adam_elem(p, g, m, v, clr, sbc2, b1: float, b2: float, eps: float, wd: float, clip: float):
    g = g.clamp(-clip, clip) + wd * p
    m.copy_(m + (g - m) * (1.0 - b1))
    v.copy_(v * b2 + (1.0 - b2) * g * g)
    p.copy_(p - clr * (m / (v.sqrt() / sbc2 + eps)))


def _adamax_elem(p, g, m, u, clr, sbc2, b1: float, b2: float, eps: float, wd: float, clip: float):
    g = g.clamp(-clip, clip) + wd * p
    m.copy_(m + (g - m) * (1.0 - b1))
    u.copy_(torch.maximum(u * b2, g.abs() + eps))
    p.copy_(p - clr * (m / u))


class SlotOpt:
    """torch.optim.Adam / Adamax maths (gradient-value clip ±c, then L2 weight decay added to the gradient, moments,
    bias-corrected step) applied to stacked weights [models, ...]. Every model has its own step counter, so models
    that join the pool later get their own bias correction, and its own learning-rate factor `scale` (upgrade C).
    With `fn` (a fused Triton kernel, see fused_kernel) the whole update is one pass over memory per weight tensor;
    otherwise standard PyTorch ops are used."""
    B1, B2, EPS = 0.9, 0.999, 1e-8

    def __init__(self, params, spec, fn=None):
        self.adam = spec["optimizer"] == "adam"
        self.lr, self.wd, self.fn = float(spec["lr"]), float(spec["weight_decay"]), fn
        self.params = params
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = torch.zeros(params[0].shape[0], dtype=torch.float64, device=params[0].device)
        self.scale = torch.ones(params[0].shape[0], dtype=torch.float64, device=params[0].device)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, clip):
        self.t += 1.0                                       # on the device: no host↔device sync in the loop
        dt = self.params[0].dtype
        clr = (self.lr * self.scale / (1.0 - torch.pow(self.B1, self.t))).to(dt)
        sbc2 = torch.sqrt(1.0 - torch.pow(self.B2, self.t)).to(dt)
        clip = float(clip)
        for p, m, v in zip(self.params, self.m, self.v):
            shp = (-1,) + (1,) * (p.dim() - 1)
            c, s2 = clr.view(shp), sbc2.view(shp)
            g = p.grad
            if self.fn is not None and all(t.is_contiguous() for t in (p, g, m, v)):
                # every weight tensor as [models, numel]: one compiled shape family for any architecture
                A = p.shape[0]
                try:
                    self.fn(p.view(A, -1), g.view(A, -1), m.view(A, -1), v.view(A, -1), clr.view(-1, 1),
                            sbc2.view(-1, 1), self.B1, self.B2, self.EPS, self.wd, clip)
                    continue
                except Exception:
                    # a compile failure happens before anything is written: switch to the standard ops (the same
                    # update, checked against the kernel at start-up) for the rest of the run
                    self.fn = None
            g.clamp_(-clip, clip)
            if self.wd:
                g.add_(p, alpha=self.wd)
            m.lerp_(g, 1.0 - self.B1)
            if self.adam:
                v.mul_(self.B2).addcmul_(g, g, value=1.0 - self.B2)
                d = v.sqrt().div_(s2).add_(self.EPS)
            else:
                torch.maximum(v.mul_(self.B2), g.abs_().add_(self.EPS), out=v)
                d = v
            p.sub_(torch.div(m, d).mul_(c))

    def select(self, keep, params):
        self.params = params
        self.m = [m[keep] for m in self.m]
        self.v = [v[keep] for v in self.v]
        self.t = self.t[keep]
        self.scale = self.scale[keep]

    def append(self, params, n):
        self.params = params
        z = lambda x: torch.zeros((n,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)  # noqa: E731
        self.m = [torch.cat([m, z(m)]) for m in self.m]
        self.v = [torch.cat([v, z(v)]) for v in self.v]
        self.t = torch.cat([self.t, z(self.t)])
        self.scale = torch.cat([self.scale, torch.ones_like(z(self.scale))])


def _relax_recompiles():
    """The fused kernel compiles once per shape family ([models, numel], each size 1 or > 1). Allow more recompiles
    than PyTorch's default of 8 and do not tie unrelated sizes together (duck shapes), so a long run never hits it."""
    try:
        import torch._dynamo.config as dc
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dc, name):
                setattr(dc, name, max(64, int(getattr(dc, name))))
    except Exception:
        pass
    try:
        import torch.fx.experimental._config as fc
        if hasattr(fc, "use_duck_shape"):
            fc.use_duck_shape = False
    except Exception:
        pass


def triton_available():
    return importlib.util.find_spec("triton") is not None


_FUSED = {}


def fused_kernel(kind, dev):
    """(compiled update function or None, description). Used only if Triton is installed and the compiled kernel
    matches the standard update in a start-up self-test (models joining at different times included)."""
    if dev.type != "cuda":
        return None, "standard PyTorch ops (CPU)"
    if not triton_available():
        return None, "standard PyTorch ops (Triton not installed)"
    if kind in _FUSED:
        return _FUSED[kind]
    try:
        _relax_recompiles()
        fn = torch.compile(_adam_elem if kind == "adam" else _adamax_elem, dynamic=True, fullgraph=True)
        g = torch.Generator(device=dev).manual_seed(0)
        spec_t = {"optimizer": kind, "lr": 1e-3, "weight_decay": 1e-5}

        def make(n):
            return [torch.randn(s, generator=g, device=dev) for s in ((n, 96, 40), (n, 40), (n, 1))]
        base = make(3)
        P1 = [b.clone().requires_grad_() for b in base]; P2 = [b.clone().requires_grad_() for b in base]
        A, B = SlotOpt(P1, spec_t, fn), SlotOpt(P2, spec_t, None)
        for it in range(8):
            if it == 4:                                   # a model joins: per-model step counters must stay exact
                extra = make(1)
                P1 = [torch.cat([p.detach(), e]).requires_grad_() for p, e in zip(P1, extra)]
                P2 = [torch.cat([p.detach(), e]).requires_grad_() for p, e in zip(P2, extra)]
                A.append(P1, 1); B.append(P2, 1)
            if it == 6:                                   # models leave until one is left (size-1 shapes)
                keep = torch.tensor([False, False, True, False], device=dev)
                P1 = [p.detach()[keep].clone().requires_grad_() for p in P1]
                P2 = [p.detach()[keep].clone().requires_grad_() for p in P2]
                A.select(keep, P1); B.select(keep, P2)
            grads = [torch.randn(p.shape, generator=g, device=dev) * 4 for p in P1]
            for p, gr in zip(P1, grads):
                p.grad = gr.clone()
            for p, gr in zip(P2, grads):
                p.grad = gr.clone()
            A.step(5.0); B.step(5.0)
        torch.cuda.synchronize()
        ok = all(torch.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(P1, P2))
        res = (fn, "fused Triton kernel") if ok else (None, "standard PyTorch ops (fused kernel did not match)")
    except Exception as e:
        res = (None, f"standard PyTorch ops (fused kernel failed: {type(e).__name__})")
    _FUSED[kind] = res
    return res


def _pad(lists, dev):
    n = torch.tensor([len(a) for a in lists], device=dev)
    out = np.zeros((len(lists), max(1, int(max(len(a) for a in lists)))), dtype=np.int64)
    for i, a in enumerate(lists):
        out[i, :len(a)] = a
    return torch.as_tensor(out, device=dev), n


def _val_loss(pred, y, IDX, n, bs, global_rmse=False):
    """authors' validation loss per model: mean over consecutive batches of sqrt(MSE + 1e-6);
    global_rmse (upgrade G): the RMSE over all of the model's validation rows"""
    A, L = IDX.shape
    se = (pred - y.gather(1, IDX)) ** 2
    mask = (torch.arange(L, device=IDX.device).unsqueeze(0) < n.unsqueeze(1)).float()
    if global_rmse:
        return torch.sqrt((se * mask).sum(1) / n.clamp(min=1).to(se.dtype))
    nch = math.ceil(L / bs); pad = nch * bs - L
    if pad:
        se = F.pad(se, (0, pad)); mask = F.pad(mask, (0, pad))
    se = se.view(A, nch, bs); mask = mask.view(A, nch, bs)
    cnt = mask.sum(2)
    r = torch.sqrt((se * mask).sum(2) / cnt.clamp(min=1) + 1e-6)
    ok = (cnt > 0).float()
    return (r * ok).sum(1) / ok.sum(1).clamp(min=1)


def label_of(job, seed=True):
    seed = f" · seed {job.member}" if (seed and job.member) else ""
    if job.final:
        return "all-data model" + seed
    return f"{job.scheme} r{job.rep} f{job.fold}" + (f" · {job.group}" if job.group else "") + seed


PLATEAU = 10          # (C) default for runs made before plateau_patience existed
LR_FLOOR = 1 / 64     # (C) the learning rate never drops below this fraction of the set value


def ema_weight(t, decay):
    """(E) per-model weight of the newest weights in the moving average: 1 − min(decay, (1+t)/(10+t)) — the
    warm-up term keeps the average from being dominated by the random initial weights"""
    d = torch.clamp((1.0 + t) / (10.0 + t), max=float(decay))
    return (1.0 - d).float()


# ======================================================================================================================
# the pool
# ======================================================================================================================
class Pool:
    """Trains `jobs` with at most P models on the GPU at once; when models stop, their slots are refilled with the
    next jobs at the following epoch boundary. Callbacks:
       prepare(job)            right before a model starts (sets job.extra['target'])
       on_step(info)           ~1×/s during an epoch
       on_round(info)          after every epoch round, with the status of every model in the pool
       on_done(job, result)    the moment a model stops (best weights, test predictions, history)"""

    def __init__(self, ctx, jobs, P):
        self.ctx, self.jobs, self.P = ctx, list(jobs), max(1, int(P))

    def run(self, prepare, on_step, on_round, on_done, stop_check):
        ctx, spec, dev, X = self.ctx, self.ctx.spec, self.ctx.dev, self.ctx.X
        N, Fdim = X.shape
        bs, E, pat = int(spec["batch"]), int(spec["epochs"]), int(spec["patience"])
        clip, p_drop = float(spec["clip"]), float(spec["dropout"])
        arch = ctx.arch
        plateau = bool(spec.get("plateau"))
        p_pat = max(1, int(spec.get("plateau_patience", PLATEAU)))
        p_delta = float(spec.get("plateau_delta", 0.0))
        ema_on = bool(spec.get("ema"))
        mse_loss = spec.get("loss", "rmse") == "mse"                                # (F)
        val_global = spec.get("val_metric", "batch") == "global"                    # (G)
        rank_on, adv_on = bool(spec.get("rank")), bool(spec.get("adv"))             # (R), (H)
        rank_w, rank_c = float(spec.get("rank_weight", 0.5)), max(2, int(spec.get("rank_chunk", 8)))
        adv_w, adv_ramp = float(spec.get("adv_weight", 0.1)), max(1, int(spec.get("adv_ramp", 10)))
        mol_on = getattr(arch, "kind", "") == "mol"                                # (M) drug index of a row
        molr = (lambda r: X.di[r]) if mol_on else (lambda r: None)
        if rank_on or adv_on:
            scode = torch.as_tensor(ctx.ds.scodes, device=dev)
            gkey = torch.as_tensor(ctx.ds.drug_codes, device=dev) * int(ctx.ds.n_studies) + scode
        aux = aopt = tsd = None
        ema_p = pbest = pwait = None
        gen = torch.Generator(device=dev).manual_seed(int(ctx.seed) * 1000 + 17)
        torch.manual_seed(int(ctx.seed) * 1000 + 1_000_003)                       # dropout masks
        queue = list(self.jobs)
        act, info = [], {}                                                       # info: id(job) -> python state
        params = best_p = opt = mu = sd = Y = best = wait = TR = ntr = VA = nva = None
        base = torch.arange(bs, device=dev).unsqueeze(0)
        rnd, last_ui, t_start = 0, 0.0, time.perf_counter()
        while act or queue:
            # ---------------- fill free slots ----------------
            n_new = min(self.P - len(act), len(queue))
            if n_new > 0:
                new = [queue.pop(0) for _ in range(n_new)]
                for j in new:
                    prepare(j)
                mu_n = torch.zeros(n_new, Fdim, device=dev); sd_n = torch.ones(n_new, Fdim, device=dev)
                if spec.get("feat_scale", True) and isinstance(X, TPairX):
                    for i, j in enumerate(new):
                        m_, s = X.stats(torch.as_tensor(j.tr, device=dev))
                        mu_n[i] = m_
                        sd_n[i] = torch.where(torch.isfinite(s) & (s >= 1e-8), s, torch.ones_like(s))
                elif spec.get("feat_scale", True):
                    for i, j in enumerate(new):
                        Xi = X[torch.as_tensor(j.tr, device=dev)]
                        mu_n[i] = Xi.mean(0)
                        s = Xi.std(0, unbiased=False)
                        sd_n[i] = torch.where(torch.isfinite(s) & (s >= 1e-8), s, torch.ones_like(s))
                        del Xi
                Y_n = torch.stack([torch.as_tensor(j.extra["target"], device=dev) for j in new])
                p_n = nets.init_batched(n_new, arch, spec["init"], gen, dev)
                if rank_on or adv_on:
                    tsd_n = torch.stack([Y_n[i][torch.as_tensor(j.tr, device=dev)].std(unbiased=False)
                                         .clamp(min=1e-6) for i, j in enumerate(new)])
                    tsd = tsd_n if tsd is None else torch.cat([tsd, tsd_n])
                if adv_on:
                    a_n = init_aux(n_new, arch.feat_width, int(ctx.ds.n_studies), gen, dev)
                    if aux is None:
                        aux, aopt = a_n, SlotOpt(a_n, spec, ctx.fused_fn)
                    else:
                        aux = [torch.cat([a.detach(), q.detach()]).requires_grad_() for a, q in zip(aux, a_n)]
                        aopt.append(aux, n_new)
                if params is None:
                    params = p_n
                    best_p = [p.detach().clone() for p in p_n]
                    opt = SlotOpt(params, spec, ctx.fused_fn)
                    mu, sd, Y = mu_n, sd_n, Y_n
                    best = torch.full((n_new,), float("inf"), device=dev)
                    wait = torch.zeros(n_new, dtype=torch.long, device=dev)
                    pbest = torch.full((n_new,), float("inf"), device=dev)
                    pwait = torch.zeros(n_new, dtype=torch.long, device=dev)
                    ema_p = [p.detach().clone() for p in p_n] if ema_on else None
                else:
                    params = [torch.cat([p.detach(), q.detach()]).requires_grad_() for p, q in zip(params, p_n)]
                    best_p = [torch.cat([b, q.detach().clone()]) for b, q in zip(best_p, p_n)]
                    opt.append(params, n_new)
                    mu, sd, Y = torch.cat([mu, mu_n]), torch.cat([sd, sd_n]), torch.cat([Y, Y_n])
                    best = torch.cat([best, torch.full((n_new,), float("inf"), device=dev)])
                    wait = torch.cat([wait, torch.zeros(n_new, dtype=torch.long, device=dev)])
                    pbest = torch.cat([pbest, torch.full((n_new,), float("inf"), device=dev)])
                    pwait = torch.cat([pwait, torch.zeros(n_new, dtype=torch.long, device=dev)])
                    if ema_on:
                        ema_p = [torch.cat([e, q.detach().clone()]) for e, q in zip(ema_p, p_n)]
                now = time.perf_counter()
                for j in new:
                    info[id(j)] = {"ep": 0, "best_ep": 0, "hist": {"train": [], "val": []}, "t0": now, "round0": rnd + 1}
                act += new
                TR, ntr = _pad([j.tr_in for j in act], dev)
                VA, nva = _pad([j.va for j in act], dev)
                del p_n, mu_n, sd_n, Y_n
            A = len(act)
            rnd += 1
            t_round = time.perf_counter()
            # ---------------- one epoch for every model in the pool ----------------
            L = TR.shape[1]
            if rank_on:                                                          # (R) drug-chunked mini-batches
                order = _chunked_order(TR, ntr, gkey, rank_c, gen)
            else:
                keys = torch.rand(A, L, generator=gen, device=dev)
                keys.masked_fill_(torch.arange(L, device=dev).unsqueeze(0) >= ntr.unsqueeze(1), 2.0)
                order = keys.argsort(1)
            if adv_on:                                                           # (H) ramp-up over the first epochs
                coef = torch.tensor([adv_w * min(1.0, (info[id(j)]["ep"] + 1) / adv_ramp) for j in act], device=dev)
            steps = math.ceil(int(ntr.max()) / bs)
            ema_decay = 1.0 - 1.0 / max(1, steps)                                # (E) ≈ the last epoch of steps
            tot = torch.zeros(A, device=dev)
            for s in range(steps):
                pos = (base + s * bs) % ntr.unsqueeze(1)
                rows = TR.gather(1, order.gather(1, pos))                         # [A, bs]
                xb = (X[rows] - mu.unsqueeze(1)) / sd.unsqueeze(1)
                if rank_on or adv_on:                                             # (R) / (H): extra loss terms
                    if adv_on:
                        out, feat = nets.forward_batched(params, xb, arch, p_drop, True, return_feat=True,
                                                         mol=molr(rows))
                    else:
                        out = nets.forward_batched(params, xb, arch, p_drop, True, mol=molr(rows))
                    yb = Y.gather(1, rows)
                    mse_b = ((out - yb) ** 2).mean(1)
                    le = torch.sqrt(mse_b + 1e-6)
                    loss = mse_b if mse_loss else le
                    lscale = loss_scale(tsd, mse_loss)
                    if rank_on:
                        loss = loss + rank_w * lscale * rank_loss(out, yb, gkey[rows], tsd)
                    if adv_on:
                        loss = loss + adv_loss(aux, feat, coef * lscale, scode[rows])
                        aopt.zero_grad()
                    opt.zero_grad()
                    loss.sum().backward()
                    le = le.detach()
                    if adv_on:
                        aopt.step(clip)
                    opt.step(clip)
                elif mse_loss:                                                    # (F)
                    out = nets.forward_batched(params, xb, arch, p_drop, True, mol=molr(rows))
                    mse_b = ((out - Y.gather(1, rows)) ** 2).mean(1)
                    opt.zero_grad()
                    mse_b.sum().backward()
                    le = torch.sqrt(mse_b.detach() + 1e-6)                        # shown as RMSE
                else:
                    out = nets.forward_batched(params, xb, arch, p_drop, True, mol=molr(rows))
                    le = torch.sqrt(((out - Y.gather(1, rows)) ** 2).mean(1) + 1e-6)
                    opt.zero_grad()
                    le.sum().backward()
                if not (rank_on or adv_on):
                    opt.step(clip)
                if ema_on:
                    with torch.no_grad():
                        w = ema_weight(opt.t, ema_decay)
                        for e, p in zip(ema_p, params):
                            e.lerp_(p.detach(), w.view((-1,) + (1,) * (p.dim() - 1)))
                tot += le.detach()
                now = time.perf_counter()
                if now - last_ui >= 1.0:
                    last_ui = now
                    if stop_check():
                        raise Stopped()
                    on_step({"round": rnd, "step": s + 1, "steps": steps, "active": A, "slots": self.P,
                             "queued": len(queue), "train_loss": float(tot.mean()) / (s + 1),
                             "seconds": now - t_start})
            t_train = time.perf_counter() - t_round
            # ---------------- validation (eval mode) + early stopping ----------------
            eval_p = ema_p if ema_on else params
            with torch.no_grad():
                Lv = VA.shape[1]
                rc = max(1, int((256 * 2 ** 20) // max(1, A * arch.row_floats * 4)))
                pv = []
                for c0 in range(0, Lv, rc):
                    xb = (X[VA[:, c0:c0 + rc]] - mu.unsqueeze(1)) / sd.unsqueeze(1)
                    pv.append(nets.forward_batched(eval_p, xb, arch, p_drop, False, mol=molr(VA[:, c0:c0 + rc])))
                vl = _val_loss(torch.cat(pv, 1), Y, VA, nva, bs, val_global)
                imp = vl <= best
                best = torch.where(imp, vl, best)
                wait = torch.where(imp, torch.zeros_like(wait), wait + 1)
                if bool(imp.any()):
                    for k in range(len(params)):
                        best_p[k][imp] = eval_p[k].detach()[imp]
                if plateau:                                                   # (C) halve the LR on a plateau
                    sig = vl < pbest - p_delta
                    pbest = torch.where(sig, vl, pbest)
                    pwait = torch.where(sig, torch.zeros_like(pwait), pwait + 1)
                    red = pwait >= p_pat
                    if bool(red.any()):
                        opt.scale = torch.where(red, (opt.scale * 0.5).clamp(min=LR_FLOOR), opt.scale)
                        pwait = torch.where(red, torch.zeros_like(pwait), pwait)
            tl = (tot / steps).cpu().numpy(); vl_c = vl.cpu().numpy(); imp_c = imp.cpu().numpy()
            wait_c = wait.cpu().numpy(); best_c = best.cpu().numpy(); lr_c = opt.scale.cpu().numpy() * opt.lr
            for q, j in enumerate(act):
                s_ = info[id(j)]
                s_["ep"] += 1
                s_["hist"]["train"].append(float(tl[q])); s_["hist"]["val"].append(float(vl_c[q]))
                if imp_c[q]:
                    s_["best_ep"] = s_["ep"]
            eps_ = np.array([info[id(j)]["ep"] for j in act])
            stop_now = (wait_c >= pat) | (eps_ >= E)
            now = time.perf_counter()
            on_round({"round": rnd, "active": A, "slots": self.P, "queued": len(queue), "steps": steps,
                      "round_seconds": now - t_round, "train_seconds": t_train, "patience": pat, "epochs": E,
                      "seconds": now - t_start,
                      "models": [{"key": j.mkey, "label": label_of(j), "lr": float(lr_c[q]), "scheme": j.scheme, "ep": int(eps_[q]),
                                  "train": float(tl[q]), "val": float(vl_c[q]), "best": float(best_c[q]),
                                  "best_ep": int(info[id(j)]["best_ep"]), "wait": int(wait_c[q]),
                                  "improved": bool(imp_c[q]), "stopping": bool(stop_now[q]),
                                  "seconds": now - info[id(j)]["t0"]} for q, j in enumerate(act)]})
            if stop_now.any():
                for q in np.where(stop_now)[0]:
                    on_done(act[q], self._result(act[q], int(q), best_p, mu, sd, best_c, info))
                keep = ~stop_now
                act = [j for j, k_ in zip(act, keep) if k_]
                if act:
                    kt = torch.as_tensor(keep, device=dev)
                    params = [p.detach()[kt].clone().requires_grad_() for p in params]
                    best_p = [b[kt] for b in best_p]
                    opt.select(kt, params)
                    mu, sd, Y, best, wait = mu[kt], sd[kt], Y[kt], best[kt], wait[kt]
                    pbest, pwait = pbest[kt], pwait[kt]
                    if ema_on:
                        ema_p = [e[kt] for e in ema_p]
                    if rank_on or adv_on:
                        tsd = tsd[kt]
                    if adv_on:
                        aux = [a.detach()[kt].clone().requires_grad_() for a in aux]
                        aopt.select(kt, aux)
                    TR, ntr, VA, nva = TR[kt], ntr[kt], VA[kt], nva[kt]
                else:
                    params = best_p = opt = mu = sd = Y = best = wait = TR = ntr = VA = nva = None
                    ema_p = pbest = pwait = None
                    aux = aopt = tsd = None
            if stop_check():
                raise Stopped()

    def _result(self, job, q, best_p, mu, sd, best_c, info):
        ctx = self.ctx
        p_q = [p[q:q + 1] for p in best_p]
        pred = np.zeros(0, dtype=np.float64)
        if len(job.te):
            with torch.no_grad():
                out = []
                te = torch.as_tensor(job.te, device=ctx.dev)
                ch = min(8192, max(256, int((512 * 2 ** 20) // (ctx.arch.row_floats * 4))))
                clip = job.extra.get("clip")                # leave-study-out: inputs clipped to the training range
                if clip is not None:
                    lo_t = torch.as_tensor(np.asarray(clip[0], np.float32), device=ctx.dev)
                    hi_t = torch.as_tensor(np.asarray(clip[1], np.float32), device=ctx.dev)
                for c0 in range(0, len(te), ch):
                    xr = ctx.X[te[c0:c0 + ch]]
                    if clip is not None:
                        xr = torch.maximum(torch.minimum(xr, hi_t), lo_t)
                    xb = ((xr - mu[q]) / sd[q]).unsqueeze(0)
                    mol = ctx.X.di[te[c0:c0 + ch]].unsqueeze(0) if getattr(ctx.arch, "kind", "") == "mol" else None
                    out.append(nets.forward_batched(p_q, xb, ctx.arch, float(ctx.spec["dropout"]), False, mol=mol)[0])
                pred = torch.cat(out).double().cpu().numpy()
        s_ = info.pop(id(job))
        return {"params": [p.detach().cpu() for p in p_q], "mu": mu[q].cpu().numpy(), "sd": sd[q].cpu().numpy(),
                "pred_t": pred, "best_epoch": int(s_["best_ep"]), "epochs_run": int(s_["ep"]),
                "best_val": float(best_c[q]), "hist": s_["hist"], "seconds": time.perf_counter() - s_["t0"],
                "round0": s_["round0"]}


def predict_pairs(ctx, params, mu, sd, di, ci, lo=None, hi=None):
    """output of ONE trained model (batched-form weights [1, ...], e.g. a finished job's result) for arbitrary
    (drug index, cell index) pairs of the context's pair tables (CSA cross-study test); eval mode, model units.
    lo / hi: per-feature range the raw inputs are clipped to before scaling (None = no clipping)"""
    dev = ctx.dev
    lo_t = None if lo is None else torch.as_tensor(np.asarray(lo, dtype=np.float32), device=dev)
    hi_t = None if hi is None else torch.as_tensor(np.asarray(hi, dtype=np.float32), device=dev)
    p_q = [p.to(dev) for p in params]
    mu_t = torch.as_tensor(np.asarray(mu, dtype=np.float32), device=dev)
    sd_t = torch.as_tensor(np.asarray(sd, dtype=np.float32), device=dev)
    di_t = torch.as_tensor(np.asarray(di, dtype=np.int64), device=dev)
    ci_t = torch.as_tensor(np.asarray(ci, dtype=np.int64), device=dev)
    ch = min(8192, max(256, int((512 * 2 ** 20) // (ctx.arch.row_floats * 4))))
    out = []
    with torch.no_grad():
        for c0 in range(0, len(di_t), ch):
            xr = ctx.X.pairs(di_t[c0:c0 + ch], ci_t[c0:c0 + ch])
            if lo_t is not None:
                xr = torch.maximum(torch.minimum(xr, hi_t), lo_t)
            xb = ((xr - mu_t) / sd_t).unsqueeze(0)
            mol = di_t[c0:c0 + ch].unsqueeze(0) if getattr(ctx.arch, "kind", "") == "mol" else None
            out.append(nets.forward_batched(p_q, xb, ctx.arch, float(ctx.spec["dropout"]), False, mol=mol)[0])
    return torch.cat(out).double().cpu().numpy() if out else np.zeros(0)


# ======================================================================================================================
# run context
# ======================================================================================================================
class Context:
    def __init__(self, ds, spec, seed, dev=None):
        self.ds, self.spec, self.seed = ds, spec, int(seed)
        self.dev = dev or device()
        set_exact_math()
        self.X = TPairX(ds.X, self.dev) if hasattr(ds.X, "stats") else torch.as_tensor(ds.X, device=self.dev)
        dg, cl, _ = ds.interaction_pairs
        self.n_pairs = len(dg)
        self.arch = nets.make_arch(spec, ds.features, (dg, cl), nets.fac_index(ds.features), self.dev)
        self.fused_fn, self.opt_name = fused_kernel(spec["optimizer"], self.dev) if spec.get("fused", True) \
            else (None, "standard PyTorch ops (fused kernel switched off)")

    def release(self):
        self.X = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================================================================
# time estimates
# ======================================================================================================================
class StepTime:
    """seconds per optimizer step as a function of the number of models trained together: s(A) = a + b·A,
    fitted on the epochs measured so far (falls back to a proportional model with a fixed-overhead share)."""

    def __init__(self):
        self.pts = []                                       # (A, seconds per step)

    def add(self, A, sec):
        self.pts.append((float(A), float(sec)))
        self.pts = self.pts[-400:]

    def __call__(self, A):
        if not self.pts:
            return float("nan")
        P = np.array(self.pts)
        As = np.unique(P[:, 0])
        if len(As) >= 2 and As.max() - As.min() >= 2:
            b, a = np.polyfit(P[:, 0], P[:, 1], 1)
            if b > 0 and a > 0:
                return float(a + b * A)
        A0, s0 = P[-1]
        return float(s0 * (0.35 + 0.65 * A / max(A0, 1.0)))


def expected_ends(n_models, dist, default=140):
    """expected stop epochs of n models (empirical quantiles of finished models' epochs, else a default)"""
    if n_models <= 0:
        return np.zeros(0)
    if dist is not None and len(dist) >= 3:
        q = (np.arange(n_models) + 0.5) / n_models
        return np.quantile(np.asarray(dist, float), q)
    return np.full(n_models, float(default))


def simulate_pool(active_remaining, queued_epochs, P, steps, step_time):
    """seconds until the pool is empty: the active models still need active_remaining[k] epochs, the queued models
    need queued_epochs[k] epochs each and join as slots free up (at most P at once)."""
    act = [float(a) for a in active_remaining if a > 1e-9]
    q = [max(1.0, float(x)) for x in queued_epochs]
    t = 0.0
    while act or q:
        while len(act) < P and q:
            act.append(q.pop(0))
        d = min(act)
        t += d * steps * step_time(len(act))
        act = [a - d for a in act if a - d > 1e-9]
    return t
