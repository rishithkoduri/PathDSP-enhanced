"""
PathGDT — full test frontend (v3).
Upload ONE zip of DepMap+GDSC CSVs -> unpack, read all rows, build the real
5-block matrix (progress bar), DELETE the unpacked files, then train the
PathGDT model (BPE-SMILES Transformer + pathway cross-attention) with
cross-validation. Live epoch/loss/ETA, model save-to-folder + reload dropdown,
predict on new data with a saved model, IC50 output, ROC-AUC, SHAP, and
VRAM cleanup. Forces the NVIDIA RTX 4060 (cuda:0).

Run:  pip install -r requirements.txt   then   streamlit run app.py
"""
import os, io, re, time, glob, zipfile, tempfile, shutil, gc, hashlib, datetime
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             roc_auc_score, roc_curve)
from scipy.stats import pearsonr, spearmanr

import prep

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except Exception as e:
    TORCH = False; TORCH_ERR = str(e)

APPDIR = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(APPDIR, "refs")
MODELS_DIR = os.path.join(APPDIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- GPU: force NVIDIA cuda:0 (AMD iGPU is never CUDA) + speed knobs ----
if TORCH and torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
else:
    DEVICE = torch.device("cpu") if TORCH else None

# torch.compile needs a working Triton backend (absent on Windows by default).
HAS_TRITON = False
if TORCH:
    try:
        import triton  # noqa: F401
        HAS_TRITON = True
    except Exception:
        HAS_TRITON = False
    try:
        torch._dynamo.config.suppress_errors = True   # any compile failure -> silent eager fallback
    except Exception:
        pass


# ---- version-proof AMP helpers (no deprecation spam) ----
def amp_autocast(enabled):
    try: return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError): return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(enabled):
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=enabled)


# path_sa=False -> fast (no 588x588 pathway self-attention); cross-attention kept.
PRESETS = {
    "Fast":                   dict(d=96,  layers=1, heads=4, epochs=20, batch=128, lr=5e-4, max_len=120, path_sa=False),
    "Balanced":               dict(d=128, layers=2, heads=8, epochs=30, batch=128, lr=3e-4, max_len=140, path_sa=False),
    "Research (max quality)": dict(d=192, layers=3, heads=8, epochs=45, batch=96,  lr=2e-4, max_len=160, path_sa=False),
}

# =========================================================================
# SMILES tokenizer (BPE + regex fallback), serializable
# =========================================================================
_SMI_RE = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


class SmilesTok:
    def __init__(self, method="regex", vocab_size=300, max_len=120):
        self.method, self.vocab_size, self.max_len = method, vocab_size, max_len
        self.stoi = {}; self._hf = None

    def fit(self, smis):
        specials = [PAD, UNK, BOS, EOS]
        if self.method == "bpe":
            try:
                from tokenizers import Tokenizer, models, trainers, pre_tokenizers
                tok = Tokenizer(models.BPE(unk_token=UNK))
                tok.pre_tokenizer = pre_tokenizers.Split(pattern=_SMI_RE.pattern, behavior="isolated")
                tok.train_from_iterator(smis, trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=specials))
                self._hf = tok; return self
            except Exception:
                self.method = "regex"
        vocab = set()
        for s in smis: vocab.update(_SMI_RE.findall(s))
        self.stoi = {t: i for i, t in enumerate(specials + sorted(vocab))}
        return self

    @property
    def pad_id(self): return self._hf.token_to_id(PAD) if self._hf else self.stoi[PAD]
    @property
    def size(self): return self._hf.get_vocab_size() if self._hf else len(self.stoi)

    def encode(self, s):
        if self._hf: ids = self._hf.encode(s).ids
        else: ids = [self.stoi.get(t, self.stoi[UNK]) for t in _SMI_RE.findall(s)]
        bos = self._hf.token_to_id(BOS) if self._hf else self.stoi[BOS]
        eos = self._hf.token_to_id(EOS) if self._hf else self.stoi[EOS]
        return [bos] + ids[: self.max_len - 2] + [eos]

    def batch(self, smis):
        seqs = [self.encode(s) for s in smis]
        L = min(self.max_len, max(len(s) for s in seqs))
        ids = np.full((len(seqs), L), self.pad_id, dtype="int64")
        m = np.zeros((len(seqs), L), dtype=bool)
        for i, s in enumerate(seqs):
            s = s[:L]; ids[i, :len(s)] = s; m[i, :len(s)] = True
        return ids, m

    def state(self):
        return {"method": "bpe" if self._hf else "regex", "max_len": self.max_len,
                "hf": self._hf.to_str() if self._hf else None, "stoi": self.stoi}

    @staticmethod
    def load(state):
        t = SmilesTok(state["method"], max_len=state["max_len"])
        if state["method"] == "bpe" and state.get("hf"):
            from tokenizers import Tokenizer
            t._hf = Tokenizer.from_str(state["hf"])
        else:
            t.stoi = state["stoi"]
        return t


# =========================================================================
# Molecular graph (RDKit) — for the GNN drug branch
# =========================================================================
GNN_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Na", "K", "Se", "H"]
GNN_HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
GNN_FIN = len(GNN_ELEMENTS) + 1 + 5 + len(GNN_HYB) + 1     # atom-feature dim = 27


def _oh(v, choices): return [1.0 if v == c else 0.0 for c in choices] + [1.0 if v not in choices else 0.0]


def _atom_feat(a):
    return (_oh(a.GetSymbol(), GNN_ELEMENTS) + [a.GetDegree() / 4.0, float(a.GetFormalCharge()),
            a.GetTotalNumHs() / 4.0, 1.0 * a.GetIsAromatic(), 1.0 * a.IsInRing()]
            + _oh(str(a.GetHybridization()), GNN_HYB))


def smiles_to_graph(smi):
    """SMILES -> (node_feats [n,GNN_FIN], sym-normalized adj [n,n]) or None."""
    try:
        from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
        from rdkit import Chem
    except Exception:
        return None
    m = Chem.MolFromSmiles(str(smi))
    if m is None or m.GetNumAtoms() == 0:
        return None
    n = m.GetNumAtoms()
    X = np.array([_atom_feat(a) for a in m.GetAtoms()], dtype="float32")
    A = np.eye(n, dtype="float32")
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); A[i, j] = A[j, i] = 1.0
    d = A.sum(1); dinv = 1.0 / np.sqrt(np.clip(d, 1e-8, None))
    A = (A * dinv).T * dinv                                # D^-1/2 (A+I) D^-1/2
    return X, A.astype("float32")


def build_graph_bank(drug_smiles, drug_order):
    """Padded per-drug graph tensors for the unique drugs (indexed by drug_order)."""
    graphs = [smiles_to_graph(drug_smiles.get(d, "C")) for d in drug_order]
    valid = [g for g in graphs if g is not None]
    MAX_A = max((g[0].shape[0] for g in valid), default=1)
    Fdim = valid[0][0].shape[1] if valid else GNN_FIN
    D = len(drug_order)
    GNF = np.zeros((D, MAX_A, Fdim), "float32")
    GADJ = np.zeros((D, MAX_A, MAX_A), "float32")
    GMASK = np.zeros((D, MAX_A), "float32")
    for i, g in enumerate(graphs):
        if g is None:
            GADJ[i, 0, 0] = 1.0; GMASK[i, 0] = 1.0; continue
        X, A = g; n = X.shape[0]
        GNF[i, :n] = X; GADJ[i, :n, :n] = A; GMASK[i, :n] = 1.0
    return GNF, GADJ, GMASK, Fdim


# =========================================================================
# Model
# =========================================================================
if TORCH:
    class RMSELoss(nn.Module):
        def forward(self, x, y): return torch.sqrt(F.mse_loss(x, y) + 1e-6)

    class DenseGCN(nn.Module):
        """Batched GCN (Kipf-Welling) in pure PyTorch — no torch_geometric needed."""
        def __init__(self, fin, d, layers=2, drop=0.1):
            super().__init__()
            self.inp = nn.Linear(fin, d)
            self.gcs = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.act = nn.GELU(); self.drop = nn.Dropout(drop)

        def forward(self, x, adj, mask):
            h = self.act(self.inp(x))                          # [B,N,d]
            for lin, ln in zip(self.gcs, self.lns):
                h = ln(h + self.drop(self.act(lin(torch.bmm(adj, h)))))
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)      # masked mean -> [B,d]

    class PathGDT(nn.Module):
        """Pathway tokens + BPE-SMILES transformer joined by cross-attention.
        path_sa=False skips the O(P^2) pathway self-attention (fast); the drug
        still attends over every pathway via cross-attention (interpretable)."""
        def __init__(self, n_path, n_extra, vocab, pad_id, d=128, heads=8, layers=2,
                     drop=0.1, max_len=140, path_sa=False, use_gnn=False, gnn_fin=GNN_FIN):
            super().__init__()
            self.path_sa = path_sa; self.use_gnn = use_gnn; self.n_path = n_path; self.max_len = max_len
            self.val = nn.Linear(1, d); self.pemb = nn.Embedding(n_path, d)
            if path_sa:
                enc = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, activation="gelu")
                self.cancer = nn.TransformerEncoder(enc, layers)
            else:
                self.cnorm = nn.LayerNorm(d)
            self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
            self.pos = nn.Embedding(max_len, d)
            enc2 = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, activation="gelu")
            self.drug = nn.TransformerEncoder(enc2, layers)
            self.extra = nn.Sequential(nn.Linear(n_extra, d), nn.GELU())
            fuse_in = 2 * d
            if use_gnn:
                self.gnn = DenseGCN(gnn_fin, d, layers=min(layers, 3), drop=drop)   # graph drug branch
                fuse_in = 3 * d
            self.fuse = nn.Sequential(nn.Linear(fuse_in, d), nn.GELU())
            self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
            self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
            self._attn = None

        def forward(self, path, ids, mask, extra, gnf=None, gadj=None, gmask=None):
            B, P = path.shape
            idx = torch.arange(P, device=path.device).unsqueeze(0).expand(B, P)
            ct = self.val(path.unsqueeze(-1)) + self.pemb(idx)
            ct = self.cancer(ct) if self.path_sa else self.cnorm(F.gelu(ct))
            L = ids.size(1); pos = torch.arange(L, device=ids.device).unsqueeze(0)
            h = self.drug(self.tok(ids) + self.pos(pos), src_key_padding_mask=~mask)
            m = mask.unsqueeze(-1).float(); dseq = (h * m).sum(1) / m.sum(1).clamp(min=1)
            parts = [dseq, self.extra(extra)]
            if self.use_gnn and gnf is not None:
                parts.append(self.gnn(gnf, gadj, gmask))       # + molecular-graph embedding
            drug = self.fuse(torch.cat(parts, -1))
            ctx, attn = self.cross(drug.unsqueeze(1), ct, ct, need_weights=True, average_attn_weights=True)
            self._attn = attn.detach()
            return self.head(torch.cat([drug, ctx.squeeze(1)], -1)).squeeze(-1)

    class FNNBaseline(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, 1000), nn.ELU(), nn.Dropout(0.1),
                                     nn.Linear(1000, 800), nn.ELU(), nn.Dropout(0.1),
                                     nn.Linear(800, 500), nn.ELU(), nn.Dropout(0.1),
                                     nn.Linear(500, 100), nn.ELU(), nn.Dropout(0.1),
                                     nn.Linear(100, 1))
        def forward(self, feats): return self.net(feats).squeeze(-1)


def _unwrap(m): return getattr(m, "_orig_mod", m)


def make_model(kind, meta, cfg):
    if kind == "fnn":
        m = FNNBaseline(meta["FE"].shape[1])
    else:
        m = PathGDT(len(meta["cancer_cols"]), len(meta["extra_cols"]), meta["tok_size"], meta["pad_id"],
                    d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], max_len=meta["max_len"],
                    path_sa=cfg.get("path_sa", False), use_gnn=cfg.get("use_gnn", False),
                    gnn_fin=meta.get("gnn_fin", GNN_FIN))
    return m.to(DEVICE)


def build_from_bundle(b, tok):
    kind, cfg = b["model_kind"], b["cfg"]
    if kind == "fnn":
        m = FNNBaseline(len(b["cancer_cols"]) + len(b["extra_cols"]))
    else:
        m = PathGDT(len(b["cancer_cols"]), len(b["extra_cols"]), tok.size, tok.pad_id,
                    d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
                    max_len=cfg.get("max_len", 140), path_sa=cfg.get("path_sa", True),   # old models: True
                    use_gnn=cfg.get("use_gnn", False), gnn_fin=cfg.get("gnn_fin", GNN_FIN))
    m.load_state_dict(b["state_dict"]); return m.to(DEVICE).eval()


def _fwd(run, is_fnn, bt, T, gb, use_gnn):
    """Single forward that gathers the batch's molecular graphs when the GNN is on."""
    path, ids, mask, extra, feats, y, didx = T
    if is_fnn:
        return run(feats[bt])
    if use_gnn and gb is not None:
        di = didx[bt]
        return run(path[bt], ids[bt], mask[bt], extra[bt], gb[0][di], gb[1][di], gb[2][di])
    return run(path[bt], ids[bt], mask[bt], extra[bt])


# =========================================================================
# Train one fold (with live epoch/ETA callback)
# =========================================================================
def train_fold(model, is_fnn, T, tr, te, cfg, gb=None, prog=None, on_batch=None):
    path, ids, mask, extra, feats, y, didx = T
    use_gnn = (not is_fnn) and cfg.get("use_gnn", False) and gb is not None
    amp = bool(cfg["amp"]) and DEVICE.type == "cuda"
    bs, lr, epochs, patience = cfg["batch"], cfg["lr"], cfg["epochs"], cfg["patience"]
    run = model
    if cfg.get("compile") and DEVICE.type == "cuda" and HAS_TRITON:
        try: run = torch.compile(model, dynamic=True)
        except Exception: run = model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = RMSELoss(); scaler = make_scaler(amp)
    best, best_state, wait, tl_hist, vl_hist = 1e9, None, 0, [], []

    def batches(idx, shuffle, drop_last=False):
        idx = idx.copy()
        if shuffle: np.random.shuffle(idx)
        end = len(idx) - (len(idx) % bs if drop_last and len(idx) > bs else 0)
        for i in range(0, max(end, 1), bs): yield idx[i:i + bs]

    ntr = int(len(tr) * 0.9); tr_i, va_i = tr[:ntr], tr[ntr:]
    nb_total = max(1, len(tr_i) // bs)
    every = max(1, nb_total // 20)           # ~20 live updates per epoch
    for ep in range(epochs):
        model.train(); tot = 0; nb = 0
        for bi, b in enumerate(batches(tr_i, True, drop_last=True)):
            bt = torch.as_tensor(b, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with amp_autocast(amp):
                out = _fwd(run, is_fnn, bt, T, gb, use_gnn)
                loss = crit(out, y[bt])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_value_(model.parameters(), 5); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
            if on_batch and bi % every == 0:
                on_batch(ep + 1, epochs, bi + 1, nb_total, tot / max(nb, 1))
        model.eval(); vp = []; vy = []
        with torch.no_grad():
            for b in batches(va_i, False):
                bt = torch.as_tensor(b, device=DEVICE)
                with amp_autocast(amp):
                    out = _fwd(run, is_fnn, bt, T, gb, use_gnn)
                vp.append(out.float().cpu().numpy()); vy.append(y[bt].cpu().numpy())
        vrmse = float(np.sqrt(mean_squared_error(np.concatenate(vy), np.concatenate(vp))))
        tl_hist.append(tot / max(nb, 1)); vl_hist.append(vrmse)
        if prog: prog(ep + 1, epochs, tl_hist[-1], vrmse)
        if vrmse < best - 1e-5:
            best, best_state, wait = vrmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); tp = []
    with torch.no_grad():
        for b in batches(te, False):
            bt = torch.as_tensor(b, device=DEVICE)
            with amp_autocast(amp):
                out = _fwd(run, is_fnn, bt, T, gb, use_gnn)
            tp.append(out.float().cpu().numpy())
    return np.concatenate(tp), tl_hist, vl_hist


def metrics(y, p):
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                R2=float(r2_score(y, p)), PCC=float(pearsonr(y, p)[0]), SCC=float(spearmanr(y, p)[0]))


def splitter_for(split, n, drugs, cells, folds):
    if split.startswith("mixed"):
        return KFold(folds, shuffle=True, random_state=42).split(np.arange(n))
    groups = drugs if "drug" in split else cells
    return GroupKFold(folds).split(np.arange(n), groups=groups)


def run_cv(kind, split, meta, T, cfg, folds, pbar=None, status=None, base=0.0, span=1.0, tag=""):
    n = meta["n"]; drugs = meta["drugs"]; cells = meta["cells"]; y = meta["y_used"]
    all_y, all_p, rows, curves = [], [], [], []
    tl0 = vl0 = None; last = None
    for k, (tr, te) in enumerate(splitter_for(split, n, drugs, cells, folds)):
        if pbar is not None: pbar.progress(base + span * k / folds, text=f"{tag}fold {k+1}/{folds}…")
        t0 = time.time()
        torch.manual_seed(42 + k)
        model = make_model(kind, meta, cfg)

        def cb(ep, epochs, tl, vl, _k=k, _t0=t0):
            if status is not None:
                el = time.time() - _t0; eta = el / ep * (epochs - ep)
                status.info(f"🧠 {tag}fold {_k+1}/{folds} · epoch {ep}/{epochs} · "
                            f"train {tl:.3f} · val RMSE {vl:.3f} · {el:0.0f}s elapsed · ~{eta:0.0f}s left (fold)")

        def bcb(ep, epochs, bi, nb, running, _k=k, _t0=t0):
            if status is not None:
                el = time.time() - _t0
                status.info(f"🧠 {tag}fold {_k+1}/{folds} · epoch {ep}/{epochs} · "
                            f"step {bi}/{nb} (size {cfg['batch']}) · train {running:.3f} · {el:0.0f}s elapsed")
        preds, tl, vl = train_fold(model, kind == "fnn", T, np.array(tr), np.array(te), cfg,
                                   gb=meta.get("gb"), prog=cb, on_batch=bcb)
        m = metrics(y[te], preds)
        rows.append({"fold": k + 1, **{q: round(m[q], 3) for q in ["RMSE", "MAE", "R2", "PCC", "SCC"]}})
        all_y.append(y[te]); all_p.append(preds); curves.append((tl, vl))
        if tl0 is None: tl0, vl0 = tl, vl
        last = model
        gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    yt = np.concatenate(all_y); pt = np.concatenate(all_p)
    return {"rows": rows, "yt": yt, "pt": pt, "overall": metrics(yt, pt),
            "tl": tl0, "vl": vl0, "curves": curves, "last": last}


# =========================================================================
# SHAP / pathway attribution
# =========================================================================
def pathway_reason(model, is_fnn, T, sample_idx, path_names, gb=None):
    path, ids, mask, extra, feats, y, didx = T
    model = _unwrap(model); model.eval()
    use_gnn = (not is_fnn) and getattr(model, "use_gnn", False) and gb is not None
    bt = torch.as_tensor(sample_idx, device=DEVICE)
    try:
        import shap
        if is_fnn:
            expl = shap.GradientExplainer(model, feats[torch.as_tensor(sample_idx[:100], device=DEVICE)])
            sv = expl.shap_values(feats[bt]); sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            return sv, "SHAP (GradientExplainer)"
        raise RuntimeError("grad")
    except Exception:
        p = path[bt].clone().requires_grad_(True)
        if is_fnn:
            f = feats[bt].clone().requires_grad_(True); model(f).sum().backward()
            return (f.grad * f).detach().cpu().numpy(), "Gradient×input attribution"
        if use_gnn:
            di = didx[bt]
            model(p, ids[bt], mask[bt], extra[bt], gb[0][di], gb[1][di], gb[2][di]).sum().backward()
        else:
            model(p, ids[bt], mask[bt], extra[bt]).sum().backward()
        return (p.grad * p).detach().cpu().numpy(), "Gradient×input attribution"


# =========================================================================
# Build features from an uploaded zip (unpack -> read -> DELETE)
# =========================================================================
def build_data(files, tokenizer, max_len):
    """Accepts a list of uploaded files: a .zip, the 5 loose CSVs, a prebuilt PathDSP
    .txt/.tsv matrix, and/or a .gmt pathway set. Routes by file type."""
    if not isinstance(files, (list, tuple)): files = [files]
    work = tempfile.mkdtemp(prefix="pathgdt_")
    try:
        pbar = st.progress(0.0, text="Staging uploaded files…")
        gmt_path = prebuilt = None
        for f in files:
            nm = f.name.lower(); dest = os.path.join(work, os.path.basename(f.name))
            with open(dest, "wb") as o: o.write(f.getvalue())
            if nm.endswith(".gmt"):
                gmt_path = dest                                   # pathway-set override
            elif nm.endswith((".txt", ".tsv")):
                try:                                              # is it a prebuilt matrix?
                    hdr = pd.read_csv(dest, sep=None, engine="python", nrows=1)
                    if "resp" in hdr.columns and any(c.startswith("EXP_") for c in hdr.columns):
                        prebuilt = dest
                except Exception:
                    pass
            elif nm.endswith(".zip"):
                with zipfile.ZipFile(dest) as z: z.extractall(work)
        def prog(fr, msg): pbar.progress(0.02 + 0.9 * fr, text=f"Preprocessing — {msg}")
        if prebuilt:
            matrix, drug_smiles = prep.load_prebuilt(prebuilt, REFS, progress=prog)
        else:
            matrix, drug_smiles = prep.build_matrix(work, REFS, progress=prog, gmt_path=gmt_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    pbar.progress(0.95, text="Cleaned up temporary files. Encoding SMILES…")
    df = matrix.reset_index()
    cancer_cols = [c for c in df.columns if c.startswith(("EXP_", "MUT_", "CNV_"))]
    extra_cols = [c for c in df.columns if c.startswith(("DG_", "CHEM_"))]
    tok = SmilesTok(tokenizer, max_len=max_len).fit(sorted(set(drug_smiles.values())))
    ids_np, mask_np = tok.batch([drug_smiles[d] for d in df["drug"].values])
    pbar.empty()   # clear the build progress bar so it doesn't linger under the results
    return dict(P=np.asarray(df[cancer_cols].values, "float32"),
                EX=np.asarray(df[extra_cols].values, "float32"),
                FE=np.asarray(df[cancer_cols + extra_cols].values, "float32"),
                y=np.asarray(df["resp"].values, "float32"),
                drugs=df["drug"].values, cells=df["cell"].values,
                cancer_cols=cancer_cols, extra_cols=extra_cols,
                ids=ids_np, mask=mask_np, tok=tok, drug_smiles=drug_smiles)


def to_tensors(data, normalize, use_gnn=False):
    y = data["y"].astype("float32")
    if normalize:
        mu, sd = float(y.mean()), float(y.std() + 1e-8); y_used = (y - mu) / sd
    else:
        mu, sd = 0.0, 1.0; y_used = y
    drug_order = list(dict.fromkeys(data["drugs"]))            # unique drugs, order preserved
    d2i = {d: i for i, d in enumerate(drug_order)}
    didx = np.array([d2i[d] for d in data["drugs"]], dtype="int64")
    T = (torch.tensor(data["P"], device=DEVICE), torch.tensor(data["ids"], device=DEVICE),
         torch.tensor(data["mask"], device=DEVICE), torch.tensor(data["EX"], device=DEVICE),
         torch.tensor(data["FE"], device=DEVICE), torch.tensor(y_used, device=DEVICE),
         torch.tensor(didx, device=DEVICE))
    gb = None; gnn_fin = GNN_FIN
    if use_gnn:
        GNF, GADJ, GMASK, gnn_fin = build_graph_bank(data["drug_smiles"], drug_order)
        gb = (torch.tensor(GNF, device=DEVICE), torch.tensor(GADJ, device=DEVICE),
              torch.tensor(GMASK, device=DEVICE))
    meta = dict(n=len(y), drugs=data["drugs"], cells=data["cells"], y_used=y_used, y_ln=y,
                cancer_cols=data["cancer_cols"], extra_cols=data["extra_cols"], FE=data["FE"],
                tok_size=data["tok"].size, pad_id=data["tok"].pad_id,
                max_len=data["tok"].max_len, norm=(mu, sd), gb=gb, gnn_fin=gnn_fin)
    return T, meta


def free_vram():
    gc.collect()
    if TORCH and DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()


# =========================================================================
# Predict with a saved model on a freshly-built matrix
# =========================================================================
def predict_with_bundle(bundle, data):
    tok = SmilesTok.load(bundle["tok"])
    model = build_from_bundle(bundle, tok)
    df_cols_c, df_cols_e = bundle["cancer_cols"], bundle["extra_cols"]
    # align new features to the saved schema
    dfc = pd.DataFrame(data["P"], columns=data["cancer_cols"]).reindex(columns=df_cols_c, fill_value=0.0)
    dfe = pd.DataFrame(data["EX"], columns=data["extra_cols"]).reindex(columns=df_cols_e, fill_value=0.0)
    P = torch.tensor(dfc.values.astype("float32"), device=DEVICE)
    EX = torch.tensor(dfe.values.astype("float32"), device=DEVICE)
    FE = torch.tensor(np.concatenate([dfc.values, dfe.values], 1).astype("float32"), device=DEVICE)
    ids_np, mask_np = tok.batch([data["drug_smiles"][d] for d in data["drugs"]])
    ids = torch.tensor(ids_np, device=DEVICE); mask = torch.tensor(mask_np, device=DEVICE)
    is_fnn = bundle["model_kind"] == "fnn"
    use_gnn = (not is_fnn) and bundle["cfg"].get("use_gnn", False)
    if use_gnn:
        drug_order = list(dict.fromkeys(data["drugs"])); d2i = {d: i for i, d in enumerate(drug_order)}
        GNF, GADJ, GMASK, _ = build_graph_bank(data["drug_smiles"], drug_order)
        GNF = torch.tensor(GNF, device=DEVICE); GADJ = torch.tensor(GADJ, device=DEVICE)
        GMASK = torch.tensor(GMASK, device=DEVICE)
        didx = np.array([d2i[d] for d in data["drugs"]], dtype="int64")
    preds = []
    with torch.no_grad():
        for i in range(0, len(data["drugs"]), 256):
            sl = slice(i, i + 256)
            with amp_autocast(DEVICE.type == "cuda"):
                if is_fnn:
                    out = model(FE[sl])
                elif use_gnn:
                    di = torch.as_tensor(didx[sl], device=DEVICE)
                    out = model(P[sl], ids[sl], mask[sl], EX[sl], GNF[di], GADJ[di], GMASK[di])
                else:
                    out = model(P[sl], ids[sl], mask[sl], EX[sl])
            preds.append(out.float().cpu().numpy())
    pred = np.concatenate(preds)
    mu, sd = bundle.get("norm", (0.0, 1.0))
    ln = pred * sd + mu                      # -> LN(IC50) space
    return pred, ln


def roc_block(ln_actual, ln_pred):
    """Binarise sensitivity at the median LN(IC50); ROC using -pred as score."""
    thr = float(np.median(ln_actual))
    y = (ln_actual < thr).astype(int)        # sensitive = lower IC50
    if y.min() == y.max(): return None
    score = -ln_pred
    auc = roc_auc_score(y, score); fpr, tpr, _ = roc_curve(y, score)
    return {"auc": auc, "fpr": fpr, "tpr": tpr, "thr": thr, "n_sens": int(y.sum()), "n": len(y)}


def save_final_model(kind, meta, cfg, data, split, normalize, T, status, pbar):
    """Train a final model on ALL rows and save it to /models; return download fields."""
    pbar.progress(0.97, text="Training final model on ALL data for saving…")
    torch.manual_seed(7)
    final = make_model(kind, meta, cfg)
    train_fold(final, kind == "fnn", T, np.arange(meta["n"]), np.arange(min(256, meta["n"])), cfg,
               gb=meta.get("gb"),
               prog=lambda e, ep, tl, vl: status.info(f"💾 final model · epoch {e}/{ep} · val RMSE {vl:.3f}"))
    bundle = {"model_kind": kind, "cfg": cfg, "cancer_cols": meta["cancer_cols"],
              "extra_cols": meta["extra_cols"], "tok": data["tok"].state(),
              "norm": meta["norm"], "split": split, "normalize": normalize,
              "state_dict": _unwrap(final).state_dict()}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    fname = f"pathgdt_{kind}_{split.split()[0].replace('(', '')}_{stamp}.pt"
    path = os.path.join(MODELS_DIR, fname)
    torch.save(bundle, path)
    buf = io.BytesIO(); torch.save(bundle, buf)
    del final
    return {"saved_path": path, "model_bytes": buf.getvalue(), "model_fname": fname}


# =========================================================================
# UI
# =========================================================================
st.set_page_config(page_title="PathGDT", page_icon="🧬", layout="wide")
st.title("🧬 PathGDT — Explainable Drug-Sensitivity (full test frontend)")

if not TORCH:
    st.error(f"PyTorch not installed. `pip install torch --index-url https://download.pytorch.org/whl/cu124`\n\n{TORCH_ERR}")
    st.stop()
if DEVICE.type == "cuda":
    st.success(f"⚡ GPU: {torch.cuda.get_device_name(0)} (cuda:0) — TF32 + AMP on")
else:
    st.warning("No CUDA GPU detected — running on CPU. Install the CUDA build of torch.")

mode = st.sidebar.radio("Mode", ["🚂 Train & evaluate", "🔮 Predict with a saved model"])


def cfg_from_ui():
    p = PRESETS[preset_name]
    return dict(d=p["d"], layers=p["layers"], heads=p["heads"], max_len=p["max_len"],
                epochs=epochs, batch=batch, lr=lr, patience=patience, amp=amp,
                compile=use_compile, path_sa=path_sa, use_gnn=use_gnn, gnn_fin=GNN_FIN)


def ensure_data(files, tokenizer, max_len):
    h = hashlib.md5()
    for f in sorted(files, key=lambda x: x.name):
        b = f.getvalue(); h.update(f.name.encode()); h.update(str(len(b)).encode()); h.update(b[:100_000])
    key = h.hexdigest() + f"_{tokenizer}_{max_len}"
    if st.session_state.get("data_key") != key:
        st.session_state["data"] = build_data(files, tokenizer, max_len)
        st.session_state["data_key"] = key
    return st.session_state["data"]


# =============================== TRAIN MODE ===============================
if mode.startswith("🚂"):
    with st.sidebar:
        st.header("1 · Data")
        up = st.file_uploader("Upload: a .zip, the 5 CSVs, a prebuilt .txt matrix, and/or a .gmt",
                              type=["zip", "csv", "txt", "tsv", "gmt"], accept_multiple_files=True)
        st.caption("zip **or** loose CSVs → full build · prebuilt .txt → fast path (skips ssGSEA) · "
                   ".gmt → custom pathway set")
        st.header("2 · Model & quality")
        model_name = st.radio("Architecture", ["PathGDT (BPE + cross-attention)", "FNN baseline (PathDSP)"])
        is_fnn = model_name.startswith("FNN")
        preset_name = st.selectbox("Quality preset", list(PRESETS.keys()), index=1)
        tokenizer = st.selectbox("SMILES tokenizer", ["bpe", "regex"], index=0)
        st.header("3 · Cross-validation")
        split = st.selectbox("Split (unseen test)",
                             ["mixed (random)", "leave-cell-out", "leave-drug-out",
                              "ALL (mixed + LOCO + LODO)"])
        folds = st.slider("CV folds", 2, 10, 5)
        normalize = st.checkbox("Z-score response (paper-comparable RMSE)", value=True)
        _p = PRESETS[preset_name]
        with st.expander("Advanced overrides"):
            epochs = st.slider("Max epochs / fold", 5, 100, _p["epochs"], 1)
            batch = st.select_slider("Batch size", [16, 32, 48, 64, 96, 128, 192, 256, 384], value=_p["batch"])
            lr = st.select_slider("Learning rate", [1e-4, 2e-4, 3e-4, 5e-4, 1e-3], value=_p["lr"])
            patience = st.slider("Early-stopping patience", 3, 25, 10)
            path_sa = st.checkbox("Pathway self-attention (higher quality, MUCH slower)", value=False)
            use_compile = st.checkbox("torch.compile (usually NOT worth it on a laptop GPU)", value=False)
        use_gnn = st.checkbox("GNN drug-graph branch (Transformer + GNN)", value=True,
                              help="Adds a molecular-graph GCN to the drug encoder → the full PathGDT. "
                                   "Only affects PathGDT (FNN ignores it). Needs RDKit.")
        amp = st.checkbox("Mixed precision / max VRAM efficiency (AMP)", value=True)
        run_shap = st.checkbox("SHAP pathway explanation", value=True)
        save_model = st.checkbox("Save trained model to /models", value=True)
        c1, c2 = st.columns(2)
        go = c1.button("▶ Run", type="primary", use_container_width=True)
        cmp = c2.button("▶ Compare 4", use_container_width=True)
        if st.button("🧹 Free GPU memory", use_container_width=True):
            free_vram(); st.toast("VRAM cache cleared.")

    cfg = cfg_from_ui()
    kind = "fnn" if is_fnn else "pathgdt"

    if go or cmp:
        if not up:
            st.warning("Upload your data first (zip, CSVs, or a prebuilt .txt)."); st.stop()
        if cfg.get("compile") and not HAS_TRITON:
            st.info("torch.compile needs Triton (not available on Windows) — running without it (no speed loss for you).")
        if cfg.get("path_sa"):
            st.warning("Pathway self-attention is ON — this is the slow O(588²) path. Uncheck it in Advanced for minutes-not-hours runs.")
        with st.spinner("Building features & training…"):
            try:
                data = ensure_data(up, tokenizer, cfg["max_len"])
            except FileNotFoundError as e:
                st.session_state.pop("data_key", None)
                st.error(str(e))
                st.info("Feature-building needs **all 5 files together** (DepMap expression + mutation + CNV "
                        "+ Model.csv + GDSC_DATASET.csv), because GDSC alone has no omics to score.\n\n"
                        "Upload options: **(a)** a zip of all 5, **(b)** the 5 loose CSVs together, or "
                        "**(c)** a single prebuilt `pathdsp_REAL5_input.txt` (skips feature-building entirely).")
                st.stop()
            T, meta = to_tensors(data, normalize, use_gnn=cfg.get("use_gnn", False))
            pbar = st.progress(0.0, text="Starting…"); status = st.empty()

            if cmp:
                cfolds = min(3, folds)
                combos = [("fnn", "mixed (random)"), ("pathgdt", "mixed (random)"),
                          ("fnn", "leave-cell-out"), ("pathgdt", "leave-cell-out")]
                table = []
                for i, (kd, sp) in enumerate(combos):
                    r = run_cv(kd, sp, meta, T, cfg, cfolds, pbar, status, i / 4, 1 / 4,
                               tag=f"{kd.upper()} {sp.split()[0]} · ")
                    f = pd.DataFrame(r["rows"]); row = {"model": "PathGDT" if kd == "pathgdt" else "FNN", "split": sp.split()[0]}
                    for q in ["R2", "PCC", "SCC", "RMSE", "MAE"]: row[q] = f"{f[q].mean():.3f}±{f[q].std():.3f}"
                    table.append(row); r["last"] = None      # release fold model from GPU
                st.session_state["res"] = {"type": "compare", "table": table, "cfolds": cfolds, "normalize": normalize}

            elif split.startswith("ALL"):
                sps = ["mixed (random)", "leave-cell-out", "leave-drug-out"]
                table = []; detail = {}
                for i, sp in enumerate(sps):
                    r = run_cv(kind, sp, meta, T, cfg, folds, pbar, status, i / 3, 1 / 3, tag=f"{sp.split()[0]} · ")
                    f = pd.DataFrame(r["rows"]); row = {"split": sp.split()[0]}
                    for q in ["R2", "PCC", "SCC", "RMSE", "MAE"]: row[q] = f"{f[q].mean():.3f}±{f[q].std():.3f}"
                    table.append(row); r["last"] = None; detail[sp] = r    # release fold model from GPU
                res_all = {"type": "all", "table": table, "detail_split": "leave-cell-out",
                           "model_name": model_name, "normalize": normalize, "unit_norm": meta["norm"],
                           **{k: detail["leave-cell-out"][k] for k in ["yt", "pt", "tl", "vl", "curves", "rows"]}}
                if save_model:
                    res_all.update(save_final_model(kind, meta, cfg, data, "ALL", normalize, T, status, pbar))
                st.session_state["res"] = res_all

            else:
                r = run_cv(kind, split, meta, T, cfg, folds, pbar, status, 0.0, 0.9, tag=f"{model_name.split()[0]} · ")
                res = {"type": "single", "model_name": model_name, "split": split, "normalize": normalize,
                       "norm": meta["norm"], "fold_rows": r["rows"], "yt": r["yt"], "pt": r["pt"],
                       "overall": r["overall"], "tl": r["tl"], "vl": r["vl"], "curves": r["curves"],
                       "path_names": meta["cancer_cols"]}
                if run_shap and r["last"] is not None:
                    try:
                        samp = np.random.RandomState(0).choice(meta["n"], size=min(200, meta["n"]), replace=False)
                        attr, method = pathway_reason(r["last"], is_fnn, T, samp, meta["cancer_cols"], gb=meta.get("gb"))
                        if is_fnn: attr = attr[:, :len(meta["cancer_cols"])]
                        res["shap"] = {"names": meta["cancer_cols"], "imp": np.abs(attr).mean(0),
                                       "signed": attr.mean(0), "method": method}
                    except Exception as e:
                        res["shap_err"] = str(e)
                r["last"] = None                  # release fold model from GPU
                if save_model:
                    res.update(save_final_model(kind, meta, cfg, data, split, normalize, T, status, pbar))
                st.session_state["res"] = res
            pbar.empty(); status.empty()          # clear progress UI so only results remain
            meta["gb"] = None                     # release the GPU molecular-graph bank
            try: del T
            except Exception: pass
            free_vram()

# ============================== PREDICT MODE =============================
else:
    st.subheader("🔮 Predict with a saved model")
    existing = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")), key=os.path.getmtime, reverse=True)
    src = st.radio("Model source", ["Pick from /models folder", "Upload a .pt"])
    bundle = None
    if src.startswith("Pick"):
        if existing:
            pick = st.selectbox("Saved models", [os.path.basename(p) for p in existing])
            if st.checkbox("Load this model", value=True):
                bundle = torch.load(os.path.join(MODELS_DIR, pick), map_location=DEVICE, weights_only=False)
        else:
            st.info("No models in /models yet — train one with 'Save trained model' on, or upload a .pt.")
    else:
        upm = st.file_uploader("Upload a saved model (.pt)", type=["pt"])
        if upm is not None:
            bundle = torch.load(io.BytesIO(upm.getvalue()), map_location=DEVICE, weights_only=False)
    if bundle is not None:
        st.caption(f"Loaded: **{bundle['model_kind']}** · trained split *{bundle.get('split','?')}* · "
                   f"{len(bundle['cancer_cols'])} pathway + {len(bundle['extra_cols'])} drug-extra features · "
                   f"normalised={bundle.get('normalize', False)}")
    st.markdown("**New data to predict** — a .zip, the 5 CSVs, or a prebuilt .txt (± .gmt).")
    upd = st.file_uploader("New data", type=["zip", "csv", "txt", "tsv", "gmt"],
                           accept_multiple_files=True, key="predfiles")
    if st.button("🔮 Predict", type="primary"):
        if bundle is None or not upd:
            st.warning("Load a model and upload new data first."); st.stop()
        with st.spinner("Building features & predicting…"):
            tokm = bundle["tok"].get("method", "bpe") if isinstance(bundle["tok"], dict) else "bpe"
            data = build_data(upd, tokm, bundle["cfg"].get("max_len", 140))
            pred, ln = predict_with_bundle(bundle, data)
            out = pd.DataFrame({"drug": data["drugs"], "cell": data["cells"],
                                "pred_LN_IC50": np.round(ln, 4), "pred_IC50_uM": np.round(np.exp(ln), 4)})
            ln_actual = data["y"].astype(float)
            has_actual = float(np.std(ln_actual)) > 1e-6
            if has_actual:
                out.insert(2, "actual_LN_IC50", np.round(ln_actual, 4))
                out.insert(4, "actual_IC50_uM", np.round(np.exp(ln_actual), 4))
            st.session_state["pred"] = {"out": out, "ln_actual": ln_actual, "ln_pred": ln, "has_actual": has_actual}
            free_vram()

    pr = st.session_state.get("pred")
    if pr is not None:
        st.success(f"Predicted {len(pr['out']):,} drug–cell pairs.")
        if pr["has_actual"]:
            m = metrics(pr["ln_actual"], pr["ln_pred"])
            c = st.columns(5)
            for col, k in zip(c, ["RMSE", "MAE", "R2", "PCC", "SCC"]): col.metric(k, f"{m[k]:.3f}")
            st.caption("Metrics in native LN(IC50) units, computed on the uploaded samples.")
            rb = roc_block(pr["ln_actual"], pr["ln_pred"])
            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(pr["ln_actual"], pr["ln_pred"], s=6, alpha=0.35, edgecolor="none")
                lo, hi = pr["ln_actual"].min(), pr["ln_actual"].max()
                ax.plot([lo, hi], [lo, hi], "r--", lw=1)
                ax.set_xlabel("Actual LN(IC50)"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual")
                fig.tight_layout(); st.pyplot(fig)
            with g2:
                if rb:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(rb["fpr"], rb["tpr"], lw=2, label=f"AUC = {rb['auc']:.3f}")
                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                    ax.set_title(f"ROC — sensitive vs resistant (thr LN={rb['thr']:.2f})"); ax.legend(loc="lower right")
                    fig.tight_layout(); st.pyplot(fig)
                    st.caption(f"Sensitivity = LN(IC50) below the median · {rb['n_sens']}/{rb['n']} labelled sensitive.")
        st.dataframe(pr["out"].head(1000), use_container_width=True, hide_index=True)
        st.download_button("⬇ Predictions with IC50 (.csv)", pr["out"].to_csv(index=False).encode(),
                           "pathgdt_predictions_ic50.csv")
    st.stop()

# =========================================================================
# Results (train mode)
# =========================================================================
res = st.session_state.get("res")
if res is None:
    st.info("Upload your data zip, pick settings, then **▶ Run** (or **▶ Compare 4**)."); st.stop()

if res["type"] in ("compare", "all"):
    title = "4-way comparison (FNN & PathGDT × mixed & LOCO)" if res["type"] == "compare" \
            else f"ALL splits — {res['model_name']}"
    st.subheader(title)
    st.caption(("z-scored response (RMSE comparable to PathDSP)" if res["normalize"] else "native −logIC50 units")
               + (f" · {res['cfolds']}-fold each" if res["type"] == "compare" else " · CV over your chosen folds"))
    cdf = pd.DataFrame(res["table"]); st.dataframe(cdf, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cdf2 = cdf.copy(); cdf2["R2m"] = cdf2["R2"].str.split("±").str[0].astype(float)
    if res["type"] == "compare":
        for j, mdl in enumerate(["FNN", "PathGDT"]):
            sub = cdf2[cdf2["model"] == mdl]
            ax.bar(np.arange(len(sub)) + j * 0.35, sub["R2m"], width=0.35, label=mdl)
        ax.set_xticks(np.arange(len(sub)) + 0.175); ax.set_xticklabels(sub["split"].values); ax.legend()
    else:
        ax.bar(cdf2["split"], cdf2["R2m"], color=["#1f77b4", "#d62728", "#2ca02c"])
    _lo = min(0.0, float(cdf2["R2m"].min()))
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("R²"); ax.set_ylim(_lo - 0.05, 1); ax.set_title("R² by split (can be < 0 on unseen drugs)")
    fig.tight_layout(); st.pyplot(fig)
    st.download_button("⬇ Download table (.csv)", cdf.to_csv(index=False).encode(), "pathgdt_comparison.csv")
    if res["type"] == "all":
        st.caption(f"Scatter/curves below = detail for **{res['detail_split']}** (the honest unseen-cell test).")
        res = {**res, "type": "single", "overall": metrics(res["yt"], res["pt"]),
               "fold_rows": res["rows"], "model_name": res["model_name"], "split": res["detail_split"],
               "norm": res["unit_norm"]}
    else:
        st.stop()

# ---- single-run (or ALL detail) ----
o = res["overall"]; unit = "z-score units" if res["normalize"] else "−logIC50 units"
st.subheader(f"Cross-validation — {res['model_name']} · {res['split']}")
st.caption(f"Metrics in {unit}. R²/PCC/SCC are scale-free; RMSE/MAE depend on the unit.")
c = st.columns(5)
for col, k in zip(c, ["RMSE", "MAE", "R2", "PCC", "SCC"]): col.metric(k, f"{o[k]:.3f}")
fdf = pd.DataFrame(res["fold_rows"])
mean = fdf[["RMSE", "MAE", "R2", "PCC", "SCC"]].mean(); std = fdf[["RMSE", "MAE", "R2", "PCC", "SCC"]].std()
st.caption("Per-fold (mean ± std): " + " · ".join(f"{k} {mean[k]:.3f}±{std[k]:.3f}" for k in ["R2", "RMSE", "PCC"]))
st.dataframe(fdf, use_container_width=True, hide_index=True)

# IC50 + ROC (inverse-transform to LN space)
mu, sd = res["norm"]; ln_a = res["yt"] * sd + mu; ln_p = res["pt"] * sd + mu
g1, g2 = st.columns(2)
with g1:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(res["yt"], res["pt"], s=6, alpha=0.35, edgecolor="none")
    lo, hi = min(res["yt"].min(), res["pt"].min()), max(res["yt"].max(), res["pt"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual (pooled CV)")
    fig.tight_layout(); st.pyplot(fig)
with g2:
    rb = roc_block(ln_a, ln_p)
    if rb:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rb["fpr"], rb["tpr"], lw=2, label=f"AUC = {rb['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
        ax.set_title("ROC — sensitive vs resistant"); fig.tight_layout(); st.pyplot(fig)
        st.caption(f"Binarised at median LN(IC50)={rb['thr']:.2f} · {rb['n_sens']}/{rb['n']} sensitive.")

g3, g4 = st.columns(2)
with g3:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    curves = res.get("curves")
    if curves:
        cmap = plt.get_cmap("tab10")
        for i, (tl, vl) in enumerate(curves):
            col = cmap(i % 10)
            ax.plot(vl, color=col, lw=1.7, label=f"fold {i+1}")
            ax.plot(tl, color=col, lw=1.0, ls="--", alpha=0.55)
        ax.set_title("Training curves — all folds (solid = valid, dashed = train)")
        ax.legend(fontsize=7, ncol=2, title="valid RMSE")
    else:
        ax.plot(res["tl"], label="train"); ax.plot(res["vl"], label="valid")
        ax.set_title("Fold-1 training curve"); ax.legend()
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE loss")
    fig.tight_layout(); st.pyplot(fig)
with g4:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(fdf["fold"].astype(str), fdf["R2"]); ax.set_title("R² per fold")
    ax.axhline(0, color="#888", lw=0.8); ax.set_ylim(min(0.0, float(fdf["R2"].min())) - 0.05, 1)
    fig.tight_layout(); st.pyplot(fig)

if "shap" in res:
    s = res["shap"]
    st.subheader("Why did the model predict this? — pathway attribution")
    st.caption(f"Method: {s['method']}. Bars = mean |attribution|; sign = direction "
               "(↓ toward sensitive / lower IC50, ↑ toward resistant).")
    order = np.argsort(s["imp"])[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    vals = s["signed"][order]; colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    ax.barh(range(len(order))[::-1], vals, color=colors)
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([s["names"][i] for i in order], fontsize=8)
    ax.set_xlabel("mean signed attribution (blue ↓ sensitive · red ↑ resistant)")
    ax.set_title("Top 20 pathways driving predictions"); fig.tight_layout(); st.pyplot(fig)
    st.dataframe(pd.DataFrame({"pathway": [s["names"][i] for i in order],
                               "mean|attr|": np.round(s["imp"][order], 4),
                               "direction": ["↑ resistant" if s["signed"][i] > 0 else "↓ sensitive" for i in order]}),
                 use_container_width=True, hide_index=True)
elif "shap_err" in res:
    st.warning("SHAP could not be computed: " + res["shap_err"])

st.subheader("Downloads")
d1, d2 = st.columns(2)
out = pd.DataFrame({"actual": res["yt"], "prediction": res["pt"],
                    "actual_LN_IC50": np.round(ln_a, 4), "pred_LN_IC50": np.round(ln_p, 4),
                    "pred_IC50_uM": np.round(np.exp(ln_p), 4)})
d1.download_button("⬇ Predictions + IC50 (.csv)", out.to_csv(index=False).encode(), "pathgdt_predictions.csv")
if "model_bytes" in res:
    d2.download_button("💾 Trained model (.pt)", res["model_bytes"], res["model_fname"])
    st.caption(f"Also saved to: `{res.get('saved_path','')}` — appears in the Predict-mode dropdown.")
