"""
IMPROVE cross-study analysis (CSA) benchmark — five drug-screening studies with PathDSP features (Phase 3).

data/csa/ (built by tools/build_csa.py from the benchmark's raw data):
    features.npz   drug_ids, drug_X [drugs, CHEM 256 + DG 196], cell_ids, cell_X [cells, EXP 187 + MUT 196 + CNV 196]
    response.npz   study, drug, cell (indices into the tables above), y = AAC (1 − AUC), source_row
    splits.npz     '<study>|split_<k>_<train|val|test>' and '<study>|all' — the benchmark's own split files
    drugs.tsv      improve id, name, targets used for DG, SMILES        build.json   provenance + counts

One drug or cell line has the same features in every study, so a model trained on one study can be applied to the
others as they are — the cross-study test of the benchmark:
    G[s, t]  metric of models trained on study s (one per official split) on study t, mean over the 10 splits;
             within-study (t = s): each split's own test set; cross-study (t ≠ s): all of study t
    Ga[s]    mean of G[s, t] over the four other studies        Gn[s, t] = G[s, t] / G[s, s]      Gna = mean Gn
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

STUDIES = ("CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2")
ALL = "ALL"                       # the five studies pooled (multi-study training, leave-study-out)
N_SPLITS = 10
DIR = C.DATA / "csa"
FILES = ("features.npz", "response.npz", "splits.npz", "drugs.tsv", "build.json")
CROSS_COLS = ["scheme", "repeat", "fold", "source", "target", "n", "RMSE", "MAE", "R2", "PCC", "SCC", "per_drug_PCC"]
METRICS = ("RMSE", "MAE", "R2", "PCC", "SCC", "per_drug_PCC")
HIGHER_BETTER = {"RMSE": False, "MAE": False, "R2": True, "PCC": True, "SCC": True, "per_drug_PCC": True}


def key_of(study):
    return f"csa-{study}"


def study_of(key):
    return str(key)[4:] if str(key).startswith("csa-") else ""


def available():
    return all((DIR / f).exists() for f in FILES)


def _sig():
    return tuple(int(os.stat(DIR / f).st_mtime_ns) for f in FILES)


def catalog():
    """dataset entries of the five studies (none when the package is missing)"""
    if not available():
        return {}
    try:
        st = (build_info().get("studies") or {})
    except Exception:
        st = {}
    out = {}
    for s in STUDIES:
        n = (st.get(s) or {}).get("rows")
        out[key_of(s)] = {"name": f"CSA · {s}", "kind": "csa", "builtin": True, "study": s,
                          "label": f"IMPROVE CSA benchmark — {s}" + (f" ({n:,} pairs)" if n else "") + " · response AAC"}
    n = sum(int((st.get(s) or {}).get("rows") or 0) for s in STUDIES)
    out[key_of(ALL)] = {"name": "CSA · all 5 studies", "kind": "csa", "builtin": True, "study": ALL,
                        "label": "IMPROVE CSA benchmark — all five studies pooled" + (f" ({n:,} pairs)" if n else "")
                                 + " · multi-study training · response AAC"}
    return out


@lru_cache(maxsize=2)
def _package(sig):
    with np.load(DIR / "features.npz", allow_pickle=False) as z:
        feats = {k: z[k] for k in z.files}
    with np.load(DIR / "response.npz", allow_pickle=False) as z:
        resp = {k: z[k] for k in z.files}
    with np.load(DIR / "splits.npz", allow_pickle=False) as z:
        splits = {k: z[k].astype(np.int64) for k in z.files}
    drugs = pd.read_csv(DIR / "drugs.tsv", sep="\t", dtype=str).fillna("")
    with open(DIR / "build.json", encoding="utf-8") as fh:
        info = json.load(fh)
    studies = [str(s) for s in resp["studies"]]
    if tuple(studies) != STUDIES:
        raise ValueError(f"data/csa lists the studies {studies}, expected {list(STUDIES)}")
    # display names: the drug's name, or 'name (improve id)' when two drugs share it
    ids = [str(d) for d in feats["drug_ids"]]
    name = dict(zip(drugs["drug"], drugs["name"]))
    raw = [name.get(d) or d for d in ids]
    dup = pd.Series(raw).duplicated(keep=False).to_numpy()
    labels = np.array([f"{n} ({d})" if dd else n for n, d, dd in zip(raw, ids, dup)])
    if len(set(labels.tolist())) != len(labels):
        labels = np.array(ids)
    return {"feats": feats, "resp": resp, "splits": splits, "drugs": drugs, "info": info, "labels": labels}


def package():
    if not available():
        raise FileNotFoundError(f"The CSA benchmark package is not in {DIR}.")
    return _package(_sig())


def build_info():
    with open(DIR / "build.json", encoding="utf-8") as fh:
        return json.load(fh)


def features():
    return [str(f) for f in package()["feats"]["features"]]


def _rows(study):
    """global row positions (into response.npz) of a study, ascending"""
    r = package()["resp"]
    return np.where(r["study"] == STUDIES.index(study))[0]


def dataset(study):
    """one study, or all five pooled (study = ALL) with each row's study in ds.study_codes"""
    from . import data as D
    pk = package()
    f, r = pk["feats"], pk["resp"]
    rows = np.arange(len(r["y"])) if study == ALL else _rows(study)
    di, ci = r["drug"][rows].astype(np.int64), r["cell"][rows].astype(np.int64)
    X = D.PairX(f["drug_X"], f["cell_X"], di, ci)
    cells = np.asarray(f["cell_ids"]).astype(str)
    if study == ALL:
        codes, names, label = r["study"][rows].astype(np.int64), list(STUDIES), "CSA · all 5 studies"
    else:
        codes, names, label = np.zeros(len(rows), dtype=np.int64), [study], f"CSA · {study}"
    return D.Dataset(key_of(study), label, X, r["y"][rows].astype(np.float32), pk["labels"][di],
                     cells[ci], features(), unit="aac", study=study, study_codes=codes, study_names=names)


def splits(study):
    """[(train, val, test)] × 10 as row positions within the study's dataset"""
    pk = package()
    rows = _rows(study)
    out = []
    for k in range(N_SPLITS):
        part = []
        for p in ("train", "val", "test"):
            g = pk["splits"][f"{study}|split_{k}_{p}"]
            loc = np.searchsorted(rows, g)
            if len(g) and (loc.max() >= len(rows) or not np.array_equal(rows[loc], g)):
                raise ValueError(f"{study} split {k} {p}: rows outside the study")
            part.append(loc.astype(np.int64))
        out.append(tuple(part))
    return out


def target_rows(study):
    """(drug index, cell index, y, drug display names) of every pair of a study — its '_all' set, the cross-study
    test set; indices point into the package's drug / cell tables"""
    pk = package()
    g = pk["splits"][f"{study}|all"]
    r = pk["resp"]
    di = r["drug"][g].astype(np.int64)
    return di, r["cell"][g].astype(np.int64), r["y"][g].astype(np.float64), pk["labels"][di]


def drug_info():
    """{display name: {'drug', 'targets', 'pathway', 'improve_id', 'target_source'}} for the Predict page"""
    if not available():
        return {}
    pk = package()
    t = pk["drugs"].set_index("drug")
    out = {}
    for d, lab in zip([str(x) for x in pk["feats"]["drug_ids"]], pk["labels"]):
        row = t.loc[d] if d in t.index else None
        out[str(lab)] = {"drug": str(lab), "targets": (row["targets"] if row is not None else "") or "",
                         "pathway": "", "improve_id": d,
                         "target_source": (row["target_source"] if row is not None else "") or ""}
    return out


# ======================================================================================================================
# cross-study matrix
# ======================================================================================================================
def run_rows(rid):
    """per-split metrics of one run: within-study (its CSA-split folds) + cross-study (cross.csv), long format"""
    from . import store as S
    cfg = S.config(rid)
    src = study_of(cfg.get("data", ""))
    if not src:
        return pd.DataFrame(columns=CROSS_COLS)
    f = S.folds(rid)
    parts = []
    if not f.empty and C.CSA_SCHEME in set(f["scheme"]):
        w = f[f["scheme"] == C.CSA_SCHEME].copy()
        p = S.preds(rid)
        pdp = {}
        if not p.empty:
            from . import metrics as MT
            pp = p[p["scheme"] == C.CSA_SCHEME]
            pdp = {k: MT.per_drug_pcc(g) for k, g in pp.groupby("fold")}
        parts.append(pd.DataFrame({"scheme": w["scheme"], "repeat": w["repeat"], "fold": w["fold"], "source": src,
                                   "target": src, "n": w["n_test"], **{m: w[m] for m in METRICS[:5]},
                                   "per_drug_PCC": w["fold"].map(pdp)}))
    x = S.read_csv(S.run_dir(rid) / "cross.csv")
    if not x.empty:
        parts.append(x)
    return pd.concat(parts, ignore_index=True)[CROSS_COLS] if parts else pd.DataFrame(columns=CROSS_COLS)


def run_table(rid):
    """one row per target study: mean ± sd over the splits of every metric, plus Gn (÷ within-study)"""
    d = run_rows(rid)
    if d.empty:
        return pd.DataFrame()
    src = d["source"].iloc[0]
    g = d.groupby("target")
    t = pd.DataFrame({"splits": g.size()})
    for m in METRICS:
        t[m] = g[m].mean()
        t[m + "_sd"] = g[m].std(ddof=1)
    t["pairs"] = g["n"].max()
    t = t.reindex([s for s in STUDIES if s in t.index])
    if src in t.index:
        for m in METRICS:
            base = t.loc[src, m]
            t[m + "_norm"] = t[m] / base if base and np.isfinite(base) else np.nan
    t.insert(0, "kind", ["within" if s == src else "cross" for s in t.index])
    return t.reset_index().rename(columns={"index": "target"})


def matrix(run_by_source, metric):
    """G, Gn (5 × 5, NaN where missing) and the row summaries Ga, Gna for the chosen runs {source: run id}"""
    G = pd.DataFrame(np.nan, index=list(STUDIES), columns=list(STUDIES))
    nsplit = pd.DataFrame(0, index=list(STUDIES), columns=list(STUDIES))
    for s, rid in run_by_source.items():
        if not rid:
            continue
        d = run_rows(rid)
        if d.empty:
            continue
        g = d.groupby("target")[metric]
        for t, v in g.mean().items():
            G.loc[s, t] = v
        for t, v in g.size().items():
            nsplit.loc[s, t] = int(v)
    diag = pd.Series(np.diag(G.to_numpy()), index=G.index)
    Gn = G.div(diag, axis=0)
    off = ~np.eye(len(STUDIES), dtype=bool)
    Ga = pd.Series([np.nanmean(G.to_numpy()[i][off[i]]) if np.isfinite(G.to_numpy()[i][off[i]]).any() else np.nan
                    for i in range(len(STUDIES))], index=G.index)
    Gna = pd.Series([np.nanmean(Gn.to_numpy()[i][off[i]]) if np.isfinite(Gn.to_numpy()[i][off[i]]).any() else np.nan
                     for i in range(len(STUDIES))], index=G.index)
    return G, Gn, Ga, Gna, nsplit


# ======================================================================================================================
# leave-study-out: few-shot recalibration and the comparison with single-study models
# ======================================================================================================================
FEWSHOT_K = (10, 30, 100, 300, 1000)
FEWSHOT_DRAWS = 20
FEWSHOT_PRIOR = 20          # pseudo-pairs of the shrinkage prior a = 1 (few labels: mostly a shift; many: full fit)
FEWSHOT_LEVEL = 0.90        # conformal interval level


def recalibrate(x, y, v, k0=FEWSHOT_PRIOR):
    """y ≈ a·x + b from labelled pairs (x = prediction, y = measured), the slope shrunk towards 1 with the weight of
    k0 pseudo-pairs: a = (Sxy + k0·v) / (Sxx + k0·v), v = variance of the model's predictions on the whole new study
    (no labels needed). Few pairs -> mostly an offset; many pairs -> the least-squares fit."""
    xm, ym = x.mean(), y.mean()
    sxx, sxy = float(((x - xm) ** 2).sum()), float(((x - xm) * (y - ym)).sum())
    a = (sxy + k0 * v) / (sxx + k0 * v) if (sxx + k0 * v) > 1e-12 else 1.0
    return a, float(ym - a * xm)


def fewshot(preds, folds, seed=42, ks=FEWSHOT_K, draws=FEWSHOT_DRAWS, level=FEWSHOT_LEVEL):
    """How much of the cross-study error is only the new study's AAC scale, and can intervals be trusted there?
    For every held-out study and k = 10 … 1000 labelled pairs of it (random, `draws` times):
      point   y ≈ a·ŷ + b fitted on the k pairs (recalibrate: slope shrunk towards 1), scored on the OTHER pairs
      interval split conformal on the new study: half of the k pairs fit (a, b), the other half give the width q
               (the ⌈(n+1)·level⌉-th smallest |error|); coverage and width are measured on the other pairs
    k = 0 is the model as it is; its interval width comes from the errors on the OTHER held-out studies.
    PCC and per-drug PCC do not change under y = a·ŷ + b with a > 0 — only RMSE / MAE / R² (and intervals) can."""
    from . import metrics as MT
    from .uncertainty import conformal_q
    p = preds[preds["scheme"] == "loso"] if not preds.empty else preds
    if p.empty:
        return pd.DataFrame()
    names = {}
    if not folds.empty:
        f = folds[folds["scheme"] == "loso"]
        names = dict(zip(f["fold"].astype(int), f["group"].astype(str)))
    rng = np.random.RandomState(int(seed))
    err_all = (p["actual"] - p["pred"]).abs().to_numpy(float)
    fold_all = p["fold"].to_numpy()
    rows = []
    for fold, g in p.groupby("fold"):
        y, yh = g["actual"].to_numpy(float), g["pred"].to_numpy(float)
        v = float(yh.var())
        m0 = MT.regression(y, yh)
        q0 = conformal_q(err_all[fold_all != fold], level)
        study = names.get(int(fold), str(fold))
        rows.append({"study": study, "fold": int(fold), "k": 0, "draws": 1, "RMSE": m0["RMSE"], "RMSE_sd": 0.0,
                     "MAE": m0["MAE"], "R2": m0["R2"], "R2_sd": 0.0, "a_pos": 1.0, "n_eval": len(y),
                     "coverage": float((np.abs(y - yh) <= q0).mean()) if np.isfinite(q0) else np.nan,
                     "half_width": q0 if np.isfinite(q0) else np.nan})
        for k in ks:
            if k >= len(y) - 10:
                continue
            res = []
            for _ in range(int(draws)):
                cal = rng.choice(len(y), k, replace=False)
                ev = np.ones(len(y), bool)
                ev[cal] = False
                a, b = recalibrate(yh[cal], y[cal], v)
                m = MT.regression(y[ev], a * yh[ev] + b)
                h1, h2 = cal[: k // 2], cal[k // 2:]
                a1, b1 = recalibrate(yh[h1], y[h1], v)
                q = conformal_q(np.abs(y[h2] - (a1 * yh[h2] + b1)), level)
                if np.isfinite(q):
                    cov = float((np.abs(y[ev] - (a1 * yh[ev] + b1)) <= q).mean())
                else:
                    cov, q = np.nan, np.nan
                res.append((m["RMSE"], m["MAE"], m["R2"], a > 0, cov, q))
            r = np.array(res, float)
            rows.append({"study": study, "fold": int(fold), "k": int(k), "draws": int(draws),
                         "RMSE": r[:, 0].mean(), "RMSE_sd": r[:, 0].std(ddof=1), "MAE": r[:, 1].mean(),
                         "R2": r[:, 2].mean(), "R2_sd": r[:, 2].std(ddof=1), "a_pos": r[:, 3].mean(),
                         "n_eval": int(len(y) - k),
                         "coverage": float(np.nanmean(r[:, 4])) if np.isfinite(r[:, 4]).any() else np.nan,
                         "half_width": float(np.nanmean(r[:, 5])) if np.isfinite(r[:, 5]).any() else np.nan})
    return pd.DataFrame(rows)


def loso_table(preds, folds):
    """per held-out study: pairs and every metric of the leave-study-out model (zero-shot)"""
    from . import metrics as MT
    p = preds[preds["scheme"] == "loso"] if not preds.empty else preds
    if p.empty:
        return pd.DataFrame()
    names = {}
    if not folds.empty:
        f = folds[folds["scheme"] == "loso"]
        names = dict(zip(f["fold"].astype(int), f["group"].astype(str)))
    out = []
    for fold, g in p.groupby("fold"):
        m = MT.regression(g["actual"], g["pred"])
        out.append({"study": names.get(int(fold), str(fold)), "pairs": len(g), "drugs": g["drug"].nunique(),
                    **{k: m[k] for k in ("RMSE", "MAE", "R2", "PCC", "SCC")}, "per_drug_PCC": MT.per_drug_pcc(g)})
    t = pd.DataFrame(out)
    t["order"] = t["study"].map({s: i for i, s in enumerate(STUDIES)})
    return t.sort_values("order").drop(columns="order").reset_index(drop=True)


def single_source_vs(run_by_source, metric):
    """for every target study t: the single-study models' G[s, t] over the sources s ≠ t of the chosen runs —
    best, mean and which source was best"""
    G = matrix(run_by_source, metric)[0]
    hb = HIGHER_BETTER[metric]
    rows = []
    for t in STUDIES:
        v = G[t].drop(index=t).dropna()
        if v.empty:
            rows.append({"study": t, "single_best": np.nan, "single_mean": np.nan, "best_source": "", "sources": 0})
            continue
        best = v.idxmax() if hb else v.idxmin()
        rows.append({"study": t, "single_best": float(v[best]), "single_mean": float(v.mean()), "best_source": best,
                     "sources": int(len(v))})
    return pd.DataFrame(rows)
