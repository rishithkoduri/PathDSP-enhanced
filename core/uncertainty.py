"""
Prediction intervals with cross-validated conformal prediction (Phase 1 of the extension).

Every held-out prediction of a run gets an interval  pred ± q  whose width q comes from the absolute errors of the
SAME run's held-out predictions in the OTHER folds (CV+-style split conformal: the pair being evaluated, and its
whole test fold, never take part in its own calibration). Nothing is retrained, so it works on every finished run.

  global    one q per test fold: the conformal quantile of all other folds' errors
  per drug  one q per (test fold, drug): the conformal quantile of that drug's errors in the other folds, when the
            drug has at least MIN_N of them (else the global q). This is "Mondrian" conformal prediction with the
            drug as the category: coverage is aimed at every drug, not only on average over drugs.
            Leave-drug-out has no other-fold errors of the held-out drug, so it falls back to global automatically.

conformal quantile of n errors at level 1 − α = the ⌈(n + 1)(1 − α)⌉-th smallest (the finite-sample correction).
Errors are in −log10 IC50 (predictions are converted back from the z target before anything is computed).
"""
import math

import numpy as np
import pandas as pd

MIN_N = 30                       # fewest other-fold errors of a drug for its own interval width
LEVELS = (0.80, 0.90, 0.95)


def conformal_q(err, level):
    """finite-sample conformal quantile of absolute errors (inf when there are too few to reach the level)"""
    e = np.sort(np.asarray(err, float))
    n = len(e)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * level)
    return float(e[k - 1]) if k <= n else float("inf")


def intervals(preds, level=0.90, mode="drug", min_n=MIN_N):
    """preds: held-out rows of ONE scheme (repeat, fold, drug, cell, actual, pred). Returns a copy with
    q (half width), lo, hi, covered, cal ('drug' | 'global' — which calibration set gave q)."""
    out = []
    for _, P in preds.groupby("repeat", sort=True):
        P = P.copy()
        err = (P["actual"] - P["pred"]).abs().to_numpy()
        fold = P["fold"].to_numpy()
        drug = P["drug"].to_numpy()
        q = np.full(len(P), np.nan)
        src = np.full(len(P), "global", dtype=object)
        qg = {k: conformal_q(err[fold != k], level) for k in np.unique(fold)}
        for k, v in qg.items():
            q[fold == k] = v
        if mode == "drug":
            for d in np.unique(drug):
                md = drug == d
                ed, fd = err[md], fold[md]
                for k in np.unique(fd):
                    cal = ed[fd != k]
                    if len(cal) >= min_n:
                        sel = md & (fold == k)
                        q[sel] = conformal_q(cal, level)
                        src[sel] = "drug"
        P["q"] = q
        P["lo"] = P["pred"] - q
        P["hi"] = P["pred"] + q
        P["covered"] = (P["actual"] >= P["lo"]) & (P["actual"] <= P["hi"])
        P["cal"] = src
        out.append(P)
    return pd.concat(out, ignore_index=True) if out else preds.assign(q=[], lo=[], hi=[], covered=[], cal=[])


def summary(iv, level):
    """coverage and width overall, per fold and per drug (first repeat for the per-drug view)"""
    if iv.empty:
        return {}
    fin = np.isfinite(iv["q"])
    first = iv[iv["repeat"] == iv["repeat"].min()]
    per_drug = (first.groupby("drug")
                .agg(pairs=("covered", "size"), coverage=("covered", "mean"), half_width=("q", "mean"))
                .reset_index())
    tol = 0.05
    ok = per_drug["coverage"].between(level - tol, 1.0)
    per_fold = iv.groupby(["repeat", "fold"])["covered"].mean()
    return {"level": level, "coverage": float(iv["covered"].mean()), "half_width": float(iv.loc[fin, "q"].mean()),
            "median_half_width": float(iv.loc[fin, "q"].median()), "share_drug_cal": float((iv["cal"] == "drug").mean()),
            "fold_cov_min": float(per_fold.min()), "fold_cov_max": float(per_fold.max()),
            "drugs": int(len(per_drug)), "drugs_ok": int(ok.sum()), "per_drug": per_drug}


def drug_widths(preds, level=0.90, min_n=MIN_N):
    """q per drug from ALL held-out errors of a run (+ the global q) — the interval width the Predict page puts
    around a new prediction of that run's model"""
    if preds.empty:
        return {}, float("nan")
    first = preds[preds["repeat"] == preds["repeat"].min()]
    err = (first["actual"] - first["pred"]).abs()
    qd = {d: conformal_q(g.to_numpy(), level) for d, g in err.groupby(first["drug"]) if len(g) >= min_n}
    return qd, conformal_q(err.to_numpy(), level)


def per_drug_table(preds, iv=None):
    """per drug: pairs, RMSE, MAE, PCC, mean / sd of the measured response (+ coverage and width with iv)"""
    first = preds[preds["repeat"] == preds["repeat"].min()]
    rows = []
    for d, g in first.groupby("drug"):
        a, p = g["actual"].to_numpy(float), g["pred"].to_numpy(float)
        e = p - a
        pcc = float(np.corrcoef(a, p)[0, 1]) if len(a) > 2 and a.std() > 1e-12 and p.std() > 1e-12 else float("nan")
        rows.append({"drug": d, "pairs": len(g), "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                     "PCC": pcc, "mean measured": float(a.mean()), "sd measured": float(a.std())})
    t = pd.DataFrame(rows)
    if iv is not None and not iv.empty and len(t):
        f = iv[iv["repeat"] == iv["repeat"].min()].groupby("drug").agg(coverage=("covered", "mean"), half_width=("q", "mean"))
        t = t.merge(f.reset_index(), on="drug", how="left")
    return t.sort_values("RMSE").reset_index(drop=True) if len(t) else t
