"""
Metrics (in the dataset's response unit: −log10 IC50, or AAC for the CSA studies).

Per split scheme the app reports:
  RMSE, MAE     mean ± sd — mixed: over repeats (paper: "repeated five times … standard deviation between runs"),
                or over the k folds when there is one repeat; leave-one-out: over held-out cell lines / drugs
                (the paper's LOCO / LODO numbers)
  R², PCC, SCC  on all held-out predictions pooled (mean ± sd over repeats when repeated)
  per-drug PCC  mean over drugs of the Pearson correlation between measured and predicted response across that
                drug's cell lines — the cell-specific signal that drug potency alone cannot explain
  drug-mean baseline RMSE  every test pair predicted by its drug's mean response in the training fold
  CSA splits    every metric is the mean ± sd over the benchmark's 10 official splits (as the benchmark reports)
  leave-study-out  every metric is the mean ± sd over the 5 held-out studies
"""
import numpy as np
import pandas as pd


def _corr(a, b, rank=False):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    if rank:
        a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def regression(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "PCC": np.nan, "SCC": np.nan, "n": 0}
    e = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
            "R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
            "PCC": _corr(y, p), "SCC": _corr(y, p, rank=True), "n": int(len(y))}


def per_drug_pcc(df):
    v = [_corr(g["actual"].to_numpy(), g["pred"].to_numpy()) for _, g in df.groupby("drug", sort=False)]
    v = np.array([x for x in v if np.isfinite(x)])
    return float(v.mean()) if len(v) else float("nan")


def _ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def summarize(preds, folds, scheme, grouping="logo"):
    """preds: rows (repeat, fold, drug, cell, actual, pred[, base]) of ONE scheme; folds: per-fold table of it."""
    out = {"scheme": scheme, "n_pred": int(len(preds)), "n_folds": int(len(folds))}
    if len(preds) == 0:
        return out
    reps = sorted(preds["repeat"].unique())
    if scheme in ("csa", "loso"):                           # CSA splits / leave-study-out: mean ± sd over the folds
        per_fold = [regression(g["actual"], g["pred"]) for _, g in preds.groupby("fold")]
        for q in ("RMSE", "MAE", "R2", "PCC", "SCC"):
            out[q], out[q + "_sd"] = _ms([m[q] for m in per_fold])
        out["per_drug_PCC"] = float(np.nanmean([per_drug_pcc(g) for _, g in preds.groupby("fold")]))
        if "base" in preds.columns and preds["base"].notna().all():
            out["base_RMSE"] = float(np.mean([regression(g["actual"], g["base"])["RMSE"] for _, g in preds.groupby("fold")]))
        if "best_epoch" in folds.columns and len(folds):
            out["best_epoch"] = float(folds["best_epoch"].mean())
            out["epochs_run"] = float(folds["epochs_run"].mean())
        out["repeats"] = 1
        return out
    per_rep = {r: regression(g["actual"], g["pred"]) for r, g in preds.groupby("repeat")}
    leave_one = scheme != "mixed" and grouping == "logo"
    for q in ("RMSE", "MAE"):
        if leave_one:                                       # paper: mean ± sd over held-out groups
            out[q], out[q + "_sd"] = _ms(folds[q].to_numpy())
        elif len(reps) > 1:                                 # paper: sd between repeated runs
            out[q], out[q + "_sd"] = _ms([per_rep[r][q] for r in reps])
        else:                                               # one run: sd over its folds
            out[q] = per_rep[reps[0]][q]
            out[q + "_sd"] = _ms(folds[q].to_numpy())[1]
    for q in ("R2", "PCC", "SCC"):
        vals = [per_rep[r][q] for r in reps]
        out[q], out[q + "_sd"] = _ms(vals)
    first = preds[preds["repeat"] == reps[0]]
    out["per_drug_PCC"] = per_drug_pcc(first)
    if "base" in preds.columns and preds["base"].notna().all():
        out["base_RMSE"] = regression(first["actual"], first["base"])["RMSE"]
    if "best_epoch" in folds.columns and len(folds):
        out["best_epoch"] = float(folds["best_epoch"].mean())
        out["epochs_run"] = float(folds["epochs_run"].mean())
    out["repeats"] = len(reps)
    return out


def sensitivity_class(pred, drug_values):
    """Where a predicted response sits among the measured responses of the same drug (higher −log10 IC50 =
    more sensitive). Returns (percentile 0–100 of cell lines it is MORE sensitive than, label)."""
    v = np.asarray(drug_values, float)
    if len(v) == 0:
        return float("nan"), "—"
    pct = float((v < pred).mean() * 100.0)
    label = "Sensitive" if pct >= 75 else ("Resistant" if pct <= 25 else "Intermediate")
    return pct, label
