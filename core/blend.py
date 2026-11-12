"""
Blend of two runs (v5.9) — e.g. a network and LightGBM: pred = w·A + (1 − w)·B on the SAME held-out pairs.

Nothing is retrained. The weight of a fold is fitted on the OTHER folds of the same repeat (cross-fitting): the w that
minimises the squared error of their held-out predictions, w = Σ eB(eB − eA) / Σ (eA − eB)², clipped to [0, 1]
(eA, eB = errors of A and B). A fold's own pairs never choose its weight, so the blend's scores are honest held-out
scores. Leave-study-out: a held-out study's weight comes from the four other held-out studies.

The two runs must hold exactly the same held-out pairs per (repeat, fold) — checked row by row, not assumed.
"""
import math

import numpy as np
import pandas as pd

from . import compare as CP
from . import metrics as MT

KEYS = ["repeat", "fold", "drug", "cell", "actual"]


def aligned(pa, pb, scheme):
    """the held-out rows of one scheme of both runs side by side (pred_a, pred_b), or (None, reason)"""
    a = pa[pa["scheme"] == scheme] if not pa.empty else pa
    b = pb[pb["scheme"] == scheme] if not pb.empty else pb
    if a.empty or b.empty:
        return None, "One of the runs has no held-out predictions for this evaluation."
    if len(a) != len(b):
        return None, f"The runs hold different numbers of held-out pairs ({len(a):,} vs {len(b):,})."
    sa = a.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    sb = b.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    for k in ("repeat", "fold", "drug", "cell"):
        if not (sa[k].astype(str).to_numpy() == sb[k].astype(str).to_numpy()).all():
            return None, "The runs' held-out folds differ (different dataset, seed, folds or repeats)."
    if not np.allclose(sa["actual"].to_numpy(float), sb["actual"].to_numpy(float), atol=1e-5):
        return None, "The runs' measured responses differ — not the same dataset."
    out = sa[["repeat", "fold", "drug", "cell", "actual"]].copy()
    out["pred_a"], out["pred_b"] = sa["pred"].to_numpy(float), sb["pred"].to_numpy(float)
    out["base"] = sa["base"].to_numpy(float) if "base" in sa.columns else np.nan
    return out, ""


def weights(df):
    """cross-fitted weight of A per (repeat, fold): fitted on the other folds of the same repeat"""
    ea = (df["pred_a"] - df["actual"]).to_numpy(float)
    eb = (df["pred_b"] - df["actual"]).to_numpy(float)
    g = df.assign(aa=ea * ea, bb=eb * eb, ab=ea * eb).groupby(["repeat", "fold"])[["aa", "bb", "ab"]].sum()
    out = {}
    for (r, f), _ in g.iterrows():
        o = g[(g.index.get_level_values(0) == r) & (g.index.get_level_values(1) != f)].sum()
        den = o["aa"] - 2 * o["ab"] + o["bb"]
        w = float(np.clip((o["bb"] - o["ab"]) / den, 0.0, 1.0)) if den > 1e-12 else 0.5
        out[(int(r), int(f))] = w
    return out


def _fold_table(p, col):
    rows = []
    for (r, f), g in p.groupby(["repeat", "fold"]):
        m = MT.regression(g["actual"], g[col])
        rows.append({"repeat": int(r), "fold": int(f), **{q: m[q] for q in ("RMSE", "MAE", "R2", "PCC", "SCC")},
                     "per_drug_PCC": MT.per_drug_pcc(pd.DataFrame({"drug": g["drug"].to_numpy(),
                                                                   "actual": g["actual"].to_numpy(float),
                                                                   "pred": g[col].to_numpy(float)}))})
    return pd.DataFrame(rows)


def _tests(d):
    """paired tests on per-fold differences d (negative = the blend better for RMSE)"""
    n = len(d)
    wins = int((d < 0).sum()); losses = int((d > 0).sum())
    out = {"n": n, "mean": float(d.mean()) if n else float("nan"), "wins": wins, "losses": losses,
           "p_sign": CP.sign_test(wins, wins + losses)}
    try:
        from scipy import stats
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue) if n >= 5 and np.any(d != 0) else float("nan")
        out["p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue) if n >= 2 and np.std(d) > 0 else float("nan")
    except Exception:
        out["p_wilcoxon"] = out["p_t"] = float("nan")
    return out


def blend(pa, pb, scheme, grouping="logo"):
    """-> dict(ok, reason | preds, weights, folds, summary{A, B, blend}, vs_a, vs_b)"""
    df, why = aligned(pa, pb, scheme)
    if df is None:
        return {"ok": False, "reason": why}
    if df.groupby("repeat")["fold"].nunique().min() < 2:
        return {"ok": False, "reason": "A blend needs at least two folds per repeat (its weights are fitted on the "
                                       "other folds)."}
    w = weights(df)
    wa = np.array([w[(int(r), int(f))] for r, f in zip(df["repeat"], df["fold"])])
    df["pred"] = wa * df["pred_a"] + (1 - wa) * df["pred_b"]
    ft = {k: _fold_table(df, c) for k, c in (("A", "pred_a"), ("B", "pred_b"), ("blend", "pred"))}
    summ = {}
    for k, c in (("A", "pred_a"), ("B", "pred_b"), ("blend", "pred")):
        p = df[["repeat", "fold", "drug", "cell", "actual", "base"]].assign(pred=df[c].to_numpy())
        summ[k] = MT.summarize(p, ft[k], scheme, grouping)
    tab = ft["blend"][["repeat", "fold"]].copy()
    tab["w_A"] = [w[(int(r), int(f))] for r, f in zip(tab["repeat"], tab["fold"])]
    tab["RMSE_A"], tab["RMSE_B"], tab["RMSE_blend"] = ft["A"]["RMSE"], ft["B"]["RMSE"], ft["blend"]["RMSE"]
    tab["pdPCC_A"], tab["pdPCC_B"], tab["pdPCC_blend"] = (ft["A"]["per_drug_PCC"], ft["B"]["per_drug_PCC"],
                                                          ft["blend"]["per_drug_PCC"])
    preds = df[["repeat", "fold", "drug", "cell", "actual", "pred", "base"]].assign(scheme=scheme, w_A=wa)
    return {"ok": True, "preds": preds, "weights": tab, "summary": summ,
            "vs_a": _tests((tab["RMSE_blend"] - tab["RMSE_A"]).to_numpy(float)),
            "vs_b": _tests((tab["RMSE_blend"] - tab["RMSE_B"]).to_numpy(float)),
            "w_mean": float(np.mean(list(w.values()))), "w_min": float(min(w.values())),
            "w_max": float(max(w.values()))}


def verdict(res, alpha=0.05):
    """plain-language reading: is the blend better than the better of the two runs?"""
    if not res.get("ok"):
        return res.get("reason", "")
    sa, sb = res["summary"]["A"]["RMSE"], res["summary"]["B"]["RMSE"]
    ref, t = ("A", res["vs_a"]) if sa <= sb else ("B", res["vs_b"])
    ps = [v for v in (t.get("p_sign"), t.get("p_wilcoxon"), t.get("p_t")) if v is not None and math.isfinite(v)]
    head = (f"Blend vs the better run ({ref}): RMSE {t['mean']:+.4f} per fold on average, lower on {t['wins']} of "
            f"{t['n']} folds")
    if t["mean"] < 0 and ps and max(ps) < alpha:
        return f"The blend is better. {head}; all tests p < {alpha}."
    if t["mean"] < 0 and ps and min(ps) < alpha:
        return f"Weak evidence for the blend. {head}; only some tests reach p < {alpha}."
    if t["n"] < 6:
        return f"{head}. With {t['n']} folds the tests cannot reach significance — read the per-fold rows."
    return f"No clear gain from blending. {head}; no test reaches p < {alpha}."
