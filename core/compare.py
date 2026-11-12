"""
Paired comparison of two runs evaluated on the SAME test folds (Phase 1 of the extension).

Two runs are compared fold by fold only when every shared (repeat, fold) holds exactly the same held-out drug–cell
pairs in both runs — checked from preds.csv, not assumed from the settings. Reported for B − A (negative = B better):
  per-fold ΔRMSE, wins, mean ± sd, and three tests on the per-fold differences:
  sign test (exact binomial, two-sided), Wilcoxon signed-rank (two-sided), paired t-test (two-sided).
"""
import hashlib
import math

import numpy as np


def fold_signatures(preds, scheme):
    """(repeat, fold) -> hash of the sorted held-out pairs"""
    p = preds[preds["scheme"] == scheme]
    out = {}
    for (r, f), g in p.groupby(["repeat", "fold"]):
        keys = sorted((g["drug"].astype(str) + "|" + g["cell"].astype(str)).tolist())
        out[(int(r), int(f))] = hashlib.md5("\n".join(keys).encode()).hexdigest()
    return out


def sign_test(wins, n):
    """two-sided exact binomial test of `wins` successes out of n at p = 0.5"""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired(folds_a, preds_a, folds_b, preds_b, scheme):
    sa, sb = fold_signatures(preds_a, scheme), fold_signatures(preds_b, scheme)
    common = sorted(set(sa) & set(sb))
    same = [k for k in common if sa[k] == sb[k]]
    res = {"scheme": scheme, "n_common": len(common), "n_same": len(same),
           "same_folds": len(common) > 0 and len(same) == len(common)}
    if not res["same_folds"]:
        return res
    key = lambda f: list(zip(f["repeat"].astype(int), f["fold"].astype(int)))   # noqa: E731
    fa = folds_a[folds_a["scheme"] == scheme].copy(); fa["k"] = key(fa)
    fb = folds_b[folds_b["scheme"] == scheme].copy(); fb["k"] = key(fb)
    j = fa[["k", "repeat", "fold", "RMSE", "PCC"]].merge(fb[["k", "RMSE", "PCC"]], on="k", suffixes=("_A", "_B"))
    j = j[j["k"].isin(set(same))].sort_values(["repeat", "fold"]).drop(columns="k").reset_index(drop=True)
    d = (j["RMSE_B"] - j["RMSE_A"]).to_numpy(float)
    j["ΔRMSE (B − A)"] = d
    n = len(d)
    wins_b = int((d < 0).sum()); wins_a = int((d > 0).sum()); ties = n - wins_a - wins_b
    res.update(table=j, n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)) if n > 1 else float("nan"),
               wins_b=wins_b, wins_a=wins_a, ties=ties, p_sign=sign_test(wins_b, wins_a + wins_b))
    try:
        from scipy import stats
        res["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and n >= 5 else float("nan")
        res["p_t"] = float(stats.ttest_rel(j["RMSE_B"], j["RMSE_A"]).pvalue) if n >= 2 and np.std(d) > 0 else float("nan")
    except Exception:
        res["p_wilcoxon"] = res["p_t"] = float("nan")
    return res


def verdict(res, alpha=0.05):
    """plain-language reading of a paired result"""
    if not res.get("same_folds"):
        return "Different test folds — these runs cannot be compared fold by fold."
    if abs(res["mean"]) < 1e-12:
        return "No difference: identical fold RMSEs."
    who = "B" if res["mean"] < 0 else "A"
    w = res["wins_b"] if who == "B" else res["wins_a"]
    tests = {"sign": res.get("p_sign"), "Wilcoxon": res.get("p_wilcoxon"), "t-test": res.get("p_t")}
    tests = {k: v for k, v in tests.items() if v is not None and np.isfinite(v)}
    passed = [k for k, v in tests.items() if v < alpha]
    failed = [k for k, v in tests.items() if v >= alpha]
    head = f"{who} has the lower RMSE by {abs(res['mean']):.4f} on average, on {w} of {res['n']} folds"
    if tests and not failed:
        return f"{who} is better. {head}; all tests p < {alpha}."
    if passed:
        return (f"Weak evidence for {who}. {head}; {', '.join(passed)} p < {alpha}, "
                f"{', '.join(failed)} not. Treat as a small difference that needs more folds or repeats.")
    return f"No clear difference. {head}; no test reaches p < {alpha}."
