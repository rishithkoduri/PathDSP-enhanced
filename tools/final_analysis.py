"""
Final validation plan (v6.0) — the analysis of every step. tools/final_plan.py calls these functions in order; each
writes its tables and a short Markdown report into final/<step>/ and returns a small dict for final/state.json.

  step1   the recipe on FRESH folds (seed 43): Paper exact vs Paper exact + C·D×3·E·F·G·Z, paired fold by fold
  step2   leakage audit: checks on the saved run files + four leak demonstrations (clean vs leaky, same folds)
  step3   do the explanations point to the drug's real targets? Precision@K / AUROC vs the PID pathways that
          contain the drug's annotated targets, a permutation control, a deletion test and seed-to-seed stability
  step4   unseen cell lines / unseen drugs (grouped 10-fold): Paper exact vs the recipe, with simple baselines
  finalise  reads steps 1–4 and fixes the final recipe of the train-on-everything models

Only the training runs use the GPU (step 3 computes SHAP on it when there is one). Units: −log10 IC50.
"""
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from core import compare as CP        # noqa: E402
from core import config as C         # noqa: E402
from core import data as D           # noqa: E402
from core import engine as E         # noqa: E402
from core import metrics as MT       # noqa: E402
from core import store as S          # noqa: E402
from core import uncertainty as U    # noqa: E402

OUT = ROOT / "final"
MIN_PAIRS = 8                         # a drug needs this many test pairs for a within-drug correlation
KS = (5, 10, 20)


# ======================================================================================================================
# small helpers
# ======================================================================================================================
def out_dir(step):
    p = OUT / step
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path, obj):
    D.write_json(path, obj)


def _f(x, nd=4):
    """number -> text for the Markdown reports"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if x != 0 and abs(x) < 10 ** (-nd):
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def _p(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _tests(d):
    """two-sided paired tests on differences d (sign, Wilcoxon, t)"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    neg, pos = int((d < 0).sum()), int((d > 0).sum())
    out = {"n": n, "mean": float(d.mean()) if n else float("nan"), "neg": neg, "pos": pos,
           "p_sign": CP.sign_test(neg, neg + pos)}
    try:
        from scipy import stats
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue) if n >= 5 and np.any(d != 0) else float("nan")
        out["p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue) if n >= 2 and np.std(d) > 0 else float("nan")
    except Exception:
        out["p_wilcoxon"] = out["p_t"] = float("nan")
    return out


def auroc(labels, scores):
    """Mann–Whitney AUROC of scores for binary labels (ties averaged); NaN if one class is empty"""
    from scipy.stats import rankdata
    y = np.asarray(labels, bool); s = np.asarray(scores, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_drug_auroc(p, min_n=10):
    """mean over drugs of the AUROC of the prediction for 'more sensitive than the drug's median cell line'
    (higher −log10 IC50 = more sensitive)"""
    v = []
    for _, g in p.groupby("drug", sort=False):
        if len(g) < min_n:
            continue
        a = g["actual"].to_numpy(float)
        lab = a > np.median(a)
        v.append(auroc(lab, g["pred"].to_numpy(float)))
    v = np.array([x for x in v if np.isfinite(x)])
    return float(v.mean()) if len(v) else float("nan")


def run_metrics(p, f, scheme, grouping):
    s = MT.summarize(p, f, scheme, grouping)
    s["within_drug_AUROC"] = within_drug_auroc(p)
    return s


def run_finished(rid):
    return bool(rid) and S.status(rid).get("state") == "finished" and bool(S.summary(rid))


def find_reference(model_spec_match, data="input", seed=42, schemes=("mixed",)):
    """newest finished run in runs/ whose config matches (used only for the side-by-side seed-42 reference)"""
    for r in S.list_runs():
        cfg = r["config"]
        if r["state"] != "finished" or cfg.get("data") != data or int(cfg.get("seed", -1)) != seed:
            continue
        if list(cfg.get("schemes", [])) != list(schemes):
            continue
        if model_spec_match(cfg):
            return r["id"]
    return None


def _is_plain_paper(cfg):
    sp = cfg.get("spec", {})
    return (cfg.get("model") == "paper" and int(sp.get("batch", 0)) == 8 and not sp.get("plateau")
            and int(sp.get("ensemble", 1) or 1) == 1 and not sp.get("ema") and sp.get("loss", "rmse") == "rmse"
            and sp.get("val_metric", "batch") == "batch" and sp.get("target", "raw") == "raw"
            and int(sp.get("repeats", 1)) == 1)


def _is_recipe_paper(cfg):
    sp = cfg.get("spec", {})
    return (cfg.get("model") == "paper" and int(sp.get("batch", 0)) == 8 and sp.get("plateau")
            and int(sp.get("ensemble", 1) or 1) == 3 and sp.get("ema") and sp.get("loss") == "mse"
            and sp.get("val_metric") == "global" and sp.get("target") == "drug_z" and int(sp.get("repeats", 1)) == 1)


# ======================================================================================================================
# step 1 — the recipe on fresh folds
# ======================================================================================================================
def _member_fold_table(rid, scheme, member=1):
    m = S.read_csv(S.run_dir(rid) / "members.csv")
    if m.empty:
        return pd.DataFrame()
    return m[(m["scheme"] == scheme) & (m["member"] == member)][["repeat", "fold", "RMSE", "PCC"]].copy()


def compare_pair(rid_a, rid_b, scheme, label_a, label_b):
    """B vs A on the same folds -> dict (paired tests on per-fold RMSE, B − A; negative = B better)"""
    fa, pa, fb, pb = S.folds(rid_a), S.preds(rid_a), S.folds(rid_b), S.preds(rid_b)
    res = CP.paired(fa, pa, fb, pb, scheme)
    if not res.get("same_folds"):
        raise RuntimeError(f"{label_a} and {label_b} do not hold the same test folds for '{scheme}' "
                           f"({res.get('n_same')} of {res.get('n_common')} folds identical).")
    t = res["table"].copy()
    out = {"scheme": scheme, "A": label_a, "B": label_b, "run_A": rid_a, "run_B": rid_b, "n": res["n"],
           "mean_dRMSE": res["mean"], "sd_dRMSE": res["sd"], "B_better": res["wins_b"], "A_better": res["wins_a"],
           "ties": res["ties"], "p_sign": res["p_sign"], "p_wilcoxon": res.get("p_wilcoxon"), "p_t": res.get("p_t")}
    return out, t


def step1(rid_plain, rid_recipe):
    od = out_dir("step1_fresh_folds")
    cmp_, table = compare_pair(rid_plain, rid_recipe, "mixed", "Paper exact (plain)", "Paper exact + recipe")
    fa, pa, fb, pb = S.folds(rid_plain), S.preds(rid_plain), S.folds(rid_recipe), S.preds(rid_recipe)
    m_a = run_metrics(pa[pa.scheme == "mixed"], fa[fa.scheme == "mixed"], "mixed", "logo")
    m_b = run_metrics(pb[pb.scheme == "mixed"], fb[fb.scheme == "mixed"], "mixed", "logo")
    # the recipe's single seed (member 1) against plain: separates the seed ensemble (D) from C·E·F·G·Z
    m1 = _member_fold_table(rid_recipe, "mixed", 1)
    single = None
    if not m1.empty:
        j = fa[fa.scheme == "mixed"][["repeat", "fold", "RMSE"]].merge(m1, on=["repeat", "fold"],
                                                                         suffixes=("_plain", "_single"))
        d = (j["RMSE_single"] - j["RMSE_plain"]).to_numpy(float)
        tt = _tests(d)
        single = {"RMSE_single_mean_of_folds": float(j["RMSE_single"].mean()),
                  "RMSE_plain_mean_of_folds": float(j["RMSE_plain"].mean()),
                  "mean_dRMSE": tt["mean"], "single_better": tt["neg"], "plain_better": tt["pos"], "n": tt["n"],
                  "p_sign": tt["p_sign"], "p_wilcoxon": tt["p_wilcoxon"], "p_t": tt["p_t"]}
        table = table.merge(m1.rename(columns={"RMSE": "RMSE_recipe_seed1", "PCC": "PCC_recipe_seed1"}),
                            on=["repeat", "fold"], how="left")
    pw = cmp_.get("p_wilcoxon")
    confirmed = bool(cmp_["mean_dRMSE"] < 0 and cmp_["B_better"] >= 8
                     and pw is not None and np.isfinite(pw) and pw < 0.05)
    # seed-42 reference (the runs of 2 Nov), when they are in runs/
    ref = None
    r_plain = find_reference(_is_plain_paper)
    r_rec = find_reference(_is_recipe_paper)
    if r_plain and r_rec:
        try:
            ref, _ = compare_pair(r_plain, r_rec, "mixed", "Paper exact (plain), seed 42", "Paper exact + recipe, seed 42")
        except Exception as e:
            ref = {"error": str(e)}
    table.rename(columns={"RMSE_A": "RMSE_plain", "RMSE_B": "RMSE_recipe", "PCC_A": "PCC_plain",
                          "PCC_B": "PCC_recipe", "ΔRMSE (B − A)": "dRMSE_recipe_minus_plain"}).to_csv(
        od / "fold_by_fold.csv", index=False)
    res = {"comparison": cmp_, "plain": _slim(m_a), "recipe": _slim(m_b), "single_seed": single,
           "confirmed": confirmed, "reference_seed42": ref, "seed": int(S.config(rid_recipe)["seed"])}
    _write_json(od / "step1.json", res)
    (od / "step1_fresh_folds.md").write_text(_step1_md(res), encoding="utf-8")
    return res


def _slim(m):
    keys = ("RMSE", "RMSE_sd", "MAE", "R2", "PCC", "SCC", "per_drug_PCC", "within_drug_AUROC", "base_RMSE",
            "best_epoch", "epochs_run", "n_pred", "n_folds")
    return {k: m.get(k) for k in keys if k in m}


def _step1_md(r):
    c, a, b = r["comparison"], r["plain"], r["recipe"]
    rows = pd.DataFrame([
        {"Run": "Paper exact (plain)", "RMSE": f"{_f(a['RMSE'])} ± {_f(a.get('RMSE_sd'))}", "MAE": _f(a["MAE"]),
         "R²": _f(a["R2"], 3), "PCC": _f(a["PCC"], 3), "Per-drug PCC": _f(a["per_drug_PCC"], 3),
         "Within-drug AUROC": _f(a["within_drug_AUROC"], 3)},
        {"Run": "Paper exact + C·D×3·E·F·G·Z", "RMSE": f"{_f(b['RMSE'])} ± {_f(b.get('RMSE_sd'))}", "MAE": _f(b["MAE"]),
         "R²": _f(b["R2"], 3), "PCC": _f(b["PCC"], 3), "Per-drug PCC": _f(b["per_drug_PCC"], 3),
         "Within-drug AUROC": _f(b["within_drug_AUROC"], 3)}])
    s = [f"# Step 1 — the recipe on fresh folds (seed {r['seed']})", "",
         "The recipe (C·D×3·E·F·G·Z) was chosen while looking at the test folds of seed 42. Here it is tested on "
         f"new folds (seed {r['seed']}) that played no part in choosing it. Units: −log10 IC50; mixed 10-fold.", "",
         md_table(rows), "",
         f"**Paired test (recipe − plain, per fold):** mean ΔRMSE {_f(c['mean_dRMSE'])}, recipe better on "
         f"{c['B_better']} of {c['n']} folds; sign p = {_p(c['p_sign'])}, Wilcoxon p = {_p(c['p_wilcoxon'])}, "
         f"paired t p = {_p(c['p_t'])}.", ""]
    if r.get("single_seed"):
        q = r["single_seed"]
        s += [f"**Recipe without the ensemble (seed 1 of 3) vs plain:** mean ΔRMSE {_f(q['mean_dRMSE'])}, better on "
              f"{q['single_better']} of {q['n']} folds; Wilcoxon p = {_p(q['p_wilcoxon'])}, t p = {_p(q['p_t'])}.", ""]
    ref = r.get("reference_seed42")
    if ref and "mean_dRMSE" in ref:
        s += [f"**Same comparison on the original seed-42 folds:** mean ΔRMSE {_f(ref['mean_dRMSE'])}, recipe better "
              f"on {ref['B_better']} of {ref['n']} folds (Wilcoxon p = {_p(ref['p_wilcoxon'])}).", ""]
    s += ["**Verdict:** " + ("CONFIRMED — the recipe beats plain Paper exact on fresh folds (≥ 8 of 10 folds, "
                             "Wilcoxon p < 0.05)." if r["confirmed"] else
                             "NOT CONFIRMED on fresh folds — the seed-42 gain does not hold up by the rule "
                             "(≥ 8 of 10 folds and Wilcoxon p < 0.05)."), ""]
    return "\n".join(s)


# ======================================================================================================================
# step 2 — leakage audit
# ======================================================================================================================
def _jobs_of(ds, cfg):
    seen, out = set(), []
    for j in E.make_jobs(ds, cfg):
        if j.key not in seen:
            seen.add(j.key)
            out.append(j)
    return out


def _sig(ds, rows):
    import hashlib
    keys = sorted((ds.drugs[rows].astype(str).astype(object) + "|" + ds.cells[rows].astype(str).astype(object)).tolist())
    return hashlib.md5("\n".join(keys).encode()).hexdigest()


def audit_checks(rid):
    """checks on the files of a finished run (normally step 1's recipe run). -> list of dicts"""
    import torch
    cfg = S.config(rid)
    ds = D.load(cfg["data"])
    jobs = [j for j in _jobs_of(ds, cfg) if j.scheme != "final"]
    preds = S.preds(rid)
    rows = []

    def add(name, how, ok, detail):
        rows.append({"check": name, "how it was verified": how, "result": "PASS" if ok else "FAIL", "detail": detail})

    # 1  split integrity, rebuilt from the run's own settings
    bad = []
    for j in jobs:
        tr, tr_in, va, te = map(lambda a: set(np.asarray(a).tolist()), (j.tr, j.tr_in, j.va, j.te))
        if tr & te or va & te or not va <= tr or (tr_in | va) != tr or tr_in & va:
            bad.append(j.key)
    add("Train / inner-validation / test rows are disjoint in every fold",
        "fold jobs rebuilt from config.json (core.engine.make_jobs)", not bad,
        f"{len(jobs)} folds; inner validation ⊂ training rows; test ∩ training = ∅" + (f"; failing: {bad}" if bad else ""))
    # 2  the saved predictions are exactly those test rows, each pair once per repeat
    sig_ok, n_dup = True, 0
    for s_ in sorted(set(preds["scheme"])):
        p = preds[preds["scheme"] == s_]
        n_dup += int(p.duplicated(["repeat", "drug", "cell"]).sum())
    fsig = CP.fold_signatures(preds, "mixed") if "mixed" in set(preds["scheme"]) else {}
    for j in jobs:
        if j.scheme != "mixed":
            continue
        if fsig.get((j.rep, j.fold)) != _sig(ds, j.te):
            sig_ok = False
    every = all(len(preds[(preds.scheme == s_) & (preds.repeat == r_)]) == ds.n
                for s_ in set(preds["scheme"]) if s_ == "mixed" for r_ in set(preds[preds.scheme == s_]["repeat"]))
    add("Every test prediction comes from a model that never saw that pair", "preds.csv vs the rebuilt test folds",
        sig_ok and n_dup == 0 and every, f"duplicates {n_dup}; each mixed repeat covers all {ds.n:,} pairs once: {every}")
    # 3/4  feature scaling and drug-wise z statistics of the saved best fold = training rows only
    bf = S.run_dir(rid) / "bestfold.pt"
    if bf.exists():
        b = torch.load(bf, map_location="cpu", weights_only=False)
        tr = np.asarray(b["tr"])
        Xtr = np.asarray(ds.X[tr], dtype="float64")
        mu_tr, sd_tr = Xtr.mean(0), Xtr.std(0)
        Xall = np.asarray(ds.X, dtype="float64")
        mu_all = Xall.mean(0)
        mu_s, sd_s = np.asarray(b["mu"], "float64"), np.asarray(b["sd"], "float64")
        if cfg["spec"].get("feat_scale", True):
            d_tr = float(np.abs(mu_s - mu_tr).max())
            d_all = float(np.abs(mu_s - mu_all).max())
            ok_sd = np.allclose(sd_s[sd_tr > 1e-6], sd_tr[sd_tr > 1e-6], rtol=1e-3, atol=1e-4)
            add("Feature z-scoring is fitted on the fold's training rows only",
                "mean / sd stored with the best fold vs recomputed", d_tr < 1e-4 and ok_sd and d_all > d_tr,
                f"max |μ − μ(train rows)| = {d_tr:.1e}; max |μ − μ(all rows)| = {d_all:.1e}")
        inv = b.get("inv", {})
        if inv.get("kind") == "drug_z":
            mu_d, sd_d = E.drug_stats(ds, tr)
            mu_a, _ = E.drug_stats(ds, np.arange(ds.n))
            ok = np.nanmax(np.abs(np.asarray(inv["mu"]) - mu_d)) < 1e-6 and np.nanmax(np.abs(np.asarray(inv["sd"]) - sd_d)) < 1e-6
            add("Drug-wise z target uses each drug's training responses only", "stored drug means / sds vs recomputed",
                bool(ok), f"max |μ − μ(train rows)| = {np.nanmax(np.abs(np.asarray(inv['mu']) - mu_d)):.1e}; "
                          f"vs all rows {np.nanmax(np.abs(np.asarray(inv['mu']) - mu_a)):.1e}")
    else:
        add("Feature scaling / drug-wise z fitted on training rows", "bestfold.pt", False, "bestfold.pt not found")
    # 5  conformal intervals: a fold's interval width never depends on that fold
    p = preds[(preds["scheme"] == "mixed") & (preds["repeat"] == preds["repeat"].min())].reset_index(drop=True)
    iv0 = U.intervals(p, 0.90, "drug")
    k0 = int(sorted(p["fold"].unique())[0])
    p2 = p.copy()
    m = p2["fold"] == k0
    p2.loc[m, "pred"] = p2.loc[m, "pred"] + 5.0                       # wreck fold k0's own predictions
    iv1 = U.intervals(p2, 0.90, "drug")
    same = np.allclose(iv0.loc[m.values, "q"].to_numpy(), iv1.loc[m.values, "q"].to_numpy())
    add("Conformal interval of a fold is calibrated on the other folds only",
        f"fold {k0}'s predictions shifted by +5; its interval widths recomputed", bool(same),
        "widths unchanged" if same else "widths changed")
    # 6  early stopping / best epoch on the inner validation rows (by construction; shown by demo L4)
    add("Early stopping and the best epoch use the inner validation rows, never the test rows",
        "code: engine.Pool scores job.va; test rows are predicted once, after training (runner._fold_done); "
        "demo L4 shows what test-based stopping would do", True, "by construction")
    add("Response labels are not used to build features",
        "the authors' ssGSEA / NetPEA scores and Morgan bits are computed per cell line / per drug without any "
        "drug response", True, "label-free (computed once for all cell lines: transductive, but no labels)")
    return pd.DataFrame(rows)


def _ridge_cv(ds, jobs, alpha, scale="fold", select=None, target="raw", z_stats="fold"):
    """Ridge regression over the folds (CPU). scale: 'fold' | 'all' (leak); select: None or k features chosen by
    |corr(x, y)| on 'fold' or 'all' rows ((k, mode)); target 'raw' | 'drug_z' with z_stats 'fold' | 'all' (leak)."""
    from sklearn.linear_model import Ridge
    X = np.asarray(ds.X, dtype="float64")
    y = ds.y.astype("float64")
    allr = np.arange(ds.n)
    rmse, preds = [], []
    for j in jobs:
        tr, te = np.asarray(j.tr), np.asarray(j.te)
        src = tr if scale == "fold" else allr
        mu, sd = X[src].mean(0), X[src].std(0)
        sd[sd < 1e-8] = 1.0
        Z = (X - mu) / sd
        cols = np.arange(X.shape[1])
        if select:
            k, mode = select
            r_ = tr if mode == "fold" else allr
            zc = Z[r_] - Z[r_].mean(0)
            yc = y[r_] - y[r_].mean()
            den = np.sqrt((zc ** 2).sum(0) * (yc ** 2).sum())
            corr = np.divide(zc.T @ yc, den, out=np.zeros(Z.shape[1]), where=den > 0)
            cols = np.argsort(-np.abs(corr))[:k]
        if target == "drug_z":
            mu_d, sd_d = E.drug_stats(ds, tr if z_stats == "fold" else allr)
            c = ds.drug_codes
            t = (y - mu_d[c]) / sd_d[c]
        else:
            t = y
        m = Ridge(alpha=alpha).fit(Z[np.ix_(tr, cols)], t[tr])
        p = m.predict(Z[np.ix_(te, cols)])
        if target == "drug_z":
            c = ds.drug_codes[te]
            p = p * sd_d[c] + mu_d[c]
        rmse.append(float(np.sqrt(np.mean((p - y[te]) ** 2))))
        preds.append(p)
    return np.array(rmse)


def _lgbm_cv(ds, jobs, es_on_test, progress=None):
    """LightGBM (the app's default settings, drug-wise z target) with early stopping on the inner validation rows,
    or — the leak — on the test fold"""
    from core import gbm as G
    spec = dict(C.MODEL_SPECS[C.GBM_MODEL]["defaults"])
    spec["target"] = "drug_z"
    rmse, rounds = [], []
    for i, j in enumerate(jobs):
        t, inv, _ = E.target_of(ds, j, "drug_z")
        job = dataclasses.replace(j, va=j.te) if es_on_test else j
        r = G.train(ds.X, t, job, spec, 42, n_threads=G.threads())
        p = G.predict_rows(r["booster"], ds.X, j.te, r["best_iter"])
        p = E.inverse(p, inv, ds.drug_codes[j.te])
        rmse.append(float(np.sqrt(np.mean((p - ds.y[j.te].astype("float64")) ** 2))))
        rounds.append(int(r["best_iter"]))
        if progress:
            progress(f"LightGBM {'test-stopped' if es_on_test else 'clean'} fold {i + 1}/{len(jobs)}: "
                     f"RMSE {rmse[-1]:.4f}, best round {rounds[-1]}")
    return np.array(rmse), np.array(rounds)


def leak_demos(data_key="input", seed=42, with_lgbm=True, progress=print):
    """four leaks, each clean vs leaky on the same mixed 10-fold split (seed 42 = the folds of the reported runs)"""
    ds = D.load(data_key)
    cfg = {"schemes": ["mixed"], "grouping": "logo", "folds": 10, "seed": seed, "final": False,
           "spec": {"val_frac": 0.1, "repeats": 1}}
    jobs = _jobs_of(ds, cfg)
    rows = []

    def add(name, model, clean, leaky, note):
        d = leaky - clean
        tt = _tests(d)
        rows.append({"leak": name, "model": model, "RMSE clean": float(clean.mean()), "RMSE leaky": float(leaky.mean()),
                     "Δ (leaky − clean)": float(d.mean()), "relative": float(d.mean() / clean.mean()),
                     "folds leaky looks better": tt["neg"], "folds": tt["n"], "Wilcoxon p": tt["p_wilcoxon"],
                     "note": note})
        progress(f"  {name}: clean {clean.mean():.4f} vs leaky {leaky.mean():.4f}")

    a = 10.0
    progress("L1 · feature scaling fitted on all rows")
    add("L1 · z-scoring fitted on all rows (incl. test)", "Ridge (α = 10)",
        _ridge_cv(ds, jobs, a, scale="fold"), _ridge_cv(ds, jobs, a, scale="all"),
        "our runs fit it per fold")
    progress("L2 · feature selection on all rows")
    add("L2 · top-200 features chosen on all rows", "Ridge (α = 10)",
        _ridge_cv(ds, jobs, a, select=(200, "fold")), _ridge_cv(ds, jobs, a, select=(200, "all")),
        "our runs select no features")
    progress("L3 · drug-wise z statistics from all rows")
    add("L3 · drug means / sds of the z target from all rows", "Ridge (α = 10)",
        _ridge_cv(ds, jobs, a, target="drug_z", z_stats="fold"), _ridge_cv(ds, jobs, a, target="drug_z", z_stats="all"),
        "our runs use training rows only")
    if with_lgbm:
        from core import gbm as G
        if G.available():
            progress("L4 · early stopping on the test fold (LightGBM, about 30 min on 8 cores)")
            clean, r0 = _lgbm_cv(ds, jobs, False, progress)
            leaky, r1 = _lgbm_cv(ds, jobs, True, progress)
            add("L4 · early stopping / best round chosen on the test fold", "LightGBM (app defaults, drug-wise z)",
                clean, leaky, f"best round {int(np.median(r0))} (clean) vs {int(np.median(r1))} (test-stopped), median")
        else:
            rows.append({"leak": "L4 · early stopping on the test fold", "model": "LightGBM", "note": "skipped: "
                         "LightGBM is not installed (Train page → Install LightGBM)"})
    return pd.DataFrame(rows)


def step2(rid_recipe, step1_res=None, with_lgbm=True, progress=print):
    od = out_dir("step2_leakage_audit")
    progress("Leakage audit: checks on the files of " + rid_recipe)
    checks = audit_checks(rid_recipe)
    checks.to_csv(od / "checks.csv", index=False)
    progress("Leakage audit: leak demonstrations (CPU)")
    demos = leak_demos(S.config(rid_recipe)["data"], 42, with_lgbm, progress)
    demos.to_csv(od / "leak_demos.csv", index=False)
    res = {"checks_passed": int((checks["result"] == "PASS").sum()), "checks": int(len(checks)),
           "all_pass": bool((checks["result"] == "PASS").all()),
           "demos": demos.replace({np.nan: None}).to_dict("records"),
           "selection_leak_step1": None if not step1_res else {"confirmed": step1_res.get("confirmed"),
                                                                "mean_dRMSE": step1_res["comparison"]["mean_dRMSE"]}}
    _write_json(od / "step2.json", res)
    (od / "step2_leakage_audit.md").write_text(_step2_md(checks, demos, step1_res), encoding="utf-8")
    return res


def _step2_md(checks, demos, s1):
    s = ["# Step 2 — leakage audit", "",
         "## A. Checks on our pipeline (step 1's recipe run)", "", md_table(checks), "",
         "## B. What each leak would do (same mixed 10-fold split, seed 42; clean vs leaky)", ""]
    if len(demos):
        d = demos.copy()
        for c in ("RMSE clean", "RMSE leaky", "Δ (leaky − clean)"):
            if c in d:
                d[c] = d[c].map(lambda x: _f(x))
        if "relative" in d:
            d["relative"] = d["relative"].map(lambda x: "—" if x is None or not np.isfinite(x) else f"{100 * x:+.1f} %")
        if "Wilcoxon p" in d:
            d["Wilcoxon p"] = d["Wilcoxon p"].map(_p)
        d = d.fillna("—")
        s += [md_table(d), "", "A negative Δ means the leak makes the model look better than it is. The demos use "
              "fast models (Ridge, LightGBM); the size of a leak depends on the model, the direction does not."]
    s += ["", "## C. Model-selection leak (choosing the recipe on the reported folds)", ""]
    if s1:
        c = s1["comparison"]
        s += [f"The recipe was tuned on the seed-42 folds. On fresh seed-43 folds (step 1) it is "
              f"{'CONFIRMED' if s1['confirmed'] else 'NOT confirmed'}: mean ΔRMSE {_f(c['mean_dRMSE'])}, better on "
              f"{c['B_better']} of {c['n']} folds (Wilcoxon p = {_p(c['p_wilcoxon'])})."]
    else:
        s += ["Step 1 has not run yet."]
    s += ["", "## D. Known limitation (not a leak)", "",
          "PathDSP's 10 % inner validation set is a random sample of training pairs. In leave-cell-out / "
          "leave-drug-out it therefore contains the same cell lines and drugs as training, so early stopping cannot "
          "see the unseen-cell / unseen-drug difficulty (v4 repeat 1: validation RMSE at the best epoch 0.380 / 0.367 "
          "/ 0.366 for mixed / LOCO / LODO, test 0.394 / 0.576 / 0.957). It uses no test information, so it is not "
          "leakage, but group-aware inner validation would match the task better."]
    return "\n".join(s)


# ======================================================================================================================
# step 3 — do the explanations point to the drug's targets?
# ======================================================================================================================
def target_pathways(drugs):
    """drug -> set of PID pathways that contain at least one of its GDSC-annotated target genes"""
    import csv
    from core import pathways as PW
    import csa_features as CF
    gmt = PW.load_gmt(PW.gmt_path())
    universe = set().union(*gmt.values())
    info = {}
    with open(C.REFS / "drug_info.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            info[r["drug"]] = r.get("targets", "")
    out, genes = {}, {}
    for d in drugs:
        g = CF.gdsc_targets(info.get(d, ""), universe)
        genes[d] = g
        out[d] = {p for p, s in gmt.items() if s & set(g)}
    return out, genes, sorted(gmt)


def step3(rid, n_random=20, n_perm=1000, progress=print):
    """explanations of the final recipe (step 1's recipe run, its best fold: held-out pairs, every seed)"""
    import torch
    from core import explain as X
    from core import nets
    od = out_dir("step3_explanations")
    cfg = S.config(rid)
    spec = cfg["spec"]
    ds = D.load(cfg["data"])
    b = torch.load(S.run_dir(rid) / "bestfold.pt", map_location="cpu", weights_only=False)
    states = b.get("state_dicts") or [b["state_dict"]]
    mu, sd = np.asarray(b["mu"], "float32"), np.asarray(b["sd"], "float32")
    te, tr = np.asarray(b["te"]), np.asarray(b["tr"])
    feats = list(ds.features)
    fpos = {f: i for i, f in enumerate(feats)}
    pathways = [f[len(D.DG):] for f in feats if f.startswith(D.DG)]
    cell_cols = {p: [fpos[bk + p] for bk in (D.EXP, D.MUT, D.CNV) if bk + p in fpos] for p in pathways}
    dg_col = {p: fpos[D.DG + p] for p in pathways}
    dev = E.device()
    rng = np.random.RandomState(int(cfg["seed"]))
    bg_rows = np.sort(rng.choice(tr, min(100, len(tr)), replace=False))
    Xte = np.asarray(ds.X[te], dtype="float32")
    nets_ = []
    for st_ in states:
        net = nets.build(spec, len(feats), None, feats)
        net.load_state_dict(st_)
        nets_.append(net.to(dev).eval())

    def predict(Xraw):
        z = torch.as_tensor((Xraw - mu) / sd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            return np.mean([n(z).view(-1).double().cpu().numpy() for n in nets_], axis=0)

    # SHAP of every seed on every held-out pair of the best fold
    vals = []
    for i, net in enumerate(nets_):
        progress(f"SHAP seed {i + 1}/{len(nets_)} on {len(te):,} held-out pairs…")
        bg = torch.as_tensor((np.asarray(ds.X[bg_rows], "float32") - mu) / sd, dtype=torch.float32, device=dev)
        xs = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
        v, method = X.shap_values(net, bg, xs, use_deep=isinstance(net, nets.PathDSPNet))
        vals.append(np.abs(np.asarray(v, "float64")))
    A = np.stack(vals)                                              # [seeds, N, F] |SHAP|
    drugs_te = ds.drugs[te]
    y_te = ds.y[te].astype("float64")
    base_pred = predict(Xte)
    tsets, tgenes, _ = target_pathways(sorted(set(drugs_te.tolist())))
    P = len(pathways)

    def cell_score(a):                                              # [n, F] -> per-pathway score [P]
        m = a.mean(0)
        return np.array([m[cell_cols[p]].sum() for p in pathways])

    rows, rank_rows = [], []
    drugs_ok = [d for d in sorted(set(drugs_te.tolist())) if (drugs_te == d).sum() >= MIN_PAIRS]
    for d in drugs_ok:
        md = drugs_te == d
        sc_seeds = np.stack([cell_score(A[s][md]) for s in range(len(A))])        # [seeds, P]
        sc = sc_seeds.mean(0)
        dg = A[:, md][:, :, [dg_col[p] for p in pathways]].mean(axis=(0, 1))
        T = tsets.get(d, set())
        lab = np.array([p in T for p in pathways])
        order = np.argsort(-sc)
        r = {"drug": d, "pairs": int(md.sum()), "targets": ", ".join(tgenes.get(d, [])), "n_target_pathways": int(lab.sum())}
        # stability across seeds: Spearman of the per-pathway scores, top-10 overlap
        if len(A) > 1:
            from scipy.stats import spearmanr
            rhos, jac = [], []
            for a_ in range(len(A)):
                for b_ in range(a_ + 1, len(A)):
                    rhos.append(spearmanr(sc_seeds[a_], sc_seeds[b_]).correlation)
                    ta, tb = set(np.argsort(-sc_seeds[a_])[:10]), set(np.argsort(-sc_seeds[b_])[:10])
                    jac.append(len(ta & tb) / len(ta | tb))
            r["seed_spearman"] = float(np.mean(rhos)); r["seed_top10_jaccard"] = float(np.mean(jac))
        if lab.any():
            r["AUROC_cell"] = auroc(lab, sc)
            r["AUROC_DG_control"] = auroc(lab, dg)
            for k in KS:
                top = order[:k]
                r[f"P@{k}"] = float(lab[top].mean())
                r[f"Hit@{k}"] = float(lab[top].any())
                r[f"P@{k}_random"] = float(lab.mean())
        # deletion test: replace a pathway's EXP/MUT/CNV by the training mean; within-drug correlation after deletion
        Xd = Xte[md]
        yd = y_te[md]
        p0 = base_pred[md]
        r["PCC_base"] = MT._corr(yd, p0)

        def delete(sel):
            Xm = Xd.copy()
            cols = [c for p in sel for c in cell_cols[pathways[p]]]
            Xm[:, cols] = mu[cols]
            p1 = predict(Xm)
            return float(np.abs(p1 - p0).mean()), MT._corr(yd, p1)

        for k in (5, 10):
            dp, pcc = delete(order[:k])
            rnd = [delete(rng.choice(P, k, replace=False)) for _ in range(n_random)]
            r[f"del_top{k}_dpred"] = dp; r[f"del_top{k}_PCC"] = pcc
            r[f"del_rand{k}_dpred"] = float(np.mean([x[0] for x in rnd]))
            r[f"del_rand{k}_PCC"] = float(np.nanmean([x[1] for x in rnd]))
        if lab.any():
            kT = int(lab.sum())
            dp, pcc = delete(np.where(lab)[0])
            rnd = [delete(rng.choice(P, kT, replace=False)) for _ in range(n_random)]
            r["del_target_dpred"] = dp; r["del_target_PCC"] = pcc
            r["del_target_rand_dpred"] = float(np.mean([x[0] for x in rnd]))
            r["del_target_rand_PCC"] = float(np.nanmean([x[1] for x in rnd]))
        rows.append(r)
        for i_, p_ in enumerate(order[:20]):
            rank_rows.append({"drug": d, "rank": i_ + 1, "pathway": pathways[p_], "score": float(sc[p_]),
                              "is_target_pathway": bool(lab[p_])})
    per = pd.DataFrame(rows)
    per.to_csv(od / "per_drug.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(od / "top20_pathways_per_drug.csv", index=False)
    # global ranking of cell pathways
    g = cell_score(A.mean(0))
    glob = pd.DataFrame({"pathway": pathways, "score": g}).sort_values("score", ascending=False).reset_index(drop=True)
    glob.to_csv(od / "global_pathways.csv", index=False)
    # permutation control: shuffle which drug owns which target set
    with_t = per[per["n_target_pathways"] > 0].reset_index(drop=True) if len(per) else per
    perm = {}
    if len(with_t) >= 3:
        from scipy.stats import rankdata
        names = list(with_t["drug"])
        ranks, labs = [], []
        for d in names:                                             # each drug's pathway ranks and its own targets
            md = drugs_te == d
            sc = np.mean([cell_score(A[s][md]) for s in range(len(A))], axis=0)
            ranks.append(rankdata(sc))
            labs.append(np.array([p in tsets[d] for p in pathways]))
        top10 = [np.argsort(-r_, kind="stable")[:10] for r_ in ranks]

        def mean_scores(assign):
            a_, p_ = [], []
            for i_ in range(len(names)):
                lab = labs[assign[i_]]
                n1 = int(lab.sum()); n0 = len(lab) - n1
                a_.append((ranks[i_][lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                p_.append(float(lab[top10[i_]].mean()))
            return float(np.mean(a_)), float(np.mean(p_))

        obs_auc, obs_p10 = mean_scores(np.arange(len(names)))
        prng = np.random.RandomState(12345)
        null = np.array([mean_scores(prng.permutation(len(names))) for _ in range(n_perm)])
        perm = {"observed_AUROC": obs_auc, "null_AUROC_mean": float(null[:, 0].mean()),
                "p_AUROC": float((1 + (null[:, 0] >= obs_auc).sum()) / (1 + n_perm)),
                "observed_P@10": obs_p10, "null_P@10_mean": float(null[:, 1].mean()),
                "p_P@10": float((1 + (null[:, 1] >= obs_p10).sum()) / (1 + n_perm)), "permutations": n_perm}
    res = _step3_summary(per, perm, method, rid, len(te), len(nets_))
    _write_json(od / "step3.json", res)
    (od / "step3_explanations.md").write_text(_step3_md(res, glob), encoding="utf-8")
    return res


def _paired_over_drugs(per, a, b):
    if a not in per or b not in per:
        return None
    x = per[[a, b]].dropna()
    if len(x) < 3:
        return None
    tt = _tests((x[a] - x[b]).to_numpy(float))
    return {"mean_a": float(x[a].mean()), "mean_b": float(x[b].mean()), "a_greater": tt["pos"], "n": tt["n"],
            "p_wilcoxon": tt["p_wilcoxon"]}


def _step3_summary(per, perm, method, rid, n_pairs, n_seeds):
    w = per[per["n_target_pathways"] > 0] if len(per) else per
    res = {"run": rid, "method": method, "pairs": int(n_pairs), "seeds": int(n_seeds), "drugs": int(len(per)),
           "drugs_with_targets": int(len(w)), "permutation": perm}
    if len(w):
        res["recovery"] = {k: float(np.nanmean(w[k])) for k in
                           ["AUROC_cell", "AUROC_DG_control"] + [f"P@{k}" for k in KS] + [f"Hit@{k}" for k in KS]
                           + [f"P@{k}_random" for k in KS]}
    if "seed_spearman" in per:
        res["stability"] = {"seed_spearman": float(np.nanmean(per["seed_spearman"])),
                            "seed_top10_jaccard": float(np.nanmean(per["seed_top10_jaccard"]))}
    res["deletion"] = {
        "top5_vs_random_dpred": _paired_over_drugs(per, "del_top5_dpred", "del_rand5_dpred"),
        "top10_vs_random_dpred": _paired_over_drugs(per, "del_top10_dpred", "del_rand10_dpred"),
        "top10_vs_random_PCC": _paired_over_drugs(per, "del_rand10_PCC", "del_top10_PCC"),
        "targets_vs_random_dpred": _paired_over_drugs(per, "del_target_dpred", "del_target_rand_dpred"),
        "base_PCC": float(np.nanmean(per["PCC_base"])) if "PCC_base" in per else None}
    return res


def _step3_md(r, glob):
    s = ["# Step 3 — do the explanations point to the drug's targets?", "",
         f"Model: step 1's recipe run ({r['run']}), its best fold: {r['pairs']:,} held-out pairs, {r['seeds']} seed(s), "
         f"{r['method']}. A pathway's score for a drug = mean |SHAP| of its EXP, MUT and CNV features over that drug's "
         f"pairs (cell-line side only; DG is built from the targets, so it is reported only as a circular control). "
         f"{r['drugs']} drugs with ≥ {MIN_PAIRS} pairs; {r['drugs_with_targets']} have annotated targets in PID pathways.",
         ""]
    rec = r.get("recovery")
    if rec:
        rows = [{"Metric": "AUROC (target pathways vs others)", "Model": _f(rec["AUROC_cell"], 3), "Random": "0.500"}]
        for k in KS:
            rows.append({"Metric": f"Precision@{k}", "Model": _f(rec[f"P@{k}"], 3), "Random": _f(rec[f"P@{k}_random"], 3)})
            rows.append({"Metric": f"Hit@{k}", "Model": _f(rec[f"Hit@{k}"], 3), "Random": "—"})
        rows.append({"Metric": "AUROC of the DG attribution (circular control)", "Model": _f(rec["AUROC_DG_control"], 3),
                     "Random": "0.500"})
        s += ["## A. Target recovery", "", md_table(pd.DataFrame(rows)), ""]
        pm = r.get("permutation") or {}
        if pm:
            s += [f"**Permutation control** (target sets shuffled between drugs, {pm['permutations']}×): mean AUROC "
                  f"{_f(pm['observed_AUROC'], 3)} vs {_f(pm['null_AUROC_mean'], 3)} under the null (p = {_p(pm['p_AUROC'])}); "
                  f"P@10 {_f(pm['observed_P@10'], 3)} vs {_f(pm['null_P@10_mean'], 3)} (p = {_p(pm['p_P@10'])}).", ""]
    st = r.get("stability")
    if st:
        s += ["## B. Stability across seeds", "",
              f"Per drug, the pathway scores of two seeds correlate at Spearman {_f(st['seed_spearman'], 3)}; their "
              f"top-10 pathways overlap with Jaccard {_f(st['seed_top10_jaccard'], 3)}.", ""]
    dl = r.get("deletion", {})
    s += ["## C. Deletion test (faithfulness)", "",
          "A pathway is deleted by replacing its EXP / MUT / CNV values with the training mean; the change is measured "
          "on the drug's held-out pairs (mean |Δ prediction| in z units, and the within-drug PCC with the measured "
          "response).", ""]
    lines = []
    for key, label in (("top5_vs_random_dpred", "top-5 vs 5 random pathways: mean |Δ prediction|"),
                       ("top10_vs_random_dpred", "top-10 vs 10 random: mean |Δ prediction|"),
                       ("top10_vs_random_PCC", "10 random vs top-10: within-drug PCC after deletion"),
                       ("targets_vs_random_dpred", "target pathways vs as many random: mean |Δ prediction|")):
        v = dl.get(key)
        if v:
            lines.append({"Comparison": label, "First": _f(v["mean_a"]), "Second": _f(v["mean_b"]),
                          "Drugs where first > second": f"{v['a_greater']} / {v['n']}", "Wilcoxon p": _p(v["p_wilcoxon"])})
    if lines:
        s += [md_table(pd.DataFrame(lines)), "", f"Within-drug PCC before deletion: {_f(dl.get('base_PCC'), 3)}.", ""]
    s += ["## D. Top cell-line pathways (all drugs)", "",
          md_table(glob.head(15).assign(score=lambda x: x["score"].map(lambda v: _f(v, 5)))), ""]
    return "\n".join(s)


# ======================================================================================================================
# step 4 — unseen cell lines / unseen drugs
# ======================================================================================================================
def _per_group(p, col):
    v = [float(np.sqrt(np.mean((g["pred"] - g["actual"]) ** 2))) for _, g in p.groupby(col)]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), np.array(v)


def _baseline_preds(p, scheme):
    """simple baselines from the training rows of every fold: LOCO -> the drug's mean over training cell lines
    (= preds.base); LODO -> the cell line's mean over training drugs. Also the global training mean."""
    p = p.copy()
    p["b_global"] = np.nan
    p["b_group"] = np.nan
    for (r, k), g in p.groupby(["repeat", "fold"]):
        trn = p[(p["repeat"] == r) & (p["fold"] != k)]
        idx = g.index
        p.loc[idx, "b_global"] = trn["actual"].mean()
        if scheme == "LOCO":
            m = trn.groupby("drug")["actual"].mean()
            p.loc[idx, "b_group"] = g["drug"].map(m).to_numpy()
        else:
            m = trn.groupby("cell")["actual"].mean()
            p.loc[idx, "b_group"] = g["cell"].map(m).to_numpy()
    return p


def step4(rid_plain, rid_loco, rid_lodo):
    od = out_dir("step4_unseen")
    out = {}
    md = ["# Step 4 — unseen cell lines and unseen drugs (grouped 10-fold, seed 42)", "",
          "Plain Paper exact vs the recipe, on the same folds. Leave-drug-out uses the recipe without the drug-wise z "
          "target (C·D×3·E·F·G): a held-out drug has no training responses to z-score with. RMSE paper style = mean ± "
          "sd over the held-out cell lines / drugs; pooled metrics over all held-out pairs. Units: −log10 IC50.", ""]
    for scheme, rid_r, col, base_name in (("LOCO", rid_loco, "cell", "drug's mean over training cell lines"),
                                          ("LODO", rid_lodo, "drug", "cell line's mean over training drugs")):
        cmp_, table = compare_pair(rid_plain, rid_r, scheme, "Paper exact (plain)", "Paper exact + recipe")
        table.to_csv(od / f"{scheme}_fold_by_fold.csv", index=False)
        rows = []
        pp = S.preds(rid_plain); pp = pp[pp.scheme == scheme]
        pr = S.preds(rid_r); pr = pr[pr.scheme == scheme]
        fp = S.folds(rid_plain); fp = fp[fp.scheme == scheme]
        fr = S.folds(rid_r); fr = fr[fr.scheme == scheme]
        bl = _baseline_preds(pp, scheme)
        per_g = {}
        for name, p, f in (("Paper exact (plain)", pp, fp), ("Paper exact + recipe", pr, fr)):
            m = run_metrics(p, f, scheme, "kfold")
            gm, gs, gv = _per_group(p, col)
            per_g[name] = pd.Series(gv, index=sorted(p[col].unique()))
            rows.append({"Model": name, "RMSE (paper style)": f"{_f(gm)} ± {_f(gs)}", "RMSE pooled": _f(m["RMSE"]),
                         "MAE": _f(m["MAE"]), "R²": _f(m["R2"], 3), "PCC": _f(m["PCC"], 3),
                         "Per-drug PCC": _f(m["per_drug_PCC"], 3), "Within-drug AUROC": _f(m["within_drug_AUROC"], 3),
                         "_rmse_group": gm, "_pcc": m["PCC"], "_pd": m["per_drug_PCC"]})
        for bname, bcol in ((f"Baseline: {base_name}", "b_group"), ("Baseline: global training mean", "b_global")):
            q = bl.assign(pred=bl[bcol])
            q = q[q["pred"].notna()]
            gm, gs, gv = _per_group(q, col)
            per_g[bname] = pd.Series(gv, index=sorted(q[col].unique()))
            mm = MT.regression(q["actual"], q["pred"])
            # LOCO's drug mean and the global mean are constant within a drug, so their within-drug ranking is
            # undefined (and the global mean's pooled PCC is fold-to-fold noise); LODO's cell-line mean does rank the
            # cell lines within a new drug, so its per-drug PCC and within-drug AUROC are reported
            lodo_cell = scheme == "LODO" and bcol == "b_group"
            rows.append({"Model": bname, "RMSE (paper style)": f"{_f(gm)} ± {_f(gs)}", "RMSE pooled": _f(mm["RMSE"]),
                         "MAE": _f(mm["MAE"]), "R²": _f(mm["R2"], 3),
                         "PCC": "n/a" if bcol == "b_global" else _f(mm["PCC"], 3),
                         "Per-drug PCC": _f(MT.per_drug_pcc(q), 3) if lodo_cell else "n/a",
                         "Within-drug AUROC": _f(within_drug_auroc(q), 3) if lodo_cell else "n/a",
                         "_rmse_group": gm, "_pcc": mm["PCC"], "_pd": float("nan")})
        tab = pd.DataFrame(rows)
        tab.drop(columns=[c for c in tab.columns if c.startswith("_")]).to_csv(od / f"{scheme}_summary.csv", index=False)
        # model vs the group baseline, per held-out cell line / drug
        bname = f"Baseline: {base_name}"
        vs = {}
        for name in ("Paper exact (plain)", "Paper exact + recipe"):
            j = pd.concat([per_g[name], per_g[bname]], axis=1, keys=["m", "b"]).dropna()
            tt = _tests((j["m"] - j["b"]).to_numpy(float))
            vs[name] = {"groups": tt["n"], "model_better": tt["neg"], "baseline_better": tt["pos"],
                        "mean_d": tt["mean"], "p_wilcoxon": tt["p_wilcoxon"]}
        out[scheme] = {"comparison": cmp_, "summary": tab.drop(columns=[c for c in tab.columns if c.startswith("_")])
                       .to_dict("records"), "vs_baseline": vs}
        unit = "cell lines" if scheme == "LOCO" else "drugs"
        md += [f"## {'Unseen cell lines (LOCO)' if scheme == 'LOCO' else 'Unseen drugs (LODO)'}", "",
               md_table(tab.drop(columns=[c for c in tab.columns if c.startswith("_")])), "",
               f"**Recipe vs plain (per fold):** mean ΔRMSE {_f(cmp_['mean_dRMSE'])}, recipe better on "
               f"{cmp_['B_better']} of {cmp_['n']} folds; Wilcoxon p = {_p(cmp_['p_wilcoxon'])}, t p = {_p(cmp_['p_t'])}.",
               ""]
        for name, v in vs.items():
            md.append(f"- **{name} vs the {base_name}:** better on {v['model_better']} of {v['groups']} held-out "
                      f"{unit} (mean per-{unit[:-1]} ΔRMSE {_f(v['mean_d'])}, Wilcoxon p = {_p(v['p_wilcoxon'])}).")
        md.append("")
    _write_json(od / "step4.json", out)
    (od / "step4_unseen.md").write_text("\n".join(md), encoding="utf-8")
    return out


# ======================================================================================================================
# finalise
# ======================================================================================================================
def finalise(state):
    """the final recipe of the train-on-everything models, decided from steps 1–4"""
    od = out_dir("finalise")
    s1 = state.get("results", {}).get("step1") or {}
    confirmed = bool(s1.get("confirmed"))
    ic50 = ({"label": "Paper exact + C·D×3·E·F·G·Z", "recipe": "recipe"} if confirmed else
            {"label": "Paper exact (plain)", "recipe": "plain"})
    dec = {
        "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "Step 1: the recipe must beat plain Paper exact on fresh folds on ≥ 8 of 10 folds with Wilcoxon "
                "p < 0.05. If it does, the final IC50 model uses it; otherwise plain Paper exact.",
        "step1_confirmed": confirmed,
        "final_ic50": {**ic50, "data": "processed",
                       "why_data": "processed_data · GDSC: all 153 drugs (40,365 pairs, a superset of input.txt's "
                                   "37,732), and the paper's CCLE external test runs on it"},
        "final_aac": [{"label": "Run A recipe — multi-study, raw AAC", "use": "absolute AAC for a study with no "
                       "measured pairs"},
                      {"label": "Run B recipe — S + R + M", "use": "ranking cell lines within a drug; recalibrate "
                       "with ~100 measured pairs"}],
    }
    _write_json(od / "decision.json", dec)
    (od / "decision.md").write_text(
        "# Final decision\n\n" + dec["rule"] + "\n\n"
        f"- Step 1 confirmed the recipe: **{'yes' if confirmed else 'no'}**\n"
        f"- Final IC50 model: **{ic50['label']}**, trained on all of processed_data · GDSC\n"
        "- Final AAC models: Run A recipe and Run B recipe, each trained on all 587,709 CSA pairs\n", encoding="utf-8")
    return dec


def final_summary(state):
    """FINAL_SUMMARY.md: every step's report in one file + the all-data models"""
    parts = ["# PathDSP — final validation (v6.0)", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')}. Runs: " +
             ", ".join(f"{k} = {v}" for k, v in state.get("runs", {}).items()), ""]
    for step, fname in (("step1_fresh_folds", "step1_fresh_folds.md"), ("step2_leakage_audit", "step2_leakage_audit.md"),
                        ("step3_explanations", "step3_explanations.md"), ("step4_unseen", "step4_unseen.md"),
                        ("finalise", "decision.md")):
        p = OUT / step / fname
        if p.exists():
            parts += [p.read_text(encoding="utf-8"), ""]
    fin = []
    for key, label in (("F1-ic50", "IC50 · processed_data"), ("F2-aac-A", "AAC · CSA all 5 · Run A recipe"),
                       ("F3-aac-B", "AAC · CSA all 5 · Run B recipe")):
        rid = state.get("runs", {}).get(key)
        if not rid:
            continue
        meta = S.read_json(S.model_paths(rid)[1], {}) or {}
        ext = S.read_json(S.run_dir(rid) / "external_ccle.json", None)
        st = S.status(rid)
        line = (f"- **{label}:** run {rid}, models/{rid}.pt ({meta.get('kind', '?')}, {meta.get('n_train', '?'):,} "
                f"training pairs, upgrades {meta.get('upgrades') or 'none'}, target {meta.get('target')}); "
                f"trained in {st.get('elapsed', 0) / 3600:.1f} h") if isinstance(meta.get("n_train"), int) else \
               f"- **{label}:** run {rid}"
        if isinstance(ext, dict) and "RMSE" in ext:
            line += f"; CCLE external test RMSE {_f(ext['RMSE'], 3)}, PCC {_f(ext['PCC'], 3)} ({ext['pairs']:,} pairs)"
        fin.append(line)
    if fin:
        parts += ["# Train-on-everything models (used by Predict)", "", *fin, ""]
    (OUT / "FINAL_SUMMARY.md").write_text("\n".join(parts), encoding="utf-8")
    return OUT / "FINAL_SUMMARY.md"


def pack():
    """final/final_results.zip: every report, table and JSON (no model files) — send this one file"""
    import zipfile
    z = OUT / "final_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.suffix in (".md", ".csv", ".json") and p != z:
                zf.write(p, p.relative_to(OUT))
        for rid in (json.loads((OUT / "state.json").read_text(encoding="utf-8")).get("runs", {}).values()
                    if (OUT / "state.json").exists() else []):
            for name in ("config.json", "summary.json", "folds.csv", "members.csv", "external_ccle.json",
                         "shap_blocks.csv", "shap_pathways.csv"):
                p = S.run_dir(rid) / name
                if p.exists():
                    zf.write(p, f"runs/{rid}/{name}")
            pr = S.run_dir(rid) / "preds.csv"
            if pr.exists():
                zf.write(pr, f"runs/{rid}/preds.csv")
    return z
