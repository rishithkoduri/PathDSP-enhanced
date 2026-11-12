"""RESULTS — metrics per evaluation, training curves, timing, SHAP pathway insight, CCLE external test, comparison."""
import numpy as np
import pandas as pd
import streamlit as st

from core import compare as CP
from core import config as C
from core import data as D
from core import store as S
from core import uncertainty as UQ

from . import charts as CH
from . import common as U
from . import theme as T

LIVE = ("starting", "loading", "running", "explaining")


def _metric_tiles(m, scheme, data=""):
    unit = D.unit_of(data)
    ref = C.PAPER_REF.get(scheme, {}) if unit["ic50"] else {}           # the paper's numbers are −log10 IC50 on GDSC

    def sub(k):
        v, sd = U.pm(m, k)
        r = ref.get(k)
        rs = f" · paper {r[0]:.2f}" + (f" ± {r[1]:.2f}" if r and r[1] else "") if r else ""
        return v, (sd + rs).strip(" ·")

    items = []
    for k, lab in (("RMSE", "RMSE"), ("MAE", "MAE"), ("R2", "R²"), ("PCC", "PCC"), ("SCC", "Spearman")):
        v, s = sub(k)
        items.append({"label": lab, "value": v, "sub": s, "hl": k == "RMSE"})
    items.append({"label": "Per-drug PCC", "value": U.f3(m.get("per_drug_PCC")),
                  "sub": "cell ranking within each drug", "red": True})
    T.kpis(items)
    base = m.get("base_RMSE")
    if base is not None and np.isfinite(base) and np.isfinite(m.get("RMSE", np.nan)):
        gain = 100 * (1 - m["RMSE"] / base)
        st.caption(f"Drug-mean baseline (each pair predicted by its drug's training mean): RMSE {base:.3f} — "
                   f"the model is {gain:.1f}% lower. Per-drug PCC of that baseline is 0 by construction.")
    k = m.get("n_folds", 0) // max(1, m.get("repeats", 1))
    sd_note = {"mixed": (f"± = sd between the repeated {k}-fold runs" if m.get("repeats", 1) > 1 else
                         f"± = sd over the {k} folds"),
               "LOCO": "RMSE / MAE: mean ± sd over held-out cell lines (paper style); R², PCC, Spearman: all held-out pairs pooled",
               "LODO": "RMSE / MAE: mean ± sd over held-out drugs (paper style); R², PCC, Spearman: all held-out pairs pooled",
               "csa": "every metric: mean ± sd over the benchmark's 10 official splits (each split's own test set)",
               "loso": "every metric: mean ± sd over the 5 held-out studies (each tested on ALL of its pairs)"}
    st.caption(f"{unit['name']} · {m.get('n_folds', 0)} fold-models · {m.get('n_pred', 0):,} held-out predictions · "
               + sd_note.get(scheme, ""))


def _ensemble_note(rid, scheme, f):
    """(D) seed ensemble: the averaged prediction vs every seed on its own, on the same folds"""
    mem = U.run_csv(rid, "members.csv")
    if mem.empty or f.empty:
        return
    mem = mem[mem["scheme"] == scheme]
    if mem.empty:
        return
    keys = ["repeat", "fold"]
    single = mem.groupby(keys, as_index=False)[["RMSE", "PCC"]].mean()           # mean over seeds, per fold
    j = f[keys + ["RMSE", "PCC"]].merge(single, on=keys, suffixes=("", "_single"))
    if j.empty:
        return
    n = int(mem["member"].max())
    ens, one = float(j["RMSE"].mean()), float(j["RMSE_single"].mean())
    wins = int((j["RMSE"] < j["RMSE_single"]).sum())
    best_seed = mem.groupby("member")["RMSE"].mean()
    T.kpis([
        {"label": f"{n}-seed ensemble RMSE", "value": f"{ens:.4f}", "sub": f"averaged predictions · {len(j)} folds",
         "hl": True},
        {"label": "Single seed RMSE", "value": f"{one:.4f}",
         "sub": "mean of the seeds on their own · seeds " + " / ".join(f"{v:.4f}" for v in best_seed.values)},
        {"label": "Ensemble gain", "value": f"{one - ens:+.4f}",
         "sub": f"RMSE lower on {wins} of {len(j)} folds", "red": True},
        {"label": "PCC", "value": f'{float(j["PCC"].mean()):.4f}',
         "sub": f'single seed {float(j["PCC_single"].mean()):.4f} · mean over folds'},
    ])
    st.caption("D · seed ensemble — every fold was trained by several seeds; the metrics above this note use their "
               "averaged predictions, this row compares them with the seeds on their own (same folds).")


def _scheme_tab(rid, scheme, m, folds, preds, curves, data=""):
    _metric_tiles(m, scheme, data)
    _ensemble_note(rid, scheme, folds[folds["scheme"] == scheme] if not folds.empty else folds)
    if scheme == C.LOSO_SCHEME:
        _loso_section(rid, folds, preds)
    st.write("")
    p = preds[(preds["scheme"] == scheme) & (preds["repeat"] == preds[preds["scheme"] == scheme]["repeat"].min())]
    f = folds[folds["scheme"] == scheme]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="nt-label">Predicted vs measured · held-out pairs (dot size = pairs)</div>',
                    unsafe_allow_html=True)
        if len(p):
            CH.show(CH.scatter_density(p["actual"], p["pred"], unit=D.unit_of(data)["name"]), 330)
    with c2:
        st.markdown('<div class="nt-label">Training curves · validation loss per model, mean train / val</div>',
                    unsafe_allow_html=True)
        cv = curves[curves["scheme"] == scheme] if not curves.empty else curves
        if len(cv):
            CH.show(CH.loss_curves(cv), 330)
    c3, c4 = st.columns([1.4, 1])
    with c3:
        st.markdown('<div class="nt-label">Test RMSE of every fold-model (red = mean)</div>', unsafe_allow_html=True)
        if len(f):
            CH.show(CH.fold_strip(f), 220)
    with c4:
        st.markdown('<div class="nt-label">Training time</div>', unsafe_allow_html=True)
        gbm = S.config(rid).get("model") == C.GBM_MODEL
        if len(f):
            T.kpis([
                {"label": "Per model", "value": U.hms(f["seconds"].mean()),
                 "sub": ("not recorded for imported runs" if f["seconds"].isna().all() else
                         ("wall clock, one at a time on the CPU" if gbm else
                          f'wall clock, {int(f["parallel"].max())} trained together'))},
                {"label": "Boosting rounds" if gbm else "Epochs", "value": f'{f["epochs_run"].mean():.0f}',
                 "sub": f'best {"round" if gbm else "epoch"} {f["best_epoch"].mean():.0f} (mean)'},
            ])
    _uncertainty(rid, scheme, data)
    with st.expander("Per-fold table"):
        st.dataframe(f.drop(columns=["scheme"], errors="ignore"), hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("RMSE", "MAE", "R2", "PCC", "SCC")})
    with st.expander("Per-drug metrics"):
        pall = preds[preds["scheme"] == scheme]
        iv, _ = U.conformal(rid, scheme, 0.90, "drug")
        t = UQ.per_drug_table(pall, iv)
        if "coverage" in t.columns:
            t["coverage"] = 100 * t["coverage"]
        st.caption("First repeat · sorted by RMSE (best first) · coverage and ± width of the 90 % per-drug conformal intervals.")
        st.dataframe(t, hide_index=True, height=min(520, 38 + 35 * len(t)),
                     column_config={**{k: st.column_config.NumberColumn(format="%.4f") for k in ("RMSE", "MAE", "PCC")},
                                    "mean measured": st.column_config.NumberColumn(format="%.3f"),
                                    "sd measured": st.column_config.NumberColumn(format="%.3f"),
                                    "coverage": st.column_config.NumberColumn("coverage 90 %", format="%.0f %%"),
                                    "half_width": st.column_config.NumberColumn("± width", format="%.3f")})
        st.download_button("Per-drug CSV", t.to_csv(index=False).encode(), f"{rid}_{scheme}_per_drug.csv", mime="text/csv",
                           key=f"dl_d_{scheme}")
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Predictions CSV", p.to_csv(index=False).encode(), f"{rid}_{scheme}_predictions.csv",
                       mime="text/csv", key=f"dl_p_{scheme}")
    d2.download_button("Folds CSV", f.to_csv(index=False).encode(), f"{rid}_{scheme}_folds.csv", mime="text/csv",
                       key=f"dl_f_{scheme}")


def _loso_section(rid, folds, preds):
    """leave-study-out: per held-out study, the comparison with single-study models, few-shot recalibration"""
    from core import csa as CSA
    t = CSA.loso_table(preds, folds)
    if t.empty:
        return
    st.markdown('<div class="nt-label" style="margin-top:14px">Per held-out study · trained on the other four, tested '
                'on all of its pairs (zero-shot)</div>', unsafe_allow_html=True)
    st.dataframe(t, hide_index=True, column_config={
        **{k: st.column_config.NumberColumn(format="%.4f") for k in ("RMSE", "MAE", "R2", "PCC", "SCC")},
        "R2": st.column_config.NumberColumn("R²", format="%.4f"),
        "per_drug_PCC": st.column_config.NumberColumn("per-drug PCC", format="%.4f")})

    # ---------------------------------------------------------------- vs single-study models
    st.markdown('<div class="nt-label" style="margin-top:14px">Multi-study vs single-study models · the same unseen '
                'study</div>', unsafe_allow_html=True)
    cfg = S.config(rid)
    runs = [r for r in S.list_runs() if D.is_csa(r["config"].get("data", "")) and r["state"] != "failed"
            and C.CSA_SCHEME in (r["config"].get("schemes") or []) and r["config"].get("cross")]
    lab = {r["id"]: U.run_label(r) for r in runs}
    c = st.columns(len(CSA.STUDIES) + 1)
    metric = c[0].selectbox("Metric", list(CSA.METRICS), index=list(CSA.METRICS).index("PCC"), key=f"lo_metric_{rid}",
                            format_func=lambda m: {"R2": "R²", "per_drug_PCC": "per-drug PCC", "SCC": "Spearman"}.get(m, m))
    pick = {}
    for i, s in enumerate(CSA.STUDIES):
        cand = [r["id"] for r in runs if CSA.study_of(r["config"].get("data", "")) == s]
        same = [x for x in cand if S.config(x).get("model") == cfg.get("model")]
        default = same[0] if same else None
        opts = [None] + cand
        pick[s] = c[i + 1].selectbox(f"{s} model", opts, index=opts.index(default) if default in opts else 0,
                                     key=f"lo_pick_{rid}_{s}",
                                     format_func=lambda x: "—" if x is None else lab.get(x, x)[:42])
    sv = CSA.single_source_vs(pick, metric)
    mine = t.set_index("study")[metric if metric != "per_drug_PCC" else "per_drug_PCC"]
    sv["multi-study (this run)"] = sv["study"].map(mine)
    hb = CSA.HIGHER_BETTER[metric]
    sgn = 1.0 if hb else -1.0
    sv["gain vs best single"] = sgn * (sv["multi-study (this run)"] - sv["single_best"])
    sv["gain vs mean single"] = sgn * (sv["multi-study (this run)"] - sv["single_mean"])
    show = sv[["study", "multi-study (this run)", "single_best", "best_source", "single_mean", "sources",
               "gain vs best single", "gain vs mean single"]].rename(
        columns={"single_best": "best single-study", "best_source": "trained on", "single_mean": "mean single-study"})
    st.dataframe(show, hide_index=True, column_config={
        k: st.column_config.NumberColumn(format="%.4f") for k in ("multi-study (this run)", "best single-study",
                                                                  "mean single-study", "gain vs best single",
                                                                  "gain vs mean single")})
    st.caption("For every study: this run's model (trained on the other four studies together) against the single-study "
               "models of the chosen CSA-split runs tested on that same study (their 10-split mean, G[s, t]). Gain > 0 "
               "= the multi-study model is better (for RMSE / MAE: lower). Choose one CSA-split run per study "
               "(default: the latest with the same model).")

    # ---------------------------------------------------------------- few-shot
    fs = U.run_csv(rid, "fewshot.csv")
    st.markdown('<div class="nt-label" style="margin-top:14px">Few-shot recalibration · k measured pairs of the new '
                'study fit y ≈ a·ŷ + b</div>', unsafe_allow_html=True)
    if fs.empty:
        st.caption("Computed when the run finishes.")
        return
    fmt = {"R2": "R²", "RMSE": "RMSE", "MAE": "MAE", "coverage": "90 % interval coverage", "half_width": "interval ±"}
    m2 = st.segmented_control("Few-shot metric", list(fmt), default="R2", key=f"fs_m_{rid}", required=True,
                              format_func=lambda m: fmt[m], label_visibility="collapsed") or "R2"
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if fs[m2].notna().any():
            CH.show(CH.fewshot_lines(fs.dropna(subset=[m2]), m2), 300)
    with c2:
        wide = fs.pivot_table(index="study", columns="k", values=m2).reset_index()
        wide.columns = ["study"] + [("zero-shot" if k == 0 else f"k={k}") for k in wide.columns[1:]]
        st.dataframe(wide, hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in wide.columns[1:]})
    st.caption("k = 0 is the model as it is (its interval width comes from the errors on the OTHER held-out studies). "
               "For k > 0, k random pairs of the held-out study (20 draws) estimate its AAC scale, y ≈ a·ŷ + b with "
               "the slope shrunk towards 1 (few pairs: mostly an offset); the other pairs are scored. Intervals: half "
               "of the k pairs fit the scale, the other half set the 90 % width (split conformal on the new study; a 90 % "
               "width needs at least 9 such pairs, so intervals start at k = 30). "
               "The rise from zero-shot shows how much of the cross-study error is only the study's assay scale; PCC "
               "and per-drug PCC (the ranking) do not change under this recalibration — that part must come from the "
               "model.")


def _uncertainty(rid, scheme, data=""):
    """conformal prediction intervals for the held-out predictions (core/uncertainty.py)"""
    st.markdown('<div class="nt-label" style="margin-top:16px">Uncertainty · conformal prediction intervals</div>',
                unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.4, 1.6, 3])
    level = c1.segmented_control("Level", list(UQ.LEVELS), default=0.90, key=f"uq_lv_{scheme}", required=True,
                                 format_func=lambda v: f"{v:.0%}", label_visibility="collapsed") or 0.90
    mode = c2.segmented_control("Calibration", ["drug", "global"], default="drug", key=f"uq_md_{scheme}", required=True,
                                format_func=lambda m: "calibrated per drug" if m == "drug" else "one global width",
                                label_visibility="collapsed") or "drug"
    iv, s = U.conformal(rid, scheme, level, mode)
    if not s:
        st.caption("Intervals appear when held-out predictions exist.")
        return
    unit = D.unit_of(data)
    T.kpis([
        {"label": "Coverage", "value": f"{100 * s['coverage']:.1f} %", "hl": True,
         "sub": f"target {level:.0%} · folds {100 * s['fold_cov_min']:.1f}–{100 * s['fold_cov_max']:.1f} %"},
        {"label": "Interval", "value": f"± {s['median_half_width']:.3f}",
         "sub": f"median half-width, {unit['name']} (mean ± {s['half_width']:.3f})"
                + (f" · IC50 within ×{10 ** s['median_half_width']:.1f}" if unit["ic50"] else "")},
        {"label": "Drugs on target", "value": f"{s['drugs_ok']}/{s['drugs']}", "red": True,
         "sub": f"coverage ≥ {100 * (level - 0.05):.0f} % for the drug"},
        {"label": "Calibrated per drug", "value": f"{100 * s['share_drug_cal']:.0f} %",
         "sub": f"of predictions; the rest use the global width (drug with < {UQ.MIN_N} errors in the other folds)"},
    ])
    st.caption("Each held-out prediction gets pred ± q. q is the conformal quantile of the absolute errors of the same "
               "run's held-out predictions in the OTHER folds (cross-validated split conformal; per drug = Mondrian by drug), "
               "so no pair calibrates its own interval and nothing is retrained."
               + (" Leave-study-out: the other folds are the OTHER held-out studies, so this is the coverage of "
                  "intervals calibrated on unseen studies and applied to a new unseen study." if scheme == C.LOSO_SCHEME
                  else ""))


def _paired_section(current):
    """paired fold-by-fold test of two runs (core/compare.py)"""
    st.markdown('<div class="nt-label" style="margin-top:18px">Paired test · two runs on the same test folds</div>',
                unsafe_allow_html=True)
    runs = [r for r in S.list_runs() if r["state"] != "failed" and not U.run_csv(r["id"], "folds.csv").empty]
    ids = [r["id"] for r in runs]
    if len(ids) < 2:
        st.caption("Needs two runs with finished folds.")
        return
    lab = {r["id"]: U.run_label(r) for r in runs}
    c1, c2, c3 = st.columns([2.3, 2.3, 1])
    b_default = ids.index(current) if current in ids else 0
    b = c2.selectbox("Run B", ids, index=b_default, key="cmp_b", format_func=lambda i: lab[i])
    a_opts = [i for i in ids if i != b]
    a = c1.selectbox("Run A (reference)", a_opts, index=0, key="cmp_a", format_func=lambda i: lab[i])
    sa = set(U.run_csv(a, "folds.csv")["scheme"]); sb = set(U.run_csv(b, "folds.csv")["scheme"])
    common = [s for s in C.SCHEMES if s in sa & sb]
    if not common:
        T.banner("These two runs share no evaluation.")
        return
    scheme = c3.selectbox("Evaluation", common, key="cmp_s", format_func=lambda s: C.SCHEMES[s]["name"])
    res = U.paired(a, b, scheme)
    if not res.get("same_folds"):
        T.banner(T.esc(CP.verdict(res)) + f" ({res.get('n_same', 0)} of {res.get('n_common', 0)} shared folds hold the "
                 "same held-out pairs.)")
        return
    f4 = lambda v: "—" if v is None or not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")  # noqa: E731
    T.kpis([
        {"label": "ΔRMSE (B − A)", "value": f"{res['mean']:+.4f}", "hl": True,
         "sub": f"mean over {res['n']} folds · sd {res['sd']:.4f} · negative = B better"},
        {"label": "Folds won", "value": f"B {res['wins_b']} · A {res['wins_a']}",
         "sub": f"{res['ties']} ties" if res["ties"] else "lower test RMSE per fold"},
        {"label": "Sign test", "value": f4(res.get("p_sign")), "sub": "exact binomial, two-sided p"},
        {"label": "Wilcoxon", "value": f4(res.get("p_wilcoxon")), "sub": "signed-rank, two-sided p"},
        {"label": "Paired t-test", "value": f4(res.get("p_t")), "sub": "two-sided p"},
    ])
    st.markdown(f'<div class="nt-note" style="margin:6px 0 8px 0">{T.esc(CP.verdict(res))}</div>', unsafe_allow_html=True)
    st.dataframe(res["table"], hide_index=True, height=min(420, 38 + 35 * len(res["table"])),
                 column_config={k: st.column_config.NumberColumn(format="%.4f")
                                for k in ("RMSE_A", "RMSE_B", "PCC_A", "PCC_B", "ΔRMSE (B − A)")})


def _blend_section(current):
    """(v5.9) blend of two runs' held-out predictions — e.g. a network + LightGBM (core/blend.py)"""
    st.markdown('<div class="nt-label" style="margin-top:22px">Blend · w·A + (1 − w)·B of two runs on the same test '
                'folds (e.g. a network + LightGBM)</div>', unsafe_allow_html=True)
    runs = [r for r in S.list_runs() if r["state"] != "failed" and not U.run_csv(r["id"], "preds.csv").empty]
    ids = [r["id"] for r in runs]
    if len(ids) < 2:
        st.caption("Needs two runs with held-out predictions.")
        return
    lab = {r["id"]: U.run_label(r) for r in runs}
    cfg_of = {r["id"]: r["config"] for r in runs}
    c1, c2, c3 = st.columns([2.3, 2.3, 1])
    a = c1.selectbox("Run A", ids, index=ids.index(current) if current in ids else 0, key="bl_a",
                     format_func=lambda i: lab[i])
    data_a = cfg_of[a].get("data")
    b_opts = [i for i in ids if i != a]
    pa_ = U.run_csv(a, "preds.csv")
    n_a = pa_.groupby("scheme").size().to_dict() if not pa_.empty else {}

    def same_folds(i):                                      # same data, seed, k and repeats, same held-out counts
        ci, ca = cfg_of[i], cfg_of[a]
        if ci.get("data") != data_a or ci.get("seed") != ca.get("seed") or ci.get("folds") != ca.get("folds") \
                or (ci.get("spec") or {}).get("repeats") != (ca.get("spec") or {}).get("repeats"):
            return False
        pi = U.run_csv(i, "preds.csv")
        n_i = pi.groupby("scheme").size().to_dict() if not pi.empty else {}
        return any(n_i.get(s_) == n for s_, n in n_a.items())

    def rmse(i):
        sm = U.scheme_summaries(i)
        v = [m.get("RMSE") for m in sm.values() if m.get("RMSE") is not None]
        return min(v) if v else np.inf
    gbm_a = cfg_of[a].get("model") == C.GBM_MODEL
    match = sorted([i for i in b_opts if same_folds(i)],
                   key=lambda i: ((cfg_of[i].get("model") == C.GBM_MODEL) == gbm_a, rmse(i)))
    b = c2.selectbox("Run B", b_opts, index=b_opts.index(match[0]) if match else 0, key=f"bl_b_{a}",
                     format_func=lambda i: lab[i],
                     help="default: the best run on the same held-out folds — of the other kind (network ↔ LightGBM) "
                          "when there is one")
    sa = set(U.run_csv(a, "preds.csv")["scheme"]); sb = set(U.run_csv(b, "preds.csv")["scheme"])
    common = [s for s in C.SCHEMES if s in sa & sb]
    if not common:
        T.banner("These two runs share no evaluation.")
        return
    scheme = c3.selectbox("Evaluation", common, key="bl_s", format_func=lambda s: C.SCHEMES[s]["name"])
    with st.spinner("Blending…"):
        res = U.blend(a, b, scheme)
    if not res.get("ok"):
        T.banner(T.esc(res.get("reason", "")))
        return
    from core import blend as BL
    sm = res["summary"]

    def val(k, q):
        v = sm[k].get(q)
        return "—" if v is None or not np.isfinite(v) else f"{v:.4f}"
    T.kpis([
        {"label": "RMSE · blend", "value": val("blend", "RMSE"), "hl": True,
         "sub": f'A {val("A", "RMSE")} · B {val("B", "RMSE")}'},
        {"label": "R² · blend", "value": val("blend", "R2"), "sub": f'A {val("A", "R2")} · B {val("B", "R2")}'},
        {"label": "PCC · blend", "value": val("blend", "PCC"), "sub": f'A {val("A", "PCC")} · B {val("B", "PCC")}'},
        {"label": "Per-drug PCC · blend", "value": val("blend", "per_drug_PCC"), "red": True,
         "sub": f'A {val("A", "per_drug_PCC")} · B {val("B", "per_drug_PCC")}'},
        {"label": "Weight of A", "value": f'{res["w_mean"]:.2f}',
         "sub": f'mean over folds · range {res["w_min"]:.2f}–{res["w_max"]:.2f}'},
    ])
    f4 = lambda v: "—" if v is None or not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")  # noqa: E731
    va, vb = res["vs_a"], res["vs_b"]
    T.kpis([
        {"label": "Blend − A · ΔRMSE", "value": f'{va["mean"]:+.4f}',
         "sub": f'blend lower on {va["wins"]} of {va["n"]} folds · sign {f4(va["p_sign"])} · Wilcoxon '
                f'{f4(va["p_wilcoxon"])} · t {f4(va["p_t"])}'},
        {"label": "Blend − B · ΔRMSE", "value": f'{vb["mean"]:+.4f}',
         "sub": f'blend lower on {vb["wins"]} of {vb["n"]} folds · sign {f4(vb["p_sign"])} · Wilcoxon '
                f'{f4(vb["p_wilcoxon"])} · t {f4(vb["p_t"])}'},
    ])
    st.markdown(f'<div class="nt-note" style="margin:6px 0 8px 0">{T.esc(BL.verdict(res))}</div>',
                unsafe_allow_html=True)
    st.dataframe(res["weights"], hide_index=True, height=min(420, 38 + 35 * len(res["weights"])),
                 column_config={**{k: st.column_config.NumberColumn(format="%.4f")
                                   for k in ("RMSE_A", "RMSE_B", "RMSE_blend", "pdPCC_A", "pdPCC_B", "pdPCC_blend")},
                                "w_A": st.column_config.NumberColumn("weight of A", format="%.2f")})
    st.caption("Each fold's weight is fitted on the OTHER folds' held-out predictions of the same repeat (cross-fitting: "
               "the w minimising their squared error, between 0 and 1), so the blend's metrics are held-out scores — no "
               "fold chooses its own weight and nothing is retrained. Metrics: the same summary as the runs' own "
               "tabs. ΔRMSE < 0 = the blend is better.")
    d1, _ = st.columns([1, 4])
    d1.download_button("Blended predictions CSV", res["preds"].to_csv(index=False).encode(),
                       f"blend_{a[:15]}_{b[:15]}_{scheme}.csv", mime="text/csv", key="bl_dl")


def _shap_tab(rid, summ, data=""):
    feat, blocks, pw = U.run_csv(rid, "shap_features.csv"), U.run_csv(rid, "shap_blocks.csv"), U.run_csv(rid, "shap_pathways.csv")
    info = (summ or {}).get("shap") or {}
    if blocks.empty:
        st.caption(info.get("error") or "SHAP is computed automatically when a run finishes.")
        return
    T.chips([x for x in (info.get("method", "SHAP"), info.get("fold") and f'best fold: {info["fold"]}',
                         info.get("background") and f'{info["background"]} background rows',
                         info.get("explained") and f'{info["explained"]} held-out pairs explained',
                         f'unit: {info.get("unit", C.UNIT)}') if x], "dim")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown('<div class="nt-label">Share of attribution per feature block</div>', unsafe_allow_html=True)
        CH.show(CH.blocks_bar(blocks), 220)
    with c2:
        st.markdown('<div class="nt-label">Top pathways · mean |SHAP| summed over DG / EXP / MUT / CNV</div>',
                    unsafe_allow_html=True)
        CH.show(CH.pathways_bar(pw, 15), 400)
    st.markdown('<div class="nt-label">Top features</div>', unsafe_allow_html=True)
    st.dataframe(feat.head(25)[["feature", "block", "mean_abs", "direction"]], hide_index=True,
                 column_config={"mean_abs": st.column_config.ProgressColumn("mean |SHAP|", format="%.4f",
                                                                            min_value=0.0, max_value=float(feat["mean_abs"].max()))})
    st.caption(("Positive SHAP pushes a prediction towards sensitivity (higher −log10 IC50 = lower IC50). "
                if D.unit_of(data)["ic50"] else
                "Positive SHAP pushes a prediction towards sensitivity (higher AAC = more growth inhibition). ")
               + "Direction = how the attribution moves with the feature value. Attribution is association, not causation.")
    att = U.run_csv(rid, "attention_pathways.csv")
    if not att.empty:
        st.markdown('<div class="nt-label" style="margin-top:14px">Drug-conditioned attention \u00b7 pathways the model '
                    'weighs most (pathway-graph model)</div>', unsafe_allow_html=True)
        CH.show(CH.attention_bar(att, 15), 400)
        st.caption(f"Mean attention over the explained held-out pairs; uniform = {att['uniform'].iloc[0]:.4f} "
                   "(1 / pathways). Top-5 share = how often the pathway is among the 5 most weighted for a pair.")
        st.dataframe(att.head(30), hide_index=True,
                     column_config={"mean_attention": st.column_config.NumberColumn("mean attention", format="%.4f"),
                                    "top5_share": st.column_config.NumberColumn("in top 5", format="%.2f"),
                                    "uniform": None, "most_by_drug": "drug weighing it most"})


def _external_tab(rid):
    ext = S.read_json(S.run_dir(rid) / "external_ccle.json", None)
    if ext is None:
        st.caption("External test on CCLE (paper: train on GDSC, apply to CCLE).")
        if st.button("Run the CCLE test", key="ext_run", type="primary"):
            with st.spinner("Predicting 11k CCLE pairs…"):
                from core.runner import external_test
                ext = external_test(rid)
            st.rerun()
        return
    if ext.get("error"):
        T.banner(T.esc(ext["error"]))
        return
    r = C.PAPER_REF["ccle"]
    T.kpis([
        {"label": "RMSE", "value": U.f3(ext["RMSE"]), "sub": f"paper {r['RMSE'][0]:.2f}", "hl": True},
        {"label": "MAE", "value": U.f3(ext["MAE"]), "sub": f"paper {r['MAE'][0]:.2f}"},
        {"label": "PCC", "value": U.f3(ext["PCC"]), "sub": "all pairs"},
        {"label": "Spearman", "value": U.f3(ext["SCC"]), "sub": "all pairs"},
        {"label": "Per-drug PCC", "value": U.f3(ext["per_drug_PCC"]), "sub": "within each drug", "red": True},
        {"label": "Pairs", "value": f'{ext["pairs"]:,}', "sub": f'{ext["drugs"]} drugs · {ext["cells"]} cell lines'},
    ])
    st.caption(f"Model: {'all-data model' if ext.get('model_kind') == 'final' else 'best-fold model'} · CCLE responses are capped at 8 µM, so part of the gap to GDSC "
               "is a dataset difference (as the paper notes).")


def _cross_tab(rid, cfg):
    """CSA cross-study generalization: this run's row of the benchmark matrix + the 5 × 5 matrix over runs"""
    from core import csa as CSA
    src = CSA.study_of(cfg.get("data", ""))
    t = CSA.run_table(rid)
    st.markdown(f'<div class="nt-label">This run · trained on {T.esc(src)} · each official-split model tested on '
                f'every study</div>', unsafe_allow_html=True)
    if t.empty:
        st.caption("Rows appear as the CSA-split models finish (cross-study rows need the cross-study test switched on "
                   "when the run starts, and the raw target).")
    else:
        show = t[["target", "kind", "splits", "pairs", "RMSE", "RMSE_sd", "R2", "PCC", "SCC", "per_drug_PCC",
                  "RMSE_norm", "R2_norm"]].rename(columns={"RMSE_sd": "± RMSE", "per_drug_PCC": "per-drug PCC",
                                                           "RMSE_norm": "RMSE ÷ within", "R2_norm": "R² ÷ within"})
        st.dataframe(show, hide_index=True, column_config={
            **{k: st.column_config.NumberColumn(format="%.4f") for k in ("RMSE", "± RMSE", "R2", "PCC", "SCC",
                                                                         "per-drug PCC")},
            "RMSE ÷ within": st.column_config.NumberColumn(format="%.2f"),
            "R² ÷ within": st.column_config.NumberColumn(format="%.2f"),
            "pairs": st.column_config.NumberColumn(format="%d")})
        st.caption("Within = the 10 splits' own test sets; cross = all pairs of the other study (its _all set), "
                   "mean ± sd over the 10 split models. AAC; ÷ within = the benchmark's normalised score Gn "
                   "(RMSE ÷ within > 1 / R² ÷ within < 1 = the model generalises worse than within its own study). "
                   "Cross-study inputs are clipped to each feature's range in the split's training data, so unseen "
                   "drugs cannot land hundreds of sd outside what the network saw.")

    # ---------------------------------------------------------------- the matrix over runs
    st.markdown('<div class="nt-label" style="margin-top:16px">Benchmark matrix · one run per source study</div>',
                unsafe_allow_html=True)
    runs = [r for r in S.list_runs() if D.is_csa(r["config"].get("data", "")) and r["state"] != "failed"
            and C.CSA_SCHEME in (r["config"].get("schemes") or [])]
    tag = U.model_tag(cfg)
    lab = {r["id"]: U.run_label(r) for r in runs}
    c = st.columns(len(CSA.STUDIES) + 1)
    metric = c[0].selectbox("Metric", list(CSA.METRICS), index=list(CSA.METRICS).index("R2"), key="x_metric",
                            format_func=lambda m: {"R2": "R²", "per_drug_PCC": "per-drug PCC", "SCC": "Spearman"}.get(m, m))
    pick = {}
    for i, s in enumerate(CSA.STUDIES):
        cand = [r["id"] for r in runs if CSA.study_of(r["config"].get("data", "")) == s]
        same = [x for x in cand if U.model_tag(S.config(x)) == tag]            # default: same model + upgrades
        default = rid if s == src else (same[0] if same else None)
        opts = [None] + cand
        pick[s] = c[i + 1].selectbox(f"from {s}", opts, index=opts.index(default) if default in opts else 0,
                                     key=f"x_pick_{rid}_{s}",
                                     format_func=lambda x: "—" if x is None else lab.get(x, x)[:42])
    G, Gn, Ga, Gna, ns = CSA.matrix(pick, metric)
    hb = CSA.HIGHER_BETTER[metric]
    if G.notna().to_numpy().sum() == 0:
        st.caption("No finished CSA-split folds in the chosen runs yet.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="nt-label">G · {T.esc(metric)} (mean over splits)</div>', unsafe_allow_html=True)
        CH.show(CH.matrix_heat(G, ".3f", hb), 300)
    with c2:
        st.markdown('<div class="nt-label">Gn · ÷ within-study (diagonal = 1)</div>', unsafe_allow_html=True)
        CH.show(CH.matrix_heat(Gn, ".2f", hb), 300)
    summ = pd.DataFrame({"source": list(CSA.STUDIES), "within G[s,s]": [G.loc[s, s] for s in CSA.STUDIES],
                         "Ga (cross mean)": Ga.values, "Gna (cross mean ÷ within)": Gna.values,
                         "studies tested": [int((ns.loc[s] > 0).sum()) for s in CSA.STUDIES]})
    summ = summ[summ["studies tested"] > 0]
    st.dataframe(summ, hide_index=True, column_config={
        "within G[s,s]": st.column_config.NumberColumn(format="%.4f"),
        "Ga (cross mean)": st.column_config.NumberColumn(format="%.4f"),
        "Gna (cross mean ÷ within)": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Rows = the study the models were trained on, columns = the study they were tested on. "
               "G[s, t] = mean over the 10 split models; Ga = mean over the other four studies; Gn = G ÷ G[s, s]; "
               "Gna = mean Gn over the other four — the IMPROVE CSA scores. Pick one run per source study "
               "(default: the latest run with the same model and upgrades as this one).")
    d1, _ = st.columns([1, 4])
    d1.download_button("Matrix CSV", pd.concat({"G": G, "Gn": Gn}).to_csv().encode(), f"csa_matrix_{metric}.csv",
                       mime="text/csv", key="x_dl")


def _compare():
    rows = []
    for r in S.list_runs():
        c = r["config"]
        if r["state"] in ("failed",):
            continue
        sm = U.scheme_summaries(r["id"])
        timing = (S.summary(r["id"]) or {}).get("timing", {})
        for s, m in sm.items():
            no_int = c.get("model") == "proposed" and not (c.get("spec") or {}).get("interaction", True)
            gm = (c.get("spec") or {}).get("g_mode") if (c.get("spec") or {}).get("arch") == "graph" else None
            rows.append({"run": r["id"], "model": C.MODEL_SPECS.get(c.get("model"), {}).get("name", "?")
                         + (" · no interaction module" if no_int else "") + (f" · {gm}" if gm else ""),
                         "upgrades": C.upgrades_tag(c.get("spec") or {}) or "—",
                         "dropout · wd": ("—" if c.get("model") == C.GBM_MODEL else
                                          f'{(c.get("spec") or {}).get("dropout", "?")} · '
                                          f'{float((c.get("spec") or {}).get("weight_decay", 0)):g}'),
                         "target": {"drug_z": "drug-wise z", "study_z": "study-wise z (S)"}.get(
                             (c.get("spec") or {}).get("target"), "raw"),
                         "data": D.dataset_name(c.get("data", "?")), "evaluation": C.SCHEMES[s]["name"],
                         "RMSE": m.get("RMSE"), "± ": m.get("RMSE_sd"), "MAE": m.get("MAE"), "R²": m.get("R2"),
                         "PCC": m.get("PCC"), "per-drug PCC": m.get("per_drug_PCC"),
                         "folds": m.get("n_folds"), "time": U.hms(timing.get("wall_seconds")) if timing else "—",
                         "state": r["state"]})
    if not rows:
        st.caption("Finished runs appear here.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, height=min(600, 38 + 35 * len(df)),
                 column_config={k: st.column_config.NumberColumn(format="%.4f")
                                for k in ("RMSE", "± ", "MAE", "R²", "PCC", "per-drug PCC")})
    st.caption("Same dataset + evaluation = directly comparable. Lower RMSE / MAE and higher R² / PCC / per-drug PCC are better.")


def render():
    runs = S.list_runs()
    if not runs:
        T.banner("No runs yet — start one on <b>Train</b>.")
        return
    ids = [r["id"] for r in runs]
    want = st.session_state.get("res_run")
    lab = {r["id"]: U.run_label(r) for r in runs}
    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    rid = c1.selectbox("Run", ids, index=ids.index(want) if want in ids else 0, format_func=lambda i: lab[i],
                       key="res_select", label_visibility="collapsed")
    st.session_state["res_run"] = rid
    state = S.live_state(S.status(rid))
    with c2.popover("Delete", width="stretch", disabled=state in LIVE):
        st.caption(f"Delete run {rid} and its model?")
        if st.button("Delete permanently", key="res_del", type="primary"):
            S.delete_run(rid)
            st.session_state.pop("res_run", None)
            st.rerun()

    def body():
        cfg = S.config(rid)
        stt = S.status(rid)
        state_now = S.live_state(stt)
        summ = S.summary(rid)
        spec = cfg.get("spec", {})
        timing = (summ or {}).get("timing") or {}
        dev_name = str(timing.get("device") or (stt.get("gpu") or {}).get("name") or "")
        st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:2px 0 6px 0">'
                    f'{T.state_chip(state_now)}{T.chip(C.MODEL_SPECS.get(cfg.get("model"), {}).get("name", "?"), "red")}'
                    f'{T.chip(D.dataset_name(cfg.get("data", "?")))}'
                    f'{T.chip("total " + U.hms(timing.get("wall_seconds") or stt.get("elapsed")))}'
                    f'{T.chip(dev_name, "dim") if dev_name else ""}</div>',
                    unsafe_allow_html=True)
        T.chips(U.settings_chips(spec, cfg.get("folds"), D.unit_of(cfg.get("data", ""))["name"])
                + ([cfg["note"]] if cfg.get("note") else []), "dim")
        sums = U.scheme_summaries(rid)
        folds, preds, curves = U.run_csv(rid, "folds.csv"), U.run_csv(rid, "preds.csv"), U.run_csv(rid, "curves.csv")
        tabs = [C.SCHEMES[s]["name"] for s in sums] + ["SHAP · pathways"]
        ext_on = D.external_for(cfg.get("data", "")) and S.model_paths(rid)[0].exists()
        if ext_on:
            tabs.append("CCLE external")
        csa_on = D.is_csa(cfg.get("data", "")) and (D.meta(cfg.get("data", "")) or {}).get("study") != "ALL"
        if csa_on:
            tabs.append("Cross-study")
        tabs.append("Compare runs")
        tt = st.tabs(tabs)
        for i, (s, m) in enumerate(sums.items()):
            with tt[i]:
                _scheme_tab(rid, s, m, folds, preds, curves, cfg.get("data", ""))
        with tt[len(sums)]:
            _shap_tab(rid, summ, cfg.get("data", ""))
        if ext_on:
            with tt[len(sums) + 1]:
                _external_tab(rid)
        if csa_on:
            with tt[len(sums) + 1 + int(bool(ext_on))]:
                _cross_tab(rid, cfg)
        with tt[-1]:
            _compare()
            _paired_section(rid)
            _blend_section(rid)
        if not sums:
            st.caption("Metrics appear as soon as the first fold-model finishes.")

    st.fragment(body, run_every=2 if state in LIVE else None)()
