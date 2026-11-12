"""PREDICT — pick a trained model, a drug and a cancer cell line."""
import numpy as np
import streamlit as st

from core import data as D
from core import metrics as MT
from core import store as S

from . import charts as CH
from . import common as U
from . import theme as T


def _model_label(m):
    cv = (m.get("cv") or {}).get("mixed") or next(iter((m.get("cv") or {}).values()), {}) or {}
    rm = f" · CV RMSE {cv['RMSE']:.3f}" if cv.get("RMSE") is not None else ""
    kind = "all-data model" if m.get("kind") == "final" else "best fold"
    tgt = " · z-target" if m.get("target") == "drug_z" else ""
    tag = m.get("tag", "?") + (f"+{m['upgrades']}" if m.get("upgrades") else "")
    role = _role(m)
    return (f"{role}  ·  " if role else "") + f"{tag}  ·  {m.get('data_name', '?')}  ·  {kind}{tgt}{rm}  ·  {m['id'][:15]}"


def _role(m):
    """presentation name of a model: Base paper (the authors' FNN as published), Model A (same FNN + recipe and
    drug-wise z target), Model B (pathway-graph model trained on all 5 CSA studies; Run A raw AAC, Run B S·R·M)"""
    rid, up = m.get("run_id") or m.get("id", ""), m.get("upgrades") or ""
    final = rid.rsplit("_", 1)[-1] if m.get("kind") == "final" and "_F" in rid else ""
    fin = f" · {final.split('-')[0]} final" if final else ""
    if m.get("model") == "paper" and not up:
        return "BASE PAPER (rerun)"
    if m.get("model") == "paper" and m.get("target") == "drug_z":
        return "MODEL A" + fin
    if m.get("model") == "graph" and str(m.get("data", "")) == "csa-ALL":
        run = "Run B" if m.get("target") == "study_z" else "Run A"
        return f"MODEL B · {run}" + (fin or " · leave-one-study-out")
    return ""


ROLE_ORDER = ["MODEL A", "BASE PAPER (rerun)", "MODEL A · F1 final", "MODEL B · Run A · leave-one-study-out",
              "MODEL B · Run B · leave-one-study-out", "MODEL B · Run A · F2 final", "MODEL B · Run B · F3 final"]


def _order(m):
    """demo models first (Model A, Base paper, Model A final, Model B …); within a role the newest run that has held-out
    predictions of the main scheme (10-fold or leave-one-study-out), so its interval and held-out error can be shown"""
    role = _role(m)
    rank = ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER)
    main = 0 if set(m.get("cv") or {}) & {"mixed", "loso"} else 1
    return rank, main, "".join(chr(255 - ord(ch)) for ch in m["id"])


@st.fragment(run_every=2)
def _watch_models(n_known):
    """reloads the page when a training run adds a model"""
    if len(S.list_models()) != n_known:
        st.rerun(scope="app")


def render():
    models = U.models()
    _watch_models(len(models))
    if not models:
        T.banner("No trained model yet — train one on <b>Train</b> (every run saves its model automatically).")
        return

    # ------------------------------------------------------------------ 01 model
    T.step(1, "Model", "every finished run adds one")
    ids = [m["id"] for m in models]
    by_id = {m["id"]: m for m in models}
    ids = sorted(ids, key=lambda i: _order(by_id[i]))
    mid = st.selectbox("Model", ids, format_func=lambda i: _model_label(by_id[i]), key="p_model",
                       label_visibility="collapsed")
    meta = by_id[mid]
    cv = (meta.get("cv") or {})
    cvm = cv.get("mixed") or next(iter(cv.values()), {}) or {}
    T.chips([meta.get("model_name", "?"), meta.get("data_name", "?"),
             ("trained on all " if meta.get("kind") == "final" else "trained on ") + f"{meta.get('n_train', 0):,} pairs",
             f"CV RMSE {U.f3(cvm.get('RMSE'))}" + (f" ± {cvm['RMSE_sd']:.3f}" if cvm.get("RMSE_sd") else ""),
             f"PCC {U.f3(cvm.get('PCC'))}", f"per-drug PCC {U.f3(cvm.get('per_drug_PCC'))}",
             "target drug-wise z" if meta.get("target") == "drug_z" else f"target {D.unit_of(meta['data'])['name']}",
             "runs on CPU"], "dim")

    ds = U.dataset(meta["data"])
    unit = D.unit_of(meta["data"])
    cinfo, dinfo = U.cell_info(), U.drug_info()

    # ------------------------------------------------------------------ 02 drug + cell line
    T.step(2, "Drug and cancer cell line", f"{len(ds.drug_list)} drugs · {len(ds.cell_list)} cell lines in this dataset")
    c1, c2 = st.columns(2)
    with c1:
        drugs = _search("Search drug", ds.drug_list, lambda d: _drug_text(d, dinfo), f"p_dq_{meta['data']}",
                        "name, target or pathway — e.g. BRAF, Lapatinib, apoptosis", "drugs")
        if not drugs:
            return
        drug = st.selectbox("Drug", drugs, key=f"p_drug_{meta['data']}")
        di = dinfo.get(drug, {})
        prof = [f"Targets: {di.get('targets') or '—'}", f"Pathway: {di.get('pathway') or '—'}"]
        if di.get("gdsc_id"):
            prof.append(f"GDSC id {di['gdsc_id']}")
        if di.get("improve_id"):
            prof.append(f"IMPROVE id {di['improve_id']}")
        st.markdown('<div class="nt-note">' + " · ".join(T.esc(x) for x in prof) + "</div>", unsafe_allow_html=True)
    with c2:
        cell_labels = {c: _cell_label(c, cinfo) for c in ds.cell_list}
        order = sorted(ds.cell_list, key=lambda c: cell_labels[c].lower())
        cells = _search("Search cancer cell line", order, lambda c: _cell_text(c, cinfo, cell_labels), f"p_cq_{meta['data']}",
                        "name, cancer type, tissue or id — e.g. A375, melanoma, breast", "cell lines")
        if not cells:
            return
        cell = st.selectbox("Cancer cell line", cells, format_func=lambda c: cell_labels[c], key=f"p_cell_{meta['data']}")
        ci = cinfo.get(cell, {})
        prof = [f"Cell line: {ci.get('name') or cell}", f"Tissue: {ci.get('tissue') or ci.get('lineage') or '—'}",
                f"Cancer type: {ci.get('cancer_type') or '—'}"]
        if ci.get("tcga"):
            prof.append(f"TCGA {ci['tcga']}")
        prof.append(f"Sanger {ci.get('sanger_id') or cell}")
        st.markdown('<div class="nt-note">' + " · ".join(T.esc(x) for x in prof) + "</div>", unsafe_allow_html=True)
    st.write("")
    go = st.button("Predict sensitivity", type="primary", width="stretch", key="p_go")
    key = (mid, drug, cell)
    if go:
        st.session_state["p_last"] = key
    if st.session_state.get("p_last") != key:
        return

    # ------------------------------------------------------------------ 03 output
    T.step(3, "Prediction", "")
    pr = U.predictor(mid)
    x = ds.feature_row(drug, cell)[None, :]
    if list(pr.features) != list(ds.features):
        pos = {f: i for i, f in enumerate(ds.features)}
        x = np.array([[x[0, pos[f]] if f in pos else np.nan for f in pr.features]], dtype="float32")
    pred = float(pr(x, np.array([drug]))[0])
    row = ds.row_of(drug, cell)
    measured = float(ds.y[row]) if row is not None else None
    vals, cells = ds.drug_responses(drug)
    pct, label = MT.sensitivity_class(pred, vals)
    held = _heldout(meta["run_id"], drug, cell)
    ic50 = 10 ** (-pred) if unit["ic50"] else float("nan")
    err_v, err_s = "—", "pair not measured in the dataset"
    if measured is not None:
        if held is not None:
            err_v, err_s = f"{abs(held - measured):.4f}", f"held-out prediction {held:.3f} (this pair was in a test fold)"
        else:
            err_v, err_s = f"{abs(pred - measured):.4f}", "|this prediction − measured|"
    qd, qg = U.drug_widths(meta["run_id"], 0.90)
    q = qd.get(drug, qg) if (qd or np.isfinite(qg)) else float("nan")
    iv_item = []
    if np.isfinite(q):
        src = "conformal, from " + ("this drug's held-out errors" if drug in qd else "all drugs' held-out errors")
        iv_item = [{"label": "90 % interval", "value": f"{pred - q:.2f} – {pred + q:.2f}",
                    "sub": (f"IC50 {_fmt_ic50(10 ** (-(pred + q)))}–{_fmt_ic50(10 ** (-(pred - q)))} µM · " if unit["ic50"]
                            else f"{unit['name']} · ") + src}]
    meas_sub = "not screened"
    if measured is not None:
        meas_sub = f"IC50 {_fmt_ic50(10 ** (-measured))} µM" if unit["ic50"] else f"{unit['name']}, measured"
    T.kpis([
        {"label": "Predicted score", "value": f"{pred:.4f}", "sub": unit["long"], "hl": True},
        *iv_item,
        *([{"label": "IC50", "value": _fmt_ic50(ic50), "sub": "estimated, µM"}] if unit["ic50"] else []),
        {"label": "Measured", "value": f"{measured:.4f}" if measured is not None else "—", "sub": meas_sub},
        {"label": "Absolute error", "value": err_v, "sub": err_s},
        {"label": "Diagnosis", "value": label, "red": label == "Sensitive",
         "sub": f"more sensitive than {pct:.0f}% of {len(vals)} cell lines tested with {drug}"},
    ])
    st.write("")
    nm = cinfo.get(cell, {}).get("name") or cell
    ct = cinfo.get(cell, {}).get("cancer_type") or ""
    what = (f"an IC50 of about <b>{_fmt_ic50(ic50)} µM</b>" if unit["ic50"] else
            f"an AAC of <b>{pred:.3f}</b> (share of growth inhibited over the tested doses)")
    T.banner(f"<b>Interpretation</b> — for <b>{T.esc(drug)}</b> on <b>{T.esc(nm)}</b>"
             f"{(' (' + T.esc(ct) + ')') if ct else ''}, the model predicts {what}: "
             f"<b>{label.lower()}</b> relative to the other cell lines screened with this drug "
             f"(rank {100 - pct:.0f}% from the most sensitive). Research use only — not a clinical recommendation.",
             ok=label != "Sensitive")
    st.write("")
    st.markdown(f'<div class="nt-label">{T.esc(drug)} across all {len(vals)} screened cell lines · '
                f'red = this prediction · {"black" if T.mode() == "light" else "white"} = measured</div>', unsafe_allow_html=True)
    CH.show(CH.drug_strip(vals, cells, pred, measured, {c: _cell_label(c, cinfo) for c in cells}, unit["name"]), 170)


def _search(label, options, text_of, key, placeholder, noun):
    """a search box above a list: keeps the options whose text contains every word typed (case-insensitive)"""
    q = st.text_input(label, key=key, placeholder=placeholder, icon=":material/search:")
    words = [w for w in q.lower().split() if w]
    if not words:
        return list(options)
    hits = [o for o in options if all(w in text_of(o) for w in words)]
    if hits:
        st.markdown(f'<div class="nt-match"><b>{len(hits)}</b> of {len(options)} {noun} match “{T.esc(q.strip())}”</div>',
                    unsafe_allow_html=True)
    else:
        T.banner(f"No {noun} match “{T.esc(q.strip())}” in this dataset — try a shorter word, or clear the search.")
    return hits


def _drug_text(d, dinfo):
    i = dinfo.get(d) or {}
    return " ".join(str(x) for x in (d, i.get("targets"), i.get("pathway"), i.get("gdsc_id"), i.get("improve_id")) if x).lower()


def _cell_text(c, cinfo, labels):
    i = cinfo.get(c) or {}
    return " ".join(str(x) for x in (labels.get(c, c), c, i.get("tissue"), i.get("lineage"), i.get("cancer_type"),
                                     i.get("tcga"), i.get("sanger_id"), i.get("depmap_id"), i.get("cosmic_id")) if x).lower()


def _cell_label(c, cinfo):
    i = cinfo.get(c) or {}
    name = i.get("name") or c
    ctype = i.get("cancer_type") or i.get("lineage") or ""
    return f"{name} — {ctype} [{c}]" if ctype else f"{name} [{c}]"


def _fmt_ic50(v):
    if not np.isfinite(v):
        return "—"
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def _heldout(rid, drug, cell):
    p = U.run_csv(rid, "preds.csv")
    if p.empty:
        return None
    for sch in ("mixed", "csa", "loso"):
        q = p[(p["scheme"] == sch) & (p["drug"] == drug) & (p["cell"] == cell)]
        if not q.empty:
            return float(q.sort_values(["repeat", "fold"]).iloc[0]["pred"])
    return None
