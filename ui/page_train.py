"""TRAIN — choose model, data and evaluation; the model's own defaults are pre-filled."""
import datetime as _dt
import hashlib
import json
import os
import re

import streamlit as st

from core import config as C
from core import data as D
from core import nets as N
from core import store as S
from core.runner import validate

from . import common as U
from . import theme as T


def _reset(model):
    for k in list(st.session_state):
        if k.startswith(f"t_{model}_"):
            del st.session_state[k]


def _n_models(ds, schemes, grouping, repeats, folds, final, seeds=1):
    n = int(final)
    for s in schemes:
        if s == C.CSA_SCHEME:
            from core import csa as CSA
            n += CSA.N_SPLITS
        elif s == C.LOSO_SCHEME:
            n += max(1, ds.n_studies)
        elif s != "mixed" and grouping == "logo":
            n += len(ds.cell_list) if s == "LOCO" else len(ds.drug_list)
        else:
            n += folds * repeats
    return n * max(1, int(seeds))


def _wd_input(col, model, d):
    return float(col.selectbox("Weight decay (L2)", C.WEIGHT_DECAYS, index=C.WEIGHT_DECAYS.index(float(d["weight_decay"])),
                               key=f"t_{model}_wd", format_func=lambda v: "0" if v == 0 else f"{v:.0e}",
                               help="Paper: 1e-5 — added to the gradient as in torch.optim (L2 penalty)"))


def _upgrades(model, spec, d, multi=False, csa=False):
    """A–D (config.UPGRADES); each is a separate switch, all off by default"""
    ups = C.UPGRADES
    ok = lambda k: model in ups[k]["models"]                                          # noqa: E731
    st.markdown('<div class="nt-label" style="margin-top:10px">Upgrades — each on its own switch · all off = the model '
                'exactly as defined above</div>', unsafe_allow_html=True)
    u = st.columns(5)

    def label(k):
        return f'{ups[k]["letter"]} · {ups[k]["name"]}'

    def hlp(k):
        return ("" if ok(k) else "Proposed system only. ") + ups[k]["help"]
    with u[0]:
        spec["fac"] = bool(st.toggle(label("fac"), value=bool(d.get("fac")), key=f"t_{model}_fac",
                                     disabled=not ok("fac"), help=hlp("fac"))) and ok("fac")
        if spec["fac"]:
            spec["fac_dim"] = int(st.selectbox("Rank", C.FAC_RANKS, index=C.FAC_RANKS.index(int(d.get("fac_dim", 16))),
                                               key=f"t_{model}_facdim",
                                               help="size of the drug and cell embeddings whose dot product is added"))
    with u[1]:
        inter = bool(spec.get("interaction"))
        g = st.toggle(label("gate"), value=bool(d.get("gate")), key=f"t_{model}_gate",
                      disabled=not ok("gate") or not inter,
                      help=hlp("gate") + ("" if inter or not ok("gate") else " — switch the interaction module on first."))
        spec["gate"] = bool(g) and ok("gate") and inter
    with u[2]:
        spec["plateau"] = bool(st.toggle(label("plateau"), value=bool(d.get("plateau")), key=f"t_{model}_plateau",
                                         help=hlp("plateau")))
        if spec["plateau"]:
            spec["plateau_patience"] = int(st.number_input(
                "LR patience (epochs)", 1, 50, int(d.get("plateau_patience", 5)), 1, key=f"t_{model}_ppat",
                help="epochs without a significant improvement before the learning rate is halved"))
            spec["plateau_delta"] = float(st.selectbox(
                "Min. improvement", (0.0, 5e-4, 1e-3, 2e-3, 5e-3), index=2, key=f"t_{model}_pdelta",
                format_func=lambda v: "any" if v == 0 else f"{v:g}",
                help="a validation loss lower than the best by less than this does not reset the LR patience"))
    with u[3]:
        on = st.toggle(label("ensemble"), value=int(d.get("ensemble", 1)) > 1, key=f"t_{model}_ens",
                       help=hlp("ensemble"))
        n = 1
        if on:
            n = int(st.number_input("Seeds per fold", 2, C.MAX_ENSEMBLE, 3, 1, key=f"t_{model}_ens_n",
                                    help="models trained per fold; their predictions are averaged"))
        spec["ensemble"] = n
    with u[4]:
        spec["ema"] = bool(st.toggle(label("ema"), value=bool(d.get("ema")), key=f"t_{model}_ema", help=hlp("ema")))
    u2 = st.columns(5)
    with u2[0]:
        spec["loss"] = "mse" if st.toggle(label("loss"), value=d.get("loss", "rmse") == "mse", key=f"t_{model}_mse",
                                          help=hlp("loss")) else "rmse"
    with u2[1]:
        spec["val_metric"] = "global" if st.toggle(label("val_metric"), value=d.get("val_metric", "batch") == "global",
                                                   key=f"t_{model}_gval", help=hlp("val_metric")) else "batch"
    with u2[2]:
        spec["rank"] = bool(st.toggle(label("rank"), value=bool(d.get("rank")), key=f"t_{model}_rank",
                                      help=hlp("rank")))
        if spec["rank"]:
            spec["rank_weight"] = float(st.selectbox("Ranking weight", C.RANK_WEIGHTS,
                                                     index=C.RANK_WEIGHTS.index(float(d.get("rank_weight", 0.5))),
                                                     key=f"t_{model}_rankw", format_func=lambda v: f"{v:g}",
                                                     help="loss = training loss + this × ranking loss"))
    with u2[3]:
        on = st.toggle(label("adv"), value=bool(d.get("adv")) and multi, key=f"t_{model}_adv", disabled=not multi,
                       help=("" if multi else "Needs 'CSA · all 5 studies'. ") + C.UPGRADES["adv"]["help"])
        spec["adv"] = bool(on) and multi
        if spec["adv"]:
            spec["adv_weight"] = float(st.selectbox("Adversarial weight", C.ADV_WEIGHTS,
                                                    index=C.ADV_WEIGHTS.index(float(d.get("adv_weight", 0.1))),
                                                    key=f"t_{model}_advw", format_func=lambda v: f"{v:g}",
                                                    help="strength of the reversed gradient once ramped up"))
    with u2[4]:
        spec["mol"] = bool(st.toggle(label("mol"), value=False, key=f"t_{model}_mol", disabled=not csa,
                                     help=("" if csa else "CSA datasets only. ") + C.UPGRADES["mol"]["help"])) and csa
    if model in ("paper", "repo") and C.upgrades_tag(spec):
        st.caption(f'{C.MODEL_SPECS[model]["name"]} with {C.upgrades_tag(spec)} is no longer the exact recipe — the run is '
                   f'marked {C.MODEL_SPECS[model]["tag"]}+{C.upgrades_tag(spec)} everywhere.')


@U.ttl_cache(3)
def _catalog():
    return D.catalog(), D.available()


@st.fragment(run_every=3)
def _watch_data_folder(known):
    """a dataset file copied into data/ appears without reloading the page"""
    if tuple(D.catalog()) != known:
        _catalog.clear()
        st.rerun(scope="app")


def _save_upload(f):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(f.name)) or "dataset.txt"
    if name in D._BUILTIN_FILES:
        name = "custom_" + name
    dst = C.DATA / name
    tmp = C.DATA / (name + ".part")
    with open(tmp, "wb") as out:
        while True:
            chunk = f.read(16 * 2 ** 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dst)
    return name


def _data_section():
    T.step(2, "Data", "built-in datasets + any dataset file you put into the data folder")
    cat, avail = _catalog()
    keys = list(cat)
    _watch_data_folder(tuple(keys))
    label = lambda k: cat[k]["name"] + ("" if avail.get(k) else "  (missing)")       # noqa: E731
    if st.session_state.get("t_data") not in keys:
        st.session_state.pop("t_data", None)
    if len(keys) <= 5:
        data = st.segmented_control("Dataset", keys, default="input", key="t_data", required=True, format_func=label,
                                    label_visibility="collapsed") or "input"
    else:
        data = st.selectbox("Dataset", keys, index=keys.index("input"), key="t_data", format_func=label,
                            label_visibility="collapsed")
    with st.expander("Add a dataset — upload a file, or copy it into the data folder"):
        st.markdown(
            '<div class="nt-note">Accepted: a table (.txt / .tsv / .csv, or a .zip holding one) in the layout of the '
            "authors' input.txt — columns <code>drug</code>, <code>cell</code>, feature columns with the prefixes "
            "<code>CHEM_</code> <code>DG_</code> (or <code>TARGETnNEIGHBOR_</code>) <code>EXP_</code> <code>MUT_</code> "
            "<code>CNV_</code>, and a response column <code>resp</code> (−log10 IC50; <code>LN_IC50</code> or "
            "<code>IC50</code> in µM are converted) — or another processed_data zip. Files copied into "
            f"<code>{T.esc(C.DATA)}</code> appear here automatically.</div>", unsafe_allow_html=True)
        up = st.file_uploader("Dataset file", type=["txt", "tsv", "csv", "zip"], key="t_upload",
                              label_visibility="collapsed")
        if up is not None and st.session_state.get("t_upload_done") != (up.name, up.size):
            with st.spinner(f"Saving {up.name}…"):
                saved = _save_upload(up)
            st.session_state["t_upload_done"] = (up.name, up.size)
            _catalog.clear()
            st.toast(f"Saved {saved} to the data folder")
            st.rerun()
    if not avail.get(data):
        T.banner(f"<b>{T.esc(cat[data].get('archive', cat[data]['name']))}</b> is not in "
                 f"<code>{T.esc(C.DATA)}</code>. Put it there — it appears here automatically.")
        return data, None
    try:
        with st.spinner("Preparing the dataset (first time only: large files take 10–60 s)…"):
            ds = U.dataset(data)
    except Exception as e:
        T.banner(f"<b>{T.esc(cat[data]['name'])}</b> could not be read: {T.esc(e)}")
        if not cat[data].get("builtin"):
            if st.button("Remove this file from the data folder", key="t_rm_data"):
                p = cat[data].get("path")
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                except OSError:
                    pass
                _catalog.clear()
                st.session_state.pop("t_data", None)
                st.rerun()
        return data, None
    sm = D.summary(ds)
    T.chips([cat[data]["label"], f"{sm['pairs']:,} drug–cell pairs", f"{sm['drugs']} drugs",
             f"{sm['cells']} cell lines", f"{sm['features']:,} features",
             " · ".join(f"{k} {v}" for k, v in sm["blocks"].items()), "response " + D.unit_of(data)["long"]], "dim")
    if D.is_csa(data):
        st.markdown('<div class="nt-note">IMPROVE cross-study benchmark (CCLE · CTRPv2 · gCSI · GDSCv1 · GDSCv2): PathDSP '
                    "features built for all five studies the same way (tools/build_csa.py), so a cell line or drug has the "
                    "same features everywhere. Response = AAC (1 − AUC; higher = more sensitive). Choose <b>CSA splits</b> "
                    "below for the benchmark's own 10 splits and the cross-study test.</div>", unsafe_allow_html=True)
    return data, ds


@st.fragment(run_every=2)
def _triton_panel():
    ti = U.triton_info()
    job = U.triton_job()
    chk = U.fused_check_result()
    req, pin = U.triton_requirement()
    if ti["installed"]:
        state = f"Triton {ti['version'] or ''} installed"
        if chk and chk.get("triton"):
            ok = "fused" in str(chk.get("adamax", "")) and "standard" not in str(chk.get("adamax", ""))
            state += " · self-test passed — fused kernel in use" if ok else f" · self-test: {chk.get('adamax')}"
        else:
            state += " · press Test to check it on your GPU"
    else:
        state = "Triton not installed — standard PyTorch optimizer (correct, about 2× slower update step)"
    T.chips([state, f"PyTorch {ti.get('torch') or '?'}", f"matching build: {req}"], "dim")
    running = bool(job and job.get("state") == "running")
    c = st.columns([1.4, 1.2, 3])
    if c[0].button("Installing…" if running else ("Reinstall Triton" if ti["installed"] else "Install Triton"),
                   key="t_tri_install", width="stretch", disabled=running or S.active_run() is not None,
                   help=f"pip install --upgrade \"{req}\" into this environment (needs internet, ~1–3 min)"):
        U.start_triton_install()
        st.rerun(scope="fragment")
    if c[1].button("Test", key="t_tri_test", width="stretch", disabled=running or not ti["installed"]
                   or S.active_run() is not None, help="Compiles the fused kernel and compares it with PyTorch's optimizer"):
        with st.spinner("Compiling and testing the fused kernel on the GPU (first time ≈ 1 min)…"):
            try:
                U.check_fused()
            except Exception as e:
                st.error(f"Test failed: {e}")
        st.rerun(scope="fragment")
    if job:
        if running:
            st.caption(f"Installing {job.get('requirement')} …")
            st.code(U.triton_log_tail() or "starting pip…", language="text")
        elif job.get("state") == "failed":
            st.caption(f"Installing {job.get('requirement')} failed — the standard optimizer is used (training is "
                       "unaffected).")
            with st.expander("pip output"):
                st.code(U.triton_log_tail(30), language="text")
        elif job.get("state") == "done" and not chk:
            st.caption("Triton installed — press Test to check the fused kernel on your GPU.")


@st.fragment(run_every=2)
def _gbm_panel():
    """(v5.9) LightGBM: installed? — install it from the app when it is missing"""
    from core import gbm as G
    job = U.gbm_job()
    running = bool(job and job.get("state") == "running")
    if G.available() and not running:
        T.chips([f"LightGBM {G.version() or ''} installed", f"CPU · {G.threads()} physical cores"], "dim")
        return
    if not running:
        T.banner("LightGBM is not installed in this environment. Press <b>Install LightGBM</b> (pip, ~1 min, needs "
                 "internet) or run <code>.venv\\Scripts\\python -m pip install lightgbm</code>.")
    c = st.columns([1.4, 4])
    if c[0].button("Installing…" if running else "Install LightGBM", key="t_gbm_install", width="stretch",
                   disabled=running or S.active_run() is not None):
        U.start_gbm_install()
        st.rerun(scope="fragment")
    if job:
        if running:
            st.code(U.gbm_log_tail() or "starting pip…", language="text")
        elif job.get("state") == "failed":
            st.caption("Installing LightGBM failed — the pip output:")
            st.code(U.gbm_log_tail(30), language="text")
        elif job.get("state") == "done" and G.available():
            st.rerun(scope="app")


def render():
    act = S.active_run()
    if act:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            T.banner(f"<b>A run is training</b> — {T.esc(act['id'])}. One run at a time uses the GPU; open "
                     f"<b>Live</b> to follow it or stop it.")
        with c2:
            if st.button("Open live", key="t_open_live", width="stretch"):
                st.switch_page(st.session_state["_pages"]["live"])

    # ------------------------------------------------------------------ 01 model
    T.step(1, "Model", "the PathDSP networks + LightGBM")
    model = st.segmented_control("Model", C.MODEL_KEYS, default="paper", key="t_model", required=True,
                                 format_func=lambda k: C.MODEL_SPECS[k]["name"], label_visibility="collapsed") or "paper"
    ms = C.MODEL_SPECS[model]
    st.markdown(f'<div class="nt-card"><div class="nt-label">{T.esc(ms["tag"])}</div>'
                f'<div style="font-size:1.02rem;margin:6px 0 10px 0">{T.esc(ms["one_line"])}</div>'
                + "".join(f'<div class="nt-note">— {T.esc(n)}</div>' for n in ms["notes"]) + "</div>",
                unsafe_allow_html=True)
    if model == C.GBM_MODEL:
        _gbm_panel()

    # ------------------------------------------------------------------ 02 data
    data, ds = _data_section()
    if ds is None:
        return

    # ------------------------------------------------------------------ 03 evaluation
    T.step(3, "Evaluation", "what the held-out test sets are")
    is_csa = D.is_csa(data)
    multi = (D.meta(data) or {}).get("study") == "ALL"
    opts = [s for s in C.SCHEMES if (s != C.CSA_SCHEME or (is_csa and not multi)) and (s != C.LOSO_SCHEME or multi)]
    schemes = st.pills("Evaluation", opts, selection_mode="multi",
                       default=[C.LOSO_SCHEME if multi else (C.CSA_SCHEME if is_csa else "mixed")],
                       key="t_schemes_all" if multi else ("t_schemes_csa" if is_csa else "t_schemes"),
                       format_func=lambda s: f'{C.SCHEMES[s]["name"]} · {C.SCHEMES[s]["long"]}',
                       label_visibility="collapsed") or []
    schemes = [s for s in opts if s in schemes]
    cross = False
    if C.CSA_SCHEME in schemes:
        cross = st.toggle("Cross-study test — every split model also predicts all pairs of the four other studies",
                          value=True, key="t_cross",
                          help="The benchmark's generalization test: G[s, t] = metric of the 10 split models trained on "
                               "study s, tested on all of study t (within-study = each split's own test set). Results → "
                               "Cross-study shows this run's row and the 5 × 5 matrix once runs for the other studies exist.")
    folds = int(st.session_state.get(f"t_{model}_folds", C.N_FOLDS))
    grouping = "logo"
    if any(s in ("LOCO", "LODO") for s in schemes):
        grouping = st.segmented_control(
            "Unseen cell lines / drugs", ["logo", "kfold"], default="logo", key="t_grouping", required=True,
            format_func=lambda g: "Leave-one-out (paper)" if g == "logo" else f"Grouped {folds}-fold (much faster)") or "logo"
    if multi:
        st.markdown('<div class="nt-note">Multi-study training: the model learns from four studies at once and is '
                    "tested on all pairs of the fifth, which it never saw (5 models). Results → Leave-study-out also "
                    "shows few-shot recalibration (k labelled pairs of the new study) and the comparison with "
                    "single-study models. Recommended: target <b>study-wise z (S)</b>.</div>", unsafe_allow_html=True)
    final = st.toggle("Also train the all-data model (used by Predict)", value=True, key="t_final")

    # ------------------------------------------------------------------ 04 settings
    T.step(4, "Settings", f"defaults of {ms['name']} — change only if you mean to")
    d = ms["defaults"]
    gbm = model == C.GBM_MODEL
    c = st.columns(5)
    if gbm:                                                 # (v5.9) LightGBM: boosting rounds instead of epochs
        epochs = c[0].number_input("Max boosting rounds", 10, 50000, int(d["epochs"]), 100, key=f"t_{model}_epochs",
                                   help="trees added at most; early stopping usually ends a model much earlier")
        patience = c[1].number_input("Early-stopping rounds", 1, 5000, int(d["patience"]), 10, key=f"t_{model}_patience",
                                     help="A model stops after this many rounds without a lower validation RMSE; "
                                          "the best round is kept")
        batch = int(d["batch"])
        gb_leaves = int(c[2].selectbox("Leaves per tree", C.GB_LEAVES, index=C.GB_LEAVES.index(int(d["gb_leaves"])),
                                       key=f"t_{model}_leaves", help="tree size — more leaves = more interactions "
                                                                     "per tree, more risk of overfitting"))
        lr = c[3].number_input("Learning rate", 0.001, 1.0, float(d["lr"]), 0.01, format="%.3f", key=f"t_{model}_lr",
                               help="shrinkage of every tree; lower = more rounds, usually a little better")
    else:
        epochs = c[0].number_input("Max epochs", 1, 5000, int(d["epochs"]), 10, key=f"t_{model}_epochs")
        patience = c[1].number_input("Early-stopping patience", 1, 500, int(d["patience"]), 1, key=f"t_{model}_patience",
                                     help="A model stops after this many epochs without a better validation loss")
        batch = c[2].number_input("Batch size", 1, 4096, int(d["batch"]), 1, key=f"t_{model}_batch")
        lr = c[3].number_input("Learning rate", 1e-6, 1e-1, float(d["lr"]), 1e-4, format="%.5f", key=f"t_{model}_lr")
    folds = int(c[4].number_input("Folds (k)", 2, 20, C.N_FOLDS, 1, key=f"t_{model}_folds",
                                  help="k of the k-fold cross-validation (Mixed, and Grouped k-fold). Paper: 10"))
    c2 = st.columns(5)
    repeats = c2[0].number_input(f"Repeats of {folds}-fold", 1, 20, int(d["repeats"]), 1, key=f"t_{model}_repeats",
                                 help="Paper: 10-fold CV repeated 5 times with different splits; ± is the sd between "
                                      "repeats. Leave-one-out folds are not repeated.")
    seed = int(c2[1].number_input("Seed", 0, 10 ** 6, 42, 1, key="t_seed"))
    c3 = st.columns(5)
    if gbm:
        dropout = float(d["dropout"])
    else:
        dropout = c3[0].selectbox("Dropout", C.DROPOUTS, index=C.DROPOUTS.index(float(d["dropout"])),
                                  key=f"t_{model}_dropout", help="Paper: 0.1 — higher = stronger regularization")
    spec = dict(d)
    spec.update(epochs=int(epochs), patience=int(patience), batch=int(batch), lr=float(lr), repeats=int(repeats),
                dropout=float(dropout))
    if gbm:
        spec["gb_leaves"] = gb_leaves
        spec["gb_min_leaf"] = int(c3[0].selectbox("Min. pairs per leaf", C.GB_MIN_LEAF,
                                                  index=C.GB_MIN_LEAF.index(int(d["gb_min_leaf"])),
                                                  key=f"t_{model}_minleaf", help="a leaf needs at least this many "
                                                                                 "training pairs"))
        spec["gb_colsample"] = float(c3[1].selectbox("Features per tree", C.GB_COLSAMPLE,
                                                     index=C.GB_COLSAMPLE.index(float(d["gb_colsample"])),
                                                     key=f"t_{model}_colsample", format_func=lambda v: f"{v:.0%}",
                                                     help="share of the 1,031 features each tree may use"))
        spec["gb_subsample"] = float(c3[2].selectbox("Pairs per tree", C.GB_SUBSAMPLE,
                                                     index=C.GB_SUBSAMPLE.index(float(d["gb_subsample"])),
                                                     key=f"t_{model}_subsample", format_func=lambda v: f"{v:.0%}",
                                                     help="share of the training pairs each tree is fitted on"))
        spec["gb_l2"] = float(c3[3].selectbox("L2 on leaf values", C.GB_L2, index=C.GB_L2.index(float(d["gb_l2"])),
                                              key=f"t_{model}_l2", format_func=lambda v: f"{v:g}",
                                              help="lambda_l2 — shrinks the values of small leaves"))
    unit = D.unit_of(data)["name"]
    t_opts = ["raw", "drug_z"] + (["study_z"] if is_csa else [])
    t_def = "study_z" if multi else ("raw" if cross else d["target"])
    spec["target"] = c2[2].selectbox(                       # each context keeps its own default + key
        "Response target", t_opts, index=t_opts.index(t_def) if t_def in t_opts else 0,
        key=f"t_{model}_target" + ("_x" if cross else "") + ("_m" if multi else ("_c" if is_csa else "")),
        format_func=lambda t: {"raw": f"raw {unit}", "drug_z": "drug-wise z-score",
                               "study_z": "study-wise z (S)"}[t],
        help=f"drug-wise z: the network learns (y − drug mean) / drug sd, both fitted on each training fold. "
             f"study-wise z (S): (y − study mean) / study sd, so studies with different AAC scales can be learned "
             f"together; a study the model never saw gets the average source study's scale. Predictions are "
             f"converted back to {unit} before any metric. Paper: raw.")
    if model == "proposed":
        with c2[3]:
            st.markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
            spec["interaction"] = st.toggle("Interaction module", value=True, key=f"t_{model}_inter",
                                            help="off = ablation: the same model without the drug × cell pathway module")
    if not gbm:
        spec["weight_decay"] = _wd_input(c3[1], model, d)
    if model == "graph":
        g = st.columns(5)
        spec["g_mode"] = g[0].selectbox("Readout", C.GRAPH_MODES, index=C.GRAPH_MODES.index(d["g_mode"]),
                                        key=f"t_{model}_gmode",
                                        format_func=lambda m: "hybrid" if m == "hybrid" else "pure graph",
                                        help="hybrid: the PathDSP network on [features ; graph summary]. pure: a small head "
                                             "on the drug-conditioned graph summary only (far fewer weights).")
        spec["g_dim"] = int(g[1].selectbox("Node size", C.GRAPH_DIMS, index=C.GRAPH_DIMS.index(int(d["g_dim"])),
                                           key=f"t_{model}_gdim", help="size of each pathway node's vector"))
        spec["g_layers"] = int(g[2].selectbox("Message-passing layers", C.GRAPH_LAYERS,
                                              index=C.GRAPH_LAYERS.index(int(d["g_layers"])), key=f"t_{model}_glay",
                                              help="rounds in which each pathway takes in its neighbours"))
        spec["g_tau"] = float(g[3].selectbox("Edge threshold", C.GRAPH_TAUS, index=C.GRAPH_TAUS.index(float(d["g_tau"])),
                                             key=f"t_{model}_gtau", format_func=lambda v: f"Jaccard \u2265 {v:g}",
                                             help="two pathways are linked when this share of their genes overlaps "
                                                  "(0.05: ~34 neighbours per pathway)"))
        spec["g_graph"] = g[4].selectbox("Pathway graph", C.GRAPH_KINDS, index=0, key=f"t_{model}_ggraph",
                                         format_func=lambda k: "PID gene overlap" if k == "pid" else "random (control)",
                                         help="random = control experiment: the same number of links per pathway and "
                                              "the same link weights, but rewired at random (degree-preserving). If "
                                              "the real graph is not better than this, the pathway structure is not "
                                              "what helps.")
    if gbm:
        st.caption("Upgrades A–M are training options of the neural networks; LightGBM does not use them. Its "
                   "training loss is the MSE and its early stopping the RMSE on the whole validation set (what F and G "
                   "give the networks).")
    else:
        _upgrades(model, spec, d, multi, is_csa)
    T.chips(U.settings_chips(spec, folds, unit), "dim")
    if st.button("Reset to the model's defaults", key="t_reset", type="tertiary"):
        _reset(model); st.rerun()

    # ------------------------------------------------------------------ 05 performance
    if gbm:
        T.step(5, "Performance", "LightGBM trains on the CPU")
        from core import gbm as G
        c3 = st.columns([3, 2])
        c3[0].markdown(f'<div class="nt-note">One model at a time on all {G.threads()} physical CPU cores (the GPU is '
                       "not used). Large CSA datasets are binned from row batches, so memory stays well below the size "
                       "of the feature matrix.</div>", unsafe_allow_html=True)
        max_perf = c3[1].toggle("Max performance", value=True, key="t_perf_gbm",
                                help="gives the training process high CPU priority")
        parallel = "auto"
    else:
        max_perf, parallel = _gpu_performance(spec)

    cfg = {"model": model, "data": data, "schemes": schemes, "grouping": grouping, "folds": folds, "seed": seed,
           "final": bool(final), "parallel": parallel, "perf": "max" if max_perf else "balanced", "spec": spec,
           "cross": bool(cross),
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    _run_section(model, cfg, ds, schemes, grouping, folds, final, spec, act)


def _gpu_performance(spec):
    T.step(5, "Performance", "how hard the GPU is pushed")
    c3 = st.columns([2.2, 1.2, 3])
    par_mode = c3[0].segmented_control("Fold-models trained at once", ["auto", "manual"], default="auto",
                                       key="t_par_mode", required=True,
                                       format_func=lambda m: "Auto (fill free VRAM)" if m == "auto" else "Manual") or "auto"
    parallel = "auto"
    if par_mode == "manual":
        parallel = int(c3[1].number_input("Models at once", 1, 512, 10, 1, key="t_par_n"))
    with c3[2]:
        st.markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
        max_perf = st.toggle("Max performance", value=True, key="t_perf",
                             help="Uses 95 % of the free GPU memory for fold-models (else 85 %) and gives the training "
                                  "process high CPU priority. Freed GPU slots are always refilled with waiting models.")
    spec["fused"] = st.toggle("Fused optimizer kernel (Triton) when available", value=True, key="t_fused",
                              help="One GPU kernel per weight tensor for the optimizer step instead of ~10; used only "
                                   "if Triton is installed and the kernel passes its self-test against PyTorch.")
    _triton_panel()
    return max_perf, parallel


def _run_section(model, cfg, ds, schemes, grouping, folds, final, spec, act):
    err = validate(cfg)
    if not err and spec.get("interaction") and len(ds.interaction_pairs[0]) == 0:
        err = "This dataset has no DG_ columns matching its EXP_/MUT_/CNV_ pathways — switch the interaction module off."
    if not err and model == "graph":
        from core import pathways as PW
        err = PW.check(ds.features, spec.get("g_tau", 0.05))
    if not err and spec.get("fac") and min(len(a) for a in N.fac_index(ds.features)) == 0:
        err = "The factorization term (A) needs both drug (CHEM_ / DG_) and cell (EXP_ / MUT_ / CNV_) features."
    n_models = _n_models(ds, schemes, grouping, int(spec["repeats"]), folds, final, spec.get("ensemble", 1))

    # ------------------------------------------------------------------ 06 run
    T.step(6, "Run", f"{n_models:,} models to train")
    bkey = hashlib.md5(json.dumps({k: v for k, v in cfg.items() if k != "created"}, sort_keys=True, default=str)
                       .encode()).hexdigest()
    bench = st.session_state.get("t_bench", {}).get(bkey)
    gbm = model == C.GBM_MODEL
    word = "rounds" if gbm else "epochs"
    if bench:
        dv = bench["device"]
        T.kpis([
            {"label": "Expected time", "value": U.hms(bench["expected_s"]), "sub": bench["epochs_basis"], "hl": True},
            {"label": "Range", "value": f'{U.hms(bench["min_s"])}',
             "sub": f'fastest possible · worst case {U.hms(bench["max_s"])} (all {spec["epochs"]} {word})'},
            {"label": "Models at once", "value": str(bench["parallel"]),
             "sub": f'{bench["jobs"]} models · one after another on the CPU' if gbm else
                    f'{bench["jobs"]} models · freed slots are refilled'},
            {"label": "Round time" if bench.get("unit") == "round" else "Step time",
             "value": f'{list(bench["step_ms"].values())[0]:.1f} ms',
             "sub": f'{dv.get("name", "?")} · {bench.get("optimizer", "")}'},
        ])
        st.write("")
    if err:
        T.banner(T.esc(err))
    b1, b2, _ = st.columns([1.3, 1.3, 3])
    if b1.button("Estimate time", key="t_bench_btn", width="stretch", disabled=bool(err or act),
                 help="LightGBM: bins the largest fold and times 30 boosting rounds (large CSA data ≈ 1–2 min)"
                      if model == C.GBM_MODEL else
                      "Measures the real step time on your GPU (≈ 15 s) and simulates the run"):
        with st.spinner("Measuring on the CPU…" if gbm else "Measuring on the GPU…"):
            try:
                st.session_state.setdefault("t_bench", {})[bkey] = U.benchmark(cfg)
                st.rerun()
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
    if b2.button("Start training", key="t_start", type="primary", width="stretch", disabled=bool(err or act)):
        rid = U.start_run(cfg)
        st.session_state["live_run"] = rid
        st.switch_page(st.session_state["_pages"]["live"])
    st.caption("Training runs in its own process — closing the browser does not stop it.")
