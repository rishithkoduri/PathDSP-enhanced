"""LIVE — the training process, refreshed every second."""
import time

import pandas as pd
import streamlit as st

from core import config as C
from core import data as D
from core import store as S

from . import charts as CH
from . import common as U
from . import theme as T

LIVE = ("starting", "loading", "running", "explaining")


def _pick_run():
    runs = S.list_runs()
    if not runs:
        return None, runs
    act = next((r for r in runs if r["state"] in LIVE), None)
    ids = [r["id"] for r in runs]
    want = st.session_state.get("live_run")
    if act and (want is None or want not in ids or S.live_state(S.status(want)) not in LIVE):
        want = act["id"]
    if want not in ids:
        want = ids[0]
    lab = {r["id"]: U.run_label(r) for r in runs}
    rid = st.selectbox("Run", ids, index=ids.index(want), format_func=lambda i: lab[i], key="live_select",
                       label_visibility="collapsed")
    st.session_state["live_run"] = rid
    return rid, runs


def _status_text(m, pat):
    if m.get("stopping"):
        return "stopping — best epoch restored"
    if m.get("improved"):
        return "improved"
    return f"no improvement ×{m.get('wait', 0)}"


def _model_table(models, pat, show_lr=False, unit="epoch"):
    df = pd.DataFrame([{"model": m["label"], "epoch": m["ep"], "train": m["train"], "val": m["val"], "best val": m["best"],
                        "best ep": m["best_ep"], "no improvement": m["wait"], "status": _status_text(m, pat),
                        **({"lr": m.get("lr")} if show_lr else {}), "time": U.hms(m.get("seconds"))} for m in models])
    st.dataframe(df, hide_index=True, height=min(460, 38 + 35 * len(df)), width="stretch",
                 column_config={
                     "lr": st.column_config.NumberColumn("learning rate", format="%.2e", width="small",
                                                         help="C · halved after the LR patience without a significant improvement"),
                     "epoch": st.column_config.NumberColumn(unit, width="small"),
                     "train": st.column_config.NumberColumn("train loss", format="%.4f"),
                     "val": st.column_config.NumberColumn("val loss", format="%.4f"),
                     "best val": st.column_config.NumberColumn("best val", format="%.4f"),
                     "best ep": st.column_config.NumberColumn("best ep", width="small"),
                     "no improvement": st.column_config.ProgressColumn(
                         f"no improvement (of {pat})", format=f"%d / {pat}", min_value=0, max_value=pat),
                     "status": st.column_config.TextColumn("status"),
                     "time": st.column_config.TextColumn("running", width="small")})


def _finished_table(rid, st_):
    f = U.run_csv(rid, "folds.csv")
    rows = []
    if not f.empty:
        for _, r in f.iloc[::-1].iterrows():
            g = r.get("group")
            rows.append({"model": f"{r['scheme']} r{int(r['repeat'])} f{int(r['fold'])}" + (f" · {g}" if isinstance(g, str) and g else ""),
                         "test RMSE": U.f3(r["RMSE"], 4), "test PCC": U.f3(r["PCC"], 4), "best ep": int(r["best_epoch"]),
                         "epochs": int(r["epochs_run"]), "time": U.hms(r["seconds"])})
    for r in (st_.get("recent") or []):
        if r.get("model") == "all-data model":
            rows.insert(0, {"model": "all-data model", "test RMSE": "— (no test set)", "test PCC": "—",
                            "best ep": r.get("best_epoch"), "epochs": r.get("epochs"), "time": U.hms(r.get("seconds"))})
    if not rows:
        st.caption("Finished models appear here the moment they stop.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, height=min(320, 38 + 35 * len(df)), width="stretch")


def render():
    rid, runs = _pick_run()
    if rid is None:
        T.banner("No runs yet — start one on <b>Train</b>.")
        return

    @st.fragment(run_every=1)
    def live():
        st_ = S.status(rid)
        state = S.live_state(st_)
        cfg = S.config(rid)
        ms = C.MODEL_SPECS.get(cfg.get("model"), {})
        p = st_.get("pool") or {}
        eta = st_.get("eta") or {}
        n, d = int(st_.get("n_jobs") or 0), int(st_.get("done_jobs") or 0)
        pat = int(p.get("patience") or cfg.get("spec", {}).get("patience", 30))
        up_tag = C.upgrades_tag(cfg.get("spec") or {})
        gbm = cfg.get("model") == C.GBM_MODEL                         # (v5.9) LightGBM: rounds on the CPU
        word = "round" if gbm else "epoch"
        # ---------- header + controls
        h1, h2, h3 = st.columns([6, 1, 1], vertical_alignment="center")
        with h1:
            st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">'
                        f'{T.state_chip(state)}{T.chip(ms.get("name", "?"), "red")}'
                        f'{T.chip("upgrades " + up_tag, "red") if up_tag else ""}'
                        f'{T.chip(D.dataset_name(cfg.get("data", "?")))}'
                        f'{T.chip(" · ".join(C.SCHEMES[s]["name"] for s in cfg.get("schemes", [])) or "all-data model")}'
                        f'{T.chip(st_.get("message", ""), "dim")}'
                        f'{T.chip("optimizer: " + st_["optimizer"], "dim") if st_.get("optimizer") else ""}</div>',
                        unsafe_allow_html=True)
        with h2:
            if state in LIVE:
                stopping = (S.run_dir(rid) / C.STOP_FILE).exists()
                if st.button("Stopping…" if stopping else "Stop", key="live_stop", width="stretch", disabled=stopping,
                             help="Stops within a second; finished models are kept and the run can be resumed"):
                    U.stop_run(rid)
            elif state in ("stopped", "interrupted", "failed"):
                if st.button("Resume", key="live_resume", type="primary", width="stretch",
                             disabled=S.active_run() is not None):
                    U.resume_run(rid)
        with h3:
            if st.button("Results", key="live_results", width="stretch"):
                st.session_state["res_run"] = rid
                st.switch_page(st.session_state["_pages"]["results"])
        st.write("")
        # ---------- KPIs
        elapsed = (st_.get("prior_seconds", 0) + time.time() - st_.get("started", time.time())) if state in LIVE \
            else st_.get("elapsed")
        gi = st_.get("gpu") or {}
        proc = gi.get("proc_mb", gi.get("torch_mb"))
        running = state == "running"
        T.kpis([
            {"label": "Elapsed", "value": U.hms(elapsed),
             "sub": "wall clock" + (" · incl. earlier sessions" if st_.get("prior_seconds") else "")},
            {"label": "Time left", "value": U.hms(eta.get("expected")) if running and eta else ("—" if state in LIVE else "0s"),
             "sub": (f'range {U.hms(eta.get("min"))} – {U.hms(eta.get("max"))} · {eta.get("basis", "")}' if running and eta
                     else ((f"measuring the first {word}…" if not gbm else "binning the data / first rounds…")
                           if state in LIVE else "")), "hl": running},
            {"label": "Models done", "value": f"{d}/{n}" if n else "—",
             "sub": (f'{p.get("queued", 0)} waiting' + ("" if gbm else " for a GPU slot")) if running
             else "finished fold-models"},
            {"label": "Rounds" if gbm else "Epochs",
             "value": (f'{p["epoch_min"]}–{p["epoch_max"]}' if p.get("epoch_min") is not None and
                       p["epoch_min"] != p["epoch_max"] else str(p.get("epoch_max", "—"))) if running else "—",
             "sub": (f'boosting rounds of the model · max {cfg.get("spec", {}).get("epochs", "—")}' if gbm else
                     f'of the models on the GPU · max {cfg.get("spec", {}).get("epochs", "—")}')},
            {"label": "On the CPU" if gbm else "On the GPU",
             "value": f'{p.get("active", "—")}/{p.get("slots", st_.get("parallel", "—"))}' if running else "—",
             "sub": "LightGBM model training / at once" if gbm else "models training / GPU slots"},
            {"label": "Speed", "value": f'{p["samples_per_s"] / 1000:.1f}k/s' if p.get("samples_per_s") else "—",
             "sub": (f'{p.get("step_ms", 0):.1f} ms per round' if gbm else
                     f'{p.get("step_ms", 0):.1f} ms per step · {p.get("round_seconds", 0):.0f} s per epoch')
             if p.get("step_ms") else "training samples per second"},
            {"label": "GPU memory", "value": f'{proc / 1024:.2f} GB' if gi.get("cuda") and proc is not None
             else (gi.get("name") or "—"),
             "sub": (f'held by this training run (PyTorch) · of {gi.get("total_mb", 0) / 1024:.1f} GB'
                     if gi.get("cuda") else "training device")},
        ])
        st.write("")
        T.progress("All models", d / n if n else 0, f"{d}/{n}")
        if running and p.get("steps"):
            T.progress(f'Current epoch — {p.get("active", 0)} models', p.get("step", 0) / p["steps"],
                       f'step {p.get("step", 0):,}/{p["steps"]:,} · loss {p.get("train_loss_now", 0):.4f}')
        if state == "loading":
            T.progress("Loading data", st_.get("load_progress", 0), st_.get("message", ""))
        # ---------- curve + results so far
        c1, c2 = st.columns([1.5, 1])
        with c1:
            curve = st_.get("curve") or []
            last = curve[-1] if curve else None
            st.markdown(f'<div class="nt-label">Mean loss per {word} over all models</div>'
                        + (f'<div class="nt-note" style="margin-top:2px">{word} {last[0]} · train {last[1]:.4f} · '
                           f'val {last[2]:.4f} · {last[3]} models</div>' if last else ""), unsafe_allow_html=True)
            ch = CH.live_curve(curve)
            if ch is not None:
                CH.show(ch, 260)
            else:
                st.caption(f"The curve appears after the first {word}.")
        with c2:
            st.markdown('<div class="nt-label">Results so far — finished held-out folds</div>', unsafe_allow_html=True)
            sums = U.scheme_summaries(rid)
            if sums:
                items = []
                for s, m in sums.items():
                    v, sd = U.pm(m, "RMSE")
                    items.append({"label": f'{C.SCHEMES[s]["name"]} RMSE', "value": v,
                                  "sub": f'{sd} · {m["n_folds"]} folds · per-drug PCC {U.f3(m.get("per_drug_PCC"))}'})
                T.kpis(items)
            else:
                st.caption("Metrics appear when the first fold-model finishes.")
        # ---------- every model on the GPU
        models = st_.get("models") or []
        if running and models:
            st.markdown('<div class="nt-label" style="margin-top:10px">'
                        + (f'Model on the CPU — status after its last round · it stops after {pat} rounds without a '
                           f'lower validation RMSE' if gbm else
                           f'Models on the GPU — status after their last epoch · a model stops after {pat} epochs '
                           f'without improvement') + '</div>', unsafe_allow_html=True)
            _model_table(models, pat, show_lr=bool(cfg.get("spec", {}).get("plateau")), unit=word)
        st.markdown('<div class="nt-label" style="margin-top:10px">Finished models</div>', unsafe_allow_html=True)
        _finished_table(rid, st_)
        if state == "failed":
            st.error(st_.get("message", "failed"))
            with st.expander("Error details"):
                st.code(st_.get("error", ""), language="text")
        with st.expander("Console log"):
            st.code(U.log_tail(rid) or "(empty)", language="text")

    live()
