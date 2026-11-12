"""Altair charts in the Nothing style (black, white, greys, one red accent; mono labels)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from . import theme as T
from .theme import RED


def _p(k):
    """a colour of the active theme (light / dark)"""
    return T.pal()[k]

alt.data_transformers.disable_max_rows()
FONT = "Space Mono, ui-monospace, Consolas, monospace"


def _cfg(ch, height=300):
    return (ch.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=_p("mute"), titleColor=_p("mute"), gridColor=_p("c_grid"), gridDash=[2, 4], domainColor=_p("c_domain"),
                            tickColor=_p("c_domain"), labelFont=FONT, titleFont=FONT, labelFontSize=10, titleFontSize=10,
                            titleFontWeight="normal")
            .configure_legend(labelColor=_p("c_legend"), titleColor=_p("mute"), labelFont=FONT, titleFont=FONT, labelFontSize=10,
                              titleFontSize=10, orient="top", symbolType="circle")
            .configure_title(color=_p("ink"), font=FONT, fontSize=11, anchor="start", fontWeight="normal"))


def show(ch, height=300):
    st.altair_chart(_cfg(ch, height), width="stretch", theme=None)


def scatter_density(actual, pred, unit="−log10 IC50", bins=56):
    """predicted vs measured as a dot matrix: dot size = number of pairs in the cell"""
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    lo = float(np.nanmin([a.min(), p.min()])); hi = float(np.nanmax([a.max(), p.max()]))
    pad = (hi - lo) * 0.03; lo -= pad; hi += pad
    w = (hi - lo) / bins
    df = pd.DataFrame({"x": np.floor((a - lo) / w), "y": np.floor((p - lo) / w)}).value_counts().reset_index(name="n")
    df["measured"] = lo + (df["x"] + 0.5) * w
    df["predicted"] = lo + (df["y"] + 0.5) * w
    scale = alt.Scale(domain=[lo, hi], nice=False)
    pts = alt.Chart(df).mark_circle(color=_p("ink")).encode(
        x=alt.X("measured:Q", scale=scale, title=f"measured {unit}"),
        y=alt.Y("predicted:Q", scale=scale, title=f"predicted {unit}"),
        size=alt.Size("n:Q", scale=alt.Scale(type="sqrt", range=[4, 70]), legend=None),
        opacity=alt.Opacity("n:Q", scale=alt.Scale(type="log", range=[0.35, 1]), legend=None),
        tooltip=[alt.Tooltip("measured:Q", format=".2f"), alt.Tooltip("predicted:Q", format=".2f"), "n:Q"])
    diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color=RED, strokeDash=[4, 4], strokeWidth=1.2) \
        .encode(x="x:Q", y="y:Q")
    return diag + pts


def loss_curves(curves, max_models=400):
    """validation loss of every fold-model (thin grey) + mean train / val (white / red)"""
    c = curves.copy()
    c["model"] = c["scheme"].astype(str) + " r" + c["repeat"].astype(str) + " f" + c["fold"].astype(str)
    keep = c["model"].drop_duplicates().iloc[:max_models]
    c = c[c["model"].isin(set(keep))]
    thin = alt.Chart(c).mark_line(strokeWidth=0.6, opacity=0.25, color=_p("dim")).encode(
        x=alt.X("epoch:Q", title="epoch", axis=alt.Axis(format="d", tickMinStep=1)), y=alt.Y("val:Q", title="loss (RMSE)", scale=alt.Scale(zero=False)),
        detail="model:N")
    m = c.groupby("epoch", as_index=False).agg(train=("train", "mean"), val=("val", "mean"), n=("val", "size"))
    m = m[m["n"] >= max(1, int(0.2 * c["model"].nunique()))]
    mm = m.melt(id_vars=["epoch"], value_vars=["train", "val"], var_name="mean", value_name="loss")
    mean = alt.Chart(mm).mark_line(strokeWidth=2).encode(
        x="epoch:Q", y="loss:Q",
        color=alt.Color("mean:N", scale=alt.Scale(domain=["train", "val"], range=[_p("ink"), RED]), title=None),
        tooltip=["epoch:Q", "mean:N", alt.Tooltip("loss:Q", format=".4f")])
    return thin + mean


def live_curve(points):
    """points: [[epoch, mean train, mean val, models], ...] — every model counted at its own epoch number"""
    if not points:
        return None
    df = pd.DataFrame(points, columns=["epoch", "train", "val", "models"])
    mm = df.melt(id_vars=["epoch"], value_vars=["train", "val"], var_name="loss", value_name="value")
    return alt.Chart(mm).mark_line(strokeWidth=2, point=len(df) <= 40).encode(
        x=alt.X("epoch:Q", title="epoch (each model's own)", axis=alt.Axis(format="d", tickMinStep=1)),
        y=alt.Y("value:Q", title="mean loss (RMSE)", scale=alt.Scale(zero=False)),
        color=alt.Color("loss:N", scale=alt.Scale(domain=["train", "val"], range=[_p("ink"), RED]), title=None))


def fold_strip(folds, metric="RMSE"):
    f = folds.reset_index(drop=True).copy()
    f["#"] = np.arange(1, len(f) + 1)
    f["label"] = (f["scheme"].astype(str) + " r" + f["repeat"].astype(str) + " f" + f["fold"].astype(str)
                  + f["group"].fillna("").astype(str).map(lambda g: f" · {g}" if g else ""))
    pts = alt.Chart(f).mark_circle(size=34, color=_p("ink"), opacity=0.9).encode(
        x=alt.X("#:Q", title="fold-model", axis=alt.Axis(format="d", tickMinStep=1)), y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        tooltip=["label:N", alt.Tooltip(f"{metric}:Q", format=".4f"), "best_epoch:Q", "epochs_run:Q",
                 alt.Tooltip("seconds:Q", format=".0f")])
    rule = alt.Chart(pd.DataFrame({"m": [float(f[metric].mean())]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(y="m:Q")
    return pts + rule


def blocks_bar(blocks):
    b = blocks.copy()
    return alt.Chart(b).mark_bar(cornerRadiusEnd=6, height=16).encode(
        y=alt.Y("block:N", sort="-x", title=None),
        x=alt.X("share:Q", title="share of total |SHAP| (%)"),
        color=alt.Color("block:N", scale=alt.Scale(domain=list(_p("blocks")), range=list(_p("blocks").values())), legend=None),
        tooltip=["block:N", alt.Tooltip("share:Q", format=".1f")])


def pathways_bar(pathways, top=15):
    p = pathways.head(top).copy()
    p["pathway"] = p["pathway"].str.replace("PID_", "", regex=False).str.replace("_PATHWAY", "", regex=False)
    order = p["pathway"].tolist()
    m = p.melt(id_vars=["pathway"], value_vars=["DG", "EXP", "MUT", "CNV"], var_name="block", value_name="mean |SHAP|")
    return alt.Chart(m).mark_bar(height=13).encode(
        y=alt.Y("pathway:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("mean |SHAP|:Q", stack="zero"),
        color=alt.Color("block:N", scale=alt.Scale(domain=["DG", "EXP", "MUT", "CNV"],
                                                     range=[_p("blocks")[k] for k in ("DG", "EXP", "MUT", "CNV")]),
                        title=None),
        tooltip=["pathway:N", "block:N", alt.Tooltip("mean |SHAP|:Q", format=".4f")])


def attention_bar(att, top=15):
    a = att.head(top).copy()
    a["pathway"] = a["pathway"].str.replace("PID_", "", regex=False).str.replace("_PATHWAY", "", regex=False)
    order = a["pathway"].tolist()
    bars = alt.Chart(a).mark_bar(height=13, color=_p("ink")).encode(
        y=alt.Y("pathway:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("mean_attention:Q", title="mean attention weight"),
        tooltip=["pathway:N", alt.Tooltip("mean_attention:Q", format=".4f"), alt.Tooltip("top5_share:Q", format=".2f"),
                 "most_by_drug:N"])
    rule = alt.Chart(pd.DataFrame({"u": [float(att["uniform"].iloc[0])]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(x="u:Q")
    return bars + rule


def drug_strip(values, cells, pred, measured=None, labels=None, unit="−log10 IC50"):
    """every measured response of the drug (grey dots) with the prediction (red) and the measurement (white)"""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"response": values, "cell": [labels.get(c, c) if labels else c for c in cells],
                       "j": rng.uniform(-1, 1, len(values))})
    pts = alt.Chart(df).mark_circle(size=26, color=_p("c_point"), opacity=0.75).encode(
        x=alt.X("response:Q", title=f"{unit}   ← resistant · sensitive →", scale=alt.Scale(zero=False)),
        y=alt.Y("j:Q", axis=None, scale=alt.Scale(domain=[-2.2, 2.2])),
        tooltip=["cell:N", alt.Tooltip("response:Q", format=".3f")])
    marks = [{"what": "prediction", "x": float(pred)}]
    if measured is not None and np.isfinite(measured):
        marks.append({"what": "measured", "x": float(measured)})
    rules = alt.Chart(pd.DataFrame(marks)).mark_rule(strokeWidth=2.5).encode(
        x="x:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["prediction", "measured"], range=[RED, _p("ink")]), title=None),
        tooltip=["what:N", alt.Tooltip("x:Q", format=".3f")])
    return pts + rules


def matrix_heat(G, fmt=".3f", higher_better=True, title_x="tested on (target study)", title_y="trained on (source)"):
    """5 × 5 cross-study matrix as a heatmap with values (darker red = better), NaN cells blank"""
    df = G.copy()
    df.index.name = "source"
    long = df.reset_index().melt(id_vars="source", var_name="target", value_name="v").dropna()
    order = list(G.columns)
    base = alt.Chart(long).encode(x=alt.X("target:N", sort=order, title=title_x, axis=alt.Axis(labelAngle=0)),
                                  y=alt.Y("source:N", sort=list(G.index), title=title_y))
    rect = base.mark_rect(stroke=_p("c_stroke"), strokeWidth=2).encode(
        color=alt.Color("v:Q", legend=None, scale=alt.Scale(range=[_p("c_low"), RED] if higher_better else [RED, _p("c_low")])),
        tooltip=["source:N", "target:N", alt.Tooltip("v:Q", format=fmt)])
    text = base.mark_text(font=FONT, fontSize=12, color=_p("ink")).encode(text=alt.Text("v:Q", format=fmt))
    return rect + text


def fewshot_lines(fs, metric="R2"):
    """few-shot recalibration: metric vs k (log axis, zero-shot drawn at k = 1) per held-out study"""
    d = fs.copy()
    d["k_plot"] = d["k"].where(d["k"] > 0, 1)
    d["label"] = d["k"].map(lambda k: "zero-shot" if k == 0 else str(k))
    pal = _p("study")
    return alt.Chart(d).mark_line(point=True, strokeWidth=1.6).encode(
        x=alt.X("k_plot:Q", scale=alt.Scale(type="log"), title="labelled pairs of the new study (k; 1 = zero-shot)"),
        y=alt.Y(f"{metric}:Q", title={"R2": "R²", "coverage": "coverage of the 90 % interval",
                                       "half_width": "interval half-width (±)"}.get(metric, metric),
                scale=alt.Scale(zero=False)),
        color=alt.Color("study:N", scale=alt.Scale(range=pal), title=None),
        tooltip=["study:N", "label:N", alt.Tooltip(f"{metric}:Q", format=".3f")])
