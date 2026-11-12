"""
PathDSP-X — Explainable drug sensitivity prediction through cancer pathway enrichment.
Start with:   streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="PathDSP-X", page_icon="●", layout="wide", initial_sidebar_state="collapsed")

from core import config as C  # noqa: E402
from core import store as S  # noqa: E402
from ui import common as U  # noqa: E402
from ui import page_live, page_predict, page_results, page_train  # noqa: E402
from ui import theme as T  # noqa: E402

T.init_mode()          # Light / Dark: from the header switch, or ?theme=light in the URL
T.inject()


@st.fragment(run_every=2)
def brand():
    act = S.active_run()
    chips = []
    if act:
        stt = act["status"]
        n, d = stt.get("n_jobs") or 0, stt.get("done_jobs") or 0
        pct = f" · {100 * d / n:.0f}%" if n else ""
        chips.append(f'<span class="nt-chip red"><span class="nt-dot live"></span>TRAINING · '
                     f'{T.esc(U.model_tag(act["config"]))}{pct}</span>')
    chips.append(U.gpu_chip())
    gstate, gtext = U.gpu_state(act)
    dot = {"in use": "live", "idle": "white"}.get(gstate, "")
    st.markdown(f'<div class="nt-brand"><div><div class="nt-logo">PATHDSP<b>-X</b></div>'
                f'<div class="nt-sub">{T.esc(C.APP_TITLE)} · Team G1589</div></div>'
                f'<div class="nt-right"><div class="nt-chips">{"".join(chips)}</div>'
                f'<div class="nt-gpu-state {gstate.replace(" ", "-")}"><span class="nt-dot {dot}"></span>'
                f'{T.esc(gtext)}</div></div></div>', unsafe_allow_html=True)


head, switch = st.columns([10, 1.3], vertical_alignment="top")
with head:
    brand()
with switch:
    T.mode_switch()

pages = [
    st.Page(page_train.render, title="Train", icon=":material/tune:", url_path="train", default=True),
    st.Page(page_live.render, title="Live", icon=":material/monitoring:", url_path="live"),
    st.Page(page_results.render, title="Results", icon=":material/insights:", url_path="results"),
    st.Page(page_predict.render, title="Predict", icon=":material/biotech:", url_path="predict"),
]
st.session_state["_pages"] = {p.url_path: p for p in pages}
st.navigation(pages, position="top").run()
