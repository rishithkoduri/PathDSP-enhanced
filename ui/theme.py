"""Nothing-OS styling: black, white, one red accent, dot-matrix display type (Doto), Space Grotesk / Space Mono."""
import html

import streamlit as st

RED = "#D71921"

# two palettes; every colour in the CSS and the charts comes from the active one (Light / Dark switch in the header)
PALETTES = {
    "dark": dict(bg="#000000", grid_dot="#151515", header="rgba(0,0,0,.85)", ink="#F2F2F2", strong="#FFFFFF",
                 mute="#8C8C8C", dim="#4A4A4A", line="#1F1F1F", card="#0B0B0B", chip_bg="#080808", chip_line="#2A2A2A",
                 chip_ink="#D6D6D6", input="#0B0B0B", code="#161616", dot_off="#222222", banner="#0A0A0A",
                 banner_ink="#E6E6E6", panel="#060606", btn2="#000000", btn2_line="#3A3A3A", soft="#DDDDDD",
                 c_grid="#1B1B1B", c_domain="#2C2C2C", c_legend="#CFCFCF", c_point="#6A6A6A", c_low="#1A1A1A",
                 c_stroke="#000000", study=["#FFFFFF", RED, "#9A9A9A", "#FF8A8A", "#5A5A5A"],
                 blocks={"CHEM": "#F2F2F2", "DG": RED, "EXP": "#BDBDBD", "MUT": "#7A7A7A", "CNV": "#454545"}),
    "light": dict(bg="#F5F5F3", grid_dot="#DCDCD8", header="rgba(245,245,243,.88)", ink="#141414", strong="#000000",
                  mute="#6B6B6B", dim="#B5B5B0", line="#DCDCD8", card="#FFFFFF", chip_bg="#FFFFFF", chip_line="#CFCFCA",
                  chip_ink="#2A2A2A", input="#FFFFFF", code="#ECECE8", dot_off="#D6D6D2", banner="#FFFFFF",
                  banner_ink="#1A1A1A", panel="#FFFFFF", btn2="#FFFFFF", btn2_line="#BDBDB8", soft="#333333",
                  c_grid="#E6E6E2", c_domain="#C9C9C4", c_legend="#333333", c_point="#9A9A96", c_low="#EDEDE9",
                  c_stroke="#FFFFFF", study=["#141414", RED, "#8A8A8A", "#F08080", "#4A4A4A"],
                  blocks={"CHEM": "#141414", "DG": RED, "EXP": "#6E6E6E", "MUT": "#A3A3A3", "CNV": "#CFCFCA"}),
}


def mode():
    """'light' or 'dark' — the header switch (st.session_state['ui_light']); dark by default"""
    return "light" if st.session_state.get("ui_light") else "dark"


def pal():
    return PALETTES[mode()]


# kept for imports that expect the old names (dark values)
WHITE, MUTE, DIM, LINE, CARD = "#F2F2F2", "#8C8C8C", "#4A4A4A", "#1F1F1F", "#0B0B0B"
BLOCK_COLORS = PALETTES["dark"]["blocks"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Doto:wght@600;800;900&family=Space+Grotesk:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');
:root { --red:__red__; --white:__ink__; --ink:__ink__; --strong:__strong__; --mute:__mute__; --line:__line__; --card:__card__;
  --chip-bg:__chip_bg__; --chip-line:__chip_line__; --chip-ink:__chip_ink__; --input:__input__; --code:__code__;
  --dot-off:__dot_off__; --banner:__banner__; --banner-ink:__banner_ink__; --panel:__panel__; --btn2:__btn2__;
  --btn2-line:__btn2_line__; --soft:__soft__; }
.stApp { background-color:__bg__; color:var(--ink);
  background-image: radial-gradient(__grid_dot__ 1px, transparent 1.3px); background-size: 18px 18px; }
header[data-testid="stHeader"] { background: __header__; backdrop-filter: blur(6px); }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1320px; }
html, body, [class*="css"], .stMarkdown, p, label, input, textarea, button, select { font-family: 'Space Grotesk', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Doto', 'Space Mono', monospace !important; text-transform: uppercase; letter-spacing: .06em; font-weight: 900 !important; color:var(--strong); }
code, pre, .stCode { font-family: 'Space Mono', monospace !important; }
p code, .nt-note code, li code { color:var(--ink) !important; background:var(--code) !important; border-radius:6px; padding:1px 5px; }
hr { border-color: var(--line) !important; }

/* ---------- brand bar ---------- */
.nt-brand { display:flex; align-items:flex-end; justify-content:space-between; gap:18px; flex-wrap:nowrap;
  border-bottom:1px solid var(--line); padding: 4px 2px 14px 2px; margin-bottom: 18px; }
.nt-brand > div:first-child { min-width:0; }
.nt-right { flex-shrink:0; }
@media (max-width: 1100px) { .nt-brand { flex-wrap:wrap; } .nt-right { align-items:flex-start !important; } }
.nt-logo { font-family:'Doto','Space Mono',ui-monospace,Consolas,monospace; font-weight:900; font-size:2.6rem; line-height:.9; letter-spacing:.08em; color:var(--strong); }
.nt-logo b { color:var(--red); }
.nt-sub { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; color:var(--mute); font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; margin-top:6px; }
.nt-chips { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.nt-right { display:flex; flex-direction:column; align-items:flex-end; gap:7px; }
.nt-gpu-state { display:flex; align-items:center; gap:7px; font-family:'Space Mono',ui-monospace,Consolas,monospace;
  font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--mute); }
.nt-gpu-state.in-use { color:var(--strong); }
.nt-gpu-state.none { color:var(--red); }

/* ---------- labels, chips, pills ---------- */
.nt-label { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; color:var(--mute); }
.nt-step { display:flex; align-items:baseline; gap:12px; margin: 22px 0 10px 0; }
.nt-step .n { font-family:'Doto','Space Mono',ui-monospace,Consolas,monospace; font-weight:900; font-size:1.55rem; color:var(--red); }
.nt-step .t { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; letter-spacing:.18em; text-transform:uppercase; color:var(--strong); }
.nt-step .h { font-family:'Space Grotesk', system-ui, sans-serif; font-size:.82rem; color:var(--mute); }
.nt-chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--chip-line); border-radius:999px; padding:3px 11px;
  font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; font-size:.7rem; letter-spacing:.06em; color:var(--chip-ink); background:var(--chip-bg); white-space:nowrap; }
.nt-chip.red { border-color: var(--red); color:var(--strong); }
.nt-chip.dim { color: var(--mute); }
.nt-dot { width:7px; height:7px; border-radius:50%; background:var(--mute); display:inline-block; }
.nt-dot.red { background: var(--red); }
.nt-dot.white { background: var(--strong); }
.nt-dot.live { background: var(--red); animation: ntpulse 1.1s ease-in-out infinite; }
@keyframes ntpulse { 0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(215,25,33,.6); } 50% { opacity:.35; box-shadow:0 0 0 5px rgba(215,25,33,0); } }

/* ---------- cards & KPI tiles ---------- */
.nt-card { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:16px 18px; }
.nt-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
.nt-kpi { background:var(--card); border:1px solid var(--line); border-radius:20px; padding:14px 16px 12px 16px; min-height:104px; }
.nt-kpi .v { font-family:'Doto','Space Mono',ui-monospace,Consolas,monospace; font-weight:900; font-size:2.05rem; line-height:1.05; color:var(--strong); margin-top:6px; word-break:break-word; }
.nt-kpi .v.red { color: var(--red); }
.nt-kpi .v.sm { font-size:1.55rem; line-height:1.25; }
.nt-kpi .v.xs { font-size:1.2rem; line-height:1.3; word-break:normal; overflow-wrap:anywhere; }
.nt-kpi .s { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; font-size:.68rem; color:var(--mute); margin-top:6px; letter-spacing:.03em; }
.nt-kpi.hl { border-color: var(--red); }

/* ---------- dot progress ---------- */
.nt-prog { margin: 6px 0 12px 0; }
.nt-prog .row { display:flex; justify-content:space-between; margin-bottom:6px; }
.nt-dots { display:flex; justify-content:space-between; align-items:center; height:10px; }
.nt-dots i { display:block; width:min(7px, 1vw); height:min(7px, 1vw); border-radius:50%; background:var(--dot-off); flex:0 0 auto; }
.nt-dots i.on { background:var(--ink); }
.nt-dots i.head { background:var(--red); }

/* ---------- banners ---------- */
.nt-banner { border:1px solid var(--line); border-left:3px solid var(--red); border-radius:14px; padding:12px 16px;
  background:var(--banner); color:var(--banner-ink); font-size:.92rem; }
.nt-banner.ok { border-left-color:var(--ink); }
.nt-note { color:var(--mute); font-size:.8rem; }
.nt-match { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; font-size:.68rem; letter-spacing:.06em; color:var(--mute); margin:-4px 0 6px 2px; }
.nt-match b { color:var(--red); font-weight:700; }

/* ---------- streamlit widgets ---------- */
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button, div[data-testid="stFormSubmitButton"] button {
  border-radius:999px !important; font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace !important; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem !important; }
div[data-testid="stButton"] button[kind="primary"] { background:var(--red) !important; border-color:var(--red) !important; color:#fff !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { filter:brightness(1.12); }
div[data-testid="stButton"] button[kind="secondary"] { background:var(--btn2) !important; border:1px solid var(--btn2-line) !important; color:var(--ink) !important; }
div[data-testid="stButton"] button[kind="secondary"]:hover { border-color:var(--strong) !important; }
div[data-testid="stButton"] button:disabled { opacity:.32 !important; cursor:not-allowed !important; filter:none !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input { background:var(--input) !important; border-radius:14px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family:'Space Mono',ui-monospace,Menlo,Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; font-size:.76rem; }
div[data-testid="stExpander"] details { border:1px solid var(--line) !important; border-radius:16px !important; background:var(--panel); }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stCaptionContainer"] { color: var(--mute) !important; }
__light__
</style>
"""

# Streamlit's own widgets follow the dark base theme in .streamlit/config.toml; in light mode these rules repaint them
LIGHT_EXTRA = """
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label, .stTabs [data-baseweb="tab"] p, div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p, [data-testid="stCheckbox"] p, .stRadio p, [data-testid="stToggle"] p,
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stTooltipIcon"] svg, [data-testid="stIconMaterial"], header[data-testid="stHeader"] button,
[data-testid="stHeaderActionElements"] a { color: var(--ink) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"], [data-baseweb="textarea"],
.stNumberInput input, .stTextInput input, .stTextArea textarea { background:var(--input) !important; color:var(--ink) !important;
  border-color:var(--chip-line) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] svg, .stNumberInput button { color:var(--ink) !important; }
.stNumberInput button { background:var(--input) !important; }
input::placeholder, textarea::placeholder { color:var(--mute) !important; opacity:1; }
[data-baseweb="popover"] ul, [data-baseweb="menu"], [data-baseweb="popover"] [role="listbox"] { background:var(--card) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"], [data-baseweb="popover"] [role="option"] * { color:var(--ink) !important; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] { background:var(--code) !important; }
[data-baseweb="tag"] { background:var(--code) !important; color:var(--ink) !important; }
[data-baseweb="tag"] span { color:var(--ink) !important; }
[data-testid="stCode"] pre, .stCode pre, pre { background:var(--code) !important; color:var(--ink) !important; }
[data-testid="stCode"] code, [data-testid="stCode"] code * { color:var(--ink) !important; }
[data-testid="stAlert"] { background:var(--card) !important; color:var(--ink) !important; border:1px solid var(--line); }
[data-testid="stAlert"] p { color:var(--ink) !important; }
[data-testid="stProgress"] > div > div { background:var(--dot-off) !important; }
/* Streamlit >= 1.60 draws inputs and selects with react-aria (no data-baseweb) */
body { background:var(--input); }
.stApp div:has(> input:not([type="checkbox"]):not([type="radio"]):not([type="range"])), .stApp div:has(> textarea),
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] { background-color:var(--input) !important;
  border-color:var(--chip-line) !important; }
.stApp input:not([type="checkbox"]):not([type="radio"]), .stApp textarea { color:var(--ink) !important; background:transparent !important; }
[data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg,
[data-testid="stNumberInput"] button, [data-testid="stTextInputIcon"] * { color:var(--ink) !important; }
[data-testid$="VirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] { background:var(--card) !important;
  border:1px solid var(--line) !important; }
[role="listbox"] [role="option"], [role="listbox"] [role="option"] * { color:var(--ink) !important; }
[role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"],
[role="listbox"] [role="option"][data-focused="true"] { background:var(--code) !important; }
div[data-testid="stButton"] button[kind="primary"] p, div[data-testid="stButton"] button[kind="primary"] * { color:#fff !important; }
div[data-testid="stButton"] button[kind="secondary"] p { color:var(--ink) !important; }
.stApp button[data-testid*="secondary"], .stApp [data-testid="stPopover"] button, .stApp [data-testid="stPopoverButton"],
.stApp [data-testid="stDownloadButton"] button { background:var(--btn2) !important; border:1px solid var(--btn2-line) !important;
  color:var(--ink) !important; }
.stApp button[data-testid*="secondary"] *, .stApp [data-testid="stPopover"] button *, .stApp [data-testid="stDownloadButton"] button * { color:var(--ink) !important; }
[data-testid="stPopoverBody"] { background:var(--card) !important; color:var(--ink) !important; }
.stApp [data-testid="stButtonGroup"] button:not([data-selected="true"]):not([aria-checked="true"]):not([kind$="Active"]) {
  background:var(--btn2) !important; color:var(--ink) !important; border-color:var(--btn2-line) !important; }
.stApp [data-testid="stButtonGroup"] button:not([data-selected="true"]):not([aria-checked="true"]):not([kind$="Active"]) * { color:var(--ink) !important; }
/* the data grid is drawn on a canvas in the dark theme's colours: invert it to a light grid (red stays red) */
div[data-testid="stDataFrame"] > div { filter: invert(1) hue-rotate(180deg); }
"""


def css():
    p = pal()
    out = CSS.replace("__light__", LIGHT_EXTRA if mode() == "light" else "").replace("__red__", RED)
    for k, v in p.items():
        if isinstance(v, str):
            out = out.replace(f"__{k}__", v)
    return out


def init_mode():
    """first run of a session: take the theme from the URL (?theme=light) so a page reload keeps it"""
    if "ui_light" not in st.session_state:
        st.session_state["ui_light"] = st.query_params.get("theme") == "light"


def _remember_mode():
    st.query_params["theme"] = "light" if st.session_state.get("ui_light") else "dark"


def mode_switch():
    """the Light / Dark switch shown in the header"""
    st.toggle(":material/light_mode: Light", key="ui_light", on_change=_remember_mode,
              help="Switch between the dark and the light theme")


def inject():
    st.markdown(css(), unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def step(n, title, hint=""):
    st.markdown(f'<div class="nt-step"><span class="n">{n:02d}</span><span class="t">{esc(title)}</span>'
                f'<span class="h">{esc(hint)}</span></div>', unsafe_allow_html=True)


def chip(text, kind=""):
    return f'<span class="nt-chip {kind}">{esc(text)}</span>'


def chips(items, kind=""):
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 8px 0">' +
                "".join(chip(t, kind) for t in items) + "</div>", unsafe_allow_html=True)


def kpi(label, value, sub="", red=False, hl=False):
    return (f'<div class="nt-kpi{" hl" if hl else ""}"><div class="nt-label">{esc(label)}</div>'
            f'<div class="v{" red" if red else ""}{_size(value)}">{esc(value)}</div>'
            f'<div class="s">{esc(sub)}</div></div>')


def _size(value):
    """tile font: words such as 'Intermediate' get the small size instead of being broken in two"""
    v = str(value)
    if len(v) > 7 and " " not in v.strip() and not any(ch.isdigit() for ch in v):
        return " xs"
    return " sm" if len(v) > 9 else ""


def kpis(items):
    """items: list of dicts(label, value, sub, red, hl)"""
    st.markdown('<div class="nt-grid">' + "".join(kpi(**i) for i in items) + "</div>", unsafe_allow_html=True)


def progress(label, frac, right="", n=72):
    frac = 0.0 if frac is None else max(0.0, min(1.0, float(frac)))
    on = int(round(frac * n))
    dots = "".join('<i class="head"></i>' if (k == on - 1 and frac < 1) else ('<i class="on"></i>' if k < on else "<i></i>")
                   for k in range(n))
    st.markdown(f'<div class="nt-prog"><div class="row"><span class="nt-label">{esc(label)}</span>'
                f'<span class="nt-label" style="color:var(--soft)">{esc(right)}</span></div><div class="nt-dots">{dots}</div></div>',
                unsafe_allow_html=True)


def banner(text, ok=False):
    st.markdown(f'<div class="nt-banner{" ok" if ok else ""}">{text}</div>', unsafe_allow_html=True)


STATE_STYLE = {
    "running": ("live", "TRAINING"), "loading": ("live", "LOADING DATA"), "starting": ("live", "STARTING"),
    "explaining": ("live", "COMPUTING SHAP"), "finished": ("white", "FINISHED"), "stopped": ("", "STOPPED"),
    "failed": ("red", "FAILED"), "interrupted": ("red", "INTERRUPTED"), "imported": ("white", "IMPORTED"),
}


def state_chip(state):
    dot, text = STATE_STYLE.get(state, ("", str(state).upper()))
    return f'<span class="nt-chip {"red" if dot in ("live", "red") else ""}"><span class="nt-dot {dot}"></span>{text}</span>'
