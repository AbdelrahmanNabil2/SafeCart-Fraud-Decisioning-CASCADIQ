"""
CASCADIQ — SafeCart Fraud Decisioning Command Center
TAG AI Team Development Program · Case 06 · Prepared by Abdelrahman

Run:  streamlit run app.py
Training, the LP, and Monte Carlo live in precompute.py / the notebook.
This app only loads those artifacts and the source CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CYAN, GOLD, AMBER, RED = "#2DD4E8", "#F5C14A", "#FB923C", "#F43F5E"
MINT = "#34D399"

THEMES = {
    "dark": {
        "bg": "#070B12",
        "bg2": "#0A141A",
        "card": "#101A22",
        "card2": "#15232C",
        "ink": "#E8F2F4",
        "muted": "#8AA3AB",
        "line": "#1C3340",
        "sub": "#B7C9CE",
        "pill_bg": "#0E1A22",
        "pill_fg": "#C5D8DC",
        "tab_on": "#14303A",
        "glow": "rgba(45,212,232,.13)",
        "glow2": "rgba(245,193,74,.07)",
        "hero_from": "rgba(16,28,36,.94)",
        "hero_to": "rgba(12,40,48,.90)",
        "halo": "rgba(45,212,232,.20)",
        "callout_fg": "#C5D8DC",
    },
    "light": {
        "bg": "#F3F7F8",
        "bg2": "#E8F1F3",
        "card": "#FFFFFF",
        "card2": "#F7FBFC",
        "ink": "#102028",
        "muted": "#5A727A",
        "line": "#D3E1E5",
        "sub": "#3E555C",
        "pill_bg": "#EEF5F6",
        "pill_fg": "#2A4048",
        "tab_on": "#E5F6F9",
        "glow": "rgba(14,140,160,.10)",
        "glow2": "rgba(200,140,20,.07)",
        "hero_from": "rgba(255,255,255,.96)",
        "hero_to": "rgba(226,244,247,.94)",
        "halo": "rgba(14,140,160,.16)",
        "callout_fg": "#2A4048",
    },
}

ACTIONS = ["Approve", "Step-up Authentication", "Manual Review", "Temporary Hold"]


def wrap_action(name: str) -> str:
    return name.replace(" ", "<br>", 1) if " " in name else name


ACTION_COLOR = {
    "Approve": CYAN,
    "Step-up Authentication": GOLD,
    "Manual Review": AMBER,
    "Temporary Hold": RED,
}
BANDS = ["Low", "Medium", "High", "Critical"]
NUM_FEATS = [
    "Customer_Tenure_Months",
    "Device_Known_Flag",
    "New_Device_Age_Hours",
    "IP_Distance_From_Usual_KM",
    "Failed_Logins_24H",
    "Password_Reset_48H",
    "Transaction_Amount_EGP",
    "Amount_vs_Customer_Avg",
    "Shipping_Address_New_Flag",
]
MISS, REVIEW_COST, STEP_COST, FALSEHOLD = 8000, 60, 8, 220
DETERRENCE = 0.5
REVIEW_CAP, HOLD_CAP = 1800, 350
PLOT = {"displayModeBar": False, "responsive": True}

st.set_page_config(
    page_title="CASCADIQ · SafeCart Fraud Decisioning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── data ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CASCADIQ…")
def load_bundle():
    with open(DATA / "summary.json") as f:
        summary = json.load(f)

    scored = pd.read_csv(DATA / "current_scored.csv")
    scored["Event_Date_Time"] = pd.to_datetime(scored["Event_Date_Time"])

    dec_path = DATA / "decision_output.csv"
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        extra = [c for c in dec.columns if c == "Event_ID" or c not in scored.columns]
        scored = scored.merge(dec[extra], on="Event_ID", how="left")

    if "Risk_Band" not in scored.columns:
        scored["Risk_Band"] = pd.cut(
            scored["fraud_prob"],
            bins=[-0.01, 0.01, 0.05, 0.15, 1.0],
            labels=BANDS,
        ).astype(str)
    if "Priority_Rank" not in scored.columns:
        scored["Priority_Rank"] = scored["fraud_prob"].rank(ascending=False, method="first").astype(int)

    hist = pd.read_csv(DATA / "02_SafeCart_Historical_Data.csv", low_memory=False)
    hist = hist.drop_duplicates().reset_index(drop=True)
    hist["Event_Date_Time"] = pd.to_datetime(hist["Event_Date_Time"])
    hist = hist.sort_values("Event_Date_Time").reset_index(drop=True)
    hist["hour"] = hist["Event_Date_Time"].dt.hour
    hist["dow"] = hist["Event_Date_Time"].dt.dayofweek
    hist["month"] = hist["Event_Date_Time"].dt.to_period("M").astype(str)

    mc = np.load(DATA / "monte_carlo_loss_prevented_pct.npy")

    catch = float(summary["review_catch_rate"])
    p = scored["fraud_prob"].to_numpy()
    scored = scored.copy()
    scored["cost_approve"] = p * MISS
    scored["cost_stepup"] = STEP_COST + p * (1 - DETERRENCE) * MISS
    scored["cost_review"] = REVIEW_COST + p * (1 - catch) * MISS
    scored["cost_hold"] = (1 - p) * FALSEHOLD
    cost_col = {
        "Approve": "cost_approve",
        "Step-up Authentication": "cost_stepup",
        "Manual Review": "cost_review",
        "Temporary Hold": "cost_hold",
    }
    scored["expected_cost"] = 0.0
    for act, col in cost_col.items():
        mask = scored["action"] == act
        scored.loc[mask, "expected_cost"] = scored.loc[mask, col]
    return summary, scored, hist, mc


@st.cache_data(show_spinner=False)
def neighbor_index(_hist: pd.DataFrame):
    h = _hist.copy()
    med = h[NUM_FEATS].median(numeric_only=True)
    h[NUM_FEATS] = h[NUM_FEATS].fillna(med)
    mu = h[NUM_FEATS].mean()
    sd = h[NUM_FEATS].std().replace(0, 1)
    fraud = h.loc[h["Chargeback_Confirmed_30D"] == 1, ["Event_ID"] + NUM_FEATS].copy()
    H = ((fraud[NUM_FEATS] - mu) / sd).to_numpy()
    return fraud.reset_index(drop=True), H, mu.to_numpy(), sd.to_numpy(), med.to_dict()


def simulate_mc(
    p,
    action,
    n_sims,
    det_lo,
    det_hi,
    seed,
    miss,
    review_cost,
    step_cost,
    falsehold,
    catch,
):
    """Vectorized Monte Carlo. Actions stay frozen; only deterrence is redrawn."""
    action = np.asarray(action)
    p = np.asarray(p, dtype=float)
    is_approve = action == "Approve"
    is_step = action == "Step-up Authentication"
    is_review = action == "Manual Review"
    is_hold = action == "Temporary Hold"
    do_nothing = float((p * miss).sum())
    approve_c = float((p[is_approve] * miss).sum())
    review_c = float((review_cost + p[is_review] * (1.0 - catch) * miss).sum())
    hold_c = float(((1.0 - p[is_hold]) * falsehold).sum())
    step_fixed = float(step_cost * int(is_step.sum()))
    step_risk = float((p[is_step] * miss).sum())
    rng = np.random.default_rng(int(seed))
    det = rng.uniform(float(det_lo), float(det_hi), int(n_sims))
    total = approve_c + review_c + hold_c + step_fixed + step_risk * (1.0 - det)
    loss_pct = 100.0 * (do_nothing - total) / do_nothing
    floor_cost = approve_c + review_c + hold_c + step_fixed + step_risk
    floor_pct = 100.0 * (do_nothing - floor_cost) / do_nothing
    return loss_pct, float(floor_pct), do_nothing


def nearest_fraud(row, hist, k=3):
    fraud, H, mu, sd, med = neighbor_index(hist)
    x = np.array(
        [row[c] if c in row.index and pd.notna(row[c]) else med[c] for c in NUM_FEATS],
        dtype=float,
    )
    dist = np.linalg.norm(H - (x - mu) / sd, axis=1)
    idx = np.argsort(dist)[:k]
    out = fraud.iloc[idx].copy()
    out["distance"] = dist[idx]
    return out


summary, scored, hist, mc = load_bundle()
base_rate = float(summary["base_rate"])
catch_rate = float(summary["review_catch_rate"])
n_hist = int(summary["n_historical"])
n_ops = len(scored)
ac = summary["action_counts"]
approve_used = int(ac.get("Approve", 0))
step_used = int(ac.get("Step-up Authentication", 0))
review_used = int(ac.get("Manual Review", 0))
hold_used = int(ac.get("Temporary Hold", 0))


# ── theme ───────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="brand">
      <svg width="28" height="26" viewBox="0 0 32 26" fill="none">
        <rect x="1" y="16" width="8" height="9" rx="2" fill="#2DD4E8" opacity="0.35"/>
        <rect x="12" y="8" width="8" height="17" rx="2" fill="#2DD4E8" opacity="0.7"/>
        <rect x="23" y="1" width="8" height="24" rx="2" fill="#2DD4E8"/>
      </svg>
      <div><h2>CASCADIQ</h2><p>SafeCart fraud decisioning</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)
is_dark = st.sidebar.toggle("Dark mode", value=False, key="theme_dark")
T = THEMES["dark" if is_dark else "light"]
INK, MUTED, LINE, SURFACE = T["ink"], T["muted"], T["line"], T["card"]

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp, p, h1, h2, h3, h4 {{ font-family: 'Outfit', sans-serif; }}
.stApp {{
  background:
    radial-gradient(ellipse 90% 50% at 8% -8%, {T['glow']}, transparent 52%),
    radial-gradient(ellipse 50% 40% at 100% 0%, {T['glow2']}, transparent 46%),
    {T['bg']};
  color: {T['ink']};
}}
.block-container {{ padding-top: 1.8rem; padding-bottom: 3.2rem; padding-left: 2.4rem; padding-right: 2.4rem; max-width: 1440px; }}
div[data-testid="stVerticalBlock"] {{ gap: 1.35rem !important; }}
div[data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
div[data-testid="stElementContainer"] {{ margin-bottom: 0.15rem; }}
div[data-testid="stCaptionContainer"] {{ margin: 0.15rem 0 0.85rem !important; }}
div[data-testid="column"] {{ overflow: visible; padding-left: 0.2rem; padding-right: 0.2rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {T['bg2']} 0%, {T['bg']} 100%);
  border-right: 1px solid {T['line']};
}}
section[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
  background: {T['card']}; border: 1px solid {T['line']}; border-radius: 10px;
  padding: 11px 12px; margin-bottom: 8px; color: {T['ink']}; font-size: 0.98rem;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{ border-color: {CYAN}; }}
.brand {{ display:flex; gap:10px; align-items:center; margin: 2px 0 12px; }}
.brand h2 {{ margin:0; font-size:1.2rem; letter-spacing:.1em; font-weight:700; line-height:1.2; color:{T['ink']}; }}
.brand p {{ margin:0; color:{T['muted']}; font-size:.76rem; }}
.hero {{
  background: linear-gradient(135deg, {T['hero_from']}, {T['hero_to']});
  border: 1px solid {T['line']}; border-radius: 18px; padding: 22px 26px 18px;
  margin-bottom: 22px; position: relative; overflow: hidden;
}}
.hero:before {{
  content:''; position:absolute; right:-50px; top:-70px; width:260px; height:240px;
  background: radial-gradient(circle, {T['halo']}, transparent 70%);
}}
.kicker {{ color:{CYAN}; letter-spacing:.10em; font-size:.84rem; font-weight:600; text-transform:uppercase; margin-bottom:8px; }}
.hero h1 {{ margin:6px 0 0; font-size:2.15rem; letter-spacing:.12em; font-weight:700; line-height:1.15; color:{T['ink']}; }}
.hero-sub {{ color:{T['sub']}; font-size:1.02rem; margin:10px 0 14px; max-width:840px; line-height:1.45; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; }}
.pill {{
  border:1px solid {T['line']}; background:{T['pill_bg']}; color:{T['pill_fg']};
  border-radius:999px; padding:4px 11px; font-size:.78rem; font-weight:500;
}}
.pill.cyan {{ border-color:#1AA7B8; color:{CYAN}; }}
.pill.gold {{ border-color:#B45309; color:{GOLD}; }}
.pill.red  {{ border-color:#9F1239; color:{RED}; }}
.pill.mint {{ border-color:#047857; color:{MINT}; }}
.kpi {{
  background: linear-gradient(180deg,{T['card2']},{T['card']}); border: 1px solid {T['line']};
  border-radius:16px; padding:22px 20px 18px; height:100%; min-height:168px;
  box-sizing: border-box;
}}
.kpi.cyan {{ border-left:3px solid {CYAN}; }}
.kpi.gold {{ border-left:3px solid {GOLD}; }}
.kpi.red  {{ border-left:3px solid {RED}; }}
.kpi.mint {{ border-left:3px solid {MINT}; }}
.kpi.amber{{ border-left:3px solid {AMBER}; }}
.kpi-label {{ color:{T['muted']}; font-size:.86rem; letter-spacing:.05em; text-transform:uppercase; margin-bottom:10px; line-height:1.4; font-weight:600; }}
.kpi-value {{ font-size:1.85rem; font-weight:700; color:{T['ink']}; line-height:1.25; font-variant-numeric: tabular-nums; word-break: break-word; margin: 4px 0 8px; }}
.kpi-hint {{ color:{T['muted']}; font-size:.95rem; margin-top:10px; line-height:1.45; }}
.panel {{ background:{T['card']}; border:1px solid {T['line']}; border-radius:16px; padding:18px 20px; margin-bottom:16px; }}
.panel h3 {{ margin:0 0 8px; font-size:1.02rem; color:{T['ink']}; }}
.panel p, .panel li {{ color:{T['sub']}; font-size:1.02rem; line-height:1.55; }}
.callout {{ border-radius:14px; padding:16px 18px; margin:14px 0 20px; border:1px solid {T['line']}; }}
.callout.cyan {{ background:rgba(45,212,232,.10); border-color:#1AA7B8; }}
.callout.gold {{ background:rgba(245,193,74,.12); border-color:#B45309; }}
.callout.red  {{ background:rgba(244,63,94,.10); border-color:#9F1239; }}
.callout.mint {{ background:rgba(52,211,153,.10); border-color:#047857; }}
.callout p {{ margin:0; color:{T['callout_fg']}; line-height:1.55; font-size:1.02rem; }}
.flag {{ display:inline-block; border-radius:999px; padding:3px 10px; font-size:.75rem; margin:0 6px 6px 0; border:1px solid {T['line']}; }}
.flag.on  {{ background:rgba(244,63,94,.12); color:{RED}; border-color:#9F1239; }}
.flag.off {{ background:{T['card']}; color:{T['muted']}; }}
.cap-head {{ display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:{T['sub']}; font-size:.85rem; margin-bottom:6px; line-height:1.35; }}
.cap-track {{ height:8px; background:{T['line']}; border-radius:99px; overflow:hidden; }}
.cap-fill {{ height:100%; border-radius:99px; }}
.mono {{ font-family:'JetBrains Mono', monospace; }}
div[data-testid="stMetric"] {{
  background: linear-gradient(180deg,{T['card2']},{T['card']}); border:1px solid {T['line']};
  border-radius:14px; padding:14px 16px;
}}
.stTabs {{ margin-top: 0.55rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:14px; margin-bottom: 1.05rem; padding-top: 6px; }}
.stTabs [data-baseweb="tab"] {{
  background:{T['card']}; border:1px solid {T['line']}; border-radius:10px; color:{T['muted']}; padding:11px 16px; font-size:1rem;
}}
.stTabs [aria-selected="true"] {{ background:{T['tab_on']} !important; color:{CYAN} !important; border-color:#1AA7B8; }}
.stDownloadButton button, .stButton button {{
  background:{T['tab_on']}; color:{CYAN}; border:1px solid #1AA7B8; border-radius:10px;
}}
div[data-testid="stExpander"] {{ background:{T['card']}; border:1px solid {T['line']}; border-radius:12px; }}
div[data-testid="stMarkdown"] p {{ margin: 0 !important; color: {T['ink']}; }}
div[data-testid="stCaption"] {{ line-height: 1.5; margin-bottom: 0.35rem !important; color: {T['muted']} !important; font-size: 1rem !important; }}
h1, h2, h3, h4 {{ color: {T['ink']} !important; }}
h2 {{ line-height: 1.35 !important; margin: 0.15rem 0 0.7rem !important; font-size: 1.7rem !important; }}
h3, h4 {{ margin-top: 0.4rem !important; margin-bottom: 0.6rem !important; }}
div[data-testid="stPlotlyChart"] {{ margin: 0.35rem 0 1.1rem; overflow: visible; }}
.js-plotly-plot .plotly, .js-plotly-plot .main-svg {{ overflow: visible !important; }}
.stTabs [data-baseweb="tab"] {{ white-space: nowrap; line-height: 1.35; }}
div[data-testid="stDataFrame"] {{ margin: 0.4rem 0 1rem; }}
.section-gap {{ height: 16px; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {T['ink']}; }}
[data-testid="stWidgetLabel"] {{ color: {T['ink']} !important; }}
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {T['ink']} !important; }}
div[data-testid="stToggle"] label, div[data-testid="stToggle"] p {{ color: {T['ink']} !important; }}
section[data-testid="stSidebar"] hr {{ border-color: {T['line']}; }}
[data-testid="stHeader"] {{ background: {T['bg']}; }}
</style>
""",
    unsafe_allow_html=True,
)


def _layout_has_y2(fig) -> bool:
    return "yaxis2" in fig.to_dict().get("layout", {})


def _bar_vals(arr) -> list:
    if arr is None:
        return []
    return [v for v in np.asarray(arr).ravel().tolist() if v is not None]


def style_fig(fig, height=400, title=None, ylabel=None, xlabel=None):
    named = sum(1 for tr in fig.data if getattr(tr, "name", None))
    show_legend = named > 1
    has_y2 = _layout_has_y2(fig)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=INK),
            x=0,
            xanchor="left",
            y=0.98,
            pad=dict(b=10, t=4),
        )
        if title
        else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color=MUTED, size=14),
        height=height,
        margin=dict(
            l=64,
            r=80 if has_y2 else 44,
            t=84 if title else 44,
            b=108 if show_legend else 68,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0,
            xanchor="left",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            itemwidth=80,
        ),
        showlegend=show_legend,
        bargap=0.32,
        hoverlabel=dict(bgcolor=SURFACE, font_size=13, font_family="Outfit", font_color=INK),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    xa = dict(
        gridcolor=LINE,
        zeroline=False,
        linecolor=LINE,
        automargin=True,
        title_standoff=16,
        tickfont=dict(size=13),
        ticksuffix=" ",
    )
    ya = dict(
        gridcolor=LINE,
        zeroline=False,
        linecolor=LINE,
        automargin=True,
        title_standoff=12,
        tickfont=dict(size=13),
        ticksuffix=" ",
    )
    if xlabel:
        xa["title"] = xlabel
    if ylabel:
        ya["title"] = ylabel
    fig.update_layout(xaxis=xa, yaxis=ya)
    if has_y2:
        fig.update_layout(yaxis2=dict(automargin=True, title_standoff=14, tickfont=dict(size=12)))
        fig.update_layout(margin_r=72)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    vmax_v, vmax_h = 0.0, 0.0
    has_v = has_h = False
    for tr in fig.data:
        if getattr(tr, "type", None) != "bar":
            continue
        ori = getattr(tr, "orientation", None) or "v"
        if ori == "h":
            has_h = True
            vals = _bar_vals(tr.x)
            if vals:
                vmax_h = max(vmax_h, max(vals))
        else:
            has_v = True
            vals = _bar_vals(tr.y)
            if vals:
                vmax_v = max(vmax_v, max(vals))
    if has_v and vmax_v > 0 and not has_y2:
        fig.update_layout(yaxis_range=[0, vmax_v * 1.22])
    if has_h and vmax_h > 0:
        existing = fig.layout.xaxis.range
        lo = float(existing[0]) if existing else 0.0
        hi = float(existing[1]) if existing else vmax_h
        fig.update_xaxes(range=[lo, max(hi, vmax_h) * 1.12])
    return fig


def kpi_row(items):
    cols = st.columns(len(items), gap="large")
    for col, it in zip(cols, items):
        label, value = it[0], it[1]
        hint = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else "cyan"
        col.markdown(
            f'<div class="kpi {tone}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-hint">{hint}</div></div>',
            unsafe_allow_html=True,
        )


def cap_bar(label, used, cap, color):
    pct = min(100.0, 100.0 * used / cap) if cap else 0
    st.markdown(
        f"""<div class="panel" style="padding:14px 16px;margin-bottom:12px;">
          <div class="cap-head"><span>{label}</span>
          <span class="mono">{used:,} / {cap:,} · {pct:.0f}%</span></div>
          <div class="cap-track"><div class="cap-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def heading(kicker, title, sub=None):
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if sub:
        st.caption(sub)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def flags(row):
    bits = [
        ("Unrecognized device", int(row.get("Device_Known_Flag", 1)) == 0),
        ("Password reset 48h", int(row.get("Password_Reset_48H", 0)) == 1),
        ("New shipping address", int(row.get("Shipping_Address_New_Flag", 0)) == 1),
        ("Location missing", int(row.get("IP_dist_missing", 0)) == 1),
        ("Spend-history missing", int(row.get("Amt_ratio_missing", 0)) == 1),
    ]
    return "".join(f'<span class="flag {"on" if on else "off"}">{name}</span>' for name, on in bits)


def lift(rate: float) -> float:
    return rate / base_rate if base_rate else 0.0


# ── sidebar ───────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div class="pills"><span class="pill cyan">SHADOW MODE</span>'
    '<span class="pill gold">AUC 0.808</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("")

PAGES = [
    "Command Center",
    "Data & Quality",
    "Risk Intelligence",
    "Model Lab",
    "Decision Policy",
    "Monte Carlo",
    "Live Operations",
    "Responsible AI",
]
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data window**")
st.sidebar.caption("History · Jan 2025 – Aug 2026")
st.sidebar.caption("Ops file · 24–26 Aug 2026")
st.sidebar.caption(f"{n_hist:,} clean historical events")
st.sidebar.caption(f"{n_ops:,} current operational events")
st.sidebar.markdown("---")
st.sidebar.caption("Case 06 · TAG AI Team Development Program")
st.sidebar.caption("Prepared by Abdelrahman")
st.sidebar.caption("Every number traces to the reproducible notebook.")


# ── Command Center ───────────────────────────────────────────────────
def page_home():
    miss_pct = summary["missed_by_review"] / summary["total_fraud"] * 100
    st.markdown(
        f"""
        <div class="hero">
          <div class="kicker">TAG AI · Case 06 · SafeCart Head of Fraud Operations</div>
          <h1>CASCADIQ</h1>
          <p class="hero-sub">A capacity-aware escalation system built on evidence, not intuition — four actions, one linear program, and a fairness finding that changes the monitoring plan.</p>
          <div class="pills">
            <span class="pill cyan">4-week shadow-mode pilot</span>
            <span class="pill mint">−{summary['mc_mean_pct']:.1f}% expected loss</span>
            <span class="pill gold">Hold {hold_used} / {HOLD_CAP} used</span>
            <span class="pill red">{miss_pct:.1f}% of fraud currently missed</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    kpi_row(
        [
            (
                "Missed by current review",
                f"{miss_pct:.1f}%",
                f"{summary['missed_by_review']:,} of {summary['total_fraud']:,} confirmed fraud",
                "red",
            ),
            (
                "Review catch rate today",
                f"{catch_rate*100:.1f}%",
                f"{summary['caught_by_review']:,} caught — the bar the pilot must clear",
                "gold",
            ),
            (
                "Expected loss reduction",
                f"{summary['mc_mean_pct']:.1f}%",
                f"90% CI {summary['mc_p5_pct']:.1f} – {summary['mc_p95_pct']:.1f}% · floor {summary['floor_pct']:.1f}%",
                "mint",
            ),
            (
                "Champion model",
                "0.808 AUC",
                f"PR-AUC {summary.get('champion_cv_ap', 0.183):.3f} (leads all 8 models) · process recall {summary.get('process_recall', catch_rate)*100:.1f}% / precision {summary.get('process_precision', 1)*100:.0f}%",
                "cyan",
            ),
        ]
    )
    st.markdown("")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### From 2,500 live events to four actions")
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=["Current ops file", "Approve", "Step-up", "Manual Review", "Temporary Hold"],
                    color=["#1AA7B8", CYAN, GOLD, AMBER, RED],
                    pad=22,
                    thickness=18,
                    line=dict(color=LINE, width=0.5),
                ),
                link=dict(
                    source=[0, 0, 0, 0],
                    target=[1, 2, 3, 4],
                    value=[approve_used, step_used, review_used, hold_used],
                    color=[
                        "rgba(45,212,232,0.28)",
                        "rgba(245,193,74,0.28)",
                        "rgba(251,146,60,0.28)",
                        "rgba(244,63,94,0.28)",
                    ],
                ),
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit", color=INK, size=13),
            height=380,
            margin=dict(l=16, r=16, t=20, b=16),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT)
        st.caption(
            "The LP spends almost all of the scarce Hold cap and only 28% of Manual Review, "
            "because cheap step-up authentication absorbs the middle of the score distribution."
        )
    with right:
        st.markdown("#### Capacity — the scarce resource is Hold")
        cap_bar("Temporary Hold (senior approval above cap)", hold_used, HOLD_CAP, RED)
        cap_bar("Manual Review", review_used, REVIEW_CAP, AMBER)
        cap_bar("Step-up Authentication (uncapped)", step_used, n_ops, GOLD)
        st.markdown(
            f"""
            <div class="callout mint">
              <p><strong>What management is being asked to approve:</strong> a 4-week shadow-mode pilot. The policy scores every event and logs its recommendation. No live customer action changes until the pilot clears review. Success bar: confirmed-fraud recall ≥ {catch_rate * 100:.1f}% without breaching 1,800 review / 350 hold, and without widening the nationality / gender gap.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### The economics in one picture")
    pc = {
        "Do nothing<br>(approve all)": summary["do_nothing_cost"],
        "Naive 2-bucket<br>(no step-up)": summary["naive2_cost"],
        "LP-optimal<br>(4 actions)": summary["lp_optimal_cost"],
    }
    saved = summary["do_nothing_cost"] - summary["lp_optimal_cost"]
    extra = summary["naive2_cost"] - summary["lp_optimal_cost"]
    fig = go.Figure(
        go.Bar(
            x=list(pc.keys()),
            y=list(pc.values()),
            marker_color=[MUTED, GOLD, CYAN],
            text=[f"{v:,.0f} EGP" for v in pc.values()],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} EGP<extra></extra>",
        )
    )
    st.plotly_chart(style_fig(fig, height=380, ylabel="Expected cost (EGP)"), use_container_width=True, config=PLOT)

    held_amt = scored.loc[scored["action"] == "Temporary Hold", "Transaction_Amount_EGP"].sum()
    kpi_row(
        [
            ("LP vs do-nothing", f"{saved:,.0f} EGP", "expected cost avoided on this 2,500-event file", "mint"),
            ("LP vs naive 2-bucket", f"{extra:,.0f} EGP", "the value of adding a step-up tier, solved jointly", "gold"),
            (
                "Amount in the file",
                f"{scored['Transaction_Amount_EGP'].sum()/1e6:.1f}M EGP",
                f"{held_amt/1e6:.2f}M EGP sitting in Hold",
                "cyan",
            ),
        ]
    )
    with st.expander("How to walk this page in a defense"):
        st.markdown(
            """
            1. Start with the miss rate — it is SafeCart's own process, not an external benchmark.
            2. Show the Sankey: Hold is almost fully used; review is not. That is the LP talking, not a threshold guess.
            3. Close with the ask: shadow mode, named recall/capacity/fairness bars, four weeks.
            """
        )


# ── Data & Quality ────────────────────────────────────────────────────
def page_data():
    heading(
        "Integrity",
        "Data understanding & leakage control",
        "35,070 historical rows in → 35,000 trustworthy rows out. Decision-time features only.",
    )
    kpi_row(
        [
            (
                "Raw historical rows",
                f"{n_hist + summary['n_duplicates_dropped']:,}",
                "source file before cleaning",
                "cyan",
            ),
            (
                "Exact duplicates dropped",
                f"−{summary['n_duplicates_dropped']}",
                "byte-identical pairs, verified before drop",
                "gold",
            ),
            ("Clean rows used", f"{n_hist:,}", "the training universe for every model", "mint"),
            ("Base fraud rate", f"{base_rate*100:.2f}%", f"{summary['total_fraud']:,} chargebacks in 30 days", "red"),
        ]
    )

    t1, t2, t3 = st.tabs(["Pipeline & coverage", "Missingness", "Leakage wall"])

    with t1:
        monthly = (
            hist.groupby("month", observed=True)
            .agg(
                events=("Event_ID", "count"),
                fraud=("Chargeback_Confirmed_30D", "sum"),
                rate=("Chargeback_Confirmed_30D", "mean"),
            )
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["events"], name="Events", marker_color="#1A3A44"))
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["rate"] * 100,
                name="Fraud rate %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=RED, width=2),
                marker=dict(size=6),
            )
        )
        fig.update_layout(
            yaxis=dict(title="Events", gridcolor=LINE, zeroline=False),
            yaxis2=dict(title="Fraud rate %", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(
            style_fig(fig, height=460, title="Monthly volume and fraud rate — history is long enough to trust a time split"),
            use_container_width=True,
            config=PLOT,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="panel">
                  <h3>Entity & target</h3>
                  <p>One row is one login/transaction event, identified by <span class="mono">Event_ID</span> — not one customer. The same customer can appear many times. The target is <span class="mono">Chargeback_Confirmed_30D</span>, the ground-truth harm the decision exists to prevent.</p>
                </div>
                <div class="panel">
                  <h3>Coverage</h3>
                  <p>History runs from <strong>{hist['Event_Date_Time'].min():%d %b %Y}</strong> to <strong>{hist['Event_Date_Time'].max():%d %b %Y}</strong>. The current ops file is a {n_ops:,}-event window from {scored['Event_Date_Time'].min():%d %b %Y %H:%M} to {scored['Event_Date_Time'].max():%d %b %Y %H:%M}.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            outcomes = hist["Manual_Review_Outcome"].value_counts()
            fig = go.Figure(
                go.Pie(
                    labels=outcomes.index,
                    values=outcomes.values,
                    hole=0.62,
                    marker=dict(colors=[CYAN, RED, GOLD]),
                    textinfo="label+percent",
                )
            )
            st.plotly_chart(style_fig(fig, height=300, title="Historical review outcomes"), use_container_width=True, config=PLOT)

        st.markdown("##### Outlier scan — caps, not corruptions")
        st.dataframe(
            pd.DataFrame(
                [
                    ["New_Device_Age_Hours", "720h (30d)", "Operational ceiling, not an error"],
                    ["Customer_Tenure_Months", "96 months", "8-year cap, left as observed"],
                    ["Amount_vs_Customer_Avg", "8.0×", "Two rows sit on a generation clip"],
                    ["Failed_Logins_24H", "12", "Plausible lockout ceiling"],
                ],
                columns=["Field", "Cap observed", "Reading"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("No negatives, non-finites, or impossible values. Caps stay in the data.")

    with t2:
        miss_df = pd.DataFrame(summary["missingness"]).T.reset_index().rename(columns={"index": "Field"})
        miss_df.columns = ["Field", "% missing", "Fraud rate when missing", "Fraud rate when present"]
        miss_long = []
        for _, r in miss_df.iterrows():
            miss_long += [
                {"Field": r["Field"], "When": "Missing", "Fraud rate %": r["Fraud rate when missing"] * 100},
                {"Field": r["Field"], "When": "Present", "Fraud rate %": r["Fraud rate when present"] * 100},
            ]
        mlong = pd.DataFrame(miss_long)
        fig = go.Figure()
        for when, color in [("Missing", RED), ("Present", CYAN)]:
            sl = mlong[mlong["When"] == when]
            fig.add_trace(
                go.Bar(
                    name=when,
                    x=sl["Field"],
                    y=sl["Fraud rate %"],
                    marker_color=color,
                    text=[f"{v:.2f}%" for v in sl["Fraud rate %"]],
                    textposition="outside",
                )
            )
        fig.update_layout(barmode="group")
        st.plotly_chart(
            style_fig(fig, height=400, title="Missing is mildly informative — we keep the rows", ylabel="Fraud rate (%)"),
            use_container_width=True,
            config=PLOT,
        )
        show = miss_df.copy()
        show["% missing"] = show["% missing"].map(lambda x: f"{x:.2f}%")
        show["Fraud rate when missing"] = (show["Fraud rate when missing"] * 100).map(lambda x: f"{x:.2f}%")
        show["Fraud rate when present"] = (show["Fraud rate when present"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="callout cyan">
              <p>Impute with the median and add a missing-indicator flag. Dropping the rows would throw away
              a weak but real signal and shrink an already-rare target.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown(
            """
            <div class="callout red">
              <p><strong>Decision-time boundary.</strong> Anything timestamped after the decision, or that is the
              target itself, is excluded from every feature set — including the champion, the LP, and the reason codes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["Manual_Review_Outcome", "After risk action", "Leakage — the reviewer's own call"],
                    ["Chargeback_Confirmed_30D", "Future outcome (target)", "The label, not a feature"],
                    ["Customer_Gender", "Audit-only", "Never a model input or reason code"],
                    ["Nationality_Group", "Audit-only", "Never a model input or reason code"],
                ],
                columns=["Field", "Why it is excluded", "How we treat it"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("##### What is allowed at decision time")
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": NUM_FEATS
                    + ["IP_dist_missing", "Amt_ratio_missing", "hour", "dow", "Account_Value_Band", "Payment_Method"],
                    "Role": ["numeric"] * 9 + ["missing flag", "missing flag", "time", "time", "categorical", "categorical"],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ── Risk Intelligence ───────────────────────────────────────────────
def page_risk():
    heading(
        "Signals",
        "Risk intelligence — six lifts, one combination that jumps 5×",
        "No single flag dominates. That is exactly why a scoring model earns its place over a hard rule.",
    )
    unknown = summary["eda"]["Device_Known_Flag"]["0"] * 100
    pwd = summary["eda"]["Password_Reset_48H"]["1"] * 100
    ship = summary["eda"]["Shipping_Address_New_Flag"]["1"] * 100
    combo = (
        hist.loc[
            (hist["Device_Known_Flag"] == 0) & (hist["Password_Reset_48H"] == 1),
            "Chargeback_Confirmed_30D",
        ].mean()
        * 100
    )
    kpi_row(
        [
            ("Unknown device", f"{unknown:.1f}%", f"{lift(unknown/100):.1f}× baseline", "red"),
            ("Password reset 48h", f"{pwd:.1f}%", f"{lift(pwd/100):.1f}× baseline", "gold"),
            ("New shipping address", f"{ship:.1f}%", f"{lift(ship/100):.1f}× baseline", "amber"),
            (
                "Unknown device × reset",
                f"{combo:.1f}%",
                f"{lift(combo/100):.1f}× baseline — the combination that matters",
                "cyan",
            ),
        ]
    )

    t1, t2, t3, t4 = st.tabs(["Flag lifts", "Combinations", "Money, logins, tenure", "When it happens"])

    with t1:
        rows = [
            ("Base rate (all events)", base_rate * 100, MUTED),
            ("Unknown device", unknown, RED),
            ("Password reset in 48h", pwd, GOLD),
            ("New shipping address", ship, AMBER),
        ]
        fig = go.Figure(
            go.Bar(
                x=[r[0] for r in rows],
                y=[r[1] for r in rows],
                marker_color=[r[2] for r in rows],
                text=[f"{r[1]:.1f}%" for r in rows],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            )
        )
        st.plotly_chart(
            style_fig(fig, height=430, title="Univariate fraud rate by the three case-cited flags", ylabel="Fraud rate (%)"),
            use_container_width=True,
            config=PLOT,
        )
        pay = hist.groupby("Payment_Method", observed=True)["Chargeback_Confirmed_30D"].agg(["mean", "count"]).reset_index()
        band = hist.groupby("Account_Value_Band", observed=True)["Chargeback_Confirmed_30D"].agg(["mean", "count"]).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(
                go.Bar(
                    x=pay["Payment_Method"],
                    y=pay["mean"] * 100,
                    marker_color=CYAN,
                    text=[f"{v*100:.1f}%" for v in pay["mean"]],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=340, title="Payment method — Card is riskier than COD", ylabel="Fraud rate (%)"),
                use_container_width=True,
                config=PLOT,
            )
        with c2:
            band["Account_Value_Band"] = pd.Categorical(band["Account_Value_Band"], ["Low", "Medium", "High"], ordered=True)
            band = band.sort_values("Account_Value_Band")
            fig = go.Figure(
                go.Bar(
                    x=band["Account_Value_Band"],
                    y=band["mean"] * 100,
                    marker_color=GOLD,
                    text=[f"{v*100:.1f}%" for v in band["mean"]],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=340, title="Account-value band — almost flat, so it is not a shortcut", ylabel="Fraud rate (%)"),
                use_container_width=True,
                config=PLOT,
            )

    with t2:
        st.markdown(
            "The model earns its keep when flags stack. An unknown device *and* a password reset "
            "is no longer a 4× story — it is a 5× story."
        )
        c1, c2 = st.columns(2)
        for cols, title, col in [
            (["Device_Known_Flag", "Password_Reset_48H"], "Unknown device × password reset", c1),
            (["Device_Known_Flag", "Shipping_Address_New_Flag"], "Unknown device × new shipping address", c2),
        ]:
            g = hist.groupby(cols, observed=True)["Chargeback_Confirmed_30D"].mean().unstack()
            z = g.values * 100
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=[f"{cols[1]} = {c}" for c in g.columns],
                    y=[("Unknown device" if i == 0 else "Known device") for i in g.index],
                    colorscale=[[0, "#0E1A22"], [1, RED]],
                    text=[[f"{v:.1f}%" for v in row] for row in z],
                    texttemplate="%{text}",
                    hovertemplate="%{y} · %{x}<br>%{z:.2f}%<extra></extra>",
                    colorbar=dict(title="Fraud %"),
                )
            )
            with col:
                st.plotly_chart(style_fig(fig, height=360, title=title), use_container_width=True, config=PLOT)

    with t3:
        c1, c2 = st.columns(2)
        amt = hist.copy()
        amt["bin"] = pd.cut(
            amt["Amount_vs_Customer_Avg"].fillna(hist["Amount_vs_Customer_Avg"].median()),
            bins=[0, 2, 4, 6, 9],
            labels=["0–2×", "2–4×", "4–6×", "6–8×"],
        )
        a = amt.groupby("bin", observed=True)["Chargeback_Confirmed_30D"].mean()
        with c1:
            fig = go.Figure(
                go.Bar(
                    x=a.index.astype(str),
                    y=a.values * 100,
                    marker_color=GOLD,
                    text=[f"{v*100:.1f}%" for v in a.values],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=360, title="Spend vs the customer's own average — a 4× climb", ylabel="Fraud rate (%)"),
                use_container_width=True,
                config=PLOT,
            )
            st.caption("This lift is as large as unknown-device, and it is not one of the three binary flags. The score has to see it.")
        fl = hist.copy()
        fl["bin"] = pd.cut(fl["Failed_Logins_24H"], bins=[-0.1, 2, 5, 8, 20], labels=["0–2", "3–5", "6–8", "9–12"])
        f = fl.groupby("bin", observed=True)["Chargeback_Confirmed_30D"].mean()
        with c2:
            fig = go.Figure(
                go.Bar(
                    x=f.index.astype(str),
                    y=f.values * 100,
                    marker_color=AMBER,
                    text=[f"{v*100:.1f}%" for v in f.values],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=360, title="Failed logins in 24h — 1.6% → 5.4%", ylabel="Fraud rate (%)"),
                use_container_width=True,
                config=PLOT,
            )

        ten = hist.copy()
        ten["bin"] = pd.qcut(ten["Customer_Tenure_Months"], 4, labels=["Newest quartile", "Q2", "Q3", "Longest tenure"])
        t = ten.groupby("bin", observed=True)["Chargeback_Confirmed_30D"].mean()
        fig = go.Figure(
            go.Bar(
                x=t.index.astype(str),
                y=t.values * 100,
                marker_color=CYAN,
                text=[f"{v*100:.1f}%" for v in t.values],
                textposition="outside",
            )
        )
        st.plotly_chart(
            style_fig(fig, height=360, title="Tenure — newer accounts run hotter, gently", ylabel="Fraud rate (%)"),
            use_container_width=True,
            config=PLOT,
        )
        fig = go.Figure()
        for lab, mask, color in [
            ("Legitimate", hist["Chargeback_Confirmed_30D"] == 0, CYAN),
            ("Confirmed fraud", hist["Chargeback_Confirmed_30D"] == 1, RED),
        ]:
            fig.add_trace(
                go.Histogram(
                    x=hist.loc[mask, "Transaction_Amount_EGP"],
                    name=lab,
                    histnorm="probability density",
                    opacity=0.55,
                    marker_color=color,
                    nbinsx=40,
                )
            )
        fig.update_layout(barmode="overlay")
        st.plotly_chart(
            style_fig(
                fig,
                height=360,
                title="Amount (EGP) — fraud sits slightly to the right, not in a different universe",
                xlabel="Transaction amount (EGP)",
            ),
            use_container_width=True,
            config=PLOT,
        )

    with t4:
        hour = hist.groupby("hour", observed=True)["Chargeback_Confirmed_30D"].mean() * 100
        fig = go.Figure(
            go.Scatter(
                x=hour.index,
                y=hour.values,
                mode="lines+markers",
                line=dict(color=CYAN, width=2),
                marker=dict(size=7),
                fill="tozeroy",
                fillcolor="rgba(45,212,232,0.12)",
            )
        )
        fig.add_hline(
            y=base_rate * 100,
            line_dash="dot",
            line_color=GOLD,
            annotation_text="baseline",
            annotation_font_color=GOLD,
            annotation_position="right",
        )
        st.plotly_chart(
            style_fig(fig, height=380, title="Hour of day — a wobble, not a regime change", ylabel="Fraud rate (%)", xlabel="Hour"),
            use_container_width=True,
            config=PLOT,
        )
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        dow = hist.groupby("dow", observed=True)["Chargeback_Confirmed_30D"].mean() * 100
        fig = go.Figure(
            go.Bar(
                x=[dow_map[i] for i in dow.index],
                y=dow.values,
                marker_color=CYAN,
                text=[f"{v:.2f}%" for v in dow.values],
                textposition="outside",
            )
        )
        st.plotly_chart(
            style_fig(fig, height=340, title="Day of week — similarly quiet", ylabel="Fraud rate (%)"),
            use_container_width=True,
            config=PLOT,
        )
        st.caption("Time-of-day and weekday are kept as features (the model can use a weak signal) but they are not a policy on their own.")


# ── Model Lab ───────────────────────────────────────────────────────
def page_model():
    heading(
        "Champion",
        "Eight approaches, tested identically — the linear ones win",
        "5-fold time-respecting cross-validation. Same features, same folds. ROC-AUC and PR-AUC both reported. Complexity did not help.",
    )
    mr = summary["model_results"]
    prauc = summary.get("model_prauc", {})
    ranked = sorted(mr.items(), key=lambda kv: -kv[1])
    names = [n for n, _ in ranked]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="ROC-AUC",
            y=list(names)[::-1],
            x=[mr[n] for n in names][::-1],
            orientation="h",
            marker_color=CYAN,
            text=[f"{mr[n]:.3f}" for n in names][::-1],
            textposition="outside",
            hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>",
        )
    )
    if prauc:
        fig.add_trace(
            go.Bar(
                name="PR-AUC",
                y=list(names)[::-1],
                x=[prauc.get(n, 0) for n in names][::-1],
                orientation="h",
                marker_color=GOLD,
                text=[f"{prauc.get(n, 0):.3f}" for n in names][::-1],
                textposition="outside",
                hovertemplate="%{y}<br>PR-AUC %{x:.3f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", xaxis=dict(range=[0, 0.95]))
    st.plotly_chart(
        style_fig(fig, height=560, title="Mean 5-fold TimeSeriesSplit — ROC-AUC and PR-AUC (notebook §7)", xlabel="Score"),
        use_container_width=True,
        config=PLOT,
    )
    st.markdown(
        f"""
        <div class="callout gold">
          <p><strong>Why both metrics.</strong> At a {base_rate*100:.2f}% base rate (~28:1), ROC-AUC is diluted by the huge true-negative pool and can look better than it deserves. PR-AUC is the more decision-relevant ranking score on the rare class (random-guess floor ≈ {base_rate:.3f}). Logistic Regression leads both lists, so the champion does not depend on which metric is treated as primary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prauc:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Ranked by AUC")
            st.dataframe(
                pd.DataFrame(
                    [{"Model": n, "AUC": f"{mr[n]:.3f}", "PR-AUC": f"{prauc.get(n, 0):.3f}"} for n, _ in ranked]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            pr_rank = sorted(prauc.items(), key=lambda kv: -kv[1])
            st.markdown("##### Ranked by PR-AUC")
            st.dataframe(
                pd.DataFrame(
                    [{"Model": n, "PR-AUC": f"{v:.3f}", "AUC": f"{mr.get(n, 0):.3f}"} for n, v in pr_rank]
                ),
                use_container_width=True,
                hide_index=True,
            )
    ap = float(summary.get("champion_cv_ap", 0.183))
    proc_p = float(summary.get("process_precision", 1.0))
    proc_r = float(summary.get("process_recall", catch_rate))
    hold_ops = summary.get("champion_ops", {}).get("hold_cap", {"precision": 0.155, "recall": 0.599})
    kpi_row(
        [
            ("Champion", "Logistic Regression", "Ties Linear SVM at AUC 0.808 · leads PR-AUC at 0.183 · natively calibrated", "cyan"),
            ("PR-AUC (Average Precision)", f"{ap:.3f}", f"{ap / base_rate:.1f}× the {base_rate*100:.2f}% base rate — ranking quality on the rare class", "mint"),
            ("Current-process recall", f"{proc_r*100:.1f}%", "Confirmed-fraud catch rate of today's manual review — the bar the pilot must clear", "gold"),
            ("Current-process precision", f"{proc_p*100:.0f}%", "Every 'Confirmed Fraud' review outcome was a later chargeback (0 false positives in that bucket)", "cyan"),
        ]
    )
    t1, t2, t3 = st.tabs(["Why linear won", "What the score sees", "Precision & recall"])
    with t1:
        st.markdown(
            """
            <div class="callout mint">
              <p><strong>Champion: Logistic Regression.</strong> It ties the top AUC (0.808) and leads PR-AUC (0.183).
              Probabilities go straight into the linear program with no second calibration step.
              Every non-linear family scored lower on both metrics, on the same time-respecting folds.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["TimeSeriesSplit, 5 folds", "No future event trains a past fold"],
                    ["Identical feature matrix", "Leakage columns and protected attributes never enter"],
                    ["AUC as the selection metric", "Ranking quality; logistic still ties the top"],
                    ["PR-AUC reported alongside", "More sensitive to the rare class under 28:1 imbalance; logistic leads this list too"],
                    ["Calibration needed by the LP", "Expected cost = p × loss — a bad probability is a bad policy"],
                    ["Linear SVM ties AUC", "Slightly lower PR-AUC (0.179 vs 0.183) and needs extra calibration to become a probability"],
                ],
                columns=["Design choice", "Why it is the right one for this case"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    with t2:
        st.markdown("Decision-time features only. Gender and nationality are absent on purpose.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Block": ["Identity / device"] * 3 + ["Session"] * 3 + ["Money"] * 3 + ["Context"] * 4,
                    "Feature": [
                        "Customer_Tenure_Months",
                        "Device_Known_Flag",
                        "New_Device_Age_Hours",
                        "IP_Distance_From_Usual_KM",
                        "Failed_Logins_24H",
                        "Password_Reset_48H",
                        "Transaction_Amount_EGP",
                        "Amount_vs_Customer_Avg",
                        "Shipping_Address_New_Flag",
                        "IP_dist_missing / Amt_ratio_missing",
                        "hour / dow",
                        "Account_Value_Band",
                        "Payment_Method",
                    ],
                    "Intuition": [
                        "Newer accounts run slightly hotter",
                        "Unknown device is a 4× flag",
                        "A freshly linked device is a cousin of unknown",
                        "Far-from-usual access",
                        "Failed logins climb 1.6% → 5.4%",
                        "Reset in 48h is a 2.7× flag",
                        "Fraud amounts sit slightly higher",
                        "Spend vs the customer's own mean is a 4× climb",
                        "New ship-to is a 2.7× flag",
                        "Missingness is weakly informative, so we keep it",
                        "Weak time signal, retained not relied on",
                        "Nearly flat — not a shortcut",
                        "Card hotter than COD, still not a rule",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with t3:
        ops = summary.get("champion_ops", {})
        f1p = ops.get("f1", {})
        holdp = ops.get("hold_cap", {})
        hr = ops.get("hold_review", {})
        st.markdown(
            f"""
            <div class="callout gold">
              <p><strong>From the notebook (§2 and §7b), made explicit.</strong> Today's process recall is {proc_r*100:.1f}% (885 of 1,204 confirmed fraud caught at review). Precision of the "Confirmed Fraud" outcome is {proc_p*100:.0f}% because that bucket is the chargeback-confirmed set. The champion is not a single threshold. Below: ranking quality (PR-AUC) and precision / recall at the capacity points the LP actually uses — same TimeSeriesSplit as the AUC table.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        kpi_row(
            [
                ("Champion PR-AUC", f"{ap:.3f}", "5-fold TimeSeriesSplit, same protocol as AUC in the notebook", "mint"),
                ("Hold-cap recall", f"{holdp.get('recall', 0)*100:.1f}%", "top 14% of scores — same scarcity as the 350/day Hold cap", "red"),
                ("Hold-cap precision", f"{holdp.get('precision', 0)*100:.1f}%", f"versus {base_rate*100:.2f}% base rate — about {holdp.get('precision', 0)/base_rate:.1f}× lift", "gold"),
                ("F1-optimal recall", f"{f1p.get('recall', 0)*100:.1f}%", f"precision {f1p.get('precision', 0)*100:.1f}% at threshold {f1p.get('threshold', 0):.3f}", "cyan"),
            ]
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["Current process (Confirmed Fraud outcome)", f"{proc_r*100:.1f}%", f"{proc_p*100:.0f}%", "Notebook §2 — catch rate of today's review"],
                    ["Champion @ Hold capacity (top 14%)", f"{holdp.get('recall', 0)*100:.1f}%", f"{holdp.get('precision', 0)*100:.1f}%", "Same scarcity as Temporary Hold (350 / 2,500)"],
                    ["Champion @ F1-optimal threshold", f"{f1p.get('recall', 0)*100:.1f}%", f"{f1p.get('precision', 0)*100:.1f}%", "Last time-fold, threshold chosen to maximise F1"],
                    ["Champion @ Hold + Review capacity (top 86%)", f"{hr.get('recall', 0)*100:.1f}%", f"{hr.get('precision', 0)*100:.1f}%", "Almost all fraud flagged; precision near base rate"],
                ],
                columns=["Operating point", "Recall", "Precision", "How to read it"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        pr_path = DATA / "pr_curve.json"
        if pr_path.exists():
            pr = json.loads(pr_path.read_text())
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pr["recall"],
                    y=pr["precision"],
                    mode="lines",
                    name="Champion PR curve",
                    line=dict(color=CYAN, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(45,212,232,0.12)",
                )
            )
            fig.add_hline(
                y=pr.get("base_rate", base_rate),
                line_dash="dot",
                line_color=GOLD,
                annotation_text="base rate",
                annotation_font_color=GOLD,
                annotation_position="right",
            )
            if holdp:
                fig.add_trace(
                    go.Scatter(
                        x=[holdp.get("recall")],
                        y=[holdp.get("precision")],
                        mode="markers+text",
                        name="Hold cap",
                        marker=dict(size=12, color=RED),
                        text=["Hold cap"],
                        textposition="top center",
                    )
                )
            if f1p:
                fig.add_trace(
                    go.Scatter(
                        x=[f1p.get("recall")],
                        y=[f1p.get("precision")],
                        mode="markers+text",
                        name="F1-optimal",
                        marker=dict(size=12, color=GOLD),
                        text=["F1-optimal"],
                        textposition="top right",
                    )
                )
            st.plotly_chart(
                style_fig(
                    fig,
                    height=460,
                    title="Precision–recall curve (last time fold)",
                    xlabel="Recall",
                    ylabel="Precision",
                ),
                use_container_width=True,
                config=PLOT,
            )
        st.markdown(
            f"""
            <div class="callout mint">
              <p><strong>How to say this in a defense.</strong> Quote AUC 0.808 for ranking and PR-AUC {ap:.3f} for the rare-class ranking ({ap/base_rate:.1f}× a random list). Quote process recall {proc_r*100:.1f}% as the pilot bar. Quote Hold-cap precision {holdp.get('precision', 0)*100:.1f}% / recall {holdp.get('recall', 0)*100:.1f}% when asked what the scarce 350 holds actually catch. The LP does not pick one threshold — it assigns four actions jointly — so these operating points are the capacity-matched reading of the same score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Decision Policy ───────────────────────────────────────────────────
def page_policy():
    heading(
        "Optimization",
        "A linear program, not a threshold guess",
        "2,500 events × 4 actions, solved jointly for minimum expected cost under both capacity caps at once.",
    )
    kpi_row(
        [
            ("Expected loss reduction", f"{summary['mc_mean_pct']:.1f}%", "mean of 20,000 Monte Carlo runs", "mint"),
            ("90% interval", f"{summary['mc_p5_pct']:.1f} – {summary['mc_p95_pct']:.1f}%", "deterrence rate is Uniform(20–80%)", "gold"),
            ("Worst-case floor", f"{summary['floor_pct']:.1f}%", "even if step-up deters zero fraud", "red"),
            ("Hold cap used", f"{hold_used} / {HOLD_CAP}", "the binding constraint — review is not", "cyan"),
        ]
    )
    t1, t2 = st.tabs(["Economics", "Where the policy landed"])

    with t1:
        c1, c2 = st.columns([1.35, 1])
        with c1:
            pc = {
                "Do nothing<br>(approve all)": summary["do_nothing_cost"],
                "Naive 2-bucket<br>(no step-up)": summary["naive2_cost"],
                "LP-optimal<br>(4 actions)": summary["lp_optimal_cost"],
            }
            fig = go.Figure(
                go.Bar(
                    x=list(pc.keys()),
                    y=list(pc.values()),
                    marker_color=[MUTED, GOLD, CYAN],
                    text=[f"{v:,.0f}" for v in pc.values()],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=400, title="Total expected cost across 2,500 events (EGP)", ylabel="EGP"),
                use_container_width=True,
                config=PLOT,
            )
        with c2:
            st.markdown("##### Unit economics the LP sees")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Missed fraud (approve a fraudster)", f"{MISS:,} EGP", "p × 8,000"],
                        ["Manual review handling", f"{REVIEW_COST} EGP", "plus residual miss × 8,000"],
                        ["Step-up authentication", f"{STEP_COST} EGP", "plus (1 − deterrence) × p × 8,000"],
                        ["False hold (hold a legitimate)", f"{FALSEHOLD} EGP", "(1 − p) × 220"],
                        ["Step-up deterrence (point)", "50%", "stressed as Uniform 20–80% in simulation"],
                    ],
                    columns=["Cost component", "Value", "How it enters the LP"],
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                """
                <div class="callout gold">
                  <p>The naive policy dumps the top 350 scores into Hold and the next 1,800 into Review.
                  It never uses step-up. The LP finds that most of the middle of the distribution is cheaper
                  to challenge with an 8 EGP step-up than to spend a reviewer on.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t2:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = go.Figure(
                go.Bar(
                    x=[wrap_action(o) for o in ACTIONS],
                    y=[ac.get(o, 0) for o in ACTIONS],
                    marker_color=[ACTION_COLOR[o] for o in ACTIONS],
                    text=[ac.get(o, 0) for o in ACTIONS],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=420, title="Assigned actions on the current file", ylabel="Events"),
                use_container_width=True,
                config=PLOT,
            )
        with c2:
            stats = (
                scored.groupby("action", observed=True)
                .agg(
                    events=("Event_ID", "count"),
                    mean_p=("fraud_prob", "mean"),
                    median_p=("fraud_prob", "median"),
                    amount=("Transaction_Amount_EGP", "sum"),
                )
                .reindex(ACTIONS)
                .reset_index()
            )
            stats["mean_p"] = (stats["mean_p"] * 100).map(lambda x: f"{x:.2f}%")
            stats["median_p"] = (stats["median_p"] * 100).map(lambda x: f"{x:.2f}%")
            stats["amount"] = stats["amount"].map(lambda x: f"{x:,.0f}")
            stats.columns = ["Action", "Events", "Mean risk", "Median risk", "Amount (EGP)"]
            st.dataframe(stats, use_container_width=True, hide_index=True)
            cap_bar("Temporary Hold", hold_used, HOLD_CAP, RED)
            cap_bar("Manual Review", review_used, REVIEW_CAP, AMBER)

        by_band = pd.crosstab(scored["Risk_Band"], scored["action"]).reindex(index=BANDS, columns=ACTIONS).fillna(0)
        fig = go.Figure(
            go.Heatmap(
                z=by_band.values,
                x=by_band.columns,
                y=by_band.index,
                colorscale=[[0, "#0E1A22"], [1, CYAN]],
                text=by_band.values.astype(int),
                texttemplate="%{text}",
                colorbar=dict(title="Events"),
            )
        )
        st.plotly_chart(style_fig(fig, height=360, title="Risk band × recommended action"), use_container_width=True, config=PLOT)

        fig = go.Figure()
        for act in ACTIONS:
            sl = scored.loc[scored["action"] == act, "fraud_prob"]
            fig.add_trace(go.Box(y=sl, name=act, marker_color=ACTION_COLOR[act], boxmean=True))
        st.plotly_chart(
            style_fig(fig, height=360, title="Score distribution by assigned action", ylabel="Fraud probability"),
            use_container_width=True,
            config=PLOT,
        )

    st.markdown(
        f"""
        <div class="callout cyan">
          <p><strong>Uncertainty is not a footnote.</strong> Step-up deterrence is a stated assumption, not a measured rate. Open the <strong>Monte Carlo</strong> section for the 20,000-run distribution, the 90% interval ({summary['mc_p5_pct']:.1f}% – {summary['mc_p95_pct']:.1f}%), and the zero-deterrence floor of {summary['floor_pct']:.1f}%.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Monte Carlo ──────────────────────────────────────────────────────
def page_mc():
    n_runs = int(len(mc))
    mc_std = float(np.std(mc))
    mc_min, mc_max = float(np.min(mc)), float(np.max(mc))
    p1, p10, p25, p50, p75, p90, p99 = (float(np.percentile(mc, q)) for q in (1, 10, 25, 50, 75, 90, 99))
    share_ge_60 = float((mc >= 60).mean() * 100)
    share_ge_65 = float((mc >= 65).mean() * 100)
    share_ge_70 = float((mc >= 70).mean() * 100)

    heading(
        "Uncertainty",
        "Monte Carlo — 20,000 draws on a stated assumption",
        "Step-up deterrence is not in the data. We draw it Uniform(20–80%) and recompute expected loss each time.",
    )
    kpi_row(
        [
            ("Mean loss prevented", f"{summary['mc_mean_pct']:.1f}%", f"across {n_runs:,} simulated draws", "mint"),
            ("90% interval", f"{summary['mc_p5_pct']:.1f} – {summary['mc_p95_pct']:.1f}%", "5th to 95th percentile of the distribution", "gold"),
            ("Zero-deterrence floor", f"{summary['floor_pct']:.1f}%", "Hold and Review still earn their keep if step-up deters nothing", "red"),
            ("Spread (std. dev.)", f"{mc_std:.1f} pp", f"range {mc_min:.1f}% – {mc_max:.1f}%", "cyan"),
        ]
    )

    st.markdown(
        f"""
        <div class="callout gold">
          <p><strong>What is being simulated.</strong> The LP is solved once, at a 50% point estimate for step-up deterrence. Monte Carlo does not re-solve the LP. It keeps the assigned actions fixed and redraws deterrence from Uniform(20%, 80%) for the {step_used:,} step-up events, then recomputes expected cost. Review catch rate stays at {catch_rate * 100:.1f}%. The result is a distribution of loss prevented versus do-nothing, not a distribution of policies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc, nbinsx=64, marker_color=CYAN, opacity=0.85, name="runs"))
        fig.add_vline(
            x=summary["mc_mean_pct"],
            line_color=RED,
            line_width=2,
            annotation_text="mean",
            annotation_font_color=RED,
            annotation_position="top",
        )
        fig.add_vrect(
            x0=summary["mc_p5_pct"],
            x1=summary["mc_p95_pct"],
            fillcolor=GOLD,
            opacity=0.14,
            line_width=0,
            annotation_text="90% interval",
            annotation_font_color=GOLD,
            annotation_position="top left",
        )
        fig.add_vline(
            x=summary["floor_pct"],
            line_dash="dot",
            line_color=MUTED,
            annotation_text="floor",
            annotation_font_color=MUTED,
            annotation_position="bottom",
        )
        st.plotly_chart(
            style_fig(
                fig,
                height=480,
                title="Distribution of loss prevented (%)",
                xlabel="Loss prevented (%)",
                ylabel="Simulations",
            ),
            use_container_width=True,
            config=PLOT,
        )
    with c2:
        xs = np.sort(mc)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        fig = go.Figure(
            go.Scatter(
                x=xs,
                y=cdf * 100,
                mode="lines",
                line=dict(color=CYAN, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(45,212,232,0.12)",
                name="empirical CDF",
            )
        )
        fig.add_vline(x=summary["mc_mean_pct"], line_color=RED, line_width=1.5, line_dash="dot")
        fig.add_hline(
            y=5,
            line_dash="dot",
            line_color=GOLD,
            annotation_text="5th pct",
            annotation_font_color=GOLD,
            annotation_position="right",
        )
        fig.add_hline(
            y=95,
            line_dash="dot",
            line_color=GOLD,
            annotation_text="95th pct",
            annotation_font_color=GOLD,
            annotation_position="right",
        )
        st.plotly_chart(
            style_fig(
                fig,
                height=480,
                title="Empirical CDF",
                xlabel="Loss prevented (%)",
                ylabel="Share of runs at or below (%)",
            ),
            use_container_width=True,
            config=PLOT,
        )

    st.markdown("##### Percentiles of loss prevented")
    st.dataframe(
        pd.DataFrame(
            {
                "Percentile": ["1st", "5th", "10th", "25th", "50th (median)", "75th", "90th", "95th", "99th"],
                "Loss prevented": [
                    f"{p1:.1f}%",
                    f"{summary['mc_p5_pct']:.1f}%",
                    f"{p10:.1f}%",
                    f"{p25:.1f}%",
                    f"{p50:.1f}%",
                    f"{p75:.1f}%",
                    f"{p90:.1f}%",
                    f"{summary['mc_p95_pct']:.1f}%",
                    f"{p99:.1f}%",
                ],
                "How to read it": [
                    "Almost the worst draw in 20,000",
                    "Lower bound of the 90% interval quoted in the deck",
                    "Still well above the zero-deterrence floor",
                    "Lower quartile — a conservative working number",
                    "Typical simulated outcome",
                    "Upper quartile",
                    "Optimistic but not the tail",
                    "Upper bound of the 90% interval quoted in the deck",
                    "Almost the best draw in 20,000",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    kpi_row(
        [
            ("Share of runs ≥ 60%", f"{share_ge_60:.0f}%", "of 20,000 draws", "cyan"),
            ("Share of runs ≥ 65%", f"{share_ge_65:.0f}%", "near the mean outcome", "mint"),
            ("Share of runs ≥ 70%", f"{share_ge_70:.0f}%", "upper half of the deterrence range", "gold"),
            ("Zero-deterrence floor", f"{summary['floor_pct']:.1f}%", "not a simulated draw — a hard lower bound", "red"),
        ]
    )

    st.markdown(
        f"""
        <div class="callout mint">
          <p><strong>How to say this in a defense.</strong> Quote the mean of {summary['mc_mean_pct']:.1f}% as the expected reduction. Quote {summary['mc_p5_pct']:.1f}% – {summary['mc_p95_pct']:.1f}% as the 90% interval. If someone asks “what if step-up deters nobody?”, the floor is still {summary['floor_pct']:.1f}% because Hold and Review do not depend on that assumption. Do not quote a single simulated draw as the answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Assumptions, held fixed"):
        st.dataframe(
            pd.DataFrame(
                [
                    ["Step-up deterrence", "Uniform(20%, 80%)", "Stated assumption — the only random input"],
                    ["Point used in the LP", "50%", "LP is solved once; Monte Carlo does not re-allocate"],
                    ["Missed-fraud cost", f"{MISS:,} EGP", "Held fixed"],
                    ["Review handling cost", f"{REVIEW_COST} EGP", "Held fixed"],
                    ["Step-up handling cost", f"{STEP_COST} EGP", "Held fixed"],
                    ["False-hold cost", f"{FALSEHOLD} EGP", "Held fixed"],
                    ["Review catch rate", f"{catch_rate * 100:.1f}%", "Historical catch rate, held fixed"],
                    ["Assigned actions", f"{n_ops:,} events, frozen", "Approve / step-up / review / hold counts do not change across draws"],
                ],
                columns=["Input", "Value", "Treatment in the simulation"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    heading(
        "Sandbox",
        "Live Monte Carlo — change the assumption and rerun",
        "The recorded result above stays frozen. This sandbox redraws deterrence on the same frozen actions so you can stress the assumption live.",
    )

    c1, c2, c3 = st.columns(3)
    det_lo_pct, det_hi_pct = c1.slider(
        "Step-up deterrence range (%)",
        min_value=0,
        max_value=100,
        value=(20, 80),
        step=5,
        help="Recorded result uses Uniform(20%, 80%).",
    )
    n_live = c2.select_slider("Number of draws", options=[1_000, 2_000, 5_000, 10_000, 20_000], value=5_000)
    catch_live = c3.slider(
        "Review catch rate (%)",
        min_value=40,
        max_value=95,
        value=int(round(catch_rate * 100)),
        step=1,
        help=f"Recorded result uses the historical catch rate of {catch_rate * 100:.1f}%.",
    )

    miss_live, review_live, step_live, hold_live, seed_live = int(MISS), int(REVIEW_COST), int(STEP_COST), int(FALSEHOLD), 42
    with st.expander("Cost knobs (optional)"):
        k1, k2, k3, k4 = st.columns(4)
        miss_live = k1.number_input("Missed-fraud cost (EGP)", min_value=1000, max_value=20000, value=int(MISS), step=500)
        review_live = k2.number_input("Review cost (EGP)", min_value=0, max_value=500, value=int(REVIEW_COST), step=10)
        step_live = k3.number_input("Step-up cost (EGP)", min_value=0, max_value=200, value=int(STEP_COST), step=1)
        hold_live = k4.number_input("False-hold cost (EGP)", min_value=0, max_value=1000, value=int(FALSEHOLD), step=10)
        seed_live = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if det_hi_pct <= det_lo_pct:
        st.warning("Raise the top of the deterrence range above the bottom so the draw has a range to sample.")
        return

    overlay = st.checkbox("Overlay the recorded 20,000-run distribution", value=True)

    live_pct, live_floor, _ = simulate_mc(
        scored["fraud_prob"].to_numpy(),
        scored["action"].to_numpy(),
        n_live,
        det_lo_pct / 100.0,
        det_hi_pct / 100.0,
        seed_live,
        miss_live,
        review_live,
        step_live,
        hold_live,
        catch_live / 100.0,
    )

    live_mean = float(np.mean(live_pct))
    live_p5, live_p95 = float(np.percentile(live_pct, 5)), float(np.percentile(live_pct, 95))
    live_std = float(np.std(live_pct))

    kpi_row(
        [
            ("Live mean", f"{live_mean:.1f}%", f"recorded mean {summary['mc_mean_pct']:.1f}%", "mint"),
            ("Live 90% interval", f"{live_p5:.1f} – {live_p95:.1f}%", f"recorded {summary['mc_p5_pct']:.1f} – {summary['mc_p95_pct']:.1f}%", "gold"),
            ("Live floor", f"{live_floor:.1f}%", f"recorded floor {summary['floor_pct']:.1f}%", "red"),
            ("Live spread", f"{live_std:.1f} pp", f"{n_live:,} draws · Uniform({det_lo_pct:.0f}–{det_hi_pct:.0f}%)", "cyan"),
        ]
    )

    fig = go.Figure()
    if overlay:
        fig.add_trace(
            go.Histogram(
                x=mc,
                nbinsx=50,
                name="Recorded 20,000",
                marker_color=MUTED,
                opacity=0.45,
                histnorm="probability density",
            )
        )
    fig.add_trace(
        go.Histogram(
            x=live_pct,
            nbinsx=50,
            name="Live sandbox",
            marker_color=CYAN,
            opacity=0.75,
            histnorm="probability density",
        )
    )
    fig.add_vline(x=live_mean, line_color=RED, line_width=2, annotation_text="live mean", annotation_font_color=RED, annotation_position="top")
    fig.add_vrect(
        x0=live_p5,
        x1=live_p95,
        fillcolor=GOLD,
        opacity=0.12,
        line_width=0,
        annotation_text="live 90%",
        annotation_font_color=GOLD,
        annotation_position="top left",
    )
    fig.add_vline(
        x=live_floor,
        line_dash="dot",
        line_color=MUTED,
        annotation_text="live floor",
        annotation_font_color=MUTED,
        annotation_position="bottom",
    )
    fig.update_layout(barmode="overlay")
    st.plotly_chart(
        style_fig(
            fig,
            height=460,
            title="Live distribution of loss prevented (%)",
            xlabel="Loss prevented (%)",
            ylabel="Density",
        ),
        use_container_width=True,
        config=PLOT,
    )

    delta = live_mean - float(summary["mc_mean_pct"])
    st.markdown(
        f"""
        <div class="callout cyan">
          <p><strong>Reading the sandbox.</strong> Actions stay frozen. You only changed the uncertainty around step-up, review catch, and unit costs. Live mean is {live_mean:.1f}%, which is {abs(delta):.1f} percentage points {"above" if delta >= 0 else "below"} the recorded mean of {summary["mc_mean_pct"]:.1f}%. Quote the recorded result in the deck. Use this panel to answer “what if deterrence is weaker / catch rate is worse?”</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Live Operations ──────────────────────────────────────────────────
def page_ops():
    heading(
        "Queue",
        "Live operational decision output",
        "All 2,500 scored events — filter the queue, open a dossier, inspect nearest historical fraud, download the slice.",
    )
    f1, f2, f3, f4 = st.columns([1.1, 1.3, 0.9, 1.1])
    band_opts = [b for b in BANDS if b in set(scored["Risk_Band"].dropna().astype(str))]
    band_filter = f1.multiselect("Risk band", band_opts)
    action_filter = f2.multiselect("Recommended action", ACTIONS)
    human_col = "Human_Review_Required" if "Human_Review_Required" in scored.columns else None
    human_filter = f3.selectbox("Human review", ["All", "Yes", "No"] if human_col else ["All"])
    search_id = f4.text_input("Search Event_ID", placeholder="EVT-CUR-…")

    df = scored.copy()
    if band_filter:
        df = df[df["Risk_Band"].isin(band_filter)]
    if action_filter:
        df = df[df["action"].isin(action_filter)]
    if human_col and human_filter != "All":
        df = df[df[human_col] == human_filter]
    if search_id:
        df = df[df["Event_ID"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
    df = df.sort_values("Priority_Rank").reset_index(drop=True)

    human_n = int((df[human_col] == "Yes").sum()) if human_col and len(df) else 0
    kpi_row(
        [
            ("Matching events", f"{len(df):,}", f"of {n_ops:,} in the ops file", "cyan"),
            (
                "Mean fraud risk",
                f"{df['fraud_prob'].mean()*100:.2f}%" if len(df) else "—",
                f"max {df['fraud_prob'].max()*100:.1f}%" if len(df) else "",
                "red",
            ),
            ("Amount in slice", f"{df['Transaction_Amount_EGP'].sum()/1e6:.2f}M" if len(df) else "—", "EGP", "gold"),
            ("Human review in slice", f"{human_n:,}", "Hold + Review", "amber"),
        ]
    )

    c1, c2 = st.columns([1.35, 1])
    with c1:
        mix = df["action"].value_counts().reindex(ACTIONS).fillna(0)
        fig = go.Figure(
            go.Bar(
                x=[wrap_action(a) for a in ACTIONS],
                y=mix.values,
                marker_color=[ACTION_COLOR[a] for a in ACTIONS],
                text=mix.values.astype(int),
                textposition="outside",
            )
        )
        st.plotly_chart(style_fig(fig, height=340, title="Actions in this slice"), use_container_width=True, config=PLOT)
    with c2:
        if "Reason_Code_1" in df.columns and len(df):
            rc = pd.concat([df["Reason_Code_1"], df["Reason_Code_2"]]).value_counts().head(8)
            fig = go.Figure(
                go.Bar(
                    y=rc.index[::-1],
                    x=rc.values[::-1],
                    orientation="h",
                    marker_color=CYAN,
                    text=rc.values[::-1],
                    textposition="outside",
                )
            )
            st.plotly_chart(
                style_fig(fig, height=380, title="Top reason codes in this slice", xlabel="Mentions"),
                use_container_width=True,
                config=PLOT,
            )

    display_cols = [
        c
        for c in [
            "Priority_Rank",
            "Event_ID",
            "fraud_prob",
            "Risk_Band",
            "action",
            "Transaction_Amount_EGP",
            "Payment_Method",
            "Reason_Code_1",
            "Reason_Code_2",
        ]
        if c in df.columns
    ]
    st.markdown("**Queue** — click a row to open the dossier. Sorted by priority rank.")
    state = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
        key="ops_grid",
        column_config={
            "Priority_Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "fraud_prob": st.column_config.ProgressColumn(
                "Fraud risk",
                min_value=0,
                max_value=float(scored["fraud_prob"].max()),
                format="%.1%",
            ),
            "Transaction_Amount_EGP": st.column_config.NumberColumn("Amount (EGP)", format="%.0f"),
            "action": st.column_config.TextColumn("Recommended action"),
        },
    )
    rows = []
    if state is not None:
        sel = getattr(state, "selection", None)
        rows = list(getattr(sel, "rows", []) or []) if sel is not None else []
        if not rows and isinstance(state, dict):
            rows = state.get("selection", {}).get("rows", [])

    if not len(df):
        st.info("No events match these filters.")
        return

    pick = df.iloc[rows[0]] if rows else df.iloc[0]
    tone = {
        "Temporary Hold": "red",
        "Manual Review": "amber",
        "Step-up Authentication": "gold",
        "Approve": "mint",
    }.get(pick["action"], "cyan")

    st.markdown("---")
    st.markdown(f"### Dossier · `{pick['Event_ID']}`")
    owner = str(pick["Action_Owner"]) if "Action_Owner" in pick.index and pd.notna(pick.get("Action_Owner")) else "—"
    kpi_row(
        [
            ("Fraud risk", f"{pick['fraud_prob']*100:.2f}%", str(pick.get("Risk_Band", "")), "red"),
            ("Action", pick["action"], owner[:48], tone),
            (
                "Amount",
                f"{pick['Transaction_Amount_EGP']:,.0f} EGP",
                f"{pick['Payment_Method']} · {pick['Account_Value_Band']} value",
                "gold",
            ),
            (
                "Priority rank",
                f"#{int(pick.get('Priority_Rank', 0))}",
                f"{pick.get('Human_Review_Required', '—')} human review",
                "cyan",
            ),
        ]
    )

    ip = pick.get("IP_Distance_From_Usual_KM")
    amt_vs = pick.get("Amount_vs_Customer_Avg")
    ip_s = f"{ip:.0f} km" if pd.notna(ip) else "—"
    amt_s = f"{amt_vs:.2f}×" if pd.notna(amt_vs) else "—"
    when = pd.to_datetime(pick["Event_Date_Time"]).strftime("%d %b %Y %H:%M")
    r1 = pick.get("Reason_Code_1", "—")
    r2 = pick.get("Reason_Code_2", "—")
    st.markdown(
        f"""
        <div class="panel">
          <h3>Analyst-facing reason codes</h3>
          <p><span class="pill red">{r1}</span> <span class="pill gold">{r2}</span></p>
          <p style="margin-top:10px;">{flags(pick)}</p>
          <p>Tenure {pick['Customer_Tenure_Months']:.0f} months · failed logins {pick['Failed_Logins_24H']:.0f} · IP distance {ip_s} · amount vs customer avg {amt_s} · {when}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    costs = {
        "Approve": pick["cost_approve"],
        "Step-up Authentication": pick["cost_stepup"],
        "Manual Review": pick["cost_review"],
        "Temporary Hold": pick["cost_hold"],
    }
    fig = go.Figure(
        go.Bar(
            x=list(costs.keys()),
            y=list(costs.values()),
            marker_color=[ACTION_COLOR[a] if a == pick["action"] else "#2A3A42" for a in costs],
            text=[f"{v:,.0f}" for v in costs.values()],
            textposition="outside",
        )
    )
    st.plotly_chart(
        style_fig(
            fig,
            height=360,
            title="Expected cost of each action on this event (EGP)<br><sup>highlighted bar = assigned</sup>",
            ylabel="EGP",
        ),
        use_container_width=True,
        config=PLOT,
    )

    st.markdown("##### Nearest historical confirmed-fraud neighbors")
    st.caption(
        "Euclidean distance on the same nine numeric features the notebook uses. Precedent an analyst can inspect, not just a probability."
    )
    try:
        nbrs = nearest_fraud(pick, hist, k=3)
        show_n = nbrs.copy()
        show_n["Device"] = show_n["Device_Known_Flag"].map({0: "Unknown", 1: "Known"})
        show_n["Reset 48h"] = show_n["Password_Reset_48H"].map({0: "No", 1: "Yes"})
        show_n["New ship-to"] = show_n["Shipping_Address_New_Flag"].map({0: "No", 1: "Yes"})
        st.dataframe(
            show_n.rename(
                columns={
                    "distance": "Distance",
                    "Transaction_Amount_EGP": "Amount (EGP)",
                    "Customer_Tenure_Months": "Tenure (mo)",
                }
            )[["Event_ID", "Distance", "Amount (EGP)", "Tenure (mo)", "Device", "Reset 48h", "New ship-to"]],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.caption(f"Neighbor lookup unavailable for this row ({exc}).")

    st.download_button(
        "Download this slice as CSV",
        df.to_csv(index=False).encode("utf-8"),
        "cascadiq_filtered_decisions.csv",
        "text/csv",
    )


# ── Responsible AI ─────────────────────────────────────────────────
def page_rai():
    heading(
        "Oversight",
        "Equal allocation, unequal protection — the finding that matters",
        "Hold-tier share is nearly identical across groups. Confirmed-fraud recall inside that tier is not.",
    )
    st.markdown("##### Fairness check 1 — allocation share on the current 2,500-event file (notebook §14)")
    st.caption("Audit-only cut. Gender and nationality never enter the model or the reason codes.")
    if {"Customer_Gender", "action"}.issubset(scored.columns):
        a1, a2 = st.columns(2)
        with a1:
            g = pd.crosstab(scored["Customer_Gender"], scored["action"], normalize="index") * 100
            g = g.reindex(columns=[c for c in ACTIONS if c in g.columns]).round(1)
            st.dataframe(g, use_container_width=True)
        with a2:
            if "Nationality_Group" in scored.columns:
                ntab = pd.crosstab(scored["Nationality_Group"], scored["action"], normalize="index") * 100
                ntab = ntab.reindex(columns=[c for c in ACTIONS if c in ntab.columns]).round(1)
                st.dataframe(ntab, use_container_width=True)
    st.markdown("##### Fairness check 2 — Hold-tier recall on history")
    for col_name, title in [
        ("Nationality_Group", "Nationality — allocation is fair; catch-rate inside Hold is not"),
        ("Customer_Gender", "Gender — the same pattern, smaller gap"),
    ]:
        rows = summary["fairness"][col_name]
        groups = [r["group"] for r in rows]
        recalls = [r["fraud_recall_in_hold_tier_pct"] for r in rows]
        alloc = [r["pct_in_hold_tier"] for r in rows]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Share placed in Hold",
                x=groups,
                y=alloc,
                marker_color=CYAN,
                text=[f"{v:.1f}%" for v in alloc],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Confirmed-fraud recall in Hold",
                x=groups,
                y=recalls,
                marker_color=RED,
                text=[f"{v:.1f}%" for v in recalls],
                textposition="outside",
            )
        )
        fig.update_layout(barmode="group")
        st.plotly_chart(style_fig(fig, height=460, title=title, ylabel="%"), use_container_width=True, config=PLOT)
        st.dataframe(
            pd.DataFrame(
                {
                    "Group": groups,
                    "Events": [r["n"] for r in rows],
                    "Confirmed fraud": [r["fraud_events"] for r in rows],
                    "Hold allocation %": alloc,
                    "Hold recall %": recalls,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    nat = {r["group"]: r["fraud_recall_in_hold_tier_pct"] for r in summary["fairness"]["Nationality_Group"]}
    gap = max(nat.values()) - min(nat.values())
    st.markdown(
        f"""
        <div class="callout red">
          <p><strong>The gap is {gap:.1f} points of recall, not of allocation.</strong> Egyptian confirmed-fraud in the Hold tier is caught at {nat.get('Egyptian', 0):.1f}%; Regional at {nat.get('Regional', 0):.1f}%. Who gets held is almost the same (~14%). Whether the hold actually catches confirmed fraud is not.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="panel">
              <h3>What we did</h3>
              <p>Gender and nationality stay out of the model and out of every reason code.
              No demographic threshold. The gap is flagged for the pilot with a named owner,
              investigated as a feature-coverage question — not "fixed" by a group-based rule
              the case forbids.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="panel">
              <h3>Privacy</h3>
              <p>Reason codes and case evidence are analyst-facing only, never shown to customers.
              No new personal data is collected beyond what checkout already gathers.
              Protected attributes are audit-only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("##### Pilot monitoring plan")
    st.dataframe(
        pd.DataFrame(
            [
                ["Recall vs 73.5% historical bar", "Weekly", "Fraud Operations lead"],
                ["Review ≤ 1,800 / Hold ≤ 350", "Daily", "Capacity owner on-call"],
                ["Nationality Hold-recall gap", "Weekly", "Responsible-AI owner (named)"],
                ["Gender Hold-recall gap", "Weekly", "Responsible-AI owner (named)"],
                ["Shadow vs live disagreement sample", "Weekly", "QA analyst"],
                ["Go / no-go for live actions", "Week 4", "Head of Fraud Operations"],
            ],
            columns=["Metric", "Cadence", "Owner"],
        ),
        use_container_width=True,
        hide_index=True,
    )


{
    "Command Center": page_home,
    "Data & Quality": page_data,
    "Risk Intelligence": page_risk,
    "Model Lab": page_model,
    "Decision Policy": page_policy,
    "Monte Carlo": page_mc,
    "Live Operations": page_ops,
    "Responsible AI": page_rai,
}[page]()
