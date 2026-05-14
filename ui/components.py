"""
ui/components.py — Reusable HTML/Streamlit UI components.

Extracted from streamlit_app.py (Fix #3: split monolith).
Provides card(), alert(), badge(), metrics_row(), b64_image().
"""
from __future__ import annotations
from typing import Any
import streamlit as st


def card(content: str, accent: bool = False):
    cls = "aml-card aml-card-accent" if accent else "aml-card"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)


def alert(msg: str, kind: str = "info"):
    st.markdown(f'<div class="aml-alert aml-alert-{kind}">{msg}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "phase") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def metrics_row(items: list[tuple[str, Any]], accent: bool = False):
    """items = [(label, value), ...]"""
    cls_extra = "" if accent else " aml-metric-plain"
    tiles = "".join(
        f'<div class="aml-metric{cls_extra}"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for l, v in items
    )
    st.markdown(f'<div class="aml-metrics">{tiles}</div>', unsafe_allow_html=True)


def b64_image(b64: str, caption: str = ""):
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:10px;" alt="{caption}"/>',
        unsafe_allow_html=True,
    )


APP_CSS = """
<style>
/* ── Global backgrounds — lifted from near-black to dark-navy-gray ─────── */
[data-testid="stAppViewContainer"] { background: #16172a; color: #e8e8f5; }
[data-testid="stSidebar"]          { background: #1c1d32; border-right: 1px solid #32335a; }
[data-testid="stSidebar"] *        { color: #d8d8f0 !important; }

/* ── Typography — all text brighter and easier to read ─────────────────── */
h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #f0f0ff !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #dcdcfa !important; }
h3 { font-size: 1rem  !important; font-weight: 600 !important; color: #c8c8ee !important; }
p, li    { color: #c2c2e0; line-height: 1.7; }
label    { color: #c2c2e0 !important; }
caption  { color: #9898c0 !important; }
small    { color: #9898c0 !important; }
code     { color: #a5f3a5 !important; background: #1a2e1a !important; }

/* ── Cards ──────────────────────────────────────────────────────────────── */
.aml-card        { background: #1f2040; border: 1px solid #35366a; border-radius: 12px; padding: 1.1rem 1.25rem; margin-bottom: 0.75rem; }
.aml-card-accent { border-color: #8b5cf6 !important; }

/* ── Metric tiles ────────────────────────────────────────────────────────── */
.aml-metrics { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
.aml-metric  { background: #1f2040; border: 1px solid #35366a; border-radius: 10px; padding: 0.65rem 1rem; min-width: 110px; text-align: center; }
.aml-metric .val         { font-size: 1.3rem; font-weight: 700; color: #7ec8fa; }
.aml-metric .lbl         { font-size: 0.68rem; color: #9898c0; text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }
.aml-metric-plain .val   { color: #e8e8f5; }

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge        { display: inline-block; font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; letter-spacing: .04em; text-transform: uppercase; }
.badge-phase  { background: #2a2550; color: #c4b5fd; border: 1px solid #6d5abf; }
.badge-green  { background: #0d2e18; color: #6ee7a0; border: 1px solid #1a6634; }
.badge-red    { background: #2e1010; color: #fca5a5; border: 1px solid #7f1d1d; }
.badge-amber  { background: #2e1e08; color: #fcd34d; border: 1px solid #92400e; }
.badge-blue   { background: #0e1e38; color: #7ec8fa; border: 1px solid #1e4a7f; }

/* ── Alerts ─────────────────────────────────────────────────────────────── */
.aml-alert             { border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; font-size: 0.875rem; border-left: 3px solid; }
.aml-alert-info        { background: #0e2038; border-color: #3b82f6; color: #bfdbfe; }
.aml-alert-warning     { background: #2e1e08; border-color: #f59e0b; color: #fde68a; }
.aml-alert-success     { background: #062812; border-color: #22c55e; color: #a7f3c0; }
.aml-alert-error       { background: #2e0e18; border-color: #ef4444; color: #fecaca; }

/* ── Feature rows ────────────────────────────────────────────────────────── */
.feat-row     { background: #1f2040; border: 1px solid #35366a; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
.feat-formula { font-family: monospace; font-size: 0.78rem; color: #6ee7a0; background: #0c1a0c; border-radius: 6px; padding: 3px 8px; border: 1px solid #1a3a20; display: inline-block; margin: 4px 0; }

/* ── SHAP bar chart ──────────────────────────────────────────────────────── */
.shap-row      { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.shap-label    { font-family:monospace; font-size:.75rem; color:#d0d0ee; width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.shap-bar-wrap { flex:1; background:#2a2b50; border-radius:4px; height:10px; }
.shap-bar-fill { height:10px; border-radius:4px; }
.shap-pct      { font-size:.7rem; color:#9898c0; width:42px; text-align:right; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button          { background: linear-gradient(135deg, #5b5ef5, #3b82f6) !important; color: #ffffff !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 1.2rem !important; }
.stButton > button:hover    { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; }
.stButton > button:disabled { background: #2a2b45 !important; color: #6060a0 !important; }

/* ── Form inputs ─────────────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div > input {
  background: #1f2040 !important; border-color: #35366a !important; color: #e8e8f5 !important;
}

/* ── Streamlit native text overrides ─────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p   { color: #c2c2e0; }
[data-testid="stMarkdownContainer"] li  { color: #c2c2e0; }
[data-testid="stCaptionContainer"]      { color: #9898c0 !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Expander headers ────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary      { color: #d0d0f0 !important; font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: #f0f0ff !important; }

/* ── Radio / checkbox labels ─────────────────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { color: #c8c8ee !important; }

/* ── Slider value ────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #9898c0 !important; }

/* ── Metric widget ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] label          { color: #9898c0 !important; font-size: 0.72rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8e8f5 !important; }
</style>
"""
