"""
ui/pages/distributions.py — Feature Distribution Explorer

A dedicated page for exploring all feature distributions:
  - Summary metrics (numeric vs categorical, skew overview)
  - Numeric columns: histograms + KDE + skewness stats table
  - Categorical columns: bar charts + pie charts
  - Per-column deep-dive with skew interpretation

Accessible at any stage after data ingestion (phase >= eda).
"""
from __future__ import annotations

import math
import io
import base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import streamlit as st

from utils.serialization import b64_to_df
from ui.components import alert, badge, metrics_row, b64_image
from ui.graph_helpers import get_state, fig_b64


# ── Design tokens (match existing dark theme) ─────────────────────────────────
PALETTE = [
    "#7ec8fa", "#c6f135", "#f28e2b", "#e15759", "#76b7b2",
    "#b07aa1", "#59a14f", "#edc948", "#ff9da7", "#9c755f",
]
BG        = "#1a1a2e"
AX_BG     = "#16172a"
TEXT      = "#e8e8f5"
GRID_CLR  = "#2a2b45"
SPINE_CLR = "#35366a"

SKEW_THRESHOLDS = {
    "Highly left-skewed":   (-float("inf"), -1.0),
    "Moderately left-skewed": (-1.0, -0.5),
    "Approximately symmetric": (-0.5, 0.5),
    "Moderately right-skewed": (0.5, 1.0),
    "Highly right-skewed":  (1.0, float("inf")),
}


def _skew_label(skew: float) -> str:
    for label, (lo, hi) in SKEW_THRESHOLDS.items():
        if lo <= skew < hi:
            return label
    return "Unknown"


def _skew_color(skew: float) -> str:
    abs_s = abs(skew)
    if abs_s < 0.5:
        return "#59a14f"   # green — symmetric
    if abs_s < 1.0:
        return "#edc948"   # amber — moderate
    return "#e15759"       # red — high skew


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     AX_BG,
        "axes.edgecolor":     SPINE_CLR,
        "axes.labelcolor":    TEXT,
        "axes.titlecolor":    TEXT,
        "axes.grid":          True,
        "grid.color":         GRID_CLR,
        "grid.linewidth":     0.5,
        "xtick.color":        "#aaa",
        "ytick.color":        "#aaa",
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     10,
        "axes.labelsize":     8,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "figure.titlesize":   13,
        "figure.titleweight": "bold",
        "text.color":         TEXT,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

def _render_numeric_grid(df: pd.DataFrame, num_cols: list, cols_per_row: int = 3) -> str:
    """Histogram + KDE per numeric column, with skew annotation."""
    if not num_cols:
        return ""
    _apply_dark_theme()
    n     = len(num_cols)
    ncols = min(cols_per_row, n)
    nrows = math.ceil(n / ncols)
    fig_w = 5.5 * ncols
    fig_h = max(4.5, nrows * 4.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold",
                 color=TEXT, y=1.01)

    for i, col in enumerate(num_cols):
        ax   = axes[i // ncols][i % ncols]
        data = df[col].dropna()
        color = PALETTE[i % len(PALETTE)]

        if data.empty:
            ax.set_title(col, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa")
            continue

        # Histogram
        ax.hist(data, bins=30, color=color, edgecolor="#000",
                linewidth=0.3, density=True, alpha=0.80)

        # KDE overlay
        try:
            data.plot.kde(ax=ax, color="#ffffff", linewidth=1.6)
        except Exception:
            pass

        # Mean / median lines
        mean_v   = data.mean()
        median_v = data.median()
        ax.axvline(mean_v,   color="#7ec8fa", linewidth=1.3, linestyle="--",
                   label=f"μ {mean_v:.3g}")
        ax.axvline(median_v, color="#c6f135", linewidth=1.3, linestyle=":",
                   label=f"med {median_v:.3g}")

        # Stats annotation
        skew_v = data.skew()
        std_v  = data.std()
        stats_txt = (
            f"n={len(data):,}\n"
            f"mean={mean_v:.3g}\n"
            f"std={std_v:.3g}\n"
            f"skew={skew_v:.2f}"
        )
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0e1a",
                          edgecolor=SPINE_CLR, alpha=0.90),
                color=TEXT)

        ax.set_title(col, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7.5)
        ax.set_ylabel("Density", fontsize=7.5)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False, labelcolor=TEXT)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    return fig_b64(fig)


def _render_skew_overview(df: pd.DataFrame, num_cols: list) -> str:
    """Horizontal bar chart of skewness for all numeric columns."""
    if not num_cols:
        return ""
    _apply_dark_theme()

    skews = df[num_cols].skew().sort_values()
    colors = [_skew_color(s) for s in skews.values]

    fig_h = max(4, len(skews) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)

    bars = ax.barh(range(len(skews)), skews.values,
                   color=colors, edgecolor="#000", linewidth=0.3, alpha=0.90)
    ax.set_yticks(range(len(skews)))
    ax.set_yticklabels(skews.index.tolist(), fontsize=8)
    ax.axvline(0, color="#555", linewidth=0.8, linestyle="-")
    ax.axvline(-0.5, color="#444", linewidth=0.6, linestyle="--")
    ax.axvline(0.5,  color="#444", linewidth=0.6, linestyle="--")
    ax.axvline(-1.0, color="#555", linewidth=0.6, linestyle=":")
    ax.axvline(1.0,  color="#555", linewidth=0.6, linestyle=":")

    for bar, val in zip(bars, skews.values):
        ax.text(
            val + (0.03 if val >= 0 else -0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7, color=TEXT,
        )

    ax.set_xlabel("Skewness", fontsize=8.5)
    ax.set_title("Skewness Overview — all numeric columns", fontsize=11,
                 fontweight="bold", pad=10)

    # Legend
    patches = [
        mpatches.Patch(color="#59a14f", label="Symmetric (|skew|<0.5)"),
        mpatches.Patch(color="#edc948", label="Moderate (0.5–1.0)"),
        mpatches.Patch(color="#e15759", label="High (|skew|≥1.0)"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower right", frameon=False,
              labelcolor=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig_b64(fig)


def _render_categorical_grid(df: pd.DataFrame, cat_cols: list, max_categories: int = 10) -> str:
    """Bar + pie side-by-side for each categorical column."""
    if not cat_cols:
        return ""
    _apply_dark_theme()

    n_features = len(cat_cols)
    fig_w = 16
    row_h = 4.5
    fig_h = max(5, n_features * row_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("Categorical Feature Distributions", fontsize=14,
                 fontweight="bold", color=TEXT, y=1.0)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(n_features, 3, figure=fig,
                           width_ratios=[5, 3, 2], hspace=0.6, wspace=0.35)

    for row_idx, col in enumerate(cat_cols):
        ax_bar    = fig.add_subplot(gs[row_idx, 0])
        ax_pie    = fig.add_subplot(gs[row_idx, 1])
        ax_legend = fig.add_subplot(gs[row_idx, 2])
        ax_legend.axis("off")

        for ax in [ax_bar, ax_pie]:
            ax.set_facecolor(AX_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(SPINE_CLR)

        vc = df[col].fillna("(missing)").astype(str).value_counts()
        if len(vc) > max_categories:
            top   = vc.iloc[:max_categories]
            other = vc.iloc[max_categories:].sum()
            vc    = pd.concat([top, pd.Series({"Other": other})])

        labels = vc.index.tolist()
        counts = vc.values.tolist()
        colors = (PALETTE * math.ceil(len(labels) / len(PALETTE)))[: len(labels)]
        total  = sum(counts)

        # Bar chart
        bars = ax_bar.barh(range(len(labels)), counts, color=colors,
                           edgecolor="#000", linewidth=0.4, alpha=0.90)
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels([str(lb)[:28] for lb in labels], fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Count", fontsize=8)
        ax_bar.set_title(f"{col}  ({df[col].nunique()} unique)",
                         fontsize=10, fontweight="bold")
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.tick_params(colors="#aaa")

        max_count = max(counts) if counts else 1
        for bar, count in zip(bars, counts):
            ax_bar.text(
                bar.get_width() + max_count * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({count/total:.1%})",
                va="center", ha="left", fontsize=7, color=TEXT,
            )

        # Pie chart
        wedges, _, autotexts = ax_pie.pie(
            counts, labels=None, colors=colors,
            autopct="%1.1f%%", pctdistance=0.70, startangle=90,
            wedgeprops={"edgecolor": "#000", "linewidth": 0.8},
            textprops={"fontsize": 8, "color": TEXT},
        )
        for at in autotexts:
            at.set_fontsize(7.5)
            at.set_color(TEXT)
        ax_pie.set_title(f"Share — {col}", fontsize=9, fontweight="bold")
        ax_pie.set_aspect("equal")

        # Legend panel
        patches = [mpatches.Patch(color=c, label=str(lb)[:22])
                   for c, lb in zip(colors, labels)]
        ax_legend.legend(handles=patches, loc="center left",
                         bbox_to_anchor=(0.0, 0.5), fontsize=7.5,
                         frameon=False, ncol=1, labelcolor=TEXT)

    return fig_b64(fig)


def _render_summary_pies(df: pd.DataFrame, num_cols: list, cat_cols: list) -> str:
    """Three summary pies: dtype split, skewness profile, null-rate profile."""
    _apply_dark_theme()

    # Skew buckets
    if num_cols:
        skews    = df[num_cols].skew(numeric_only=True)
        sym_n    = int((skews.abs() < 0.5).sum())
        mod_n    = int(((skews.abs() >= 0.5) & (skews.abs() < 1.0)).sum())
        high_n   = int((skews.abs() >= 1.0).sum())
    else:
        sym_n = mod_n = high_n = 0

    # Null-rate buckets
    all_cols = num_cols + cat_cols
    if all_cols:
        null_pcts = df[all_cols].isna().mean() * 100
        no_null   = int((null_pcts == 0).sum())
        mild_null = int(((null_pcts > 0) & (null_pcts < 20)).sum())
        mod_null  = int(((null_pcts >= 20) & (null_pcts < 50)).sum())
        sev_null  = int((null_pcts >= 50).sum())
    else:
        no_null = mild_null = mod_null = sev_null = 0

    # dtype split
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Dataset Distribution Overview", fontsize=13,
                 fontweight="bold", color=TEXT)

    for ax in axes:
        ax.set_facecolor(BG)

    def _pie(ax, values, labels, colors, title_str):
        filtered = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
        if not filtered:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#aaa")
            ax.axis("off")
            ax.set_title(title_str, fontsize=10, fontweight="bold")
            return
        vs, ls, cs = zip(*filtered)
        wedges, _, autotexts = ax.pie(
            vs, labels=None, colors=cs,
            autopct="%1.0f%%", pctdistance=0.70, startangle=90,
            wedgeprops={"edgecolor": "#000", "linewidth": 0.8},
        )
        for at in autotexts:
            at.set_fontsize(8.5)
            at.set_color(TEXT)
        patches = [mpatches.Patch(color=c, label=f"{l} ({v})")
                   for c, l, v in zip(cs, ls, vs)]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), fontsize=8.5,
                  frameon=False, ncol=2, labelcolor=TEXT)
        ax.set_title(title_str, fontsize=10.5, fontweight="bold")
        ax.set_aspect("equal")

    _pie(axes[0],
         [n_num, n_cat],
         ["Numeric", "Categorical"],
         ["#7ec8fa", "#f28e2b"],
         "Column Type Split")

    _pie(axes[1],
         [sym_n, mod_n, high_n],
         ["Symmetric\n(|skew|<0.5)", "Moderate\n(0.5–1.0)", "High\n(|skew|≥1.0)"],
         ["#59a14f", "#edc948", "#e15759"],
         "Skewness Profile\n(numeric cols)")

    _pie(axes[2],
         [no_null, mild_null, mod_null, sev_null],
         ["None", "Mild\n(<20%)", "Moderate\n(20–50%)", "Severe\n(≥50%)"],
         ["#59a14f", "#7ec8fa", "#f28e2b", "#e15759"],
         "Null-Rate Profile")

    plt.tight_layout()
    return fig_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Skew stats table
# ─────────────────────────────────────────────────────────────────────────────

def _skew_table(df: pd.DataFrame, num_cols: list):
    """Render a sortable skewness stats table."""
    if not num_cols:
        return

    rows = []
    for col in num_cols:
        data  = df[col].dropna()
        skew  = data.skew() if len(data) > 2 else float("nan")
        kurt  = data.kurt() if len(data) > 3 else float("nan")
        null_pct = df[col].isna().mean() * 100
        rows.append({
            "Column":     col,
            "Dtype":      str(df[col].dtype),
            "Non-null N": len(data),
            "Null %":     round(null_pct, 1),
            "Mean":       round(data.mean(), 4) if len(data) else None,
            "Median":     round(data.median(), 4) if len(data) else None,
            "Std":        round(data.std(), 4) if len(data) > 1 else None,
            "Min":        round(data.min(), 4) if len(data) else None,
            "Max":        round(data.max(), 4) if len(data) else None,
            "Skewness":   round(skew, 3) if not math.isnan(skew) else None,
            "Kurtosis":   round(kurt, 3) if not math.isnan(kurt) else None,
            "Skew label": _skew_label(skew) if not math.isnan(skew) else "—",
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(
        table_df.style.applymap(
            lambda v: (
                "background-color:#0d2e18;color:#6ee7a0" if isinstance(v, float) and abs(v) < 0.5
                else "background-color:#2e1e08;color:#fcd34d" if isinstance(v, float) and 0.5 <= abs(v) < 1.0
                else "background-color:#2e1010;color:#fca5a5" if isinstance(v, float) and abs(v) >= 1.0
                else ""
            ),
            subset=["Skewness"],
        ),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────

def page_distributions(_rt):
    s = st.session_state.get("pipeline_state", {})

    st.markdown(
        f'{badge("Explorer")} <h1 style="display:inline;margin-left:.5rem;">'
        f'Feature Distribution Explorer</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Explore distributions, skewness, and value breakdowns for every feature in your dataset.")

    tid = st.session_state.get("tid")
    if not tid:
        alert("No active session. Load a dataset first via Data Fusion.", "warning")
        return

    # Load dataframe from runtime store (cached)
    cache_key = "_dist_df_loaded"
    df_cache  = _rt.get_key(tid, cache_key)

    if df_cache is None:
        try:
            full_state = get_state(tid)
            df_b64 = full_state.get("df_parquet_b64")
            if not df_b64:
                alert("Dataset not found in session state. Please re-run the pipeline from Data Fusion.", "warning")
                return
            df = b64_to_df(df_b64)
            _rt.set_key(tid, cache_key, "loaded")
            _rt.set_key(tid, "_dist_df", df.to_json())
        except Exception as e:
            alert(f"Could not load dataset: {e}", "error")
            return
    else:
        try:
            df = pd.read_json(_rt.get_key(tid, "_dist_df"))
        except Exception as e:
            alert(f"Could not deserialize cached dataset: {e}", "error")
            return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── Top metrics ──────────────────────────────────────────────────────────
    n_high_skew = 0
    if num_cols:
        skews = df[num_cols].skew()
        n_high_skew = int((skews.abs() >= 1.0).sum())

    metrics_row([
        ("Total columns",    df.shape[1]),
        ("Rows",             f"{df.shape[0]:,}"),
        ("Numeric cols",     len(num_cols)),
        ("Categorical cols", len(cat_cols)),
        ("High-skew cols",   n_high_skew),
        ("Cols with nulls",  int(df.isna().any().sum())),
    ])

    # ── Column filter & search ───────────────────────────────────────────────
    st.markdown("---")
    col_filter_search = st.text_input(
        "🔍 Filter columns by name",
        placeholder="Type to filter…",
        key="dist_filter",
    )
    filter_q = col_filter_search.strip().lower()
    if filter_q:
        num_cols = [c for c in num_cols if filter_q in c.lower()]
        cat_cols = [c for c in cat_cols if filter_q in c.lower()]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_overview, tab_numeric, tab_skew, tab_categorical, tab_single = st.tabs([
        "📊 Overview",
        "📈 Numeric Distributions",
        "📐 Skew Analysis",
        "🗂 Categorical Distributions",
        "🔬 Single Column Deep-Dive",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Overview summary pies
    # ════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown("### Dataset-level distribution overview")
        st.caption("High-level summary of column types, skewness profile, and null rates across the full dataset.")

        with st.spinner("Generating overview charts…"):
            cache_k = "_dist_overview_b64"
            b64 = _rt.get_key(tid, cache_k)
            if not b64:
                b64 = _render_summary_pies(df, num_cols, cat_cols)
                _rt.set_key(tid, cache_k, b64)
            if b64:
                b64_image(b64, "Distribution overview")
            else:
                alert("No data to display.", "info")

        # Null summary table
        st.markdown("#### Null rate by column")
        null_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  [str(df[c].dtype) for c in df.columns],
            "Null count": df.isna().sum().values,
            "Null %": (df.isna().mean() * 100).round(2).values,
            "Unique values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(null_df.style.background_gradient(
            subset=["Null %"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Numeric histograms
    # ════════════════════════════════════════════════════════════════════════
    with tab_numeric:
        if not num_cols:
            alert("No numeric columns found (or all filtered out).", "info")
        else:
            st.markdown(f"### Distributions for **{len(num_cols)}** numeric column(s)")
            st.caption("Each subplot shows a histogram + KDE overlay, with mean (blue dashed) and median (green dotted) lines.")

            cols_per_row = st.select_slider(
                "Columns per row", options=[1, 2, 3, 4], value=3,
                key="dist_cols_per_row",
            )
            with st.spinner("Rendering histograms…"):
                cache_k = f"_dist_num_{filter_q}_{cols_per_row}"
                b64 = _rt.get_key(tid, cache_k)
                if not b64:
                    b64 = _render_numeric_grid(df, num_cols, cols_per_row)
                    _rt.set_key(tid, cache_k, b64)
                if b64:
                    b64_image(b64, "Numeric distributions")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Skewness analysis
    # ════════════════════════════════════════════════════════════════════════
    with tab_skew:
        if not num_cols:
            alert("No numeric columns found (or all filtered out).", "info")
        else:
            st.markdown("### Skewness overview")
            st.caption(
                "**Skewness** measures the asymmetry of a distribution. "
                "|skew| < 0.5 → symmetric (✅ good for most models). "
                "0.5–1.0 → moderate skew. > 1.0 → high skew (consider log-transform or winsorization)."
            )

            with st.spinner("Rendering skewness chart…"):
                cache_k = f"_dist_skew_{filter_q}"
                b64 = _rt.get_key(tid, cache_k)
                if not b64:
                    b64 = _render_skew_overview(df, num_cols)
                    _rt.set_key(tid, cache_k, b64)
                if b64:
                    b64_image(b64, "Skewness overview")

            st.markdown("#### Detailed skewness & descriptive stats table")
            st.caption("Click any column header to sort. Skewness cells are colour-coded: 🟢 symmetric · 🟡 moderate · 🔴 high.")
            _skew_table(df, num_cols)

            # Skew distribution breakdown
            if num_cols:
                skews = df[num_cols].skew()
                sym_n  = int((skews.abs() < 0.5).sum())
                mod_n  = int(((skews.abs() >= 0.5) & (skews.abs() < 1.0)).sum())
                high_n = int((skews.abs() >= 1.0).sum())

                st.markdown("#### Skewness breakdown")
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Symmetric (|skew|<0.5)", sym_n)
                c2.metric("⚠️ Moderate (0.5–1.0)", mod_n)
                c3.metric("🔴 High (|skew|≥1.0)", high_n)

                if high_n > 0:
                    high_skew_cols = skews[skews.abs() >= 1.0].sort_values(key=abs, ascending=False)
                    with st.expander(f"🔴 {high_n} highly skewed column(s) — click to expand"):
                        for col, sv in high_skew_cols.items():
                            direction = "right" if sv > 0 else "left"
                            st.markdown(
                                f"**`{col}`** — skew = `{sv:.3f}` ({direction}-skewed) · "
                                f"Suggest: `log_transform` / `winsorize`"
                            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Categorical distributions
    # ════════════════════════════════════════════════════════════════════════
    with tab_categorical:
        if not cat_cols:
            alert("No categorical columns found (or all filtered out).", "info")
        else:
            st.markdown(f"### Distributions for **{len(cat_cols)}** categorical column(s)")
            st.caption("Each row shows a bar chart (count) and a pie chart (share). Top 10 categories shown; rest grouped as 'Other'.")

            max_cats = st.slider("Max categories shown per column", 5, 25, 10,
                                 key="dist_max_cats")
            with st.spinner("Rendering categorical charts…"):
                cache_k = f"_dist_cat_{filter_q}_{max_cats}"
                b64 = _rt.get_key(tid, cache_k)
                if not b64:
                    b64 = _render_categorical_grid(df, cat_cols, max_cats)
                    _rt.set_key(tid, cache_k, b64)
                if b64:
                    b64_image(b64, "Categorical distributions")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Single-column deep-dive
    # ════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown("### Single-column deep-dive")
        st.caption("Select any column to see its full distribution, stats, and value breakdown.")

        all_cols = num_cols + cat_cols
        if not all_cols:
            alert("No columns match the current filter.", "info")
        else:
            chosen = st.selectbox("Choose a column", all_cols, key="dist_single_col")
            data   = df[chosen]
            is_num = chosen in num_cols

            # Stats
            st.markdown(f"#### `{chosen}` — {'Numeric' if is_num else 'Categorical'}")
            null_pct = data.isna().mean() * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Non-null count", f"{data.notna().sum():,}")
            c2.metric("Null %",         f"{null_pct:.1f}%")
            c3.metric("Unique values",  f"{data.nunique():,}")
            c4.metric("Dtype",          str(data.dtype))

            if is_num:
                clean = data.dropna()
                skew_v = clean.skew()
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Mean",   f"{clean.mean():.4g}")
                col_b.metric("Median", f"{clean.median():.4g}")
                col_c.metric("Std",    f"{clean.std():.4g}")
                skew_color_emoji = "✅" if abs(skew_v) < 0.5 else ("⚠️" if abs(skew_v) < 1.0 else "🔴")
                col_d.metric("Skewness", f"{skew_color_emoji} {skew_v:.3f}")

                st.caption(f"**Skew interpretation:** {_skew_label(skew_v)}")
                if abs(skew_v) >= 0.5:
                    rec = "log_transform" if skew_v > 0 else "reflect then log_transform"
                    st.info(f"💡 Recommendation: consider `{rec}` or `winsorize` to reduce skewness.")

                # Plot
                _apply_dark_theme()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor(BG)
                for ax in axes:
                    ax.set_facecolor(AX_BG)
                    for sp in ax.spines.values():
                        sp.set_edgecolor(SPINE_CLR)

                color = PALETTE[0]
                # Histogram
                axes[0].hist(clean, bins=40, color=color, edgecolor="#000",
                             linewidth=0.3, density=True, alpha=0.85)
                try:
                    clean.plot.kde(ax=axes[0], color="#ffffff", linewidth=1.8)
                except Exception:
                    pass
                axes[0].axvline(clean.mean(), color="#7ec8fa", linestyle="--",
                                linewidth=1.4, label=f"Mean {clean.mean():.3g}")
                axes[0].axvline(clean.median(), color="#c6f135", linestyle=":",
                                linewidth=1.4, label=f"Median {clean.median():.3g}")
                axes[0].set_title(f"Distribution — {chosen}", fontsize=10, fontweight="bold")
                axes[0].set_xlabel("Value"); axes[0].set_ylabel("Density")
                axes[0].legend(fontsize=7.5, frameon=False, labelcolor=TEXT)
                axes[0].spines[["top", "right"]].set_visible(False)

                # Box plot
                bp = axes[1].boxplot(clean, vert=True, patch_artist=True,
                                     boxprops=dict(facecolor=color, alpha=0.7),
                                     medianprops=dict(color="#fff", linewidth=2),
                                     whiskerprops=dict(color="#aaa"),
                                     capprops=dict(color="#aaa"),
                                     flierprops=dict(marker="o", markersize=3,
                                                     markerfacecolor="#e15759", alpha=0.6))
                axes[1].set_title(f"Box plot — {chosen}", fontsize=10, fontweight="bold")
                axes[1].set_ylabel("Value")
                axes[1].set_xticks([])
                axes[1].spines[["top", "right"]].set_visible(False)

                plt.tight_layout()
                b64 = fig_b64(fig)
                b64_image(b64, f"Deep-dive: {chosen}")

                # Percentiles table
                st.markdown("#### Percentile breakdown")
                pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
                pct_df = pd.DataFrame({
                    "Percentile": [f"p{p}" for p in pcts],
                    "Value":      [round(float(np.percentile(clean, p)), 4) for p in pcts],
                })
                st.dataframe(pct_df, use_container_width=False, hide_index=True)

            else:
                # Categorical deep-dive
                vc = data.fillna("(missing)").astype(str).value_counts()
                st.markdown(f"#### Value counts — top {min(30, len(vc))} categories")
                st.dataframe(
                    pd.DataFrame({
                        "Value":  vc.index[:30],
                        "Count":  vc.values[:30],
                        "Share %": (vc.values[:30] / len(data) * 100).round(2),
                    }),
                    use_container_width=True, hide_index=True,
                )

                # Simple bar
                top_n = min(20, len(vc))
                vc_top = vc.iloc[:top_n]
                _apply_dark_theme()
                fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.4 + 1)))
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(AX_BG)
                colors = (PALETTE * math.ceil(top_n / len(PALETTE)))[:top_n]
                ax.barh(range(top_n), vc_top.values, color=colors,
                        edgecolor="#000", linewidth=0.3, alpha=0.90)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([str(lb)[:30] for lb in vc_top.index], fontsize=8.5)
                ax.invert_yaxis()
                ax.set_xlabel("Count"); ax.set_title(f"Top {top_n} values — {chosen}",
                                                      fontsize=10, fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                b64 = fig_b64(fig)
                b64_image(b64, f"Bar chart: {chosen}")
