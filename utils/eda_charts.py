"""
utils/eda_charts.py — Professional EDA chart generators.

Public API
----------
  generate_categorical_charts(df, cat_cols, max_categories=10) -> b64_png_str
      Bar + pie side-by-side layout, one row per column, legend panel.

  generate_numeric_distributions(df, num_cols, title) -> b64_png_str
      Histogram + KDE overlay, 3 columns per row, with descriptive stats.
"""

import io
import base64
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd


# ── Shared design tokens ──────────────────────────────────────────────────────

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor":    "#ffffff",
        "axes.facecolor":      "#fafafa",
        "axes.edgecolor":      "#cccccc",
        "axes.grid":           True,
        "grid.color":          "#e8e8e8",
        "grid.linewidth":      0.6,
        "font.family":         "DejaVu Sans",
        "axes.titlesize":      10,
        "axes.labelsize":      8,
        "xtick.labelsize":     7.5,
        "ytick.labelsize":     7.5,
        "figure.titlesize":    13,
        "figure.titleweight":  "bold",
    })


def _b64_fig(fig, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Categorical charts  —  bar + pie side-by-side
# ═══════════════════════════════════════════════════════════════════════════════

def generate_categorical_charts(
    df: pd.DataFrame,
    cat_cols: list,
    max_categories: int = 10,
    title: str = "Categorical Feature Charts (train set)",
) -> str:
    """
    Generate one clear bar + pie chart pair per categorical column.

    Parameters
    ----------
    df             : DataFrame (train set only)
    cat_cols       : list of categorical column names to plot
    max_categories : top-N categories to show (rest grouped as 'Other')
    title          : overall figure title

    Returns
    -------
    Base64-encoded PNG string ready for _embed_plot()
    """
    if not cat_cols:
        return ""

    _apply_theme()

    n_features = len(cat_cols)
    fig_w = 18
    row_h = 4.5
    fig_h = max(5, n_features * row_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    gs = gridspec.GridSpec(
        n_features, 3,
        figure=fig,
        width_ratios=[5, 3, 2],
        hspace=0.55,
        wspace=0.35,
    )

    for row_idx, col in enumerate(cat_cols):
        ax_bar    = fig.add_subplot(gs[row_idx, 0])
        ax_pie    = fig.add_subplot(gs[row_idx, 1])
        ax_legend = fig.add_subplot(gs[row_idx, 2])
        ax_legend.axis("off")

        # Value counts
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
        bars = ax_bar.barh(
            range(len(labels)), counts,
            color=colors, edgecolor="white", linewidth=0.6,
        )
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels([str(lb)[:30] for lb in labels], fontsize=8.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Count", fontsize=8)
        ax_bar.set_title(f"{col}  ({df[col].nunique()} unique values)", fontsize=10, fontweight="bold")
        ax_bar.tick_params(axis="x", labelsize=7.5)
        ax_bar.spines[["top", "right"]].set_visible(False)

        max_count = max(counts) if counts else 1
        for bar, count in zip(bars, counts):
            ax_bar.text(
                bar.get_width() + max_count * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({count/total:.1%})",
                va="center", ha="left", fontsize=7,
            )

        # Pie chart
        wedge_props = {"edgecolor": "white", "linewidth": 1.0}
        wedges, texts, autotexts = ax_pie.pie(
            counts,
            labels=None,
            colors=colors,
            autopct="%1.1f%%",
            pctdistance=0.70,
            startangle=90,
            wedgeprops=wedge_props,
            textprops={"fontsize": 8},
        )
        for at in autotexts:
            at.set_fontsize(7.5)
        ax_pie.set_title(f"Share — {col}", fontsize=9, fontweight="bold")
        ax_pie.set_aspect("equal")

        # Legend
        patches = [
            mpatches.Patch(color=c, label=str(lb)[:24])
            for c, lb in zip(colors, labels)
        ]
        ax_legend.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            fontsize=7.5,
            frameon=False,
            ncol=1,
        )

    return _b64_fig(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Numeric distribution charts  —  histogram + KDE + stats
# ═══════════════════════════════════════════════════════════════════════════════

def generate_numeric_distributions(
    df: pd.DataFrame,
    num_cols: list,
    title: str = "Numeric Feature Distributions (train set)",
) -> str:
    """
    One histogram + KDE per numeric column, 3 per row.
    Each subplot shows a mini-stats annotation (mean, median, std, skew).

    Returns
    -------
    Base64-encoded PNG string.
    """
    if not num_cols:
        return ""

    _apply_theme()

    n     = len(num_cols)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig_w = 5.5 * ncols
    fig_h = max(4, nrows * 4.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, col in enumerate(num_cols):
        ax   = axes[i // ncols][i % ncols]
        data = df[col].dropna()

        if data.empty:
            ax.set_title(col, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Histogram
        ax.hist(
            data, bins=30,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.4,
            density=True, alpha=0.80,
        )

        # KDE overlay
        try:
            data.plot.kde(ax=ax, color="#e15759", linewidth=1.8)
        except Exception:
            pass

        # Mean / median lines
        mean_val   = data.mean()
        median_val = data.median()
        ax.axvline(mean_val,   color="#283593", linewidth=1.2,
                   linestyle="--", label=f"Mean {mean_val:.2g}")
        ax.axvline(median_val, color="#388e3c", linewidth=1.2,
                   linestyle=":",  label=f"Median {median_val:.2g}")

        # Mini stats text box
        std_val  = data.std()
        skew_val = data.skew()
        stats_txt = (
            f"n={len(data):,}\n"
            f"mean={mean_val:.3g}\n"
            f"std={std_val:.3g}\n"
            f"skew={skew_val:.2f}"
        )
        ax.text(
            0.97, 0.97, stats_txt,
            transform=ax.transAxes,
            fontsize=6.5, verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdbdbd", alpha=0.85),
        )

        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    return _b64_fig(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Numeric summary pie charts  —  skewness / null-rate overview
# ═══════════════════════════════════════════════════════════════════════════════

def generate_numeric_summary_pies(
    df: pd.DataFrame,
    num_cols: list,
    title: str = "Numeric Feature Summary",
) -> str:
    """
    Generate overview pie charts summarising numeric feature properties:
      - Data type breakdown (int vs float)
      - Skewness profile (symmetric / moderate / high skew)
      - Null-rate profile (none / mild / moderate / severe)

    Returns
    -------
    Base64-encoded PNG string.
    """
    if not num_cols:
        return ""

    _apply_theme()
    sub_df = df[num_cols]

    # ── Skewness buckets ─────────────────────────────────────────────────────
    skews = sub_df.skew(numeric_only=True)
    symmetric   = int((skews.abs() < 0.5).sum())
    mod_skew    = int(((skews.abs() >= 0.5) & (skews.abs() < 1.0)).sum())
    high_skew   = int((skews.abs() >= 1.0).sum())

    # ── Null-rate buckets ─────────────────────────────────────────────────────
    null_pcts = sub_df.isna().mean() * 100
    no_null   = int((null_pcts == 0).sum())
    mild_null = int(((null_pcts > 0) & (null_pcts < 20)).sum())
    mod_null  = int(((null_pcts >= 20) & (null_pcts < 50)).sum())
    sev_null  = int((null_pcts >= 50).sum())

    # ── dtype split ───────────────────────────────────────────────────────────
    int_cols   = int(sub_df.select_dtypes(include="integer").shape[1])
    float_cols = int(sub_df.shape[1] - int_cols)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def _pie(ax, values, labels, colors, title_str):
        filtered = [(v, l) for v, l in zip(values, labels) if v > 0]
        if not filtered:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            ax.set_title(title_str, fontsize=10, fontweight="bold")
            return
        vs, ls = zip(*filtered)
        cs = colors[:len(vs)]
        wedges, _, autotexts = ax.pie(
            vs, labels=None, colors=cs,
            autopct="%1.0f%%", pctdistance=0.70, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(8)
        patches = [mpatches.Patch(color=c, label=f"{l} ({v})")
                   for c, l, v in zip(cs, ls, vs)]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), fontsize=8, frameon=False, ncol=2)
        ax.set_title(title_str, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")

    _pie(
        axes[0],
        [int_cols, float_cols],
        ["Integer", "Float"],
        ["#4e79a7", "#f28e2b"],
        "Data Type Split",
    )
    _pie(
        axes[1],
        [symmetric, mod_skew, high_skew],
        ["Symmetric\n(|skew|<0.5)", "Moderate\n(0.5-1.0)", "High\n(|skew|≥1.0)"],
        ["#59a14f", "#edc948", "#e15759"],
        "Skewness Profile",
    )
    _pie(
        axes[2],
        [no_null, mild_null, mod_null, sev_null],
        ["None", "Mild\n(<20%)", "Moderate\n(20-50%)", "Severe\n(≥50%)"],
        ["#59a14f", "#4e79a7", "#f28e2b", "#e15759"],
        "Null-Rate Profile",
    )

    plt.tight_layout()
    return _b64_fig(fig)
