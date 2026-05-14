"""
utils/autoeda_report.py — Automatic EDA report generator (v5)

Generates a comprehensive HTML EDA report similar to pandas profiling,
without requiring the heavy ydata-profiling dependency.

Covers:
  - Dataset overview (shape, types, missing, duplicates)
  - Per-column statistics (numeric + categorical)
  - Correlation matrix heatmap
  - Target distribution
  - Missing value heatmap
  - Feature-target relationships
  - Data quality warnings

Usage:
    from utils.autoeda_report import generate_autoeda_report

    html = generate_autoeda_report(df, target="Survived", problem_type="classification")
    with open("eda_report.html", "w") as f:
        f.write(html)
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config_loader import cfg
from utils.logger import get_logger
from utils.stats_safety import clean_numeric_frame_for_corr, safe_corr

log = get_logger(__name__)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _overview(df: pd.DataFrame, target: str) -> dict:
    missing = df.isna().mean()
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_numeric": int(df.select_dtypes("number").shape[1]),
        "n_categorical": int(df.select_dtypes(["str", "object", "category", "bool"]).shape[1]),
        "n_datetime": int(df.select_dtypes(["datetime", "datetimetz"]).shape[1]),
        "total_missing_pct": round(float(missing.mean() * 100), 2),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def _numeric_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "mean":    round(float(s.mean()), 4),
        "std":     round(float(s.std()), 4),
        "min":     round(float(s.min()), 4),
        "p25":     round(float(s.quantile(0.25)), 4),
        "median":  round(float(s.median()), 4),
        "p75":     round(float(s.quantile(0.75)), 4),
        "max":     round(float(s.max()), 4),
        "skew":    round(float(s.skew()), 4),
        "kurtosis":round(float(s.kurtosis()), 4),
        "zeros":   int((s == 0).sum()),
        "n_unique":int(s.nunique()),
    }


def _categorical_stats(series: pd.Series) -> dict:
    n_unique = int(series.nunique())
    top = series.value_counts().head(10)
    return {
        "n_unique": n_unique,
        "top_values": {str(k): int(v) for k, v in top.items()},
        "mode":    str(series.mode().iloc[0]) if len(series.mode()) else "",
        "missing_pct": round(float(series.isna().mean() * 100), 2),
    }


def _plot_distributions(df: pd.DataFrame, target: str, max_cols: int = 20) -> str:
    """Grid of distribution plots for numeric columns."""
    num_cols = [c for c in df.select_dtypes("number").columns if c != target][:max_cols]
    if not num_cols:
        return ""
    n = len(num_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        df[col].dropna().hist(ax=ax, bins=30, color="#4f8ef7", alpha=0.8, edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_correlation(df: pd.DataFrame, target: str) -> str:
    num_df = clean_numeric_frame_for_corr(df)
    if num_df.shape[1] < 2:
        return ""
    method = cfg("autoeda.correlation_method", default="pearson")
    try:
        corr = num_df.corr(method=method)
    except Exception:
        corr = num_df.corr()
    n = len(corr)
    size = max(6, min(n * 0.6, 16))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(f"Correlation Matrix ({method})", fontsize=12)
    # Annotate if small
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_missing(df: pd.DataFrame) -> str:
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.3)))
    missing.plot.barh(ax=ax, color="#e06c75", edgecolor="white")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Value Rate by Column")
    ax.tick_params(labelsize=8)
    for i, v in enumerate(missing):
        ax.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=7)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_target(df: pd.DataFrame, target: str, problem_type: str) -> str:
    if target not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    if problem_type == "classification":
        vc = df[target].value_counts()
        vc.plot.bar(ax=ax, color="#61afef", edgecolor="white")
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    else:
        df[target].dropna().hist(ax=ax, bins=40, color="#98c379", edgecolor="white")
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _data_quality_warnings(df: pd.DataFrame, target: str) -> list[str]:
    warnings = []
    # Missing
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 50:
            warnings.append(f"🔴 '{col}' has {pct:.1f}% missing values")
        elif pct > 20:
            warnings.append(f"🟡 '{col}' has {pct:.1f}% missing values")
    # Constant
    for col in df.columns:
        if df[col].nunique() <= 1:
            warnings.append(f"🔴 '{col}' is constant (zero variance)")
    # Duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        warnings.append(f"🟡 {n_dup} duplicate rows detected")
    # High cardinality
    for col in df.select_dtypes(["str", "object", "category"]).columns:
        if col == target:
            continue
        n = df[col].nunique()
        if n > 100:
            warnings.append(f"🟡 '{col}' has {n} unique values (high cardinality)")
    # Target leakage candidates
    num_df = clean_numeric_frame_for_corr(df)
    if target in num_df.columns:
        others = [c for c in num_df.columns if c != target]
        for col in others:
            try:
                corr = safe_corr(num_df[col], num_df[target])
                if corr is not None and abs(corr) > 0.95:
                    warnings.append(f"🔴 '{col}' has {corr:.3f} correlation with target — possible leakage")
            except Exception:
                pass
    return warnings


def generate_autoeda_report(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "classification",
    title: str = "AutoEDA Report",
) -> str:
    """
    Generate a self-contained HTML EDA report.

    Returns:
        HTML string containing the full report.
    """
    log.info("Generating AutoEDA report", extra={"shape": df.shape, "target": target})

    overview   = _overview(df, target)
    warnings   = _data_quality_warnings(df, target)
    dist_img   = _plot_distributions(df, target)
    corr_img   = _plot_correlation(df, target)
    miss_img   = _plot_missing(df)
    target_img = _plot_target(df, target, problem_type)

    # Per-column stats
    col_stats = []
    for col in df.columns:
        entry = {"name": col, "dtype": str(df[col].dtype),
                 "missing_pct": round(float(df[col].isna().mean() * 100), 1)}
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["type"] = "numeric"
            entry.update(_numeric_stats(df[col]))
        else:
            entry["type"] = "categorical"
            entry.update(_categorical_stats(df[col]))
        col_stats.append(entry)

    def img_block(b64: str, caption: str) -> str:
        if not b64:
            return ""
        return f'<div class="chart-block"><h3>{caption}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"/></div>'

    warn_html = ""
    if warnings:
        warn_html = "<ul>" + "".join(f"<li>{w}</li>" for w in warnings) + "</ul>"
    else:
        warn_html = "<p>✅ No major data quality issues found.</p>"

    col_rows = ""
    for s in col_stats:
        stats_str = ""
        if s["type"] == "numeric":
            stats_str = (f"mean={s.get('mean','?')}, std={s.get('std','?')}, "
                         f"min={s.get('min','?')}, max={s.get('max','?')}, "
                         f"skew={s.get('skew','?')}")
        else:
            top = list(s.get("top_values", {}).items())[:3]
            top_str = ", ".join(f"{k}({v})" for k, v in top)
            stats_str = f"n_unique={s.get('n_unique','?')}, top: {top_str}"
        col_rows += f"""
        <tr>
            <td><code>{s['name']}</code></td>
            <td>{s['type']}</td>
            <td>{s['dtype']}</td>
            <td>{s['missing_pct']}%</td>
            <td style="font-size:12px">{stats_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1e1e2e; color: #cdd6f4; margin: 0; padding: 20px; }}
  h1   {{ color: #89b4fa; border-bottom: 2px solid #313244; padding-bottom: 8px; }}
  h2   {{ color: #cba6f7; margin-top: 30px; }}
  h3   {{ color: #89dceb; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 16px; margin: 20px 0; }}
  .card {{ background: #313244; border-radius: 12px; padding: 16px; }}
  .card .val {{ font-size: 28px; font-weight: 700; color: #89b4fa; }}
  .card .lbl {{ font-size: 12px; color: #a6adc8; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th    {{ background: #313244; padding: 10px; text-align: left; color: #cba6f7; }}
  td    {{ padding: 8px 10px; border-bottom: 1px solid #313244; font-size: 13px; }}
  tr:hover td {{ background: #2a2a3e; }}
  .warn-box {{ background: #1e1e2e; border-left: 4px solid #f38ba8;
               padding: 12px 16px; border-radius: 4px; margin: 12px 0; }}
  .warn-box li {{ margin: 4px 0; }}
  .chart-block {{ background: #313244; border-radius: 12px; padding: 16px; margin: 16px 0; }}
  code {{ background: #313244; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #a6e3a1; }}
</style>
</head>
<body>
<h1>📊 {title}</h1>
<p style="color:#a6adc8">Target: <code>{target}</code> &nbsp;|&nbsp;
   Problem: <code>{problem_type}</code> &nbsp;|&nbsp;
   Generated: <code>{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</code></p>

<h2>📋 Dataset Overview</h2>
<div class="grid">
  <div class="card"><div class="val">{overview['n_rows']:,}</div><div class="lbl">Rows</div></div>
  <div class="card"><div class="val">{overview['n_cols']}</div><div class="lbl">Columns</div></div>
  <div class="card"><div class="val">{overview['n_numeric']}</div><div class="lbl">Numeric</div></div>
  <div class="card"><div class="val">{overview['n_categorical']}</div><div class="lbl">Categorical</div></div>
  <div class="card"><div class="val">{overview['total_missing_pct']}%</div><div class="lbl">Missing (avg)</div></div>
  <div class="card"><div class="val">{overview['n_duplicate_rows']}</div><div class="lbl">Duplicate Rows</div></div>
  <div class="card"><div class="val">{overview['memory_mb']} MB</div><div class="lbl">Memory</div></div>
</div>

<h2>⚠️ Data Quality Warnings</h2>
<div class="warn-box">{warn_html}</div>

{img_block(target_img, f"🎯 Target Distribution: {target}")}

{img_block(miss_img, "❓ Missing Value Rate")}

{img_block(dist_img, "📈 Feature Distributions")}

{img_block(corr_img, "🔗 Correlation Matrix")}

<h2>📊 Column Statistics</h2>
<table>
  <thead>
    <tr><th>Column</th><th>Type</th><th>Dtype</th><th>Missing</th><th>Statistics</th></tr>
  </thead>
  <tbody>
    {col_rows}
  </tbody>
</table>

</body>
</html>"""

    return html
