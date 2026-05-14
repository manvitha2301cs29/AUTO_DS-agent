"""
ui/pages/compare.py — Multi-dataset training + comparison (v5)

Allows users to train multiple datasets and compare:
  - Metrics side-by-side (F1, ROC-AUC, R², RMSE, etc.)
  - Feature importance across datasets
  - Model choice across datasets
  - Drift summary per dataset

Usage:
    from ui.pages.compare import page_compare
    page_compare()
"""

from __future__ import annotations
import streamlit as st
import pandas as pd

from ui.components import alert, badge, metrics_row
from ui.state_store import get_store


def page_compare() -> None:
    store = get_store()

    st.markdown(
        f'{badge("Compare")} <h1 style="display:inline;margin-left:.5rem;">'
        f'Multi-Dataset Comparison</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Compare multiple AutoML runs side-by-side.")

    datasets = store.datasets
    if len(datasets) < 2:
        st.info(
            "Train at least **2 datasets** to compare. "
            "Each completed pipeline run is automatically registered here."
        )
        if datasets:
            st.write("**Registered datasets:**")
            for d in datasets:
                m = d.get("metrics", {})
                st.write(f"• `{d['name']}` — {_fmt_metrics(m)}")
        return

    # ── Metrics comparison table ─────────────────────────────────────────────
    st.markdown("#### 📊 Metrics Comparison")
    rows = []
    for d in datasets:
        m = d.get("metrics", {})
        rows.append({
            "Dataset":    d["name"],
            "F1 weighted": _fmt(m.get("f1_weighted")),
            "ROC-AUC":    _fmt(m.get("roc_auc")),
            "Accuracy":   _fmt(m.get("accuracy")),
            "R²":         _fmt(m.get("r2")),
            "RMSE":       _fmt(m.get("rmse")),
            "MAE":        _fmt(m.get("mae")),
        })
    df = pd.DataFrame(rows).set_index("Dataset")
    # Drop all-empty columns
    df = df.loc[:, (df != "—").any(axis=0)]
    st.dataframe(df, width="stretch")

    # ── Best model per dataset ───────────────────────────────────────────────
    st.markdown("#### 🤖 Model Selection")
    model_rows = []
    for d in datasets:
        model_rows.append({
            "Dataset":   d["name"],
            "Best Model": d.get("best_model_key") or d.get("best_model", "—"),
            "CV Score":   _fmt(d.get("cv_score")),
            "Problem":    d.get("problem_type", "—"),
        })
    st.dataframe(pd.DataFrame(model_rows).set_index("Dataset"), width="stretch")

    # ── Primary metric bar chart ─────────────────────────────────────────────
    st.markdown("#### 📈 Primary Metric Comparison")
    chart_data = {}
    for d in datasets:
        m = d.get("metrics", {})
        primary = m.get("f1_weighted") or m.get("r2") or m.get("accuracy")
        if primary is not None:
            chart_data[d["name"]] = float(primary)

    if chart_data:
        chart_df = pd.DataFrame(
            {"Dataset": list(chart_data.keys()), "Score": list(chart_data.values())}
        ).set_index("Dataset")
        st.bar_chart(chart_df)

    # ── Drift comparison ─────────────────────────────────────────────────────
    drift_rows = [
        {
            "Dataset":  d["name"],
            "Severity": d.get("drift_severity", "—"),
            "Score":    _fmt(d.get("drift_score")),
            "Flagged Features": d.get("n_drifted_features", "—"),
        }
        for d in datasets
    ]
    if any(r["Severity"] != "—" for r in drift_rows):
        st.markdown("#### 🌊 Drift Summary")
        st.dataframe(pd.DataFrame(drift_rows).set_index("Dataset"), width="stretch")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _fmt_metrics(m: dict) -> str:
    parts = []
    for k in ("f1_weighted", "roc_auc", "r2", "accuracy"):
        if k in m and m[k] is not None:
            parts.append(f"{k}={m[k]:.4f}")
    return ", ".join(parts) or "no metrics"
