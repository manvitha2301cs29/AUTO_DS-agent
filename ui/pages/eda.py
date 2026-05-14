"""ui/pages/eda.py — Phase 2: EDA Agent page.

Pipeline flow handled here:
  1. EDA agent already ran (state has preprocessing_decisions).
  2. User reviews/overrides decisions and clicks "Approve EDA".
  3. resume_graph_sync runs: hitl_eda -> leakage_agent -> STOPS at hitl_leakage.
  4. Page re-renders showing the leakage report.
  5. User clicks "Approve Leakage & run Feature Agent".
  6. resume_graph_sync runs: hitl_leakage -> feature_agent -> STOPS at hitl_features.
  7. Phase advances to "features".
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from utils.serialization import b64_to_df
from utils.stats_safety import clean_numeric_frame_for_corr
from ui.components import alert, badge, metrics_row, b64_image
from ui.graph_helpers import get_state, resume_graph_sync, safe_state, persist, fig_b64
from ui.state_store import update_store
from ui.workflow_controls import reopen_workflow_stage

STRATEGIES = [
    'keep_as_is','drop','median_impute','mean_impute','knn_impute',
    'mode_impute','winsorize','log_transform','label_encode','onehot_encode','standardize',
]


def _to_python(obj):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _render_leakage_report(leakage_report: dict) -> None:
    """Render the leakage agent's structured report."""
    st.markdown("### 🛡 Leakage Detection Report")
    overall = leakage_report.get("overall_risk", "unknown")
    colour = {"low": "#22c55e", "medium": "#facc15", "high": "#ef4444"}.get(overall, "#888")
    st.markdown(
        f'<div style="background:#1a1a2e;border:1px solid {colour};border-radius:8px;'
        f'padding:0.75rem 1rem;margin-bottom:1rem;">'
        f'<span style="color:{colour};font-weight:700;">Overall risk: {overall.upper()}</span>'
        f'<br><span style="color:#aaa;font-size:.9rem;">{leakage_report.get("leakage_summary","")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    high_risk  = leakage_report.get("high_risk_columns", [])
    medium_risk = leakage_report.get("medium_risk_columns", [])
    dropped    = leakage_report.get("dropped_by_leakage", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk columns", len(high_risk))
    c2.metric("Medium-risk columns", len(medium_risk))
    c3.metric("Dropped by leakage", len(dropped))

    if dropped:
        alert(
            f"🚫 <strong>{len(dropped)} column(s) dropped</strong> due to leakage risk: "
            + ", ".join(f"<code>{c}</code>" for c in dropped),
            "error",
        )
    if high_risk:
        with st.expander("🔴 High-risk columns", expanded=True):
            for col in high_risk:
                verdict = leakage_report.get("llm_verdicts", {}).get(col, {})
                st.markdown(f"**`{col}`** — {verdict.get('reasoning', 'High leakage score')}")
    if medium_risk:
        with st.expander("🟡 Medium-risk columns"):
            for col in medium_risk:
                verdict = leakage_report.get("llm_verdicts", {}).get(col, {})
                st.markdown(f"**`{col}`** — {verdict.get('reasoning', 'Moderate leakage score')}")

    user_kept = leakage_report.get("user_kept_columns", [])
    if user_kept:
        kept_list = ", ".join(f"<code>{c}</code>" for c in user_kept)
        alert(
            "⚠️ User kept flagged columns for downstream modeling: " + kept_list,
            "warning",
        )


def page_eda(_rt):
    s = st.session_state.pipeline_state
    decisions = s.get("preprocessing_decisions", {})
    if not decisions:
        alert("Run the pipeline from Data Fusion first.", "warning")
        return

    tid = st.session_state.tid
    st.markdown(f'{badge("Phase 2")} <h1 style="display:inline;margin-left:.5rem;">EDA Agent</h1>', unsafe_allow_html=True)
    st.caption("Review dataset analysis, preprocessing decisions, and leakage detection.")

    analysis = s.get("eda_analysis", {})
    metrics_row([
        ("Columns analysed", analysis.get("n_columns_analysed", len(decisions))),
        ("Numeric cols",     len(analysis.get("numeric_columns", []))),
        ("High-null cols",   len(analysis.get("high_null_columns", []))),
        ("High-skew cols",   len(analysis.get("high_skew_columns", []))),
    ])

    if s.get("eda_report"):
        with st.expander("📊 EDA Agent Report", expanded=True):
            st.write(s["eda_report"])
            if s.get("global_notes"):
                st.caption(s["global_notes"])

    # ── Target distribution ──────────────────────────────────────────────────
    with st.expander("📈 Target distribution plot"):
        cached_b64 = _rt.get_key(tid, "_plt_target")
        if not cached_b64:
            if st.button("Generate target distribution plot"):
                full_state = get_state(tid)
                df = b64_to_df(full_state["df_parquet_b64"])
                tgt = full_state["target"]
                fig, ax = plt.subplots(figsize=(7, 3))
                fig.patch.set_facecolor("#1a1a2e")
                ax.set_facecolor("#1a1a2e")
                if full_state["problem_type"] == "classification":
                    vc = df[tgt].value_counts().sort_index()
                    ax.bar(vc.index.astype(str), vc.values, color="#c6f135", alpha=0.85, edgecolor="#888", linewidth=0.5)
                    ax.set_title("Class Distribution", color="#e0e0e0", fontsize=13, pad=10)
                    ax.set_xlabel("Class", color="#aaa")
                else:
                    ax.hist(df[tgt].dropna(), bins=40, color="#c6f135", alpha=0.85, edgecolor="#888", linewidth=0.3)
                    ax.set_title("Target Distribution", color="#e0e0e0", fontsize=13, pad=10)
                    ax.set_xlabel(tgt, color="#aaa")
                ax.set_ylabel("Count", color="#aaa")
                ax.tick_params(colors="#aaa", labelsize=10)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
                ax.yaxis.grid(True, color="#333", linewidth=0.5, linestyle="--")
                ax.set_axisbelow(True)
                plt.tight_layout()
                b64 = fig_b64(fig)
                plt.close(fig)
                _rt.set_key(tid, "_plt_target", b64)
                cached_b64 = b64
        if cached_b64:
            b64_image(cached_b64, "Target distribution")

    # ── Correlation heatmap ──────────────────────────────────────────────────
    with st.expander("🔥 Correlation heatmap"):
        cached_corr = _rt.get_key(tid, "_plt_corr")
        if not cached_corr:
            if st.button("Generate correlation heatmap"):
                try:
                    import seaborn as sns
                except ImportError:
                    alert("seaborn not installed — run: pip install seaborn", "warning")
                    return
                full_state = get_state(tid)
                df = b64_to_df(full_state["df_parquet_b64"])
                numeric_df = clean_numeric_frame_for_corr(df)
                if numeric_df.shape[1] >= 2:
                    corr = numeric_df.corr()
                    sz = max(8, min(14, numeric_df.shape[1]))
                    fig, ax = plt.subplots(figsize=(sz, sz * 0.8))
                    fig.patch.set_facecolor("#1a1a2e")
                    ax.set_facecolor("#1a1a2e")
                    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                    sns.heatmap(
                        corr, mask=mask, ax=ax,
                        annot=len(num_cols) <= 15, fmt=".2f",
                        cmap="coolwarm", center=0,
                        linewidths=0.4, linecolor="#2a2a3e",
                        annot_kws={"size": 9, "color": "white"},
                        cbar_kws={"shrink": 0.8},
                    )
                    cbar = ax.collections[0].colorbar
                    cbar.ax.yaxis.set_tick_params(color="#aaa")
                    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa")
                    cbar.ax.set_facecolor("#1a1a2e")
                    ax.set_title("Feature Correlation Matrix", color="#e0e0e0", fontsize=13, pad=12)
                    ax.tick_params(colors="#ccc", labelsize=9)
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    b64 = fig_b64(fig)
                    plt.close(fig)
                    _rt.set_key(tid, "_plt_corr", b64)
                    cached_corr = b64
                else:
                    alert("Not enough clean numeric columns for correlation heatmap.", "warning")
        if cached_corr:
            b64_image(cached_corr, "Correlation heatmap")

    # ── Preprocessing decisions ──────────────────────────────────────────────
    st.markdown("### Preprocessing decisions")
    st.caption("Review or override the agent's strategy for each column.")

    eda_approved     = bool(s.get("hitl_eda_approved"))
    leakage_approved = bool(s.get("hitl_leakage_approved"))
    overrides = {}

    header_cols = st.columns([3, 2, 4])
    header_cols[0].markdown("**Column**")
    header_cols[1].markdown("**Strategy**")
    header_cols[2].markdown("**Rationale**")

    for col_name, info in decisions.items():
        agent_strategy = info.get("strategy", "keep_as_is")
        c1, c2, c3 = st.columns([3, 2, 4])
        c1.code(col_name)
        if eda_approved:
            c2.write(agent_strategy)
            overrides[col_name] = agent_strategy
        else:
            sel = c2.selectbox(
                f"strategy_{col_name}", STRATEGIES,
                index=STRATEGIES.index(agent_strategy) if agent_strategy in STRATEGIES else 0,
                key=f"eda_strat_{col_name}", label_visibility="collapsed",
            )
            overrides[col_name] = sel
        c3.caption(info.get("rationale", ""))

    st.markdown("---")

    # ── STEP 1: EDA approval -> triggers leakage agent ────────────────────────
    if not eda_approved:
        if st.button("✅ Approve EDA & run Leakage Detection →", width="stretch"):
            with st.spinner("Leakage Detection Agent scanning for data leakage…"):
                try:
                    resolved = {
                        col: {
                            "strategy": str(overrides[col]),
                            "rationale": str(decisions[col].get("rationale", "")),
                        }
                        for col in decisions
                    }
                    resolved = _to_python(resolved)
                    # Graph runs: hitl_eda -> leakage_agent -> STOPS at hitl_leakage
                    state = resume_graph_sync(
                        {
                            "preprocessing_decisions": resolved,
                            "hitl_eda_approved": True,
                            "openai_api_key": st.session_state.openai_key,
                        },
                        tid,
                    )
                    if state.get("loop_verdict") == "error":
                        alert(f"❌ Pipeline error: {state.get('loop_reasoning','')}", "error")
                        return
                    # Stay on EDA page — leakage report now available for review
                    persist(tid, state, phase="eda")
                    st.session_state.pipeline_state = safe_state(state)
                    update_store(pipeline_state=safe_state(state), phase="eda")
                    st.success("✅ EDA approved — Leakage Detection complete. Review results below.")
                    st.rerun()
                except Exception as e:
                    alert(f"Leakage detection error: {e}", "error")
        return

    reopen_col1, reopen_col2 = st.columns(2)
    if reopen_col1.button("↩ Reopen EDA Decisions", width="stretch"):
        reopen_workflow_stage(
            "eda",
            tid=tid,
            phase="eda",
            note="[Workflow] Reopened EDA decisions for revision.",
        )

    # ── STEP 2: Show leakage report + approval -> triggers feature agent ───────
    leakage_report = s.get("leakage_report") or {}
    if leakage_report:
        _render_leakage_report(leakage_report)
    else:
        alert("ℹ️ No leakage issues detected — leakage agent found no flagged columns.", "info")

    if leakage_approved:
        alert("✅ EDA & Leakage approved — Feature Agent has already run. Navigate forward via the sidebar.", "success")
        if reopen_col2.button("↩ Reopen Leakage Review", width="stretch"):
            reopen_workflow_stage(
                "leakage",
                tid=tid,
                phase="eda",
                note="[Workflow] Reopened leakage review for revision.",
            )
        return

    if reopen_col2.button("↩ Reopen Leakage Review", width="stretch"):
        reopen_workflow_stage(
            "leakage",
            tid=tid,
            phase="eda",
            note="[Workflow] Reopened leakage review for revision.",
        )

    flagged_candidates = []
    for col_group in (
        leakage_report.get("dropped_by_leakage", []),
        leakage_report.get("high_risk_columns", []),
        leakage_report.get("medium_risk_columns", []),
    ):
        for col in col_group:
            if col not in flagged_candidates:
                flagged_candidates.append(col)

    kept_default = leakage_report.get("user_kept_columns", [])
    keep_flagged = []
    if flagged_candidates:
        st.markdown("### Leakage Overrides")
        st.caption(
            "If a flagged column is genuinely available at prediction time and important for the model, "
            "keep it here. This skips the automatic leakage drop, but the warning stays visible."
        )
        keep_flagged = st.multiselect(
            "Keep selected flagged columns",
            options=flagged_candidates,
            default=[c for c in kept_default if c in flagged_candidates],
            help="Use this only for columns you are confident are not target leakage.",
            key=f"leakage_keep_cols_{tid}",
        )

    st.markdown("---")
    if st.button("✅ Approve Leakage Report & run Feature Agent →", width="stretch"):
        with st.spinner("Feature Agent engineering features…"):
            try:
                next_report = dict(leakage_report)
                keep_set = set(keep_flagged)
                if keep_set:
                    next_report["user_kept_columns"] = list(keep_flagged)
                    next_report["dropped_by_leakage"] = [
                        c for c in leakage_report.get("dropped_by_leakage", [])
                        if c not in keep_set
                    ]
                    next_report["n_dropped"] = len(next_report["dropped_by_leakage"])
                else:
                    next_report["user_kept_columns"] = []

                next_decisions = _to_python(dict(s.get("preprocessing_decisions") or {}))
                for col in keep_flagged:
                    current = next_decisions.get(col, {})
                    rationale = str(current.get("rationale", ""))
                    if current.get("strategy") == "drop" and rationale.startswith("[Leakage Agent]"):
                        next_decisions[col] = {
                            "strategy": "keep_as_is",
                            "rationale": "[User Override] Kept after leakage review.",
                        }

                # Graph runs: hitl_leakage -> feature_agent -> STOPS at hitl_features
                state = resume_graph_sync(
                    {
                        "hitl_leakage_approved": True,
                        "leakage_report": _to_python(next_report),
                        "preprocessing_decisions": next_decisions,
                        "openai_api_key": st.session_state.openai_key,
                    },
                    tid,
                )
                if state.get("loop_verdict") == "error":
                    alert(f"❌ Pipeline error: {state.get('loop_reasoning','')}", "error")
                    return
                persist(tid, state, phase="features")
                st.session_state.pipeline_state = safe_state(state)
                st.session_state.phase = "features"
                update_store(pipeline_state=safe_state(state), phase="features")
                st.success(f"✅ Feature Agent proposed {len(state.get('feature_proposals', []))} features!")
                st.rerun()
            except Exception as e:
                alert(f"Feature agent error: {e}", "error")
