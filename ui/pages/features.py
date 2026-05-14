"""ui/pages/features.py — Phase 3: Feature Engineering page."""
from __future__ import annotations
import streamlit as st

from utils.serialization import b64_to_df, df_to_b64
from ui.components import alert, badge, metrics_row
from ui.graph_helpers import get_state, resume_graph_sync, safe_state, persist
from ui.state_store import update_store
from ui.workflow_controls import reopen_workflow_stage


def page_features():
    s = st.session_state.pipeline_state
    proposals = s.get("feature_proposals", [])
    if not proposals:
        alert("EDA phase not complete.", "warning")
        return

    tid         = st.session_state.tid
    retry_count = s.get("retry_count", 0)
    analysis    = s.get("feature_analysis", {})
    approved    = bool(s.get("hitl_features_approved"))
    previously_selected = set(s.get("selected_features", []) or [])

    st.markdown(f'{badge("Phase 3")} <h1 style="display:inline;margin-left:.5rem;">Feature Engineering Agent</h1>', unsafe_allow_html=True)
    st.caption("Review AI-proposed features and select which to include.")

    if retry_count > 0:
        alert(f"🔁 <strong>Retry {retry_count}</strong> — Feature Agent proposed a new set based on orchestrator feedback.", "info")

    computable = [p for p in proposals if p.get("_computable")]
    metrics_row([
        ("Total proposed", len(proposals)),
        ("Computable",     len(computable)),
        ("Retry count",    retry_count),
    ], accent=True)

    if analysis.get("strategy_summary"):
        with st.expander("🧪 Feature Strategy", expanded=True):
            st.write(analysis["strategy_summary"])

    st.markdown("### Feature proposals")
    st.caption("Select which engineered features to include in your dataset.")

    RISK_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    selected_names = []

    for p in proposals:
        is_comp  = p.get("_computable", False)
        risk     = p.get("leakage_risk", "low")
        col_name = p["name"]

        check_col, info_col = st.columns([1, 10])
        if is_comp and not approved:
            checked = check_col.checkbox(
                f"Include feature {col_name}",
                value=(col_name in previously_selected) if previously_selected else True,
                key=f"feat_{col_name}",
                label_visibility="collapsed",
            )
        else:
            checked = is_comp
            check_col.write("✓" if is_comp else "✗")

        with info_col:
            risk_str = RISK_EMOJI.get(risk, "🟢")
            corr_str = f"  corr: `{p['_corr']:.3f}`" if p.get("_corr") is not None else ""
            vif_str  = f"  VIF: `{p['_vif']}`"         if p.get("_vif") is not None else ""
            badges_md = f"`{col_name}` {risk_str} {corr_str}{vif_str}"
            if not is_comp:
                badges_md += " ⚠️ *unresolvable*"
            st.markdown(badges_md)
            st.markdown(f'<div class="feat-formula">{p.get("formula","")}</div>', unsafe_allow_html=True)
            st.caption(f"💡 {p.get('benefit','')}")
            if risk != "low":
                st.caption(f"⚠️ {p.get('leakage_reason','')}")

        if is_comp and checked:
            selected_names.append(col_name)

    st.markdown("---")

    if approved:
        applied = s.get("selected_features", [])
        alert(f"✅ Features approved — {len(applied)} applied: {', '.join(f'`{f}`' for f in applied) or 'none'}.", "success")
        if st.button("↩ Reopen Feature Selection", width="stretch"):
            reopen_workflow_stage(
                "features",
                tid=tid,
                phase="features",
                note="[Workflow] Reopened feature selection for revision.",
            )
        return

    st.markdown(f"**{len(selected_names)}** of {len(computable)} computable features selected.")
    col_a, col_b = st.columns([3, 1])

    with col_a:
        if st.button(f"⚡ Apply {len(selected_names)} feature(s) & run Split Agent →"):
            with st.spinner("Split Agent analysing your data…"):
                try:
                    full_state = get_state(tid)
                    from agents.feature_agent import safe_eval_feature
                    df = b64_to_df(full_state["df_parquet_b64"])
                    df_eng = df.copy()
                    created = []
                    for p in proposals:
                        if p["name"] in selected_names and p.get("_computable"):
                            s_feat = safe_eval_feature(df_eng, p["formula"])
                            if s_feat is not None:
                                df_eng[p["name"]] = s_feat
                                created.append(p["name"])

                    state = resume_graph_sync(
                        {
                            "df_engineered_parquet_b64": df_to_b64(df_eng),
                            "selected_features": created,
                            "hitl_features_approved": True,
                            "openai_api_key": st.session_state.openai_key,
                        },
                        tid,
                    )
                    if state.get("loop_verdict") == "error":
                        alert(f"❌ Pipeline error: {state.get('loop_reasoning','')}", "error")
                        return
                    persist(tid, state, phase="split")
                    st.session_state.pipeline_state = safe_state(state)
                    st.session_state.phase = "split"
                    update_store(pipeline_state=safe_state(state), phase="split")
                    st.success(f"✅ {len(created)} features applied — Split Agent ready!")
                    st.rerun()
                except Exception as e:
                    alert(f"Feature apply error: {e}", "error")

    with col_b:
        if st.button("⏩ Skip features"):
            with st.spinner("Skipping features…"):
                try:
                    full_state = get_state(tid)
                    # Ensure df_engineered_parquet_b64 is set even when skipping,
                    # so downstream agents (split, model) always find a consistent df key.
                    eng_b64 = full_state.get("df_engineered_parquet_b64") or full_state.get("df_parquet_b64", "")
                    state = resume_graph_sync(
                        {
                            "selected_features": [],
                            "df_engineered_parquet_b64": eng_b64,
                            "hitl_features_approved": True,
                            "openai_api_key": st.session_state.openai_key,
                        },
                        tid,
                    )
                    persist(tid, state, phase="split")
                    st.session_state.pipeline_state = safe_state(state)
                    st.session_state.phase = "split"
                    update_store(pipeline_state=safe_state(state), phase="split")
                    st.success("✅ Skipped features — Split Agent ready!")
                    st.rerun()
                except Exception as e:
                    alert(f"Skip error: {e}", "error")
