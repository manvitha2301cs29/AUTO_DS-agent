"""
ui/sidebar.py — Sidebar navigation with pipeline flow viz + NL query (v5.1)

v5.1 additions:
  - Problem type badge (Classification / Regression) shown next to dataset name
  - Target column shown with AI type label so user always knows what they picked
  - ⚠️ indicator on navigation items that have active validation issues
    (e.g. ingest shows ⚠️ if target has warnings; model_selection shows ⚠️ if
     incompatible models were previously confirmed)
  - Column-intel cache is cleared on new session
"""
from __future__ import annotations
import os
import streamlit as st
from db.session_store import delete_session, list_sessions
from ui.graph_helpers import get_state, safe_state
from ui.runtime_context import get_runtime_store
from ui.state_store import get_store, update_store, reset_store
from ui.pipeline_viz import render_pipeline_flow, render_progress_bar

PHASES = [
    ("fusion",          "0 · Data Fusion"),
    ("eda",             "1 · EDA Agent"),
    ("distributions",   "📊 Distribution Explorer"),
    ("features",        "2 · Feature Engineering"),
    ("split",           "3 · Split Agent"),
    ("model_selection", "3.5 · Model Selection"),
    ("models",          "4 · Model Agent"),
    ("eval",            "5 · Evaluation"),
    ("export",          "6 · Export"),
    ("compare",         "📊 Compare Datasets"),
]
PHASE_ORDER = [p for p, _ in PHASES if p != "compare"]


def _compute_phase_gates(s: dict) -> dict:
    """
    Single source of truth for phase unlock gates.
    Accepts a pipeline_state dict and returns {phase_id: bool}.
    Both sidebar AND streamlit_app.py call this — do NOT duplicate this logic.
    """
    eda_done           = bool(s.get("eda_report") or s.get("preprocessing_decisions"))
    hitl_eda_done      = bool(s.get("hitl_eda_approved"))
    hitl_leakage_done  = bool(s.get("hitl_leakage_approved"))
    feature_started    = s.get("feature_proposals") is not None or bool(s.get("feature_analysis"))
    hitl_features_done = bool(s.get("hitl_features_approved"))
    hitl_split_done    = bool(s.get("hitl_split_approved"))
    model_done         = bool(s.get("best_model_key"))
    eval_done          = bool(s.get("eval_metrics"))

    return {
        "fusion":          True,
        "eda":             eda_done,
        "distributions":   eda_done,
        # Feature Engineering should unlock only after leakage review is approved
        # (or when feature proposals already exist in state). This keeps the
        # intended EDA -> Distribution Explorer -> Feature flow intact.
        "features":        hitl_leakage_done or feature_started,
        "split":           hitl_features_done,
        "model_selection": hitl_split_done,
        "models":          hitl_split_done,
        "eval":            model_done,
        "export":          eval_done,
        "compare":         True,
    }


def _phase_unlocked(p: str) -> bool:
    s = st.session_state.get("pipeline_state", {})
    return _compute_phase_gates(s).get(p, False)


def _derive_phase_from_state(state: dict, fallback: str = "fusion") -> str:
    gates = _compute_phase_gates(state)
    for phase in reversed(PHASE_ORDER):
        if gates.get(phase):
            return phase
    return fallback


# ── Validation indicator helpers ─────────────────────────────────────────────

def _target_has_warning(s: dict) -> bool:
    """True if current target + problem_type has any validation issues."""
    target    = s.get("target")
    prob_type = s.get("problem_type")
    df_b64    = s.get("df_parquet_b64")
    if not (target and prob_type and df_b64):
        return False
    try:
        from utils.serialization import b64_to_df
        from utils.column_intelligence import validate_target
        df = b64_to_df(df_b64)
        v  = validate_target(df, target, prob_type, s.get("column_descriptions", {}))
        return v.severity in ("warning", "error")
    except Exception:
        return False


def _model_selection_has_warning(s: dict) -> bool:
    """True if previously confirmed models contain incompatible ones."""
    try:
        from ui.pages.model_selection import _task_compat
        prob_type = s.get("problem_type", "classification")
        confirmed = s.get("user_selected_models", []) + s.get("user_added_models", [])
        return any(not _task_compat(mk, prob_type)[0] for mk in confirmed)
    except Exception:
        return False


def _phase_warning_icon(p_id: str, s: dict) -> str:
    """Return '⚠️ ' prefix if this phase has an active validation warning."""
    if p_id in ("fusion", "ingest") and _target_has_warning(s):
        return "⚠️ "
    if p_id == "model_selection" and _model_selection_has_warning(s):
        return "⚠️ "
    return ""


# ── Problem type badge ────────────────────────────────────────────────────────

def _problem_type_badge(prob_type: str) -> str:
    if prob_type == "classification":
        return (
            '<span style="background:#0d2e18;color:#6ee7a0;padding:2px 8px;'
            'border-radius:4px;font-size:0.72rem;font-weight:700;'
            'border:1px solid #59a14f33;">🏷 Classification</span>'
        )
    if prob_type == "regression":
        return (
            '<span style="background:#1a2e08;color:#d4f56a;padding:2px 8px;'
            'border-radius:4px;font-size:0.72rem;font-weight:700;'
            'border:1px solid #c6f13533;">📈 Regression</span>'
        )
    return ""


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚡ AutoML v5.1")
        st.markdown("---")

        store = get_store()

        # API key
        key_in = st.text_input(
            "OpenAI API Key",
            value=store.openai_key,
            type="password",
            key="_key_input",
        )
        if key_in != store.openai_key:
            update_store(openai_key=key_in)
            st.session_state["openai_key"] = key_in

        st.markdown("---")

        # ── Active dataset info ───────────────────────────────────────────────
        s         = st.session_state.get("pipeline_state", {})
        ds_name   = st.session_state.get("dataset_name") or store.dataset_name
        target    = s.get("target")
        prob_type = s.get("problem_type")

        if ds_name or target:
            st.markdown("**Active dataset**")
            if ds_name:
                st.caption(f"📁 {ds_name}")
            if prob_type:
                st.markdown(_problem_type_badge(prob_type), unsafe_allow_html=True)
            if target:
                # Show target with its AI type label if available
                col_descs  = s.get("column_descriptions", {})
                type_label = col_descs.get(target, {}).get("type_label", "")
                label_str  = f" · *{type_label}*" if type_label else ""
                has_warn   = _target_has_warning(s)
                warn_icon  = " ⚠️" if has_warn else " ✅"
                st.caption(f"🎯 Target: **{target}**{label_str}{warn_icon}")
                if has_warn:
                    st.caption(
                        "Target has validation issues — "
                        "click **0 · Data Fusion** to review."
                    )
            n_rows = s.get("n_rows")
            n_cols = s.get("n_cols")
            if n_rows and n_cols:
                st.caption(f"📊 {n_rows:,} rows × {n_cols} cols")
            st.markdown("---")

        # ── Pipeline progress bar ─────────────────────────────────────────────
        render_progress_bar()

        st.markdown("**Pipeline phases**")
        for p_id, p_label in PHASES:
            unlocked = _phase_unlocked(p_id)
            is_cur   = st.session_state.get("phase") == p_id
            try:
                done = (
                    p_id in PHASE_ORDER
                    and PHASE_ORDER.index(p_id)
                    < PHASE_ORDER.index(st.session_state.get("phase", "fusion"))
                )
            except ValueError:
                done = False

            warn_prefix = _phase_warning_icon(p_id, s)
            status_prefix = (
                "✅ " if done and not warn_prefix
                else (warn_prefix if warn_prefix else ("▶ " if is_cur else "🔒 " if not unlocked else "  "))
            )
            label = status_prefix + p_label

            if st.button(
                label,
                key=f"nav_{p_id}",
                use_container_width=True,
                disabled=not unlocked,
            ):
                st.session_state["phase"] = p_id
                update_store(phase=p_id)
                st.rerun()

        st.markdown("---")

        # ── NL Query panel ────────────────────────────────────────────────────
        with st.expander("🤖 Ask AutoML", expanded=False):
            from ui.nl_query import render_nl_query_panel
            render_nl_query_panel()

        st.markdown("---")
        st.markdown("**Session**")
        tid = st.session_state.get("tid") or store.tid
        if tid:
            st.code(tid[:16] + "…", language=None)
        if st.button("🗑 New session", width="stretch"):
            # Clear column-intelligence cache on new session
            stale_ci = [k for k in st.session_state if k.startswith("_col_intel_")]
            for k in stale_ci:
                del st.session_state[k]
            reset_store()
            st.session_state["tid"]            = None
            st.session_state["phase"]          = "fusion"
            st.session_state["pipeline_state"] = {}
            st.session_state["dataset_name"]   = ""
            st.rerun()

        with st.expander("Past sessions"):
            sessions = list_sessions()
            for sess in (sessions or [])[:8]:
                full_tid = sess.get("thread_id", "")
                s_id     = full_tid[:16]
                s_ph     = sess.get("phase", "?")
                ds_name_s = sess.get("dataset_name") or "Unnamed dataset"
                st.caption(f"{ds_name_s} — {s_id}… [{s_ph}]")
                restore_col, delete_col = st.columns([5, 1])
                if restore_col.button(
                    "Restore",
                    key=f"restore_{full_tid}",
                    width="stretch",
                ):
                    state_data     = get_state(full_tid)
                    restored_phase = _derive_phase_from_state(state_data, s_ph or "fusion")
                    st.session_state["tid"]            = full_tid
                    st.session_state["pipeline_state"] = safe_state(state_data)
                    st.session_state["phase"]          = restored_phase
                    st.session_state["dataset_name"]   = sess.get("dataset_name") or ""
                    update_store(
                        tid=full_tid,
                        phase=restored_phase,
                        pipeline_state=safe_state(state_data),
                        dataset_name=sess.get("dataset_name") or "",
                    )
                    st.rerun()
                if delete_col.button(
                    "Delete",
                    key=f"delete_{full_tid}",
                    width="stretch",
                    help="Delete this saved session",
                ):
                    delete_session(full_tid)
                    rt_store = get_runtime_store()
                    if rt_store is not None:
                        rt_store.delete(full_tid)
                    current_tid = st.session_state.get("tid") or store.tid
                    if current_tid == full_tid:
                        stale_ci = [k for k in st.session_state if k.startswith("_col_intel_")]
                        for k in stale_ci:
                            del st.session_state[k]
                        reset_store()
                        st.session_state["tid"]            = None
                        st.session_state["phase"]          = "fusion"
                        st.session_state["pipeline_state"] = {}
                        st.session_state["dataset_name"]   = ""
                    st.rerun()

        # ── Pipeline flow diagram ─────────────────────────────────────────────
        with st.expander("📊 Pipeline flow", expanded=False):
            render_pipeline_flow()
