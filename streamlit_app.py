"""
streamlit_app.py - AutoML LangGraph v5 entry point

v5 additions on top of v4:
  - Central state store (ui/state_store.py) replaces scattered session_state
  - Pipeline flow visualization (ui/pipeline_viz.py)
  - Natural language query panel (ui/nl_query.py)
  - Multi-dataset comparison page (ui/pages/compare.py)
  - AutoEDA report tab in Ingest phase
  - Config-driven architecture (config.yaml)
  - Progress tracking per pipeline step

v5.1 additions:
  - Column Intelligence (LLM explains every column in plain English)
  - Target validation (blocks unusable targets: IDs, names, dates, wrong task type)
  - Model compatibility guard (clf-only vs reg-only models warn + block)
  - Feature Distribution Explorer page
  - model_agent fixes: HistGB -inf bug, CatBoost score regression, adaptive hyperparams

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
import os
import types
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="AutoML v5.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Singletons ----------------------------------------------------------------
from ui.runtime_store   import RuntimeStore
from ui.runtime_context import set_runtime_store
from ui.graph_helpers   import init_helpers
from ui.components      import APP_CSS
from ui.state_store     import get_store, update_store, reset_store

@st.cache_resource
def _get_runtime_store():
    return RuntimeStore()

_rt = _get_runtime_store()
init_helpers(None)
set_runtime_store(_rt)
st.session_state["runtime_store_obj"] = _rt

# Streamlit shim so agents that import streamlit still work
_fake_st = types.SimpleNamespace()
_fake_st.session_state = {"runtime_objects": _rt._raw, "runtime_store_obj": _rt}
sys.modules.setdefault("streamlit", _fake_st)

# -- CSS -----------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)

# -- Migrate legacy session_state keys to central store -----------------------
store = get_store()

for _old_key, _store_attr in [
    ("tid",            "tid"),
    ("phase",          "phase"),
    ("pipeline_state", "pipeline_state"),
    ("openai_key",     "openai_key"),
    ("dataset_name",   "dataset_name"),
]:
    if _old_key in st.session_state and not getattr(store, _store_attr, None):
        setattr(store, _store_attr, st.session_state[_old_key])

if store.tid is not None:
    st.session_state["tid"]            = store.tid
if store.phase:
    st.session_state["phase"]          = store.phase
if store.pipeline_state:
    st.session_state["pipeline_state"] = store.pipeline_state
st.session_state["openai_key"]         = store.openai_key
st.session_state["progress_msgs"]      = store.progress_msgs

if "pipeline_state" not in st.session_state:
    st.session_state["pipeline_state"] = {}

# -- Error banners -------------------------------------------------------------
_ps = st.session_state.get("pipeline_state", {})

# Generic pipeline error
if _ps.get("loop_verdict") == "error":
    from ui.components import alert
    src = _ps.get("_agent_error_source", "unknown agent")
    msg = _ps.get("loop_reasoning", "")
    alert(
        f"❌ <strong>Pipeline error in {src}</strong> — {msg}<br>"
        "Fix the issue and restart from the affected phase.",
        "error",
    )

# Target validation warning — surface if a bad target somehow slipped through
# (e.g. session was restored from a pre-validation run)
if _ps.get("target") and _ps.get("problem_type"):
    _target      = _ps["target"]
    _prob_type   = _ps["problem_type"]
    _col_descs   = _ps.get("column_descriptions", {})
    _phase       = st.session_state.get("phase", "fusion")

    # Only show on fusion or eda phase so it's not repeated on every page
    if _phase in ("fusion", "ingest", "eda"):
        try:
            import pandas as pd
            from utils.serialization import b64_to_df
            from utils.column_intelligence import validate_target

            _df_b64 = _ps.get("df_parquet_b64")
            if _df_b64:
                _df = b64_to_df(_df_b64)
                _v  = validate_target(_df, _target, _prob_type, _col_descs)
                if _v.severity == "warning":
                    from ui.components import alert
                    _titles = "; ".join(i["title"] for i in _v.issues)
                    alert(
                        f"⚠️ Target column <strong>'{_target}'</strong> has potential issues: {_titles}. "
                        "Go to Data Fusion to review.",
                        "warning",
                    )
        except Exception:
            pass  # non-fatal — don't break the app if validation fails

# -- Database ------------------------------------------------------------------
from db.session_store import init_db
init_db()

# -- Sidebar -------------------------------------------------------------------
from ui.sidebar import render_sidebar
render_sidebar()

# -- Page routing --------------------------------------------------------------
def _render_phase(phase: str) -> None:
    with st.container():
        if phase == "fusion":
            from ui.pages.fusion import page_fusion
            page_fusion()
            return
        if phase == "ingest":
            from ui.pages.ingest import page_ingest
            page_ingest()
            return
        if phase == "eda":
            from ui.pages.eda import page_eda
            page_eda(_rt)
            return
        if phase == "distributions":
            from ui.pages.distributions import page_distributions
            page_distributions(_rt)
            return
        if phase == "features":
            from ui.pages.features import page_features
            page_features()
            return
        if phase == "split":
            from ui.pages.split import page_split
            page_split(_rt)
            return
        if phase == "model_selection":
            from ui.pages.model_selection import page_model_selection
            page_model_selection(_rt)
            return
        if phase == "models":
            from ui.pages.models import page_models
            page_models(_rt)
            return
        if phase == "eval":
            from ui.pages.eval import page_eval
            page_eval(_rt)
            return
        if phase == "export":
            from ui.pages.export import page_export
            page_export(_rt)
            return

        from ui.pages.compare import page_compare
        page_compare()


def _last_valid_phase(state: dict, requested_phase: str) -> str:
    """Return the phase the user is allowed to view.

    Uses _compute_phase_gates() from ui.sidebar — single source of truth.
    """
    if requested_phase == "compare":
        return "compare"

    from ui.sidebar import _compute_phase_gates
    unlocked = _compute_phase_gates(state)

    if unlocked.get(requested_phase):
        return requested_phase

    phase_order = [
        "fusion", "eda", "distributions", "features", "split",
        "model_selection", "models", "eval", "export",
    ]
    for phase in reversed(phase_order):
        if unlocked.get(phase):
            return phase
    return "fusion"


# -- Session-level column intelligence cache cleanup --------------------------
# When user starts a new session (tid changes) we clear any stale col-intel
# cache keys so the next upload gets a fresh LLM analysis.
def _cleanup_col_intel_cache():
    stale = [k for k in st.session_state if k.startswith("_col_intel_")]
    current_tid = st.session_state.get("tid")
    has_active_upload = any(k.startswith("_df_") for k in st.session_state)
    has_merged_dataset = "_fusion_merged_b64" in st.session_state
    has_dataset_context = bool(st.session_state.get("dataset_name"))

    # Only purge when there is no active pipeline and no in-progress dataset.
    # Fresh uploads use _col_intel_* before a tid exists, so clearing here on
    # every rerun would reset the step flow back to step 1 after each click.
    if not current_tid and not (has_active_upload or has_merged_dataset or has_dataset_context) and stale:
        for k in stale:
            del st.session_state[k]

_cleanup_col_intel_cache()

# -- Render --------------------------------------------------------------------
current_phase   = st.session_state.get("phase", "fusion")
validated_phase = _last_valid_phase(
    st.session_state.get("pipeline_state", {}), current_phase
)
if validated_phase != current_phase:
    st.session_state["phase"] = validated_phase
    update_store(
        phase=validated_phase,
        pipeline_state=st.session_state.get("pipeline_state", {}),
    )
    current_phase = validated_phase

if st.session_state.get("_last_rendered_phase") != current_phase:
    st.session_state["_last_rendered_phase"] = current_phase

_render_phase(current_phase)
