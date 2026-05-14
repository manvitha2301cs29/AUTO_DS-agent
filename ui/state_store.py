"""
ui/state_store.py — Central Streamlit state store (v5)

Replaces scattered st.session_state[key] references throughout the UI
with a single typed store. Provides:
  - AppState dataclass with all UI-level fields
  - get_store() / update_store() helpers
  - Pipeline progress tracking
  - NL query history

Usage:
    from ui.state_store import get_store, update_store, mark_phase_done

    store = get_store()
    store.phase          # "eda"
    store.pipeline_state # full PipelineState dict
    store.progress       # {"eda_agent": "done", "model_agent": "running", ...}

    update_store(phase="eval", pipeline_state=new_state)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Progress step definitions (in pipeline order)
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("ingest",        "📂 Data Fusion"),
    ("eda_agent",     "🔍 EDA Agent"),
    ("leakage_agent", "🛡 Leakage Check"),
    ("feature_agent", "⚙️  Feature Engineering"),
    ("split_agent",   "✂️  Train/Test Split"),
    ("model_agent",   "🤖 Model Training"),
    ("eval_agent",    "📊 Evaluation"),
    ("ensemble",      "🏆 Ensemble"),
    ("export",        "📦 Export"),
]

STEP_IDS   = [s for s, _ in PIPELINE_STEPS]
STEP_LABELS = {s: l for s, l in PIPELINE_STEPS}

STATUS_ICONS = {
    "pending":  "⏳",
    "running":  "🔄",
    "done":     "✅",
    "error":    "❌",
    "skipped":  "⏭",
}


# ─────────────────────────────────────────────────────────────────────────────
# App state dataclass
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_TO_STEP = {
    "fusion":           "ingest",
    "ingest":           "ingest",
    "eda":              "eda_agent",
    "features":         "feature_agent",
    "split":            "split_agent",
    "model_selection":  "model_agent",
    "models":           "model_agent",
    "eval":             "eval_agent",
    "export":           "export",
    "compare":          "export",   # compare has no dedicated step, map to export
}

_ORDERED_PROGRESS_STEPS = [
    "ingest",
    "eda_agent",
    "leakage_agent",
    "feature_agent",
    "split_agent",
    "model_agent",
    "eval_agent",
    "ensemble",
    "export",
]

@dataclass
class AppState:
    # ── Navigation ─────────────────────────────────────────────────────────
    phase: str = "fusion"
    tid: str | None = None
    dataset_name: str = ""

    # ── Pipeline state (PipelineState dict from LangGraph) ─────────────────
    pipeline_state: dict = field(default_factory=dict)

    # ── Auth ───────────────────────────────────────────────────────────────
    openai_key: str = ""

    # ── Progress tracking ──────────────────────────────────────────────────
    progress: dict[str, str] = field(default_factory=lambda: {
        s: "pending" for s in STEP_IDS
    })
    current_step: str = "ingest"
    progress_msgs: list[str] = field(default_factory=list)

    # ── NL query history ───────────────────────────────────────────────────
    nl_history: list[dict] = field(default_factory=list)   # [{query, result, ts}]

    # ── Multi-dataset tracking ─────────────────────────────────────────────
    datasets: list[dict] = field(default_factory=list)     # [{name, tid, metrics}]
    comparison_mode: bool = False

    # ── UI preferences ─────────────────────────────────────────────────────
    show_pipeline_viz: bool = True
    autoeda_html: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Store helpers
# ─────────────────────────────────────────────────────────────────────────────

_STORE_KEY = "_automl_v5_store"


def get_store() -> AppState:
    """Return the current AppState, initialising defaults if needed."""
    if _STORE_KEY not in st.session_state:
        st.session_state[_STORE_KEY] = AppState(
            openai_key=os.getenv("OPENAI_API_KEY", ""),
        )
    return st.session_state[_STORE_KEY]


def update_store(**kwargs) -> AppState:
    """Atomically update one or more AppState fields."""
    store = get_store()
    for k, v in kwargs.items():
        if hasattr(store, k):
            setattr(store, k, v)
    phase = kwargs.get("phase", store.phase)
    current_step = _PHASE_TO_STEP.get(phase)
    pipeline_state = kwargs.get("pipeline_state")
    if isinstance(pipeline_state, dict):
        store.progress = {step: "pending" for step in STEP_IDS}
        store.progress["ingest"] = "done"

        # ── Progress rules match the NEW pipeline order exactly ───────────────
        # Pipeline: eda → hitl_eda → leakage → hitl_leakage → feature → hitl_features
        #           → split → hitl_split → hitl_model_selection → model → hitl_models
        #           → eval → ensemble → hitl_ensemble → orchestrator → hitl_loop

        eda_done      = bool(pipeline_state.get("eda_report") or pipeline_state.get("preprocessing_decisions"))
        leakage_done  = bool(pipeline_state.get("hitl_leakage_approved"))          # leakage HITL explicitly approved
        feature_done  = bool(pipeline_state.get("hitl_features_approved"))         # features HITL approved → feature agent done
        split_done    = bool(pipeline_state.get("hitl_split_approved"))
        model_done    = bool(pipeline_state.get("best_model_key"))
        eval_done     = bool(pipeline_state.get("eval_metrics"))
        ensemble_done = bool(pipeline_state.get("ensemble_report"))
        export_done   = phase == "export" and eval_done

        if eda_done:      store.progress["eda_agent"]     = "done"
        if leakage_done:  store.progress["leakage_agent"] = "done"
        if feature_done:  store.progress["feature_agent"] = "done"
        if split_done:    store.progress["split_agent"]   = "done"
        if model_done:    store.progress["model_agent"]   = "done"
        if eval_done:     store.progress["eval_agent"]    = "done"
        if ensemble_done: store.progress["ensemble"]      = "done"
        if export_done:   store.progress["export"]        = "done"

        if pipeline_state.get("loop_verdict") == "error" and current_step:
            store.progress[current_step] = "error"

    if current_step and store.progress.get(current_step) == "pending":
        store.progress[current_step] = "done" if phase == "export" else "running"
        store.current_step = current_step
    return store


def mark_step(step_id: str, status: str) -> None:
    """Mark a pipeline step as pending/running/done/error/skipped."""
    store = get_store()
    if step_id in store.progress:
        store.progress[step_id] = status
        if status == "running":
            store.current_step = step_id


def mark_phase_done(phase: str) -> None:
    """Advance current phase and mark corresponding step as done."""
    store = get_store()
    store.progress[phase] = "done"


def add_progress_msg(msg: str) -> None:
    """Append a message to the running log."""
    store = get_store()
    store.progress_msgs.append(msg)
    if len(store.progress_msgs) > 200:
        store.progress_msgs = store.progress_msgs[-200:]


def add_nl_result(query: str, result: str) -> None:
    """Record an NL query + result."""
    from datetime import datetime, timezone
    store = get_store()
    store.nl_history.append({
        "query": query,
        "result": result,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def add_dataset(name: str, tid: str, metrics: dict, pipeline_state: dict | None = None) -> None:
    """Register a trained dataset for comparison."""
    store = get_store()
    ps = pipeline_state or {}
    store.datasets = [d for d in store.datasets if d.get("name") != name]
    store.datasets.append({
        "name":               name,
        "tid":                tid,
        "metrics":            metrics,
        "best_model_key":     ps.get("best_model_key", "—"),
        "cv_score":           ps.get("best_cv_score"),
        "problem_type":       ps.get("problem_type", "—"),
        "drift_severity":     ps.get("drift_severity", "—"),
        "drift_score":        ps.get("drift_score"),
        "n_drifted_features": ps.get("n_drifted_features", "—"),
    })


def reset_store() -> None:
    """Full reset for new session."""
    st.session_state[_STORE_KEY] = AppState(
        openai_key=os.getenv("OPENAI_API_KEY", ""),
    )
