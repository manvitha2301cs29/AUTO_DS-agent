"""
ui/graph_helpers.py — LangGraph execution helpers and session persistence.

Extracted from streamlit_app.py (Fix #3: split monolith).
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np

from utils.serialization import sanitize_for_msgpack


# ── These are injected at app startup by streamlit_app.py ─────────────────────
# graph and _cfg are set by calling init_helpers(graph) once.
_graph = None
_cfg_fn = None


def init_helpers(graph):
    global _graph, _cfg_fn
    _graph = graph
    _cfg_fn = lambda tid: {"configurable": {"thread_id": tid}}


def cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}


def _ensure_graph():
    """Compile the LangGraph lazily so the Streamlit UI can render quickly."""
    global _graph, _cfg_fn
    if _graph is None:
        from agents.pipeline import compile_graph
        from db.session_store import init_db, resolve_writable_db_path
        from utils.config_loader import cfg as app_cfg

        db_path = resolve_writable_db_path(
            os.getenv("AUTOML_DB_PATH", app_cfg("pipeline.db_path", default="db/automl_history.db"))
        )
        init_db(db_path)
        _graph = compile_graph(db_path=db_path)
        _cfg_fn = lambda tid: {"configurable": {"thread_id": tid}}
    return _graph


def get_state(tid: str) -> dict:
    try:
        graph = _ensure_graph()
        snap = graph.get_state(cfg(tid))
        return sanitize_for_msgpack(dict(snap.values)) if snap else {}
    except Exception:
        return {}


def run_graph_sync(input_state: dict, tid: str) -> dict:
    graph = _ensure_graph()
    input_state = sanitize_for_msgpack(dict(input_state))
    input_state["_thread_id"] = tid
    events = list(graph.stream(input_state, config=cfg(tid), stream_mode="values"))
    return sanitize_for_msgpack(events[-1]) if events else {}


def resume_graph_sync(patch: dict, tid: str) -> dict:
    graph = _ensure_graph()
    merged_state = sanitize_for_msgpack(get_state(tid))
    merged_state.update(sanitize_for_msgpack(dict(patch)))
    merged_state["_thread_id"] = tid
    graph.update_state(cfg(tid), merged_state)
    events = list(graph.stream(None, config=cfg(tid), stream_mode="values"))
    return sanitize_for_msgpack(events[-1]) if events else {}


def update_graph_state(patch: dict, tid: str) -> None:
    graph = _ensure_graph()
    merged_state = sanitize_for_msgpack(get_state(tid))
    merged_state.update(sanitize_for_msgpack(dict(patch)))
    merged_state["_thread_id"] = tid
    graph.update_state(cfg(tid), merged_state)


def np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def safe_state(state: dict) -> dict:
    """Strip large binary blobs before storing in st.session_state."""
    skip = {
        "df_parquet_b64", "df_engineered_parquet_b64",
        "_X_train_t_b64", "_X_test_t_b64",
        "_y_train_b64", "_y_test_b64",
        "_preprocessor_b64", "_label_encoder_b64",
        "openai_api_key",
    }
    return {k: v for k, v in state.items() if k not in skip}


def persist(tid: str, state: dict, dataset_name: str | None = None, phase: str | None = None):
    from db.session_store import upsert_session, sync_agent_messages
    upsert_session(
        tid,
        dataset_name=dataset_name,
        target=state.get("target"),
        problem_type=state.get("problem_type"),
        phase=phase or "ingest",
        best_model=state.get("best_model_key"),
        cv_score=state.get("best_cv_score"),
        retry_count=state.get("retry_count", 0),
    )
    if state.get("agent_messages"):
        sync_agent_messages(tid, state["agent_messages"])


def fig_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
