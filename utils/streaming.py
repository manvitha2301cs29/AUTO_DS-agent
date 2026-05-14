"""
utils/streaming.py — Live LLM streaming for Streamlit (v2)

Updated NODE_META and ORDERED_NODES to include two new agents:
  - leakage_agent / hitl_leakage
  - ensemble_agent / hitl_ensemble

Everything else (streaming engine, persist, public API) unchanged from v1.
"""

from __future__ import annotations
import os
import time

import numpy as np
import streamlit as st

# ── Patch msgpack to handle numpy scalar types ─────────────────────────────────
# LangGraph's SQLite checkpointer uses msgpack to serialise PipelineState.
# Any numpy.float64 / int64 / bool_ / ndarray that leaks into agent output dicts
# causes "Type is not msgpack serializable: numpy.float64".
# We patch msgpack's Packer at import time so it transparently converts numpy
# scalars to their native Python equivalents before encoding.
try:
    import msgpack

    _original_packb = msgpack.packb

    def _numpy_safe_packb(data, **kwargs):
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                converted = [_convert(v) for v in obj]
                return converted if isinstance(obj, list) else tuple(converted)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        return _original_packb(_convert(data), **kwargs)

    msgpack.packb = _numpy_safe_packb
except ImportError:
    pass  # msgpack not installed — LangGraph will raise its own error


NODE_META: dict[str, dict] = {
    "eda_agent":          {"icon": "📊", "label": "EDA Agent",             "color": "#e3f2fd", "border": "#1565c0"},
    "leakage_agent":      {"icon": "🚨", "label": "Leakage Detection",     "color": "#fff3e0", "border": "#bf360c"},
    "feature_agent":      {"icon": "🔧", "label": "Feature Engineering",   "color": "#f3e5f5", "border": "#6a1b9a"},
    "split_agent":        {"icon": "✂️",  "label": "Split Agent",           "color": "#e8f5e9", "border": "#2e7d32"},
    "model_agent":        {"icon": "🔍", "label": "Model Selection",        "color": "#fff3e0", "border": "#e65100"},
    "eval_agent":         {"icon": "📈", "label": "Evaluation Agent",       "color": "#fce4ec", "border": "#880e4f"},
    "ensemble_agent":     {"icon": "🎯", "label": "Ensemble Learning",      "color": "#e8eaf6", "border": "#283593"},
    "orchestrator_agent": {"icon": "🤖", "label": "Orchestrator",           "color": "#e0f7fa", "border": "#006064"},
    "hitl_eda":           {"icon": "⏸️",  "label": "EDA gate",              "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_leakage":       {"icon": "⏸️",  "label": "Leakage gate",          "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_features":      {"icon": "⏸️",  "label": "Feature gate",          "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_split":         {"icon": "⏸️",  "label": "Split gate",            "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_models":        {"icon": "⏸️",  "label": "Model gate",            "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_ensemble":      {"icon": "⏸️",  "label": "Ensemble gate",         "color": "#f5f5f5", "border": "#bdbdbd"},
    "hitl_loop":          {"icon": "🔁",  "label": "Loop decision gate",    "color": "#fff9c4", "border": "#f9a825"},
}

ORDERED_NODES = [
    "eda_agent", "hitl_eda",
    "leakage_agent", "hitl_leakage",
    "feature_agent", "hitl_features",
    "split_agent", "hitl_split",
    "model_agent", "hitl_models",
    "eval_agent",
    "ensemble_agent", "hitl_ensemble",
    "orchestrator_agent", "hitl_loop",
]


def _run_with_streaming(graph, input_or_none, config: dict) -> dict:
    node_placeholders: dict[str, object] = {}
    node_accumulated:  dict[str, str]    = {}
    current_node: str  = ""
    final_state:  dict = {}

    progress_bar = st.progress(0, text="Starting agents…")
    status_empty = st.empty()
    stream_area  = st.container()

    def _ensure_node_ui(node: str) -> None:
        if node in node_placeholders:
            return
        meta = NODE_META.get(node, {"icon": "⚙️", "label": node, "color": "#fafafa", "border": "#9e9e9e"})
        with stream_area:
            st.markdown(
                f"<div style='background:{meta['color']};border-left:4px solid {meta['border']};"
                f"border-radius:6px;padding:8px 14px;margin:6px 0 2px 0;"
                f"font-weight:600;font-size:0.9rem;'>"
                f"{meta['icon']} {meta['label']}</div>",
                unsafe_allow_html=True,
            )
            node_placeholders[node] = st.empty()
            node_accumulated[node]  = ""

    def _update_progress(node: str) -> None:
        try:
            idx = ORDERED_NODES.index(node)
            pct = max(5, int((idx + 1) / len(ORDERED_NODES) * 100))
        except ValueError:
            pct = 50
        meta = NODE_META.get(node, {"icon": "⚙️", "label": node})
        progress_bar.progress(min(pct, 95), text=f"{meta['icon']} {meta['label']}…")
        status_empty.caption(f"Running: **{meta['label']}**")

    def _freeze_node(node: str) -> None:
        if node not in node_placeholders:
            return
        text = node_accumulated.get(node, "")
        if text:
            node_placeholders[node].markdown(
                f"<div style='font-size:0.85rem;line-height:1.5;white-space:pre-wrap;"
                f"color:#212121;padding:4px 0;'>{text}</div>",
                unsafe_allow_html=True,
            )
        else:
            node_placeholders[node].caption(
                "_No LLM output — agent used rule-based or sklearn computation._"
            )

    try:
        stream = graph.stream(input_or_none, config=config, stream_mode=["messages", "values"])
        for raw in stream:
            if isinstance(raw, tuple) and len(raw) == 2:
                mode, payload = raw
            else:
                mode, payload = "values", raw

            if mode == "messages":
                if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
                    continue
                msg_chunk, meta = payload
                node  = meta.get("langgraph_node", "")
                token = getattr(msg_chunk, "content", "")
                if not node or not token or not isinstance(token, str):
                    continue
                if node != current_node:
                    if current_node:
                        _freeze_node(current_node)
                    current_node = node
                    _ensure_node_ui(node)
                    _update_progress(node)
                node_accumulated[node] += token
                node_placeholders[node].markdown(
                    f"<div style='font-size:0.85rem;line-height:1.5;white-space:pre-wrap;'>"
                    f"{node_accumulated[node]}▌</div>",
                    unsafe_allow_html=True,
                )
            elif mode == "values":
                if not isinstance(payload, dict):
                    continue
                final_state.update(payload)
                if current_node:
                    _freeze_node(current_node)
                msgs = payload.get("agent_messages", [])
                if msgs:
                    status_empty.caption(f"✅ {msgs[-1]}")

    except StopIteration:
        pass
    except Exception as exc:
        progress_bar.empty()
        status_empty.error(f"Stream error: {exc}")
        raise

    if current_node:
        _freeze_node(current_node)
    progress_bar.progress(100, text="✅ Complete")
    time.sleep(0.4)
    progress_bar.empty()
    status_empty.empty()
    return final_state


def _persist_state(state: dict) -> None:
    try:
        from db import upsert_session, sync_agent_messages
        tid = st.session_state.get("thread_id")
        if not tid:
            return
        upsert_session(
            tid,
            dataset_name=st.session_state.get("dataset_name"),
            target=state.get("target"),
            problem_type=state.get("problem_type"),
            phase=st.session_state.get("phase", "ingest"),
            best_model=state.get("best_model_key"),
            cv_score=state.get("best_cv_score"),
            retry_count=state.get("retry_count", 0),
        )
        if state.get("agent_messages"):
            sync_agent_messages(tid, state["agent_messages"])
    except Exception:
        pass


_GRAPH_CACHE = None


def _get_compiled_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        from agents.pipeline import compile_graph
        _GRAPH_CACHE = compile_graph(db_path=os.getenv("AUTOML_DB_PATH", "automl_history.db"))
    return _GRAPH_CACHE


def run_graph_streaming(input_state: dict) -> dict:
    graph  = _get_compiled_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {**input_state, "_thread_id": st.session_state.thread_id}
    final = _run_with_streaming(graph, input_state, config)
    if final:
        st.session_state.agent_state = final
        _persist_state(final)
    return final


def resume_graph_streaming(patch: dict | None = None) -> dict:
    graph  = _get_compiled_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph.update_state(config, {**(patch or {}), "_thread_id": st.session_state.thread_id})
    final = _run_with_streaming(graph, None, config)
    if final:
        st.session_state.agent_state = final
        _persist_state(final)
    return final


def resume_graph_no_stream(patch: dict | None = None) -> dict:
    graph  = _get_compiled_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph.update_state(config, {**(patch or {}), "_thread_id": st.session_state.thread_id})
    final: dict = {}
    for event in graph.stream(None, config=config, stream_mode="values"):
        if isinstance(event, dict):
            final.update(event)
    if final:
        st.session_state.agent_state = final
        _persist_state(final)
    return final
