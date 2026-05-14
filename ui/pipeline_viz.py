"""
ui/pipeline_viz.py — Pipeline flow visualization component (v5)

Renders an interactive pipeline DAG showing:
  - Each agent step with status (pending / running / done / error)
  - Current active step highlighted
  - Retry loops shown as back-edges
  - Clickable steps navigate to that phase

Usage:
    from ui.pipeline_viz import render_pipeline_flow
    render_pipeline_flow()   # renders inside current Streamlit column
"""

from __future__ import annotations
import streamlit as st
from ui.state_store import get_store, PIPELINE_STEPS, STATUS_ICONS


def _status_color(status: str) -> str:
    return {
        "pending":  "#313244",
        "running":  "#f59e0b",
        "done":     "#22c55e",
        "error":    "#ef4444",
        "skipped":  "#6b7280",
    }.get(status, "#313244")


def _status_text_color(status: str) -> str:
    return {
        "pending":  "#6b7280",
        "running":  "#fff8eb",
        "done":     "#dcfce7",
        "error":    "#fee2e2",
        "skipped":  "#9ca3af",
    }.get(status, "#cdd6f4")


def render_pipeline_flow() -> None:
    """Render the full pipeline as a horizontal flow diagram."""
    store = get_store()
    progress = store.progress

    # Build HTML for each step
    steps_html = ""
    for i, (step_id, label) in enumerate(PIPELINE_STEPS):
        status = progress.get(step_id, "pending")
        icon   = STATUS_ICONS.get(status, "⏳")
        color  = _status_color(status)
        text_c = _status_text_color(status)
        border = "2px solid #f59e0b" if status == "running" else f"1px solid {color}"
        shadow = "box-shadow: 0 0 12px #f59e0b88;" if status == "running" else ""

        step_html = f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:90px;">
          <div style="
            background:{color}; border:{border}; border-radius:10px;
            padding:10px 12px; text-align:center; cursor:pointer;
            {shadow} transition:all .2s; width:82px;
          ">
            <div style="font-size:18px">{icon}</div>
            <div style="font-size:10px; color:{text_c}; font-weight:600;
                        line-height:1.3; margin-top:4px">{label}</div>
          </div>
        </div>"""
        steps_html += step_html

        # Arrow connector (except after last step)
        if i < len(PIPELINE_STEPS) - 1:
            arrow_color = "#4b5563" if status == "pending" else "#6366f1"
            steps_html += f"""
            <div style="display:flex;align-items:center;padding:0 4px">
              <div style="color:{arrow_color};font-size:20px;font-weight:700">›</div>
            </div>"""

    # Retry annotation
    retry_count = store.pipeline_state.get("retry_count", 0)
    retry_html = ""
    if retry_count > 0:
        verdict = store.pipeline_state.get("loop_verdict", "")
        retry_html = f"""
        <div style="margin-top:8px; font-size:11px; color:#f59e0b;
                    background:#2d1f00; border:1px solid #78350f;
                    border-radius:6px; padding:4px 10px; display:inline-block">
          🔁 Retry loop active (attempt {retry_count}/3) — verdict: {verdict}
        </div>"""

    html = f"""
    <div style="overflow-x:auto; padding:12px 0 4px">
      <div style="display:flex; align-items:center; gap:2px; min-width:max-content; padding:4px">
        {steps_html}
      </div>
      {retry_html}
    </div>"""

    st.markdown(html, unsafe_allow_html=True)


def render_progress_bar() -> None:
    """Compact progress bar showing overall completion percentage."""
    store = get_store()
    progress = store.progress
    total = len(progress)
    done  = sum(1 for s in progress.values() if s == "done")
    pct   = int(done / total * 100) if total else 0

    st.progress(pct / 100, text=f"Pipeline progress: {done}/{total} steps complete ({pct}%)")


def render_step_log(max_msgs: int = 30) -> None:
    """Render the last N agent messages as a scrollable log."""
    store = get_store()
    msgs  = store.pipeline_state.get("agent_messages", [])
    if not msgs:
        return

    last_msgs = msgs[-max_msgs:]
    log_rows  = ""
    for msg in reversed(last_msgs):
        color = "#f87171" if "❌" in str(msg) else "#86efac" if "✅" in str(msg) else "#a6adc8"
        log_rows += f'<div style="font-size:11px; color:{color}; padding:2px 0; font-family:monospace">{msg}</div>'

    st.markdown(f"""
    <div style="background:#1e1e2e; border:1px solid #313244; border-radius:8px;
                padding:10px; max-height:200px; overflow-y:auto;">
      {log_rows}
    </div>""", unsafe_allow_html=True)
