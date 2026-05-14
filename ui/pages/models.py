"""ui/pages/models.py — Phase 5: Model Agent page (redesigned UI)."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st

from ui.components import alert, badge, metrics_row, b64_image
from ui.graph_helpers import resume_graph_sync, safe_state, persist, fig_b64, get_state, update_graph_state
from ui.state_store import update_store
from ui.runtime_context import (
    append_training_event, clear_training_monitor,
    get_training_events, get_training_progress,
    set_runtime_store, set_training_progress,
)
from utils.serialization import sanitize_for_msgpack
from ui.workflow_controls import reopen_workflow_stage

_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=1)

_STAGE_LABELS = {
    "split": "Splitting data",
    "preprocessing": "Preprocessing features",
    "queued": "Starting Model Agent",
    "llm_recommendation": "Choosing model families",
    "tuning": "Tuning models",
    "final_fit": "Training final model",
    "complete": "Completed",
}

_STAGE_ICONS = {
    "queued":             "⏳",
    "llm_recommendation": "🧠",
    "tuning":             "⚙️",
    "final_fit":          "🏋️",
    "complete":           "✅",
}

_TRAINING_CSS = """
<style>
.train-shell {
  background: #0d0d18;
  border: 1px solid #1e1e35;
  border-radius: 16px;
  padding: 1.5rem 1.75rem;
  margin-top: 0.5rem;
}
.train-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
}
.train-title {
  font-size: 1rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: .02em;
}
.train-elapsed {
  margin-left: auto;
  font-size: 0.75rem;
  color: #9999bb;
  font-family: monospace;
}
.train-progress-wrap {
  background: #1f2040;
  border-radius: 8px;
  height: 8px;
  overflow: hidden;
  margin-bottom: 1.1rem;
}
.train-progress-fill {
  height: 8px;
  border-radius: 8px;
  background: linear-gradient(90deg, #6366f1, #3b82f6);
}
.stage-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 1.1rem;
}
.stage-pill {
  font-size: 0.68rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid #35366a;
  color: #9999bb;
  background: #191a35;
  letter-spacing: .03em;
  text-transform: uppercase;
}
.stage-pill.active {
  background: #1e1b3a;
  border-color: #6366f1;
  color: #a5b4fc;
}
.stage-pill.done {
  background: #0a2010;
  border-color: #16a34a;
  color: #4ade80;
}
.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.6rem;
  margin-bottom: 1.1rem;
}
.model-card {
  background: #191a35;
  border: 1px solid #2a2a40;
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  text-align: center;
}
.model-card.tuning {
  border-color: #6366f1;
  background: #1a1835;
}
.model-card.done {
  border-color: #16a34a;
  background: #0a2010;
}
.model-card.pending {
  opacity: 0.45;
}
.model-card .mc-name {
  font-size: 0.78rem;
  font-weight: 700;
  color: #ffffff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.model-card .mc-score {
  font-size: 0.72rem;
  color: #93c5fd;
  font-weight: 700;
  margin-top: 2px;
  font-family: monospace;
}
.model-card .mc-status {
  font-size: 0.68rem;
  color: #9999bb;
  margin-top: 2px;
}
.model-card.tuning .mc-status { color: #c4b5fd; font-weight: 600; }
.model-card.done   .mc-status { color: #6ee7b7; font-weight: 600; }
.event-log {
  background: #0a0a14;
  border: 1px solid #1f2040;
  border-radius: 8px;
  padding: 0.65rem 0.9rem;
  max-height: 160px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 0.72rem;
}
.event-row {
  display: flex;
  gap: 0.5rem;
  padding: 2px 0;
  border-bottom: 1px solid #14141e;
  color: #aaaacc;
  line-height: 1.5;
}
.event-row:last-child { border-bottom: none; color: #ffffff; font-weight: 500; }
.event-dot { color: #6366f1; flex-shrink: 0; }
.results-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.results-table th {
  text-align: left;
  padding: 0.45rem 0.8rem;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #9999bb;
  border-bottom: 1px solid #2a2a40;
}
.results-table td {
  padding: 0.55rem 0.8rem;
  color: #c9c9f5;
  border-bottom: 1px solid #1f2040;
}
.results-table tr:last-child td { border-bottom: none; }
.results-table tr:hover td { background: #191a35; }
.score-cell { color: #60a5fa; font-family: monospace; font-weight: 700; }
.best-cell { color: #fbbf24; font-size: 1rem; }
.rank-cell { color: #9090b8; font-size: 0.75rem; }
.status-ok  { color: #4ade80; font-size: 0.7rem; }
.status-err { color: #f87171; font-size: 0.7rem; }
.optuna-cfg {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.cfg-pill {
  background: #191a35;
  border: 1px solid #2a2a40;
  border-radius: 8px;
  padding: 0.4rem 0.85rem;
  font-size: 0.75rem;
  color: #aaaacc;
}
.cfg-pill strong { color: #c9c9f5; }
</style>
"""


def _run_model_pipeline(rt, state: dict, tid: str) -> dict:
    import os
    from agents.model_agent import model_agent
    from agents.eval_agent import eval_agent
    from agents.ensemble_agent import ensemble_agent
    from agents.orchestrator import orchestrator_agent
    from ui.graph_helpers import update_graph_state, get_state
    from utils.serialization import sanitize_for_msgpack
    set_runtime_store(rt)
    current = dict(get_state(tid))
    current.update(state)
    current["_thread_id"] = tid
    n_forced = len(current.get("_forced_model_candidates") or [])
    if n_forced > 0:
        os.environ["MAX_CANDIDATE_MODELS"] = str(max(3, n_forced))
    model_out = model_agent(current); current.update(model_out)
    eval_out  = eval_agent(current);  current.update(eval_out)
    ensemble_out = ensemble_agent(current); current.update(ensemble_out)
    orch_out  = orchestrator_agent(current); current.update(orch_out)
    final = sanitize_for_msgpack(current)
    update_graph_state(final, tid)
    return final


def _stage_pills_html(current_stage: str) -> str:
    stages = ["queued", "llm_recommendation", "tuning", "final_fit", "complete"]
    order = {s: i for i, s in enumerate(stages)}
    cur_idx = order.get(current_stage, 0)
    pills = []
    for s in stages:
        idx = order[s]
        cls = "done" if idx < cur_idx else ("active" if idx == cur_idx else "")
        icon = _STAGE_ICONS.get(s, "")
        label = _STAGE_LABELS.get(s, s)
        pills.append(f'<span class="stage-pill {cls}">{icon} {label}</span>')
    return f'<div class="stage-row">{"".join(pills)}</div>'


def _model_cards_html(candidates: list[str], current_model: str | None,
                      partial_results: list[dict]) -> str:
    # Filter out -inf scores (model crashed) — show as error instead of a number
    scored = {
        r["model_key"]: r.get("best_score")
        for r in partial_results
        if isinstance(r.get("best_score"), (int, float)) and r.get("best_score") != float("-inf")
    }
    failed = {
        r["model_key"]
        for r in partial_results
        if isinstance(r.get("best_score"), float) and r.get("best_score") == float("-inf")
    }
    cards = []
    for mk in candidates:
        if mk == current_model:
            cls, status = "tuning", "⚙️ tuning…"
        elif mk in scored:
            cls, status = "done", f"✓ {scored[mk]:.4f}"
        elif mk in failed:
            cls, status = "done", "✗ error"
        else:
            cls, status = "pending", "waiting"
        short = mk.replace("_", " ").replace("gradient boosting", "grad boost")
        score_html = f'<div class="mc-score">{scored[mk]:.4f}</div>' if mk in scored else ""
        cards.append(
            f'<div class="model-card {cls}">'
            f'<div class="mc-name">{short}</div>'
            f'{score_html}'
            f'<div class="mc-status">{status}</div>'
            f'</div>'
        )
    return f'<div class="model-grid">{"".join(cards)}</div>'


def _event_log_html(events: list[dict]) -> str:
    rows = [
        f'<div class="event-row"><span class="event-dot">›</span>{e.get("message","")}</div>'
        for e in events[-10:]
    ]
    inner = "".join(rows) if rows else '<div class="event-row">Waiting for updates…</div>'
    return f'<div class="event-log">{inner}</div>'


def _training_dashboard_html(pct: int, stage: str, elapsed_str: str,
                              candidates: list[str], current_model: str | None,
                              partial_results: list[dict], events: list[dict]) -> str:
    return (
        f'<div class="train-shell">'
        f'<div class="train-header">'
        f'<span class="train-title">⚙️ Model Training in Progress</span>'
        f'<span class="train-elapsed">Elapsed: {elapsed_str}</span>'
        f'</div>'
        f'<div class="train-progress-wrap">'
        f'<div class="train-progress-fill" style="width:{max(2, min(pct, 99))}%"></div>'
        f'</div>'
        f'{_stage_pills_html(stage)}'
        f'{_model_cards_html(candidates, current_model, partial_results)}'
        f'{_event_log_html(events)}'
        f'</div>'
    )


def _launch_model_training(_rt, s: dict, tid: str):
    """Training monitor — renders a self-contained dashboard, no background bleed."""
    st.markdown(_TRAINING_CSS, unsafe_allow_html=True)

    # Guard: previous rerun finished
    if st.session_state.get("_training_complete_tid") == tid:
        st.session_state.pop("_training_complete_tid", None)
        try:
            fresh = get_state(tid)
            if fresh and fresh.get("best_model_key"):
                st.session_state.pipeline_state = safe_state(fresh)
        except Exception:
            pass
        st.rerun()
        return

    display_candidates = [
        c.get("model_key", "") for c in (s.get("_forced_model_candidates") or [])
        if c.get("model_key")
    ]
    n_trials = int(s.get("optuna_trials", 20))
    cv_folds = int(s.get("cv_folds", 3))

    # Guard: already running (rerender during training)
    if st.session_state.get("_training_running_tid") == tid:
        st.markdown(
            f'<div class="optuna-cfg">'
            f'<div class="cfg-pill">🔬 <strong>{len(display_candidates)}</strong> models</div>'
            f'<div class="cfg-pill">🎯 <strong>{n_trials}</strong> trials</div>'
            f'<div class="cfg-pill">📐 <strong>{cv_folds}</strong>-fold CV</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        shell = st.empty()
        import time as _t
        started = _t.time()
        for _ in range(600):
            prog = get_training_progress(tid) or {}
            pct  = int(prog.get("percent", 1) or 1)
            stage = prog.get("stage", "queued")
            current_model = prog.get("current_model")
            events = get_training_events(tid) or []
            elapsed = int(_t.time() - started)
            elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
            shell.markdown(
                _training_dashboard_html(pct, stage, elapsed_str, display_candidates, current_model, [], events),
                unsafe_allow_html=True,
            )
            if prog.get("status") == "done" or pct >= 100:
                break
            _t.sleep(0.35)
        try:
            fresh = get_state(tid)
            if fresh and fresh.get("best_model_key"):
                st.session_state.pipeline_state = safe_state(fresh)
                st.session_state.pop("_training_running_tid", None)
                persist(tid, fresh, phase="models")
                update_store(pipeline_state=safe_state(fresh), phase="models")
        except Exception:
            pass
        st.rerun()
        return

    # ── First launch ──────────────────────────────────────────────────────────
    st.session_state["_training_running_tid"] = tid
    clear_training_monitor(tid)
    set_training_progress(tid, {"status": "running", "stage": "queued",
                                "percent": 5, "message": "Queuing model training…"})
    append_training_event(tid, "Model Agent queued.", stage="queued")

    future = _MODEL_EXECUTOR.submit(
        _run_model_pipeline,
        _rt,
        {
            "_thread_id": tid,
            "optuna_trials": n_trials,
            "cv_folds": cv_folds,
            "openai_api_key": st.session_state.openai_key,
            "hitl_split_approved": True,
            "hitl_models_approved": True,
            "hitl_model_selection_approved": bool(s.get("hitl_model_selection_approved", True)),
            "_forced_model_candidates": s.get("_forced_model_candidates") or [],
            "user_selected_models": s.get("user_selected_models") or [],
            "user_added_models": s.get("user_added_models") or [],
            "user_model_instructions": s.get("user_model_instructions", ""),
            "max_candidate_models": max(3, len(display_candidates)),
        },
        tid,
    )

    st.markdown(
        f'<div class="optuna-cfg">'
        f'<div class="cfg-pill">🔬 <strong>{len(display_candidates)}</strong> models selected</div>'
        f'<div class="cfg-pill">🎯 <strong>{n_trials}</strong> Optuna trials each</div>'
        f'<div class="cfg-pill">📐 <strong>{cv_folds}</strong>-fold cross-validation</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    shell = st.empty()
    started_at = time.time()
    partial_results: list[dict] = []

    while not future.done():
        prog = get_training_progress(tid) or {}
        pct   = int(prog.get("percent", 1) or 1)
        stage = prog.get("stage", "queued")
        current_model = prog.get("current_model")
        elapsed = int(time.time() - started_at)
        since   = max(0.0, time.time() - float(prog.get("updated_at") or started_at))
        events  = get_training_events(tid) or []

        if elapsed > 30 and since > 600:
            future.cancel()
            st.session_state.pop("_training_running_tid", None)
            shell.empty()
            alert("⏱️ Training timed out after 10 min of inactivity. Please retry.", "error")
            return

        # Parse live scores from event messages
        for e in events:
            msg = e.get("message", "")
            if "Finished" in msg and "score" in msg:
                for mk in display_candidates:
                    if mk.replace("_", " ") in msg.replace("_", " "):
                        try:
                            score_val = float(msg.split("score")[-1].strip().rstrip("."))
                            if not any(r["model_key"] == mk for r in partial_results):
                                partial_results.append({"model_key": mk, "best_score": score_val})
                        except Exception:
                            pass

        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
        shell.markdown(
            _training_dashboard_html(pct, stage, elapsed_str, display_candidates,
                                     current_model, partial_results, events),
            unsafe_allow_html=True,
        )
        time.sleep(0.35)

    # ── Done ──────────────────────────────────────────────────────────────────
    st.session_state.pop("_training_running_tid", None)
    shell.empty()

    try:
        final_state = future.result()
    except Exception as exc:
        alert(f"❌ Training error: {exc}", "error")
        return

    if final_state.get("loop_verdict") == "error":
        alert(f"❌ Pipeline error: {final_state.get('loop_reasoning', '')}", "error")
        return

    persist(tid, final_state, phase="models")
    new_ps = safe_state(final_state)
    st.session_state.pipeline_state = new_ps
    st.session_state.phase = "models"
    update_store(pipeline_state=new_ps, phase="models")

    best  = final_state.get("best_model_key", "?")
    score = final_state.get("best_cv_score", 0) or 0
    st.markdown(
        f'<div class="aml-alert aml-alert-success" style="font-size:1rem;font-weight:700;">'
        f'✅ Training complete — <strong>{best}</strong> &nbsp;CV = {score:.4f}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.session_state["_training_complete_tid"] = tid
    time.sleep(0.8)
    st.rerun()


def _score_or_neg_inf(row: dict) -> float:
    score = row.get("best_score")
    return float(score) if isinstance(score, (int, float)) else float("-inf")


def _results_table_html(tuning_results: list[dict], best_model_key: str) -> str:
    rows_sorted = sorted(tuning_results, key=_score_or_neg_inf, reverse=True)
    rows_html = ""
    for i, r in enumerate(rows_sorted):
        mk    = r.get("model_key", "?")
        score = r.get("best_score")
        is_ok = isinstance(score, (int, float)) and score != float("-inf")
        trials = r.get("n_trials_completed", 0)
        score_td  = f'<td class="score-cell">{score:.4f}</td>' if is_ok else '<td class="rank-cell">—</td>'
        err_msg   = r.get("error") or ("all trials returned -inf" if score == float("-inf") else "Failed")
        status_td = f'<td class="status-ok">✓ OK</td>' if is_ok else f'<td class="status-err">✗ {str(err_msg)[:28]}</td>'
        best_td   = '<td class="best-cell">🏆</td>' if mk == best_model_key else '<td></td>'
        rows_html += (
            f'<tr>'
            f'<td class="rank-cell">{i+1}</td>'
            f'<td style="font-weight:600;color:#c9c9f5">{mk}</td>'
            f'{score_td}'
            f'<td style="color:#aaaacc;font-size:.75rem">{trials}</td>'
            f'{status_td}'
            f'{best_td}'
            f'</tr>'
        )
    return (
        f'<table class="results-table">'
        f'<thead><tr><th>#</th><th>Model</th><th>CV Score</th><th>Trials</th><th>Status</th><th></th></tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def page_models(_rt):
    s = st.session_state.pipeline_state
    if not s.get("hitl_split_approved"):
        alert("Complete the Split phase first.", "warning")
        return

    tid = st.session_state.tid
    st.markdown(_TRAINING_CSS, unsafe_allow_html=True)

    # Reload fresh state if training completed externally
    if s.get("hitl_model_selection_approved") and not s.get("best_model_key"):
        try:
            fresh = get_state(tid)
            if fresh and fresh.get("best_model_key"):
                new_ps = safe_state(fresh)
                st.session_state.pipeline_state = new_ps
                update_store(pipeline_state=new_ps, phase="models")
                s = new_ps
        except Exception:
            pass

    # Launch training if approved but not yet done
    # Render ONLY the training dashboard — no model-selection cards or LLM rationales above it
    if s.get("hitl_model_selection_approved") and not s.get("best_model_key"):
        # Inject a scroll-to-top + clear previous page CSS to prevent stale content bleed
        st.markdown(
            "<style>section.main > div { padding-top: 1rem !important; }</style>"
            "<script>window.scrollTo(0,0);</script>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{badge("Phase 5")} <h1 style="display:inline;margin-left:.5rem;">Model Agent</h1>',
            unsafe_allow_html=True,
        )
        st.caption("Optuna hyperparameter tuning across all selected model families.")
        _launch_model_training(_rt, s, tid)
        return  # Hard return — nothing below renders during training

    if not s.get("best_model_key"):
        alert("Complete the Model Selection phase (4.5) first.", "warning")
        return

    # ── Results view ──────────────────────────────────────────────────────────
    best_model_key = s.get("best_model_key", "?")
    best_cv        = s.get("best_cv_score")
    tuning_results = s.get("tuning_results", [])
    retry_count    = s.get("retry_count", 0)
    analysis       = s.get("model_analysis", {})
    baseline_score = s.get("_baseline_score")
    approved       = bool(s.get("hitl_models_approved"))
    dummy_score    = analysis.get("dummy_baseline_score")

    st.markdown(
        f'{badge("Phase 5")} <h1 style="display:inline;margin-left:.5rem;">Model Agent</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Optuna-tuned model comparison and selection.")

    if retry_count > 0:
        alert(f"🔁 <strong>Retry {retry_count}</strong> — Model Agent tried new families based on feedback.", "info")

    improvement = (best_cv - baseline_score) if (best_cv is not None and baseline_score is not None) else None
    items = [
        ("Best model",    best_model_key.replace("_clf", "").replace("_reg", "")),
        ("CV score",      f"{best_cv:.4f}" if best_cv else "—"),
        ("Models tuned",  len(tuning_results)),
        ("Optuna trials", analysis.get("optuna_trials", 20)),
    ]
    if baseline_score is not None:
        items.append(("Naive baseline", f"{baseline_score:.4f}"))
    if improvement is not None:
        items.append(("Lift vs baseline", f"+{improvement:.4f}"))
    if dummy_score is not None:
        items.append(("Dummy model", f"{dummy_score:.4f}"))
        if best_cv is not None and dummy_score > 0:
            items.append(("Lift vs dummy", f"{best_cv - dummy_score:+.4f}"))
    metrics_row(items, accent=True)

    if tuning_results:
        st.markdown(
            f'<div class="aml-card">{_results_table_html(tuning_results, best_model_key)}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📊 Model comparison chart"):
        cached_b64 = _rt.get_key(tid, "_plt_model_cmp")
        if not cached_b64 and tuning_results:
            if st.button("Generate comparison chart"):
                from utils.ml_helpers import plot_model_comparison
                successful = [r for r in tuning_results if isinstance(r.get("best_score"), (int, float))]
                if successful:
                    fig = plot_model_comparison(successful)
                    b64 = fig_b64(fig)
                    _rt.set_key(tid, "_plt_model_cmp", b64)
                    cached_b64 = b64
                else:
                    alert("No successful model runs to plot.", "warning")
        if cached_b64:
            b64_image(cached_b64, "Model comparison")

    # ── All models final hyperparameters ──────────────────────────────────────
    tuning_results = s.get("tuning_results") or []
    if tuning_results or s.get("best_params"):
        with st.expander("⚙️ Final hyperparameters — all models", expanded=False):
            if tuning_results:
                for r in sorted(tuning_results, key=lambda x: x.get("best_score") or float("-inf"), reverse=True):
                    mk    = r.get("model_key", "?")
                    score = r.get("best_score")
                    params= r.get("best_params") or {}
                    score_str = f"{score:.4f}" if isinstance(score, float) and score != float("-inf") else "—"
                    is_best = mk == s.get("best_model_key")
                    label = f"{'🏆 ' if is_best else ''}{mk}  —  CV {score_str}"
                    with st.expander(label, expanded=is_best):
                        if params:
                            param_rows = [{"Parameter": k, "Value": str(v)} for k, v in sorted(params.items())]
                            st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)
                        else:
                            st.caption("No hyperparameters recorded.")
            elif s.get("best_params"):
                st.markdown(f"**{s.get('best_model_key', 'Best model')}**")
                param_rows = [{"Parameter": k, "Value": str(v)} for k, v in sorted(s["best_params"].items())]
                st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)

    if s.get("model_recommendation"):
        with st.expander("💡 Agent reasoning", expanded=True):
            st.markdown(s["model_recommendation"])

    st.markdown("---")
    if approved:
        alert("✅ Models approved — Evaluation complete. Navigate forward via the sidebar.", "success")
        back_col1, back_col2 = st.columns(2)
        if back_col1.button("↩ Reopen Model Selection", width="stretch"):
            reopen_workflow_stage(
                "model_selection",
                tid=tid,
                phase="model_selection",
                note="[Workflow] Reopened model selection after training review.",
            )
        if back_col2.button("↩ Reopen Split Settings", width="stretch"):
            reopen_workflow_stage(
                "split",
                tid=tid,
                phase="split",
                note="[Workflow] Reopened split settings after training review.",
            )
    else:
        if st.button("✅ Approve models & run Evaluation →"):
            with st.spinner("Eval Agent computing metrics + SHAP…"):
                try:
                    current_state = get_state(tid) or {}
                    if current_state.get("eval_metrics"):
                        patch = sanitize_for_msgpack({"hitl_models_approved": True})
                        update_graph_state(patch, tid)
                        new_ps = safe_state({**current_state, **patch})
                        st.session_state.pipeline_state = new_ps
                        st.session_state.phase = "eval"
                        update_store(pipeline_state=new_ps, phase="eval")
                        persist(tid, new_ps, phase="eval")
                        st.success("✅ Evaluation complete!")
                        st.rerun()
                    else:
                        from agents.eval_agent import eval_agent
                        from agents.ensemble_agent import ensemble_agent
                        from agents.orchestrator import orchestrator_agent
                        full = dict(current_state)
                        full["hitl_models_approved"] = True
                        full["openai_api_key"] = st.session_state.openai_key
                        full.update(eval_agent(full))
                        full.update(ensemble_agent(full))
                        full.update(orchestrator_agent(full))
                        final = sanitize_for_msgpack(full)
                        update_graph_state(final, tid)
                        if final.get("loop_verdict") == "error":
                            alert(f"❌ Pipeline error: {final.get('loop_reasoning', '')}", "error")
                            return
                        persist(tid, final, phase="eval")
                        st.session_state.pipeline_state = safe_state(final)
                        st.session_state.phase = "eval"
                        update_store(pipeline_state=safe_state(final), phase="eval")
                        st.success("✅ Evaluation complete!")
                        st.rerun()
                except Exception as e:
                    alert(f"Eval error: {e}", "error")
