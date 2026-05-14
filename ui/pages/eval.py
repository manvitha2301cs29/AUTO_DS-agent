"""ui/pages/eval.py — Phase 6: Evaluation page."""
from __future__ import annotations
from typing import Any
import pandas as pd
import streamlit as st

from ui.components import alert, badge, metrics_row, b64_image
from ui.graph_helpers import resume_graph_sync, safe_state, persist, fig_b64, get_state
from ui.state_store import update_store
from utils.runtime import get_or_restore_runtime

BAR_COLOURS = [
    '#c6f135','#a8d62b','#8cba22','#739f1b','#5c8515',
    '#4a6d10','#3a560c','#2c4109','#1f2d06','#141e04',
]

VERDICT_META = {
    "accept":         ("✅", "Accept results",            "success"),
    "retry_features": ("🔧", "Retry Feature Engineering", "warning"),
    "retry_models":   ("🔄", "Retry Model Selection",     "warning"),
    "retry_both":     ("⚠️",  "Retry Features + Models",  "error"),
    "error":          ("❌", "Pipeline Error",            "error"),
}


def page_eval(_rt):
    s = st.session_state.pipeline_state

    if not s.get("best_model_key"):
        alert("Complete the Models phase first.", "warning")
        return

    if not s.get("eval_metrics"):
        st.markdown(f'{badge("Phase 6")} <h1 style="display:inline;margin-left:.5rem;">Evaluation</h1>', unsafe_allow_html=True)
        st.caption("Run the evaluation agent to score your best model.")
        alert(f"Best model selected: <strong>{s.get('best_model_key')}</strong> — ready to evaluate.", "info")
        if st.button("Run Evaluation Agent", width="stretch"):
            tid = st.session_state.tid
            with st.spinner("Evaluation Agent running…"):
                try:
                    full_state = get_state(tid)
                    rt, _ = get_or_restore_runtime(full_state, required=("best_model", "X_test_t", "y_test"))
                    missing_before = [k for k in ("best_model", "X_test_t", "y_test") if k not in rt]
                    if missing_before:
                        alert(
                            "Evaluation cannot start because runtime artifacts are missing: "
                            f"{', '.join(missing_before)}. Please go back to Split and run the model step again.",
                            "error",
                        )
                        return
                    state = resume_graph_sync(
                        {"openai_api_key": st.session_state.openai_key},
                        tid,
                    )
                    if state.get("loop_verdict") == "error":
                        alert(f"Evaluation error: {state.get('loop_reasoning','')}", "error")
                        return
                    if not state.get("eval_metrics"):
                        alert(
                            "Evaluation did not produce metrics. Please rerun the Model phase once and then try Evaluation again.",
                            "error",
                        )
                        return
                    persist(tid, state, phase="eval")
                    st.session_state.pipeline_state = safe_state(state)
                    st.session_state.phase = "eval"
                    update_store(pipeline_state=safe_state(state), phase="eval")
                    st.rerun()
                except Exception as e:
                    alert(f"Evaluation error: {e}", "error")
        return

    tid         = st.session_state.tid
    metrics     = s.get("eval_metrics", {})
    shap        = s.get("shap_importance", [])
    verdict     = s.get("loop_verdict", "accept")
    reasoning   = s.get("loop_reasoning", "")
    suggestion  = s.get("loop_suggestion", "")
    prob_type   = s.get("problem_type", "classification")
    eval_report = s.get("eval_report", "")

    st.markdown(f'{badge("Phase 6")} <h1 style="display:inline;margin-left:.5rem;">Evaluation</h1>', unsafe_allow_html=True)
    st.caption("Model performance, SHAP feature importance, and orchestrator verdict.")

    v_icon, v_label, v_kind = VERDICT_META.get(verdict, ("?", verdict, "info"))
    alert(f"{v_icon} <strong>Orchestrator verdict: {v_label}</strong><br>{reasoning}", v_kind)

    if suggestion:
        st.info(f"💡 **Orchestrator suggestion:** {suggestion}")

    if prob_type == "classification":
        items = [
            ("F1 weighted",  f"{metrics.get('f1_weighted', 0):.4f}"),
            ("Accuracy",     f"{metrics.get('accuracy', 0):.4f}"),
            ("ROC-AUC",      f"{metrics.get('roc_auc', 0):.4f}" if metrics.get("roc_auc") else "—"),
        ]
    else:
        items = [
            ("R²",   f"{metrics.get('r2', 0):.4f}"),
            ("RMSE", f"{metrics.get('rmse', 0):.4f}"),
            ("MAE",  f"{metrics.get('mae', 0):.4f}"),
        ]
    metrics_row(items, accent=True)

    # Eval report — structured markdown rendering
    with st.expander("📝 Eval Agent Report", expanded=True):
        if eval_report:
            st.markdown(eval_report)
        else:
            if st.button("📝 Generate Eval Report (streaming)"):
                full_state = __import__("ui.graph_helpers", fromlist=["get_state"]).get_state(tid)
                if full_state:
                    from utils.agent_utils import stream_llm_text
                    import json as _json
                    metrics_payload = {
                        k: v for k, v in full_state.get("eval_metrics", {}).items()
                        if k != "confusion_matrix"
                    }
                    eval_analysis = full_state.get("eval_analysis", {})
                    payload = {
                        "model_key":             full_state.get("best_model_key", "unknown"),
                        "problem_type":          full_state.get("problem_type", ""),
                        "metrics":               metrics_payload,
                        "train_metrics":         eval_analysis.get("train_metrics", {}),
                        "calibration_metrics":   eval_analysis.get("calibration_metrics", {}),
                        "prediction_confidence": eval_analysis.get("prediction_confidence", {}),
                        "top_features":          (full_state.get("shap_importance") or [])[:10],
                        "n_test_samples":        eval_analysis.get("n_test_samples", "?"),
                    }
                    from agents.eval_agent import SYSTEM_PROMPT as _EVAL_SYSTEM
                    placeholder = st.empty()
                    streamed = ""
                    for chunk in stream_llm_text(
                        api_key=st.session_state.openai_key,
                        model_name=__import__("os").getenv("EVAL_MODEL", "gpt-4o-mini"),
                        system_prompt=_EVAL_SYSTEM,
                        user_content=_json.dumps(payload, indent=2),
                        temperature=0.2, max_tokens=900,
                    ):
                        streamed += chunk
                        placeholder.markdown(streamed + "▌")
                    placeholder.markdown(streamed)

    # SHAP inline bars
    if shap:
        st.markdown("### 🎯 What drives the prediction?")
        sorted_shap = sorted(shap, key=lambda x: x["importance"], reverse=True)
        top_shap    = sorted_shap[:10]
        total_imp   = sum(x["importance"] for x in sorted_shap) or 1
        max_imp     = top_shap[0]["importance"] if top_shap else 1

        html = '<div style="margin-bottom:1rem;">'
        for i, f in enumerate(top_shap):
            pct     = f["importance"] / max_imp * 100
            contrib = f["importance"] / total_imp * 100
            colour  = BAR_COLOURS[min(i, len(BAR_COLOURS) - 1)]
            name    = (f["feature"][:28] + "…") if len(f.get("feature", "")) > 30 else f.get("feature", "")
            html += (
                f'<div class="shap-row">'
                f'<span style="font-size:.7rem;color:#9898c0;width:20px;text-align:right">{i+1}</span>'
                f'<span class="shap-label" title="{f.get("feature","")}">{name}</span>'
                f'<div class="shap-bar-wrap"><div class="shap-bar-fill" style="width:{pct:.1f}%;background:{colour}"></div></div>'
                f'<span class="shap-pct">{contrib:.1f}%</span>'
                f'</div>'
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with st.expander("📊 SHAP bar chart"):
        if shap:
            cached_shap = _rt.get_key(tid, "_plt_shap") if tid else None
            if not cached_shap:
                # Auto-generate from in-memory shap data (no button needed)
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import base64, io as _bio
                    top = sorted(shap, key=lambda x: x.get("importance", 0), reverse=True)[:15]
                    features = [f.get("feature", "?") for f in reversed(top)]
                    values   = [f.get("importance", 0) for f in reversed(top)]
                    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.4)))
                    colours  = ["#c6f135" if i >= len(features) - 3 else "#60a5fa" for i in range(len(features))]
                    ax.barh(features, values, color=colours)
                    ax.set_xlabel("Mean |SHAP value|")
                    ax.set_title("Feature Importance (SHAP)")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    buf = _bio.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                    plt.close(fig)
                    buf.seek(0)
                    cached_shap = base64.b64encode(buf.read()).decode()
                    if tid:
                        _rt.set_key(tid, "_plt_shap", cached_shap)
                except Exception as _e:
                    st.caption(f"Could not render SHAP chart: {_e}")
            if cached_shap:
                b64_image(cached_shap, "SHAP importance")
        else:
            st.caption("SHAP importance not yet computed. Run evaluation first.")

    if prob_type == "classification" and metrics.get("confusion_matrix"):
        with st.expander("📊 Confusion Matrix & ROC Curve", expanded=True):
            col_cm, col_roc = st.columns(2)
            with col_cm:
                # v5: use pre-computed b64 from eval_analysis
                cm_b64 = s.get("eval_analysis", {}).get("confusion_matrix_b64", "")
                if cm_b64:
                    b64_image(cm_b64, "Confusion Matrix")
                else:
                    cached_cm = _rt.get_key(tid, "_plt_cm")
                    if not cached_cm:
                        if st.button("Generate confusion matrix"):
                            from utils.ml_helpers import plot_confusion_matrix_fig
                            le  = _rt.get_key(tid, "label_encoder")
                            fig = plot_confusion_matrix_fig(metrics["confusion_matrix"], le.classes_ if le else None)
                            b64 = fig_b64(fig)
                            _rt.set_key(tid, "_plt_cm", b64)
                            cached_cm = b64
                    if cached_cm:
                        b64_image(cached_cm, "Confusion matrix")

            with col_roc:
                roc_b64 = s.get("eval_analysis", {}).get("roc_curve_b64", "")
                if roc_b64:
                    b64_image(roc_b64, "ROC Curve")
                else:
                    st.caption("ROC curve not available (multiclass or no predict_proba)")

    # ── v5: Error analysis ────────────────────────────────────────────────────
    error_analysis = s.get("eval_analysis", {}).get("error_analysis", {})
    if error_analysis:
        with st.expander("🔬 Error Analysis — Where the Model Fails", expanded=False):
            if prob_type == "classification":
                per_class = error_analysis.get("per_class_error_rates", {})
                if per_class:
                    st.markdown("**Per-class error rates**")
                    df_err = pd.DataFrame([
                        {"Class": cls, "Samples": v["n"], "Errors": v["errors"],
                         "Error Rate": f"{v['error_rate']:.1%}"}
                        for cls, v in per_class.items()
                    ])
                    st.dataframe(df_err, width="stretch", hide_index=True)
                confused = error_analysis.get("most_confused_pairs", [])
                if confused:
                    st.markdown("**Most confused class pairs**")
                    df_conf = pd.DataFrame(confused)
                    st.dataframe(df_conf, width="stretch", hide_index=True)
            else:
                res_stats = error_analysis.get("residual_stats", {})
                if res_stats:
                    st.markdown(f"**Residual stats** — mean: `{res_stats.get('mean', 0):.4f}`, "
                                f"std: `{res_stats.get('std', 0):.4f}`, "
                                f"P95 |error|: `{res_stats.get('p95_abs', 0):.4f}`")
                worst = error_analysis.get("worst_predictions", [])
                if worst:
                    st.markdown("**Worst predictions (largest residuals)**")
                    st.dataframe(pd.DataFrame(worst), width="stretch", hide_index=True)

    # ── v5: Retrain alert ──────────────────────────────────────────────────────
    retrain_alert = s.get("eval_analysis", {}).get("retrain_alert", "")
    if retrain_alert:
        alert(retrain_alert, "warning")

    # ── HITL loop ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### HITL Decision")

    if verdict == "accept":
        st.markdown("The orchestrator recommends **accepting** these results. Proceed to export, or override and retry.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Accept results & go to Export"):
            with st.spinner("Finalising…"):
                try:
                    state = resume_graph_sync(
                        {"hitl_loop_approved": True, "loop_verdict": "accept", "hitl_final_accepted": True},
                        tid,
                    )
                    persist(tid, state, phase="export")
                    st.session_state.pipeline_state = safe_state(state)
                    st.session_state.phase = "export"
                    update_store(pipeline_state=safe_state(state), phase="export")
                    st.rerun()
                except Exception as e:
                    alert(f"Accept error: {e}", "error")

    with st.expander("🔁 Override: retry pipeline"):
        user_verdict = st.radio(
            "Retry mode",
            ["retry_features", "retry_models", "retry_both"],
            format_func=lambda x: {
                "retry_features": "🔧 Retry features",
                "retry_models":   "🔄 Retry models",
                "retry_both":     "⚠️ Retry both",
            }[x],
        )
        feat_instructions  = st.text_area("Feature instructions (optional)")
        model_instructions = st.text_area("Model instructions (optional)")

        proposals = s.get("feature_proposals", [])
        if user_verdict in ("retry_features", "retry_both"):
            feat_names    = [p["name"] for p in proposals if p.get("_computable")]
            kept_features = st.multiselect("Keep these features", feat_names, default=feat_names[:3])
        else:
            kept_features = []

        if st.button("🔁 Execute retry"):
            with st.spinner("Running loop-back retry…"):
                try:
                    new_retry = s.get("retry_count", 0) + 1
                    patch: dict[str, Any] = {
                        "hitl_loop_approved": True,
                        "loop_verdict": user_verdict,
                        "retry_count": new_retry,
                        "user_feature_instructions": feat_instructions,
                        "user_model_instructions":   model_instructions,
                        "eval_metrics": None, "shap_importance": None, "eval_report": None,
                        "orchestrator_decision": None,
                        "hitl_models_approved": False, "hitl_final_accepted": False,
                        "openai_api_key": st.session_state.openai_key,
                    }
                    if user_verdict in ("retry_features", "retry_both"):
                        prev = {p["name"]: p for p in proposals}
                        kept = [prev[n] for n in kept_features if n in prev]
                        patch.update({
                            "feature_proposals": kept or None,
                            "selected_features": [f["name"] for f in kept],
                            "hitl_features_approved": False,
                            "hitl_split_approved":    False,
                            "best_model_key": None, "best_cv_score": None, "tuning_results": None,
                        })
                    new_state = resume_graph_sync(patch, tid)
                    dest = "features" if user_verdict in ("retry_features", "retry_both") else "models"
                    persist(tid, new_state, phase=dest)
                    st.session_state.pipeline_state = safe_state(new_state)
                    st.session_state.phase = dest
                    update_store(pipeline_state=safe_state(new_state), phase=dest)
                    st.success(f"✅ Retry complete — at {dest} phase")
                    st.rerun()
                except Exception as e:
                    alert(f"Loop error: {e}", "error")
