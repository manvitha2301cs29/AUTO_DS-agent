"""
ui/pages/model_selection.py — Phase 4.5: HITL Model Selection

New in v3:
  - LLM recommends candidate models based on dataset characteristics
  - User can deselect LLM-recommended models
  - User can add extra models not in the LLM list
  - User can set per-model custom instructions
  - Imbalance-aware: shows loss function and class_weight settings
  - Approved selection is passed to model_agent via state
"""
from __future__ import annotations
import json
import os
import streamlit as st
import pandas as pd

from ui.components import alert, badge, metrics_row, card
from ui.graph_helpers import get_state, safe_state, persist, update_graph_state, resume_graph_sync
from ui.state_store import update_store
from utils.serialization import sanitize_for_msgpack
from ui.workflow_controls import reopen_workflow_stage


# All valid model keys the model_agent knows about
ALL_MODEL_KEYS = [
    "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees",
    "hist_gradient_boosting", "gradient_boosting", "logistic_regression",
    "ridge", "lasso", "elasticnet", "svm", "mlp", "adaboost",
]

MODEL_DESCRIPTIONS = {
    "xgboost":               "XGBoost — fast gradient boosting, handles missing values natively",
    "lightgbm":              "LightGBM — leaf-wise boosting, very fast on large datasets",
    "catboost":              "CatBoost — handles categoricals natively, minimal preprocessing",
    "random_forest":         "Random Forest — robust bagging ensemble, low variance",
    "extra_trees":           "Extra Trees — randomised splits, fast, low correlation with RF",
    "hist_gradient_boosting":"Histogram GBM (sklearn) — fast native boosting with NaN support",
    "gradient_boosting":     "Gradient Boosting (sklearn) — classic boosting, slower but solid",
    "logistic_regression":   "Logistic Regression — linear, fast, interpretable baseline",
    "ridge":                 "Ridge Regression — L2 regularised linear model (regression only)",
    "lasso":                 "Lasso — L1 sparse linear model (regression only)",
    "elasticnet":            "ElasticNet — combined L1+L2 regularisation (regression only)",
    "svm":                   "SVM — high-dimensional kernel classifier, slow on large data",
    "mlp":                   "MLP Neural Net — multi-layer perceptron, non-linear",
    "adaboost":              "AdaBoost — sequential boosting of weak learners (classification)",
}

# ── Model compatibility ───────────────────────────────────────────────────────
# Models that ONLY work for one task type.
# "both" models are not listed here — they work for either.
_CLF_ONLY = frozenset({
    "logistic_regression",  # outputs class probabilities; no regression variant
    "adaboost",             # AdaBoostClassifier only (no regressor in pipeline)
})
_REG_ONLY = frozenset({
    "ridge",        # RidgeRegressor — predicts continuous values only
    "lasso",        # Lasso — regression only
    "elasticnet",   # ElasticNet — regression only
})

# Human-readable reason shown in the warning
_COMPAT_REASON = {
    "logistic_regression": "Logistic Regression outputs class probabilities — it cannot predict continuous values.",
    "adaboost":            "AdaBoost (classifier variant) predicts discrete class labels, not continuous targets.",
    "ridge":               "Ridge Regression predicts continuous numeric values — it cannot predict class labels.",
    "lasso":               "Lasso is a regression model that minimises continuous prediction error.",
    "elasticnet":          "ElasticNet combines L1+L2 for regression — it has no classification output.",
}

def _task_compat(model_key: str, problem_type: str) -> tuple[bool, str]:
    """
    Returns (is_compatible, reason_string).
    is_compatible=False means this model CANNOT be used for problem_type.
    """
    if problem_type == "regression" and model_key in _CLF_ONLY:
        return False, _COMPAT_REASON.get(model_key, f"'{model_key}' is a classification-only model.")
    if problem_type == "classification" and model_key in _REG_ONLY:
        return False, _COMPAT_REASON.get(model_key, f"'{model_key}' is a regression-only model.")
    return True, ""
    """Ask the LLM for model recommendations given dataset context."""
    from utils.agent_utils import call_llm_json
    from utils.config_loader import cfg
    import numpy as np

    model_name = os.getenv("MODEL_AGENT_MODEL") or os.getenv("OPENAI_MODEL", cfg("llm.default_model", default="gpt-4o-mini"))

    imbalance = state.get("imbalance_analysis") or {}
    imbalance_recs = state.get("imbalance_recommendations") or {}
    n_rows = state.get("n_rows", 0)
    n_cols = state.get("n_cols", 0)
    problem_type = state.get("problem_type", "classification")

    # Build the allowed model key list dynamically so the LLM never recommends
    # a classification-only model for a regression task (or vice versa).
    allowed_keys = [
        k for k in ALL_MODEL_KEYS
        if _task_compat(k, problem_type)[0]
    ]

    SYSTEM = f"""\
You are a senior ML engineer recommending models for a user to choose from.

Given dataset metadata, recommend 4-6 models ranked by expected performance.
For each model, explain WHY it suits this specific dataset.
Consider: dataset size, class imbalance, feature types, problem type.

IMPORTANT — problem_type is '{problem_type}'.
Only recommend models from the allowed list below. Do NOT suggest models that are
incompatible with the task type (e.g. logistic_regression for regression, or
ridge/lasso/elasticnet for classification).

For imbalanced classification: prefer tree-based models that support class_weight
(xgboost, lightgbm, random_forest, hist_gradient_boosting).
For small datasets (<500 rows): prefer simpler models (logistic_regression, random_forest).
For large datasets (>50k rows): prefer lightgbm or hist_gradient_boosting for speed.

Return ONLY valid JSON, no markdown:
{{
  "candidates": [
    {{
      "model_key": "<one of the allowed model keys>",
      "rank": 1,
      "rationale": "<2 sentences on why this fits the dataset>",
      "expected_strength": "<high|medium|low>",
      "caveat": "<one sentence limitation or watch-out>"
    }}
  ],
  "overall_recommendation": "<2 sentences on overall strategy>"
}}

Allowed model keys: {', '.join(allowed_keys)}
"""
    payload = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "problem_type": problem_type,
        "imbalance_severity": imbalance.get("severity", "unknown"),
        "imbalance_ratio": imbalance.get("imbalance_ratio"),
        "n_classes": imbalance.get("n_classes"),
        "recommended_loss": imbalance_recs.get("recommended_loss"),
        "class_weight": imbalance_recs.get("class_weight"),
        "has_catboost": True,
        "user_model_instructions": state.get("user_model_instructions", ""),
    }
    result, err = call_llm_json(
        api_key=api_key,
        model_name=model_name,
        system_prompt=SYSTEM,
        user_content=json.dumps(payload, indent=2, default=str),
        temperature=0.2,
        max_tokens=1500,
    )
    if result is None or not result.get("candidates"):
        # fallback
        fallback = ["xgboost", "lightgbm", "random_forest", "hist_gradient_boosting"]
        return [
            {"model_key": k, "rank": i+1, "rationale": MODEL_DESCRIPTIONS.get(k, k),
             "expected_strength": "high" if i == 0 else "medium", "caveat": ""}
            for i, k in enumerate(fallback)
        ], ""
    return result.get("candidates", []), result.get("overall_recommendation", "")


def page_model_selection(_rt):
    s = st.session_state.pipeline_state
    if not s.get("hitl_split_approved"):
        alert("Complete the Split phase first.", "warning")
        return

    tid = st.session_state.tid
    approved = bool(s.get("hitl_model_selection_approved"))

    st.markdown(
        f'{badge("Phase 4.5")} <h1 style="display:inline;margin-left:.5rem;">Model Selection</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Review LLM-recommended models, deselect ones you don't want, or add your own.")

    if approved:
        selected = s.get("user_selected_models", [])
        added = s.get("user_added_models", [])
        alert(f"✅ Model selection approved — {len(selected) + len(added)} model(s) queued for training.", "success")
        metrics_row([
            ("Selected models", len(selected)),
            ("User-added models", len(added)),
            ("Total to train", len(selected) + len(added)),
        ])
        if st.button("↩ Reopen Model Selection", width="stretch"):
            reopen_workflow_stage(
                "model_selection",
                tid=tid,
                phase="model_selection",
                note="[Workflow] Reopened model selection for revision.",
            )
        return

    # ── Imbalance advisory ────────────────────────────────────────────────────
    imbalance = s.get("imbalance_analysis") or {}
    imbalance_recs = s.get("imbalance_recommendations") or {}
    severity = imbalance.get("severity", "balanced")
    if severity in ("moderate", "severe"):
        alert(
            f"⚖️ <strong>Imbalance detected ({severity}, ratio {imbalance.get('imbalance_ratio', '?')}:1)</strong> — "
            f"Recommended loss: <code>{imbalance_recs.get('recommended_loss', 'weighted_cross_entropy')}</code>. "
            f"Prefer models that support <code>class_weight='balanced'</code>.",
            "warning",
        )

    # ── Load or fetch LLM candidates ─────────────────────────────────────────
    llm_candidates = s.get("llm_model_candidates") or []
    overall_rec = ""

    if not llm_candidates:
        with st.spinner("Asking LLM to recommend models for your dataset…"):
            try:
                api_key = st.session_state.openai_key
                full_state = get_state(tid)
                llm_candidates, overall_rec = _get_llm_candidates(full_state or s, api_key)
                # Store in state so we don't re-fetch on rerun
                update_graph_state(sanitize_for_msgpack({
                    "llm_model_candidates": llm_candidates,
                }), tid)
                st.session_state.pipeline_state["llm_model_candidates"] = llm_candidates
            except Exception as e:
                alert(f"LLM recommendation error: {e}", "error")
                llm_candidates = [
                    {"model_key": k, "rank": i+1, "rationale": MODEL_DESCRIPTIONS.get(k, k),
                     "expected_strength": "medium", "caveat": ""}
                    for i, k in enumerate(["xgboost", "lightgbm", "random_forest"])
                ]

    if overall_rec:
        with st.expander("💡 LLM Overall Recommendation", expanded=True):
            st.markdown(overall_rec)

    # ── Model selection UI ────────────────────────────────────────────────────
    st.markdown("### LLM-Recommended Models")
    st.caption("Select which models to train. You can deselect models or add your own below.")

    strength_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}

    llm_keys = [c["model_key"] for c in llm_candidates]
    problem_type = s.get("problem_type", "classification")

    # ── Filter out any LLM-hallucinated incompatible candidates ──────────────
    compat_llm_candidates = []
    for cand in llm_candidates:
        mk = cand.get("model_key", "")
        ok, reason = _task_compat(mk, problem_type)
        if not ok:
            # Remove and add a note — LLM shouldn't have suggested this
            cand = dict(cand)
            cand["_incompatible"] = True
            cand["_reason"] = reason
        compat_llm_candidates.append(cand)

    selected_models = []
    incompatible_selected = []
    previously_selected = set(s.get("user_selected_models", []) or [])

    for cand in compat_llm_candidates:
        mk = cand.get("model_key", "")
        sb = strength_badge.get(cand.get("expected_strength", "medium"), "🟡")
        is_incompat = cand.get("_incompatible", False)
        reason      = cand.get("_reason", "")

        col_chk, col_info = st.columns([1, 8])
        with col_chk:
            # Default incompatible models to unchecked
            default_checked = (mk in previously_selected) if previously_selected else (not is_incompat)
            checked = st.checkbox("", value=default_checked and not is_incompat, key=f"model_chk_{mk}")
        with col_info:
            if is_incompat:
                st.markdown(
                    f'<div style="background:#2e1010;border-left:4px solid #e15759;'
                    f'border-radius:6px;padding:8px 12px;margin:2px 0;">'
                    f'<span style="color:#fca5a5;font-weight:700;">⚠️ {mk}</span>'
                    f'<span style="color:#aaa;font-size:0.85rem;"> — incompatible with <strong>{problem_type}</strong></span><br>'
                    f'<span style="color:#fcd34d;font-size:0.82rem;">❌ {reason}</span><br>'
                    f'<span style="color:#7ec8fa;font-size:0.82rem;">💡 Go back to Data Fusion and change the problem type, or deselect this model.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"**{sb} {mk}** — {cand.get('rationale', MODEL_DESCRIPTIONS.get(mk, ''))}"
                )
                caveat = cand.get("caveat", "")
                if caveat:
                    st.caption(f"⚠️ {caveat}")

        if checked:
            selected_models.append(mk)
            if is_incompat:
                incompatible_selected.append((mk, reason))

    # ── User-added models ─────────────────────────────────────────────────────
    st.markdown("### ➕ Add Your Own Models")
    st.caption("Pick extra models not in the LLM list above.")

    llm_keys = [c["model_key"] for c in llm_candidates]
    available_extra = [k for k in ALL_MODEL_KEYS if k not in llm_keys]

    def _extra_label(k: str) -> str:
        ok, _ = _task_compat(k, problem_type)
        prefix = "⚠️ INCOMPATIBLE — " if not ok else ""
        return f"{prefix}{k} — {MODEL_DESCRIPTIONS.get(k, k)}"

    user_added = st.multiselect(
        "Add models to training queue",
        options=available_extra,
        format_func=_extra_label,
        default=[k for k in (s.get("user_added_models", []) or []) if k in available_extra],
        key="user_extra_models",
    )

    # Warn immediately if user added incompatible models
    user_added_incompat = [(k, _task_compat(k, problem_type)[1])
                           for k in user_added if not _task_compat(k, problem_type)[0]]
    if user_added_incompat:
        for mk, reason in user_added_incompat:
            task_word = "regression" if mk in _CLF_ONLY else "classification"
            st.markdown(
                f'<div style="background:#2e1010;border-left:4px solid #e15759;'
                f'border-radius:6px;padding:10px 14px;margin:4px 0;">'
                f'<span style="color:#fca5a5;font-weight:700;">❌ {mk} is a {task_word}-only model</span><br>'
                f'<span style="color:#fcd34d;font-size:0.88rem;">{reason}</span><br>'
                f'<span style="color:#7ec8fa;font-size:0.85rem;">💡 Remove it from the list above, '
                f'or go back to Data Fusion and change the problem type to <strong>{task_word}</strong>.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Custom instructions ───────────────────────────────────────────────────
    with st.expander("📝 Additional instructions for the Model Agent (optional)"):
        model_instructions = st.text_area(
            "e.g. 'prefer faster models', 'try more regularisation', 'avoid SVM'",
            value=s.get("user_model_instructions", ""),
            height=80,
            key="model_instructions_box",
        )

    # ── Summary & compatibility gate ─────────────────────────────────────────
    all_incompat = incompatible_selected + user_added_incompat
    total_to_train = len(selected_models) + len(user_added)

    if total_to_train == 0:
        alert("Select at least one model to continue.", "warning")
    else:
        metrics_row([
            ("LLM picks selected", len(selected_models)),
            ("User-added",         len(user_added)),
            ("Total to train",     total_to_train),
        ])

    # Global incompatibility banner — shown whenever any checked model is wrong type
    if all_incompat:
        incompat_names = ", ".join(f"`{mk}`" for mk, _ in all_incompat)
        task_flip = "classification" if problem_type == "regression" else "regression"
        st.markdown(
            f'<div style="background:#2e1010;border:1px solid #e15759;border-radius:8px;'
            f'padding:14px 18px;margin:12px 0;">'
            f'<div style="color:#fca5a5;font-size:1rem;font-weight:700;">⛔ Cannot start training</div>'
            f'<div style="color:#e8e8f5;margin:6px 0;">'
            f'The following model(s) are <strong>not compatible</strong> with your '
            f'<strong>{problem_type}</strong> task: {incompat_names}</div>'
            f'<div style="color:#fcd34d;font-size:0.9rem;margin-bottom:6px;">'
            f'These models are designed for <strong>{task_flip}</strong> tasks only and will '
            f'fail or produce meaningless results on a {problem_type} target.</div>'
            f'<div style="color:#7ec8fa;font-size:0.88rem;">'
            f'✏️ <strong>To fix:</strong> deselect the incompatible model(s) above, '
            f'<em>or</em> go back to <strong>Data Fusion</strong> and change '
            f'the problem type to <strong>{task_flip}</strong> if that was the intent.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    confirm_blocked = (total_to_train == 0) or bool(all_incompat)
    if st.button(
        "✅ Confirm model selection & start training →",
        disabled=confirm_blocked,
        help=(
            "Fix the incompatible model(s) above first." if all_incompat else
            "Select at least one model." if total_to_train == 0 else ""
        ),
    ):
        all_models = selected_models + user_added

        # Build forced candidate list for model_agent
        forced_candidates = []
        for mk in all_models:
            existing = next((c for c in llm_candidates if c.get("model_key") == mk), None)
            if existing:
                forced_candidates.append(existing)
            else:
                forced_candidates.append({
                    "model_key": mk,
                    "rationale": f"User-selected: {MODEL_DESCRIPTIONS.get(mk, mk)}",
                    "expected_strength": "medium",
                    "caveat": "",
                })

        patch = sanitize_for_msgpack({
            "hitl_model_selection_approved": True,
            "user_selected_models": selected_models,
            "user_added_models": user_added,
            "llm_model_candidates": llm_candidates,
            "user_model_instructions": model_instructions or s.get("user_model_instructions", ""),
            "_forced_model_candidates": forced_candidates,
        })
        update_graph_state(patch, tid)
        st.session_state.pipeline_state = safe_state({**s, **patch})
        update_store(pipeline_state=safe_state({**s, **patch}), phase="models")
        st.session_state.phase = "models"
        st.rerun()
