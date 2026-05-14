"""ui/pages/split.py — Phase 4: Split Agent page."""
from __future__ import annotations
import json
import time
import numpy as np


def _sanitize_array(arr: np.ndarray) -> np.ndarray:
    """Replace inf/NaN in a transformed feature array with column medians.

    Defined at module level (not as a closure) so it can be pickled by
    concurrent.futures / multiprocessing workers without raising
    'Can't pickle local object' errors.
    """
    arr = np.array(arr, dtype=np.float64)
    arr = np.where(np.isinf(arr), np.nan, arr)
    col_medians = np.nanmedian(arr, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_medians, inds[1])
    return arr
import pandas as pd
import streamlit as st

from utils.serialization import b64_to_df, obj_to_b64, sanitize_for_msgpack
from ui.components import alert, badge, metrics_row, card
from ui.graph_helpers import (
    get_state,
    safe_state,
    persist,
    np_to_b64,
    update_graph_state,
)
from ui.runtime_context import (
    append_training_event,
    clear_training_monitor,
    set_training_progress,
)
from ui.state_store import update_store
from ui.workflow_controls import reopen_workflow_stage

SPLIT_STRATEGIES = {
    "standard":    ("Standard random split",  "Random shuffle, optionally stratified"),
    "time_series": ("Time-series split",       "Sort by datetime → last N% as test, NO shuffle"),
    "group_based": ("Group-based split",       "No group spans both train and test"),
    "stratified":  ("Stratified split",        "StratifiedShuffleSplit for imbalanced classification"),
}


def _append_training_event(_rt, tid: str, message: str, stage: str | None = None) -> None:
    append_training_event(tid, message, stage=stage)


def page_split(_rt):
    s = st.session_state.pipeline_state
    if not s.get("hitl_features_approved") and not s.get("split_strategy"):
        alert("Feature phase not complete.", "warning")
        return

    tid             = st.session_state.tid
    agent_strategy  = s.get("split_strategy", "standard")
    agent_rationale = s.get("split_rationale", "")
    agent_test_size = s.get("test_size", 0.2)
    split_warnings  = s.get("split_warnings", [])
    approved        = bool(s.get("hitl_split_approved"))
    best_model      = s.get("best_model_key")

    st.markdown(f'{badge("Phase 4")} <h1 style="display:inline;margin-left:.5rem;">Split Agent</h1>', unsafe_allow_html=True)
    st.caption("Review and apply the train/test split strategy.")

    card(
        f'<b style="color:#a78bfa">Agent Recommendation</b><br>'
        f'{badge(SPLIT_STRATEGIES.get(agent_strategy, (agent_strategy,))[0], "blue")}&nbsp;'
        f'<span style="color:#9898c0;font-size:.85rem">test size: {agent_test_size*100:.0f}%</span><br>'
        f'<p style="margin:.5rem 0 0">{agent_rationale}</p>'
    )
    for w in split_warnings:
        alert(f"⚠️ {w}", "warning")

    if approved:
        alert("✅ Split applied — Model Agent has run. Navigate forward via the sidebar.", "success")
        if best_model:
            metrics_row([
                ("Strategy",   SPLIT_STRATEGIES.get(s.get("split_strategy"), (s.get("split_strategy", "?"),))[0]),
                ("Best model", best_model),
                ("CV score",   f"{s.get('best_cv_score', 0):.4f}"),
            ], accent=True)
        if st.button("↩ Reopen Split Settings", width="stretch"):
            reopen_workflow_stage(
                "split",
                tid=tid,
                phase="split",
                note="[Workflow] Reopened split settings for revision.",
            )
        return

    # ── v3: Imbalance & loss function recommendations panel ───────────────────
    imbalance = s.get("imbalance_analysis") or {}
    imbalance_recs = s.get("imbalance_recommendations") or {}
    if imbalance and imbalance.get("severity") not in (None, "n/a"):
        severity = imbalance.get("severity", "balanced")
        ratio = imbalance.get("imbalance_ratio")
        severity_colours = {
            "balanced": "#22c55e", "mild": "#facc15",
            "moderate": "#f97316", "severe": "#ef4444",
        }
        colour = severity_colours.get(severity, "#888")
        ratio_str = f"{ratio:.1f}:1" if ratio else "n/a"

        with st.expander("⚖️ Imbalance Analysis & Recommendations", expanded=(severity in ("moderate", "severe"))):
            col_sev, col_ratio, col_min = st.columns(3)
            with col_sev:
                st.markdown(f'<div style="text-align:center"><span style="font-size:1.5rem;font-weight:700;color:{colour}">{severity.upper()}</span><br><span style="color:#9898c0;font-size:.75rem">Imbalance Severity</span></div>', unsafe_allow_html=True)
            with col_ratio:
                st.markdown(f'<div style="text-align:center"><span style="font-size:1.5rem;font-weight:700;color:#60a5fa">{ratio_str}</span><br><span style="color:#9898c0;font-size:.75rem">Majority:Minority Ratio</span></div>', unsafe_allow_html=True)
            with col_min:
                minority_pct = imbalance.get("minority_pct", "?")
                st.markdown(f'<div style="text-align:center"><span style="font-size:1.5rem;font-weight:700;color:#a78bfa">{minority_pct}%</span><br><span style="color:#9898c0;font-size:.75rem">Minority Class %</span></div>', unsafe_allow_html=True)

            st.markdown("---")
            rec_split = imbalance_recs.get("recommended_split", "stratified")
            rec_loss  = imbalance_recs.get("recommended_loss", "cross_entropy")
            rec_cw    = imbalance_recs.get("class_weight")
            smote_rec = imbalance_recs.get("smote_recommended", False)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Recommended Split:** `{rec_split}`")
                st.caption(imbalance_recs.get("split_rationale", ""))
                st.markdown(f"**Class Weight:** `{rec_cw or 'none'}`")
                if smote_rec:
                    st.markdown("**SMOTE oversampling:** Recommended ✅")
            with col_r:
                st.markdown(f"**Recommended Loss:** `{rec_loss}`")
                st.caption(imbalance_recs.get("loss_rationale", ""))

            summary = imbalance_recs.get("imbalance_handling_summary", "")
            if summary:
                if severity in ("moderate", "severe"):
                    alert(f"<strong>Imbalance Advisory:</strong> {summary}", "warning")
                else:
                    st.info(f"ℹ️ {summary}")

            dist = imbalance.get("class_distribution", {})
            if dist:
                st.markdown("**Class distribution:**")
                dist_rows = [{"Class": cls, "Count": v["count"], "Pct": f"{v['pct']}%"} for cls, v in dist.items()]
                st.dataframe(pd.DataFrame(dist_rows), hide_index=True, width="stretch")

    st.markdown("### Split strategy")
    strategy = st.radio(
        "Choose strategy",
        list(SPLIT_STRATEGIES.keys()),
        format_func=lambda k: f"{SPLIT_STRATEGIES[k][0]} — {SPLIT_STRATEGIES[k][1]}",
        index=list(SPLIT_STRATEGIES.keys()).index(agent_strategy) if agent_strategy in SPLIT_STRATEGIES else 0,
        label_visibility="collapsed",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_size = st.slider("Test size", 0.1, 0.4, float(agent_test_size), 0.05)
    with col2:
        random_seed = st.number_input("Random seed", value=int(s.get("random_seed", 42) or 42), step=1)
    with col3:
        optuna_trials = st.number_input(
            "Optuna trials",
            value=int(s.get("optuna_trials", 20) or 20),
            min_value=5,
            max_value=200,
            step=5,
        )
    with col4:
        cv_folds = st.number_input(
            "CV folds",
            value=int(s.get("cv_folds", 3) or 3),
            min_value=2,
            max_value=10,
            step=1,
        )

    df_cols = [m["name"] for m in (s.get("column_meta") or [])]
    datetime_col = group_col = None
    if strategy == "time_series":
        datetime_col = st.selectbox("Datetime column", [""] + df_cols) or None
    elif strategy == "group_based":
        group_col = st.selectbox("Group column", [""] + df_cols) or None

    with st.expander("⚙️ Advanced: custom quality thresholds (JSON)"):
        custom_thresh_str = st.text_area('e.g. {"f1_weighted": 0.80}', value="{}", height=80)

    st.markdown("---")
    if st.button("⚡ Apply split & run Model Agent → (may take several minutes)"):
        try:
            custom_thresholds = json.loads(custom_thresh_str or "{}")
        except Exception:
            alert("Custom thresholds must be valid JSON.", "error")
            return

        with st.spinner("Executing split, preprocessing, and Optuna tuning… (this may take a few minutes)"):
            try:
                from utils.ml_helpers import (
                    execute_split,
                    build_preprocessor,
                    generate_eda_plots,
                    get_preprocessor_feature_names,
                    GLOBAL_SEED,
                )
                import hashlib as _hl

                full_state = get_state(tid)
                df_key = "df_engineered_parquet_b64" if full_state.get("df_engineered_parquet_b64") else "df_parquet_b64"
                df        = b64_to_df(full_state[df_key])
                target    = full_state["target"]
                prob_type = full_state["problem_type"]

                clear_training_monitor(tid)
                set_training_progress(tid, {
                    "status": "running",
                    "stage": "split",
                    "percent": 2,
                    "message": "Preparing dataset split...",
                })
                _append_training_event(_rt, tid, "Preparing dataset and split configuration.", stage="split")

                X_train, X_test, y_train, y_test, label_encoder, X_cal, y_cal = execute_split(
                    df=df, target=target, strategy=strategy,
                    test_size=float(test_size),
                    random_seed=int(random_seed) if random_seed else GLOBAL_SEED,
                    datetime_col=datetime_col, group_col=group_col,
                    problem_type=prob_type,
                )
                set_training_progress(tid, {
                    "status": "running",
                    "stage": "split",
                    "percent": 10,
                    "message": "Split complete. Building preprocessing pipeline...",
                })
                _append_training_event(
                    _rt,
                    tid,
                    f"Split complete with strategy '{strategy}' and test size {float(test_size):.2f}.",
                    stage="split",
                )
                preprocessor, _, _ = build_preprocessor(X_train, s.get("preprocessing_decisions", {}))
                for X in (X_train, X_test):
                    nc = X.select_dtypes(include="number").columns
                    X[nc] = X[nc].replace([np.inf, -np.inf], np.nan)

                # Preprocessor fit cache (Fix #13)
                _data_sig = _hl.md5(
                    str(X_train.shape).encode() +
                    str(sorted(X_train.columns.tolist())).encode() +
                    str(list(s.get("preprocessing_decisions", {}).keys())).encode()
                ).hexdigest()[:12]
                cached_pre = _rt.get_key(tid, "_preprocessor_cache")
                if cached_pre and cached_pre.get("sig") == _data_sig:
                    preprocessor  = cached_pre["preprocessor"]
                    feature_names = cached_pre["feature_names"]
                    X_train_t = preprocessor.transform(X_train)
                    X_test_t  = preprocessor.transform(X_test)
                    _append_training_event(_rt, tid, "Reused cached preprocessor for this dataset shape.", stage="preprocessing")
                else:
                    preprocessor.fit(X_train)
                    X_train_t = preprocessor.transform(X_train)
                    X_test_t  = preprocessor.transform(X_test)
                    try:
                        feature_names = get_preprocessor_feature_names(preprocessor, X_train)
                    except Exception:
                        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]
                    _rt.set_key(tid, "_preprocessor_cache", {
                        "sig": _data_sig, "preprocessor": preprocessor, "feature_names": feature_names
                    })
                    _append_training_event(_rt, tid, "Fitted preprocessor and transformed train/test data.", stage="preprocessing")

                y_train_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
                y_test_arr  = y_test.values  if hasattr(y_test,  "values") else np.array(y_test)
                X_cal_t = y_cal_arr = None
                if X_cal is not None and y_cal is not None:
                    X_cal_t   = preprocessor.transform(X_cal)
                    y_cal_arr = y_cal.values if hasattr(y_cal, "values") else np.array(y_cal)

                # ── Sanitize transformed arrays: replace any remaining NaN/inf ─────
                # Safety net for keep_as_is columns or edge cases in custom transformers.
                # Uses module-level _sanitize_array (not a closure) to stay picklable.
                X_train_t = _sanitize_array(X_train_t)
                X_test_t  = _sanitize_array(X_test_t)
                if X_cal_t is not None:
                    X_cal_t = _sanitize_array(X_cal_t)

                baseline_score = None
                try:
                    from sklearn.dummy import DummyClassifier, DummyRegressor
                    if prob_type == "classification":
                        from sklearn.metrics import f1_score as _f1
                        d = DummyClassifier(strategy="most_frequent", random_state=42)
                        d.fit(X_train_t, y_train_arr)
                        baseline_score = float(_f1(y_test_arr, d.predict(X_test_t), average="weighted", zero_division=0))
                    else:
                        from sklearn.metrics import r2_score as _r2
                        d = DummyRegressor(strategy="mean")
                        d.fit(X_train_t, y_train_arr)
                        baseline_score = float(_r2(y_test_arr, d.predict(X_test_t)))
                except Exception:
                    pass

                _rt.update(tid, {
                    "X_train_t": X_train_t, "X_test_t": X_test_t,
                    "y_train": y_train_arr, "y_test": y_test_arr,
                    "feature_names": feature_names,
                    "label_encoder": label_encoder,
                    "preprocessor": preprocessor,
                    "X_cal_t": X_cal_t, "y_cal": y_cal_arr,
                })
                set_training_progress(tid, {
                    "status": "running",
                    "stage": "preprocessing",
                    "percent": 20,
                    "message": "Generating diagnostics and queuing model tuning...",
                })

                try:
                    eda_plots = generate_eda_plots(X_train=X_train, y_train=y_train,
                                                   target=target, problem_type=prob_type,
                                                   label_encoder=label_encoder)
                except Exception:
                    eda_plots = {}
                _append_training_event(_rt, tid, "Prepared transformed features and diagnostics for model training.", stage="preprocessing")

                update_graph_state(sanitize_for_msgpack({
                    "_X_train_t_b64":      np_to_b64(X_train_t),
                    "_X_test_t_b64":       np_to_b64(X_test_t),
                    "_y_train_b64":        np_to_b64(y_train_arr),
                    "_y_test_b64":         np_to_b64(y_test_arr),
                    "_X_cal_t_b64":        np_to_b64(X_cal_t) if X_cal_t is not None else None,
                    "_y_cal_b64":          np_to_b64(y_cal_arr) if y_cal_arr is not None else None,
                    "_feature_names":      feature_names,
                    "_preprocessor_b64":   obj_to_b64(preprocessor),
                    "_label_encoder_b64":  obj_to_b64(label_encoder) if label_encoder else None,
                    "_baseline_score":     baseline_score,
                    "_custom_thresholds":  custom_thresholds,
                    "eda_plots":           eda_plots,
                    "_thread_id":          tid,
                    "split_strategy":      strategy,
                    "test_size":           float(test_size),
                    "random_seed":         int(random_seed),
                    "optuna_trials":       int(optuna_trials),
                    "cv_folds":            int(cv_folds),
                    "hitl_split_approved": True,
                    # Reset model selection gate so user must confirm models
                    "hitl_model_selection_approved": False,
                    "user_selected_models": [],
                    "user_added_models": [],
                    "llm_model_candidates": [],
                }), tid)

                # Navigate to model selection page (HITL step 4.5)
                set_training_progress(tid, {"status": "idle", "stage": "split", "percent": 100, "message": "Split complete. Choose models next."})
                split_state_patch = sanitize_for_msgpack({
                    "hitl_split_approved": True,
                    "split_strategy": strategy,
                    "test_size": float(test_size),
                    "_baseline_score": baseline_score,
                    "optuna_trials": int(optuna_trials),
                    "cv_folds": int(cv_folds),
                    "hitl_model_selection_approved": False,
                    "user_selected_models": [],
                    "user_added_models": [],
                    "llm_model_candidates": [],
                    # Saved for script_exporter reproducibility
                    "split_info": {
                        "test_size":   float(test_size),
                        "random_seed": int(random_seed),
                        "strategy":    strategy,
                        "cv_folds":    int(cv_folds),
                    },
                })
                st.session_state.pipeline_state = safe_state({**full_state, **split_state_patch})
                st.session_state.phase = "model_selection"
                update_store(pipeline_state=safe_state({**full_state, **split_state_patch}), phase="model_selection")
                persist(tid, {**full_state, **split_state_patch}, phase="model_selection")
                st.success("✅ Split complete! Choose which models to train on the next page.")
                st.rerun()
            except Exception as e:
                alert(f"Split error: {e}", "error")
