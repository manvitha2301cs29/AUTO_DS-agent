"""
agents/state.py — Shared typed state for the AutoML LangGraph pipeline (v2).

IMPORTANT — serialisability contract
-------------------------------------
LangGraph checkpoints EVERY field in PipelineState to SQLite.
Only JSON-serialisable types may live here:
  str, int, float, bool, list, dict, None

Non-serialisable runtime objects (numpy arrays, sklearn models, etc.)
are stored separately in st.session_state["runtime_objects"][thread_id].

New fields added in v2:
  leakage_report        (leakage_agent)
  drift_report          (eval_agent / detect_drift)
  ensemble_report       (ensemble_agent)
  calibration_metrics   (eval_agent / model_calibration)
  metric_ci             (eval_agent bootstrap CIs)
  auto_dataset_insights (auto-computed on upload)
  _meta_memory_hints    (meta-learning warm-start suggestions)
"""

from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict, Annotated

MAX_AGENT_MESSAGES = 200


def _capped_add(a: list, b: list) -> list:
    merged = a + b
    if len(merged) > MAX_AGENT_MESSAGES:
        merged = merged[-MAX_AGENT_MESSAGES:]
    return merged


class PipelineState(TypedDict, total=False):

    # ── Ingest ────────────────────────────────────────────────────────────────
    df_parquet_b64: str
    target: str
    problem_type: str
    n_rows: int
    n_cols: int
    auto_dataset_insights: dict  # ← NEW: auto-computed on upload
    # Keys: n_rows, n_cols, n_numeric, n_categorical, n_datetime,
    #       missing_pct, imbalance_ratio, recommended_problem_type,
    #       type_issues (list), warnings (list)

    # ── EDA Agent ─────────────────────────────────────────────────────────────
    column_meta: list
    preprocessing_decisions: dict
    global_notes: str
    eda_report: str
    eda_analysis: dict
    # Keys: n_columns_analysed, n_decisions, global_notes, eda_report,
    #       numeric_columns, categorical_columns, high_null_columns,
    #       zero_variance_cols, high_skew_columns,
    #       decisions_summary, decisions_full

    # ── Leakage Detection Agent ───────────────────────────────────────────────
    leakage_report: dict       # ← NEW
    # Keys: overall_risk, leakage_summary, high_risk_columns,
    #       medium_risk_columns, id_risk_columns, time_risk_columns,
    #       dropped_by_leakage, target_scores, llm_verdicts,
    #       n_flagged, n_dropped

    # ── Feature Engineering Agent ─────────────────────────────────────────────
    feature_proposals: list
    selected_features: list
    df_engineered_parquet_b64: str
    feature_strategy_summary: str
    feature_analysis: dict
    # Keys: strategy_summary, n_proposed, n_computable,
    #       n_user_kept, n_user_custom, n_agent_proposed, n_poly_advanced,
    #       retry_count, user_instructions, shap_feedback_used, te_columns,
    #       computable_features, dropped_features, high_leakage_risk,
    #       proposals_full

    # ── v3: Imbalance analysis & recommendations ──────────────────────────────
    imbalance_analysis: dict
    # Keys: n_classes, class_distribution, imbalance_ratio, severity,
    #       majority_class, minority_class, minority_pct, n_samples
    imbalance_recommendations: dict
    # Keys: recommended_split, split_rationale, recommended_loss, loss_rationale,
    #       class_weight, smote_recommended, imbalance_handling_summary

    # ── v3: HITL model selection ──────────────────────────────────────────────
    hitl_model_selection_approved: bool   # user approved model candidates
    user_selected_models: list            # model keys user picked (subset of LLM recommendations)
    user_added_models: list               # extra model keys user typed in manually
    llm_model_candidates: list            # raw LLM candidate list (before HITL filter)
    _forced_model_candidates: Optional[list]  # enriched candidates with hyperparams, set by HITL page

    # ── HITL free-text ────────────────────────────────────────────────────────
    user_feature_instructions: str
    user_model_instructions: str
    user_kept_features: list
    user_custom_features: list

    # ── Split Agent ───────────────────────────────────────────────────────────
    split_strategy: str
    split_rationale: str
    split_warnings: list
    test_size: float
    datetime_column: Optional[str]
    group_column: Optional[str]
    split_analysis: dict

    # ── Drift Report ──────────────────────────────────────────────────────────
    drift_report: dict         # ← NEW (also written by eval_agent)
    # Keys: feature_stats, overall_drift_score, overall_severity,
    #       summary, flagged_features, n_flagged

    # ── Visualisation Agent ───────────────────────────────────────────────────
    eda_plots: dict

    # ── Model Selection Agent ─────────────────────────────────────────────────
    model_candidates: list
    model_recommendation: str
    tuning_results: list
    best_model_key: str
    best_params: dict
    best_cv_score: float
    best_nn_config: dict
    all_model_params: dict
    model_analysis: dict
    optuna_trials: int            # n_trials used in last run (read back on retry)
    cv_folds: int                 # cv folds used in last run (read back on retry)
    _meta_memory_hints: Optional[list]  # ← NEW: warm-start model suggestions

    # ── Ensemble Agent ────────────────────────────────────────────────────────
    ensemble_report: dict      # ← NEW
    # Keys: base_models, ensemble_results, winner, winner_score,
    #       original_best_score, improvement, replaced_best_model

    # ── Evaluation Agent ──────────────────────────────────────────────────────
    eval_metrics: dict
    shap_importance: list
    eval_report: str
    eval_analysis: dict
    # Keys (v2 additions): metric_confidence_intervals, calibration_metrics,
    #       prediction_confidence, drift_report (reference), baseline_score
    # Keys (v1): model_key, problem_type, n_test_samples, n_features,
    #       feature_names, metrics, confusion_matrix,
    #       shap_importance_full, top_10_features, eval_report,
    #       llm_payload_used, cv_score_at_training, best_params

    # Calibration metrics (top-level for easy access in UI)
    calibration_metrics: dict   # ← NEW: ECE, MCE, Brier, bins
    metric_ci: dict             # ← NEW: {metric: {mean, std, ci_low, ci_high}}
    top_shap_features: list     # top feature names from SHAP, used by feature_agent on retry

    # ── Orchestrator ──────────────────────────────────────────────────────────
    retry_count: int
    loop_verdict: str
    loop_reasoning: str
    loop_suggestion: str
    loop_feature_strategy: str
    loop_model_strategy: str
    orchestrator_decision: dict
    orchestrator_analysis: dict

    # ── User-provided OpenAI API key ──────────────────────────────────────────
    openai_api_key: Optional[str]

    # ── HITL gates ────────────────────────────────────────────────────────────
    hitl_eda_approved: bool
    hitl_features_approved: bool
    hitl_leakage_approved: bool   # ← NEW
    hitl_split_approved: bool
    hitl_viz_approved: bool
    hitl_models_approved: bool
    hitl_loop_approved: bool
    hitl_ensemble_approved: bool  # ← NEW
    hitl_final_accepted: bool

    # ── Page-refresh resilience ───────────────────────────────────────────────
    _preprocessor_b64: Optional[str]
    _label_encoder_b64: Optional[str]
    _best_model_b64: Optional[str]
    _baseline_score: Optional[float]
    _X_train_t_b64: Optional[str]
    _X_test_t_b64:  Optional[str]
    _y_train_b64:   Optional[str]
    _y_test_b64:    Optional[str]
    _feature_names: Optional[list]
    _custom_thresholds: Optional[dict]
    _thread_id: Optional[str]

    # ── Append-only message log ───────────────────────────────────────────────
    agent_messages: Annotated[list, _capped_add]
