"""
agents/orchestrator.py — Orchestrator Agent (v2)

Improvements over v1:
  - Hybrid rule-based + LLM: many decisions now made without an LLM call
    (thresholds, retries, obvious accepts) — LLM only for borderline cases
  - SHAP feedback loop: top SHAP features included in loop_suggestion so
    feature_agent can bias next-round proposals toward high-impact features
  - Meta-learning recording: records completed run to meta_memory on accept
  - Improved early-exit logic with more informative messages
  - Per-metric threshold breaches reported individually for the UI
"""

from __future__ import annotations
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json
from utils.logger import get_logger

log = get_logger(__name__)


# FIX 9: thresholds are now read from config.yaml so they can be tuned per deployment.
# Users can also override per-run via state["_custom_thresholds"].
# Kept lower than "production ready" deliberately — AutoML should accept and let
# the user decide if the score is good enough for their use case.
_DEFAULT_THRESHOLDS = {
    "classification": {"f1_weighted": 0.72, "accuracy": 0.70},
    "regression":     {"r2": 0.50},
}


def _load_thresholds_from_config() -> dict:
    """Load per-problem-type thresholds from config.yaml, falling back to defaults."""
    from utils.config_loader import cfg as _cfg
    return {
        "classification": {
            "f1_weighted": float(_cfg("evaluation.thresholds.f1_weighted", default=_DEFAULT_THRESHOLDS["classification"]["f1_weighted"])),
            "accuracy":    float(_cfg("evaluation.thresholds.accuracy",    default=_DEFAULT_THRESHOLDS["classification"]["accuracy"])),
        },
        "regression": {
            "r2": float(_cfg("evaluation.thresholds.r2", default=_DEFAULT_THRESHOLDS["regression"]["r2"])),
        },
    }

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are an AutoML orchestrator agent assessing model evaluation results.

Key rules:
- If improvement_over_baseline < 0.05, model barely beats naive predictor — always retry.
- If retry_count >= 3, always accept to avoid infinite loops.
- Prefer retry_features if top SHAP features look suspicious (high leakage risk, ID-like names).
- Prefer retry_models if features look diverse and meaningful but scores are weak.

Verdicts:
  "accept"          -> metrics meet thresholds AND beat baseline meaningfully
  "retry_features"  -> metrics poor; features may be hurting or leaking
  "retry_models"    -> metrics poor but features look fine; try different models
  "retry_both"      -> metrics very poor; full reset

Return ONLY valid JSON — no markdown:
{
  "verdict":                    "<accept|retry_features|retry_models|retry_both>",
  "score_assessment":           "<one sentence rating the scores vs thresholds>",
  "reasoning":                  "<2-3 sentences explaining the decision>",
  "suggested_feature_strategy": "<only if features involved>",
  "suggested_model_strategy":   "<only if models involved>",
  "confidence":                 "<high|medium|low>"
}
"""


def _compute_thresholds(state: dict, problem_type: str, base_override: dict | None = None) -> dict:
    # FIX 9: start from config-file values if provided, else fall back to hardcoded defaults
    base = dict(base_override or _DEFAULT_THRESHOLDS.get(problem_type, {}))

    if problem_type == "classification":
        cb = (state.get("split_analysis") or {}).get("class_balance", {})
        if cb:
            min_pct = min(cb.values())
            if min_pct < 0.05:
                base["f1_weighted"] = 0.50; base.pop("accuracy", None)
            elif min_pct < 0.15:
                base["f1_weighted"] = 0.62; base.pop("accuracy", None)
    elif problem_type == "regression":
        stats = ((state.get("model_analysis") or {}).get("target_stats", {})
                 or (state.get("eval_analysis") or {}).get("metrics", {}))
        mean_v = abs(stats.get("mean", 1) or 1)
        std_v  = stats.get("std", 0) or 0
        cv_    = std_v / mean_v
        if cv_ > 2.0:
            base["r2"] = 0.25
        elif cv_ > 1.0:
            base["r2"] = 0.40

    custom = state.get("_custom_thresholds") or {}
    base.update(custom)
    return base


def _build_threshold_breaches(metrics: dict, thresholds: dict) -> list[dict]:
    return [
        {
            "metric":    k,
            "threshold": v,
            "actual":    round(metrics.get(k, 0), 4),
            "passed":    metrics.get(k, 0) >= v,
        }
        for k, v in thresholds.items()
    ]


def _accept_result(reason, metrics, thresholds, state, retry_count, confidence="high") -> dict:
    breaches = _build_threshold_breaches(metrics, thresholds)

    # Record run in meta-memory on accept
    try:
        from utils.meta_memory import record_run
        record_run(state)
    except Exception:
        pass

    orchestrator_analysis = {
        "verdict": "accept", "confidence": confidence,
        "score_assessment": reason, "reasoning": reason,
        "suggested_feature_strategy": "", "suggested_model_strategy": "",
        "retry_count": retry_count, "problem_type": state.get("problem_type", ""),
        "thresholds_used": thresholds,
        "metrics_evaluated": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "threshold_breaches": breaches,
        "engineered_features_count": len(state.get("selected_features", [])),
        "top_shap_features": state.get("shap_importance", [])[:5],
    }
    return {
        "loop_verdict":          "accept",
        "loop_reasoning":        reason,
        "loop_suggestion":       "",
        "loop_feature_strategy": "",
        "loop_model_strategy":   "",
        "orchestrator_decision": {"verdict": "accept", "confidence": confidence, "reasoning": reason},
        "orchestrator_analysis": orchestrator_analysis,
        "agent_messages":        [f"[Orchestrator] Verdict: accept. {reason}"],
    }


@agent_error_handler("Orchestrator Agent")
def orchestrator_agent(state: PipelineState) -> dict:
    log.info("Orchestrator agent started", extra={"retry_count": state.get("retry_count", 0)})
    api_key      = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model_name   = os.getenv("ORCHESTRATOR_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    problem_type = state["problem_type"]
    metrics      = state.get("eval_metrics", {})
    retry_count  = state.get("retry_count", 0)
    # FIX 9: merge config-file thresholds with dynamic adjustments
    _config_thresholds = _load_thresholds_from_config()
    thresholds   = _compute_thresholds(state, problem_type, base_override=_config_thresholds.get(problem_type))

    # ── Rule-based early-exits (no LLM needed) ────────────────────────────────
    if retry_count >= MAX_RETRIES:
        return _accept_result(
            f"Maximum retries ({MAX_RETRIES}) reached — accepting current results.",
            metrics, thresholds, state, retry_count,
        )

    baseline     = state.get("_baseline_score")
    primary_key  = "f1_weighted" if problem_type == "classification" else "r2"
    actual_score = metrics.get(primary_key, 0) or 0
    improvement  = round(actual_score - (baseline or 0), 4) if baseline is not None else None

    breaches   = _build_threshold_breaches(metrics, thresholds)
    all_passed = all(b["passed"] for b in breaches)
    baseline_ok = (improvement is None) or (improvement >= 0.05)

    # ── SHAP feedback: extract top feature names for retry suggestions ─────────
    shap_imp = state.get("shap_importance") or []
    top_shap_names = [s["feature"] for s in shap_imp[:5] if "feature" in s]
    shap_feedback_msg = (
        f"Top SHAP features from this run: {top_shap_names}. "
        "Consider engineering interactions between these in the next round."
        if top_shap_names else ""
    )

    # ── Happy path: all thresholds pass + baseline improvement OK ─────────────
    if all_passed and baseline_ok:
        reason = (
            f"All thresholds passed ({primary_key}={round(actual_score, 4)}) "
            + (f"with +{improvement:.4f} over baseline." if improvement is not None
               else "— no baseline for comparison.")
        )
        return _accept_result(reason, metrics, thresholds, state, retry_count)

    # ── Hybrid rule: trivial cases that don't need LLM ────────────────────────
    leakage_risk = (state.get("leakage_report") or {}).get("overall_risk", "low")
    drift_sev    = (state.get("drift_report") or {}).get("overall_severity", "low")
    drift_cols   = list((state.get("drift_report") or {}).get("drifted_features", []))

    # Fix #15: drift detection enforced — surface drift as a concrete warning
    # and add it to the SHAP feedback message sent to feature_agent on retry.
    drift_warning = ""
    if drift_sev in ("high", "medium"):
        drift_warning = (
            f"Data drift detected (severity={drift_sev}) in features: "
            f"{drift_cols[:5]}. Consider removing or transforming these columns."
        )
        shap_feedback_msg = (shap_feedback_msg + " " + drift_warning).strip()

    # If high leakage risk was detected and score is poor → retry features
    if leakage_risk == "high" and not all_passed and retry_count < 2:
        feat_strategy = f"High leakage risk detected. Remove suspect features. {shap_feedback_msg}"
        return {
            "loop_verdict":          "retry_features",
            "loop_reasoning":        f"High leakage + poor metrics after {retry_count} retries.",
            "loop_suggestion":       feat_strategy,
            "loop_feature_strategy": feat_strategy,
            "loop_model_strategy":   "",
            "retry_count":           retry_count + 1,   # FIX 5: single source of truth
            "orchestrator_decision": {"verdict": "retry_features", "confidence": "high"},
            "orchestrator_analysis": {
                "verdict": "retry_features", "threshold_breaches": breaches,
                "top_shap_features": shap_imp[:5], "leakage_risk": leakage_risk,
                "drift_severity": drift_sev, "drift_columns": drift_cols,
            },
            "agent_messages": [f"[Orchestrator] Rule-based retry_features (leakage={leakage_risk})."],
        }

    # Severe drift alone can trigger a retry even when metrics look OK
    if drift_sev == "high" and all_passed and retry_count < 2:
        feat_strategy = (
            f"High data drift detected — model may not generalise. {drift_warning} "
            f"{shap_feedback_msg}"
        )
        return {
            "loop_verdict":          "retry_features",
            "loop_reasoning":        f"High drift (severity=high) despite passing thresholds — "
                                     f"generalisation risk. Drifted cols: {drift_cols[:5]}",
            "loop_suggestion":       feat_strategy,
            "loop_feature_strategy": feat_strategy,
            "loop_model_strategy":   "",
            "retry_count":           retry_count + 1,   # Fix #2: increment here, not only in UI
            "orchestrator_decision": {"verdict": "retry_features", "confidence": "medium"},
            "orchestrator_analysis": {
                "verdict": "retry_features", "threshold_breaches": breaches,
                "top_shap_features": shap_imp[:5], "drift_severity": drift_sev,
            },
            "agent_messages": [f"[Orchestrator] Rule-based retry_features (drift={drift_sev})."],
        }

    # ── Full LLM assessment for borderline cases ──────────────────────────────
    payload = {
        "problem_type": problem_type, "retry_count": retry_count,
        "thresholds": thresholds,
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "metric_ci": state.get("eval_analysis", {}).get("metric_confidence_intervals", {}),
        "threshold_breaches": breaches,
        "engineered_features": state.get("selected_features", []),
        "top_shap_features": shap_imp[:10],
        "best_cv_score": state.get("best_cv_score"),
        "baseline_score": baseline,
        "improvement_over_baseline": improvement,
        "leakage_risk": leakage_risk,
        "drift_severity": drift_sev,
        "shap_feedback": shap_feedback_msg,
        "ensemble_result": state.get("ensemble_report", {}).get("winner", "none"),
    }

    result, _llm_err = call_llm_json(
        api_key=api_key, model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
        user_content=json.dumps(payload, indent=2),
        temperature=0.1, max_tokens=700,
    )
    if result is None:
        result = {"verdict": "accept",
                  "reasoning": f"Orchestrator error — auto-accepting: {_llm_err}",
                  "score_assessment": "Error during assessment.", "confidence": "low"}

    verdict = result.get("verdict", "accept")
    if verdict not in ("accept", "retry_features", "retry_models", "retry_both"):
        verdict = "accept"

    feat_strategy  = result.get("suggested_feature_strategy", "") or ""
    model_strategy = result.get("suggested_model_strategy", "")  or ""

    # Append SHAP feedback to feature strategy suggestion
    if shap_feedback_msg and "features" in verdict:
        feat_strategy = f"{feat_strategy} {shap_feedback_msg}".strip()

    suggestion = feat_strategy or model_strategy

    orchestrator_analysis = {
        "verdict":                    verdict,
        "confidence":                 result.get("confidence", "medium"),
        "score_assessment":           result.get("score_assessment", ""),
        "reasoning":                  result.get("reasoning", ""),
        "suggested_feature_strategy": feat_strategy,
        "suggested_model_strategy":   model_strategy,
        "retry_count":                retry_count,
        "problem_type":               problem_type,
        "thresholds_used":            thresholds,
        "metrics_evaluated":          {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "threshold_breaches":         breaches,
        "engineered_features_count":  len(state.get("selected_features", [])),
        "top_shap_features":          shap_imp[:5],
        "leakage_risk":               leakage_risk,
        "drift_severity":             drift_sev,
        "shap_feedback":              shap_feedback_msg,
    }

    # Record to meta-memory on accept
    if verdict == "accept":
        try:
            from utils.meta_memory import record_run
            record_run(state)
        except Exception:
            pass

    return {
        "loop_verdict":          verdict,
        "loop_reasoning":        result.get("reasoning", ""),
        "loop_suggestion":       suggestion,
        "loop_feature_strategy": feat_strategy,
        "loop_model_strategy":   model_strategy,
        # FIX 5: orchestrator is the SINGLE source of truth for retry_count.
        # The UI must NOT also increment it — doing so causes double-counting
        # (retry_count += 2 per retry, effectively halving MAX_RETRIES).
        "retry_count":           retry_count + 1 if verdict != "accept" else retry_count,
        "orchestrator_decision": result,
        "orchestrator_analysis": orchestrator_analysis,
        "agent_messages": [
            f"[Orchestrator] Verdict: {verdict}. "
            f"Confidence: {result.get('confidence', '?')}. "
            f"{result.get('reasoning', '')[:80]}…"
        ],
    }


def is_below_threshold(metrics: dict, problem_type: str, state: dict | None = None) -> bool:
    thresholds = _compute_thresholds(state or {}, problem_type)
    return any(metrics.get(k, 1.0) < v for k, v in thresholds.items())
