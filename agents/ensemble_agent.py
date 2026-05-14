"""
agents/ensemble_agent.py — Ensemble Learning Agent

Builds Voting and Stacking ensembles from the top-performing models produced
by model_agent.  Called after eval_agent when loop_verdict == "accept".

Strategies:
  VotingClassifier / VotingRegressor   — soft/hard voting over top-N models
  StackingClassifier / StackingRegressor — base models + meta-learner (Ridge/LR)

The agent:
  1. Selects ALL successfully tuned models by CV score from tuning_results
     (FIX 6: was top-3 only — now uses all models the user selected)
  2. Trains both a Voting and a Stacking ensemble on X_train_t / y_train
  3. Evaluates both on X_test_t / y_test
  4. Picks the best and writes it to runtime as "best_model" if it beats the
     current best_model score

FIX 1: Runtime store updated via RuntimeStore, not st.session_state directly.
FIX 10: Stacking has a wall-clock timeout guard to avoid hanging on large datasets.

All sklearn objects stored in runtime_objects (never in graph state).
"""

from __future__ import annotations
import threading
import time
from typing import Any

from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler
from utils.logger import get_logger

log = get_logger(__name__)
from utils.runtime import get_or_restore_runtime
from ui.runtime_context import get_runtime_store


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_model(model, X_test_t, y_test, problem_type: str) -> float:
    y_pred = model.predict(X_test_t)
    if problem_type == "classification":
        return float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    return float(r2_score(y_test, y_pred))


def _supports_predict_proba(model) -> bool:
    return hasattr(model, "predict_proba")


def _fit_with_timeout(model, X_train, y_train, timeout_seconds: float):
    """
    Fit a sklearn model in a daemon thread with a wall-clock timeout.
    FIX 7: uses threading instead of signal.SIGALRM so it works on Windows,
    macOS, and Linux regardless of whether we are in the main thread.

    Returns the fitted model on success, raises TimeoutError on timeout,
    re-raises any exception thrown inside the thread.
    """
    result = {}
    exc_holder = {}

    def _worker():
        try:
            model.fit(X_train, y_train)
            result["model"] = model
        except Exception as e:
            exc_holder["exc"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        # Thread is still running — we cannot kill it in Python, but we can
        # stop waiting and treat it as a timeout. The daemon thread will be
        # cleaned up when the process exits.
        raise TimeoutError(
            f"Model fit timed out after {timeout_seconds:.0f}s"
        )
    if "exc" in exc_holder:
        raise exc_holder["exc"]
    return result["model"]


# ─────────────────────────────────────────────────────────────────────────────
# Agent entry point
# ─────────────────────────────────────────────────────────────────────────────

@agent_error_handler("Ensemble Agent")
def ensemble_agent(state: PipelineState) -> dict:
    log.info("Ensemble agent started", extra={"problem_type": state.get("problem_type")})
    problem_type  = state["problem_type"]
    tuning_results: list[dict] = state.get("tuning_results") or []
    best_key      = state.get("best_model_key", "")
    best_cv       = state.get("best_cv_score", 0.0) or 0.0

    # ── Fetch runtime objects ─────────────────────────────────────────────────
    rt, tid = get_or_restore_runtime(
        state, required=("best_model", "X_train_t", "X_test_t", "y_train", "y_test")
    )
    missing = [k for k in ("best_model", "X_train_t", "X_test_t", "y_train", "y_test") if k not in rt]
    if missing:
        return {"agent_messages": [f"[Ensemble Agent] ❌ Missing runtime objects: {missing}"]}

    best_model    = rt["best_model"]
    X_train_t     = rt["X_train_t"]
    X_test_t      = rt["X_test_t"]
    y_train       = rt["y_train"]
    y_test        = rt["y_test"]
    all_models: dict[str, Any] = rt.get("all_trained_models", {})

    # ── FIX 6: Use ALL successfully tuned models, not just top-3 ─────────────
    # Previously capped at [:3] — this wasted the runner-up fits done by model_agent
    # and excluded user-selected models beyond rank 3.
    sorted_results = sorted(
        [r for r in tuning_results if r.get("best_score") is not None],
        key=lambda r: r["best_score"], reverse=True
    )

    if len(sorted_results) < 2:
        return {
            "ensemble_report": {"skipped": True, "reason": "Need ≥2 tuned models for ensembling"},
            "agent_messages": ["[Ensemble Agent] Skipped — fewer than 2 tuned models available."],
        }

    # Build estimator list from runtime store or current best
    estimators: list[tuple[str, Any]] = []
    for r in sorted_results:
        key = r["model_key"]
        model_obj = all_models.get(key) or (best_model if key == best_key else None)
        if model_obj is not None:
            estimators.append((key, model_obj))

    if len(estimators) < 2:
        return {
            "ensemble_report": {"skipped": True, "reason": "Could not retrieve ≥2 trained models from runtime"},
            "agent_messages": ["[Ensemble Agent] Skipped — could not retrieve model objects."],
        }

    log.info(f"Ensemble agent using {len(estimators)} models: {[e[0] for e in estimators]}")

    # Current best score on test set
    current_score = _score_model(best_model, X_test_t, y_test, problem_type)

    # Ensemble must beat both test-set score AND CV score to avoid selecting
    # an ensemble that got lucky on the test split but has worse CV stability.
    min_threshold = max(current_score, best_cv)

    results: list[dict] = []
    best_ensemble = None
    best_ensemble_name = ""
    best_ensemble_score = min_threshold  # must beat this to replace

    # Pre-compute size-dependent constants used by both voting and stacking blocks
    n_samples = len(X_train_t)
    stacking_cv = 3 if n_samples > 10_000 else 5
    stacking_timeout = min(300.0, max(60.0, n_samples * 0.015))
    voting_timeout   = min(120.0, max(30.0,  n_samples * 0.005))
    try:
        if problem_type == "classification":
            use_soft = all(_supports_predict_proba(m) for _, m in estimators)
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting="soft" if use_soft else "hard",
                n_jobs=-1,
            )
            fitted_v = _fit_with_timeout(voting_clf, X_train_t, y_train, voting_timeout)
            score = _score_model(fitted_v, X_test_t, y_test, problem_type)
            results.append({"name": "voting", "score": score, "voting": "soft" if use_soft else "hard"})
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble = fitted_v
                best_ensemble_name = "voting"
        else:
            voting_reg = VotingRegressor(estimators=estimators, n_jobs=-1)
            fitted_v = _fit_with_timeout(voting_reg, X_train_t, y_train, voting_timeout)
            score = _score_model(fitted_v, X_test_t, y_test, problem_type)
            results.append({"name": "voting", "score": score})
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble = fitted_v
                best_ensemble_name = "voting"
    except (TimeoutError, Exception) as exc:
        results.append({"name": "voting", "error": str(exc)})
        log.warning("Voting ensemble failed", extra={"error": str(exc)})

    # ── Stacking ensemble (FIX 10: with wall-clock timeout guard) ────────────
    # StackingClassifier(cv=5) refits all base models 5× — very slow on large
    # datasets. Cap at 300s; fall back gracefully if timeout or error occurs.
    try:
        if problem_type == "classification":
            meta = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=meta,
                cv=stacking_cv,
                n_jobs=-1,
                passthrough=False,
            )
            fitted = _fit_with_timeout(stacking_clf, X_train_t, y_train, stacking_timeout)
            score = _score_model(fitted, X_test_t, y_test, problem_type)
            results.append({"name": "stacking", "score": score, "meta_learner": "LogisticRegression"})
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble = fitted
                best_ensemble_name = "stacking"
        else:
            meta = Ridge(alpha=1.0)
            stacking_reg = StackingRegressor(
                estimators=estimators,
                final_estimator=meta,
                cv=stacking_cv,
                n_jobs=-1,
            )
            fitted = _fit_with_timeout(stacking_reg, X_train_t, y_train, stacking_timeout)
            score = _score_model(fitted, X_test_t, y_test, problem_type)
            results.append({"name": "stacking", "score": score, "meta_learner": "Ridge"})
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble = fitted
                best_ensemble_name = "stacking"
    except (TimeoutError, Exception) as exc:
        results.append({"name": "stacking", "error": str(exc)})
        log.warning("Stacking ensemble failed or timed out", extra={"error": str(exc)})

    # ── FIX 1: Update runtime via RuntimeStore, not st.session_state ─────────
    # ensemble_agent runs inside a LangGraph node which may execute in a background
    # thread where st.session_state is unavailable. Use get_runtime_store() instead.
    state_updates: dict = {}
    if best_ensemble is not None:
        try:
            rt_store = get_runtime_store()
            if rt_store is not None and hasattr(rt_store, "update"):
                rt_store.update(tid, {"best_model": best_ensemble})
            else:
                # Fallback: update the rt dict in-place (same object model_agent stored)
                rt["best_model"] = best_ensemble
        except Exception as _e:
            log.warning("Could not update runtime with ensemble model", extra={"error": str(_e)})

        state_updates["best_model_key"] = f"{best_ensemble_name}_ensemble"
        state_updates["best_cv_score"]  = best_ensemble_score

        # Re-evaluate ensemble on test set so state metrics reflect ensemble performance
        try:
            from sklearn.metrics import (
                f1_score as _f1, r2_score as _r2,
                accuracy_score as _acc, mean_squared_error as _mse,
                mean_absolute_error as _mae,
            )
            y_pred_ens = best_ensemble.predict(X_test_t)
            if problem_type == "classification":
                ens_metrics = {
                    "accuracy":    float(_acc(y_test, y_pred_ens)),
                    "f1_weighted": float(_f1(y_test, y_pred_ens, average="weighted", zero_division=0)),
                    "_source":     f"{best_ensemble_name}_ensemble",
                }
            else:
                ens_metrics = {
                    "r2":      float(_r2(y_test, y_pred_ens)),
                    "rmse":    float(_mse(y_test, y_pred_ens) ** 0.5),
                    "mae":     float(_mae(y_test, y_pred_ens)),
                    "_source": f"{best_ensemble_name}_ensemble",
                }
            state_updates["eval_metrics"] = ens_metrics
        except Exception as _e:
            log.warning("Ensemble metric re-evaluation failed", extra={"error": str(_e)})

    ensemble_report = {
        "base_models":         [r["model_key"] for r in sorted_results],
        "n_base_models":       len(estimators),
        "ensemble_results":    results,
        "winner":              best_ensemble_name or "none_beat_baseline",
        "winner_score":        best_ensemble_score,
        "original_best_score": current_score,
        "best_cv_score":       best_cv,
        "min_threshold":       min_threshold,
        "improvement":         round(best_ensemble_score - current_score, 4),
        "replaced_best_model": best_ensemble is not None,
        "stacking_cv_folds":   stacking_cv,
    }

    msg = (
        f"[Ensemble Agent] Winner: {best_ensemble_name or 'none'}. "
        f"Score: {best_ensemble_score:.4f} (was {current_score:.4f}). "
        f"Improvement: {best_ensemble_score - current_score:+.4f}. "
        f"Used {len(estimators)} base models."
    )

    return {
        "ensemble_report": ensemble_report,
        "agent_messages":  [msg],
        **state_updates,
    }
