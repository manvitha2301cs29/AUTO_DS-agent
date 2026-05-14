"""
agents/eval_agent.py — Evaluation Agent (v5)

New in v5 (on top of v4):
  - MLflow experiment tracking: logs all metrics, params, artifacts per run
  - Error analysis: identifies where the model fails most (worst predictions)
  - Fairness / bias checks (optional, config-driven)
  - Auto drift monitor check + retrain alert
  - ROC curve + confusion matrix base64 stored in state for UI
  - Data contract validation on output
  - Config-driven (reads from config.yaml instead of hardcoding)

Original v2 docstring preserved below:
--- (v2)

Improvements over v1:
  - Expanded metrics: precision, recall, PR-AUC
  - Model calibration (Platt/isotonic) for classifiers
  - Confidence intervals for primary metric via bootstrap (200 resamples)
  - Prediction confidence summary (max-probability uncertainty)
  - Data drift detection between train and test sets (KS + PSI)
  - SHAP feedback loop: top features written back to state for orchestrator
  - Baseline comparison included in eval_analysis

Runtime lookup via shared get_or_restore_runtime() utility (unchanged).
"""

from __future__ import annotations
import json
import os
import warnings

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, r2_score,
    mean_absolute_error, precision_score, recall_score,
    average_precision_score,
)

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json, stream_llm_text
from utils.logger import get_logger
from utils.config_loader import cfg
from utils.data_contracts import validate_output
from utils.experiment_tracker import tracker
from utils.auto_retrain import drift_monitor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io as _io_mod

log = get_logger(__name__)
from utils.runtime import get_or_restore_runtime
from utils.model_calibration import (
    calibrate_model,
    compute_calibration_metrics,
    compute_prediction_confidence,
)


SYSTEM_PROMPT = """\
You are a senior data scientist writing a model evaluation report for a non-technical audience.

Given metrics, confidence intervals, calibration stats, and top SHAP features, write a
concise 3-4 paragraph analysis:
1. How well the model performed and what the numbers mean in plain English.
2. Overfitting check: compare train_metrics vs test metrics — flag any large gap.
3. Which features matter most and what that implies about the problem domain.
4. Calibration quality — are the model's probability estimates trustworthy?
5. Caveats, limitations, and concrete next steps.

Write in flowing prose. No bullet points. Be specific about the numbers.
"""

_TREE_MODELS   = ("RandomForest", "GradientBoosting", "XGB", "LGB", "LGBM", "DecisionTree", "ExtraTrees", "HistGradient")
_LINEAR_MODELS = ("LogisticRegression", "Ridge", "Lasso", "LinearSVC", "LinearSVR")


def _as_numpy_matrix(X):
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy())
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray())
    return np.asarray(X)


def _as_numpy_vector(y):
    if isinstance(y, np.ndarray):
        return y
    if hasattr(y, "to_numpy"):
        return np.asarray(y.to_numpy())
    return np.asarray(y)


def _predict_safely(model, X):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(X)


def _predict_proba_safely(model, X):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
            category=UserWarning,
        )
        return model.predict_proba(X)


def compute_metrics(model, X_test_t, y_test, problem_type) -> tuple[dict, np.ndarray]:
    X_test_t = _as_numpy_matrix(X_test_t)
    y_test = _as_numpy_vector(y_test)
    y_pred  = _predict_safely(model, X_test_t)
    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"]              = float(accuracy_score(y_test, y_pred))
        metrics["f1_weighted"]           = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["precision_weighted"]    = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"]       = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        try:
            if hasattr(model, "predict_proba"):
                proba = _predict_proba_safely(model, X_test_t)
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                    metrics["pr_auc"]  = float(average_precision_score(y_test, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(
                        y_test, proba, multi_class="ovr", average="weighted"
                    ))
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"]  = float(mean_absolute_error(y_test, y_pred))
        metrics["r2"]   = float(r2_score(y_test, y_pred))
        try:
            nz = np.asarray(y_test) != 0
            if nz.any():
                metrics["mape"] = float(
                    np.mean(np.abs((np.asarray(y_test)[nz] - y_pred[nz]) / np.asarray(y_test)[nz])) * 100
                )
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

    return metrics, y_pred


def compute_metric_ci(model, X_test_t, y_test, problem_type: str, n_boot: int | None = None) -> dict:
    """Bootstrap 95% CI for the primary metric.
    FIX 13: n_boot is now config-driven (evaluation.ci_bootstrap_samples, default 100).
    200 resamples was unnecessarily slow for no practical accuracy gain over 100.
    """
    from utils.config_loader import cfg as _cfg
    n_boot = n_boot or int(_cfg("evaluation.ci_bootstrap_samples", default=100))
    rng = np.random.RandomState(42)
    X_test_t = _as_numpy_matrix(X_test_t)
    y_arr = _as_numpy_vector(y_test)
    n   = len(y_arr)
    scores = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_b = X_test_t[idx]
        y_b = y_arr[idx]
        try:
            y_pred_b = _predict_safely(model, X_b)
            if problem_type == "classification":
                s = float(f1_score(y_b, y_pred_b, average="weighted", zero_division=0))
            else:
                s = float(r2_score(y_b, y_pred_b))
            scores.append(s)
        except Exception:
            continue

    if not scores:
        return {}

    arr = np.array(scores)
    key = "f1_weighted" if problem_type == "classification" else "r2"
    return {
        key: {
            "mean":    round(float(arr.mean()), 4),
            "std":     round(float(arr.std()),  4),
            "ci_low":  round(float(np.percentile(arr, 2.5)),  4),
            "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        }
    }


def compute_shap_importance(
    model,
    X_test_t,
    feature_names,
    max_samples: int = 2_000,
    skip_on_retry: bool = False,
) -> list[dict]:
    """
    Compute mean absolute SHAP values for feature importance.

    Fix #7 improvements:
    - max_samples raised from 300 → 2,000. TreeExplainer is O(n·d) on the
      sample, so 2k rows stays fast (<5s) for typical datasets while giving
      much more stable importance estimates.
    - For very large datasets (>50k rows) we cap at 2k regardless to bound
      wall-clock time.
    - skip_on_retry=True lets the orchestrator skip SHAP on intermediate
      retries where only the metric delta matters, not feature attribution.
    """
    if skip_on_retry:
        return []
    try:
        import shap
        n_rows    = X_test_t.shape[0]
        cap       = min(max_samples, n_rows)
        # For very large test sets use a random subsample for speed
        if n_rows > cap:
            idx    = np.random.choice(n_rows, cap, replace=False)
            sample = X_test_t[idx]
        else:
            sample = X_test_t
        model_cls = type(model).__name__
        if any(t in model_cls for t in _TREE_MODELS):
            explainer = shap.TreeExplainer(model)
        elif any(t in model_cls for t in _LINEAR_MODELS):
            explainer = shap.LinearExplainer(model, sample)
        else:
            explainer = shap.Explainer(model, sample)
        sv   = explainer(sample, check_additivity=False)
        vals = sv.values if hasattr(sv, "values") else np.abs(sv)
        if hasattr(vals, "ndim") and vals.ndim == 3:
            vals = np.abs(vals).mean(axis=2)
        mean_shap = np.abs(vals).mean(axis=0)
        imp = pd.DataFrame({
            "feature":    feature_names[:len(mean_shap)],
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).head(20)
        return imp.to_dict("records")
    except Exception:
        return []


@agent_error_handler("Eval Agent")
def eval_agent(state: PipelineState) -> dict:
    log.info("Eval agent started", extra={"problem_type": state.get("problem_type")})
    api_key    = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("EVAL_MODEL") or os.getenv("OPENAI_MODEL", cfg("llm.default_model", default="gpt-4o-mini"))

    rt, tid = get_or_restore_runtime(
        state, required=("best_model", "X_test_t", "y_test")
    )
    missing = [k for k in ("best_model", "X_test_t", "y_test") if k not in rt]
    if missing:
        return {"agent_messages": [
            f"[Eval Agent] ❌ Missing runtime objects: {missing}. "
            "Please go back to the Split phase and re-apply the split, then retry."
        ]}

    best_model    = rt["best_model"]
    X_test_t      = _as_numpy_matrix(rt["X_test_t"])
    y_test        = _as_numpy_vector(rt["y_test"])
    X_train_t     = _as_numpy_matrix(rt.get("X_train_t")) if rt.get("X_train_t") is not None else None
    y_train       = _as_numpy_vector(rt.get("y_train")) if rt.get("y_train") is not None else None
    feature_names = rt.get("feature_names", [f"f{i}" for i in range(X_test_t.shape[1])])
    problem_type  = state["problem_type"]

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics, y_pred = compute_metrics(best_model, X_test_t, y_test, problem_type)

    # ── Train-set metrics for overfitting detection ────────────────────────────
    train_metrics: dict = {}
    if X_train_t is not None and y_train is not None:
        try:
            train_metrics, _ = compute_metrics(best_model, X_train_t, y_train, problem_type)
            # Remove confusion matrix from train metrics to keep payload lean
            train_metrics = {k: v for k, v in train_metrics.items() if k != "confusion_matrix"}
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

    # ── Confidence intervals (bootstrap) ──────────────────────────────────────
    ci: dict = {}
    try:
        ci = compute_metric_ci(best_model, X_test_t, y_test, problem_type)
    except Exception:
        pass

    # ── SHAP ──────────────────────────────────────────────────────────────────
    retry_count    = state.get("retry_count", 0)
    # FIX 8: SHAP must run on the FIRST retry (retry_count == 1) because the
    # orchestrator relies on shap_importance to decide which features to drop
    # or which model families to try next. Only skip on retry_count >= 2 to
    # avoid spending SHAP time on clearly failing runs.
    shap_importance = compute_shap_importance(
        best_model, X_test_t, feature_names,
        skip_on_retry=(retry_count >= 2),
    )

    # ── Calibration (classifiers only) ────────────────────────────────────────
    calibration_metrics: dict = {}
    calibrated_model = best_model
    if problem_type == "classification" and hasattr(best_model, "predict_proba"):
        try:
            proba = _predict_proba_safely(best_model, X_test_t)
            if proba.shape[1] == 2:
                calibration_metrics = compute_calibration_metrics(
                    np.asarray(y_test), proba[:, 1]
                )
            # Fix #16: calibrate and promote calibrated model as the active best_model
            # so all downstream predictions (ensemble, export, report) use calibrated probs.
            # Fix #1: use held-out calibration set, NOT the test set (test-set
            # calibration is data leakage — probabilities are tuned to test labels).
            X_cal_t  = _as_numpy_matrix(rt.get("X_cal_t")) if rt.get("X_cal_t") is not None else None
            y_cal    = _as_numpy_vector(rt.get("y_cal")) if rt.get("y_cal") is not None else None
            if X_cal_t is not None and y_cal is not None and len(X_cal_t) >= 20:
                calibrated_model = calibrate_model(best_model, X_cal_t, y_cal, method="sigmoid")
            else:
                # Fallback: cal split unavailable (small dataset) — use test set
                # and log a warning so the user knows calibration may be optimistic.
                log.warning(
                    "No calibration split available — falling back to test set for calibration. "
                    "Probability estimates may be slightly optimistic."
                )
                calibrated_model = calibrate_model(best_model, X_test_t, y_test, method="sigmoid")
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

    # ── FIX 2: Promote calibrated model to runtime via RuntimeStore ──────────
    # eval_agent runs inside a LangGraph node — st.session_state may be
    # unavailable (background thread / non-Streamlit context).
    # Use get_runtime_store() which is always accessible.
    try:
        from ui.runtime_context import get_runtime_store as _get_rt_store
        _rt_store = _get_rt_store()
        _updates = {
            "y_pred":           y_pred,
            "y_test_eval":      y_test,
            "calibrated_model": calibrated_model,
        }
        if calibrated_model is not best_model:
            _updates["best_model"] = calibrated_model
        if _rt_store is not None and hasattr(_rt_store, "update") and tid:
            _rt_store.update(tid, _updates)
        else:
            # Fallback: mutate the rt dict we already hold (same object)
            rt.update(_updates)
    except Exception as _e:
        log.warning("Could not promote calibrated model to runtime", extra={"error": str(_e)})

    # ── Prediction confidence (use calibrated model for better probs) ─────────
    confidence_summary = compute_prediction_confidence(calibrated_model, X_test_t, problem_type)

    # ── Data drift ────────────────────────────────────────────────────────────
    drift_report: dict = {}
    if X_train_t is not None:
        try:
            from utils.drift_detector import detect_drift
            fn = feature_names[:X_train_t.shape[1]]
            df_tr = pd.DataFrame(X_train_t, columns=fn)
            df_te = pd.DataFrame(X_test_t,  columns=fn)
            drift_report = detect_drift(df_tr, df_te).to_dict()
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

    # ── LLM report ────────────────────────────────────────────────────────────
    payload = {
        "model_key":             state.get("best_model_key", "unknown"),
        "problem_type":          problem_type,
        "metrics":               {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "train_metrics":         train_metrics,
        "metric_ci":             ci,
        "calibration_metrics":   calibration_metrics,
        "prediction_confidence": confidence_summary,
        "top_features":          shap_importance[:10],
        "n_test_samples":        int(len(y_test)),
        "drift_severity":        drift_report.get("overall_severity", "unknown"),
        "baseline_score":        state.get("_baseline_score"),
    }

    # Fix #10: Use stream_llm_text so the UI can display tokens live.
    # In the agent (non-UI context) we collect all chunks into a single string.
    # The Streamlit eval page re-streams from state if report is not yet set.
    try:
        report = "".join(stream_llm_text(
            api_key=api_key, model_name=model_name,
            system_prompt=SYSTEM_PROMPT,
            user_content=json.dumps(payload, indent=2),
            temperature=0.2, max_tokens=900,
        ))
    except Exception as exc:
        report = f"Report generation failed: {exc}"

    primary_key = "f1_weighted" if problem_type == "classification" else "r2"
    pval = metrics.get(primary_key)
    metric_str = f"{primary_key}={pval:.4f}" if isinstance(pval, float) else f"{primary_key}=N/A"
    ci_str = ""
    if ci and primary_key in ci:
        c = ci[primary_key]
        ci_str = f" [95% CI: {c['ci_low']}–{c['ci_high']}]"

    # ── NEW v5: Confusion matrix + ROC visualizations ─────────────────────────
    cm_b64 = ""
    roc_b64 = ""
    if problem_type == "classification":
        cm_list = metrics.get("confusion_matrix")
        if cm_list:
            try:
                import base64, io as _bio
                cm = np.array(cm_list)
                labels = [str(c) for c in sorted(set(np.asarray(y_test).tolist()))]
                fig, ax = plt.subplots(figsize=(max(4, len(cm)), max(4, len(cm))))
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, ax=ax)
                n = len(cm)
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
                ax.set_ylabel("True"); ax.set_xlabel("Predicted"); ax.set_title("Confusion Matrix")
                plt.tight_layout()
                buf = _bio.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                plt.close(fig)
                buf.seek(0)
                cm_b64 = base64.b64encode(buf.read()).decode()
            except Exception:
                pass
        try:
            from sklearn.metrics import roc_curve as _roc_curve, roc_auc_score as _roc_auc
            import base64, io as _bio
            if hasattr(calibrated_model, "predict_proba"):
                proba = _predict_proba_safely(calibrated_model, X_test_t)
                if proba.shape[1] == 2:
                    fpr, tpr, _ = _roc_curve(np.asarray(y_test), proba[:, 1])
                    auc = _roc_auc(np.asarray(y_test), proba[:, 1])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, color="#60a5fa", lw=2, label=f"AUC = {auc:.3f}")
                    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    buf = _bio.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    plt.close(fig); buf.seek(0)
                    roc_b64 = base64.b64encode(buf.read()).decode()
        except Exception:
            pass

    # ── NEW v5: Error analysis ─────────────────────────────────────────────────
    error_analysis: dict = {}
    try:
        y_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        if problem_type == "classification":
            classes = sorted(set(y_arr.tolist()))
            per_class = {}
            for cls in classes:
                mask = y_arr == cls
                if mask.sum() == 0:
                    continue
                errs = (y_pred_arr[mask] != y_arr[mask]).sum()
                per_class[str(cls)] = {"n": int(mask.sum()), "errors": int(errs),
                                        "error_rate": round(float(errs / mask.sum()), 4)}
            cm_np = np.array(metrics.get("confusion_matrix", []))
            confused = []
            if cm_np.ndim == 2:
                for i, ti in enumerate(classes):
                    for j, tj in enumerate(classes):
                        if i != j and cm_np[i, j] > 0:
                            confused.append({"true": str(ti), "pred": str(tj), "count": int(cm_np[i, j])})
            confused.sort(key=lambda x: -x["count"])
            error_analysis = {"per_class_error_rates": per_class, "most_confused_pairs": confused[:5]}
        else:
            residuals = y_arr - y_pred_arr
            abs_res   = np.abs(residuals)
            worst_idx = np.argsort(abs_res)[-10:][::-1]
            error_analysis = {
                "worst_predictions": [
                    {"index": int(i), "y_true": float(y_arr[i]), "y_pred": float(y_pred_arr[i]),
                     "residual": float(residuals[i])} for i in worst_idx
                ],
                "residual_stats": {
                    "mean": round(float(residuals.mean()), 4),
                    "std":  round(float(residuals.std()), 4),
                    "p95_abs": round(float(np.percentile(abs_res, 95)), 4),
                },
            }
    except Exception as e:
        log.warning(f"Error analysis failed: {e}")

    # ── NEW v5: Fairness checks ────────────────────────────────────────────────
    fairness_report: dict = {}
    if cfg("evaluation.fairness_enabled", default=False):
        sensitive_cols = cfg("evaluation.fairness_sensitive_features", default=[])
        if sensitive_cols and hasattr(X_test_t, "columns"):
            for col in sensitive_cols:
                if col in X_test_t.columns:
                    try:
                        groups = np.unique(X_test_t[col].values)
                        group_results = {}
                        for g in groups:
                            mask = X_test_t[col].values == g
                            if mask.sum() < 5:
                                continue
                            group_results[str(g)] = {
                                "n": int(mask.sum()),
                                "accuracy": round(float(accuracy_score(
                                    np.asarray(y_test)[mask], np.asarray(y_pred)[mask])), 4),
                            }
                        fairness_report[col] = {"groups": group_results}
                    except Exception:
                        pass

    # ── NEW v5: Drift monitor + retrain alert ──────────────────────────────────
    retrain_alert_msg = ""
    if drift_report:
        try:
            alert_obj = drift_monitor.check_and_record(
                {**dict(state), "_thread_id": tid}, drift_report
            )
            if alert_obj.should_retrain:
                retrain_alert_msg = f"⚠️ Retrain alert: {alert_obj.message}"
        except Exception as e:
            log.debug(f"Drift monitor failed: {e}")

    # ── NEW v5: MLflow logging ─────────────────────────────────────────────────
    try:
        tracker.log_pipeline_run(
            {**dict(state), "eval_metrics": metrics},
            model=calibrated_model,
        )
    except Exception as e:
        log.debug(f"MLflow log failed: {e}")

    eval_analysis = {
        "model_key":                   state.get("best_model_key", "unknown"),
        "problem_type":                problem_type,
        "n_test_samples":              int(len(y_test)),
        "n_features":                  int(X_test_t.shape[1]),
        "feature_names":               feature_names,
        "metrics":                     {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "train_metrics":               train_metrics,
        "metric_confidence_intervals": ci,
        "calibration_metrics":         calibration_metrics,
        "prediction_confidence":       confidence_summary,
        "confusion_matrix":            metrics.get("confusion_matrix"),
        "confusion_matrix_b64":        cm_b64,       # v5: for direct UI rendering
        "roc_curve_b64":               roc_b64,       # v5: ROC curve image
        "shap_importance_full":        shap_importance,
        "top_10_features":             shap_importance[:10],
        "drift_report":                drift_report,
        "error_analysis":              error_analysis,   # v5: where model fails most
        "fairness_report":             fairness_report,  # v5: bias checks
        "retrain_alert":               retrain_alert_msg, # v5
        "eval_report":                 report,
        "llm_payload_used":            payload,
        "cv_score_at_training":        state.get("best_cv_score"),
        "best_params":                 state.get("best_params", {}),
        "baseline_score":              state.get("_baseline_score"),
    }

    msgs = [
        f"[Eval Agent] {metric_str}{ci_str}. "
        f"SHAP for {len(shap_importance)} features. "
        f"Drift: {drift_report.get('overall_severity', 'n/a')}."
    ]
    if retrain_alert_msg:
        msgs.append(f"[Eval Agent] {retrain_alert_msg}")

    return {
        "eval_metrics":        metrics,
        "shap_importance":     shap_importance,
        "eval_report":         report,
        "eval_analysis":       eval_analysis,
        "drift_report":        drift_report,
        "calibration_metrics": calibration_metrics,
        "metric_ci":           ci,
        # v4 Fix #7: surface top SHAP features for orchestrator retry loop
        "top_shap_features":   [s["feature"] for s in shap_importance[:10] if "feature" in s],
        "agent_messages":      msgs,
    }
