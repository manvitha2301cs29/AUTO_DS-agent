"""
utils/model_calibration.py — Probability Calibration & Uncertainty Estimation

Provides:
  calibrate_model()        — Platt scaling (sigmoid) or isotonic regression
  compute_calibration_metrics() — ECE, MCE, reliability diagram data
  compute_uncertainty()    — prediction intervals via bootstrap or conformal

Used after eval_agent on classification models to improve probability estimates.
"""

from __future__ import annotations
from typing import Any
import warnings

import numpy as np


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


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(model, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "sigmoid"):
    """
    Calibrate a fitted classifier using CalibratedClassifierCV.

    Parameters
    ----------
    model   : already-fitted sklearn-compatible classifier
    X_cal   : calibration features (NEVER training data — use a held-out cal set)
    y_cal   : calibration labels
    method  : "sigmoid" (Platt) or "isotonic"

    Returns a calibrated classifier that wraps the original.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",  # model is already fitted
        )
        calibrated.fit(X_cal, y_cal)
        return calibrated
    except Exception:
        return model  # fall back to uncalibrated


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
    and per-bin reliability data for plotting a calibration curve.

    Works for binary classification (y_prob = probability of positive class).

    Returns
    -------
    {
        "ece": float,
        "mce": float,
        "brier_score": float,
        "bins": [{"mean_conf": float, "mean_acc": float, "n": int}]
    }
    """
    try:
        from sklearn.metrics import brier_score_loss
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = None

    bins_data: list[dict] = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n   = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        n_bin    = int(mask.sum())
        mean_conf = float(y_prob[mask].mean())
        mean_acc  = float(y_true[mask].mean())
        cal_err   = abs(mean_conf - mean_acc)
        ece += (n_bin / n) * cal_err
        mce = max(mce, cal_err)
        bins_data.append({"mean_conf": round(mean_conf, 4), "mean_acc": round(mean_acc, 4), "n": n_bin})

    return {
        "ece":         round(ece, 4),
        "mce":         round(mce, 4),
        "brier_score": round(brier, 4) if brier is not None else None,
        "bins":        bins_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_uncertainty_regression(
    model,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bootstrap: int = 50,
    confidence: float = 0.90,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Bootstrap prediction intervals for regression models.

    Trains n_bootstrap models on bootstrap samples of training data,
    then uses the distribution of predictions as an uncertainty estimate.

    Returns
    -------
    {
        "mean_predictions": list,
        "lower_bound": list,
        "upper_bound": list,
        "interval_width_mean": float,
        "confidence": float,
    }
    """
    rng = np.random.RandomState(random_state)
    preds: list[np.ndarray] = []

    model_cls = type(model)
    try:
        params = model.get_params()
    except Exception:
        params = {}

    for i in range(n_bootstrap):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_b, y_b = X_train[idx], y_train[idx]
        try:
            m = model_cls(**params)
            m.fit(X_b, y_b)
            preds.append(_predict_safely(m, X_test))
        except Exception:
            continue

    if not preds:
        mean_pred = _predict_safely(model, X_test)
        return {
            "mean_predictions":    mean_pred.tolist(),
            "lower_bound":         mean_pred.tolist(),
            "upper_bound":         mean_pred.tolist(),
            "interval_width_mean": 0.0,
            "confidence":          confidence,
        }

    preds_arr = np.array(preds)  # shape (n_bootstrap, n_samples)
    alpha = (1 - confidence) / 2
    lower = np.quantile(preds_arr, alpha,     axis=0)
    upper = np.quantile(preds_arr, 1 - alpha, axis=0)
    mean  = preds_arr.mean(axis=0)

    return {
        "mean_predictions":    mean.tolist(),
        "lower_bound":         lower.tolist(),
        "upper_bound":         upper.tolist(),
        "interval_width_mean": float(np.mean(upper - lower)),
        "confidence":          confidence,
        "n_bootstrap_used":    len(preds),
    }


def compute_prediction_confidence(
    model,
    X_test: np.ndarray,
    problem_type: str,
) -> dict[str, Any]:
    """
    Simple confidence summary for any model.
    Classification: max class probability per sample.
    Regression: not directly available — returns None.
    """
    if problem_type != "classification":
        return {"available": False, "reason": "Confidence intervals require regression bootstrap."}

    if not hasattr(model, "predict_proba"):
        return {"available": False, "reason": "Model does not support predict_proba."}

    try:
        proba = _predict_proba_safely(model, X_test)
        max_conf = proba.max(axis=1)
        return {
            "available":        True,
            "mean_confidence":  round(float(max_conf.mean()), 4),
            "pct_high_conf":    round(float((max_conf >= 0.9).mean() * 100), 1),
            "pct_uncertain":    round(float((max_conf < 0.6).mean() * 100), 1),
            "min_confidence":   round(float(max_conf.min()), 4),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
