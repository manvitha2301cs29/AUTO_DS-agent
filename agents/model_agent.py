"""
agents/model_agent.py — Model Selection + Optuna Tuning Agent (v2)

Improvements over v1:
  - Meta-learning warm-start: suggest_models() seeds candidate list from similar past runs
  - Parallel Optuna: n_jobs=-1 where sampler supports it; joblib for independent trials
  - Loss function selection: select_loss_function() chooses optimal objective
  - All trained models stored in runtime for ensemble_agent to use
  - Caching: completed Optuna studies cached by (model_key, data_hash) key
  - Basic NAS: for deep_learning, Optuna also searches number of layers/units
    with LLM-chosen architectural constraints (retained from v1)
  - Improved error handling: per-model fallback, never crashes entire agent
  - class_weight / sample_weight injection retained from v1 via utils.runtime
"""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
import time
import warnings

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json
from utils.logger import get_logger

log = get_logger(__name__)
from utils.runtime import (
    get_or_restore_runtime,
    store_arrays_to_state,
    inject_class_weight,
    needs_sample_weight,
    compute_sample_weight,
)
from utils.serialization import obj_to_b64
from utils.ml_helpers import compute_baseline_score, select_loss_function, GLOBAL_SEED, set_global_seed
from utils.config_loader import cfg
from ui.runtime_context import append_training_event, get_runtime_store, set_training_progress

# ── Apply global seed at import time ─────────────────────────────────────────
set_global_seed()

optuna.logging.set_verbosity(optuna.logging.WARNING)
_DEEP_LEARNING_ENABLED = os.getenv("AUTOML_ENABLE_DEEP_LEARNING", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# ── Optuna study cache: {cache_key: best_params} ─────────────────────────────
# FIX 14: cap cache at 200 entries (LRU-style, oldest first) to prevent unbounded
# memory growth across many sessions in a long-running process.
_STUDY_CACHE: dict[str, dict] = {}
_STUDY_CACHE_MAX = 200

def _cache_put(key: str, value: dict) -> None:
    """Insert into study cache, evicting oldest entry when at capacity."""
    if key in _STUDY_CACHE:
        _STUDY_CACHE.pop(key)           # remove to re-insert at end (LRU)
    elif len(_STUDY_CACHE) >= _STUDY_CACHE_MAX:
        _STUDY_CACHE.pop(next(iter(_STUDY_CACHE)))  # evict oldest
    _STUDY_CACHE[key] = value


def _get_runtime_bucket(tid: str):
    rt_store = get_runtime_store()
    if rt_store is not None:
        return rt_store
    return None


def _append_training_event(tid: str, message: str, stage: str | None = None) -> None:
    try:
        append_training_event(tid, message, stage=stage)
    except Exception:
        pass


def _update_training_progress(tid: str, **updates) -> None:
    try:
        set_training_progress(tid, updates)
    except Exception:
        pass


def _resolve_training_tid(state: dict, fallback: str = "") -> str:
    tid = str(state.get("_thread_id") or fallback or "")
    if tid:
        return tid
    rt_store = get_runtime_store()
    if rt_store is not None:
        raw = getattr(rt_store, "_raw", {})
        if len(raw) == 1:
            return next(iter(raw.keys()))
    return ""


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


@contextmanager
def _suppress_lgbm_feature_name_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
            category=UserWarning,
        )
        yield


def _predict_safely(model, X):
    with _suppress_lgbm_feature_name_warning():
        return model.predict(X)


def _predict_proba_safely(model, X):
    with _suppress_lgbm_feature_name_warning():
        return model.predict_proba(X)


def _fallback_candidate_templates(problem_type: str) -> list[dict]:
    """Return default hyperparameter search spaces for all supported model types.

    This covers EVERY model key that _make_model() accepts so that user-selected
    models without LLM-enriched hyperparams always get a sensible search space.
    """
    shared: dict[str, dict] = {
        "xgboost": {
            "rationale": "Robust boosted-tree baseline for tabular data.",
            "hyperparams": {
                "n_estimators":      {"type": "int",   "low": 100,  "high": 500},
                "max_depth":         {"type": "int",   "low": 3,    "high": 9},
                "learning_rate":     {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
                "subsample":         {"type": "float", "low": 0.6,  "high": 1.0},
                "colsample_bytree":  {"type": "float", "low": 0.5,  "high": 1.0},
                "reg_alpha":         {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
                "reg_lambda":        {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
                "min_child_weight":  {"type": "int",   "low": 1,    "high": 10},
            },
        },
        "random_forest": {
            "rationale": "Stable bagged-tree baseline that handles nonlinear structure well.",
            "hyperparams": {
                "n_estimators":     {"type": "int",   "low": 100, "high": 500},
                "max_depth":        {"type": "int",   "low": 3,   "high": 20},
                "min_samples_split":{"type": "int",   "low": 2,   "high": 20},
                "min_samples_leaf": {"type": "int",   "low": 1,   "high": 10},
                "max_features":     {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7]},
                "max_samples":      {"type": "float", "low": 0.6, "high": 1.0},
            },
        },
        "hist_gradient_boosting": {
            "rationale": "Fast histogram boosting with native NaN support and early stopping.",
            "hyperparams": {
                "learning_rate":      {"type": "float", "low": 0.01,  "high": 0.3,   "log": True},
                "max_depth":          {"type": "int",   "low": 3,     "high": 10},
                "max_leaf_nodes":     {"type": "int",   "low": 15,    "high": 63},
                "min_samples_leaf":   {"type": "int",   "low": 10,    "high": 40},
                "l2_regularization":  {"type": "float", "low": 1e-4,  "high": 1.0,   "log": True},
                "max_features":       {"type": "float", "low": 0.5,   "high": 1.0},
            },
        },
        "extra_trees": {
            "rationale": "High-variance ensemble that often performs well on medium tabular data.",
            "hyperparams": {
                "n_estimators": {"type": "int", "low": 100, "high": 400},
                "max_depth": {"type": "int", "low": 3, "high": 20},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
            },
        },
        "lightgbm": {
            "rationale": "Fast gradient boosting with leaf-wise growth.",
            "hyperparams": {
                "n_estimators":     {"type": "int",   "low": 100,  "high": 500},
                "max_depth":        {"type": "int",   "low": 3,    "high": 10},
                "learning_rate":    {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
                "num_leaves":       {"type": "int",   "low": 15,   "high": 127},
                "subsample":        {"type": "float", "low": 0.6,  "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5,  "high": 1.0},
                "reg_alpha":        {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
                "reg_lambda":       {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
                "min_child_samples":{"type": "int",   "low": 10,   "high": 60},
            },
        },
        "catboost": {
            "rationale": "Gradient boosting with native categorical support and built-in early stopping.",
            "hyperparams": {
                "iterations":          {"type": "int",   "low": 100,  "high": 600},
                "depth":               {"type": "int",   "low": 4,    "high": 8},
                "learning_rate":       {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "l2_leaf_reg":         {"type": "float", "low": 1.0,  "high": 10.0, "log": True},
                "bagging_temperature": {"type": "float", "low": 0.0,  "high": 1.0},
                "random_strength":     {"type": "float", "low": 0.1,  "high": 3.0},
                "border_count":        {"type": "int",   "low": 32,   "high": 128},
                "min_data_in_leaf":    {"type": "int",   "low": 1,    "high": 20},
            },
        },
        "gradient_boosting": {
            "rationale": "Classic sklearn gradient boosting baseline.",
            "hyperparams": {
                "n_estimators": {"type": "int", "low": 80, "high": 300},
                "max_depth": {"type": "int", "low": 2, "high": 8},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            },
        },
        "mlp": {
            "rationale": "Multi-layer perceptron for learning complex feature interactions.",
            "hyperparams": {
                # Use strings instead of tuples — Optuna SQLite storage cannot
                # persist tuple categorical choices and silently marks trials as FAIL.
                "hidden_layer_sizes": {"type": "categorical", "choices": ["64", "128", "64-32", "128-64", "256-128"]},
                "alpha":              {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
                "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            },
        },
        "svm": {
            "rationale": "SVM with RBF kernel for moderate-sized datasets.",
            "hyperparams": {
                "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            },
        },
    }

    # Classification-only models
    clf_only: dict[str, dict] = {
        "logistic_regression": {
            "rationale": "Linear baseline classifier; fast and interpretable.",
            "hyperparams": {
                "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
                "solver": {"type": "categorical", "choices": ["lbfgs", "saga"]},
            },
        },
        "adaboost": {
            "rationale": "Adaptive boosting ensemble for binary/multiclass classification.",
            "hyperparams": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 2.0, "log": True},
            },
        },
    }

    # Regression-only models
    reg_only: dict[str, dict] = {
        "ridge": {
            "rationale": "L2-regularised linear regression baseline.",
            "hyperparams": {
                "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            },
        },
        "lasso": {
            "rationale": "L1-regularised regression with automatic feature selection.",
            "hyperparams": {
                "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            },
        },
        "elasticnet": {
            "rationale": "Elastic Net combines L1+L2 regularisation for correlated features.",
            "hyperparams": {
                "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
                "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0},
            },
        },
    }

    if problem_type == "classification":
        pool = {**shared, **clf_only}
    else:
        pool = {**shared, **reg_only}

    return [{"model_key": mk, **spec} for mk, spec in pool.items()]


def _ensure_minimum_candidates(candidates: list[dict], problem_type: str, min_candidates: int) -> list[dict]:
    merged = list(candidates or [])
    seen = {str(c.get("model_key") or "").strip() for c in merged if str(c.get("model_key") or "").strip()}
    if len(seen) >= min_candidates:
        return merged
    for fallback in _fallback_candidate_templates(problem_type):
        model_key = str(fallback.get("model_key") or "").strip()
        if not model_key or model_key in seen:
            continue
        merged.append(fallback)
        seen.add(model_key)
        if len(seen) >= min_candidates:
            break
    return merged


def _dedupe_and_limit_candidates(candidates: list[dict], max_candidates: int) -> list[dict]:
    filtered: list[dict] = []
    seen: set[str] = set()
    for cand in candidates:
        model_key = str(cand.get("model_key") or "").strip()
        if not model_key or model_key in seen:
            continue
        seen.add(model_key)
        filtered.append(cand)
        if len(filtered) >= max_candidates:
            break
    return filtered


def _tuning_percent(completed_trials: int, total_trials: int) -> int:
    frac = completed_trials / max(total_trials, 1)
    return max(35, min(95, 35 + int(frac * 60)))


# ─────────────────────────────────────────────────────────────────────────────
# LLM System Prompt (unchanged from v1 + meta-learning addendum)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a world-class AutoML engineer whose sole job is to produce the best possible
Optuna hyperparameter search spaces for a given dataset.

You will receive a rich JSON payload describing the dataset in full detail. You MUST
use every signal in that payload to calibrate your search spaces. Generic / copy-paste
ranges are unacceptable — every bound must be justified by the data characteristics.

═══════════════════════════════════════════════════════════════
DATASET SIGNALS AND HOW TO USE THEM
═══════════════════════════════════════════════════════════════

n_samples  (critical for every decision)
  < 500    → very small: avoid complex models (deep_learning, large XGB/LGBM);
             prefer simpler models (logistic_regression, ridge, svm, random_forest
             with low n_estimators); use tight regularisation ranges; lower n_estimators.
  500–5k   → medium: balanced choices; moderate depth/estimators.
  5k–50k   → large: boosting methods shine; wider depth ranges OK.
  > 50k    → very large: hist_gradient_boosting, lightgbm, xgboost preferred for speed;
             skip svm entirely; deep_learning viable.

n_features  (controls depth, complexity, regularisation)
  < 10     → shallow trees work (max_depth 2–6); strong regularisation for linear models.
  10–50    → moderate depth OK (max_depth 3–10); standard regularisation ranges.
  > 50     → feature-rich: use colsample/subsample aggressively; prefer L1 or ElasticNet
             for linear; increase regularisation strength ranges.

class_imbalance_ratio  (classification only — heavily influences model and regularisation)
  < 2:1    → balanced: standard scoring, no special handling needed.
  2:1–5:1  → mild imbalance: prefer models with class_weight support; tighten C/alpha.
  5:1–20:1 → severe: MUST use models with class_weight; tree depth may need tightening
             to avoid majority-class bias; prefer recall-oriented search.
  > 20:1   → extreme: restrict to models that natively handle extreme imbalance
             (hist_gradient_boosting, xgboost, lightgbm, random_forest);
             avoid logistic_regression and svm unless regularisation is very strong.

high_skew_columns  (from EDA)
  Many skewed features → tree-based models are robust; linear models benefit from
  log-transformed inputs (already handled upstream), but still prefer trees.

high_null_columns  (from EDA)
  Columns with > 20 % nulls → hist_gradient_boosting and lightgbm handle natively;
  random_forest with imputation is robust; prefer these over xgboost or linear models.

categorical_columns  (from EDA)
  Many categorical features → catboost handles natively; lightgbm with categorical
  support also good; tree-based models in general better than linear.

n_engineered_features  (from feature_analysis)
  High ratio of engineered to raw features → data may be sparse; prefer models with
  built-in regularisation (hist_gradient_boosting, xgboost with L1/L2).

shap_top_features  (from feature_analysis — if provided)
  Fewer than 5 features driving > 80 % importance → sparse signal; strong regularisation;
  consider simpler models to avoid overfitting.

n_trials  (Optuna budget — calibrate search space WIDTH accordingly)
  < 30 trials → narrow search spaces with tight, focused ranges; avoid categorical
               params with many choices; reduce the number of params searched.
  30–80 trials → moderate width; 3–5 hyperparams per model is ideal.
  > 80 trials → wider ranges OK; more categorical choices acceptable.

cv_folds  (affects variance of each trial — influences how wide to search)
  3 folds → high variance per trial → narrower, safer ranges.
  5 folds → standard → balanced ranges.
  ≥ 7 folds → low variance → can afford wider exploration.

drift_severity  (if provided)
  medium/high → prefer tree-based models robust to distributional shift
               (hist_gradient_boosting, lightgbm, xgboost, random_forest);
               avoid SVM and linear models.

meta_memory_hints  (past runs on similar datasets — highest priority signal)
  If provided, anchor your primary candidates on these model keys. They have a
  proven track record on similar data. Adjust their hyperparams to the current
  dataset size/features, do not copy ranges blindly.

previous_best  (last run's winner)
  On retry, INCLUDE this model and WIDEN its search space by ~30 % to explore
  neighbourhoods the previous run may have missed. Do NOT simply repeat prior ranges.

═══════════════════════════════════════════════════════════════
HYPERPARAMETER RANGE CALIBRATION RULES (MANDATORY)
═══════════════════════════════════════════════════════════════

n_estimators / iterations
  • n_samples < 1000  → low=50,  high=200
  • n_samples < 5000  → low=80,  high=350
  • n_samples < 20000 → low=100, high=500
  • n_samples ≥ 20000 → low=200, high=800

max_depth (trees)
  • n_features < 10   → low=2, high=6
  • n_features < 30   → low=3, high=10
  • n_features ≥ 30   → low=3, high=15
  • If high imbalance (> 5:1): reduce high by 2 to avoid overfitting majority class.

learning_rate (always log=true)
  • Tight budget (n_trials < 30) → low=0.02, high=0.2
  • Standard                     → low=0.005, high=0.3
  • Large dataset (> 20k)        → low=0.005, high=0.5

regularisation (C for SVM/LR, alpha for Ridge/Lasso, l2_leaf_reg for CatBoost)
  Always log=true. Range anchored on dataset complexity:
  • Simple (n_features < 15, n_samples < 2000) → low=0.1,   high=10.0
  • Moderate                                    → low=0.01,  high=100.0
  • Complex (n_features > 50 or > 20k samples) → low=0.001, high=1000.0

subsample / colsample_bytree / bagging_fraction
  • n_features < 20  → low=0.7, high=1.0
  • n_features ≥ 20  → low=0.5, high=1.0
  If imbalance > 5:1: lower bound to 0.4 to see minority class more often.

min_samples_leaf / min_child_samples (regularisation for trees)
  • n_samples < 2000  → low=1,  high=20
  • n_samples < 10000 → low=5,  high=50
  • n_samples ≥ 10000 → low=10, high=100

num_leaves (LightGBM)
  • Derived from max_depth: num_leaves high ≈ 2^max_depth_high × 0.75
  • Floor at 16, ceiling at 256.

═══════════════════════════════════════════════════════════════
MODEL SELECTION PRIORITY MATRIX
═══════════════════════════════════════════════════════════════

Always recommend exactly 3–5 candidates. Selection priority:

CLASSIFICATION
  • Default top picks (data-size agnostic): xgboost, lightgbm, random_forest
  • Add hist_gradient_boosting if n_samples > 5000 OR many null columns
  • Add catboost if many categorical columns (has_catboost must be true)
  • Add logistic_regression as interpretable baseline if n_features < 30
  • Add extra_trees as diversity pick if n_features > 20
  • Avoid svm if n_samples > 10000 (too slow)
  • Avoid adaboost if imbalance_ratio > 5 (sensitive to noise)
  • deep_learning: ONLY if n_samples ≥ 1000 AND n_features ≥ 8 AND AUTOML_ENABLE_DEEP_LEARNING

REGRESSION
  • Default top picks: xgboost, lightgbm, random_forest
  • Add hist_gradient_boosting if n_samples > 5000 OR many nulls
  • Add ridge/elasticnet as fast linear baseline if n_features < 50
  • Add extra_trees for high-dimensional datasets
  • Avoid lasso if correlated features expected (use elasticnet instead)
  • Avoid svm if n_samples > 5000

RETRY LOGIC (retry_count > 0)
  • Keep the previous_best model with widened ranges (+30 %)
  • Replace the worst-performing model from last run with a fresh alternative
  • Add at least one model NOT tried in the previous run

═══════════════════════════════════════════════════════════════
DEEP LEARNING (model_key == "deep_learning")
═══════════════════════════════════════════════════════════════

Only include if n_samples ≥ 1000 AND n_features ≥ 8.
MUST include nn_config block when selected:
  hidden_activation : "relu" | "tanh" | "elu" | "selu"
    → "relu" for large datasets; "elu"/"selu" for small/sparse; "tanh" for balanced.
  use_dropout       : true  if n_samples < 5000 OR imbalance_ratio > 3
  use_l2            : true  if n_features > 20 OR high multicollinearity expected
  use_batch_norm    : true  if n_samples > 2000 AND n_features > 10
  overfitting_risk  : "low" | "medium" | "high"
    → "high" if n_samples < 2000; "medium" if 2000–10000; "low" if > 10000
  rationale         : one sentence justifying ALL four choices together

Hyperparams for deep_learning must include:
  n_layers       : {"type": "int",   "low": 1, "high": 3}   (2–4 if n_samples > 5000)
  units          : {"type": "categorical", "choices": [32, 64, 128, 256]}
  dropout_rate   : {"type": "float", "low": 0.1, "high": 0.5}  (only if use_dropout)
  learning_rate  : {"type": "float", "low": 0.0001, "high": 0.01, "log": true}
  batch_size     : {"type": "categorical", "choices": [16, 32, 64, 128]}

═══════════════════════════════════════════════════════════════
MODEL-SPECIFIC PARAMETER CONSTRAINTS (STRICTLY ENFORCED)
═══════════════════════════════════════════════════════════════

hist_gradient_boosting — ALLOWED params ONLY (others silently stripped):
  learning_rate, max_depth, max_leaf_nodes, min_samples_leaf,
  l2_regularization, max_features (float 0.5–1.0)
  DO NOT USE: n_estimators, max_iter, subsample, colsample_bytree,
  n_jobs, reg_alpha, reg_lambda, num_leaves — these cause TypeError -> -inf.
  max_features must be a float between 0.5 and 1.0.

catboost — ALLOWED params ONLY:
  iterations, depth, learning_rate, l2_leaf_reg, bagging_temperature,
  random_strength, border_count (int, keep 32–128 max), min_data_in_leaf, subsample
  DO NOT USE: n_estimators, n_jobs, verbose, od_type, od_wait (injected automatically).
  border_count: use low=32, high=128 (NOT 255 — too wide wastes trials).
  depth: use low=4, high=8 (NOT 3 — shallow trees hurt CatBoost).
  l2_leaf_reg: low=1.0, high=10.0 (NOT 20.0 — too high kills accuracy).

lightgbm — DO NOT USE: n_estimators (use num_leaves instead for leaf-wise growth).
  min_child_samples low should be ≥ 5 to avoid overfitting on small folds.

xgboost — n_estimators low ≥ 50, use eval_metric is injected automatically.



{
  "candidates": [
    {
      "model_key": "<one of: random_forest | extra_trees | hist_gradient_boosting | gradient_boosting | logistic_regression | ridge | lasso | elasticnet | svm | xgboost | lightgbm | catboost | adaboost | mlp | deep_learning>",
      "rationale": "<one precise sentence citing specific dataset signals: e.g. '5200 samples with 23 features and mild 2.4:1 imbalance suits XGBoost's gradient boosting with moderate depth and colsample regularisation'>",
      "hyperparams": {
        "<param_name>": {
          "type": "int | float | categorical",
          "low": <number — omit for categorical>,
          "high": <number — omit for categorical>,
          "log": <true | false — only for float/int, true when range spans > 2 orders of magnitude>,
          "choices": [<list — only for categorical>]
        }
      },
      "nn_config": { <ONLY present when model_key == "deep_learning"> }
    }
  ],
  "recommendation": "<one sentence naming the single expected best model and WHY, citing the most important dataset signal>"
}

VALIDATION CHECKLIST (apply before outputting):
  ✓ Every "low" < "high"
  ✓ log=true on every float param spanning > 2 orders of magnitude
  ✓ No param named "random_state" or "verbose" or "n_jobs" (injected by pipeline)
  ✓ nn_config present IFF model_key == "deep_learning"
  ✓ Exactly 3–5 candidates total
  ✓ Each rationale cites at least one specific number from the dataset payload
  ✓ No duplicate model_key values
  ✓ hist_gradient_boosting: no n_estimators, no max_iter, no subsample, no colsample_bytree
  ✓ catboost: no n_estimators, no n_jobs; border_count high ≤ 128; depth low ≥ 4
  ✓ lightgbm: min_child_samples low ≥ 5
"""


# ─────────────────────────────────────────────────────────────────────────────
# Keras Neural Network Wrapper (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

class KerasNNWrapper(BaseEstimator):
    def __init__(self, problem_type="classification", n_classes=2, n_features=10,
                 hidden_activation="relu", use_dropout=True, use_l2=False,
                 use_batch_norm=False, n_layers=2, units=64, dropout_rate=0.3,
                 l2_lambda=1e-4, learning_rate=1e-3, batch_size=32,
                 epochs=100, random_state=GLOBAL_SEED):
        self.problem_type = problem_type; self.n_classes = n_classes
        self.n_features = n_features; self.hidden_activation = hidden_activation
        self.use_dropout = use_dropout; self.use_l2 = use_l2
        self.use_batch_norm = use_batch_norm; self.n_layers = n_layers
        self.units = units; self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda; self.learning_rate = learning_rate
        self.batch_size = batch_size; self.epochs = epochs
        self.random_state = random_state
        self.model_ = None; self.classes_ = None; self._class_map = {}

    def _build_model(self, input_dim):
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("tensorflow is required for deep_learning.")
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        l2_reg = keras.regularizers.l2(self.l2_lambda) if self.use_l2 else None
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=(input_dim,)))
        for _ in range(self.n_layers):
            model.add(keras.layers.Dense(self.units, activation=self.hidden_activation,
                                         kernel_regularizer=l2_reg))
            if self.use_batch_norm:
                model.add(keras.layers.BatchNormalization())
            if self.use_dropout:
                model.add(keras.layers.Dropout(self.dropout_rate))
        if self.problem_type == "classification":
            if self.n_classes == 2:
                model.add(keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(keras.layers.Dense(self.n_classes, activation="softmax"))
                loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:
            model.add(keras.layers.Dense(1, activation="linear"))
            loss = "mse"; metrics = ["mae"]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=loss, metrics=metrics)
        return model

    def fit(self, X, y, sample_weight=None):
        try:
            from tensorflow import keras
        except ImportError:
            raise
        if self.problem_type == "classification":
            classes = np.unique(y)
            self.classes_ = classes
            self._class_map = {c: i for i, c in enumerate(classes)}
            y_enc = np.array([self._class_map[c] for c in y])
            self.n_classes = len(classes)
        else:
            y_enc = np.array(y, dtype=float)
        self.model_ = self._build_model(X.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]
        val_split = 0.1 if len(X) >= 100 else 0.0
        self.model_.fit(X, y_enc, epochs=self.epochs, batch_size=self.batch_size,
                        validation_split=val_split, callbacks=callbacks,
                        verbose=0, sample_weight=sample_weight)
        return self

    def predict(self, X):
        raw = self.model_.predict(X, verbose=0)
        if self.problem_type == "classification":
            if self.n_classes == 2:
                idx = (raw.flatten() > 0.5).astype(int)
            else:
                idx = raw.argmax(axis=1)
            inv_map = {v: k for k, v in self._class_map.items()}
            return np.array([inv_map[i] for i in idx])
        return raw.flatten()

    def predict_proba(self, X):
        raw = self.model_.predict(X, verbose=0)
        if self.n_classes == 2:
            p = raw.flatten()
            return np.column_stack([1 - p, p])
        return raw

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in [
            "problem_type", "n_classes", "n_features", "hidden_activation",
            "use_dropout", "use_l2", "use_batch_norm", "n_layers", "units",
            "dropout_rate", "l2_lambda", "learning_rate", "batch_size",
            "epochs", "random_state"
        ]}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Model factories (unchanged structure, exposed for ensemble_agent)
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(model_key: str, params: dict, problem_type: str,
                n_classes: int = 2, n_features: int = 10,
                nn_config: dict | None = None,
                parallel_jobs: int = -1):
    p = params.copy()
    if model_key == "random_forest":
        return (RandomForestClassifier(**p, random_state=GLOBAL_SEED, n_jobs=parallel_jobs)
                if problem_type == "classification"
                else RandomForestRegressor(**p, random_state=GLOBAL_SEED, n_jobs=parallel_jobs))
    if model_key == "extra_trees":
        return (ExtraTreesClassifier(**p, random_state=GLOBAL_SEED, n_jobs=parallel_jobs)
                if problem_type == "classification"
                else ExtraTreesRegressor(**p, random_state=GLOBAL_SEED, n_jobs=parallel_jobs))
    if model_key == "hist_gradient_boosting":
        # HistGradientBoosting has a STRICT set of valid constructor params.
        # The LLM sometimes hallucinates XGB/LGBM params (e.g. n_estimators,
        # subsample, colsample_bytree) which cause TypeError on every trial -> -inf.
        # Whitelist only the known valid params and strip everything else.
        import sklearn as _sklearn_mod
        _sklearn_ver = tuple(int(x) for x in _sklearn_mod.__version__.split(".")[:2])
        _hgb_supports_cw = _sklearn_ver >= (1, 6)
        _HGB_VALID = {
            "loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
            "min_samples_leaf", "l2_regularization", "max_features", "max_bins",
            "categorical_features", "monotonic_cst", "interaction_cst", "warm_start",
            "early_stopping", "scoring", "validation_fraction", "n_iter_no_change",
            "tol", "class_weight",
        }
        p_clean = {k: v for k, v in p.items() if k in _HGB_VALID}
        if not _hgb_supports_cw or problem_type != "classification":
            p_clean.pop("class_weight", None)
        # When max_iter is tuned, disable early_stopping to avoid silent conflicts
        if "max_iter" in p_clean:
            p_clean.setdefault("early_stopping", False)
        if problem_type == "classification":
            return HistGradientBoostingClassifier(**p_clean, random_state=GLOBAL_SEED)
        else:
            return HistGradientBoostingRegressor(**p_clean, random_state=GLOBAL_SEED)
    if model_key == "gradient_boosting":
        return (GradientBoostingClassifier(**p, random_state=GLOBAL_SEED)
                if problem_type == "classification"
                else GradientBoostingRegressor(**p, random_state=GLOBAL_SEED))
    if model_key == "logistic_regression":
        # sklearn 1.8+: n_jobs and penalty are both deprecated — remove them.
        # saga solver supports both l1 and l2 via l1_ratio; lbfgs is l2-only.
        p = {k: v for k, v in p.items() if k not in ("n_jobs", "penalty")}
        return LogisticRegression(**p, random_state=GLOBAL_SEED, max_iter=1000)
    if model_key == "ridge":
        return Ridge(**p)
    if model_key == "lasso":
        return Lasso(**p, random_state=GLOBAL_SEED, max_iter=2000)
    if model_key == "elasticnet":
        return ElasticNet(**p, random_state=GLOBAL_SEED, max_iter=2000)
    if model_key == "svm":
        return (SVC(**p, random_state=GLOBAL_SEED, probability=True)
                if problem_type == "classification"
                else SVR(**p))
    if model_key == "xgboost":
        common = {**p, "random_state": 42, "n_jobs": parallel_jobs, "verbosity": 0}
        return (XGBClassifier(**common, eval_metric="logloss")
                if problem_type == "classification"
                else XGBRegressor(**common))
    if model_key == "lightgbm":
        common = {**p, "random_state": 42, "n_jobs": parallel_jobs, "verbose": -1}
        return (LGBMClassifier(**common) if problem_type == "classification"
                else LGBMRegressor(**common))
    if model_key == "catboost" and _HAS_CATBOOST:
        # Strip any params CatBoost doesn't recognise (LLM hallucinations like
        # n_estimators, subsample with wrong key, etc.) to prevent trial failures.
        _CB_VALID = {
            "iterations", "depth", "learning_rate", "l2_leaf_reg",
            "bagging_temperature", "random_strength", "border_count",
            "grow_policy", "min_data_in_leaf", "subsample",
            "od_type", "od_wait",
        }
        p_clean = {k: v for k, v in p.items() if k in _CB_VALID}
        # Add overfitting detector early stopping — prevents score regression on
        # later iterations (was causing lower scores vs previous runs)
        p_clean.setdefault("od_type", "Iter")
        p_clean.setdefault("od_wait", 30)
        common = {**p_clean, "random_seed": GLOBAL_SEED, "verbose": 0, "thread_count": 1}
        return (CatBoostClassifier(**common) if problem_type == "classification"
                else CatBoostRegressor(**common))
    if model_key == "adaboost" and problem_type == "classification":
        return AdaBoostClassifier(**p, random_state=GLOBAL_SEED)
    if model_key == "mlp":
        p = dict(p)  # copy to avoid mutating shared params dict
        hls = p.get("hidden_layer_sizes", "128")
        # Decode string encoding (e.g. "128-64") back to tuple (128, 64)
        if isinstance(hls, str):
            p["hidden_layer_sizes"] = tuple(int(x) for x in hls.split("-"))
        return (MLPClassifier(**p, random_state=GLOBAL_SEED, max_iter=500,
                              early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)
                if problem_type == "classification"
                else MLPRegressor(**p, random_state=GLOBAL_SEED, max_iter=500,
                                  early_stopping=True, n_iter_no_change=20, validation_fraction=0.1))
    if model_key == "deep_learning":
        cfg = nn_config or {}
        return KerasNNWrapper(
            problem_type=problem_type, n_classes=n_classes, n_features=n_features,
            hidden_activation=cfg.get("hidden_activation", "relu"),
            use_dropout=cfg.get("use_dropout", True),
            use_l2=cfg.get("use_l2", False),
            use_batch_norm=cfg.get("use_batch_norm", False),
            **p,
        )
    raise ValueError(f"Unknown model_key: {model_key}")


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_objective(model_key, hyperparams_spec, problem_type, X_train_t, y_train,
                    n_classes, n_features, nn_config, cv_folds, scoring,
                    model_n_jobs, cv_n_jobs):
    def objective(trial):
        params = {}
        for pname, spec in hyperparams_spec.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, int(spec["low"]), int(spec["high"]))
            elif ptype == "float":
                params[pname] = trial.suggest_float(
                    pname, float(spec["low"]), float(spec["high"]),
                    log=spec.get("log", False))
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])

        params = inject_class_weight(model_key, params, y_train, problem_type)

        try:
            model = _make_model(
                model_key,
                params,
                problem_type,
                n_classes,
                n_features,
                nn_config,
                parallel_jobs=model_n_jobs,
            )
        except TypeError as te:
            # LLM suggested an unsupported param — log it clearly so it's
            # debuggable, then prune the trial instead of silently returning -inf.
            log.warning(f"[{model_key}] _make_model TypeError (bad param from LLM?): {te}")
            raise optuna.exceptions.TrialPruned(f"Bad param: {te}")
        except Exception:
            return float("-inf")

        cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=GLOBAL_SEED)
              if problem_type == "classification"
              else KFold(n_splits=cv_folds, shuffle=True, random_state=GLOBAL_SEED))

        if needs_sample_weight(model_key, y_train, problem_type):
            sw = compute_sample_weight(y_train)
            scores = []
            for tr_idx, val_idx in cv.split(X_train_t, y_train):
                X_tr, X_val = X_train_t[tr_idx], X_train_t[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                sw_tr = sw[tr_idx]
                try:
                    m = _make_model(
                        model_key,
                        params,
                        problem_type,
                        n_classes,
                        n_features,
                        nn_config,
                        parallel_jobs=model_n_jobs,
                    )
                    m.fit(X_tr, y_tr, sample_weight=sw_tr)
                    from sklearn.metrics import f1_score
                    s = float(f1_score(y_val, _predict_safely(m, X_val), average="weighted", zero_division=0))
                    scores.append(s)
                except Exception as e:
                    log.warning(f"Manual CV fold failed for {model_key}: {e}")
                    scores.append(float("-inf"))
            valid = [s for s in scores if np.isfinite(s)]
            return float(np.mean(valid)) if valid else float("-inf")

        # ── cross_val_score with progressive fallbacks ─────────────────────
        # On Windows, joblib multiprocessing can silently kill worker processes
        # for certain models (HistGradientBoosting, etc.), causing all scores
        # to be NaN/-inf with no visible exception. Strategy:
        #   1. Try parallel CV (n_jobs=cv_n_jobs)
        #   2. If all non-finite  → retry serial (n_jobs=1)
        #   3. If still failing   → manual fold loop for full error visibility
        import traceback as _tb

        def _run_cv(n_jobs_override=None):
            _nj = n_jobs_override if n_jobs_override is not None else cv_n_jobs
            with _suppress_lgbm_feature_name_warning():
                return cross_val_score(
                    model, X_train_t, y_train,
                    cv=cv, scoring=scoring, n_jobs=_nj,
                )

        # Attempt 1: parallel
        try:
            scores = _run_cv()
            valid = scores[np.isfinite(scores)]
            if len(valid) > 0:
                return float(np.mean(valid))
            log.warning(
                f"cross_val_score returned all non-finite for {model_key} "
                f"(parallel n_jobs={cv_n_jobs}) — retrying serial"
            )
        except Exception as e:
            log.warning(
                f"cross_val_score (parallel) failed for {model_key}: {e}\n"
                + _tb.format_exc()
            )

        # Attempt 2: serial — avoids joblib multiprocessing issues on Windows
        try:
            scores = _run_cv(n_jobs_override=1)
            valid = scores[np.isfinite(scores)]
            if len(valid) > 0:
                return float(np.mean(valid))
            log.warning(
                f"cross_val_score serial also non-finite for {model_key} "
                f"— falling back to manual fold loop"
            )
        except Exception as e:
            log.warning(
                f"cross_val_score (serial) failed for {model_key}: {e}\n"
                + _tb.format_exc()
            )

        # Attempt 3: manual fold loop — maximum error visibility
        fold_scores = []
        from sklearn.metrics import f1_score as _f1, r2_score as _r2
        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train_t, y_train)):
            X_tr, X_val = X_train_t[tr_idx], X_train_t[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            try:
                _m = _make_model(
                    model_key, params, problem_type,
                    n_classes, n_features, nn_config, parallel_jobs=1,
                )
                _m.fit(X_tr, y_tr)
                if problem_type == "classification":
                    s = float(_f1(y_val, _predict_safely(_m, X_val),
                                  average="weighted", zero_division=0))
                else:
                    s = float(_r2(y_val, _predict_safely(_m, X_val)))
                fold_scores.append(s)
            except Exception as fold_exc:
                log.warning(
                    f"Manual fold {fold_idx} failed for {model_key}: {fold_exc}\n"
                    + _tb.format_exc()
                )
                fold_scores.append(float("-inf"))

        valid_folds = [s for s in fold_scores if np.isfinite(s)]
        if valid_folds:
            return float(np.mean(valid_folds))

        log.error(
            f"All CV attempts failed for {model_key}. Fold scores: {fold_scores}"
        )
        return float("-inf")

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Data hash for Optuna cache key
# ─────────────────────────────────────────────────────────────────────────────

def _data_hash(X_train_t: np.ndarray, y_train: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(X_train_t[:10].tobytes())
    h.update(y_train[:10].tobytes())
    h.update(str(X_train_t.shape).encode())
    return h.hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Agent entry point
# ─────────────────────────────────────────────────────────────────────────────

@agent_error_handler("Model Agent")
def model_agent(state: PipelineState) -> dict:
    log.info("Model agent started", extra={"problem_type": state.get("problem_type"), "retry_count": state.get("retry_count", 0)})
    api_key = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_AGENT_MODEL") or os.getenv("OPENAI_MODEL", cfg("llm.default_model", default="gpt-4o-mini"))
    tid_for_logs = _resolve_training_tid(state)
    _update_training_progress(
        tid_for_logs,
        status="running",
        stage="llm_recommendation",
        percent=24,
        message="Model Agent starting up...",
    )
    _append_training_event(tid_for_logs, "Model Agent entered startup.", stage="llm_recommendation")
    log.info("Model agent restoring runtime", extra={"thread_id": tid_for_logs})

    rt, tid = get_or_restore_runtime(
        state, required=("X_train_t", "X_test_t", "y_train", "y_test")
    )
    tid = _resolve_training_tid(state, tid or tid_for_logs)
    log.info(
        "Model agent runtime restore finished",
        extra={
            "thread_id": tid,
            "runtime_keys": sorted(rt.keys()),
            "has_train": "X_train_t" in rt,
            "has_test": "X_test_t" in rt,
            "has_y_train": "y_train" in rt,
            "has_y_test": "y_test" in rt,
        },
    )
    _append_training_event(
        tid,
        "Model Agent restored runtime objects and is validating required arrays.",
        stage="llm_recommendation",
    )
    missing = [k for k in ("X_train_t", "X_test_t", "y_train", "y_test") if k not in rt]
    if missing:
        log.error("Model agent missing runtime objects", extra={"thread_id": tid, "missing": missing})
        _append_training_event(
            tid,
            f"Model Agent is missing required runtime objects: {', '.join(missing)}.",
            stage="llm_recommendation",
        )
        return {"agent_messages": [
            f"[Model Agent] ❌ Missing runtime objects: {missing}. "
            "Please go back to the Split phase and re-apply the split."
        ]}

    X_train_t     = _as_numpy_matrix(rt["X_train_t"])
    X_test_t      = _as_numpy_matrix(rt["X_test_t"])
    y_train       = _as_numpy_vector(rt["y_train"])
    y_test        = _as_numpy_vector(rt["y_test"])
    rt["X_train_t"] = X_train_t
    rt["X_test_t"] = X_test_t
    rt["y_train"] = y_train
    rt["y_test"] = y_test
    feature_names = rt.get("feature_names", [f"f{i}" for i in range(X_train_t.shape[1])])
    problem_type  = state["problem_type"]

    n_samples  = len(X_train_t)
    n_features = X_train_t.shape[1]
    n_classes  = len(np.unique(y_train)) if problem_type == "classification" else 2
    retry_count = state.get("retry_count", 0)

    # ── Config-driven Optuna settings (v5: reads config.yaml) ─────────────────
    # TPE sampler uses ~25 random warm-up trials before guided search begins.
    # Default 50 ensures ~25 genuinely guided trials per model after warm-up.
    n_trials      = int(state.get("optuna_trials") or os.getenv("OPTUNA_TRIALS",  str(cfg("tuning.n_trials",    default=50))))
    optuna_n_jobs = int(os.getenv("OPTUNA_N_JOBS",  str(cfg("tuning.n_jobs",      default=1))))
    cv_folds      = int(state.get("cv_folds") or os.getenv("CV_FOLDS", str(cfg("cv.n_splits", default=5))))
    max_candidate_models = max(
        3,
        int(state.get("max_candidate_models") or os.getenv("MAX_CANDIDATE_MODELS", str(cfg("tuning.max_candidate_models", default=3)))),
    )
    min_candidate_models = 3
    optuna_timeout_seconds = float(
        os.getenv("OPTUNA_TIMEOUT_SECONDS", str(cfg("tuning.timeout_seconds", default=0)))
    )
    # 0  → no wall-clock cap; every model runs its full n_trials.
    # >0 → total budget (seconds) divided equally across all candidates up-front,
    #      so every model gets a guaranteed slice rather than first-come-first-served.
    tuning_deadline = time.monotonic() + optuna_timeout_seconds if optuna_timeout_seconds > 0 else None
    tuning_model_n_jobs = 1 if optuna_n_jobs != 1 else -1
    cv_n_jobs = 1 if optuna_n_jobs != 1 else -1
    scoring       = "f1_weighted" if problem_type == "classification" else "r2"

    _update_training_progress(
        tid,
        status="running",
        stage="llm_recommendation",
        percent=28,
        message="Choosing candidate model families...",
    )
    _append_training_event(
        tid,
        f"Model Agent started with {n_trials} Optuna trials and {cv_folds}-fold CV.",
        stage="llm_recommendation",
    )
    log.info(
        "Model agent startup configuration prepared",
        extra={
            "thread_id": tid,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "max_candidate_models": max_candidate_models,
            "optuna_timeout_seconds": optuna_timeout_seconds,
            "optuna_n_jobs": optuna_n_jobs,
            "model_name": model_name,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
        },
    )

    # ── NEW v5: Baseline dummy model comparison ────────────────────────────────
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import f1_score as _f1, r2_score as _r2
    dummy_score: float | None = None
    try:
        if problem_type == "classification":
            dummy = DummyClassifier(strategy="most_frequent", random_state=GLOBAL_SEED)
        else:
            dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train_t, y_train)
        y_dummy = dummy.predict(X_test_t)
        dummy_score = float(_f1(y_test, y_dummy, average="weighted", zero_division=0)) \
            if problem_type == "classification" else float(_r2(y_test, y_dummy))
        log.info(f"Dummy baseline score: {dummy_score:.4f}")
    except Exception:
        pass

    # ── Loss function selection ────────────────────────────────────────────────
    loss_info = select_loss_function(problem_type, y_train, {
        "n_samples": n_samples, "n_features": n_features
    })

    # ── Meta-learning warm-start ───────────────────────────────────────────────
    meta_hints: list[str] = []
    try:
        from utils.meta_memory import suggest_models
        meta_hints = suggest_models(state, k=5)
    except Exception:
        pass

    # ── Data hash for Optuna caching ──────────────────────────────────────────
    d_hash = _data_hash(X_train_t, y_train)

    # ── LLM model recommendation ──────────────────────────────────────────────
    class_balance = None
    if problem_type == "classification":
        vals, counts = np.unique(y_train, return_counts=True)
        class_balance = {str(v): round(float(c / len(y_train)), 3) for v, c in zip(vals, counts)}

    split_analysis = state.get("split_analysis") or {}
    user_instructions = state.get("user_model_instructions", "").strip()
    loop_model_strategy = state.get("loop_model_strategy", "").strip()

    # Fix #9: include early drift report in payload so LLM can prefer
    # drift-robust model families when drift is detected pre-training.
    drift_report   = state.get("drift_report") or {}
    drift_severity = drift_report.get("overall_severity", "low")
    drifted_cols   = list(drift_report.get("drifted_features", []))[:5]

    payload = {
        # ── Core dataset dimensions ──────────────────────────────────────────
        "n_samples": n_samples,
        "n_features": n_features,
        "problem_type": problem_type,
        "n_classes": n_classes,
        "class_balance": class_balance,

        # ── Optuna budget (LLM calibrates search space width to this) ────────
        "n_trials": n_trials,
        "cv_folds": cv_folds,

        # ── EDA signals ──────────────────────────────────────────────────────
        "feature_names": list(feature_names)[:50] if feature_names else [],
        "numeric_columns":       (state.get("eda_analysis") or {}).get("numeric_columns", []),
        "categorical_columns":   (state.get("eda_analysis") or {}).get("categorical_columns", []),
        "high_null_columns":     (state.get("eda_analysis") or {}).get("high_null_columns", []),
        "zero_variance_columns": (state.get("eda_analysis") or {}).get("zero_variance_cols", []),
        "high_skew_columns":     (state.get("eda_analysis") or {}).get("high_skew_columns", []),

        # ── Auto dataset insights (from ingest) ───────────────────────────────
        "missing_pct_overall": (state.get("auto_dataset_insights") or {}).get("missing_pct", 0.0),
        "type_issues":         (state.get("auto_dataset_insights") or {}).get("type_issues", []),
        "dataset_warnings":    (state.get("auto_dataset_insights") or {}).get("warnings", []),

        # ── Imbalance analysis (classification only) ─────────────────────────
        "imbalance_ratio":    (state.get("imbalance_analysis") or {}).get("imbalance_ratio", 1.0),
        "imbalance_severity": (state.get("imbalance_analysis") or {}).get("severity", "none"),
        "smote_recommended":  (state.get("imbalance_recommendations") or {}).get("smote_recommended", False),

        # ── Feature engineering signals ──────────────────────────────────────
        "n_engineered_features": (state.get("feature_analysis") or {}).get("n_computable", 0),
        "n_dropped_features":    (state.get("feature_analysis") or {}).get("n_dropped", 0),
        "shap_top_features": [
            f["feature"] for f in (state.get("shap_importance") or [])[:10]
            if isinstance(f, dict) and "feature" in f
        ],

        # ── Split analysis ────────────────────────────────────────────────────
        "split_strategy": state.get("split_strategy", "random"),
        "test_size":       state.get("test_size", 0.2),
        "split_analysis":  split_analysis,

        # ── Drift signals ─────────────────────────────────────────────────────
        "drift_severity":   drift_severity,
        "drifted_columns":  drifted_cols,
        "drift_note": (
            f"WARNING: Data drift detected (severity={drift_severity}) in columns "
            f"{drifted_cols}. Prefer tree-based or robust models "
            "(e.g. hist_gradient_boosting, lightgbm, xgboost) that handle distributional "
            "shift better than linear or SVM models."
            if drift_severity in ("high", "medium") else ""
        ),

        # ── Meta-learning & history ───────────────────────────────────────────
        "meta_memory_hints":  meta_hints[:5],
        "loss_recommendation": loss_info,
        "previous_best":      state.get("best_model_key"),
        "retry_count":        retry_count,
        "has_catboost":       _HAS_CATBOOST,

        # ── User instructions ─────────────────────────────────────────────────
        "user_instructions": user_instructions or loop_model_strategy or None,
    }
    log.info(
        "Model agent prepared LLM payload",
        extra={
            "thread_id": tid,
            "payload_keys": sorted(payload.keys()),
            "meta_hint_count": len(meta_hints[:5]),
        },
    )

    # ── NEW v3: HITL forced candidates override ───────────────────────────────
    # If the user approved model selection in the HITL page, use their choices.
    # _forced_model_candidates is a list of {model_key, rationale, hyperparams?}
    forced_candidates = state.get("_forced_model_candidates") or []
    if forced_candidates and state.get("hitl_model_selection_approved"):
        # User made explicit selections — merge hyperparams spec from LLM result
        # with the forced list, falling back to default search spaces.
        log.info(
            "Model agent using user-forced candidate list",
            extra={"forced": [c.get("model_key") for c in forced_candidates]},
        )
        # Enrich forced candidates with dataset-aware hyperparams from LLM.
        # We issue a second LLM call ONLY when needed to produce hyperparams for
        # user-selected models. FIX 11: if the HITL page already stored LLM-generated
        # hyperparams in _forced_model_candidates, skip the redundant LLM call entirely.
        forced_keys = [c["model_key"] for c in forced_candidates]
        needs_llm_enrichment = any(not fc.get("hyperparams") for fc in forced_candidates)
        llm_hyp_map = {}
        if needs_llm_enrichment:
            forced_user_prompt = (
                "IMPORTANT: The user has explicitly selected ONLY these models: "
                + json.dumps(forced_keys) + ".\n"
                "You MUST return candidates for ALL of these models and NO others.\n"
                "Do NOT substitute, add, or remove any model from this list.\n"
                "Produce the best possible Optuna hyperparameter search spaces for each, "
                "calibrated to the dataset payload below.\n\n"
                + json.dumps(payload, indent=2, default=str)
            )
            try:
                _llm_enriched, _ = call_llm_json(
                    api_key=api_key or "",
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=forced_user_prompt,
                    temperature=0.1,
                    max_tokens=3500,
                    max_attempts=int(cfg("llm.backoff_max_attempts", default=3)),
                    base_delay=float(cfg("llm.backoff_base_delay", default=1.0)),
                    request_timeout=float(os.getenv("MODEL_AGENT_LLM_TIMEOUT_SECONDS", "60")),
                )
                if _llm_enriched:
                    for c in (_llm_enriched.get("candidates") or []):
                        mk = c.get("model_key", "")
                        if mk in forced_keys:
                            llm_hyp_map[mk] = c
            except Exception:
                pass
        else:
            log.info("Skipping LLM enrichment — all forced candidates already have hyperparams.")

        enriched_forced = []
        for fc in forced_candidates:
            mk = fc.get("model_key", "")
            llm_entry = llm_hyp_map.get(mk)
            if llm_entry and llm_entry.get("hyperparams"):
                # LLM returned good hyperparams for this model — use them.
                # Preserve the original forced candidate's model_key/rationale and
                # layer in LLM hyperparams + nn_config on top.
                enriched_forced.append({
                    **fc,
                    "hyperparams": llm_entry["hyperparams"],
                    "rationale":   llm_entry.get("rationale") or fc.get("rationale", ""),
                    **(({"nn_config": llm_entry["nn_config"]}) if llm_entry.get("nn_config") else {}),
                })
                log.info(f"LLM enriched hyperparams for forced candidate: {mk}")
            else:
                # LLM did not return hyperparams for this model — fall back to defaults.
                fallback_hp = next(
                    (t for t in _fallback_candidate_templates(problem_type) if t["model_key"] == mk),
                    None,
                )
                enriched = {**fc, **(fallback_hp or {})}
                # Preserve original model_key in case fallback clobbered it
                enriched["model_key"] = mk
                enriched_forced.append(enriched)
                log.warning(f"Using fallback hyperparams for forced candidate: {mk} (LLM did not return it)")
        candidates = enriched_forced
        # Build a meaningful reasoning message from per-model rationales
        rationale_lines = []
        for c in candidates:
            mk  = c.get("model_key", "?")
            rat = c.get("rationale", "").strip()
            if rat:
                rationale_lines.append(f"- **{mk}**: {rat}")
            else:
                rationale_lines.append(f"- **{mk}**: selected by user.")
        hitl_reasoning = (
            f"User selected {len(candidates)} model(s) for tuning. "
            f"Hyperparameters enriched by LLM where available.\n\n"
            + "\n".join(rationale_lines)
        )
        llm_result = {"recommendation": hitl_reasoning}
        _append_training_event(tid, f"Using {len(candidates)} user-selected model(s): {[c['model_key'] for c in candidates]}", stage="llm_recommendation")
    else:
        _update_training_progress(
            tid, status="running", stage="llm_recommendation", percent=32,
            message="Choosing model families for tuning...",
        )
        _append_training_event(tid, f"Asking {model_name} to recommend model families.", stage="llm_recommendation")
        log.info("Model agent calling LLM for candidates", extra={"thread_id": tid, "model_name": model_name})
        llm_result, llm_err = call_llm_json(
            api_key=api_key or "",
            model_name=model_name,
            system_prompt=SYSTEM_PROMPT,
            user_content=json.dumps(payload, indent=2, default=str),
            temperature=0.1,
            max_tokens=3500,
            max_attempts=int(cfg("llm.backoff_max_attempts", default=3)),
            base_delay=float(cfg("llm.backoff_base_delay", default=1.0)),
            request_timeout=float(os.getenv("MODEL_AGENT_LLM_TIMEOUT_SECONDS", "60")),
        )
        try:
            if llm_result is None:
                raise llm_err or RuntimeError("LLM returned no result.")
            log.info(
                "Model agent received LLM candidate response",
                extra={"thread_id": tid, "candidate_count": len(llm_result.get("candidates", []) or [])},
            )
            candidates = llm_result.get("candidates", [])
            _append_training_event(
                tid,
                f"Selected {len(candidates)} candidate model families for tuning.",
                stage="llm_recommendation",
            )
        except Exception as exc:
            log.warning("Model agent LLM candidate selection failed", extra={"thread_id": tid, "error": str(exc)})
            candidates = _fallback_candidate_templates(problem_type)[:min_candidate_models]
            for c in candidates:
                c["rationale"] = f"LLM fallback: {c.get('rationale', '')} ({exc})"
            llm_result = {"recommendation": f"LLM error — using {len(candidates)}-model fallback."}
            _append_training_event(tid, f"LLM recommendation failed, fallback used: {exc}", stage="llm_recommendation")

    # ── Optuna tuning (parallel n_jobs=-1 where supported) ────────────────────
    if not _DEEP_LEARNING_ENABLED:
        candidates = [c for c in candidates if c.get("model_key") != "deep_learning"]
        if not candidates:
            candidates = _fallback_candidate_templates(problem_type)[:min_candidate_models]
    original_candidate_count = len({str(c.get("model_key") or "").strip() for c in candidates if str(c.get("model_key") or "").strip()})
    # When user explicitly selected models via HITL, respect their choices — don't pad to min
    if not (forced_candidates and state.get("hitl_model_selection_approved")):
        candidates = _ensure_minimum_candidates(candidates, problem_type, min_candidate_models)
    candidates = _dedupe_and_limit_candidates(candidates, max(max_candidate_models, len(candidates)))
    if len(candidates) > original_candidate_count:
        _append_training_event(
            tid,
            f"Added fallback model families so at least {min_candidate_models} models are tuned.",
            stage="tuning",
        )
    elif len(candidates) < original_candidate_count:
        _append_training_event(
            tid,
            f"Limited tuning to the top {len(candidates)} model families for faster iteration.",
            stage="tuning",
        )
    log.info(
        "Model agent finalized candidate list",
        extra={
            "thread_id": tid,
            "candidate_models": [c.get("model_key", "") for c in candidates],
            "deep_learning_enabled": _DEEP_LEARNING_ENABLED,
            "max_candidate_models": max_candidate_models,
        },
    )
    _append_training_event(
        tid,
        "Candidate model families finalized. Starting Optuna tuning next.",
        stage="tuning",
    )

    total_models = len(candidates)
    total_trials = max(total_models * n_trials, 1)
    log.info(
        "Model agent entering Optuna tuning",
        extra={"thread_id": tid, "total_models": total_models, "total_trials": total_trials},
    )
    _update_training_progress(
        tid,
        status="running",
        stage="tuning",
        total_models=total_models,
        total_trials=total_trials,
        completed_trials=0,
        percent=_tuning_percent(0, total_trials),
        message="Preparing model tuning...",
    )
    tuning_results: list[dict] = []
    best_model = None
    best_score = float("-inf")
    best_key   = ""
    best_params: dict = {}
    best_nn_cfg: dict = {}
    all_trained_models: dict = {}

    # ── Pre-broadcast all model slots as "queued" so the UI can render all
    # model cards immediately (instead of showing nothing until each model starts).
    _update_training_progress(
        tid,
        stage="tuning",
        all_model_slots=[c.get("model_key", "") for c in candidates],
        total_models=total_models,
        completed_trials=0,
        total_trials=total_trials,
        percent=_tuning_percent(0, total_trials),
        message="Preparing to tune all models...",
    )

    for cand_idx, cand in enumerate(candidates, start=1):
        model_key   = cand.get("model_key", "")
        hp_spec     = cand.get("hyperparams", {})
        nn_config   = cand.get("nn_config") or {}

        if model_key == "catboost" and not _HAS_CATBOOST:
            continue

        # Check Optuna cache
        cache_key = f"{model_key}_{d_hash}_{retry_count}"
        # Fix #7: _STUDY_CACHE is a module-level dict that is NOT shared across
        # worker processes when OPTUNA_N_JOBS != 1 (multiprocessing).
        # Only use the in-process cache when running single-threaded.
        _cache_enabled = (optuna_n_jobs == 1)
        if _cache_enabled and cache_key in _STUDY_CACHE:
            cached = _STUDY_CACHE[cache_key]
            tuning_results.append({**cached, "model_key": model_key, "_cached": True})
            if cached["best_score"] > best_score:
                best_score  = cached["best_score"]
                best_key    = model_key
                best_params = cached["best_params"]
                best_nn_cfg = nn_config
            continue

        objective = _make_objective(
            model_key,
            hp_spec,
            problem_type,
            X_train_t,
            y_train,
            n_classes,
            n_features,
            nn_config,
            cv_folds,
            scoring,
            tuning_model_n_jobs,
            cv_n_jobs,
        )
        trial_offset = (cand_idx - 1) * n_trials
        _update_training_progress(
            tid,
            stage="tuning",
            current_model=model_key,
            current_model_index=cand_idx,
            total_models=total_models,
            completed_trials=trial_offset,
            total_trials=total_trials,
            percent=_tuning_percent(trial_offset, total_trials),
            message=f"Tuning {model_key.replace('_', ' ')} ({cand_idx}/{total_models})...",
        )
        _append_training_event(
            tid,
            f"Started tuning {model_key.replace('_', ' ')} ({cand_idx}/{total_models}).",
            stage="tuning",
        )

        try:
            # ── Per-model timeout: divide the total budget equally up-front ──────
            # This guarantees every model gets a fair slice instead of first-come-
            # first-served where early models can exhaust the entire budget.
            model_timeout_seconds = None
            if tuning_deadline is not None:
                remaining_tuning_seconds = max(0.0, tuning_deadline - time.monotonic())
                if remaining_tuning_seconds <= 0:
                    _append_training_event(
                        tid,
                        "Stopped additional tuning because the overall time budget was exhausted.",
                        stage="tuning",
                    )
                    break
                # Equal upfront slice — NOT leftover / remaining_models
                model_timeout_seconds = optuna_timeout_seconds / max(total_models, 1)

            # Use TPE sampler. MedianPruner only kicks in after enough warmup
            # to avoid pruning slow-but-valid models like CatBoost.
            sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED, n_startup_trials=max(5, n_trials // 4))
            # Disable pruner for slow models (catboost, mlp) — they need full trials
            if model_key in ("catboost", "mlp", "svm", "deep_learning"):
                pruner = optuna.pruners.NopPruner()
            else:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=max(5, n_trials // 4),
                    n_warmup_steps=5,
                    interval_steps=2,
                )
            study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            # ── Fix closure bug: capture loop variables by value, not reference ──
            # Defining the callback inside the loop means model_key / cand_idx /
            # trial_offset are captured by reference.  When the callback fires
            # (possibly after the loop has advanced), they hold the wrong values.
            # Bind them as default-argument values to snapshot them at definition time.
            def _make_progress_callback(_model_key, _cand_idx, _trial_offset):
                def _cb(study_obj, _trial):
                    completed = min(_trial_offset + len(study_obj.trials), total_trials)
                    best_val = study_obj.best_value if study_obj.best_trial is not None else None
                    pct = _tuning_percent(completed, total_trials)
                    best_txt = f" best={best_val:.4f}" if isinstance(best_val, (int, float)) else ""
                    _update_training_progress(
                        tid,
                        stage="tuning",
                        current_model=_model_key,
                        current_model_index=_cand_idx,
                        total_models=total_models,
                        completed_trials=completed,
                        total_trials=total_trials,
                        percent=pct,
                        best_score=best_val,
                        message=(
                            f"Tuning {_model_key.replace('_', ' ')} "
                            f"trial {len(study_obj.trials)}/{n_trials}{best_txt}"
                        ),
                    )
                return _cb

            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=optuna_n_jobs,
                timeout=model_timeout_seconds,
                show_progress_bar=False,
                callbacks=[_make_progress_callback(model_key, cand_idx, trial_offset)],
            )

            best_trial_score  = study.best_value
            best_trial_params = study.best_params
            completed_so_far  = trial_offset + len(study.trials)

            # Treat -inf as a failed run: all trials returned -inf means the
            # model crashed on every attempt (e.g. class_weight incompatibility).
            _score_valid = best_trial_score != float("-inf")
            _stored_score = round(best_trial_score, 4) if _score_valid else None
            _stored_error = None if _score_valid else "all trials returned -inf"

            run_result = {
                "model_key":          model_key,
                "best_score":         _stored_score,
                "best_params":        best_trial_params if _score_valid else {},
                "n_trials_completed": len(study.trials),
                "rationale":          cand.get("rationale", ""),
            }
            if _stored_error:
                run_result["error"] = _stored_error
            tuning_results.append(run_result)
            if _cache_enabled and _score_valid:
                _cache_put(cache_key, run_result)

            # ── Fix: send explicit "finished" progress update for this model ──
            # Without this, the UI stays on the last mid-trial callback message
            # (e.g. "Tuning xgboost trial 20/20 best=0.8292") and the model card
            # never transitions out of the "tuning..." state.
            if _score_valid:
                _update_training_progress(
                    tid,
                    stage="tuning",
                    current_model=model_key,
                    current_model_index=cand_idx,
                    total_models=total_models,
                    completed_trials=min(completed_so_far, total_trials),
                    total_trials=total_trials,
                    percent=_tuning_percent(min(completed_so_far, total_trials), total_trials),
                    best_score=best_trial_score,
                    message=f"Finished {model_key.replace('_', ' ')} ✓ CV={best_trial_score:.4f}",
                    model_finished=model_key,
                )
                _append_training_event(
                    tid,
                    f"Finished {model_key.replace('_', ' ')} with best CV score {best_trial_score:.4f}.",
                    stage="tuning",
                )
            else:
                _update_training_progress(
                    tid,
                    stage="tuning",
                    current_model=model_key,
                    current_model_index=cand_idx,
                    total_models=total_models,
                    completed_trials=min(completed_so_far, total_trials),
                    total_trials=total_trials,
                    percent=_tuning_percent(min(completed_so_far, total_trials), total_trials),
                    best_score=None,
                    message=f"{model_key.replace('_', ' ')} failed — all trials returned -inf",
                    model_finished=model_key,
                )
                _append_training_event(
                    tid,
                    f"{model_key.replace('_', ' ')} failed: all Optuna trials returned -inf.",
                    stage="tuning",
                )
            if best_trial_score > best_score:
                best_score  = best_trial_score
                best_key    = model_key
                best_params = best_trial_params
                best_nn_cfg = nn_config

        except Exception as exc:
            _update_training_progress(
                tid,
                stage="tuning",
                current_model=model_key,
                current_model_index=cand_idx,
                total_models=total_models,
                completed_trials=min(trial_offset, total_trials),
                total_trials=total_trials,
                percent=_tuning_percent(trial_offset, total_trials),
                message=f"{model_key.replace('_', ' ')} failed: {exc}",
            )
            _append_training_event(
                tid,
                f"{model_key.replace('_', ' ')} failed during tuning: {exc}",
                stage="tuning",
            )
            tuning_results.append({
                "model_key": model_key, "best_score": None,
                "error": str(exc), "n_trials_completed": 0,
                "best_params": {}, "rationale": cand.get("rationale", ""),
            })

    # ── Final fit of best model on full training set ───────────────────────────
    if best_key:
        try:
            _update_training_progress(
                tid,
                stage="final_fit",
                current_model=best_key,
                total_models=total_models,
                completed_trials=total_trials,
                total_trials=total_trials,
                percent=97,
                message=f"Fitting final {best_key.replace('_', ' ')} model...",
            )
            _append_training_event(
                tid,
                f"Training final {best_key.replace('_', ' ')} model on full training data.",
                stage="final_fit",
            )
            final_params = inject_class_weight(best_key, best_params, y_train, problem_type)
            best_model_obj = _make_model(best_key, final_params, problem_type,
                                         n_classes, n_features, best_nn_cfg or None)
            if needs_sample_weight(best_key, y_train, problem_type):
                sw = compute_sample_weight(y_train)
                best_model_obj.fit(X_train_t, y_train, sample_weight=sw)
            else:
                best_model_obj.fit(X_train_t, y_train)

            best_model = best_model_obj
            all_trained_models[best_key] = best_model_obj

            # Fit ALL runner-up models for ensemble (all successfully tuned models)
            sorted_results = sorted(
                [r for r in tuning_results if r.get("best_score") is not None],
                key=lambda r: r["best_score"], reverse=True
            )
            # Train every runner-up, not just max_candidate_models - 1.
            # This ensures all user-selected models are available for ensemble.
            for r in sorted_results[1:]:
                mk = r["model_key"]
                if mk in all_trained_models:
                    continue
                try:
                    mp = inject_class_weight(mk, r["best_params"], y_train, problem_type)
                    m  = _make_model(mk, mp, problem_type, n_classes, n_features,
                                     parallel_jobs=tuning_model_n_jobs)
                    if needs_sample_weight(mk, y_train, problem_type):
                        sw = compute_sample_weight(y_train)
                        m.fit(X_train_t, y_train, sample_weight=sw)
                    else:
                        m.fit(X_train_t, y_train)
                    all_trained_models[mk] = m
                    _append_training_event(
                        tid,
                        f"Runner-up model {mk} fitted successfully for ensemble.",
                        stage="final_fit",
                    )
                except Exception as runner_exc:
                    log.warning(
                        "Runner-up model fit failed",
                        extra={"thread_id": tid, "model_key": mk, "error": str(runner_exc)},
                    )
                    _append_training_event(
                        tid,
                        f"Runner-up model {mk} fit failed (skipping): {runner_exc}",
                        stage="final_fit",
                    )
            # Warn if we ended up with fewer than min_candidate_models trained models
            if len(all_trained_models) < min_candidate_models:
                _append_training_event(
                    tid,
                    f"Warning: only {len(all_trained_models)} model(s) trained successfully "
                    f"(target was {min_candidate_models}). Ensemble may be limited.",
                    stage="final_fit",
                )

        except Exception as exc:
            _update_training_progress(
                tid,
                status="error",
                stage="final_fit",
                percent=100,
                message=f"Final fit failed: {exc}",
            )
            _append_training_event(tid, f"Final fit failed: {exc}", stage="final_fit")
            return {"agent_messages": [f"[Model Agent] Final fit failed: {exc}"]}

    # ── Compute baseline ───────────────────────────────────────────────────────
    baseline = compute_baseline_score(y_train, y_test, problem_type)

    # ── Store in runtime ──────────────────────────────────────────────────────
    bucket = _get_runtime_bucket(tid)
    if bucket is not None:
        try:
            if hasattr(bucket, "update"):
                bucket.update(tid, {
                    "best_model": best_model,
                    "X_train_t": X_train_t,
                    "X_test_t": X_test_t,
                    "y_train": y_train,
                    "y_test": y_test,
                    "feature_names": feature_names,
                    "all_trained_models": all_trained_models,
                })
        except Exception:
            pass

    # ── Build model_analysis ─────────────────────────────────────────────────
    model_analysis = {
        "recommendation":       llm_result.get("recommendation", ""),
        "n_candidates":         len(candidates),
        "n_tuned_successfully": sum(1 for r in tuning_results if r.get("best_score") is not None),
        "best_model_key":       best_key,
        "best_cv_score":        round(best_score, 4) if best_score > float("-inf") else None,
        "best_params":          best_params,
        "n_train_samples":      n_samples,
        "n_features":           n_features,
        "cv_folds":             cv_folds,
        "optuna_trials":        n_trials,
        "problem_type":         problem_type,
        "retry_count":          retry_count,
        "meta_memory_hints":    meta_hints,
        "loss_function_chosen": loss_info,
        "user_instructions":    user_instructions,
        "candidates_detail": [
            {
                "model_key":          r["model_key"],
                "rationale":          r.get("rationale", ""),
                "best_score":         r.get("best_score"),
                "best_params":        r.get("best_params", {}),
                "n_trials_completed": r.get("n_trials_completed", 0),
                "is_best":            r["model_key"] == best_key,
                "cached":             r.get("_cached", False),
            }
            for r in tuning_results
        ],
    }
    _update_training_progress(
        tid,
        status="awaiting_review",
        stage="complete",
        current_model=best_key,
        total_models=total_models,
        completed_trials=total_trials,
        total_trials=total_trials,
        percent=100,
        best_score=best_score if best_score > float("-inf") else None,
        message=f"Training complete. Best model: {best_key or 'none'}",
    )
    _append_training_event(
        tid,
        f"Training complete. Best model: {best_key or 'none'}"
        + (f" with CV score {best_score:.4f}." if best_score > float("-inf") else "."),
        stage="complete",
    )

    # rt is the flat dict of arrays returned by get_or_restore_runtime;
    # pass it directly rather than the fragile rt.get(tid, rt) pattern.
    state_arr_updates = store_arrays_to_state(rt)

    # v5: inject dummy_score into model_analysis for UI comparison
    model_analysis["dummy_baseline_score"] = dummy_score

    return {
        "model_candidates":    [c["model_key"] for c in candidates],
        "model_recommendation": llm_result.get("recommendation", ""),
        "tuning_results":      tuning_results,
        "best_model_key":      best_key,
        "best_params":         best_params,
        "best_cv_score":       round(best_score, 4) if best_score > float("-inf") else None,
        "best_nn_config":      best_nn_cfg,
        "model_analysis":      model_analysis,
        "_baseline_score":     baseline,
        "_meta_memory_hints":  meta_hints,
        "_best_model_b64":     obj_to_b64(best_model) if best_model is not None else None,
        "optuna_trials":       n_trials,
        "cv_folds":            cv_folds,
        **state_arr_updates,
        "agent_messages": [
            f"[Model Agent] Best: {best_key} (CV={best_score:.4f}). "
            f"Baseline: {baseline:.4f}. "
            + (f"Dummy: {dummy_score:.4f}. " if dummy_score is not None else "")
            + f"Meta-hints: {meta_hints[:3]}. "
            f"Loss: {loss_info['loss_function']}."
        ],
    }
