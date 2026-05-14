"""
utils/runtime.py — Shared runtime-object helpers.

Improvement #4: extracted _restore_arrays_from_state() which was duplicated
verbatim in both model_agent.py and eval_agent.py. Single source of truth.

Additional improvements:
- store_arrays_to_state(): symmetric counterpart to restore; split/model agents
  can call this to persist arrays back to graph state for page-refresh resilience.
- get_or_restore_runtime(): consolidates the repeated two-phase "canonical bucket
  → scan all buckets → restore from b64" lookup pattern that was duplicated across
  model_agent and eval_agent into one reusable function.
- needs_sample_weight / inject_class_weight / compute_sample_weight / is_imbalanced:
  centralises the imbalanced-class detection logic that was duplicated (and slightly
  inconsistent) between make_objective() and the final-fit block in model_agent.
  The original final-fit check omitted extra_trees and hist_gradient_boosting from
  sample-weight handling — this version covers all models correctly.
"""
from __future__ import annotations
import base64
import io
from typing import Any

import numpy as np
from ui.runtime_context import get_runtime_store
from utils.serialization import b64_to_obj


# ─────────────────────────────────────────────────────────────────────────────
# Array ↔ graph-state persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

_ARR_KEY_MAP = [
    ("X_train_t", "_X_train_t_b64"),
    ("X_test_t",  "_X_test_t_b64"),
    ("y_train",   "_y_train_b64"),
    ("y_test",    "_y_test_b64"),
    # Calibration split (Fix: avoids calibrating on the test set)
    ("X_cal_t",   "_X_cal_t_b64"),
    ("y_cal",     "_y_cal_b64"),
]


def restore_arrays_from_state(rt: dict, state: dict) -> dict:
    """
    Restore numpy arrays from b64-encoded graph state when the runtime
    store is missing them (e.g. after a server restart or worker recycle).

    Parameters
    ----------
    rt    : current runtime bucket for this thread_id
    state : full PipelineState dict from the LangGraph checkpointer

    Returns a new dict with any recoverable arrays added.
    """
    def _load(key: str):
        b64 = state.get(key)
        if b64 and isinstance(b64, str):
            try:
                return np.load(
                    io.BytesIO(base64.b64decode(b64)), allow_pickle=False
                )
            except Exception:
                return None
        return None

    restored = dict(rt)

    for arr_key, state_key in _ARR_KEY_MAP:
        if arr_key not in restored:
            arr = _load(state_key)
            if arr is not None:
                restored[arr_key] = arr

    if "feature_names" not in restored and state.get("_feature_names"):
        restored["feature_names"] = state["_feature_names"]

    if "best_model" not in restored and state.get("_best_model_b64"):
        try:
            restored["best_model"] = b64_to_obj(state["_best_model_b64"])
        except Exception:
            pass

    if "preprocessor" not in restored and state.get("_preprocessor_b64"):
        try:
            restored["preprocessor"] = b64_to_obj(state["_preprocessor_b64"])
        except Exception:
            pass

    if "label_encoder" not in restored and state.get("_label_encoder_b64"):
        try:
            restored["label_encoder"] = b64_to_obj(state["_label_encoder_b64"])
        except Exception:
            pass

    return restored


def store_arrays_to_state(rt: dict, existing_state: dict | None = None) -> dict:
    """
    Symmetric counterpart to restore_arrays_from_state.

    Serialise numpy arrays present in *rt* to base64 NPY strings suitable for
    storage in PipelineState (JSON-serialisable, survives SQLite checkpoint).

    Parameters
    ----------
    rt             : runtime bucket dict containing numpy arrays
    existing_state : optional existing state dict to merge into (non-destructive)

    Returns a dict of {state_key: b64_string} ready to merge into the LangGraph
    return dict from split_agent or any agent that writes arrays.

    Example usage in split_agent after building rt
    -----------------------------------------------
        state_updates = store_arrays_to_state(rt)
        return {**other_updates, **state_updates}
    """
    updates: dict[str, Any] = dict(existing_state or {})

    def _dump(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        np.save(buf, arr)
        return base64.b64encode(buf.getvalue()).decode()

    for arr_key, state_key in _ARR_KEY_MAP:
        arr = rt.get(arr_key)
        if arr is not None and isinstance(arr, np.ndarray):
            updates[state_key] = _dump(arr)

    if "feature_names" in rt:
        updates["_feature_names"] = rt["feature_names"]

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Consolidated runtime-object lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_or_restore_runtime(
    state: dict,
    required: tuple[str, ...] = ("best_model", "X_test_t", "y_test"),
) -> tuple[dict, str]:
    """
    Robustly locate all required runtime objects using a three-phase strategy.

    This consolidates the duplicated lookup patterns from model_agent and
    eval_agent into a single reusable function.

    Phase 1 — Use runtime_objects[_thread_id] (canonical bucket).
    Phase 2 — Restore missing numpy arrays from b64 in graph state.
    Phase 3 — If still incomplete, scan ALL buckets and merge: take each needed
               key from whichever bucket has it first.

    Parameters
    ----------
    state    : full PipelineState dict
    required : tuple of keys that must be present for the caller to proceed

    Returns
    -------
    (merged_rt, tid_used)
        merged_rt  — dict with all recoverable objects
        tid_used   — the _thread_id string used for the canonical bucket
    """
    tid = state.get("_thread_id", "")
    runtime_store = get_runtime_store()
    if runtime_store is None:
        return restore_arrays_from_state({}, state), tid
    runtime_objects = runtime_store._raw

    # Phase 1: canonical bucket
    rt = dict(runtime_objects.get(tid, {}))

    # Phase 2: restore arrays from graph-state b64
    rt = restore_arrays_from_state(rt, state)

    # Phase 3: scan all buckets to fill any remaining gaps
    missing = [k for k in required if k not in rt]
    if missing:
        merged = dict(rt)
        for _tid, _rt in runtime_objects.items():
            if not missing:
                break
            for need in list(missing):
                if need in _rt and need not in merged:
                    merged[need] = _rt[need]
            if "feature_names" not in merged and "feature_names" in _rt:
                merged["feature_names"] = _rt["feature_names"]
            missing = [k for k in required if k not in merged]
        rt = merged

    return rt, tid


# ─────────────────────────────────────────────────────────────────────────────
# Imbalanced-class helpers  (single source of truth — replaces duplicated code
# in model_agent's make_objective and final-fit block)
# ─────────────────────────────────────────────────────────────────────────────

# Models that support class_weight='balanced' natively.
_CLASS_WEIGHT_MODELS = frozenset({
    "random_forest", "extra_trees", "logistic_regression", "svm", "lightgbm",
    # sklearn 1.5+: HistGradientBoosting supports class_weight natively
    "hist_gradient_boosting",
})

# Models requiring per-sample weights passed at fit() time (no class_weight param).
_SAMPLE_WEIGHT_MODELS = frozenset({
    "gradient_boosting",  # GradientBoosting has no class_weight; uses sample_weight
})

_IMBALANCE_THRESHOLD = 0.20   # minority class ≤ 20% of total → imbalanced


def is_imbalanced(y_train) -> bool:
    """Return True if the minority class ratio is below the threshold."""
    y_arr = np.asarray(y_train)
    _, counts = np.unique(y_arr, return_counts=True)
    if len(counts) < 2:
        return False
    return (counts.min() / counts.sum()) <= _IMBALANCE_THRESHOLD


def inject_class_weight(model_key: str, params: dict, y_train, problem_type: str) -> dict:
    """
    Inject class_weight='balanced' or scale_pos_weight into *params* for models
    that support it natively.  Does nothing for regression or balanced datasets.

    Does NOT handle sample_weight — that requires a manual CV loop.
    See needs_sample_weight / compute_sample_weight below.
    """
    if problem_type != "classification" or not is_imbalanced(y_train):
        return params

    if any(k in model_key for k in _CLASS_WEIGHT_MODELS):
        return {**params, "class_weight": "balanced"}

    # XGBoost binary classification: use scale_pos_weight
    if "xgboost" in model_key:
        y_arr = np.asarray(y_train)
        vals, counts = np.unique(y_arr, return_counts=True)
        if len(vals) == 2:
            neg, pos = int(counts[0]), int(counts[1])
            return {**params, "scale_pos_weight": float(neg / max(pos, 1))}
        # XGBoost multiclass — fall through to sample_weight path

    return params


def needs_sample_weight(model_key: str, y_train, problem_type: str) -> bool:
    """
    Return True when this model+dataset combination requires per-sample weights
    passed at fit() time (rather than a class_weight= constructor param).

    Covers:
    - GradientBoosting and HistGradientBoosting (no class_weight support)
    - XGBoost multiclass (scale_pos_weight only works for binary)

    Bug fix vs original: the original final-fit check in model_agent missed
    extra_trees and hist_gradient_boosting because it only checked for a
    hard-coded short list.  This version derives from the same frozensets as
    inject_class_weight, so coverage stays in sync automatically.
    """
    if problem_type != "classification" or not is_imbalanced(y_train):
        return False

    if any(k in model_key for k in _SAMPLE_WEIGHT_MODELS):
        return True

    if "xgboost" in model_key:
        y_arr = np.asarray(y_train)
        n_classes = len(np.unique(y_arr))
        return n_classes > 2   # multiclass only; binary uses scale_pos_weight

    return False


def compute_sample_weight(y_train) -> np.ndarray:
    """
    Compute per-sample weights inversely proportional to class frequency.
    Equivalent to what class_weight='balanced' does internally, but as an
    explicit array for models that don't support that constructor parameter.
    """
    from sklearn.utils.class_weight import compute_sample_weight as _csw
    return _csw("balanced", np.asarray(y_train))
