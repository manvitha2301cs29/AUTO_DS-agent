"""
utils/experiment_tracker.py — MLflow experiment tracking + model versioning (v5)

Wraps MLflow so every pipeline run is tracked with:
  - Parameters: model key, hyperparams, CV config, feature count
  - Metrics:    all eval_metrics (f1, roc_auc, rmse, r2, …)
  - Artifacts:  serialised model, SHAP plot, confusion matrix
  - Tags:       problem_type, dataset name, run verdict

Falls back gracefully when MLflow is not installed.

Usage:
    from utils.experiment_tracker import tracker

    with tracker.start_run(run_name="xgboost_run_1") as run_id:
        tracker.log_params({"model": "xgboost", "n_estimators": 100})
        tracker.log_metrics({"f1_weighted": 0.88, "roc_auc": 0.92})
        tracker.log_artifact_bytes(model_bytes, "model.pkl")
        tracker.log_tags({"problem_type": "classification"})

    # Or use the convenience wrapper used by eval_agent:
    tracker.log_pipeline_run(state)
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
from contextlib import contextmanager
from typing import Any

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


class ExperimentTracker:
    """
    Thin MLflow wrapper with graceful no-op fallback.
    """

    def __init__(self):
        self._enabled = cfg("mlflow.enabled", default=False) and _HAS_MLFLOW
        self._tracking_uri = cfg("mlflow.tracking_uri", default="mlruns")
        self._experiment = cfg("mlflow.experiment_name", default="automl_v5")
        self._active_run_id: str | None = None

        if self._enabled:
            try:
                mlflow.set_tracking_uri(self._tracking_uri)
                mlflow.set_experiment(self._experiment)
                log.info("MLflow tracking enabled", extra={
                    "uri": self._tracking_uri,
                    "experiment": self._experiment,
                })
            except Exception as e:
                log.warning(f"MLflow init failed — tracking disabled: {e}")
                self._enabled = False

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Context manager that starts/ends an MLflow run."""
        if not self._enabled:
            yield None
            return
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            self._active_run_id = run.info.run_id
            log.info("MLflow run started", extra={"run_id": self._active_run_id})
            try:
                yield self._active_run_id
            finally:
                self._active_run_id = None

    def log_params(self, params: dict) -> None:
        if not self._enabled:
            return
        try:
            # MLflow param values must be strings ≤ 500 chars
            safe = {
                str(k)[:250]: str(v)[:500]
                for k, v in params.items()
            }
            mlflow.log_params(safe)
        except Exception as e:
            log.debug(f"MLflow log_params failed: {e}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self._enabled:
            return
        try:
            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            }
            mlflow.log_metrics(numeric, step=step)
        except Exception as e:
            log.debug(f"MLflow log_metrics failed: {e}")

    def log_tags(self, tags: dict) -> None:
        if not self._enabled:
            return
        try:
            mlflow.set_tags({str(k): str(v)[:500] for k, v in tags.items()})
        except Exception as e:
            log.debug(f"MLflow log_tags failed: {e}")

    def log_artifact_bytes(self, data: bytes, artifact_name: str) -> None:
        if not self._enabled:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=artifact_name, delete=False) as f:
                f.write(data)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="artifacts")
            os.unlink(f.name)
        except Exception as e:
            log.debug(f"MLflow log_artifact failed: {e}")

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self._enabled:
            return
        try:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            log.info("Model logged to MLflow", extra={"path": artifact_path})
        except Exception as e:
            log.debug(f"MLflow log_model failed: {e}")

    def get_best_run(self, metric: str = "f1_weighted") -> dict | None:
        """
        Query MLflow for the best run in the current experiment by metric.
        Returns run dict or None.
        """
        if not self._enabled:
            return None
        try:
            runs = mlflow.search_runs(
                experiment_names=[self._experiment],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            if runs.empty:
                return None
            row = runs.iloc[0]
            return {
                "run_id":   row["run_id"],
                "metric":   metric,
                "value":    row.get(f"metrics.{metric}"),
                "model_uri": f"runs:/{row['run_id']}/model",
            }
        except Exception as e:
            log.debug(f"MLflow search_runs failed: {e}")
            return None

    # ── Pipeline convenience method ──────────────────────────────────────────

    def log_pipeline_run(self, state: dict, model: Any = None) -> str | None:
        """
        Log a complete AutoML pipeline run from PipelineState.
        Returns the MLflow run_id or None.
        """
        if not self._enabled:
            return None

        run_name = (
            f"{state.get('best_model_key', 'unknown')}"
            f"_retry{state.get('retry_count', 0)}"
        )
        tags = {
            "problem_type":   state.get("problem_type", ""),
            "loop_verdict":   state.get("loop_verdict", ""),
            "best_model_key": state.get("best_model_key", ""),
        }

        with self.start_run(run_name=run_name, tags=tags) as run_id:
            # Parameters
            params = {
                "model_key":        state.get("best_model_key", ""),
                "split_strategy":   state.get("split_strategy", ""),
                "test_size":        state.get("test_size", ""),
                "n_features":       len(state.get("selected_features", [])),
                "retry_count":      state.get("retry_count", 0),
                "best_cv_score":    state.get("best_cv_score", ""),
            }
            best_params = state.get("best_params", {})
            params.update({f"hp_{k}": v for k, v in (best_params or {}).items()})
            self.log_params(params)

            # Metrics
            metrics = state.get("eval_metrics", {})
            if metrics:
                self.log_metrics(metrics)

            # CI metrics
            ci = state.get("metric_ci", {})
            if ci:
                ci_flat = {
                    f"{m}_{stat}": v
                    for m, d in ci.items()
                    for stat, v in d.items()
                    if isinstance(v, (int, float))
                }
                self.log_metrics(ci_flat)

            # Log model object
            if model is not None and cfg("mlflow.log_models", default=True):
                self.log_model(model)

            return run_id
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
tracker = ExperimentTracker()
