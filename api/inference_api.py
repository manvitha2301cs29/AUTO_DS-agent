"""
api/inference_api.py — FastAPI inference API for production deployment (v5)

Exposes trained models for:
  - Real-time single-row prediction  POST /predict
  - Batch prediction                 POST /predict/batch
  - Model info                       GET  /model/info
  - Health check                     GET  /health
  - Model reload                     POST /model/reload

Usage:
    # Start the server (after training):
    python -m api.inference_api --model-path model.pkl --preprocessor-path preprocessor.pkl

    # Or run programmatically:
    from api.inference_api import create_app
    app = create_app(model, preprocessor, feature_names, problem_type)
"""

from __future__ import annotations

import io
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

import numpy as np

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_FASTAPI:

    class PredictRequest(BaseModel):
        features: dict[str, Any] = Field(
            ...,
            description="Feature values keyed by feature name",
            example={"age": 30, "income": 50000, "category": "A"},
        )
        return_proba: bool = Field(False, description="Return class probabilities")

    class PredictResponse(BaseModel):
        prediction: Any
        probabilities: dict[str, float] | None = None
        prediction_confidence: float | None = None
        latency_ms: float

    class BatchPredictRequest(BaseModel):
        rows: list[dict[str, Any]] = Field(..., description="List of feature dicts")
        return_proba: bool = False

    class BatchPredictResponse(BaseModel):
        predictions: list[Any]
        probabilities: list[dict[str, float]] | None = None
        n_rows: int
        latency_ms: float

    class ModelInfo(BaseModel):
        model_key: str
        problem_type: str
        feature_names: list[str]
        n_features: int
        loaded_at: str
        metrics: dict[str, float] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Model registry (in-memory, reloaded via POST /model/reload)
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names: list[str] = []
        self.problem_type: str = "classification"
        self.model_key: str = "unknown"
        self.loaded_at: str | None = None
        self.metrics: dict = {}

    def load_from_bytes(
        self,
        model_bytes: bytes,
        preprocessor_bytes: bytes | None,
        feature_names: list[str],
        problem_type: str,
        model_key: str = "unknown",
        metrics: dict | None = None,
    ) -> None:
        self.model        = pickle.loads(model_bytes)
        self.preprocessor = pickle.loads(preprocessor_bytes) if preprocessor_bytes else None
        self.feature_names = feature_names
        self.problem_type  = problem_type
        self.model_key     = model_key
        self.loaded_at     = datetime.now(timezone.utc).isoformat()
        self.metrics       = metrics or {}
        log.info("Model loaded into registry", extra={"model_key": model_key})

    def load_from_state(self, state: dict, runtime_objects: dict | None = None) -> None:
        """Load from a completed PipelineState dict."""
        import base64
        model_b64 = state.get("_best_model_b64")
        preprocessor_b64 = state.get("_preprocessor_b64")
        if not model_b64:
            raise ValueError("No _best_model_b64 in state")

        self.load_from_bytes(
            model_bytes=base64.b64decode(model_b64),
            preprocessor_bytes=base64.b64decode(preprocessor_b64) if preprocessor_b64 else None,
            feature_names=state.get("_feature_names") or state.get("selected_features", []),
            problem_type=state.get("problem_type", "classification"),
            model_key=state.get("best_model_key", "unknown"),
            metrics={
                k: v for k, v in (state.get("eval_metrics") or {}).items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            },
        )

    def predict(self, features: dict) -> tuple[Any, dict | None]:
        """Returns (prediction, probabilities_dict or None)."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        import pandas as pd
        row = pd.DataFrame([features])

        # Align columns
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0  # fill missing with 0

        row = row[self.feature_names]

        if self.preprocessor is not None:
            try:
                row = self.preprocessor.transform(row)
            except Exception:
                row = row.values

        pred = self.model.predict(row)[0]
        probas = None

        if hasattr(self.model, "predict_proba"):
            try:
                p = self.model.predict_proba(row)[0]
                if hasattr(self.model, "classes_"):
                    probas = {str(c): float(v) for c, v in zip(self.model.classes_, p)}
                else:
                    probas = {str(i): float(v) for i, v in enumerate(p)}
            except Exception:
                pass

        return pred, probas

    def predict_batch(self, rows: list[dict]) -> tuple[list, list | None]:
        if self.model is None:
            raise RuntimeError("No model loaded")
        import pandas as pd
        df = pd.DataFrame(rows)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        if self.preprocessor is not None:
            try:
                df = self.preprocessor.transform(df)
            except Exception:
                df = df.values
        preds = self.model.predict(df).tolist()
        all_probas = None
        if hasattr(self.model, "predict_proba"):
            try:
                p = self.model.predict_proba(df)
                classes = [str(c) for c in getattr(self.model, "classes_", range(p.shape[1]))]
                all_probas = [
                    {c: float(v) for c, v in zip(classes, row)}
                    for row in p
                ]
            except Exception:
                pass
        return preds, all_probas


_registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(registry: ModelRegistry | None = None) -> "FastAPI":
    if not _HAS_FASTAPI:
        raise ImportError("fastapi and uvicorn are required: pip install fastapi uvicorn")

    reg = registry or _registry
    app = FastAPI(
        title="AutoML Inference API",
        description="Production inference endpoint for AutoML v5 trained models",
        version="5.0.0",
    )

    if cfg("api.enable_cors", default=True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": reg.model is not None,
            "model_key": reg.model_key,
            "loaded_at": reg.loaded_at,
        }

    @app.get("/model/info", response_model=ModelInfo)
    def model_info():
        if reg.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return ModelInfo(
            model_key=reg.model_key,
            problem_type=reg.problem_type,
            feature_names=reg.feature_names,
            n_features=len(reg.feature_names),
            loaded_at=reg.loaded_at or "",
            metrics={k: float(v) for k, v in reg.metrics.items() if isinstance(v, (int, float))},
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if reg.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        t0 = time.perf_counter()
        try:
            pred, probas = reg.predict(req.features)
            confidence = max(probas.values()) if probas else None
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        latency = (time.perf_counter() - t0) * 1000
        return PredictResponse(
            prediction=pred,
            probabilities=probas if req.return_proba else None,
            prediction_confidence=confidence,
            latency_ms=round(latency, 2),
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(req: BatchPredictRequest):
        if reg.model is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        batch_max = cfg("api.batch_max_size", default=1000)
        if len(req.rows) > batch_max:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(req.rows)} exceeds max {batch_max}",
            )
        t0 = time.perf_counter()
        try:
            preds, probas = reg.predict_batch(req.rows)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        latency = (time.perf_counter() - t0) * 1000
        return BatchPredictResponse(
            predictions=preds,
            probabilities=probas if req.return_proba else None,
            n_rows=len(preds),
            latency_ms=round(latency, 2),
        )

    return app


# ── Module-level app for uvicorn ─────────────────────────────────────────────
if _HAS_FASTAPI:
    app = create_app()


if __name__ == "__main__":
    import argparse
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required: pip install uvicorn")

    parser = argparse.ArgumentParser(description="AutoML v5 Inference API")
    parser.add_argument("--model-path",        type=str, default="model.pkl")
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--feature-names",     type=str, default=None, help="comma-separated")
    parser.add_argument("--problem-type",      type=str, default="classification")
    args = parser.parse_args()

    model_bytes = Path(args.model_path).read_bytes()
    pre_bytes   = Path(args.preprocessor_path).read_bytes() if args.preprocessor_path else None
    feat_names  = args.feature_names.split(",") if args.feature_names else []

    _registry.load_from_bytes(model_bytes, pre_bytes, feat_names, args.problem_type)

    uvicorn.run(
        app,
        host=cfg("api.host", default="0.0.0.0"),
        port=cfg("api.port", default=8000),
    )
