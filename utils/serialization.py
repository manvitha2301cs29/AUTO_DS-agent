"""
utils/serialization.py — Lossless serialisation helpers.

DataFrame ↔ base64-parquet  (preserves dtypes, datetimes, categoricals)
sklearn objects ↔ base64-joblib  (for export only, NOT stored in graph state)
"""

from __future__ import annotations
import base64
import io
import joblib
import pandas as pd
import numpy as np


# ── msgpack / LangGraph state sanitizer ───────────────────────────────────────

def sanitize_for_msgpack(obj):
    """
    Recursively convert numpy scalar types to native Python equivalents so that
    LangGraph's msgpack checkpointer can serialise agent output dicts without
    raising 'Type is not msgpack serializable: numpy.float64' (or similar).

    Call this on the full return dict of any agent before returning it, e.g.:

        from utils.serialization import sanitize_for_msgpack
        return sanitize_for_msgpack({
            "feature_proposals": proposals,
            "selected_features": selected,
            ...
        })
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_msgpack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [sanitize_for_msgpack(v) for v in obj]
        return sanitized if isinstance(obj, list) else tuple(sanitized)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── DataFrame ─────────────────────────────────────────────────────────────────

def df_to_b64(df: pd.DataFrame) -> str:
    """Serialise DataFrame to base64-encoded Parquet string."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=True, engine="pyarrow")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_df(b64: str) -> pd.DataFrame:
    """Deserialise base64-encoded Parquet string back to DataFrame."""
    buf = io.BytesIO(base64.b64decode(b64))
    return pd.read_parquet(buf, engine="pyarrow")


# ── sklearn / arbitrary objects ───────────────────────────────────────────────

def obj_to_b64(obj) -> str:
    """Serialise any joblib-picklable object to base64 string."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_obj(b64: str):
    """Deserialise a base64 joblib string back to the original object."""
    buf = io.BytesIO(base64.b64decode(b64))
    return joblib.load(buf)


def pipeline_to_bytes(preprocessor, model) -> bytes:
    """
    Bundle preprocessor + model into a single sklearn Pipeline and
    return raw joblib bytes suitable for st.download_button.
    """
    from sklearn.pipeline import Pipeline
    combined = Pipeline([("preprocessor", preprocessor), ("model", model)])
    buf = io.BytesIO()
    joblib.dump(combined, buf)
    return buf.getvalue()
