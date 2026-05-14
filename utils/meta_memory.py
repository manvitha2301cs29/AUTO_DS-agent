from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_DB = Path(os.getenv("AUTOML_DB_PATH", "automl_history.db"))


def _connect(db_path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_meta_memory(db_path: Path = _DEFAULT_DB) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                n_rows INTEGER,
                n_cols INTEGER,
                n_numeric INTEGER,
                n_categorical INTEGER,
                problem_type TEXT,
                class_balance_min REAL,
                target_skew REAL,
                split_strategy TEXT,
                imbalance_bucket TEXT,
                best_model TEXT,
                best_score REAL,
                dataset_hash TEXT,
                extra_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_problem ON meta_memory(problem_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_split ON meta_memory(split_strategy)")
        conn.commit()


def _extract_meta_features(state: dict) -> dict[str, Any]:
    col_meta: list[dict] = state.get("column_meta") or []
    n_numeric = sum(1 for c in col_meta if "mean" in c and not c.get("is_target"))
    n_categorical = sum(1 for c in col_meta if "top_values" in c and not c.get("is_target"))
    problem_type = state.get("problem_type", "")
    split_analysis = state.get("split_analysis") or {}
    cb = split_analysis.get("class_balance") or {}
    class_balance_min = float(min(cb.values())) if cb else None

    target_skew = None
    if problem_type == "regression":
        for c in col_meta:
            if c.get("is_target"):
                target_skew = c.get("skew")
                break

    col_sig = "|".join(f"{c['name']}:{c['dtype']}" for c in col_meta if not c.get("is_target"))
    dataset_hash = hashlib.sha256(col_sig.encode()).hexdigest()[:16]
    split_strategy = split_analysis.get("strategy") or state.get("split_strategy", "standard")

    if class_balance_min is None:
        imbalance_bucket = "n/a"
    elif class_balance_min >= 0.40:
        imbalance_bucket = "balanced"
    elif class_balance_min >= 0.20:
        imbalance_bucket = "mild"
    elif class_balance_min >= 0.05:
        imbalance_bucket = "moderate"
    else:
        imbalance_bucket = "severe"

    total_cols = max(n_numeric + n_categorical, 1)
    return {
        "n_rows": state.get("n_rows", 0),
        "n_cols": state.get("n_cols", 0),
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "problem_type": problem_type,
        "class_balance_min": class_balance_min,
        "target_skew": target_skew,
        "dataset_hash": dataset_hash,
        "split_strategy": split_strategy,
        "imbalance_bucket": imbalance_bucket,
        "col_type_fingerprint": {
            "frac_numeric": n_numeric / total_cols,
            "frac_categorical": n_categorical / total_cols,
            "missing_pct": float(state.get("auto_dataset_insights", {}).get("missing_pct", 0) or 0),
            "target_skew_norm": min(abs(target_skew or 0) / 10.0, 1.0),
        },
    }


def record_run(state: dict, db_path: Path = _DEFAULT_DB) -> None:
    best_model = state.get("best_model_key")
    best_score = state.get("best_cv_score")
    if not best_model or best_score is None:
        return
    init_meta_memory(db_path)
    meta = _extract_meta_features(state)
    extra = {
        "selected_features": state.get("selected_features", []),
        "split_strategy": state.get("split_strategy", ""),
        "eval_metrics": {k: v for k, v in (state.get("eval_metrics") or {}).items() if k != "confusion_matrix"},
        "col_type_fingerprint": meta["col_type_fingerprint"],
    }
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO meta_memory
            (created_at, n_rows, n_cols, n_numeric, n_categorical, problem_type,
             class_balance_min, target_skew, split_strategy, imbalance_bucket,
             best_model, best_score, dataset_hash, extra_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(UTC).isoformat(),
                meta["n_rows"],
                meta["n_cols"],
                meta["n_numeric"],
                meta["n_categorical"],
                meta["problem_type"],
                meta["class_balance_min"],
                meta["target_skew"],
                meta["split_strategy"],
                meta["imbalance_bucket"],
                best_model,
                best_score,
                meta["dataset_hash"],
                json.dumps(extra),
            ),
        )
        conn.commit()


def get_similar_runs(state: dict, k: int = 5, db_path: Path = _DEFAULT_DB) -> list[dict]:
    try:
        init_meta_memory(db_path)
        query_meta = _extract_meta_features(state)
        with closing(_connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM meta_memory WHERE problem_type = ? ORDER BY created_at DESC LIMIT 200",
                (query_meta["problem_type"],),
            ).fetchall()
        if not rows:
            return []

        split_idx = {"standard": 0, "stratified": 1, "group_based": 2, "time_series": 3}
        imbalance_idx = {"balanced": 0, "mild": 1, "moderate": 2, "severe": 3, "n/a": 0}
        qfp = query_meta["col_type_fingerprint"]

        def vec(meta_or_row: dict, is_row: bool = False) -> np.ndarray:
            if is_row:
                extra = json.loads(meta_or_row.get("extra_json") or "{}")
                fp = extra.get("col_type_fingerprint", {})
            else:
                fp = qfp
            return np.array(
                [
                    np.log1p(meta_or_row.get("n_rows") or 0) / 15,
                    (meta_or_row.get("n_cols") or 0) / 50,
                    fp.get("frac_numeric", 0.5),
                    fp.get("frac_categorical", 0.5),
                    fp.get("missing_pct", 0) / 100,
                    fp.get("target_skew_norm", 0),
                    meta_or_row.get("class_balance_min") or 0.5,
                    split_idx.get(meta_or_row.get("split_strategy") or "standard", 0) / 3,
                    imbalance_idx.get(meta_or_row.get("imbalance_bucket") or "balanced", 0) / 3,
                ],
                dtype=float,
            )

        q = vec(query_meta)
        q_norm = np.linalg.norm(q) + 1e-9
        results = []
        for row in rows:
            item = dict(row)
            rv = vec(item, is_row=True)
            sim = float(np.dot(q, rv) / (q_norm * (np.linalg.norm(rv) + 1e-9)))
            results.append({**item, "_distance": 1.0 - sim})
        results.sort(key=lambda x: x["_distance"])
        return results[:k]
    except Exception:
        return []


def suggest_models(state: dict, k: int = 5, db_path: Path = _DEFAULT_DB) -> list[str]:
    similar = get_similar_runs(state, k=k, db_path=db_path)
    if not similar:
        return []
    model_scores: dict[str, float] = {}
    for rank, run in enumerate(similar):
        model_key = run.get("best_model")
        if not model_key:
            continue
        model_scores[model_key] = model_scores.get(model_key, 0.0) + (float(run.get("best_score") or 0.0) / (rank + 1))
    return [model for model, _ in sorted(model_scores.items(), key=lambda item: item[1], reverse=True)]
