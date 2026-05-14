"""
utils/auto_retrain.py — Auto-retraining pipeline + drift alerts (v5)

Extends the existing drift detection to trigger automatic retraining when
drift severity exceeds the configured threshold.

Features:
  - DriftMonitor: persistent store of drift scores across time
  - check_retrain_needed(): returns True if retraining is overdue
  - schedule_retrain(): writes a retrain job spec to SQLite
  - RetainAlert: typed alert surfaced in the Streamlit UI

Usage:
    from utils.auto_retrain import drift_monitor, RetainAlert

    alert = drift_monitor.check_and_record(state, current_drift_report)
    if alert.should_retrain:
        st.warning(alert.message)
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path(os.getenv("AUTOML_DB_PATH", "automl_history.db"))


# ─────────────────────────────────────────────────────────────────────────────
# DB schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,
    overall_score   REAL,
    severity        TEXT,
    drifted_features TEXT,
    model_key       TEXT,
    retrain_triggered INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrain_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',
    reason          TEXT,
    completed_at    TEXT
);
"""


def _init_retrain_db(db_path: Path = _DB_PATH) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(_SCHEMA)


_init_retrain_db()


# ─────────────────────────────────────────────────────────────────────────────
# Alert dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetainAlert:
    should_retrain: bool = False
    severity: str = "low"
    message: str = ""
    drifted_features: list[str] = field(default_factory=list)
    drift_score: float = 0.0
    days_since_last_retrain: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Persists drift history and decides when retraining is needed.
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self._db = db_path
        _init_retrain_db(db_path)   # ensure tables exist for this specific db

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db), check_same_thread=False)

    def record_drift(
        self,
        thread_id: str,
        drift_report: dict,
        model_key: str = "",
        retrain_triggered: bool = False,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO drift_history
                  (thread_id, recorded_at, overall_score, severity, drifted_features,
                   model_key, retrain_triggered)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    datetime.now(timezone.utc).isoformat(),
                    drift_report.get("overall_drift_score", 0.0),
                    drift_report.get("overall_severity", "low"),
                    json.dumps(list(drift_report.get("drifted_features", []))[:20]),
                    model_key,
                    int(retrain_triggered),
                ),
            )

    def days_since_last_retrain(self, thread_id: str) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT recorded_at FROM drift_history
                WHERE thread_id = ? AND retrain_triggered = 1
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        last = datetime.fromisoformat(row[0])
        return (datetime.now(timezone.utc) - last).days

    def recent_severity_trend(self, thread_id: str, n: int = 5) -> list[str]:
        """Return severity of the last n drift checks."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT severity FROM drift_history
                WHERE thread_id = ?
                ORDER BY recorded_at DESC LIMIT ?
                """,
                (thread_id, n),
            ).fetchall()
        return [r[0] for r in rows]

    def check_and_record(
        self,
        state: dict,
        drift_report: dict | None = None,
    ) -> RetainAlert:
        """
        Main entry point: record drift, decide if retraining is needed.
        """
        thread_id = state.get("_thread_id", "default")
        dr = drift_report or state.get("drift_report") or {}
        severity  = dr.get("overall_severity", "low")
        score     = float(dr.get("overall_drift_score", 0.0))
        drifted   = list(dr.get("drifted_features", []))

        interval_days = cfg("drift.drift_check_interval_days", default=7)
        days_since    = self.days_since_last_retrain(thread_id)
        auto_retrain  = cfg("drift.auto_retrain_on_drift", default=False)

        should_retrain = False
        reason = ""

        if severity == "high":
            should_retrain = True
            reason = f"High drift detected (score={score:.3f}) in {len(drifted)} features."
        elif severity == "medium":
            trend = self.recent_severity_trend(thread_id, n=3)
            if trend.count("medium") + trend.count("high") >= 2:
                should_retrain = True
                reason = "Sustained medium drift across multiple checks."
        if days_since is not None and days_since >= interval_days:
            should_retrain = True
            reason += f" {days_since} days since last retrain (threshold={interval_days})."

        self.record_drift(
            thread_id, dr,
            model_key=state.get("best_model_key", ""),
            retrain_triggered=should_retrain and auto_retrain,
        )

        if should_retrain and auto_retrain:
            self._schedule_retrain(thread_id, reason)

        alert = RetainAlert(
            should_retrain=should_retrain,
            severity=severity,
            message=reason.strip() or "No significant drift detected.",
            drifted_features=drifted[:10],
            drift_score=score,
            days_since_last_retrain=days_since,
        )
        if should_retrain:
            log.info(
                "Retrain alert triggered",
                extra={"thread_id": thread_id, "reason": reason},
            )
        return alert

    def _schedule_retrain(self, thread_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO retrain_jobs (thread_id, created_at, reason)
                VALUES (?, ?, ?)
                """,
                (thread_id, datetime.now(timezone.utc).isoformat(), reason),
            )

    def get_pending_jobs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM retrain_jobs WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows] if rows else []

    def get_drift_history(self, thread_id: str, n: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT recorded_at, overall_score, severity, drifted_features
                FROM drift_history
                WHERE thread_id = ?
                ORDER BY recorded_at DESC LIMIT ?
                """,
                (thread_id, n),
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "recorded_at":      r[0],
                "overall_score":    r[1],
                "severity":         r[2],
                "drifted_features": json.loads(r[3] or "[]"),
            })
        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
drift_monitor = DriftMonitor()
