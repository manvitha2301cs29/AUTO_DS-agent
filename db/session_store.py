"""
db/session_store.py — SQLite-backed session & chat history store.

Stores AutoML pipeline sessions (metadata + agent message log) so users
can revisit past runs.  LangGraph graph checkpoints are stored separately
in the same SQLite file via SqliteSaver.

Schema
------
sessions  — one row per pipeline run
messages  — append-only agent message log per session
"""

from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RELATIVE_DB = "db/automl_history.db"


def _local_data_db_path() -> Path:
    base = os.getenv("LOCALAPPDATA")
    if base:
        data_dir = Path(base) / "AutoML_v5"
    else:
        data_dir = Path.home() / ".automl_v5"
    return data_dir / "automl_history.db"


def resolve_db_path(db_path: str | os.PathLike | None = None) -> Path:
    """Return an absolute SQLite path for app/session history."""
    raw_path = db_path or os.getenv("AUTOML_DB_PATH") or _DEFAULT_RELATIVE_DB
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_writable_db_path(db_path: str | os.PathLike | None = None) -> Path:
    """Return a SQLite path that supports writes, falling back outside OneDrive."""
    preferred = resolve_db_path(db_path)
    candidates = [preferred]
    fallback = _local_data_db_path()
    if fallback != preferred:
        candidates.append(fallback)
    legacy_root = _PROJECT_ROOT / "automl_history.db"
    if legacy_root not in candidates:
        candidates.append(legacy_root)

    last_error: Exception | None = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(path), timeout=30) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS __automl_write_probe (id INTEGER)")
                conn.execute("DROP TABLE IF EXISTS __automl_write_probe")
                conn.commit()
            return path
        except sqlite3.Error as exc:
            last_error = exc
            continue
        except OSError as exc:
            last_error = exc
            continue

    raise sqlite3.OperationalError(
        f"Could not open a writable AutoML SQLite database. Last error: {last_error}"
    )


# Default DB path can be overridden via env var AUTOML_DB_PATH.
_DEFAULT_DB: Path | None = None


def _connect(db_path: Path | str | None = _DEFAULT_DB) -> sqlite3.Connection:
    db_path = resolve_writable_db_path(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = _DEFAULT_DB) -> None:
    """Create tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id    TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                dataset_name TEXT,
                target       TEXT,
                problem_type TEXT,
                phase        TEXT DEFAULT 'ingest',
                best_model   TEXT,
                cv_score     REAL,
                retry_count  INTEGER DEFAULT 0,
                summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id    TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id, id);
        """)


# ── Session CRUD ──────────────────────────────────────────────────────────────

def upsert_session(
    thread_id: str,
    *,
    dataset_name: str | None = None,
    target: str | None = None,
    problem_type: str | None = None,
    phase: str | None = None,
    best_model: str | None = None,
    cv_score: float | None = None,
    retry_count: int | None = None,
    summary: dict | None = None,
    db_path: Path | str | None = _DEFAULT_DB,
) -> None:
    """Insert or update a session row with whatever fields are provided."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT thread_id FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO sessions
                       (thread_id, created_at, updated_at, dataset_name, target,
                        problem_type, phase, best_model, cv_score, retry_count, summary_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    thread_id, now, now,
                    dataset_name, target, problem_type,
                    phase or "ingest",
                    best_model, cv_score,
                    retry_count or 0,
                    json.dumps(summary) if summary else None,
                ),
            )
        else:
            # Only update non-None fields
            fields: dict[str, Any] = {"updated_at": now}
            if dataset_name  is not None: fields["dataset_name"]  = dataset_name
            if target        is not None: fields["target"]         = target
            if problem_type  is not None: fields["problem_type"]   = problem_type
            if phase         is not None: fields["phase"]          = phase
            if best_model    is not None: fields["best_model"]     = best_model
            if cv_score      is not None: fields["cv_score"]       = cv_score
            if retry_count   is not None: fields["retry_count"]    = retry_count
            if summary       is not None: fields["summary_json"]   = json.dumps(summary)

            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE thread_id = ?",
                (*fields.values(), thread_id),
            )


def list_sessions(db_path: Path | str | None = _DEFAULT_DB) -> list[dict]:
    """Return all sessions, newest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT thread_id, created_at, updated_at, dataset_name, target,
                      problem_type, phase, best_model, cv_score, retry_count
               FROM sessions
               ORDER BY updated_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(thread_id: str, db_path: Path | str | None = _DEFAULT_DB) -> dict | None:
    """Return a single session row or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_session(thread_id: str, db_path: Path | str | None = _DEFAULT_DB) -> None:
    """Delete a session, its messages, and any LangGraph checkpoint rows."""
    with _connect(db_path) as conn:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        ]

        # Remove rows from any checkpoint-style tables that track a thread_id.
        for table_name in tables:
            try:
                cols = [
                    r["name"]
                    for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                ]
            except sqlite3.DatabaseError:
                continue
            if "thread_id" in cols:
                conn.execute(f'DELETE FROM "{table_name}" WHERE thread_id = ?', (thread_id,))

        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))


# ── Message log ───────────────────────────────────────────────────────────────

def append_message(
    thread_id: str,
    role: str,
    content: str,
    db_path: Path | str | None = _DEFAULT_DB,
) -> None:
    """Append an agent or user message to the log for a session."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (thread_id, timestamp, role, content) VALUES (?,?,?,?)",
            (thread_id, now, role, content),
        )


def get_messages(thread_id: str, db_path: Path | str | None = _DEFAULT_DB) -> list[dict]:
    """Return all messages for a session in chronological order."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT timestamp, role, content FROM messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def sync_agent_messages(
    thread_id: str,
    agent_messages: list[str],
    db_path: Path | str | None = _DEFAULT_DB,
) -> None:
    """
    Sync the agent_messages list from PipelineState into the messages table.
    Only appends messages that aren't already stored (idempotent).
    """
    stored = get_messages(thread_id, db_path)
    stored_agent = [m for m in stored if m["role"] == "agent"]
    new_msgs = agent_messages[len(stored_agent):]
    for msg in new_msgs:
        append_message(thread_id, "agent", msg, db_path)
