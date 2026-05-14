"""Shared runtime-store access that does not depend on Streamlit context."""
from __future__ import annotations

import time
from threading import Lock

from ui.runtime_store import RuntimeStore

_RUNTIME_STORE: RuntimeStore | None = None
_TRAINING_MONITOR: dict[str, dict] = {}
_TRAINING_MONITOR_LOCK = Lock()


def set_runtime_store(store: RuntimeStore) -> None:
    global _RUNTIME_STORE
    _RUNTIME_STORE = store


def get_runtime_store() -> RuntimeStore | None:
    return _RUNTIME_STORE


def clear_training_monitor(tid: str) -> None:
    with _TRAINING_MONITOR_LOCK:
        _TRAINING_MONITOR.pop(tid, None)


def get_training_progress(tid: str) -> dict:
    with _TRAINING_MONITOR_LOCK:
        return dict(_TRAINING_MONITOR.get(tid, {}).get("progress", {}))


def set_training_progress(tid: str, updates: dict) -> dict:
    with _TRAINING_MONITOR_LOCK:
        entry = _TRAINING_MONITOR.setdefault(tid, {})
        progress = dict(entry.get("progress", {}))
        progress.update(updates)
        progress["updated_at"] = time.time()
        entry["progress"] = progress
        return dict(progress)


def get_training_events(tid: str) -> list[dict]:
    with _TRAINING_MONITOR_LOCK:
        return list(_TRAINING_MONITOR.get(tid, {}).get("events", []))


def append_training_event(tid: str, message: str, stage: str | None = None) -> list[dict]:
    with _TRAINING_MONITOR_LOCK:
        entry = _TRAINING_MONITOR.setdefault(tid, {})
        events = list(entry.get("events", []))
        event = {"ts": time.time(), "message": message}
        if stage:
            event["stage"] = stage
        if not events or events[-1].get("message") != message:
            events.append(event)
        entry["events"] = events[-40:]
        return list(entry["events"])
