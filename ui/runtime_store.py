"""
ui/runtime_store.py - LRU+TTL in-memory store for heavy runtime objects.

Extracted from streamlit_app.py (Fix #3: split monolith).
Fix #11: all compound read-modify-write operations hold the lock for
their entire duration to eliminate TOCTOU race conditions.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from threading import Lock


class RuntimeStore:
    def __init__(self, max_sessions: int = 20, ttl_seconds: int = 3_600):
        max_sessions = int(os.getenv("RUNTIME_MAX_SESSIONS", str(max_sessions)))
        ttl_seconds  = int(os.getenv("RUNTIME_TTL_SECONDS",  str(ttl_seconds)))
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._ts: dict[str, float] = {}
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._lock = Lock()

    def _evict_expired(self):
        """Must be called with self._lock already held."""
        now = time.monotonic()
        for k in [k for k, t in self._ts.items() if now - t > self._ttl]:
            self._store.pop(k, None)
            self._ts.pop(k, None)

    def _touch(self, tid: str):
        """Must be called with self._lock already held."""
        self._store.move_to_end(tid)
        self._ts[tid] = time.monotonic()

    @property
    def _raw(self):
        return self._store

    def get_key(self, tid: str, key: str, default=None):
        with self._lock:
            self._evict_expired()
            if tid in self._store:
                self._touch(tid)
            return self._store.get(tid, {}).get(key, default)

    def get_all(self, tid: str) -> dict:
        with self._lock:
            self._evict_expired()
            if tid in self._store:
                self._touch(tid)
            return dict(self._store.get(tid, {}))

    def update(self, tid: str, updates: dict):
        with self._lock:
            self._evict_expired()
            if tid not in self._store:
                if len(self._store) >= self._max:
                    oldest = next(iter(self._store))
                    del self._store[oldest]
                    del self._ts[oldest]
            self._store.setdefault(tid, {}).update(updates)
            self._touch(tid)

    def set_key(self, tid: str, key: str, value):
        self.update(tid, {key: value})

    def delete(self, tid: str):
        with self._lock:
            self._store.pop(tid, None)
            self._ts.pop(tid, None)
