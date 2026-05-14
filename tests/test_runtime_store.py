"""
tests/test_runtime_store.py

Tests for ui/runtime_store.py:
  - get/set/delete semantics
  - LRU eviction when max_sessions exceeded
  - Thread safety (basic smoke test)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import threading
import pytest
from ui.runtime_store import RuntimeStore


class TestRuntimeStore:
    def test_set_and_get(self):
        store = RuntimeStore(max_sessions=5, ttl_seconds=3600)
        store.set_key("tid1", "model", "rf")
        assert store.get_key("tid1", "model") == "rf"

    def test_missing_key_returns_default(self):
        store = RuntimeStore(max_sessions=5, ttl_seconds=3600)
        assert store.get_key("nonexistent", "key") is None
        assert store.get_key("nonexistent", "key", default=42) == 42

    def test_update_merges(self):
        store = RuntimeStore(max_sessions=5, ttl_seconds=3600)
        store.update("tid1", {"a": 1, "b": 2})
        store.update("tid1", {"b": 99, "c": 3})
        assert store.get_key("tid1", "a") == 1
        assert store.get_key("tid1", "b") == 99
        assert store.get_key("tid1", "c") == 3

    def test_delete_removes_entry(self):
        store = RuntimeStore(max_sessions=5, ttl_seconds=3600)
        store.set_key("tid1", "x", 10)
        store.delete("tid1")
        assert store.get_key("tid1", "x") is None

    def test_get_all_returns_copy(self):
        store = RuntimeStore(max_sessions=5, ttl_seconds=3600)
        store.update("tid1", {"a": 1})
        snapshot = store.get_all("tid1")
        snapshot["a"] = 999  # mutate copy
        assert store.get_key("tid1", "a") == 1  # original unchanged

    def test_lru_eviction_at_max_sessions(self):
        store = RuntimeStore(max_sessions=3, ttl_seconds=3600)
        store.set_key("tid1", "v", 1)
        store.set_key("tid2", "v", 2)
        store.set_key("tid3", "v", 3)
        # Adding a 4th should evict the LRU (tid1)
        store.set_key("tid4", "v", 4)
        assert store.get_key("tid1", "v") is None
        assert store.get_key("tid4", "v") == 4

    def test_ttl_eviction(self):
        store = RuntimeStore(max_sessions=10, ttl_seconds=0)  # immediate expiry
        store.set_key("tid1", "v", 1)
        time.sleep(0.01)
        # Trigger eviction via any operation
        store.set_key("tid2", "v", 2)
        assert store.get_key("tid1", "v") is None

    def test_thread_safety_concurrent_writes(self):
        store = RuntimeStore(max_sessions=100, ttl_seconds=3600)
        errors = []

        def write(tid, val):
            try:
                for _ in range(50):
                    store.set_key(tid, "counter", val)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(f"t{i}", i)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
