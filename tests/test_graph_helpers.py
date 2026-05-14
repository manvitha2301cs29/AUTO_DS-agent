"""
tests/test_graph_helpers.py

Focused tests for ui.graph_helpers helper functions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui import graph_helpers


class _FakeGraph:
    def __init__(self):
        self.calls = []

    def update_state(self, config, payload):
        self.calls.append((config, payload))


class TestGraphHelpers:
    def test_update_graph_state_uses_lazy_graph_and_merges_existing_state(self, monkeypatch):
        fake_graph = _FakeGraph()
        monkeypatch.setattr(graph_helpers, "_graph", fake_graph)
        monkeypatch.setattr(graph_helpers, "get_state", lambda tid: {"existing": 1})

        graph_helpers.update_graph_state({"new_value": 2}, "tid-123")

        assert len(fake_graph.calls) == 1
        config, payload = fake_graph.calls[0]
        assert config == {"configurable": {"thread_id": "tid-123"}}
        assert payload["existing"] == 1
        assert payload["new_value"] == 2
        assert payload["_thread_id"] == "tid-123"
