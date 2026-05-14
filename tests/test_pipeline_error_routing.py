"""
tests/test_pipeline_error_routing.py

Unit tests for agents/pipeline.py covering fix #2:
  - pipeline_error_node writes correct state
  - _make_error_router routes to pipeline_error when _agent_error=True
  - _make_error_router routes to next node on clean state
  - has_agent_error correctly detects error state
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.pipeline import pipeline_error_node, _make_error_router
from utils.agent_utils import has_agent_error


class TestPipelineErrorNode:
    def test_sets_loop_verdict_error(self):
        state = {"_agent_error_source": "EDA Agent", "agent_messages": ["[EDA Agent] ❌ fail"]}
        result = pipeline_error_node(state)
        assert result["loop_verdict"] == "error"

    def test_includes_source_in_message(self):
        state = {"_agent_error_source": "Feature Agent", "agent_messages": []}
        result = pipeline_error_node(state)
        assert "Feature Agent" in result["agent_messages"][0]

    def test_extracts_error_message_from_agent_messages(self):
        state = {
            "_agent_error_source": "Model Agent",
            "agent_messages": ["[Model Agent] ❌ Unexpected error: OOM"],
        }
        result = pipeline_error_node(state)
        assert "OOM" in result["loop_reasoning"]

    def test_works_with_empty_messages(self):
        state = {"agent_messages": []}
        result = pipeline_error_node(state)
        assert result["loop_verdict"] == "error"
        assert isinstance(result["loop_reasoning"], str)


class TestMakeErrorRouter:
    def test_routes_to_pipeline_error_when_agent_error_true(self):
        router = _make_error_router("hitl_eda")
        state = {"_agent_error": True, "_agent_error_source": "EDA Agent"}
        assert router(state) == "pipeline_error"

    def test_routes_to_next_node_on_clean_state(self):
        router = _make_error_router("hitl_eda")
        state = {"eda_report": "all good"}
        assert router(state) == "hitl_eda"

    def test_routes_to_next_node_when_error_is_false(self):
        router = _make_error_router("hitl_features")
        state = {"_agent_error": False}
        assert router(state) == "hitl_features"

    def test_different_next_nodes(self):
        for next_node in ("hitl_eda", "hitl_leakage", "hitl_features", "model_agent"):
            router = _make_error_router(next_node)
            assert router({}) == next_node
            assert router({"_agent_error": True}) == "pipeline_error"


class TestHasAgentError:
    def test_false_on_empty_state(self):
        assert not has_agent_error({})

    def test_true_on_error_state(self):
        assert has_agent_error({"_agent_error": True})

    def test_false_on_explicit_false(self):
        assert not has_agent_error({"_agent_error": False})

    def test_false_on_none(self):
        assert not has_agent_error({"_agent_error": None})
