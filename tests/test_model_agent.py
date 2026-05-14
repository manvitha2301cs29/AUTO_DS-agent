"""
tests/test_model_agent.py

Focused regressions for model-agent helper behavior.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import model_agent


class _FakeRuntimeStore:
    def __init__(self, raw):
        self._raw = raw


def test_resolve_training_tid_prefers_state_thread_id(monkeypatch):
    monkeypatch.setattr(model_agent, "get_runtime_store", lambda: _FakeRuntimeStore({"tid-a": {}}))

    assert model_agent._resolve_training_tid({"_thread_id": "tid-state"}) == "tid-state"


def test_resolve_training_tid_falls_back_to_single_runtime_bucket(monkeypatch):
    monkeypatch.setattr(model_agent, "get_runtime_store", lambda: _FakeRuntimeStore({"tid-only": {"X_train_t": 1}}))

    assert model_agent._resolve_training_tid({}, "") == "tid-only"


def test_dedupe_and_limit_candidates_keeps_order():
    candidates = [
        {"model_key": "lightgbm"},
        {"model_key": "xgboost"},
        {"model_key": "lightgbm"},
        {"model_key": "random_forest"},
    ]

    assert model_agent._dedupe_and_limit_candidates(candidates, 2) == [
        {"model_key": "lightgbm"},
        {"model_key": "xgboost"},
    ]


def test_tuning_percent_stays_in_expected_range():
    assert model_agent._tuning_percent(0, 30) == 35
    assert model_agent._tuning_percent(15, 30) >= 35
    assert model_agent._tuning_percent(30, 30) == 95


def test_ensure_minimum_candidates_adds_fallback_models():
    ensured = model_agent._ensure_minimum_candidates(
        [{"model_key": "lightgbm"}],
        problem_type="classification",
        min_candidates=3,
    )

    keys = [row["model_key"] for row in ensured]
    assert keys[0] == "lightgbm"
    assert len(set(keys)) >= 3


def test_fallback_candidate_templates_cover_regression_with_multiple_models():
    templates = model_agent._fallback_candidate_templates("regression")

    assert len(templates) >= 3
    assert len({row["model_key"] for row in templates}) == len(templates)
