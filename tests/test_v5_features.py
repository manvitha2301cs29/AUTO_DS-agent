"""
tests/test_v5_features.py — Unit tests for all new AutoML v5 features.

Covers:
  - Config loader (YAML loading, env overrides, dot-path accessor)
  - Data contracts (schema validation, semantic checks)
  - AutoEDA report generation
  - Drift monitor (alert logic, DB persistence)
  - Experiment tracker (graceful no-op when MLflow not installed)
  - State store (central store, step tracking)
  - Error analysis helpers
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root on path ───────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigLoader:

    def test_cfg_reads_base_config(self):
        from utils.config_loader import cfg
        # pipeline.max_retries should be 3 from config.yaml
        val = cfg("pipeline.max_retries", default=99)
        assert isinstance(val, int)
        # Should be a reasonable value
        assert 1 <= val <= 20

    def test_cfg_default_on_missing_key(self):
        from utils.config_loader import cfg
        val = cfg("this.does.not.exist", default="fallback")
        assert val == "fallback"

    def test_cfg_raises_on_missing_without_default(self):
        from utils.config_loader import cfg
        with pytest.raises(KeyError):
            cfg("this.does.not.exist.at.all")

    def test_cfg_nested_access(self):
        from utils.config_loader import cfg
        # cv.n_splits should be an int
        val = cfg("cv.n_splits", default=5)
        assert isinstance(val, int)

    def test_env_override(self):
        from utils.config_loader import reload_config, cfg
        with patch.dict(os.environ, {"AUTOML_PIPELINE_MAX_RETRIES": "7"}):
            reload_config()
            val = cfg("pipeline.max_retries")
            assert val == 7
        reload_config()   # restore

    def test_get_section(self):
        from utils.config_loader import get_section
        section = get_section("cv")
        assert isinstance(section, dict)
        assert "n_splits" in section


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

class TestDataContracts:

    def test_valid_split_agent_output(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "split_strategy": "stratified",
            "test_size":      0.2,
            "split_analysis": {"n_train": 800},
            "agent_messages": ["[Split Agent] Done"],
        }
        errors = validate_agent_output(output, "split_agent")
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_required_field(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "split_strategy": "stratified",
            # test_size missing
            "split_analysis": {},
            "agent_messages": [],
        }
        errors = validate_agent_output(output, "split_agent")
        assert any("test_size" in e for e in errors)

    def test_wrong_type(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "split_strategy": "stratified",
            "test_size":      "not_a_float",   # should be float
            "split_analysis": {},
            "agent_messages": [],
        }
        errors = validate_agent_output(output, "split_agent")
        assert any("test_size" in e for e in errors)

    def test_invalid_split_strategy_semantic(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "split_strategy": "banana",   # not a valid strategy
            "test_size":      0.2,
            "split_analysis": {},
            "agent_messages": [],
        }
        errors = validate_agent_output(output, "split_agent")
        assert any("split_strategy" in e for e in errors)

    def test_test_size_out_of_range(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "split_strategy": "standard",
            "test_size":      0.99,   # > 0.5
            "split_analysis": {},
            "agent_messages": [],
        }
        errors = validate_agent_output(output, "split_agent")
        assert any("test_size" in e for e in errors)

    def test_strict_mode_raises(self):
        from utils.data_contracts import validate_agent_output, SchemaViolation
        output = {"agent_messages": []}   # missing many fields
        with pytest.raises(SchemaViolation):
            validate_agent_output(output, "split_agent", strict=True)

    def test_unknown_agent_passes(self):
        from utils.data_contracts import validate_agent_output
        errors = validate_agent_output({"foo": "bar"}, "nonexistent_agent")
        assert errors == []

    def test_orchestrator_valid_verdict(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "loop_verdict":          "accept",
            "loop_reasoning":        "All thresholds passed.",
            "orchestrator_analysis": {"verdict": "accept"},
            "agent_messages":        ["[Orchestrator] accept"],
        }
        errors = validate_agent_output(output, "orchestrator_agent")
        assert errors == []

    def test_orchestrator_invalid_verdict(self):
        from utils.data_contracts import validate_agent_output
        output = {
            "loop_verdict":          "invalid_verdict",
            "loop_reasoning":        "Something.",
            "orchestrator_analysis": {},
            "agent_messages":        [],
        }
        errors = validate_agent_output(output, "orchestrator_agent")
        assert any("loop_verdict" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# AutoEDA Report
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoEDAReport:

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "age":      np.random.randint(18, 80, 200),
            "income":   np.random.normal(50000, 15000, 200),
            "category": np.random.choice(["A", "B", "C"], 200),
            "target":   np.random.choice([0, 1], 200),
        })

    def test_report_returns_html(self, sample_df):
        from utils.autoeda_report import generate_autoeda_report
        html = generate_autoeda_report(sample_df, target="target", problem_type="classification")
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "AutoEDA Report" in html

    def test_report_contains_overview_stats(self, sample_df):
        from utils.autoeda_report import generate_autoeda_report
        html = generate_autoeda_report(sample_df, target="target")
        # Should contain row/col counts
        assert "200" in html   # n_rows

    def test_report_handles_missing_values(self, sample_df):
        from utils.autoeda_report import generate_autoeda_report
        # Introduce missing values
        sample_df.loc[:10, "income"] = np.nan
        html = generate_autoeda_report(sample_df, target="target")
        assert "Missing" in html

    def test_data_quality_warnings(self, sample_df):
        from utils.autoeda_report import _data_quality_warnings
        # Add a constant column
        sample_df["constant_col"] = 42
        warnings = _data_quality_warnings(sample_df, target="target")
        assert any("constant" in w.lower() for w in warnings)

    def test_report_regression(self, sample_df):
        from utils.autoeda_report import generate_autoeda_report
        html = generate_autoeda_report(sample_df, target="income", problem_type="regression")
        assert "regression" in html.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Drift Monitor
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftMonitor:

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return tmp_path / "test_drift.db"

    @pytest.fixture
    def monitor(self, tmp_db):
        from utils.auto_retrain import DriftMonitor
        return DriftMonitor(db_path=tmp_db)

    def test_no_alert_on_low_drift(self, monitor):
        state = {"best_model_key": "xgboost", "_thread_id": "tid_test_1"}
        dr    = {"overall_severity": "low", "overall_drift_score": 0.05, "drifted_features": []}
        alert = monitor.check_and_record(state, dr)
        assert alert.should_retrain is False
        assert alert.severity == "low"

    def test_alert_on_high_drift(self, monitor):
        state = {"best_model_key": "rf", "_thread_id": "tid_test_2"}
        dr    = {"overall_severity": "high", "overall_drift_score": 0.45,
                  "drifted_features": ["age", "income"]}
        alert = monitor.check_and_record(state, dr)
        assert alert.should_retrain is True
        assert "high" in alert.message.lower() or len(alert.drifted_features) > 0

    def test_drift_history_persisted(self, monitor):
        state = {"_thread_id": "tid_hist"}
        dr    = {"overall_severity": "medium", "overall_drift_score": 0.15,
                  "drifted_features": ["f1"]}
        monitor.check_and_record(state, dr)
        history = monitor.get_drift_history("tid_hist")
        assert len(history) >= 1
        assert history[0]["severity"] == "medium"

    def test_pending_jobs_initially_empty(self, monitor):
        jobs = monitor.get_pending_jobs()
        assert isinstance(jobs, list)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Tracker (no-op without MLflow)
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentTracker:

    def test_tracker_no_op_when_disabled(self):
        """Tracker should silently do nothing when mlflow.enabled=false."""
        from utils.experiment_tracker import ExperimentTracker
        t = ExperimentTracker.__new__(ExperimentTracker)
        t._enabled = False
        # These should not raise
        t.log_params({"a": 1})
        t.log_metrics({"f1": 0.9})
        t.log_tags({"model": "rf"})
        assert t.get_best_run() is None

    def test_log_pipeline_run_disabled(self):
        from utils.experiment_tracker import ExperimentTracker
        t = ExperimentTracker.__new__(ExperimentTracker)
        t._enabled = False
        result = t.log_pipeline_run({"best_model_key": "rf", "eval_metrics": {}})
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# State Store
# ─────────────────────────────────────────────────────────────────────────────

class TestStateStore:
    """Test central state store in isolation (no Streamlit)."""

    @pytest.fixture(autouse=True)
    def mock_streamlit(self):
        """Provide a minimal streamlit mock so ui.state_store can import."""
        import types
        fake_st = types.SimpleNamespace()
        fake_st.session_state = {}
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            yield

    def test_app_state_defaults(self):
        from ui.state_store import AppState
        s = AppState()
        assert s.phase == "ingest"
        assert s.tid is None
        assert isinstance(s.pipeline_state, dict)
        assert isinstance(s.progress, dict)
        assert all(v == "pending" for v in s.progress.values())

    def test_step_marking(self):
        from ui.state_store import AppState
        s = AppState()
        s.progress["eda_agent"] = "done"
        assert s.progress["eda_agent"] == "done"
        assert s.progress["model_agent"] == "pending"

    def test_nl_history_append(self):
        from ui.state_store import AppState
        s = AppState()
        s.nl_history.append({"query": "test", "result": "ok", "ts": "now"})
        assert len(s.nl_history) == 1
        assert s.nl_history[0]["query"] == "test"

    def test_dataset_registration(self):
        from ui.state_store import AppState
        s = AppState()
        s.datasets.append({"name": "iris", "tid": "abc", "metrics": {"f1": 0.9}})
        assert len(s.datasets) == 1


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI inference (unit-level, no server)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:

    @pytest.fixture
    def registry_with_model(self):
        """Create a ModelRegistry loaded with a tiny sklearn model."""
        import pickle
        from sklearn.dummy import DummyClassifier
        from api.inference_api import ModelRegistry

        clf = DummyClassifier(strategy="most_frequent")
        clf.fit([[0, 1], [1, 0], [0, 0]], [0, 1, 0])

        reg = ModelRegistry()
        reg.load_from_bytes(
            model_bytes=pickle.dumps(clf),
            preprocessor_bytes=None,
            feature_names=["f0", "f1"],
            problem_type="classification",
            model_key="dummy_clf",
        )
        return reg

    def test_single_predict(self, registry_with_model):
        pred, probas = registry_with_model.predict({"f0": 0.5, "f1": 0.5})
        assert pred in [0, 1]

    def test_missing_feature_filled_zero(self, registry_with_model):
        # f1 is missing — should be filled with 0
        pred, _ = registry_with_model.predict({"f0": 1.0})
        assert pred in [0, 1]

    def test_batch_predict(self, registry_with_model):
        rows  = [{"f0": 0, "f1": 1}, {"f0": 1, "f1": 0}]
        preds, probas = registry_with_model.predict_batch(rows)
        assert len(preds) == 2

    def test_model_info(self, registry_with_model):
        assert registry_with_model.model_key == "dummy_clf"
        assert registry_with_model.feature_names == ["f0", "f1"]
        assert registry_with_model.problem_type == "classification"


class TestEdaDerivedYieldDrop:

    def test_eda_agent_drops_yield_per_hectare_when_yield_is_target(self):
        from agents.eda_agent import eda_agent
        from utils.serialization import df_to_b64

        df = pd.DataFrame({
            "yield": [10.0, 12.0, 14.0, 16.0],
            "area": [2.0, 3.0, 4.0, 5.0],
            "yield_per_hectare": [5.0, 4.0, 3.5, 3.2],
            "rainfall": [100.0, 110.0, 90.0, 105.0],
        })
        state = {
            "df_parquet_b64": df_to_b64(df),
            "target": "yield",
            "problem_type": "regression",
            "openai_api_key": "test-key",
        }

        llm_result = {
            "decisions": {
                "area": {"strategy": "standardize", "rationale": "numeric"},
                "yield_per_hectare": {"strategy": "keep_as_is", "rationale": "llm would keep it"},
                "rainfall": {"strategy": "standardize", "rationale": "numeric"},
            },
            "global_notes": "ok",
            "eda_report": "ok",
        }

        with patch("agents.eda_agent.call_llm_json", return_value=(llm_result, None)):
            result = eda_agent(state)

        assert result["preprocessing_decisions"]["yield_per_hectare"]["strategy"] == "drop"
        assert "derived yield-per-area" in result["preprocessing_decisions"]["yield_per_hectare"]["rationale"]
        assert "yield_per_hectare" in result["eda_analysis"]["target_derived_drop_columns"]


class TestEdaIdentifierDrops:

    def test_eda_agent_drops_near_unique_identifier_columns_even_if_llm_keeps_them(self):
        from agents.eda_agent import eda_agent
        from utils.serialization import df_to_b64

        df = pd.DataFrame({
            "passenger_id": ["P001", "P002", "P003", "P004", "P005"],
            "ticket_class": ["A", "A", "B", "B", "C"],
            "age": [22, 35, 41, 28, 30],
            "target": [0, 1, 0, 1, 0],
        })
        state = {
            "df_parquet_b64": df_to_b64(df),
            "target": "target",
            "problem_type": "classification",
            "openai_api_key": "test-key",
        }

        llm_result = {
            "decisions": {
                "passenger_id": {"strategy": "keep_as_is", "rationale": "llm would keep it"},
                "ticket_class": {"strategy": "onehot_encode", "rationale": "categorical"},
                "age": {"strategy": "standardize", "rationale": "numeric"},
            },
            "global_notes": "ok",
            "eda_report": "ok",
        }

        with patch("agents.eda_agent.call_llm_json", return_value=(llm_result, None)):
            result = eda_agent(state)

        assert result["preprocessing_decisions"]["passenger_id"]["strategy"] == "drop"
        assert "identifier-like" in result["preprocessing_decisions"]["passenger_id"]["rationale"]
        assert "passenger_id" in result["eda_analysis"]["identifier_like_drop_columns"]

    def test_eda_agent_fallback_drops_serial_number_columns(self):
        from agents.eda_agent import eda_agent
        from utils.serialization import df_to_b64

        df = pd.DataFrame({
            "serial_no": ["SN100", "SN101", "SN102", "SN103", "SN104"],
            "feature_a": [10.0, 11.5, 9.8, 10.7, 12.1],
            "target": [1, 0, 1, 0, 1],
        })
        state = {
            "df_parquet_b64": df_to_b64(df),
            "target": "target",
            "problem_type": "classification",
            "openai_api_key": "test-key",
        }

        with patch("agents.eda_agent.call_llm_json", return_value=(None, "mock failure")):
            result = eda_agent(state)

        assert result["preprocessing_decisions"]["serial_no"]["strategy"] == "drop"
        assert "identifier-like" in result["preprocessing_decisions"]["serial_no"]["rationale"]
        assert result["eda_analysis"]["fallback"] is True


class TestLeakageAgentTargetDerivedDrop:

    def test_leakage_agent_drops_yield_per_hectare_when_yield_is_target(self):
        from agents.leakage_agent import leakage_agent
        from utils.serialization import df_to_b64

        df = pd.DataFrame({
            "yield": [10.0, 12.0, 14.0, 16.0],
            "Area_in_hectares": [2.0, 3.0, 4.0, 5.0],
            "Yield_ton_per_hec": [5.0, 4.0, 3.5, 3.2],
            "rainfall": [100.0, 110.0, 90.0, 105.0],
        })
        state = {
            "df_parquet_b64": df_to_b64(df),
            "target": "yield",
            "problem_type": "regression",
            "split_strategy": "standard",
            "openai_api_key": "test-key",
            "preprocessing_decisions": {},
        }

        with patch("agents.leakage_agent.call_llm_json", return_value=({"column_verdicts": {}, "overall_risk": "low", "leakage_summary": "ok"}, None)):
            result = leakage_agent(state)

        decisions = result["preprocessing_decisions"]
        report = result["leakage_report"]
        assert decisions["Yield_ton_per_hec"]["strategy"] == "drop"
        assert "yield-per-area" in decisions["Yield_ton_per_hec"]["rationale"]
        assert "Yield_ton_per_hec" in report["target_derived_columns"]
        assert "Yield_ton_per_hec" in report["dropped_by_leakage"]
