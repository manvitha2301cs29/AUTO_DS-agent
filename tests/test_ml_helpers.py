"""
tests/test_ml_helpers.py

Unit tests for utils/ml_helpers.py covering:
  - auto_dataset_insights: basic stats, imbalance, ordinal detection (fix #6),
    multi-label detection (fix #6), type warnings
  - set_global_seed: runs without error
  - compute_baseline_score: correct for classification and regression
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from utils.ml_helpers import (
    auto_dataset_insights,
    build_preprocessor,
    select_loss_function,
    set_global_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_clf_df(n=200, n_classes=2, imbalance=False):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    if imbalance:
        labels = rng.choice([0, 1], size=n, p=[0.95, 0.05])
    else:
        labels = rng.choice(list(range(n_classes)), size=n)
    X["target"] = labels
    return X


def _make_reg_df(n=200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = rng.normal(scale=10, size=n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# auto_dataset_insights
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoDatasetInsights:
    def test_basic_shape(self):
        df = _make_clf_df()
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights["n_rows"] == 200
        assert insights["n_cols"] == 3

    def test_no_false_warnings_on_clean_data(self):
        df = _make_clf_df()
        insights = auto_dataset_insights(df, "target", "classification")
        # Clean data should have no duplicate or missing warnings
        dup_warns = [w for w in insights["warnings"] if "duplicate" in w.lower()]
        assert len(dup_warns) == 0

    def test_detects_duplicates(self):
        df = _make_clf_df(n=100)
        df = pd.concat([df, df.head(10)], ignore_index=True)
        insights = auto_dataset_insights(df, "target", "classification")
        assert any("duplicate" in w.lower() for w in insights["warnings"])
        assert insights["duplicate_rows"] == 10

    def test_detects_severe_imbalance(self):
        df = _make_clf_df(n=200, imbalance=True)
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights["imbalance_ratio"] is not None
        assert insights["imbalance_ratio"] > 5

    def test_regression_no_imbalance_ratio(self):
        df = _make_reg_df()
        insights = auto_dataset_insights(df, "target", "regression")
        assert insights["imbalance_ratio"] is None

    def test_recommended_problem_type_regression(self):
        df = _make_reg_df()
        insights = auto_dataset_insights(df, "target", "regression")
        assert insights["recommended_problem_type"] == "regression"

    def test_recommended_problem_type_classification(self):
        df = _make_clf_df(n_classes=2)
        insights = auto_dataset_insights(df, "target", "classification")
        assert "classification" in insights["recommended_problem_type"]

    # Fix #6: ordinal detection
    def test_detects_ordinal_target(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(size=300),
            "b": rng.normal(size=300),
            "target": rng.integers(1, 6, size=300),  # 1-5 ordinal ratings
        })
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights.get("likely_ordinal") is True
        assert any("ordinal" in t.lower() for t in insights["type_issues"])

    def test_non_ordinal_integers_not_flagged(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(size=300),
            "target": rng.integers(0, 100, size=300),  # too many unique values
        })
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights.get("likely_ordinal") is not True

    # Fix #6: multi-label detection
    def test_detects_multi_label_target(self):
        labels = ["cat,dog", "fish", "cat,bird,dog", "bird", "dog,fish"] * 40
        df = pd.DataFrame({"a": range(200), "target": labels})
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights.get("likely_multi_label") is True
        assert any("multi" in t.lower() for t in insights["type_issues"])

    def test_normal_string_labels_not_flagged_as_multi_label(self):
        df = pd.DataFrame({
            "a": range(200),
            "target": ["cat", "dog", "bird"] * 66 + ["cat", "cat"],
        })
        insights = auto_dataset_insights(df, "target", "classification")
        assert insights.get("likely_multi_label") is not True

    def test_missing_values_reported(self):
        df = _make_clf_df(n=100)
        df.loc[:60, "a"] = np.nan  # 61% missing in col 'a'
        insights = auto_dataset_insights(df, "target", "classification")
        assert any("missing" in w.lower() for w in insights["warnings"])

    def test_constant_columns_detected(self):
        df = _make_clf_df(n=50)
        df["constant_col"] = 7
        insights = auto_dataset_insights(df, "target", "classification")
        assert "constant_col" in insights["constant_columns"]
        assert any("zero-variance" in w.lower() or "constant" in w.lower()
                   for w in insights["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# set_global_seed
# ─────────────────────────────────────────────────────────────────────────────

class TestSetGlobalSeed:
    def test_runs_without_error(self):
        seed = set_global_seed(42)
        assert seed == 42

    def test_uses_global_seed_when_none(self):
        seed = set_global_seed()
        assert isinstance(seed, int)

    def test_reproducibility(self):
        set_global_seed(0)
        a = np.random.rand(5).tolist()
        set_global_seed(0)
        b = np.random.rand(5).tolist()
        assert a == b


class TestSelectLossFunction:
    def test_classification_returns_metadata_dict(self):
        result = select_loss_function(
            "classification",
            y=np.array([0, 0, 0, 1]),
            metadata={"n_samples": 4, "n_features": 3},
        )
        assert result["loss_function"] == "log_loss"
        assert result["n_samples"] == 4
        assert result["n_features"] == 3
        assert result["imbalance_ratio"] == pytest.approx(3.0)

    def test_regression_returns_rmse(self):
        result = select_loss_function(
            "regression",
            y=np.array([1.2, 2.3, 3.4]),
            metadata={"n_features": 2},
        )
        assert result["loss_function"] == "rmse"
        assert result["problem_type"] == "regression"
        assert result["n_features"] == 2


class TestBuildPreprocessor:
    def test_handles_bool_columns_with_categorical_strategy(self):
        X = pd.DataFrame(
            {
                "num_a": [1.0, 2.0, 3.0, 4.0],
                "flag": [True, False, True, False],
                "cat_c": ["x", "y", "x", "z"],
            }
        )
        decisions = {
            "num_a": {"strategy": "standardize"},
            "flag": {"strategy": "onehot_encode"},
            "cat_c": {"strategy": "onehot_encode"},
        }

        preprocessor, numeric_cols, categorical_cols = build_preprocessor(X, decisions)
        X_t = preprocessor.fit_transform(X)

        assert X_t.shape[0] == len(X)
        assert "flag" in numeric_cols
        assert "cat_c" in categorical_cols

    def test_handles_nullable_boolean_columns(self):
        X = pd.DataFrame(
            {
                "flag": pd.Series([True, False, None, True], dtype="boolean"),
                "num_a": [10.0, 20.0, 30.0, 40.0],
            }
        )
        decisions = {
            "flag": {"strategy": "standardize"},
            "num_a": {"strategy": "standardize"},
        }

        preprocessor, _, _ = build_preprocessor(X, decisions)
        X_t = preprocessor.fit_transform(X)

        assert X_t.shape[0] == len(X)
        assert np.isfinite(X_t).all()
