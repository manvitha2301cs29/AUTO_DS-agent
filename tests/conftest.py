"""
tests/conftest.py — shared pytest fixtures and configuration.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_clf_df():
    """200-row binary classification dataframe."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num_a":  rng.normal(size=200),
        "num_b":  rng.normal(size=200),
        "cat_c":  rng.choice(["x", "y", "z"], size=200),
        "target": rng.integers(0, 2, size=200),
    })


@pytest.fixture
def small_reg_df():
    """200-row regression dataframe."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num_a":  rng.normal(size=200),
        "num_b":  rng.normal(size=200),
        "target": rng.normal(scale=10, size=200),
    })


@pytest.fixture
def minimal_pipeline_state(small_clf_df):
    """Minimal PipelineState dict for agent tests."""
    from utils.serialization import df_to_b64
    return {
        "df_parquet_b64":  df_to_b64(small_clf_df),
        "target":          "target",
        "problem_type":    "classification",
        "n_rows":          200,
        "n_cols":          4,
        "retry_count":     0,
        "agent_messages":  [],
        "openai_api_key":  "test-key",
        "preprocessing_decisions": {
            "num_a": {"strategy": "standardize",  "rationale": "numeric"},
            "num_b": {"strategy": "keep_as_is",   "rationale": "numeric"},
            "cat_c": {"strategy": "onehot_encode", "rationale": "categorical"},
        },
    }
