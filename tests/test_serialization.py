import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from utils.agent_utils import agent_error_handler
from utils.serialization import sanitize_for_msgpack


def test_sanitize_for_msgpack_converts_nested_numpy_scalars():
    payload = {
        "score": np.float64(0.91),
        "count": np.int64(7),
        "flag": np.bool_(True),
        "items": [
            {"corr": np.float64(0.42)},
            np.array([1, 2, 3]),
        ],
    }

    result = sanitize_for_msgpack(payload)

    assert result == {
        "score": 0.91,
        "count": 7,
        "flag": True,
        "items": [
            {"corr": 0.42},
            [1, 2, 3],
        ],
    }


def test_sanitize_for_msgpack_converts_nan_float_to_none():
    result = sanitize_for_msgpack({"value": np.float64(np.nan)})
    assert result == {"value": None}


def test_agent_error_handler_sanitizes_successful_agent_output():
    @agent_error_handler("Test Agent")
    def _agent(_state):
        return {"metric": np.float64(1.23)}

    assert _agent({}) == {"metric": 1.23}
