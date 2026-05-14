from __future__ import annotations

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler


@agent_error_handler("Split Agent")
def split_agent(state: PipelineState) -> dict:
    strategy = state.get("split_strategy", "standard")
    test_size = float(state.get("test_size", 0.2) or 0.2)
    return {
        "split_strategy": strategy,
        "split_rationale": state.get("split_rationale", f"Using {strategy} split."),
        "split_warnings": state.get("split_warnings", []),
        "test_size": test_size,
        "datetime_column": state.get("datetime_column"),
        "group_column": state.get("group_column"),
        "split_analysis": {
            "strategy": strategy,
            "test_size": test_size,
            "warnings": state.get("split_warnings", []),
        },
        "agent_messages": [
            f"[Split Agent] Recommended '{strategy}' split with test size {test_size:.0%}."
        ],
    }
