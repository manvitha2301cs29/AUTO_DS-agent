"""
utils/data_contracts.py — Data contracts + schema validation between agents (v5)

Every agent output is validated against a contract before being written to
PipelineState.  This prevents silent type mismatches, missing required fields,
and corrupt state from propagating downstream.

Usage:
    from utils.data_contracts import validate_agent_output, DataContract

    # In an agent:
    result = {...}
    validate_agent_output(result, "feature_agent")   # raises on violation

    # Or use the decorator:
    @validate_output("model_agent")
    def model_agent(state): ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Type


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

class SchemaViolation(ValueError):
    """Raised when an agent output violates its data contract."""
    pass


def _check_field(
    output: dict,
    field: str,
    expected_type: type | tuple,
    required: bool = True,
    nullable: bool = False,
) -> list[str]:
    errors = []
    if field not in output:
        if required:
            errors.append(f"Missing required field: '{field}'")
        return errors
    val = output[field]
    if val is None:
        if not nullable and required:
            errors.append(f"Field '{field}' must not be None")
        return errors
    if not isinstance(val, expected_type):
        errors.append(
            f"Field '{field}' expected {expected_type}, got {type(val).__name__}"
        )
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Agent contracts (schema per agent output)
# ─────────────────────────────────────────────────────────────────────────────

_CONTRACTS: dict[str, list[dict]] = {

    "eda_agent": [
        {"field": "column_meta",              "type": list,  "required": True},
        {"field": "preprocessing_decisions",  "type": dict,  "required": True},
        {"field": "eda_analysis",             "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "leakage_agent": [
        {"field": "leakage_report",           "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "feature_agent": [
        {"field": "feature_proposals",        "type": list,  "required": True},
        {"field": "selected_features",        "type": list,  "required": True},
        {"field": "feature_analysis",         "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "split_agent": [
        {"field": "split_strategy",           "type": str,   "required": True},
        {"field": "test_size",                "type": float, "required": True},
        {"field": "split_analysis",           "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "model_agent": [
        {"field": "model_candidates",         "type": list,  "required": True},
        {"field": "best_model_key",           "type": str,   "required": True},
        {"field": "best_cv_score",            "type": float, "required": True},
        {"field": "model_analysis",           "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "eval_agent": [
        {"field": "eval_metrics",             "type": dict,  "required": True},
        {"field": "eval_analysis",            "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "ensemble_agent": [
        {"field": "ensemble_report",          "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],

    "orchestrator_agent": [
        {"field": "loop_verdict",             "type": str,   "required": True},
        {"field": "loop_reasoning",           "type": str,   "required": True},
        {"field": "orchestrator_analysis",    "type": dict,  "required": True},
        {"field": "agent_messages",           "type": list,  "required": True},
    ],
}

_VALID_LOOP_VERDICTS = {"accept", "retry_features", "retry_models", "retry_both", "error"}
_VALID_SPLIT_STRATEGIES = {"standard", "time_series", "group_based", "stratified"}


def _semantic_checks(output: dict, agent_name: str) -> list[str]:
    """Domain-specific semantic validation beyond just types."""
    errors = []

    if agent_name == "orchestrator_agent":
        verdict = output.get("loop_verdict", "")
        if verdict not in _VALID_LOOP_VERDICTS:
            errors.append(f"loop_verdict '{verdict}' not in {_VALID_LOOP_VERDICTS}")

    if agent_name == "split_agent":
        strat = output.get("split_strategy", "")
        if strat not in _VALID_SPLIT_STRATEGIES:
            errors.append(f"split_strategy '{strat}' not in {_VALID_SPLIT_STRATEGIES}")
        test_size = output.get("test_size", 0.0)
        try:
            if not (0.05 <= float(test_size) <= 0.5):
                errors.append(f"test_size {test_size} outside [0.05, 0.5]")
        except (TypeError, ValueError):
            pass  # type error already caught by _check_field

    if agent_name == "model_agent":
        cv_score = output.get("best_cv_score", 0.0)
        if not (-1.0 <= cv_score <= 1.0):
            errors.append(f"best_cv_score {cv_score} outside [-1, 1]")

    if agent_name == "feature_agent":
        selected = output.get("selected_features", [])
        if len(selected) == 0:
            errors.append("selected_features is empty — at least one feature required")

    return errors


def validate_agent_output(output: dict, agent_name: str, strict: bool = False) -> list[str]:
    """
    Validate agent output against its contract.

    Args:
        output:     The dict returned by the agent.
        agent_name: Name key in _CONTRACTS.
        strict:     If True, raise SchemaViolation on any error.
                    If False, return list of error strings.

    Returns:
        List of violation strings (empty → valid).
    """
    if agent_name not in _CONTRACTS:
        return []   # No contract registered → pass through

    violations: list[str] = []
    for spec in _CONTRACTS[agent_name]:
        violations.extend(_check_field(
            output,
            field=spec["field"],
            expected_type=spec["type"],
            required=spec.get("required", True),
            nullable=spec.get("nullable", False),
        ))

    violations.extend(_semantic_checks(output, agent_name))

    if strict and violations:
        raise SchemaViolation(
            f"[{agent_name}] Contract violations:\n  "
            + "\n  ".join(violations)
        )
    return violations


def validate_output(agent_name: str, strict: bool = False):
    """
    Decorator: validates the dict returned by an agent function.
    Violations are logged to agent_messages and (if strict) raise.

    Also sanitizes numpy scalar types (float64, int64, bool_, ndarray) so that
    LangGraph's msgpack checkpointer never raises a serialization error.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any, *args, **kwargs) -> dict:
            result = fn(state, *args, **kwargs)
            if isinstance(result, dict):
                # Sanitize numpy types so msgpack checkpointer never chokes
                from utils.serialization import sanitize_for_msgpack
                result = sanitize_for_msgpack(result)

                violations = validate_agent_output(result, agent_name, strict=strict)
                if violations:
                    msgs = result.setdefault("agent_messages", [])
                    msgs.append(
                        f"[{agent_name}] ⚠️ Schema warnings: "
                        + "; ".join(violations[:3])
                    )
            return result
        return wrapper
    return decorator


def register_contract(agent_name: str, fields: list[dict]) -> None:
    """Register a custom contract for an agent (useful for plugins)."""
    _CONTRACTS[agent_name] = fields
