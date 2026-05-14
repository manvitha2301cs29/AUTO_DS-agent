"""
utils/logger.py — Structured logging for AutoML pipeline (fix #19)

Provides a single get_logger() factory that returns a consistently
configured logger for each module/agent.

Features
--------
- JSON-structured output when LOG_FORMAT=json (default: human-readable)
- Log level controlled by LOG_LEVEL env var (default: INFO)
- Agent name included in every record
- Timestamps in ISO 8601

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Agent started", extra={"agent": "model_agent", "n_trials": 30})
    log.warning("LLM fallback used", extra={"reason": str(exc)})
    log.error("Pipeline error", exc_info=True)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment
# ─────────────────────────────────────────────────────────────────────────────
_LOG_LEVEL  = os.getenv("LOG_LEVEL",  "INFO").upper()
_LOG_FORMAT = os.getenv("LOG_FORMAT", "human").lower()   # "human" | "json"

_LEVEL_MAP = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_LEVEL = _LEVEL_MAP.get(_LOG_LEVEL, logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# JSON formatter
# ─────────────────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line, ready for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Include any extra= fields the caller added
        for key, val in record.__dict__.items():
            if key not in (
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "message",
            ):
                try:
                    json.dumps(val)   # only include JSON-serialisable extras
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable formatter
# ─────────────────────────────────────────────────────────────────────────────

_HUMAN_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT  = "%Y-%m-%dT%H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# Root handler setup (runs once)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_root() -> None:
    root = logging.getLogger("automl")
    if root.handlers:
        return   # already configured
    root.setLevel(_LEVEL)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_LEVEL)
    if _LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)
    root.propagate = False


_configure_root()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'automl' namespace.

    Parameters
    ----------
    name : typically __name__ of the calling module
           e.g. 'agents.model_agent' → logger 'automl.agents.model_agent'
    """
    # Strip leading package prefix if caller passes __name__
    if not name.startswith("automl"):
        name = f"automl.{name}"
    return logging.getLogger(name)
