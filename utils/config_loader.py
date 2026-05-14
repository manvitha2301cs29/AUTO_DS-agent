"""
utils/config_loader.py — Config-driven architecture (v5)

Loads config.yaml once at startup and exposes a typed Config object.
Supports:
  - YAML base config
  - config.local.yaml override (gitignored)
  - env-var overrides (AUTOML_<SECTION>_<KEY>=value)
  - dot-path accessor: cfg("pipeline.max_retries")

Usage:
    from utils.config_loader import cfg, reload_config

    max_retries = cfg("pipeline.max_retries")       # 3
    model_name  = cfg("llm.default_model")          # "gpt-4o-mini"
    cfg("missing.key", default=None)                # None (no KeyError)
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any

# ── YAML optional (falls back to {} if not installed) ────────────────────────
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_CONFIG_DIR = Path(__file__).resolve().parent.parent   # project root
_BASE_CONFIG  = _CONFIG_DIR / "config.yaml"
_LOCAL_CONFIG = _CONFIG_DIR / "config.local.yaml"

_CONFIG: dict = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides.
    Convention: AUTOML_PIPELINE_MAX_RETRIES → config["pipeline"]["max_retries"]
    """
    prefix = "AUTOML_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        if section in config and isinstance(config[section], dict):
            # Try to coerce type from existing value
            existing = config[section].get(field)
            try:
                if isinstance(existing, bool):
                    config[section][field] = val.lower() in ("1", "true", "yes")
                elif isinstance(existing, int):
                    config[section][field] = int(val)
                elif isinstance(existing, float):
                    config[section][field] = float(val)
                else:
                    config[section][field] = val
            except (ValueError, TypeError):
                config[section][field] = val
    return config


def _load_yaml(path: Path) -> dict:
    if not _HAS_YAML or not path.exists():
        return {}
    with open(path) as f:
        return _yaml.safe_load(f) or {}


def reload_config() -> dict:
    """Reload config from disk (useful in tests)."""
    global _CONFIG
    base   = _load_yaml(_BASE_CONFIG)
    local  = _load_yaml(_LOCAL_CONFIG)
    merged = _deep_merge(base, local)
    _CONFIG = _apply_env_overrides(merged)
    return _CONFIG


class _MISSING:
    """Sentinel for missing default in cfg()."""
    pass

_sentinel = _MISSING()


def cfg(dotpath: str, default: Any = _sentinel) -> Any:
    """
    Dot-path accessor into the loaded config.

    Examples:
        cfg("pipeline.max_retries")          # 3
        cfg("cv.n_splits")                   # 5
        cfg("missing.path", default=None)    # None
    """
    if not _CONFIG:
        reload_config()
    parts = dotpath.split(".")
    node: Any = _CONFIG
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            if isinstance(default, _MISSING):
                raise KeyError(f"Config key not found: {dotpath!r}")
            return default
        node = node[part]
    return node


def get_section(section: str) -> dict:
    """Return an entire config section as a dict."""
    if not _CONFIG:
        reload_config()
    return dict(_CONFIG.get(section, {}))


# Load on import
reload_config()
