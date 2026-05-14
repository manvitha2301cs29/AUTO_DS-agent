"""
agents/leakage_agent.py — Leakage Detection Agent

Detects three categories of data leakage before model training:
  1. Target leakage   — features with suspiciously high correlation/mutual-info with target
  2. Time leakage     — datetime features used in a non-time-series context
  3. ID leakage       — high-cardinality identifier columns that should be dropped

Hybrid rule-based + LLM approach:
  - Rule-based scoring runs first (fast, deterministic, no API cost)
  - LLM reviews flagged columns and writes a human-readable risk report

Results written to:
  state["leakage_report"]          — full structured report
  state["preprocessing_decisions"] — dropped columns added with strategy="drop"
"""

from __future__ import annotations
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json
from utils.logger import get_logger
from utils.stats_safety import safe_corr

log = get_logger(__name__)
from utils.ml_helpers import detect_target_derived_columns
from utils.serialization import b64_to_df


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based scorers
# ─────────────────────────────────────────────────────────────────────────────

def _target_leakage_scores(df: pd.DataFrame, target: str, problem_type: str) -> dict[str, float]:
    """Return leakage risk score [0, 1] per non-target column."""
    scores: dict[str, float] = {}
    y = df[target]

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        score = 0.0

        # Pearson correlation (numeric only)
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
            try:
                corr = safe_corr(s, y)
                if corr is not None:
                    corr = abs(corr)
                    score = max(score, corr)
            except Exception:
                pass

        # Mutual information
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            s_filled = s.fillna(
                s.median() if pd.api.types.is_numeric_dtype(s) else s.mode().iloc[0]
            )
            if not pd.api.types.is_numeric_dtype(s_filled):
                s_filled = s_filled.astype("category").cat.codes
            X_tmp = s_filled.values.reshape(-1, 1)
            if problem_type == "classification":
                y_enc = y.astype("category").cat.codes if not pd.api.types.is_numeric_dtype(y) else y
                mi = mutual_info_classif(X_tmp, y_enc.values, random_state=42)[0]
            else:
                mi = mutual_info_regression(X_tmp, y.values, random_state=42)[0]
            mi_norm = min(1.0, mi / (np.log(df[target].nunique() + 1) + 1e-9))
            score = max(score, mi_norm)
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

        scores[col] = round(score, 4)
    return scores


def _id_leakage_flags(df: pd.DataFrame, target: str) -> dict[str, bool]:
    """Flag high-cardinality identifier columns."""
    flags: dict[str, bool] = {}
    n = len(df)
    id_keywords = {"id", "uuid", "key", "index", "rownum", "record", "pk", "uid"}
    for col in df.columns:
        if col == target:
            continue
        unique_ratio = df[col].nunique() / max(n, 1)
        name_lower = col.lower().replace("_", "").replace("-", "")
        name_hint = any(kw in name_lower for kw in id_keywords)
        flags[col] = unique_ratio > 0.95 or (unique_ratio > 0.80 and name_hint)
    return flags


def _time_leakage_flags(df: pd.DataFrame, target: str, split_strategy: str) -> dict[str, str]:
    """Flag datetime columns that may leak temporal information."""
    flags: dict[str, str] = {}
    if split_strategy == "time_series":
        return flags
    dt_keywords = {"date", "time", "ts", "timestamp", "created", "updated", "modified"}
    for col in df.columns:
        if col == target:
            continue
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        name_lower = col.lower().replace("_", "").replace("-", "")
        name_hint = any(kw in name_lower for kw in dt_keywords)
        if is_datetime:
            flags[col] = "datetime dtype — may encode future information if not split temporally"
        elif name_hint and df[col].dtype in ("object",):
            flags[col] = "name suggests timestamp — verify it does not encode post-event data"
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a data-leakage expert reviewing an automated ML pipeline.

Given rule-based leakage risk scores and flags for each feature, you must:
1. Confirm or dismiss each HIGH-risk finding (score ≥ 0.7) — high correlation
   can be legitimate if the feature genuinely predates the target event.
2. Review medium-risk columns (score 0.4–0.7) that look suspicious by name.
3. Recommend "drop" or "keep" for each flagged column with a one-sentence rationale.
4. Write a 2-3 sentence overall leakage risk summary.

Return ONLY valid JSON — no markdown:
{
  "column_verdicts": {
    "<col>": {"verdict": "drop|keep", "reason": "<one sentence>"}
  },
  "leakage_summary": "<2-3 sentence summary>",
  "overall_risk": "low|medium|high"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent entry point
# ─────────────────────────────────────────────────────────────────────────────

@agent_error_handler("Leakage Agent")
def leakage_agent(state: PipelineState) -> dict:
    log.info("Leakage agent started", extra={"target": state.get("target")})
    api_key = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model   = os.getenv("LEAKAGE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    df_key = "df_engineered_parquet_b64" if state.get("df_engineered_parquet_b64") else "df_parquet_b64"
    df           = b64_to_df(state[df_key])
    target       = state["target"]
    problem_type = state["problem_type"]
    split_strategy = state.get("split_strategy", "standard")

    target_scores = _target_leakage_scores(df, target, problem_type)
    id_flags      = _id_leakage_flags(df, target)
    time_flags    = _time_leakage_flags(df, target, split_strategy)
    target_derived = detect_target_derived_columns(df, target)

    for col in target_derived:
        target_scores[col] = max(float(target_scores.get(col, 0.0)), 0.99)

    high_risk   = {c: s for c, s in target_scores.items() if s >= 0.70}
    medium_risk = {c: s for c, s in target_scores.items() if 0.40 <= s < 0.70}
    id_risk     = {c for c, v in id_flags.items() if v}
    time_risk   = set(time_flags.keys())
    all_flagged = set(high_risk) | set(medium_risk) | id_risk | time_risk

    payload: dict[str, Any] = {
        "target": target, "problem_type": problem_type,
        "split_strategy": split_strategy, "n_rows": len(df),
        "high_target_leakage":   {c: {"score": s} for c, s in high_risk.items()},
        "medium_target_leakage": {c: {"score": s} for c, s in medium_risk.items()},
        "target_derived_candidates": target_derived,
        "id_leakage_candidates": list(id_risk),
        "time_leakage_candidates": time_flags,
    }

    llm_verdicts: dict[str, dict] = {}
    leakage_summary = "No significant leakage signals detected by rule-based analysis."
    overall_risk = "low"

    if all_flagged:
        _result, _llm_err = call_llm_json(
            api_key=api_key, model_name=model,
            system_prompt=SYSTEM_PROMPT,
            user_content=json.dumps(payload, indent=2, default=str),
            temperature=0.1, max_tokens=1200,
        )
        if _result is not None:
            llm_verdicts    = _result.get("column_verdicts", {})
            leakage_summary = _result.get("leakage_summary", leakage_summary)
            overall_risk    = _result.get("overall_risk", "medium")
        else:
            leakage_summary = f"LLM assessment failed: {_llm_err}. Applying conservative rules."
            overall_risk = "medium"
            for c in high_risk:
                llm_verdicts[c] = {"verdict": "drop", "reason": "High target correlation (rule-based fallback)"}
            for c in id_risk:
                llm_verdicts[c] = {"verdict": "drop", "reason": "High-cardinality ID column"}

    for col, reason in target_derived.items():
        llm_verdicts[col] = {"verdict": "drop", "reason": reason}
        overall_risk = "high"

    existing_decisions: dict = dict(state.get("preprocessing_decisions") or {})
    dropped_by_leakage: list[str] = []

    for col, verdict_info in llm_verdicts.items():
        if verdict_info.get("verdict") == "drop":
            existing_decisions[col] = {
                "strategy":  "drop",
                "rationale": f"[Leakage Agent] {verdict_info.get('reason', 'leakage detected')}",
            }
            dropped_by_leakage.append(col)

    for c in id_risk:
        if c not in llm_verdicts:
            existing_decisions[c] = {
                "strategy":  "drop",
                "rationale": "[Leakage Agent] High-cardinality identifier column",
            }
            dropped_by_leakage.append(c)

    leakage_report = {
        "overall_risk":       overall_risk,
        "leakage_summary":    leakage_summary,
        "high_risk_columns":  list(high_risk.keys()),
        "medium_risk_columns": list(medium_risk.keys()),
        "target_derived_columns": list(target_derived.keys()),
        "id_risk_columns":    list(id_risk),
        "time_risk_columns":  list(time_risk),
        "dropped_by_leakage": dropped_by_leakage,
        "target_scores":      target_scores,
        "llm_verdicts":       llm_verdicts,
        "n_flagged":          len(all_flagged),
        "n_dropped":          len(dropped_by_leakage),
    }

    return {
        "leakage_report":          leakage_report,
        "preprocessing_decisions": existing_decisions,
        "agent_messages": [
            f"[Leakage Agent] Overall risk: {overall_risk}. "
            f"Flagged {len(all_flagged)} columns, dropped {len(dropped_by_leakage)}. "
            f"{leakage_summary[:80]}…"
        ],
    }
