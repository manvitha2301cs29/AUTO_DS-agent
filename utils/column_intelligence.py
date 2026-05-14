"""
utils/column_intelligence.py

Two capabilities:
  1. column_descriptions(df, dataset_name, api_key) → dict[col, {meaning, data_type_label, example_values}]
     Uses an LLM to explain every column in plain English, grounded in the
     actual data samples and dataset name/domain.

  2. validate_target(df, target, problem_type) → TargetValidation
     Rule-based + LLM-assisted check for:
       • Type mismatch  (categorical target + regression task → warning)
       • Unusable target (high-cardinality free text, names, IDs → error)
       • Constant / near-constant target (zero variance → error)
       • Too many classes for regression target (>50 unique numeric values, ok; object dtype → warn)
"""
from __future__ import annotations

import os
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.agent_utils import call_llm_json
from utils.logger import get_logger

log = get_logger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _model() -> str:
    return os.getenv("EDA_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _sample_values(series: pd.Series, n: int = 5) -> list:
    """Return up to n representative non-null sample values (no duplicates)."""
    vals = series.dropna().unique()
    chosen = vals[:n].tolist()
    # Convert numpy types to plain python for JSON serialisation
    return [v.item() if hasattr(v, "item") else v for v in chosen]


def _build_col_payload(df: pd.DataFrame) -> list[dict]:
    """Compact column snapshot for the LLM prompt."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        entry: dict[str, Any] = {
            "name":         col,
            "dtype":        str(s.dtype),
            "null_pct":     round(float(s.isna().mean() * 100), 1),
            "n_unique":     int(s.nunique()),
            "sample_values": _sample_values(s),
        }
        if is_num:
            entry["min"] = round(float(s.min()), 4) if s.notna().any() else None
            entry["max"] = round(float(s.max()), 4) if s.notna().any() else None
        rows.append(entry)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Column descriptions
# ─────────────────────────────────────────────────────────────────────────────

_COL_DESC_SYSTEM = """You are a senior data scientist and domain expert.
You receive a dataset name and a JSON list of column snapshots.
Your job: explain every column in plain English so that a non-technical person
can understand what it means and why it exists in the dataset.

Rules:
- Use domain knowledge inferred from the dataset name (e.g. "Titanic" → maritime disaster).
- For each column write a concise plain-English meaning (1-2 sentences max).
- Assign a human-readable data type label: one of:
    "Unique ID", "Numeric (continuous)", "Numeric (count/integer)",
    "Categorical (binary)", "Categorical (ordinal)", "Categorical (nominal)",
    "Text / Name", "Date / Time", "Boolean", "Unknown"
- List 3 real example values exactly as they appear in the sample_values field.
- If a column name is an abbreviation or jargon term, spell it out.

Output ONLY valid JSON — no markdown, no comments.
Schema:
{
  "columns": [
    {
      "name": "<exact column name>",
      "meaning": "<plain English explanation>",
      "type_label": "<one of the labels above>",
      "examples": ["val1", "val2", "val3"]
    },
    ...
  ]
}
"""

def column_descriptions(
    df: pd.DataFrame,
    dataset_name: str,
    api_key: str,
) -> dict[str, dict]:
    """
    Returns {col_name: {"meaning": str, "type_label": str, "examples": list}}
    Falls back to a rule-based description if the LLM call fails.
    """
    payload = _build_col_payload(df)
    user_content = (
        f"Dataset name: {dataset_name}\n"
        f"Number of rows: {len(df):,}\n\n"
        f"Columns:\n{json.dumps(payload, default=str)}"
    )

    result, err = call_llm_json(
        api_key=api_key,
        model_name=_model(),
        system_prompt=_COL_DESC_SYSTEM,
        user_content=user_content,
        temperature=0.1,
        max_tokens=2000,
        max_attempts=1,
        request_timeout=45,
    )

    if err or not result or "columns" not in result:
        log.warning(f"column_descriptions LLM failed ({err}), using fallback")
        return _fallback_descriptions(df)

    out: dict[str, dict] = {}
    for item in result.get("columns", []):
        name = item.get("name", "")
        if name in df.columns:
            out[name] = {
                "meaning":    item.get("meaning", "—"),
                "type_label": item.get("type_label", "Unknown"),
                "examples":   item.get("examples", _sample_values(df[name])),
            }

    # Fill any missing columns with fallback
    for col in df.columns:
        if col not in out:
            out[col] = _fallback_descriptions(df[[col]])[col]

    return out


def _fallback_descriptions(df: pd.DataFrame) -> dict[str, dict]:
    """Rule-based fallback when LLM is unavailable."""
    out = {}
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        unique_ratio = s.nunique() / max(s.notna().sum(), 1)

        if unique_ratio > 0.95:
            type_label = "Unique ID"
            meaning = f"Appears to be a unique identifier (very high cardinality: {s.nunique()} unique values)."
        elif is_num:
            if s.nunique() <= 2:
                type_label = "Categorical (binary)"
            elif s.nunique() <= 10:
                type_label = "Numeric (count/integer)"
            else:
                type_label = "Numeric (continuous)"
            meaning = (
                f"Numeric column. Range: {s.min():.3g} – {s.max():.3g}. "
                f"Mean: {s.mean():.3g}."
            )
        else:
            n_unique = s.nunique()
            if n_unique == 2:
                type_label = "Categorical (binary)"
            elif n_unique <= 15:
                type_label = "Categorical (nominal)"
            else:
                type_label = "Text / Name"
            meaning = f"Categorical column with {n_unique} unique values."

        out[col] = {
            "meaning":    meaning,
            "type_label": type_label,
            "examples":   _sample_values(s, 3),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Target validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetValidation:
    """
    severity:  "ok" | "warning" | "error"
    issues:    list of issue dicts  {code, title, detail, suggestion}
    """
    severity: str = "ok"
    issues: list[dict] = field(default_factory=list)

    def add(self, severity: str, code: str, title: str, detail: str, suggestion: str):
        self.issues.append({
            "severity":   severity,
            "code":       code,
            "title":      title,
            "detail":     detail,
            "suggestion": suggestion,
        })
        # escalate overall severity
        if severity == "error":
            self.severity = "error"
        elif severity == "warning" and self.severity == "ok":
            self.severity = "warning"

    @property
    def blocking(self) -> bool:
        return self.severity == "error"


# Tokens that strongly indicate a column is a free-text name / identifier
_NAME_TOKENS   = {"name", "firstname", "lastname", "surname", "fullname",
                  "email", "address", "street", "city", "phone", "url", "link",
                  "description", "comment", "note", "text", "title", "remark"}
_ID_TOKENS     = {"id", "uuid", "uid", "pk", "key", "serial", "ticket",
                  "invoice", "order", "passenger", "record", "index", "rownum"}
_DATE_TOKENS   = {"date", "time", "datetime", "timestamp", "created", "updated",
                  "year", "month", "day", "dob", "birth"}


def _col_tokens(col: str) -> set[str]:
    # Split on non-alpha and also on camelCase transitions (e.g. PassengerId → passenger, id)
    lower = col.lower()
    # Insert space before each uppercase→lowercase transition for camelCase
    import re as _re
    spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', col)
    return set(_re.findall(r"[a-z]+", spaced.lower()))


def validate_target(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    col_descriptions: dict | None = None,
) -> TargetValidation:
    """
    Comprehensive rule-based validation of the chosen target column + task type.
    col_descriptions: optional {col: {meaning, type_label}} dict from column_descriptions().
    """
    v = TargetValidation()

    if target not in df.columns:
        v.add("error", "MISSING_TARGET",
              "Target column not found",
              f"'{target}' does not exist in the dataset.",
              "Choose a column that exists in the uploaded CSV.")
        return v

    s = df[target]
    is_num = pd.api.types.is_numeric_dtype(s)
    n_unique = int(s.nunique())
    n_rows = len(df)
    null_pct = float(s.isna().mean() * 100)
    tokens = _col_tokens(target)
    unique_ratio = n_unique / max(s.notna().sum(), 1)

    # Use LLM type_label if available
    type_label = (col_descriptions or {}).get(target, {}).get("type_label", "")

    # ── 1. Constant / near-constant target ───────────────────────────────────
    if n_unique <= 1:
        v.add("error", "CONSTANT_TARGET",
              "Target has only one unique value",
              f"'{target}' contains {n_unique} unique value(s) — a model cannot learn anything.",
              "Choose a different target column that varies across rows.")
        return v  # no point running more checks

    if n_unique == 2 and problem_type == "regression":
        v.add("warning", "BINARY_AS_REGRESSION",
              "Binary column selected for regression",
              f"'{target}' has only 2 unique values ({sorted(s.dropna().unique().tolist())[:2]}). "
              "Regression on a binary target is non-standard and metrics will be misleading.",
              "Change the problem type to **Classification**.")

    # ── 2. High-null target ───────────────────────────────────────────────────
    if null_pct > 30:
        v.add("warning", "HIGH_NULL_TARGET",
              f"Target has {null_pct:.1f}% missing values",
              f"{null_pct:.1f}% of rows in '{target}' are null — these rows will be dropped before training.",
              "Consider imputing or filtering rows before uploading, or choose a different target.")

    # ── 3. Free-text / Name column (cannot be predicted) ─────────────────────
    is_name_col = bool(tokens & _NAME_TOKENS) or type_label in ("Text / Name",)
    is_id_col   = (
        (bool(tokens & _ID_TOKENS) or type_label == "Unique ID")
        and unique_ratio > 0.8
    )
    is_date_col = bool(tokens & _DATE_TOKENS) or type_label == "Date / Time"

    if is_id_col and unique_ratio > 0.8:
        v.add("error", "UNUSABLE_ID_TARGET",
              f"'{target}' looks like a unique identifier",
              f"Column name and cardinality ({n_unique:,} unique / {n_rows:,} rows = {unique_ratio:.0%}) "
              "suggest this is a row ID, not a predictable outcome.",
              "Choose a meaningful outcome column (e.g. Survived, Price, Churn) as the target.")

    elif is_name_col and not is_num:
        v.add("error", "UNUSABLE_NAME_TARGET",
              f"'{target}' appears to be a free-text name or description",
              f"Free-text fields like names, emails, or addresses cannot be predicted by a classification "
              f"or regression model — each value is essentially unique ({n_unique:,} unique values).",
              "Choose a structured outcome column as your target, such as a category, score, or label.")

    elif is_date_col:
        v.add("error", "UNUSABLE_DATE_TARGET",
              f"'{target}' appears to be a date/time column",
              "Date/time columns are inputs (features) for time-based models, not prediction targets, "
              "unless you are specifically doing time-series forecasting with a dedicated pipeline.",
              "Choose a numeric or categorical outcome as target. Use this date column as a feature.")

    elif not is_num and unique_ratio > 0.85 and n_unique > 50:
        v.add("error", "HIGH_CARDINALITY_TARGET",
              f"'{target}' has too many unique values to predict reliably",
              f"{n_unique:,} unique text values in {n_rows:,} rows ({unique_ratio:.0%} unique ratio). "
              "No model can generalise to this many distinct categories.",
              "Choose a target with fewer categories (≤50), or encode this column differently.")

    # ── 4. Type mismatch: categorical target + regression ────────────────────
    if not is_num and problem_type == "regression":
        top_vals = s.dropna().value_counts().head(4).index.tolist()
        v.add("warning", "TYPE_MISMATCH_REGRESSION",
              "Categorical target selected with Regression task",
              f"'{target}' contains text/categorical values "
              f"(e.g. {top_vals}) — you cannot fit a regression model on non-numeric labels.",
              "Change the problem type to **Classification**, or choose a numeric column as target.")

    # ── 5. Type mismatch: numeric target + classification with many classes ───
    if is_num and problem_type == "classification" and n_unique > 50:
        v.add("warning", "TYPE_MISMATCH_CLASSIFICATION",
              "Continuous numeric target selected with Classification task",
              f"'{target}' has {n_unique} unique numeric values — treating each as a separate class "
              "will produce a model with extremely poor generalisation.",
              "Change the problem type to **Regression**, or bin the target into discrete ranges first.")

    # ── 6. Regression on very low-cardinality numeric ─────────────────────────
    if is_num and problem_type == "regression" and n_unique <= 5:
        v.add("warning", "LOW_CARDINALITY_REGRESSION",
              f"Target has only {n_unique} unique values — consider Classification",
              f"'{target}' only takes {n_unique} distinct values "
              f"({sorted(s.dropna().unique().tolist()[:5])}). "
              "Regression on very few discrete values is usually less effective than classification.",
              "Consider switching to **Classification** unless the values are truly continuous.")

    return v


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AI vs User description comparison
# ─────────────────────────────────────────────────────────────────────────────

_COMPARE_SYSTEM = """You are a data dictionary expert.
You are given a column from a dataset with two descriptions:
  - "ai_description": generated automatically by an AI
  - "user_description": written by the human who owns the dataset

Your job:
1. Evaluate both descriptions for accuracy, clarity, and usefulness.
2. Pick the BETTER one (or synthesise a new one if neither is great).
3. Return a short recommendation explaining WHY you chose or synthesised.

Output ONLY valid JSON:
{
  "recommendation": "ai" | "user" | "synthesised",
  "recommended_text": "<the best description text>",
  "reasoning": "<1-2 sentences why this is better>"
}
"""


def compare_descriptions(
    col: str,
    ai_desc: str,
    user_desc: str,
    col_sample: dict,
    api_key: str,
) -> dict:
    """
    Ask the LLM to compare an AI-generated vs user-written description
    and recommend the better one (or a synthesis).

    Returns:
        {
          "recommendation": "ai" | "user" | "synthesised",
          "recommended_text": str,
          "reasoning": str,
        }
    Falls back to keeping the user description if the LLM call fails.
    """
    user_content = (
        f"Column name: {col}\n"
        f"Data type: {col_sample.get('dtype', 'unknown')}\n"
        f"Sample values: {col_sample.get('sample_values', [])}\n"
        f"Unique count: {col_sample.get('n_unique', '?')}\n\n"
        f"ai_description: {ai_desc}\n"
        f"user_description: {user_desc}"
    )

    result, err = call_llm_json(
        api_key=api_key,
        model_name=_model(),
        system_prompt=_COMPARE_SYSTEM,
        user_content=user_content,
        temperature=0.1,
        max_tokens=300,
    )

    if err or not result:
        log.warning(f"compare_descriptions failed for '{col}': {err}")
        return {
            "recommendation": "user",
            "recommended_text": user_desc,
            "reasoning": "LLM comparison unavailable — keeping your description.",
        }

    return {
        "recommendation": result.get("recommendation", "user"),
        "recommended_text": result.get("recommended_text", user_desc),
        "reasoning": result.get("reasoning", ""),
    }
