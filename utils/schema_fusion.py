"""
utils/schema_fusion.py — Multi-table CSV schema analysis and join execution.

Provides:
  - analyse_schema(tables, api_key)  → list of detected relationships
  - execute_joins(tables, join_plan) → merged DataFrame + quality report
  - execute_sql_query(tables, sql)   → merged DataFrame via DuckDB
"""
from __future__ import annotations

import os
import json
import re
from typing import Any

import pandas as pd

from utils.agent_utils import call_llm_json
from utils.logger import get_logger

log = get_logger(__name__)

MAX_JOIN_OUTPUT_ROWS = int(os.getenv("MAX_JOIN_OUTPUT_ROWS", "5000000"))
try:
    from numpy._core._exceptions import _ArrayMemoryError
except Exception:
    _ArrayMemoryError = MemoryError


def _model() -> str:
    return os.getenv("EDA_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ─────────────────────────────────────────────────────────────────────────────
# Schema snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def _table_snapshot(
    name: str,
    df: pd.DataFrame,
    max_sample: int = 5,
    column_descriptions: dict | None = None,
) -> dict:
    descriptions = column_descriptions or {}
    cols = []
    for col in df.columns:
        s        = df[col]
        is_num   = pd.api.types.is_numeric_dtype(s)
        n_unique = int(s.nunique())
        n_null   = int(s.isna().sum())
        samples  = s.dropna().unique()[:max_sample].tolist()
        samples  = [v.item() if hasattr(v, "item") else v for v in samples]
        cols.append({
            "name":         col,
            "dtype":        str(s.dtype),
            "n_unique":     n_unique,
            "null_pct":     round(n_null / max(len(df), 1) * 100, 1),
            "unique_ratio": round(n_unique / max(len(df), 1), 3),
            "sample":       samples,
            "is_numeric":   is_num,
            "meaning":      descriptions.get(col, {}).get("meaning", ""),
            "type_label":   descriptions.get(col, {}).get("type_label", ""),
        })
    return {
        "table":   name,
        "n_rows":  len(df),
        "n_cols":  len(df.columns),
        "columns": cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM schema analysis
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SYSTEM = """You are a database architect and data engineer expert.

You receive JSON snapshots of multiple CSV tables (name, row count, column
metadata including unique ratios, sample values, and reviewed plain-English
column descriptions).

Your job:
1. Detect foreign key relationships between tables by matching column names,
   data types, value overlap patterns, and domain knowledge.
2. Propose a join plan — ordered list of merge operations to produce one
   unified table.
3. Assign a join type to each merge: "inner", "left", or "outer".
4. For each relationship give a confidence score 0.0–1.0.

Additional goal: propose the optimal join plan for a single modelling dataset.
Prefer a fact/event table as the base table when one exists, such as claims,
transactions, orders, visits, payments, events, or encounters. Join dimension
tables such as patients, providers, customers, products, and locations onto that
base table. Avoid many-to-many joins; if both sides of a key look non-unique,
mention the risk and prefer a more unique key or a bridge/fact table.
The Human join goal / HITL constraints are authoritative. If the human selected
a base table and requested row preservation, the first join must start from that
base table, all later joins should keep the merged result on the left, and the
final row count must equal the selected base table row count. Do not use INNER
joins from the base table unless the human explicitly allows dropping base rows.

Rules:
- A column with unique_ratio ≈ 1.0 in one table is likely a primary key.
- A column with same name / similar name in another table is likely a foreign key.
- If a primary key column has the same name as a foreign key in another table,
  that is a direct match (high confidence).
- Consider semantic similarity: customer_id ↔ cust_id ↔ CustomerID are the same.
- Prefer LEFT joins unless you are sure all rows in both tables have a match
  (inner), or you need all rows regardless (outer).
- join_order: list merge operations in the order they should be executed.
  Each step takes either an original table or the result of a previous merge.

Output ONLY valid JSON, no markdown:
{
  "relationships": [
    {
      "table_a": "<table name>",
      "key_a":   "<column in table_a>",
      "table_b": "<table name>",
      "key_b":   "<column in table_b>",
      "join_type": "inner" | "left" | "outer",
      "confidence": 0.0-1.0,
      "reasoning": "<one sentence>"
    }
  ],
  "join_order": [
    {
      "step": 1,
      "left":  "<table name or 'merged_so_far'>",
      "right": "<table name>",
      "left_key":  "<column>",
      "right_key": "<column>",
      "join_type": "inner" | "left" | "outer",
      "description": "<short human-readable description>"
    }
  ],
  "summary": "<2-3 sentences describing the overall schema and recommended strategy>"
}
"""


def analyse_schema(
    tables: dict[str, pd.DataFrame],
    api_key: str,
    column_descriptions: dict[str, dict[str, dict]] | None = None,
    join_goal: dict | None = None,
) -> dict:
    """
    Send all table schemas to the LLM and return detected relationships + join plan.

    Returns:
        {
          "relationships": [...],
          "join_order": [...],
          "summary": str,
          "error": str | None,
        }
    """
    column_descriptions = column_descriptions or {}
    join_goal = join_goal or {}
    snapshots = [
        _table_snapshot(name, df, column_descriptions=column_descriptions.get(name, {}))
        for name, df in tables.items()
    ]
    user_content = (
        f"Number of tables: {len(tables)}\n\n"
        f"Human join goal / HITL constraints:\n{json.dumps(join_goal, default=str, indent=2)}\n\n"
        f"Table schemas:\n{json.dumps(snapshots, default=str, indent=2)}"
    )

    result, err = call_llm_json(
        api_key=api_key,
        model_name=_model(),
        system_prompt=_SCHEMA_SYSTEM,
        user_content=user_content,
        temperature=0.1,
        max_tokens=2000,
    )

    if err or not result:
        log.warning(f"schema analysis failed: {err}")
        return {
            "relationships": [],
            "join_order":    [],
            "summary":       "Schema analysis failed — please define relationships manually.",
            "error":         str(err),
        }

    return {
        "relationships": result.get("relationships", []),
        "join_order":    result.get("join_order", []),
        "summary":       result.get("summary", ""),
        "error":         None,
    }


def relationship_diagnostics(
    tables: dict[str, pd.DataFrame],
    relationships: list[dict],
) -> list[dict]:
    """Validate inferred relationships against actual uploaded table values."""
    diagnostics: list[dict] = []
    for rel in relationships:
        table_a = rel.get("table_a")
        table_b = rel.get("table_b")
        key_a = rel.get("key_a")
        key_b = rel.get("key_b")

        if table_a not in tables or table_b not in tables:
            diagnostics.append({
                "relationship": f"{table_a}.{key_a} -> {table_b}.{key_b}",
                "status": "missing_table",
                "message": "One or both tables are missing.",
            })
            continue

        df_a, df_b = tables[table_a], tables[table_b]
        if key_a not in df_a.columns or key_b not in df_b.columns:
            diagnostics.append({
                "relationship": f"{table_a}.{key_a} -> {table_b}.{key_b}",
                "status": "missing_key",
                "message": "One or both key columns are missing.",
            })
            continue

        a = df_a[key_a].dropna()
        b = df_b[key_b].dropna()
        a_unique = int(a.nunique())
        b_unique = int(b.nunique())
        a_rows = int(len(df_a))
        b_rows = int(len(df_b))
        a_nulls = int(df_a[key_a].isna().sum())
        b_nulls = int(df_b[key_b].isna().sum())
        a_is_unique = a_unique == len(a) and a_nulls == 0
        b_is_unique = b_unique == len(b) and b_nulls == 0

        a_vals = set(a.unique())
        b_vals = set(b.unique())
        a_in_b = len(a_vals & b_vals) / max(len(a_vals), 1)
        b_in_a = len(a_vals & b_vals) / max(len(b_vals), 1)

        if a_is_unique and b_is_unique:
            cardinality = "one-to-one / optional one-to-one"
        elif a_is_unique and not b_is_unique:
            cardinality = f"one-to-many ({table_a} -> {table_b})"
        elif not a_is_unique and b_is_unique:
            cardinality = f"many-to-one ({table_a} -> {table_b})"
        else:
            cardinality = "many-to-many risk"

        pk_candidates = []
        if a_is_unique:
            pk_candidates.append(f"{table_a}.{key_a}")
        if b_is_unique:
            pk_candidates.append(f"{table_b}.{key_b}")

        if a_is_unique and b_in_a >= 0.98:
            fk_statement = f"{table_b}.{key_b} can act as FK to {table_a}.{key_a}"
        elif b_is_unique and a_in_b >= 0.98:
            fk_statement = f"{table_a}.{key_a} can act as FK to {table_b}.{key_b}"
        elif a_in_b >= 0.98 or b_in_a >= 0.98:
            fk_statement = "High key overlap, but PK side is not fully unique."
        else:
            fk_statement = "Weak FK coverage; verify this relationship manually."

        severity = "ok"
        if "many-to-many" in cardinality:
            severity = "error"
        elif min(a_in_b, b_in_a) < 0.98 and not (a_in_b >= 0.98 or b_in_a >= 0.98):
            severity = "warning"

        diagnostics.append({
            "relationship": f"{table_a}.{key_a} -> {table_b}.{key_b}",
            "cardinality": cardinality,
            "pk_candidates": ", ".join(pk_candidates) if pk_candidates else "none",
            "fk_check": fk_statement,
            "a_rows": a_rows,
            "b_rows": b_rows,
            "a_unique": a_unique,
            "b_unique": b_unique,
            "a_nulls": a_nulls,
            "b_nulls": b_nulls,
            "a_values_found_in_b_pct": round(a_in_b * 100, 1),
            "b_values_found_in_a_pct": round(b_in_a * 100, 1),
            "constraint_type": "Inferred from CSV values; not an actual DB constraint.",
            "status": severity,
        })

    return diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Join execution
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_key_types(df_left: pd.DataFrame, key_left: str,
                      df_right: pd.DataFrame, key_right: str
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try to coerce key columns to the same dtype before joining."""
    dl, dr = df_left.copy(), df_right.copy()
    tl, tr = str(dl[key_left].dtype), str(dr[key_right].dtype)
    if tl == tr:
        return dl, dr
    # Both numeric → cast to common float
    try:
        dl[key_left]  = pd.to_numeric(dl[key_left],  errors="coerce")
        dr[key_right] = pd.to_numeric(dr[key_right], errors="coerce")
        return dl, dr
    except Exception:
        pass
    # Otherwise cast both to str
    dl[key_left]  = dl[key_left].astype(str)
    dr[key_right] = dr[key_right].astype(str)
    return dl, dr


def _dedup_columns(df: pd.DataFrame, step_result: pd.DataFrame,
                   right_key: str) -> pd.DataFrame:
    """Drop redundant _x / _y duplicates introduced by pandas merge."""
    # Drop the right-side foreign key if it's identical to the left-side key
    drop_cols = [c for c in step_result.columns
                 if c.endswith("_y") and c[:-2] + "_x" in step_result.columns]
    for c in drop_cols:
        base = c[:-2]
        step_result = step_result.rename(columns={base + "_x": base})
        step_result = step_result.drop(columns=[c], errors="ignore")
    return step_result


def _fmt_int(n: int) -> str:
    return f"{int(n):,}"


def _estimate_join_rows(
    df_left: pd.DataFrame,
    left_key: str,
    df_right: pd.DataFrame,
    right_key: str,
    how: str,
) -> dict:
    """Estimate merge output rows without materializing the joined dataframe."""
    left_counts = df_left[left_key].dropna().value_counts()
    right_counts = df_right[right_key].dropna().value_counts()
    common_keys = left_counts.index.intersection(right_counts.index)

    matched_rows = int((left_counts.loc[common_keys] * right_counts.loc[common_keys]).sum())
    left_unmatched_rows = int(len(df_left) - left_counts.loc[common_keys].sum())
    right_unmatched_rows = int(len(df_right) - right_counts.loc[common_keys].sum())

    if how == "inner":
        estimated_rows = matched_rows
    elif how == "right":
        estimated_rows = matched_rows + right_unmatched_rows
    elif how == "outer":
        estimated_rows = matched_rows + left_unmatched_rows + right_unmatched_rows
    else:
        estimated_rows = matched_rows + left_unmatched_rows

    left_dup_keys = int((left_counts > 1).sum())
    right_dup_keys = int((right_counts > 1).sum())
    many_to_many_keys = int(((left_counts.loc[common_keys] > 1) & (right_counts.loc[common_keys] > 1)).sum())

    worst_key = None
    worst_rows = 0
    if len(common_keys):
        products = left_counts.loc[common_keys] * right_counts.loc[common_keys]
        worst_key = products.idxmax()
        worst_rows = int(products.max())

    return {
        "estimated_rows": estimated_rows,
        "matched_rows": matched_rows,
        "left_unmatched_rows": left_unmatched_rows,
        "right_unmatched_rows": right_unmatched_rows,
        "left_dup_keys": left_dup_keys,
        "right_dup_keys": right_dup_keys,
        "many_to_many_keys": many_to_many_keys,
        "worst_key": worst_key,
        "worst_rows": worst_rows,
    }


def execute_joins(
    tables: dict[str, pd.DataFrame],
    join_plan: list[dict],
    expected_rows: int | None = None,
    expected_rows_label: str = "selected base table",
) -> tuple[pd.DataFrame | None, list[dict], str | None]:
    """
    Execute a list of join steps in order.

    join_plan items: {left, right, left_key, right_key, join_type, description}

    Returns:
        (merged_df, quality_report, error_message)
        quality_report: list of {step, description, rows_before, rows_after,
                                  nulls_introduced, key_mismatches}
    """
    registry: dict[str, pd.DataFrame] = dict(tables)
    merged: pd.DataFrame | None = None
    report: list[dict] = []

    for i, step in enumerate(join_plan):
        left_name  = step.get("left",  "merged_so_far")
        right_name = step.get("right", "")
        left_key   = step.get("left_key",  "")
        right_key  = step.get("right_key", "")
        join_type  = step.get("join_type", "left")
        desc       = step.get("description", f"Step {i+1}")

        # Resolve left table
        if left_name == "merged_so_far" and merged is not None:
            df_left = merged
        elif left_name in registry:
            df_left = registry[left_name]
        else:
            return None, report, f"Step {i+1}: left table '{left_name}' not found."

        # Resolve right table
        if right_name not in registry:
            return None, report, f"Step {i+1}: right table '{right_name}' not found."
        df_right = registry[right_name]

        # Validate keys
        if left_key not in df_left.columns:
            return None, report, (
                f"Step {i+1}: key '{left_key}' not found in "
                f"{'merged result' if left_name == 'merged_so_far' else left_name}. "
                f"Available: {list(df_left.columns[:10])}"
            )
        if right_key not in df_right.columns:
            return None, report, (
                f"Step {i+1}: key '{right_key}' not found in '{right_name}'. "
                f"Available: {list(df_right.columns[:10])}"
            )

        rows_before = len(df_left)

        # Coerce types
        df_left, df_right = _coerce_key_types(df_left, left_key, df_right, right_key)

        # Key mismatch analysis
        left_vals  = set(df_left[left_key].dropna().unique())
        right_vals = set(df_right[right_key].dropna().unique())
        unmatched_left  = len(left_vals - right_vals)
        unmatched_right = len(right_vals - left_vals)

        # Perform merge
        how_map = {"inner": "inner", "left": "left", "outer": "outer", "right": "right"}
        how = how_map.get(join_type, "left")

        estimate = _estimate_join_rows(df_left, left_key, df_right, right_key, how)
        estimated_rows = estimate["estimated_rows"]
        if estimated_rows > MAX_JOIN_OUTPUT_ROWS:
            worst_key_msg = ""
            if estimate["worst_key"] is not None:
                worst_key_msg = (
                    f" The largest matching key value ({estimate['worst_key']!r}) alone "
                    f"would create {_fmt_int(estimate['worst_rows'])} rows."
                )
            return None, report, (
                f"Step {i+1}: this join would create about {_fmt_int(estimated_rows)} rows, "
                f"which is above the safety limit of {_fmt_int(MAX_JOIN_OUTPUT_ROWS)} rows. "
                f"The selected keys look many-to-many: {_fmt_int(estimate['left_dup_keys'])} duplicate-key "
                f"values on the left, {_fmt_int(estimate['right_dup_keys'])} on the right, and "
                f"{_fmt_int(estimate['many_to_many_keys'])} duplicated key values on both sides."
                f"{worst_key_msg} Choose a unique ID/key, aggregate or deduplicate one table first, "
                "or use SQL to filter the rows before joining."
            )

        try:
            merged = df_left.merge(
                df_right,
                left_on=left_key,
                right_on=right_key,
                how=how,
                suffixes=("", f"_{right_name}"),
            )
        except (MemoryError, _ArrayMemoryError):
            return None, report, (
                f"Step {i+1}: the join ran out of memory while creating about "
                f"{_fmt_int(estimated_rows)} rows. The selected keys likely create a many-to-many join. "
                "Choose a more unique key, deduplicate/aggregate one side, or filter rows with SQL first."
            )

        # Clean up duplicate key column if left_key != right_key
        if left_key != right_key and f"{right_key}_{right_name}" in merged.columns:
            merged = merged.drop(columns=[f"{right_key}_{right_name}"], errors="ignore")
        elif left_key != right_key and right_key in merged.columns:
            merged = merged.drop(columns=[right_key], errors="ignore")

        # Register result for next step
        registry["merged_so_far"] = merged

        rows_after       = len(merged)
        nulls_introduced = int(merged.isna().sum().sum())

        report.append({
            "step":              i + 1,
            "description":       desc,
            "join_type":         join_type,
            "left_key":          left_key,
            "right_key":         right_key,
            "rows_before":       rows_before,
            "rows_after":        rows_after,
            "row_delta":         rows_after - rows_before,
            "nulls_introduced":  nulls_introduced,
            "unmatched_left":    unmatched_left,
            "unmatched_right":   unmatched_right,
        })

    if expected_rows is not None and merged is not None and len(merged) != expected_rows:
        return None, report, (
            f"Final dataset row count is {_fmt_int(len(merged))}, but the HITL grain constraint "
            f"expects {_fmt_int(expected_rows)} rows from {expected_rows_label}. "
            "Review the join order/types and avoid joins that drop or duplicate base rows."
        )

    return merged, report, None


# ─────────────────────────────────────────────────────────────────────────────
# SQL query execution via DuckDB
# ─────────────────────────────────────────────────────────────────────────────

def execute_sql_query(
    tables: dict[str, pd.DataFrame],
    sql: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Execute a SQL query against the uploaded tables using DuckDB.
    Each table is registered by its filename (without .csv extension).

    Returns (result_df, error_message).
    """
    try:
        import duckdb
    except ImportError:
        return None, "DuckDB not installed. Run: pip install duckdb"

    try:
        con = duckdb.connect(database=":memory:")
        for name, df in tables.items():
            # Register with clean name (no spaces, no .csv)
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            con.register(safe_name, df)

        result = con.execute(sql).df()
        con.close()
        return result, None
    except Exception as e:
        return None, str(e)
