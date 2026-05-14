"""
agents/eda_agent.py — EDA Agent (v3)

New in v3:
  - Class imbalance analysis: detects imbalance ratio, flags severe imbalance
  - Loss function recommendations: focal loss, weighted CE, class_weight for imbalanced data
  - Split strategy recommendations: stratified split for imbalanced classification
  - Recommendations surfaced to state for split page and model_agent
"""

from __future__ import annotations
import json
import os
import pandas as pd
from typing import Any

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json, truncate_prompt_metadata
from utils.logger import get_logger

log = get_logger(__name__)
from utils.serialization import b64_to_df
from utils.ml_helpers import detect_identifier_like_columns, detect_target_derived_columns


def compute_column_stats(df: pd.DataFrame, target: str) -> list[dict]:
    stats = []
    for col in df.columns:
        s = df[col]
        non_null = max(int(s.notna().sum()), 1)
        unique = int(s.nunique())
        entry: dict[str, Any] = {
            "name":       col,
            "dtype":      str(s.dtype),
            "is_target":  col == target,
            "null_count": int(s.isna().sum()),
            "null_pct":   round(float(s.isna().mean() * 100), 2),
            "unique":     unique,
            "unique_ratio": round(float(unique / non_null), 4),
            "cardinality": "high" if unique > 50 else "medium" if unique > 10 else "low",
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            entry.update({
                "mean":         round(float(desc["mean"]), 4) if "mean" in desc else None,
                "std":          round(float(desc["std"]),  4) if "std"  in desc else None,
                "min":          round(float(desc["min"]),  4) if "min"  in desc else None,
                "max":          round(float(desc["max"]),  4) if "max"  in desc else None,
                "skew":         round(float(s.skew()),     4),
                "zero_variance": bool(s.std() == 0),
                "pct_negative":  round(float((s < 0).mean() * 100), 1),
            })
        else:
            entry["top_values"] = [str(v) for v in s.value_counts().head(5).index.tolist()]
        stats.append(entry)
    return stats


# ── v3: Imbalance analysis ────────────────────────────────────────────────────

def compute_imbalance_analysis(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    if problem_type != "classification":
        y = df[target].dropna()
        return {
            "problem_type": "regression",
            "target_mean": round(float(y.mean()), 4) if pd.api.types.is_numeric_dtype(y) else None,
            "target_std":  round(float(y.std()), 4) if pd.api.types.is_numeric_dtype(y) else None,
            "imbalance_ratio": None,
            "severity": "n/a",
        }

    y = df[target].dropna()
    counts = y.value_counts()
    n_classes = len(counts)
    total = len(y)

    class_distribution = {
        str(cls): {"count": int(cnt), "pct": round(float(cnt / total * 100), 2)}
        for cls, cnt in counts.items()
    }

    if n_classes < 2:
        return {
            "n_classes": n_classes,
            "class_distribution": class_distribution,
            "imbalance_ratio": None,
            "severity": "single_class",
        }

    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])
    imbalance_ratio = round(majority_count / max(minority_count, 1), 2)

    if imbalance_ratio >= 10:
        severity = "severe"
    elif imbalance_ratio >= 4:
        severity = "moderate"
    elif imbalance_ratio >= 1.5:
        severity = "mild"
    else:
        severity = "balanced"

    return {
        "n_classes": n_classes,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "severity": severity,
        "majority_class": str(counts.index[0]),
        "minority_class": str(counts.index[-1]),
        "minority_pct": round(minority_count / total * 100, 2),
        "n_samples": total,
    }


def rule_based_imbalance_recommendations(imbalance: dict, problem_type: str) -> dict:
    """Deterministic baseline recommendations — LLM refines these."""
    if problem_type != "classification":
        return {
            "recommended_split": "standard",
            "split_rationale": "Standard random split for regression tasks.",
            "recommended_loss": "mse",
            "loss_rationale": "MSE is the standard regression loss.",
            "class_weight": None,
            "smote_recommended": False,
            "imbalance_handling_summary": "Regression task — no class imbalance handling needed.",
        }

    severity = imbalance.get("severity", "balanced")
    ratio = imbalance.get("imbalance_ratio") or 1.0

    if severity == "balanced":
        return {
            "recommended_split": "stratified",
            "split_rationale": "Stratified split preserves class proportions.",
            "recommended_loss": "cross_entropy",
            "loss_rationale": "Standard cross-entropy for balanced classification.",
            "class_weight": None,
            "smote_recommended": False,
            "imbalance_handling_summary": "Dataset is balanced. Stratified split with standard cross-entropy.",
        }
    elif severity == "mild":
        return {
            "recommended_split": "stratified",
            "split_rationale": "Stratified split ensures minority class in both splits.",
            "recommended_loss": "weighted_cross_entropy",
            "loss_rationale": f"Mild imbalance ({ratio:.1f}:1) — class_weight='balanced' upweights minority.",
            "class_weight": "balanced",
            "smote_recommended": False,
            "imbalance_handling_summary": f"Mild imbalance ({ratio:.1f}:1). Stratified split + class_weight='balanced'.",
        }
    elif severity == "moderate":
        return {
            "recommended_split": "stratified",
            "split_rationale": "Stratified split is critical — random split risks no minority samples in test.",
            "recommended_loss": "weighted_cross_entropy",
            "loss_rationale": f"Moderate imbalance ({ratio:.1f}:1) — weighted loss prevents majority-class bias.",
            "class_weight": "balanced",
            "smote_recommended": True,
            "imbalance_handling_summary": f"Moderate imbalance ({ratio:.1f}:1). Stratified + class_weight='balanced'. Consider SMOTE.",
        }
    else:  # severe
        return {
            "recommended_split": "stratified",
            "split_rationale": "Severe imbalance — stratified split is mandatory for minority representation.",
            "recommended_loss": "focal_loss",
            "loss_rationale": f"Severe imbalance ({ratio:.1f}:1) — focal loss down-weights easy majority examples, focuses on hard minority cases.",
            "class_weight": "balanced",
            "smote_recommended": True,
            "imbalance_handling_summary": f"Severe imbalance ({ratio:.1f}:1). Mandatory: stratified + focal loss + class_weight='balanced'. Use SMOTE.",
        }


SYSTEM_PROMPT = """\
You are a senior data scientist performing automated exploratory data analysis.

Given column-level statistics, assign a preprocessing strategy to EVERY non-target column
and write brief reports.

Allowed strategies (ALL must be honoured by the downstream preprocessor):
  keep_as_is | drop | median_impute | mean_impute | knn_impute | mode_impute |
  winsorize | log_transform | label_encode | onehot_encode | standardize

Rules:
- Drop zero-variance columns and columns >70% null.
- Drop identifier-like columns and categorical columns whose cardinality is nearly equal
  to the sample count (for example passenger IDs, serial numbers, row keys, record codes).
- Numeric, skewed (|skew|>1): prefer log_transform if all positive, else winsorize.
- Numeric, normal: mean_impute + standardize if needed.
- Categorical low-cardinality (≤10 unique): onehot_encode.
- Categorical high-cardinality (>10 unique): label_encode.
- Never assign a strategy to the target column.
- No issues: keep_as_is.

IMBALANCE & SPLIT RECOMMENDATIONS — REQUIRED:
Analyse the class_distribution and imbalance_ratio provided, then fill in imbalance_recommendations.
- Split strategy options: standard | stratified | time_series | group_based
- Loss options: cross_entropy | weighted_cross_entropy | focal_loss | label_smoothing | mse
- For ANY imbalance (ratio > 1.5:1) in classification: recommend stratified split.
- Focal loss for severe imbalance (ratio >= 10:1).
- Weighted cross-entropy for moderate imbalance (ratio 4:1 – 10:1).
- Be specific about the ratio and WHY in the rationale.

Return ONLY valid JSON — no markdown:
{
  "decisions": {
    "<column_name>": {"strategy": "<strategy>", "rationale": "<one sentence>"}
  },
  "global_notes": "<2-3 sentence dataset quality summary>",
  "eda_report": "<4-6 sentence narrative on key patterns, risks, focus areas>",
  "imbalance_recommendations": {
    "recommended_split": "<standard|stratified|time_series|group_based>",
    "split_rationale": "<one sentence explaining why this split>",
    "recommended_loss": "<cross_entropy|weighted_cross_entropy|focal_loss|label_smoothing|mse>",
    "loss_rationale": "<one sentence explaining why this loss function>",
    "class_weight": "<balanced or null>",
    "smote_recommended": true or false,
    "imbalance_handling_summary": "<2 sentences summarising all recommendations>"
  }
}
"""


@agent_error_handler("EDA Agent")
def eda_agent(state: PipelineState) -> dict:
    log.info("EDA agent started", extra={"target": state.get("target"), "problem_type": state.get("problem_type")})
    api_key = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model   = os.getenv("EDA_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    df           = b64_to_df(state["df_parquet_b64"])
    target       = state["target"]
    problem_type = state["problem_type"]
    stats        = compute_column_stats(df, target)
    target_derived_drops  = detect_target_derived_columns(df, target)
    identifier_like_drops = detect_identifier_like_columns(df, target)

    # v3: imbalance analysis
    imbalance_analysis = compute_imbalance_analysis(df, target, problem_type)
    rule_recs = rule_based_imbalance_recommendations(imbalance_analysis, problem_type)
    log.info("Imbalance analysis", extra={
        "severity": imbalance_analysis.get("severity"),
        "ratio": imbalance_analysis.get("imbalance_ratio"),
    })

    _user_content = (
        f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols\n"
        f"Target: '{target}' | Problem type: {problem_type}\n\n"
        f"Class imbalance analysis:\n{json.dumps(imbalance_analysis, indent=2, default=str)}\n\n"
        f"Rule-based baseline recommendations (refine/improve these based on the data):\n"
        f"{json.dumps(rule_recs, indent=2, default=str)}\n\n"
        f"Column statistics:\n{json.dumps(truncate_prompt_metadata(stats), indent=2, default=str)}"
    )

    result, _llm_err = call_llm_json(
        api_key=api_key, model_name=model,
        system_prompt=SYSTEM_PROMPT,
        user_content=_user_content,
        temperature=0.1, max_tokens=3000,
    )

    if result is None:
        # Rule-based fallback
        fallback_decisions: dict = {}
        for s in stats:
            col = s["name"]
            if s.get("is_target"):
                continue
            if col in target_derived_drops:
                fallback_decisions[col] = {"strategy": "drop", "rationale": target_derived_drops[col]}
            elif col in identifier_like_drops:
                fallback_decisions[col] = {"strategy": "drop", "rationale": identifier_like_drops[col]}
            elif s.get("zero_variance") or s.get("null_pct", 0) > 70:
                fallback_decisions[col] = {"strategy": "drop", "rationale": "rule-based: zero-variance or >70% null"}
            elif "mean" in s:
                skew = abs(s.get("skew", 0))
                if s.get("null_pct", 0) > 0:
                    if skew > 1 and s.get("min", 0) >= 0:
                        fallback_decisions[col] = {"strategy": "log_transform", "rationale": "rule-based: skewed positive numeric"}
                    else:
                        fallback_decisions[col] = {"strategy": "median_impute", "rationale": "rule-based: numeric with missing"}
                else:
                    fallback_decisions[col] = {"strategy": "standardize", "rationale": "rule-based: complete numeric"}
            else:
                if s.get("unique", 0) <= 10:
                    fallback_decisions[col] = {"strategy": "onehot_encode", "rationale": "rule-based: low-cardinality categorical"}
                else:
                    fallback_decisions[col] = {"strategy": "label_encode", "rationale": "rule-based: high-cardinality categorical"}

        return {
            "column_meta": stats,
            "preprocessing_decisions": fallback_decisions,
            "global_notes": "",
            "eda_report": f"EDA agent LLM error - rule-based fallback: {_llm_err}",
            "eda_analysis": {
                "n_columns_analysed": len(stats), "n_decisions": len(fallback_decisions),
                "fallback": True, "imbalance_analysis": imbalance_analysis,
                "imbalance_recommendations": rule_recs,
            },
            "imbalance_analysis":      imbalance_analysis,
            "imbalance_recommendations": rule_recs,
            "split_strategy":  rule_recs["recommended_split"],
            "split_rationale": rule_recs["split_rationale"],
            "agent_messages": [f"[EDA Agent] LLM error, rule-based fallback: {_llm_err}"],
        }

    decisions = result.get("decisions", {})
    decisions.pop(target, None)
    for col, rationale in target_derived_drops.items():
        decisions[col] = {"strategy": "drop", "rationale": rationale}
    for col, rationale in identifier_like_drops.items():
        decisions[col] = {"strategy": "drop", "rationale": rationale}

    # Merge LLM recs with rule-based (LLM takes priority)
    llm_recs = result.get("imbalance_recommendations") or {}
    merged_recs = {**rule_recs, **{k: v for k, v in llm_recs.items() if v is not None}}

    eda_analysis = {
        "n_columns_analysed": len(stats),
        "n_decisions":        len(decisions),
        "global_notes":       result.get("global_notes", ""),
        "eda_report":         result.get("eda_report", ""),
        "numeric_columns":    [s["name"] for s in stats if "mean" in s and not s.get("is_target")],
        "categorical_columns":[s["name"] for s in stats if "top_values" in s and not s.get("is_target")],
        "high_null_columns":  [s["name"] for s in stats if s.get("null_pct", 0) > 20],
        "zero_variance_cols": [s["name"] for s in stats if s.get("zero_variance")],
        "high_skew_columns":  [s["name"] for s in stats if abs(s.get("skew", 0)) > 1],
        "target_derived_drop_columns": list(target_derived_drops.keys()),
        "identifier_like_drop_columns": list(identifier_like_drops.keys()),
        "decisions_summary":  {col: d.get("strategy") for col, d in decisions.items()},
        "decisions_full":     decisions,
        "imbalance_analysis": imbalance_analysis,
        "imbalance_recommendations": merged_recs,
    }

    severity = imbalance_analysis.get("severity", "balanced")
    imbalance_msg = (
        f"Imbalance: {severity} (ratio {imbalance_analysis.get('imbalance_ratio', 'n/a')}:1). "
        f"Split: {merged_recs['recommended_split']}. Loss: {merged_recs['recommended_loss']}."
        if problem_type == "classification" else ""
    )

    return {
        "column_meta":             stats,
        "preprocessing_decisions": decisions,
        "global_notes":            result.get("global_notes", ""),
        "eda_report":              result.get("eda_report", ""),
        "eda_analysis":            eda_analysis,
        "imbalance_analysis":      imbalance_analysis,
        "imbalance_recommendations": merged_recs,
        # Pre-fill split_strategy for split page (user can override)
        "split_strategy":  merged_recs.get("recommended_split", "stratified"),
        "split_rationale": merged_recs.get("split_rationale", ""),
        "agent_messages": [
            f"[EDA Agent] Analysed {len(stats)} columns. Decisions: {len(decisions)}. "
            f"{imbalance_msg} {result.get('global_notes', '')[:60]}..."
        ],
    }
