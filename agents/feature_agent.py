"""
agents/feature_agent.py — Feature Engineering Agent (v2)

Improvements over v1:
  #A  SHAP-based feedback loop — orchestrator's top SHAP features guide retry
  #B  Advanced features: polynomial interactions, target encoding proposals
  #C  SHAP + L1 automatic feature selection after engineering
  #D  Caching: hash of (df_shape, target, retry_count) avoids redundant LLM calls
  #E  AST-based safe_eval_feature() unchanged (retained from v1)
  #F  VIF computation retained from v1
"""

from __future__ import annotations
import ast
import hashlib
import json
import operator as _op
import os
import re
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import PipelineState
from utils.agent_utils import agent_error_handler, call_llm_json
from utils.logger import get_logger

log = get_logger(__name__)
from utils.serialization import b64_to_df, sanitize_for_msgpack as _sanitize
from utils.stats_safety import safe_corr


# ─────────────────────────────────────────────────────────────────────────────
# Numpy → native Python type sanitizer (prevents msgpack serialization errors)
# ─────────────────────────────────────────────────────────────────────────────

from utils.advanced_features import (
    generate_polynomial_features,
    generate_target_encoding,
)


# ─────────────────────────────────────────────────────────────────────────────
# AST-based safe feature evaluator (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

_BINARY_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub,
    ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.FloorDiv: _op.floordiv,
}
_CMP_OPS = {
    ast.Eq: _op.eq, ast.NotEq: _op.ne,
    ast.Lt: _op.lt, ast.LtE: _op.le,
    ast.Gt: _op.gt, ast.GtE: _op.ge,
}
_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}
_ALLOWED_NP = frozenset({
    "log", "log1p", "log2", "sqrt", "abs", "clip",
    "where", "maximum", "minimum", "exp", "sign",
    "floor", "ceil", "round", "nan", "inf",
})


class _ASTEval(ast.NodeVisitor):
    def __init__(self, df):
        self._df = df

    def visit_BinOp(self, node):
        op = _BINARY_OPS.get(type(node.op))
        if op is None: raise ValueError(f"Forbidden op: {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op = _UNARY_OPS.get(type(node.op))
        if op is None: raise ValueError(f"Forbidden unary: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op_node, rn in zip(node.ops, node.comparators):
            cmp = _CMP_OPS.get(type(op_node))
            if cmp is None: raise ValueError(f"Forbidden cmp: {type(op_node).__name__}")
            left = cmp(left, self.visit(rn))
        return left

    def visit_BoolOp(self, node):
        vals = [self.visit(v) for v in node.values]
        result = vals[0]
        for v in vals[1:]:
            result = (result & v) if isinstance(node.op, ast.And) else (result | v)
        return result

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr in _ALLOWED_NP):
            fn = getattr(np, node.func.attr)
            return fn(*[self.visit(a) for a in node.args],
                      **{kw.arg: self.visit(kw.value) for kw in node.keywords})
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "astype"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in ("int", "float", "bool")):
            obj = self.visit(node.func.value)
            dtype = {"int": int, "float": float, "bool": bool}[node.args[0].id]
            return obj.astype(dtype)
        raise ValueError(f"Forbidden call: {ast.dump(node.func)}")

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            key = self.visit(node.slice)
            if not isinstance(key, str): raise ValueError("Key must be str literal")
            if key not in self._df.columns: raise ValueError(f"Unknown column: {key!r}")
            return self._df[key]
        raise ValueError("Subscript only on df")

    def visit_IfExp(self, node):
        return np.where(self.visit(node.test), self.visit(node.body), self.visit(node.orelse))

    def visit_Name(self, node):
        if node.id == "df": return self._df
        if node.id == "np": return np
        raise ValueError(f"Unknown name: {node.id!r}")

    def visit_Constant(self, node): return node.value
    def visit_Num(self, node): return node.n
    def visit_Str(self, node): return node.s
    def visit_NameConstant(self, node): return node.value
    def visit_Index(self, node): return self.visit(node.value)

    def generic_visit(self, node):
        raise ValueError(f"Forbidden node: {type(node).__name__}")


def safe_eval_feature(df: pd.DataFrame, formula: str) -> "pd.Series | None":
    if not isinstance(formula, str) or not formula.strip():
        return None
    if any(s in formula for s in ("import", "__", "exec(", "open(")):
        return None
    try:
        tree   = ast.parse(formula.strip(), mode="eval")
        result = _ASTEval(df).visit(tree.body)
    except Exception:
        return None
    if not isinstance(result, pd.Series):
        try:
            result = pd.Series(result, index=df.index)
        except Exception:
            return None
    if len(result) != len(df):
        return None
    numeric = pd.to_numeric(result, errors="coerce")
    if numeric.isna().mean() > 0.5:
        return None
    return numeric


def compute_vif(df: pd.DataFrame, col_name: str, series: pd.Series) -> "float | None":
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        num = df.select_dtypes(include="number").copy()
        num[col_name] = series
        num = num.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if num.shape[1] < 2 or num.shape[0] < num.shape[1] + 1:
            return None
        idx = list(num.columns).index(col_name)
        return round(float(variance_inflation_factor(num.values.astype(float), idx)), 2)
    except Exception:
        return None


def _safe_feature_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower() or "feature"


def _heuristic_feature_fallback(
    df: pd.DataFrame,
    target: str,
    reserved: set[str],
) -> list[dict]:
    """
    Deterministic fallback so the feature phase never returns an empty proposal list
    when the LLM responds with no usable features.
    """
    proposals: list[dict] = []
    numeric_cols = [c for c in df.select_dtypes(include="number").columns.tolist() if c != target]
    categorical_cols = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns.tolist() if c != target]

    def _add(name: str, formula: str, benefit: str):
        safe_name = _safe_feature_name(name)
        if safe_name in reserved or any(p["name"] == safe_name for p in proposals):
            return
        proposals.append({
            "name": safe_name,
            "formula": formula,
            "benefit": benefit,
            "leakage_risk": "low",
            "leakage_reason": "Derived only from existing input features",
        })

    if len(numeric_cols) >= 1:
        c1 = numeric_cols[0]
        _add(f"{c1}_sq", f"df['{c1}'] ** 2", f"Captures non-linear effects of {c1}")

        positive_col = next((c for c in numeric_cols if (df[c].dropna() >= 0).all()), None)
        if positive_col:
            _add(
                f"log1p_{positive_col}",
                f"np.log1p(df['{positive_col}'])",
                f"Compresses skew in {positive_col} while preserving order",
            )

    if len(numeric_cols) >= 2:
        c1, c2 = numeric_cols[:2]
        _add(f"{c1}_x_{c2}", f"df['{c1}'] * df['{c2}']", f"Captures interaction between {c1} and {c2}")
        _add(f"{c1}_plus_{c2}", f"df['{c1}'] + df['{c2}']", f"Combines the joint magnitude of {c1} and {c2}")
        _add(
            f"{c1}_over_{c2}",
            f"df['{c1}'] / (df['{c2}'] + 1e-6)",
            f"Measures relative scale between {c1} and {c2}",
        )

    for col in categorical_cols[:2]:
        mode = df[col].dropna().mode()
        if len(mode) == 0:
            continue
        top_value = str(mode.iloc[0]).replace("'", "\\'")
        short_val = _safe_feature_name(top_value)[:20] or "top"
        _add(
            f"is_{col}_{short_val}",
            f"(df['{col}'] == '{top_value}').astype(int)",
            f"Flags the most common {col} category as a simple segment indicator",
        )

    return proposals[:6]


# ─────────────────────────────────────────────────────────────────────────────
# LLM cache (avoids redundant API calls on identical inputs)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_CACHE: dict[str, dict] = {}


def _cache_key(meta: dict) -> str:
    s = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior feature engineer for automated machine learning.

Given column metadata and the target variable, propose 5-8 NEW features.
Use only arithmetic/logical combinations of existing columns.
Never reference columns not listed in the metadata.
Use df['col'] syntax only. No imports, no __, no exec.

SHAP feedback: if top_shap_features are provided, bias proposals toward combinations
of those high-impact features and their interaction terms.

Return ONLY valid JSON - no markdown:
{
  "proposals": [
    {
      "name": "<snake_case_name>",
      "formula": "<expression using df['col'] syntax>",
      "benefit": "<one sentence on why this helps>",
      "leakage_risk": "<low|medium|high>",
      "leakage_reason": "<one sentence>"
    }
  ],
  "strategy_summary": "<2 sentences on the overall strategy>"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent entry point
# ─────────────────────────────────────────────────────────────────────────────

@agent_error_handler("Feature Agent")
def feature_agent(state: PipelineState) -> dict:
    log.info("Feature agent started", extra={"retry_count": state.get("retry_count", 0)})
    api_key = state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    model   = os.getenv("FEATURE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    df           = b64_to_df(state["df_parquet_b64"])
    target       = state["target"]
    problem_type = state["problem_type"]
    retry_count  = state.get("retry_count", 0)

    # ── Fix #6: enforce leakage — drop columns flagged by leakage_agent ───────
    leakage_report: dict = state.get("leakage_report") or {}
    leaked_cols: set[str] = set(leakage_report.get("dropped_by_leakage", []))
    preprocessing_decisions: dict = _sanitize(dict(state.get("preprocessing_decisions") or {}))
    # Also enforce any col with strategy="drop" from preprocessing decisions
    dropped_cols: set[str] = leaked_cols | {
        c for c, d in preprocessing_decisions.items()
        if isinstance(d, dict) and d.get("strategy") == "drop"
    }
    if dropped_cols:
        df = df.drop(columns=[c for c in dropped_cols if c in df.columns and c != target])

    user_kept_features   = state.get("user_kept_features", [])
    user_custom_features = state.get("user_custom_features", [])

    # ── SHAP feedback from orchestrator ───────────────────────────────────────
    # When the orchestrator triggers a feature retry, it passes back the top
    # SHAP features from the last run so we can bias new proposals toward them.
    top_shap_from_last_run: list[str] = []
    if retry_count > 0:
        shap_imp = state.get("shap_importance") or []
        top_shap_from_last_run = [s["feature"] for s in shap_imp[:5] if "feature" in s]

    retry_sections = []
    if retry_count > 0:
        agent_suggestion  = state.get("loop_suggestion", "")
        user_instructions = state.get("user_feature_instructions", "").strip()
        retry_sections.append(
            f"\n\nRETRY #{retry_count} — previous features underperformed.\n"
            f"Agent suggestion: {agent_suggestion}\n"
        )
        if top_shap_from_last_run:
            retry_sections.append(
                f"Top SHAP features from last run (bias toward these): {top_shap_from_last_run}\n"
            )
        if user_instructions:
            retry_sections.append(
                f"USER INSTRUCTION (highest priority): {user_instructions}\n"
                "Follow the user's instruction precisely."
            )
        if user_kept_features:
            retry_sections.append(
                f"\nAlready kept (do NOT reproduce): {user_kept_features}\n"
            )
        if user_custom_features:
            retry_sections.append(
                f"User-defined (do NOT reproduce): {[f['name'] for f in user_custom_features]}\n"
            )
        retry_sections.append("Produce DIFFERENT features — do not repeat previous formulas.")

    numeric     = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    meta = {
        "target": target, "problem_type": problem_type,
        "numeric_columns": numeric, "categorical_columns": categorical,
        "shape": list(df.shape),
        "sample_row": _sanitize(df.dropna().iloc[0].to_dict()) if len(df.dropna()) > 0 else {},
        "retry_count": retry_count,
        "user_kept_features": user_kept_features,
        "user_custom_feature_names": [f["name"] for f in user_custom_features],
        "top_shap_features": top_shap_from_last_run,
    }

    # ── LLM call with caching ─────────────────────────────────────────────────
    cache_k = _cache_key({**meta, "retry_sections": retry_sections})
    result  = _LLM_CACHE.get(cache_k)

    if result is None:
        result, _llm_err = call_llm_json(
            api_key=api_key, model_name=model,
            system_prompt=SYSTEM_PROMPT + "".join(retry_sections),
            user_content=json.dumps(meta, indent=2, default=str),
            temperature=0.3, max_tokens=2000,
        )
        if result is None:
            return _sanitize({
                "feature_proposals": list(user_custom_features),
                "agent_messages": [f"[Feature Agent] ERROR: {_llm_err} — keeping user-selected features only."],
            })
        _LLM_CACHE[cache_k] = result

    proposals = result.get("proposals", [])
    reserved  = set(user_kept_features) | {f["name"] for f in user_custom_features}
    agent_proposals = [p for p in proposals if p.get("name") not in reserved]

    # ── Advanced feature proposals: polynomial + target encoding ───────────────
    poly_proposals: list[dict] = []
    te_columns: list[str] = []

    if retry_count == 0 or "retry_features" in (state.get("loop_verdict") or ""):
        # Polynomial interactions from top numeric cols
        try:
            poly_proposals = generate_polynomial_features(
                df, [c for c in numeric if c != target], target, max_cols=6
            )
        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})

        # Target encoding for high-cardinality categoricals
        high_card_cats = [
            c for c in categorical
            if c != target and df[c].nunique() > 10
        ]
        if high_card_cats:
            try:
                df_with_te = generate_target_encoding(df, high_card_cats, target, problem_type)
                te_cols = [c for c in df_with_te.columns if c.startswith("te_") and c not in df.columns]
                te_columns = te_cols
                # Convert to proposals so they show in UI
                for c in te_cols[:4]:
                    poly_proposals.append({
                        "name":         c,
                        "formula":      f"# target-encoded: {c}",
                        "benefit":      f"Target encoding reduces cardinality of {c.replace('te_', '')}",
                        "leakage_risk": "medium",
                        "leakage_reason": "Uses cross-val smoothing to limit leakage",
                        "_computable":  True,
                        "_corr":        None,
                    })
            except Exception:
                pass

    # Merge proposals: user-kept > user-custom > agent > polynomial
    prev_by_name = {p["name"]: p for p in (state.get("feature_proposals") or [])}
    kept_full    = [prev_by_name[n] for n in user_kept_features if n in prev_by_name]
    all_proposals = kept_full + user_custom_features + agent_proposals + [
        p for p in poly_proposals if p["name"] not in reserved
    ]
    used_heuristic_fallback = False
    if not all_proposals:
        all_proposals = _heuristic_feature_fallback(df, target, reserved)
        used_heuristic_fallback = bool(all_proposals)

    # ── Enrich each proposal with computability + correlation ─────────────────
    # Fix #10: reject engineered features that are suspiciously correlated with
    # the target (|corr| > 0.95) — these almost certainly derive from the target
    # and would constitute hidden leakage.
    TARGET_LEAKAGE_CORR_THRESHOLD = 0.95
    enriched = []
    for p in all_proposals:
        if p.get("_computable") is not None and p["name"] in reserved:
            enriched.append(p)
            continue
        if p.get("formula", "").startswith("#"):
            # Pre-computed (target encoding) — mark as computable
            p.setdefault("_computable", True)
            enriched.append(p)
            continue
        series = safe_eval_feature(df, p.get("formula", ""))
        if series is not None:
            p["_computable"] = True
            if pd.api.types.is_numeric_dtype(df[target]):
                corr = safe_corr(series, df[target])
                p["_corr"] = round(float(corr), 4) if corr is not None else None
                # Block target-derived features (hidden leakage guard)
                if p["_corr"] is not None and abs(p["_corr"]) >= TARGET_LEAKAGE_CORR_THRESHOLD:
                    p["_computable"] = False
                    p["leakage_risk"] = "high"
                    p["leakage_reason"] = (
                        f"Engineered feature has |corr|={abs(p['_corr']):.3f} with target "
                        f"— likely target-derived. Blocked to prevent hidden leakage."
                    )
            else:
                p["_corr"] = None
            p["_vif"] = compute_vif(df, p["name"], series) if p.get("_computable") else None
        else:
            p["_computable"] = False
            p["_corr"]       = None
            p["_vif"]        = None
        enriched.append(p)

    user_label = ""
    if user_kept_features or user_custom_features:
        user_label = (
            f" ({len(user_kept_features)} user-kept, "
            f"{len(user_custom_features)} user-custom, "
            f"{len(agent_proposals)} agent-proposed, "
            f"{len(poly_proposals)} auto-advanced)"
        )

    # ── Fix #18: L1 + SHAP automatic feature selection ────────────────────────
    # Build a temporary numeric feature matrix from all computable proposals
    # and run SelectFromModel(Lasso) to auto-prune low-signal features.
    # Then cross-check against SHAP feedback from last run (if available).
    l1_selected: list[str] = []
    shap_selected: list[str] = []
    auto_pruned: list[str] = []

    computable_proposals = [p for p in enriched if p.get("_computable") and
                            not p.get("formula", "").startswith("#")]
    if len(computable_proposals) >= 3 and target in df.columns:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.feature_selection import SelectFromModel
            from sklearn.preprocessing import StandardScaler
            from utils.ml_helpers import GLOBAL_SEED

            feat_matrix = {}
            for p in computable_proposals:
                s = safe_eval_feature(df, p.get("formula", ""))
                if s is not None:
                    feat_matrix[p["name"]] = s

            if feat_matrix:
                X_sel = pd.DataFrame(feat_matrix).fillna(0).replace([np.inf, -np.inf], 0)
                y_sel = df[target]
                if not pd.api.types.is_numeric_dtype(y_sel):
                    y_sel = y_sel.astype("category").cat.codes

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_sel)

                lasso = Lasso(alpha=0.01, random_state=GLOBAL_SEED, max_iter=2000)
                sel = SelectFromModel(lasso, threshold="mean")
                sel.fit(X_scaled, y_sel)
                l1_selected = list(X_sel.columns[sel.get_support()])

                # Mark pruned features (not in L1 selection and not user-specified)
                user_names = {p["name"] for p in enriched
                              if p.get("formula", "").startswith("#") or
                              p["name"] in user_kept_features}
                for p in enriched:
                    if (p.get("_computable") and
                            p["name"] in feat_matrix and
                            p["name"] not in l1_selected and
                            p["name"] not in user_names):
                        p["_l1_pruned"] = True
                        auto_pruned.append(p["name"])

        except Exception as _silent_exc:
            log.warning("Silenced exception", extra={"error": str(_silent_exc)})  # L1 selection is best-effort; never block the pipeline

    # SHAP pruning: if we have SHAP from a previous run, keep only top-N
    # features that overlap with the current proposals.
    if top_shap_from_last_run and retry_count > 0:
        shap_set = set(top_shap_from_last_run)
        shap_selected = [p["name"] for p in enriched if p["name"] in shap_set]

    strategy_summary = result.get("strategy_summary", "")
    if used_heuristic_fallback:
        fallback_summary = (
            "LLM and advanced generators returned no usable feature proposals, "
            "so deterministic fallback features were created from the available columns."
        )
        strategy_summary = f"{strategy_summary} {fallback_summary}".strip()
    feature_analysis = _sanitize({
        "strategy_summary":    strategy_summary,
        "n_proposed":          len(enriched),
        "n_computable":        sum(1 for p in enriched if p.get("_computable")),
        "n_user_kept":         len(user_kept_features),
        "n_user_custom":       len(user_custom_features),
        "n_agent_proposed":    len(agent_proposals),
        "n_poly_advanced":     len(poly_proposals),
        "retry_count":         retry_count,
        "user_instructions":   state.get("user_feature_instructions", ""),
        "shap_feedback_used":  top_shap_from_last_run,
        "used_heuristic_fallback": used_heuristic_fallback,
        "te_columns":          te_columns,
        "computable_features": [p["name"] for p in enriched if p.get("_computable")],
        "dropped_features":    [p["name"] for p in enriched if not p.get("_computable")],
        "high_leakage_risk":   [p["name"] for p in enriched if p.get("leakage_risk") == "high"],
        "l1_selected_features": l1_selected,
        "shap_selected_features": shap_selected,
        "auto_pruned_features": auto_pruned,
        "proposals_full":      enriched,
    })

    return _sanitize({
        "feature_proposals":        enriched,
        "feature_strategy_summary": strategy_summary,
        "feature_analysis":         feature_analysis,
        "agent_messages": [
            f"[Feature Agent] Proposed {len(enriched)} features{user_label} "
            f"({sum(1 for p in enriched if p['_computable'])} computable, "
            f"{len(auto_pruned)} L1-pruned). "
            f"SHAP feedback: {top_shap_from_last_run}. "
            f"Strategy: {strategy_summary[:60]}…"
        ],
    })
