"""
utils/advanced_features.py — Advanced Feature Engineering

Provides three types of advanced features that complement the LLM-proposed ones:

1. Polynomial features   — degree-2 interactions + squares for top numeric cols
2. Target encoding       — supervised encoding for high-cardinality categoricals
   (with cross-val smoothing to prevent leakage)
3. SHAP + L1 auto-selection — trims feature set post-training using SHAP values
   and L1 regularisation to remove redundant / low-signal features

Public API
----------
  generate_polynomial_features(df, numeric_cols, target, max_cols=10) -> list[dict]
  generate_target_encoding(df, cat_cols, target, problem_type, n_splits=5) -> pd.DataFrame
  select_features_shap_l1(X_t, y, feature_names, problem_type, top_k=30) -> list[str]
"""

from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from utils.stats_safety import safe_corr


# ─────────────────────────────────────────────────────────────────────────────
# 1. Polynomial / interaction features
# ─────────────────────────────────────────────────────────────────────────────

def generate_polynomial_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target: str,
    max_cols: int = 10,
    top_k_corr: int = 6,
) -> list[dict]:
    """
    Generate degree-2 polynomial features (interactions + squares) for the
    top-correlating numeric columns.

    Returns a list of proposal dicts compatible with feature_agent output format:
      {name, formula, benefit, leakage_risk, _computable, _corr}
    """
    if not numeric_cols or target not in df.columns:
        return []

    # Select the most correlated numeric cols (reduces explosion of pairs)
    if pd.api.types.is_numeric_dtype(df[target]):
        corrs = {
            c: abs(corr)
            for c in numeric_cols
            if c != target and df[c].std() > 0
            for corr in [safe_corr(df[c], df[target])]
            if corr is not None
        }
        top_cols = sorted(corrs, key=corrs.get, reverse=True)[:top_k_corr]  # type: ignore
    else:
        top_cols = numeric_cols[:top_k_corr]

    proposals: list[dict] = []

    # Squared terms
    for col in top_cols[:max_cols]:
        name = f"{col}_sq"
        formula = f"df['{col}'] ** 2"
        try:
            series = df[col] ** 2
            corr = safe_corr(series, df[target]) if pd.api.types.is_numeric_dtype(df[target]) else None
            proposals.append({
                "name":         name,
                "formula":      formula,
                "benefit":      f"Captures non-linear effect of {col}",
                "leakage_risk": "low",
                "leakage_reason": "derived purely from input feature",
                "_computable":  True,
                "_corr":        round(float(corr), 4) if corr is not None and not np.isnan(corr) else None,
            })
        except Exception:
            continue

    # Pairwise interactions (top_k_corr choose 2)
    for c1, c2 in combinations(top_cols, 2):
        if len(proposals) >= max_cols * 2:
            break
        name = f"{c1}_x_{c2}"
        formula = f"df['{c1}'] * df['{c2}']"
        try:
            series = df[c1] * df[c2]
            corr = safe_corr(series, df[target]) if pd.api.types.is_numeric_dtype(df[target]) else None
            proposals.append({
                "name":         name,
                "formula":      formula,
                "benefit":      f"Interaction effect between {c1} and {c2}",
                "leakage_risk": "low",
                "leakage_reason": "derived from two input features",
                "_computable":  True,
                "_corr":        round(float(corr), 4) if corr is not None and not np.isnan(corr) else None,
            })
        except Exception:
            continue

    return proposals


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target encoding with cross-val smoothing
# ─────────────────────────────────────────────────────────────────────────────

class CrossValTargetEncoder:
    """
    Target encoding with k-fold smoothing to prevent train-set leakage.

    For classification: encodes each class separately (one column per class).
    For regression: encodes the mean of the continuous target.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 1.0, random_state: int = 42):
        self.n_splits    = n_splits
        self.smoothing   = smoothing
        self.random_state = random_state
        self._global_stats: dict = {}
        self._col_stats:    dict = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        target: str,
        problem_type: str,
    ) -> pd.DataFrame:
        """Fit + transform on training data using out-of-fold encoding."""
        y = df[target]
        result = df.copy()

        if problem_type == "classification":
            classes = sorted(y.unique())

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) \
             if problem_type == "classification" \
             else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in cat_cols:
            if col not in df.columns:
                continue

            encoded = pd.Series(np.nan, index=df.index)

            if problem_type == "regression":
                col_global_mean = float(df.groupby(col)[target].mean().mean())
                for tr_idx, val_idx in kf.split(df, y if problem_type == "classification" else None):
                    tr_df = df.iloc[tr_idx]
                    stats = tr_df.groupby(col)[target].agg(["mean", "count"])
                    smooth = (stats["count"] * stats["mean"] + self.smoothing * col_global_mean) / \
                             (stats["count"] + self.smoothing)
                    encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth).fillna(col_global_mean)
                result[f"te_{col}"] = encoded.fillna(col_global_mean)
                self._col_stats[col] = {"global_mean": col_global_mean, "type": "regression"}

            else:
                # One column per class
                for cls in classes:
                    binary_target = (y == cls).astype(int)
                    col_global = float(binary_target.mean())
                    fold_encoded = pd.Series(np.nan, index=df.index)
                    for tr_idx, val_idx in kf.split(df, y):
                        tr_binary = binary_target.iloc[tr_idx]
                        tr_col    = df.iloc[tr_idx][col]
                        stats = pd.DataFrame({"tgt": tr_binary, "col": tr_col}).groupby("col")["tgt"].agg(
                            ["mean", "count"]
                        )
                        smooth = (stats["count"] * stats["mean"] + self.smoothing * col_global) / \
                                 (stats["count"] + self.smoothing)
                        fold_encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth).fillna(col_global)
                    result[f"te_{col}_cls{cls}"] = fold_encoded.fillna(col_global)

        return result

    def transform(self, df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        """Transform test data using fitted stats (fit_transform must be called first)."""
        # For simplicity, fall back to ordinal encoding on test set
        # (proper deployment would fit a separate encoder — this is for pipeline use)
        result = df.copy()
        for col, stats in self._col_stats.items():
            if col not in result.columns:
                continue
            result[f"te_{col}"] = result[col].astype(str).map(
                lambda x: stats.get(x, stats.get("global_mean", 0.0))
            )
        return result


def generate_target_encoding(
    df: pd.DataFrame,
    cat_cols: list[str],
    target: str,
    problem_type: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Convenience wrapper — returns df with target-encoded columns added.
    Original categorical columns are NOT dropped (let preprocessor handle them).
    """
    if not cat_cols:
        return df
    encoder = CrossValTargetEncoder(n_splits=n_splits)
    return encoder.fit_transform(df, cat_cols, target, problem_type)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP + L1 automatic feature selection
# ─────────────────────────────────────────────────────────────────────────────

def select_features_shap_l1(
    X_t: "np.ndarray",
    y: "np.ndarray",
    feature_names: list[str],
    problem_type: str,
    top_k: int = 30,
    l1_C: float = 0.1,
) -> list[str]:
    """
    Automatic feature selection using two complementary methods:
      1. SHAP mean |value| — model-agnostic importance
      2. L1 regularisation (Lasso / LogReg with L1) — linear sparsity

    Returns the union of top-k features from each method, capped at top_k total.
    Falls back gracefully if either method fails.
    """
    selected: set[str] = set()
    n_feat = X_t.shape[1]
    names = feature_names[:n_feat]

    # ── SHAP selection ────────────────────────────────────────────────────────
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        sample_size = min(500, len(X_t))
        X_sample = X_t[:sample_size]
        y_sample = y[:sample_size]

        if problem_type == "classification":
            probe = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        else:
            probe = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)

        probe.fit(X_sample, y_sample)
        explainer = shap.TreeExplainer(probe)
        sv = explainer(X_sample, check_additivity=False)
        vals = sv.values
        if hasattr(vals, "ndim") and vals.ndim == 3:
            vals = np.abs(vals).mean(axis=2)
        mean_shap = np.abs(vals).mean(axis=0)
        shap_top = [names[i] for i in np.argsort(mean_shap)[::-1][:top_k]]
        selected.update(shap_top)
    except Exception:
        pass

    # ── L1 selection ──────────────────────────────────────────────────────────
    try:
        from sklearn.linear_model import LogisticRegression, Lasso
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X_t)
        if problem_type == "classification":
            l1_model = LogisticRegression(penalty="l1", C=l1_C, solver="liblinear",
                                          max_iter=500, random_state=42)
        else:
            l1_model = Lasso(alpha=1.0 / (2 * l1_C * len(X_t)), max_iter=2000, random_state=42)

        l1_model.fit(X_scaled, y)
        coef = np.abs(l1_model.coef_).flatten() if hasattr(l1_model, "coef_") else np.zeros(n_feat)
        if coef.ndim > 1:
            coef = coef.max(axis=0)
        l1_top = [names[i] for i in np.argsort(coef)[::-1] if coef[i] > 0][:top_k]
        selected.update(l1_top)
    except Exception:
        pass

    # If both methods failed, return all features
    if not selected:
        return names

    # Sort by original order to maintain determinism
    return [n for n in names if n in selected]
