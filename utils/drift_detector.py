"""
utils/drift_detector.py — Data Drift Detection

Detects distribution shift between training and test sets using:
  - KS test (Kolmogorov-Smirnov) for numeric features
  - PSI (Population Stability Index) for all features
  - Chi-squared test for categorical features

Used post-split to warn the user if their test set is non-representative,
which would invalidate hold-out metrics.

Public API
----------
  detect_drift(X_train, X_test, feature_names) -> DriftReport
  DriftReport.to_dict() -> dict (serialisable to PipelineState)
  DriftReport.flagged_features -> list[str]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


# ─────────────────────────────────────────────────────────────────────────────
# PSI helper
# ─────────────────────────────────────────────────────────────────────────────

def _psi_numeric(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a numeric feature.
    PSI < 0.1  → stable
    PSI 0.1–0.2 → slight shift
    PSI > 0.2  → significant shift
    """
    eps = 1e-6
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_counts = np.histogram(actual,   bins=breakpoints)[0].astype(float)

    exp_pct = (exp_counts + eps) / (len(expected) + eps)
    act_pct = (act_counts + eps) / (len(actual)   + eps)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(abs(psi), 4)


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a categorical feature."""
    eps = 1e-6
    cats = set(expected.dropna().unique()) | set(actual.dropna().unique())
    exp_vc = expected.value_counts(normalize=True)
    act_vc = actual.value_counts(normalize=True)

    psi = 0.0
    for cat in cats:
        e = float(exp_vc.get(cat, eps))
        a = float(act_vc.get(cat, eps))
        if e < eps:
            e = eps
        if a < eps:
            a = eps
        psi += (a - e) * np.log(a / e)
    return round(abs(psi), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Drift report dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    feature_stats: list[dict] = field(default_factory=list)
    overall_drift_score: float = 0.0
    overall_severity: str = "low"       # low | medium | high
    summary: str = ""

    @property
    def flagged_features(self) -> list[str]:
        """Return feature names with medium or high drift."""
        return [
            s["feature"] for s in self.feature_stats
            if s.get("severity") in ("medium", "high")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_stats":       self.feature_stats,
            "overall_drift_score": self.overall_drift_score,
            "overall_severity":    self.overall_severity,
            "summary":             self.summary,
            "flagged_features":    self.flagged_features,
            "n_flagged":           len(self.flagged_features),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def detect_drift(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    ks_pvalue_threshold: float = 0.05,
    psi_medium: float = 0.10,
    psi_high:   float = 0.20,
    max_features: int = 50,
) -> DriftReport:
    """
    Compute drift statistics for each feature.

    Parameters
    ----------
    X_train, X_test : feature DataFrames (before preprocessing)
    ks_pvalue_threshold : KS p-value below which we flag drift
    psi_medium, psi_high : PSI thresholds for severity levels
    max_features : cap to avoid very long reports on wide datasets

    Returns
    -------
    DriftReport with per-feature stats and overall assessment.
    """
    cols = [c for c in X_train.columns if c in X_test.columns][:max_features]
    feature_stats: list[dict] = []
    psi_scores: list[float] = []

    for col in cols:
        tr = X_train[col]
        te = X_test[col]
        is_numeric = pd.api.types.is_numeric_dtype(tr)

        stat: dict[str, Any] = {"feature": col, "dtype": str(tr.dtype)}

        if is_numeric:
            tr_clean = tr.dropna().values
            te_clean = te.dropna().values
            if len(tr_clean) < 5 or len(te_clean) < 5:
                continue

            # KS test
            try:
                ks_stat, ks_pval = stats.ks_2samp(tr_clean, te_clean)
                stat["ks_statistic"] = round(float(ks_stat), 4)
                stat["ks_pvalue"]    = round(float(ks_pval),  6)
                stat["ks_flagged"]   = bool(ks_pval < ks_pvalue_threshold)
            except Exception:
                stat["ks_flagged"] = False

            # PSI
            psi = _psi_numeric(tr_clean, te_clean)
            stat["psi"] = psi
            psi_scores.append(psi)

        else:
            # Categorical: chi-squared + PSI
            try:
                cats = list(set(tr.dropna().unique()) | set(te.dropna().unique()))
                tr_counts = tr.value_counts().reindex(cats, fill_value=0)
                te_counts = te.value_counts().reindex(cats, fill_value=0)
                chi2, pval, _, _ = stats.chi2_contingency(
                    np.stack([tr_counts.values, te_counts.values])
                )
                stat["chi2_statistic"] = round(float(chi2), 4)
                stat["chi2_pvalue"]    = round(float(pval),  6)
                stat["ks_flagged"]     = bool(pval < ks_pvalue_threshold)
            except Exception:
                stat["ks_flagged"] = False

            psi = _psi_categorical(tr, te)
            stat["psi"] = psi
            psi_scores.append(psi)

        # Severity from PSI
        psi_val = stat.get("psi", 0.0)
        if psi_val >= psi_high:
            stat["severity"] = "high"
        elif psi_val >= psi_medium:
            stat["severity"] = "medium"
        else:
            stat["severity"] = "low"

        feature_stats.append(stat)

    # ── Overall assessment ────────────────────────────────────────────────────
    mean_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
    n_high   = sum(1 for s in feature_stats if s.get("severity") == "high")
    n_medium = sum(1 for s in feature_stats if s.get("severity") == "medium")

    if n_high > 0 or mean_psi >= psi_high:
        overall_severity = "high"
    elif n_medium > 2 or mean_psi >= psi_medium:
        overall_severity = "medium"
    else:
        overall_severity = "low"

    summary = (
        f"Drift analysis: mean PSI={mean_psi:.3f} ({overall_severity} severity). "
        f"{n_high} features with high drift, {n_medium} with medium drift. "
        + (
            "Consider re-sampling your test set or using time-based splitting."
            if overall_severity == "high"
            else "Hold-out metrics should be reliable."
            if overall_severity == "low"
            else "Monitor these features carefully in production."
        )
    )

    return DriftReport(
        feature_stats=feature_stats,
        overall_drift_score=round(mean_psi, 4),
        overall_severity=overall_severity,
        summary=summary,
    )
