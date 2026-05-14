from __future__ import annotations

import io
import math
import random
import re
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from utils.stats_safety import clean_numeric_frame_for_corr

GLOBAL_SEED = 42

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor":    "#ffffff",
        "axes.facecolor":      "#fafafa",
        "axes.edgecolor":      "#cccccc",
        "axes.grid":           True,
        "grid.color":          "#e8e8e8",
        "grid.linewidth":      0.6,
        "font.family":         "DejaVu Sans",
        "axes.titlesize":      10,
        "axes.labelsize":      8,
        "xtick.labelsize":     7.5,
        "ytick.labelsize":     7.5,
        "figure.titlesize":    13,
        "figure.titleweight":  "bold",
    })


def _b64_fig(fig, dpi: int = 150) -> str:
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _coerce_bool_to_float(X):
    if isinstance(X, pd.DataFrame):
        return X.astype("boolean").astype("float64")
    if isinstance(X, pd.Series):
        return X.to_frame().astype("boolean").astype("float64")
    return pd.DataFrame(X).astype("boolean").astype("float64")


def _coerce_bool_to_string(X):
    if isinstance(X, pd.DataFrame):
        return X.astype("string")
    if isinstance(X, pd.Series):
        return X.to_frame().astype("string")
    return pd.DataFrame(X).astype("string")


def _winsorize_array(X):
    return np.clip(
        X,
        np.nanpercentile(X, 1, axis=0),
        np.nanpercentile(X, 99, axis=0),
    )


def normalize_feature_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_identifier_like_columns(
    df: pd.DataFrame, target: str, unique_ratio_threshold: float = 0.9
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    id_tokens = {"id", "uuid", "uid", "pk", "key", "serial", "record", "identifier", "code", "number"}
    no_context_tokens = {"serial", "record", "row", "ticket", "invoice", "order", "passenger", "class"}

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        non_null = int(s.notna().sum())
        if non_null <= 1:
            continue
        unique = int(s.nunique(dropna=True))
        unique_ratio = unique / max(non_null, 1)
        if unique_ratio < unique_ratio_threshold:
            continue
        norm = normalize_feature_name(col)
        tokens = set(re.findall(r"[a-z0-9]+", str(col).lower()))
        has_id_hint = (
            bool(tokens & id_tokens)
            or norm.endswith(("id", "uuid", "uid", "pk", "key", "serial", "identifier", "code"))
            or "serial" in norm
            or ("no" in tokens and bool(tokens & no_context_tokens))
        )
        is_categorical = not pd.api.types.is_numeric_dtype(s)
        if has_id_hint:
            reasons[col] = "auto-drop: identifier-like column with near-unique values"
        elif is_categorical:
            reasons[col] = "auto-drop: categorical column cardinality is approximately equal to the sample count"

    return reasons


def detect_target_derived_columns(df: pd.DataFrame, target: str) -> dict[str, str]:
    target_norm = normalize_feature_name(target)
    normalized_cols = {col: normalize_feature_name(col) for col in df.columns}
    area_tokens = ("area", "acre", "acres", "acreage", "hectare", "hectares", "hecture", "ha")
    ratio_tokens = ("per", "rate", "ratio", "avg", "average", "by")
    target_tokens = [tok for tok in target_norm.replace("_", " ").split() if tok]
    if not target_tokens and target_norm:
        target_tokens = [target_norm]

    derived: dict[str, str] = {}
    has_area_column = any(
        col != target and any(token in norm for token in area_tokens)
        for col, norm in normalized_cols.items()
    )
    for col, norm in normalized_cols.items():
        if col == target:
            continue
        if not any(tok in norm for tok in target_tokens):
            continue
        mentions_area = any(token in norm for token in area_tokens)
        mentions_ratio = any(token in norm for token in ratio_tokens)
        if "yield" in target_norm and has_area_column and (mentions_area or mentions_ratio):
            derived[col] = (
                "auto-drop: derived yield-per-area style feature detected while "
                "target is yield and an area column is present"
            )
        elif mentions_area and mentions_ratio:
            derived[col] = (
                f"auto-drop: column name suggests a target-derived ratio or rate based on '{target}'"
            )
    return derived


def set_global_seed(seed: int | None = None) -> int:
    seed = GLOBAL_SEED if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def auto_dataset_insights(df: pd.DataFrame, target: str, problem_type: str | None = None) -> dict:
    problem_type = problem_type or "classification"
    warnings: list[str] = []
    type_issues: list[str] = []
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate rows.")

    if any(float(df[c].isna().mean()) > 0.5 for c in df.columns):
        warnings.append("High missingness detected.")

    constant_columns = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    if constant_columns:
        warnings.append(f"Constant or zero-variance columns detected: {constant_columns}")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target]
    y = df[target]

    imbalance_ratio = None
    recommended_problem_type = problem_type
    likely_multi_label = False

    if problem_type == "classification":
        counts = y.value_counts(dropna=False)
        if not counts.empty and counts.min() > 0:
            imbalance_ratio = float(counts.max() / counts.min())
        if pd.api.types.is_integer_dtype(y) and 3 <= y.nunique(dropna=True) <= 10:
            vals = sorted(int(v) for v in y.dropna().unique().tolist())
            if vals == list(range(vals[0], vals[-1] + 1)):
                type_issues.append("Target appears ordinal.")
        joined = y.dropna().astype(str)
        likely_multi_label = bool((joined.str.contains(",") | joined.str.contains(";")).mean() > 0.2)
        if likely_multi_label:
            type_issues.append("Target appears multi-label.")
    else:
        recommended_problem_type = "regression"

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "n_numeric": int(len(numeric_cols)),
        "n_categorical": int(len(categorical_cols)),
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "warnings": warnings,
        "type_issues": type_issues,
        "imbalance_ratio": imbalance_ratio,
        "recommended_problem_type": recommended_problem_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# generate_eda_plots — full chart suite
# ═══════════════════════════════════════════════════════════════════════════════

def generate_eda_plots(
    X_train: pd.DataFrame,
    y_train,
    target: str,
    problem_type: str,
    label_encoder=None,
) -> dict[str, str]:
    """
    Generate all EDA charts and return them as base64-encoded PNG strings.

    Keys returned
    -------------
    target_distribution    : pie + bar chart of target classes (classification)
                             or histogram + KDE (regression)
    numeric_distributions  : histograms + KDE for every numeric feature
    box_plots              : per-class box plots (classification) or overall (regression)
    categorical_charts     : bar + pie pairs for each categorical feature
    correlation_heatmap    : Pearson correlation matrix of numeric features
    missing_values         : severity-coloured horizontal bar chart
    """
    _apply_theme()
    from utils.eda_charts import generate_numeric_distributions, generate_categorical_charts

    plots: dict[str, str] = {}

    categorical = [
        c for c in X_train.columns
        if not pd.api.types.is_numeric_dtype(X_train[c])
        and not pd.api.types.is_bool_dtype(X_train[c])
    ]
    numeric = [
        c for c in X_train.columns
        if c not in categorical and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    try:
        plots["target_distribution"] = _plot_target_distribution(y_train, target, problem_type)
    except Exception:
        pass

    try:
        if numeric:
            plots["numeric_distributions"] = generate_numeric_distributions(
                X_train[numeric], numeric,
                title="Numerical Feature Distributions (train set)",
            )
    except Exception:
        pass

    try:
        if numeric:
            plots["box_plots"] = _plot_box_plots(X_train[numeric], y_train, problem_type)
    except Exception:
        pass

    try:
        if categorical:
            plots["categorical_charts"] = generate_categorical_charts(
                X_train, categorical,
                title="Categorical Feature Charts (train set)",
            )
            plots["categorical_distributions"] = plots["categorical_charts"]
    except Exception:
        pass

    try:
        if len(numeric) >= 2:
            plots["correlation_heatmap"] = _plot_correlation_heatmap(X_train[numeric])
    except Exception:
        pass

    try:
        plots["missing_values"] = _plot_missing_values(X_train)
    except Exception:
        pass

    return plots


# ── Individual plot helpers ───────────────────────────────────────────────────

def _plot_target_distribution(y_train, target: str, problem_type: str) -> str:
    _apply_theme()
    if problem_type == "regression":
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle(f"Target Distribution — {target}", fontsize=13, fontweight="bold")
        data = pd.Series(y_train).dropna()

        axes[0].hist(data, bins=40, color=PALETTE[0], edgecolor="white",
                     linewidth=0.4, density=True, alpha=0.85)
        try:
            data.plot.kde(ax=axes[0], color="#e15759", linewidth=2)
        except Exception:
            pass
        axes[0].set_title("Distribution (Histogram + KDE)", fontsize=10)
        axes[0].set_xlabel(target, fontsize=9)
        axes[0].set_ylabel("Density", fontsize=9)
        axes[0].spines[["top", "right"]].set_visible(False)

        stats_text = (
            f"Count:   {len(data):,}\n"
            f"Mean:    {data.mean():.4f}\n"
            f"Median:  {data.median():.4f}\n"
            f"Std:     {data.std():.4f}\n"
            f"Min:     {data.min():.4f}\n"
            f"Max:     {data.max():.4f}\n"
            f"Skew:    {data.skew():.3f}\n"
            f"Kurt:    {data.kurt():.3f}"
        )
        axes[1].axis("off")
        axes[1].text(
            0.1, 0.5, stats_text,
            transform=axes[1].transAxes,
            fontsize=10, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8eaf6", alpha=0.8),
        )
        axes[1].set_title("Descriptive Statistics", fontsize=10)
        plt.tight_layout()
        return _b64_fig(fig)

    # Classification
    vc = pd.Series(y_train).value_counts().sort_values(ascending=False)
    labels = [str(l) for l in vc.index]
    counts = vc.values.tolist()
    colors = (PALETTE * math.ceil(len(labels) / len(PALETTE)))[: len(labels)]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Target Distribution — {target}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[5, 3, 2], wspace=0.35)
    ax_bar    = fig.add_subplot(gs[0])
    ax_pie    = fig.add_subplot(gs[1])
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis("off")

    total = sum(counts)
    bars = ax_bar.bar(range(len(labels)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels([str(l)[:20] for l in labels], rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel("Count", fontsize=9)
    ax_bar.set_title("Class Counts", fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count/total:.1%})",
            ha="center", va="bottom", fontsize=7.5,
        )

    wedges, _, autotexts = ax_pie.pie(
        counts, labels=None, colors=colors,
        autopct="%1.1f%%", pctdistance=0.72,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(7.5)
    ax_pie.set_title("Class Share", fontsize=10)
    ax_pie.set_aspect("equal")

    patches = [mpatches.Patch(color=c, label=str(l)[:24]) for c, l in zip(colors, labels)]
    ax_legend.legend(handles=patches, loc="center left", bbox_to_anchor=(0, 0.5),
                     fontsize=8, frameon=False)
    plt.tight_layout()
    return _b64_fig(fig)


def _plot_box_plots(X_numeric: pd.DataFrame, y_train, problem_type: str) -> str:
    _apply_theme()
    num_cols = list(X_numeric.columns)
    if not num_cols:
        return ""

    n     = len(num_cols)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig_w = 6 * ncols
    fig_h = max(4, nrows * 3.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Feature Box Plots (train set)", fontsize=13, fontweight="bold")

    y_series = pd.Series(y_train).reset_index(drop=True)
    X_reset  = X_numeric.reset_index(drop=True)

    if problem_type == "classification":
        classes = sorted(y_series.dropna().unique())
        class_colors = (PALETTE * math.ceil(len(classes) / len(PALETTE)))[: len(classes)]
    else:
        classes = None
        class_colors = None

    for i, col in enumerate(num_cols):
        ax = axes[i // ncols][i % ncols]

        if problem_type == "classification" and classes is not None:
            class_data, class_labels = [], []
            for cls in classes:
                vals = X_reset.loc[y_series == cls, col].dropna().values
                if len(vals) > 0:
                    class_data.append(vals)
                    class_labels.append(str(cls))
            if class_data:
                bp = ax.boxplot(
                    class_data, patch_artist=True, notch=False, widths=0.55,
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1},
                    capprops={"linewidth": 1.5},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                )
                for patch, color in zip(bp["boxes"], class_colors[:len(class_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                ax.set_xticklabels([str(l)[:12] for l in class_labels],
                                   rotation=20, ha="right", fontsize=7)
        else:
            data_vals = X_reset[col].dropna().values
            bp = ax.boxplot(
                data_vals, patch_artist=True, widths=0.4,
                medianprops={"color": "white", "linewidth": 2},
                whiskerprops={"linewidth": 1},
                capprops={"linewidth": 1.5},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            )
            bp["boxes"][0].set_facecolor(PALETTE[0])
            bp["boxes"][0].set_alpha(0.8)

        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Value", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    if problem_type == "classification" and classes is not None:
        patches = [mpatches.Patch(color=c, label=str(cls))
                   for c, cls in zip(class_colors, classes)]
        fig.legend(
            handles=patches, title="Target class",
            loc="lower center", ncol=min(len(classes), 6),
            bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=True,
        )

    plt.tight_layout()
    return _b64_fig(fig)


def _plot_correlation_heatmap(X_numeric: pd.DataFrame) -> str:
    _apply_theme()
    numeric = clean_numeric_frame_for_corr(X_numeric)
    if numeric.shape[1] < 2:
        return ""
    corr = numeric.corr(numeric_only=True)
    n = len(corr)
    side = max(6, min(n * 0.7, 20))
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    fig.suptitle("Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold")

    annot = n <= 25
    fmt   = ".2f" if annot else ""

    sns.heatmap(
        corr, ax=ax, annot=annot, fmt=fmt,
        cmap="RdYlBu_r", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        annot_kws={"size": max(5, 8 - n // 5)},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=max(6, 9 - n // 6))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=max(6, 9 - n // 6))
    plt.tight_layout()
    return _b64_fig(fig)


def _plot_missing_values(X: pd.DataFrame) -> str:
    _apply_theme()
    miss = (X.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No missing values detected in any feature",
                ha="center", va="center", fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.8))
        ax.axis("off")
        fig.suptitle("Missing Values", fontsize=12, fontweight="bold")
        return _b64_fig(fig)

    n = len(miss)
    fig_h = max(3, n * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.suptitle("Missing Values by Feature (%)", fontsize=13, fontweight="bold")

    bar_colors = [
        "#e53935" if pct >= 50 else "#fb8c00" if pct >= 20 else "#43a047"
        for pct in miss.values
    ]

    bars = ax.barh(range(n), miss.values, color=bar_colors,
                   edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(c)[:35] for c in miss.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Missing (%)", fontsize=9)
    ax.set_xlim(0, max(miss.values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, pct in zip(bars, miss.values):
        ax.text(bar.get_width() + max(miss.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", fontsize=7.5)

    legend_patches = [
        mpatches.Patch(color="#43a047", label="< 20%  (mild)"),
        mpatches.Patch(color="#fb8c00", label="20-50% (moderate)"),
        mpatches.Patch(color="#e53935", label=">= 50% (severe)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    return _b64_fig(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Professional model / evaluation plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix_fig(confusion_matrix_values, class_labels=None):
    _apply_theme()
    cm_array = np.asarray(confusion_matrix_values)
    n = cm_array.shape[0]
    side = max(5, min(n * 0.9, 14))
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    fig.suptitle("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")

    sns.heatmap(
        cm_array, annot=True, fmt="g",
        cmap="Blues", ax=ax,
        linewidths=0.5, linecolor="#e0e0e0",
        cbar_kws={"shrink": 0.75, "label": "Count"},
        annot_kws={"size": max(7, 12 - n // 3)},
    )
    if class_labels is not None:
        ax.set_xticklabels(class_labels, rotation=30, ha="right")
        ax.set_yticklabels(class_labels, rotation=0)

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_importance: list[dict]):
    _apply_theme()
    if not shap_importance:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        return fig

    top_n = min(20, len(shap_importance))
    items = shap_importance[:top_n]
    features = [str(item.get("feature", f"f{i}"))[:35] for i, item in enumerate(items)]
    importances = [item.get("importance", 0.0) for item in items]

    fig_h = max(4, top_n * 0.42)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.suptitle("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")

    max_imp = max(importances) or 1
    colors = [
        "#e53935" if v / max_imp > 0.85 else
        "#1976d2" if v / max_imp > 0.50 else
        "#64b5f6"
        for v in importances
    ]

    bars = ax.barh(range(top_n), importances, color=colors,
                   edgecolor="white", linewidth=0.4, height=0.72)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(features, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| value", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, importances):
        ax.text(bar.get_width() + max_imp * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5)

    plt.tight_layout()
    return fig


def plot_model_comparison(results: list[dict]):
    _apply_theme()
    if not results:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No model results available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        return fig

    sorted_results = sorted(results, key=lambda r: r.get("best_score", 0), reverse=True)
    models = [r.get("model_key", r.get("model", "?")) for r in sorted_results]
    scores = [r.get("best_score", 0.0) for r in sorted_results]
    trials = [r.get("n_trials_completed", 0) for r in sorted_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(models) * 0.55 + 2)))
    fig.suptitle("Model Comparison — Cross-Validation Scores", fontsize=13, fontweight="bold")

    bar_colors = [PALETTE[0]] + [PALETTE[3]] * (len(models) - 1)
    bars = axes[0].barh(range(len(models)), scores, color=bar_colors,
                        edgecolor="white", linewidth=0.4, height=0.65)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([m[:30] for m in models], fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("CV Score", fontsize=9)
    axes[0].set_title("CV Score by Model (sorted)", fontsize=10)
    axes[0].spines[["top", "right"]].set_visible(False)

    score_range = max(scores) - min(scores) if len(scores) > 1 else 0.01
    for bar, score in zip(bars, scores):
        axes[0].text(
            bar.get_width() + score_range * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", ha="left", fontsize=8,
        )
    axes[0].set_xlim(0, max(scores) * 1.12)

    axes[1].bar(range(len(models)), trials, color=PALETTE[1],
                edgecolor="white", linewidth=0.4)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels([m[:15] for m in models], rotation=30, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Trials completed", fontsize=9)
    axes[1].set_title("Optuna Trials per Model", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline export utilities
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ExportPipeline:
    preprocessor: object
    model: object

    def predict(self, X):
        return self.model.predict(self.preprocessor.transform(X))


def build_export_pipeline(preprocessor, best_model_obj):
    buf = io.BytesIO()
    joblib.dump(_ExportPipeline(preprocessor=preprocessor, model=best_model_obj), buf)
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Functions required by agents — restored from original
# ═══════════════════════════════════════════════════════════════════════════════

def compute_baseline_score(y_train, y_test, problem_type: str) -> float:
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    if problem_type == "classification":
        dummy = DummyClassifier(strategy="most_frequent", random_state=GLOBAL_SEED)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        preds = dummy.predict(np.zeros((len(y_test), 1)))
        return float(f1_score(y_test, preds, average="weighted", zero_division=0))
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    preds = dummy.predict(np.zeros((len(y_test), 1)))
    return float(r2_score(y_test, preds))


def select_loss_function(problem_type: str, y=None, metadata: dict | None = None) -> dict:
    metadata = metadata or {}

    if problem_type == "classification":
        y_arr = np.asarray(y) if y is not None else np.array([])
        class_counts = (
            pd.Series(y_arr).value_counts(dropna=False).to_dict()
            if y is not None and y_arr.size
            else {}
        )
        imbalance_ratio = None
        if class_counts:
            counts = np.asarray(list(class_counts.values()), dtype=float)
            if counts.min() > 0:
                imbalance_ratio = float(counts.max() / counts.min())
        return {
            "loss_function":    "log_loss",
            "problem_type":     problem_type,
            "n_samples":        int(metadata.get("n_samples", len(y_arr))),
            "n_features":       int(metadata.get("n_features", 0)),
            "class_counts":     class_counts,
            "imbalance_ratio":  imbalance_ratio,
            "reason":           "Probabilistic classification objective selected.",
        }

    return {
        "loss_function": "rmse",
        "problem_type":  problem_type,
        "n_samples":     int(metadata.get("n_samples", len(np.asarray(y)) if y is not None else 0)),
        "n_features":    int(metadata.get("n_features", 0)),
        "reason":        "Regression objective selected.",
    }


# ── Split helpers ─────────────────────────────────────────────────────────────

def _time_split(df_full, X, y, datetime_col, test_size):
    if not datetime_col or datetime_col not in df_full.columns:
        raise ValueError("datetime_col is required for time_series splits")
    ordered = df_full.sort_values(datetime_col)
    n_test  = max(1, int(len(ordered) * float(test_size)))
    train_df, test_df = ordered.iloc[:-n_test], ordered.iloc[-n_test:]
    return (
        train_df.drop(columns=[y.name]), test_df.drop(columns=[y.name]),
        train_df[y.name], test_df[y.name],
    )


def _group_split(X, y, group_col, test_size, random_seed):
    if not group_col or group_col not in X.columns:
        raise ValueError("group_col is required for group_based splits")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=X[group_col]))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _stratified_split(X, y, test_size, random_seed):
    return train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)


def execute_split(
    df,
    target,
    strategy,
    test_size,
    random_seed,
    datetime_col=None,
    group_col=None,
    problem_type=None,
    cal_size=0.15,
):
    random_seed  = GLOBAL_SEED if random_seed is None else int(random_seed)
    problem_type = problem_type or "classification"
    X = df.drop(columns=[target])
    y = df[target].copy()

    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    if strategy == "time_series":
        X_train, X_test, y_train, y_test = _time_split(df, X, y, datetime_col, test_size)
    elif strategy == "group_based":
        X_train, X_test, y_train, y_test = _group_split(X, y, group_col, test_size, random_seed)
    elif strategy == "stratified":
        X_train, X_test, y_train, y_test = _stratified_split(X, y, test_size, random_seed)
    else:
        strat = y if problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=strat
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )

    X_cal = y_cal = None
    if len(X_train) >= 100 and problem_type == "classification":
        if strategy == "time_series":
            n_cal = max(1, int(len(X_train) * cal_size))
            X_cal, y_cal = X_train.iloc[-n_cal:], y_train.iloc[-n_cal:]
            X_train, y_train = X_train.iloc[:-n_cal], y_train.iloc[:-n_cal]
        else:
            try:
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X_train, y_train, test_size=cal_size, random_state=random_seed, stratify=y_train
                )
            except ValueError:
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X_train, y_train, test_size=cal_size, random_state=random_seed
                )

    return X_train, X_test, y_train, y_test, label_encoder, X_cal, y_cal


# ── Preprocessor builder ──────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame, decisions: dict):
    bool_cols      = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols   = [c for c in X.select_dtypes(include="number").columns.tolist() if c not in bool_cols]
    # Use is_numeric_dtype to correctly handle StringDtype, ArrowDtype, etc.
    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols and c not in bool_cols
        and not pd.api.types.is_numeric_dtype(X[c])
    ]
    transformers   = []

    for col in numeric_cols:
        strategy = (decisions.get(col) or {}).get("strategy", "standardize")
        if strategy == "drop":
            continue
        elif strategy == "knn_impute":
            pipe = Pipeline([("impute", KNNImputer()), ("scale", StandardScaler())])
        elif strategy == "log_transform":
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log",    FunctionTransformer(np.log1p, validate=False)),
                ("scale",  StandardScaler()),
            ])
        elif strategy == "winsorize":
            pipe = Pipeline([
                ("impute",    SimpleImputer(strategy="median")),
                ("winsorize", FunctionTransformer(_winsorize_array, validate=False)),
                ("scale",     StandardScaler()),
            ])
        elif strategy == "mean_impute":
            pipe = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())])
        elif strategy == "keep_as_is":
            pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        else:
            pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append((col, pipe, [col]))

    for col in categorical_cols:
        strategy = (decisions.get(col) or {}).get("strategy", "onehot_encode")
        if strategy == "drop":
            continue
        elif strategy == "label_encode":
            from sklearn.preprocessing import OrdinalEncoder
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        else:
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        transformers.append((col, pipe, [col]))

    for col in bool_cols:
        strategy = (decisions.get(col) or {}).get("strategy", "standardize")
        if strategy == "drop":
            continue
        if strategy == "keep_as_is":
            pipe = Pipeline([("cast", FunctionTransformer(_coerce_bool_to_float, validate=False))])
        elif strategy in {"onehot_encode", "label_encode"}:
            pipe = Pipeline([
                ("cast",   FunctionTransformer(_coerce_bool_to_string, validate=False)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        elif strategy == "knn_impute":
            pipe = Pipeline([
                ("cast",   FunctionTransformer(_coerce_bool_to_float, validate=False)),
                ("impute", KNNImputer()),
                ("scale",  StandardScaler()),
            ])
        elif strategy == "log_transform":
            pipe = Pipeline([
                ("cast",   FunctionTransformer(_coerce_bool_to_float, validate=False)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("log",    FunctionTransformer(np.log1p, validate=False)),
                ("scale",  StandardScaler()),
            ])
        else:
            pipe = Pipeline([
                ("cast",   FunctionTransformer(_coerce_bool_to_float, validate=False)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("scale",  StandardScaler()),
            ])
        transformers.append((col, pipe, [col]))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, numeric_cols + bool_cols, categorical_cols


def get_preprocessor_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return X.columns.tolist()
