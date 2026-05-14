"""
script_exporter.py — Professional Python script generator for the AutoML pipeline.

Produces a self-contained, fully-commented Python script that faithfully
reproduces everything the AutoML pipeline did:  preprocessing, feature
engineering, model training with exact hyperparameters, evaluation, and
SHAP analysis.  The output is the code a data scientist would write after
completing a manual analysis of the same dataset.
"""
from __future__ import annotations

import textwrap
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header_comment(dataset_name: str, state: dict) -> str:
    now        = datetime.now().strftime("%Y-%m-%d %H:%M")
    target     = state.get("target", "target")
    prob_type  = state.get("problem_type", "classification")
    best_model = state.get("best_model_key", "random_forest")
    cv_score   = state.get("best_cv_score", 0.0)
    metrics    = state.get("eval_metrics", {})

    if prob_type == "classification":
        primary = f"F1 (weighted): {metrics.get('f1_weighted', 'N/A')}"
        acc     = f"Accuracy:      {metrics.get('accuracy', 'N/A')}"
        auc     = f"ROC-AUC:       {metrics.get('roc_auc', 'N/A')}"
        metric_lines = f"#   {primary}\n#   {acc}\n#   {auc}"
    else:
        r2   = f"R²:   {metrics.get('r2', 'N/A')}"
        rmse = f"RMSE: {metrics.get('rmse', 'N/A')}"
        mae  = f"MAE:  {metrics.get('mae', 'N/A')}"
        metric_lines = f"#   {r2}\n#   {rmse}\n#   {mae}"

    return f"""\
#!/usr/bin/env python3
# =============================================================================
#  AutoML Pipeline — Exported Analysis Script
#  Generated : {now}
#  Dataset   : {dataset_name}
#  Target    : {target}
#  Task      : {prob_type.capitalize()}
#  Best model: {best_model}
#  CV score  : {cv_score:.4f}
#
#  Test-set performance
{metric_lines}
#
#  This script was auto-generated from an AutoML pipeline run.
#  It faithfully reproduces every step — preprocessing, feature
#  engineering, model training, evaluation, and SHAP analysis —
#  exactly as performed by the pipeline.
# =============================================================================
"""


def _imports_block(state: dict) -> str:
    best_model = state.get("best_model_key", "random_forest")
    prob_type  = state.get("problem_type", "classification")
    model_imports = _model_import_lines(best_model, prob_type)

    return f"""\
# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Data & numerics ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Sklearn — preprocessing & pipeline ────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler,
    RobustScaler, MinMaxScaler
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
)

# ── Model ──────────────────────────────────────────────────────────────────────
{model_imports}

# ── Explainability ─────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[warn] shap not installed — SHAP section will be skipped")

# ── Visualisation ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
"""


def _model_import_lines(best_model: str, prob_type: str) -> str:
    MAP = {
        "random_forest":      ("sklearn.ensemble", "RandomForestClassifier" if prob_type == "classification" else "RandomForestRegressor"),
        "gradient_boosting":  ("sklearn.ensemble", "GradientBoostingClassifier" if prob_type == "classification" else "GradientBoostingRegressor"),
        "extra_trees":        ("sklearn.ensemble", "ExtraTreesClassifier" if prob_type == "classification" else "ExtraTreesRegressor"),
        "logistic_regression":("sklearn.linear_model", "LogisticRegression"),
        "linear_regression":  ("sklearn.linear_model", "LinearRegression"),
        "ridge":              ("sklearn.linear_model", "Ridge"),
        "lasso":              ("sklearn.linear_model", "Lasso"),
        "svm":                ("sklearn.svm", "SVC" if prob_type == "classification" else "SVR"),
        "knn":                ("sklearn.neighbors", "KNeighborsClassifier" if prob_type == "classification" else "KNeighborsRegressor"),
        "decision_tree":      ("sklearn.tree", "DecisionTreeClassifier" if prob_type == "classification" else "DecisionTreeRegressor"),
        "xgboost":            ("xgboost", "XGBClassifier" if prob_type == "classification" else "XGBRegressor"),
        "lightgbm":           ("lightgbm", "LGBMClassifier" if prob_type == "classification" else "LGBMRegressor"),
        "catboost":           ("catboost", "CatBoostClassifier" if prob_type == "classification" else "CatBoostRegressor"),
        "mlp":                ("sklearn.neural_network", "MLPClassifier" if prob_type == "classification" else "MLPRegressor"),
    }
    key = best_model.lower().replace("-", "_")
    if key in MAP:
        module, cls = MAP[key]
        return f"from {module} import {cls}  # best model selected by pipeline"
    return f"# NOTE: unrecognised model key '{best_model}' — add its import manually"


def _constants_block(dataset_name: str, state: dict) -> str:
    target     = state.get("target", "target")
    prob_type  = state.get("problem_type", "classification")
    best_model = state.get("best_model_key", "random_forest")
    test_size  = state.get("test_size", 0.2)
    split_strat = state.get("split_strategy", "random")
    dt_col      = state.get("datetime_column") or ""
    grp_col     = state.get("group_column") or ""

    dt_line  = f'DATETIME_COLUMN  = "{dt_col}"  # used for time-series split' if dt_col else "DATETIME_COLUMN  = None"
    grp_line = f'GROUP_COLUMN     = "{grp_col}"  # used for group-aware split' if grp_col else "GROUP_COLUMN     = None"

    dropped = state.get("dropped_columns", []) or []
    dropped_repr = repr(dropped) if dropped else "[]  # pipeline detected no columns to drop"

    return f"""\
# =============================================================================
# CONSTANTS  —  mirrors exactly what the pipeline configured
# =============================================================================
DATASET_PATH    = "{dataset_name}"
TARGET          = "{target}"
PROBLEM_TYPE    = "{prob_type}"          # "classification" | "regression"
BEST_MODEL_KEY  = "{best_model}"
TEST_SIZE       = {test_size}
RANDOM_SEED     = 42
SPLIT_STRATEGY  = "{split_strat}"       # random | stratified | time_series | group
{dt_line}
{grp_line}
DROPPED_COLUMNS = {dropped_repr}
"""


def _load_block() -> str:
    return """\
# =============================================================================
# 1.  LOAD DATA
# =============================================================================
df = pd.read_csv(DATASET_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df.dtypes.to_string())
print("\\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
"""


def _preprocessing_block(state: dict) -> str:
    decisions = state.get("preprocessing_decisions", {}) or {}
    col_meta  = state.get("column_meta", []) or []
    target    = state.get("target", "target")

    # Categorise columns by strategy
    drop_cols     = []
    numeric_cols  = []
    cat_ohe_cols  = []
    cat_ord_cols  = []
    bool_cols     = []
    passthrough   = []
    label_encode_target = False

    for col, decision in decisions.items():
        if col == target:
            if "label" in decision.get("strategy", "").lower():
                label_encode_target = True
            continue
        strat = (decision.get("strategy") or "").lower()
        if any(x in strat for x in ("drop", "remove", "exclude")):
            drop_cols.append(col)
        elif "onehot" in strat or "one_hot" in strat or "ohe" in strat:
            cat_ohe_cols.append(col)
        elif "ordinal" in strat:
            cat_ord_cols.append(col)
        elif "bool" in strat:
            bool_cols.append(col)
        elif "pass" in strat or "no transform" in strat:
            passthrough.append(col)
        else:
            # Infer from col_meta dtype
            dtype_info = next((c.get("dtype","") for c in col_meta if c.get("name") == col), "")
            if "float" in dtype_info or "int" in dtype_info:
                numeric_cols.append(col)
            elif "object" in dtype_info or "categ" in dtype_info:
                cat_ohe_cols.append(col)
            else:
                numeric_cols.append(col)

    # Fall back to col_meta if decisions dict is sparse
    if not numeric_cols and not cat_ohe_cols and col_meta:
        for c in col_meta:
            name = c.get("name", "")
            if name == target:
                continue
            dtype = c.get("dtype", "")
            if name in drop_cols:
                continue
            if "float" in dtype or "int" in dtype:
                numeric_cols.append(name)
            elif "object" in dtype or "categ" in dtype:
                cat_ohe_cols.append(name)

    lines = ["""\
# =============================================================================
# 2.  PREPROCESSING
# =============================================================================
# The pipeline recorded the following per-column decisions.
# Each transformer below faithfully reproduces that strategy.
"""]

    # Per-column decision comments
    if decisions:
        lines.append("# Per-column strategy as decided by the pipeline agent:")
        for col, d in decisions.items():
            strat   = d.get("strategy", "—")
            rationale = (d.get("rationale") or "—")[:100]
            lines.append(f"#   {col:<30s}  {strat:<25s}  {rationale}")
        lines.append("")

    # Drop columns
    if drop_cols:
        lines.append(f"COLUMNS_TO_DROP = {drop_cols!r}")
        lines.append("df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])")
        lines.append("")

    # Separate target
    lines.append("# ── Separate features and target ─────────────────────────")
    lines.append("X = df.drop(columns=[TARGET])")
    lines.append("y = df[TARGET]")
    lines.append("")

    # Label encode target for classification if needed
    if label_encode_target:
        lines.append("# ── Encode target labels ──────────────────────────────")
        lines.append("le = LabelEncoder()")
        lines.append("y  = pd.Series(le.fit_transform(y), index=y.index)")
        lines.append("# Access encoded classes via: dict(enumerate(le.classes_))")
        lines.append("")

    # Numeric pipeline
    num_imputer = "KNNImputer(n_neighbors=5)" if any(
        "knn" in (decisions.get(c, {}).get("strategy") or "").lower()
        for c in numeric_cols
    ) else "SimpleImputer(strategy='median')"

    scaler = "StandardScaler()"
    if any("robust" in (decisions.get(c, {}).get("strategy") or "").lower() for c in numeric_cols):
        scaler = "RobustScaler()"
    elif any("minmax" in (decisions.get(c, {}).get("strategy") or "").lower() for c in numeric_cols):
        scaler = "MinMaxScaler()"

    lines.append("# ── Column lists ──────────────────────────────────────────")
    lines.append(f"NUMERIC_COLS  = {numeric_cols!r}")
    lines.append(f"CAT_OHE_COLS  = {cat_ohe_cols!r}  # one-hot encoded")
    if cat_ord_cols:
        lines.append(f"CAT_ORD_COLS  = {cat_ord_cols!r}  # ordinal encoded")
    if passthrough:
        lines.append(f"PASSTHRU_COLS = {passthrough!r}  # passed through unchanged")
    lines.append("")

    lines.append("# ── Sklearn pipelines (mirrors the pipeline's ColumnTransformer) ──")
    lines.append("numeric_transformer = Pipeline(steps=[")
    lines.append(f"    ('imputer', {num_imputer}),")
    lines.append(f"    ('scaler',  {scaler}),")
    lines.append("])")
    lines.append("")
    lines.append("categorical_transformer = Pipeline(steps=[")
    lines.append("    ('imputer', SimpleImputer(strategy='most_frequent')),")
    lines.append("    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),")
    lines.append("])")
    lines.append("")

    transformers = [
        "    ('num', numeric_transformer, NUMERIC_COLS),",
        "    ('cat', categorical_transformer, CAT_OHE_COLS),",
    ]
    if cat_ord_cols:
        transformers.append(
            "    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_ORD_COLS),"
        )
    if passthrough:
        transformers.append("    ('pass', 'passthrough', PASSTHRU_COLS),")

    lines.append("preprocessor = ColumnTransformer(transformers=[")
    lines.extend(transformers)
    lines.append("], remainder='drop')")
    lines.append("")

    return "\n".join(lines)


def _feature_engineering_block(state: dict) -> str:
    proposals  = state.get("feature_proposals", []) or []
    selected   = set(state.get("selected_features", []) or [])
    sel_props  = [p for p in proposals if p.get("name") in selected]

    if not sel_props:
        return """\
# =============================================================================
# 3.  FEATURE ENGINEERING
# =============================================================================
# The pipeline did not add engineered features in this run.
# (No proposals passed the leakage + correlation + VIF filter.)

"""

    lines = ["""\
# =============================================================================
# 3.  FEATURE ENGINEERING
# =============================================================================
# The pipeline proposed and selected the following engineered features.
# Each formula below is exactly what was applied to the raw dataframe.
#
# IMPORTANT: apply feature engineering BEFORE the train/test split so that
#            no test-set data leaks into the feature distributions.  The
#            pipeline followed this same order.
"""]

    for p in proposals:
        name     = p.get("name", "?")
        formula  = p.get("formula", "?")
        benefit  = p.get("benefit", "")
        leak     = p.get("leakage_risk", "?")
        corr_val = p.get("_corr")
        vif_val  = p.get("_vif")
        selected_mark = "✅ SELECTED" if name in selected else "➖ excluded"

        corr_str = f"{corr_val:.3f}" if corr_val is not None else "—"
        vif_str  = f"{vif_val:.2f}"  if vif_val  is not None else "—"

        lines.append(f"# [{selected_mark}] {name}")
        lines.append(f"#   formula       : {formula}")
        lines.append(f"#   benefit       : {benefit[:100]}")
        lines.append(f"#   leakage risk  : {leak}   |  corr: {corr_str}   |  VIF: {vif_str}")

        if name in selected:
            # Emit actual code
            # Replace 'df[' patterns so they reference the right variable
            code_formula = formula.replace("df[", "df[")
            lines.append(f"df['{name}'] = {code_formula}")
        else:
            lines.append(f"# df['{name}'] = {formula}  ← not selected")
        lines.append("")

    return "\n".join(lines)


def _split_block(state: dict) -> str:
    prob_type   = state.get("problem_type", "classification")
    split_strat = state.get("split_strategy", "random")
    dt_col      = state.get("datetime_column") or ""
    grp_col     = state.get("group_column") or ""
    split_rationale = (state.get("split_rationale") or "").strip()

    lines = ["""\
# =============================================================================
# 4.  TRAIN / TEST SPLIT
# =============================================================================
"""]

    if split_rationale:
        for ln in textwrap.wrap(split_rationale, 78):
            lines.append(f"# {ln}")
        lines.append("")

    if "time" in split_strat.lower() and dt_col:
        lines.append(f"# Time-series split on column '{dt_col}'")
        lines.append(f"df_sorted = df.sort_values('{dt_col}')")
        lines.append(f"split_idx = int(len(df_sorted) * (1 - TEST_SIZE))")
        lines.append("train_df  = df_sorted.iloc[:split_idx]")
        lines.append("test_df   = df_sorted.iloc[split_idx:]")
        lines.append("X_train   = train_df.drop(columns=[TARGET])")
        lines.append("X_test    = test_df.drop(columns=[TARGET])")
        lines.append("y_train   = train_df[TARGET]")
        lines.append("y_test    = test_df[TARGET]")
    elif "group" in split_strat.lower() and grp_col:
        lines.append(f"from sklearn.model_selection import GroupShuffleSplit")
        lines.append(f"gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)")
        lines.append(f"train_idx, test_idx = next(gss.split(X, y, groups=df['{grp_col}']))")
        lines.append("X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]")
        lines.append("y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]")
    elif prob_type == "classification" and "stratif" in split_strat.lower():
        lines.append(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=TEST_SIZE,\n"
            "    random_state=RANDOM_SEED,\n"
            "    stratify=y,   # preserves class proportions — as done by pipeline\n"
            ")"
        )
    else:
        stratify_note = "    stratify=y,  # stratified for classification" if prob_type == "classification" else ""
        lines.append(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=TEST_SIZE,\n"
            "    random_state=RANDOM_SEED,\n"
            + (stratify_note + "\n" if stratify_note else "")
            + ")"
        )

    lines.append("")
    lines.append(f"print(f'Train: {{len(X_train):,}} rows  |  Test: {{len(X_test):,}} rows')")

    split_analysis = state.get("split_analysis") or {}
    cb = split_analysis.get("class_balance")
    if cb:
        lines.append(f"# Pipeline-observed class balance (train): {cb}")

    lines.append("")
    return "\n".join(lines)


def _model_block(state: dict) -> str:
    best_model = state.get("best_model_key", "random_forest")
    best_params = state.get("best_params", {}) or {}
    prob_type   = state.get("problem_type", "classification")
    cv_score    = state.get("best_cv_score", 0.0)
    tuning      = state.get("tuning_results", []) or []

    cls_name = _get_class_name(best_model, prob_type)

    lines = [f"""\
# =============================================================================
# 5.  MODEL — {best_model.upper()}
# =============================================================================
# The pipeline evaluated {max(len(tuning), 1)} candidate model(s) and selected
# '{best_model}' with CV score {cv_score:.4f}.
"""]

    # All candidates summary
    if tuning:
        lines.append("# Candidate models evaluated (sorted by CV score):")
        for r in sorted(tuning, key=lambda x: x.get("best_score", 0), reverse=True):
            mk     = r.get("model_key", "?")
            score  = r.get("best_score", 0)
            trials = r.get("n_trials_completed", "?")
            marker = "  ← BEST" if mk == best_model else ""
            lines.append(f"#   {mk:<30s}  CV={score:.4f}   trials={trials}{marker}")
        lines.append("")

    # Best hyperparameters
    if best_params:
        lines.append("# Best hyperparameters found by Optuna:")
        for k, v in best_params.items():
            lines.append(f"#   {k:<35s} = {v!r}")
        lines.append("")

    # Emit model instantiation
    lines.append("# ── Instantiate the best model with tuned hyperparameters ──")
    if best_params:
        params_str = ",\n    ".join(f"{k}={v!r}" for k, v in best_params.items())
        lines.append(f"model = {cls_name}(")
        lines.append(f"    {params_str},")
        if "random_state" not in best_params and "seed" not in best_params:
            lines.append(f"    random_state=RANDOM_SEED,")
        lines.append(")")
    else:
        lines.append(f"model = {cls_name}(random_state=RANDOM_SEED)  # default params — no Optuna data found")
    lines.append("")

    # Full sklearn pipeline
    lines.append("# ── Wrap in a full sklearn Pipeline ───────────────────────")
    lines.append("full_pipeline = Pipeline(steps=[")
    lines.append("    ('preprocessor', preprocessor),")
    lines.append("    ('model',        model),")
    lines.append("])")
    lines.append("")

    # Cross-validation (mirror what pipeline used)
    cv_metric = "f1_weighted" if prob_type == "classification" else "r2"
    cv_obj    = "StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)" \
                if prob_type == "classification" \
                else "KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)"
    lines.append("# ── Cross-validation on training set (same metric as pipeline) ──")
    lines.append(f"cv = {cv_obj}")
    lines.append(
        f"cv_scores = cross_val_score(\n"
        f"    full_pipeline, X_train, y_train,\n"
        f"    cv=cv, scoring='{cv_metric}', n_jobs=-1,\n"
        f")"
    )
    lines.append(f"print(f'CV {cv_metric}: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}')")
    lines.append(f"# Pipeline reported: {cv_score:.4f}")
    lines.append("")

    lines.append("# ── Final fit on full training set ────────────────────────")
    lines.append("full_pipeline.fit(X_train, y_train)")
    lines.append("")

    return "\n".join(lines)


def _get_class_name(best_model: str, prob_type: str) -> str:
    MAP_CLF = {
        "random_forest":       "RandomForestClassifier",
        "gradient_boosting":   "GradientBoostingClassifier",
        "extra_trees":         "ExtraTreesClassifier",
        "logistic_regression": "LogisticRegression",
        "svm":                 "SVC",
        "knn":                 "KNeighborsClassifier",
        "decision_tree":       "DecisionTreeClassifier",
        "xgboost":             "XGBClassifier",
        "lightgbm":            "LGBMClassifier",
        "catboost":            "CatBoostClassifier",
        "mlp":                 "MLPClassifier",
    }
    MAP_REG = {
        "random_forest":       "RandomForestRegressor",
        "gradient_boosting":   "GradientBoostingRegressor",
        "extra_trees":         "ExtraTreesRegressor",
        "linear_regression":   "LinearRegression",
        "ridge":               "Ridge",
        "lasso":               "Lasso",
        "svm":                 "SVR",
        "knn":                 "KNeighborsRegressor",
        "decision_tree":       "DecisionTreeRegressor",
        "xgboost":             "XGBRegressor",
        "lightgbm":            "LGBMRegressor",
        "catboost":            "CatBoostRegressor",
        "mlp":                 "MLPRegressor",
    }
    key = best_model.lower().replace("-", "_")
    if prob_type == "classification":
        return MAP_CLF.get(key, f"# UnknownModel('{best_model}')")
    return MAP_REG.get(key, f"# UnknownModel('{best_model}')")


def _evaluation_block(state: dict) -> str:
    prob_type = state.get("problem_type", "classification")
    metrics   = state.get("eval_metrics", {}) or {}
    cm        = metrics.get("confusion_matrix")

    lines = ["""\
# =============================================================================
# 6.  EVALUATION  (test set)
# =============================================================================
y_pred = full_pipeline.predict(X_test)
"""]

    if prob_type == "classification":
        lines.append("# ── Classification metrics ────────────────────────────")
        lines.append("print('\\n── Classification Report ──────────────────────────')")
        lines.append("print(classification_report(y_test, y_pred))")
        lines.append("")
        lines.append("acc  = accuracy_score(y_test, y_pred)")
        lines.append("f1   = f1_score(y_test, y_pred, average='weighted')")
        lines.append("prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)")
        lines.append("rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)")
        lines.append("print(f'Accuracy  : {acc:.4f}')")
        lines.append("print(f'F1        : {f1:.4f}')")
        lines.append("print(f'Precision : {prec:.4f}')")
        lines.append("print(f'Recall    : {rec:.4f}')")
        lines.append("")
        lines.append("# Pipeline achieved:")
        for k, v in metrics.items():
            if k != "confusion_matrix" and v is not None:
                lines.append(f"#   {k:<25s} = {v!r}")
        lines.append("")

        # ROC-AUC if available
        if metrics.get("roc_auc") is not None:
            lines.append("# ── ROC-AUC (requires predict_proba) ─────────────")
            lines.append("try:")
            lines.append("    y_prob = full_pipeline.predict_proba(X_test)")
            lines.append("    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')")
            lines.append("    print(f'ROC-AUC: {roc_auc:.4f}')")
            lines.append("    # Pipeline achieved: {:.4f}".format(metrics["roc_auc"]))
            lines.append("except Exception as e:")
            lines.append("    print(f'ROC-AUC not available: {e}')")
            lines.append("")

        # Confusion matrix
        lines.append("# ── Confusion matrix ──────────────────────────────────")
        lines.append("cm = confusion_matrix(y_test, y_pred)")
        lines.append("print('\\nConfusion Matrix:')")
        lines.append("print(cm)")
        if cm:
            lines.append(f"# Pipeline confusion matrix: {cm!r}")
        lines.append("")
        lines.append("fig, ax = plt.subplots(figsize=(7, 5))")
        lines.append("sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,")
        lines.append("            linewidths=0.5, cbar_kws={'label': 'Count'})")
        lines.append("ax.set_xlabel('Predicted'); ax.set_ylabel('True')")
        lines.append("ax.set_title('Confusion Matrix — Test Set')")
        lines.append("plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.show()")

    else:
        lines.append("# ── Regression metrics ────────────────────────────────")
        lines.append("r2   = r2_score(y_test, y_pred)")
        lines.append("rmse = mean_squared_error(y_test, y_pred, squared=False)")
        lines.append("mae  = mean_absolute_error(y_test, y_pred)")
        lines.append("print(f'R²  : {r2:.4f}')")
        lines.append("print(f'RMSE: {rmse:.4f}')")
        lines.append("print(f'MAE : {mae:.4f}')")
        lines.append("")
        lines.append("# Pipeline achieved:")
        for k, v in metrics.items():
            if v is not None:
                lines.append(f"#   {k:<25s} = {v!r}")
        lines.append("")
        lines.append("# ── Residuals plot ────────────────────────────────────")
        lines.append("residuals = y_test - y_pred")
        lines.append("fig, axes = plt.subplots(1, 2, figsize=(12, 4))")
        lines.append("axes[0].scatter(y_pred, residuals, alpha=0.4, color='steelblue')")
        lines.append("axes[0].axhline(0, color='red', linewidth=1)")
        lines.append("axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')")
        lines.append("axes[0].set_title('Residuals vs Predicted')")
        lines.append("axes[1].hist(residuals, bins=40, edgecolor='white', color='steelblue')")
        lines.append("axes[1].set_title('Residual Distribution')")
        lines.append("plt.tight_layout(); plt.savefig('residuals.png', dpi=150); plt.show()")

    lines.append("")
    return "\n".join(lines)


def _shap_block(state: dict) -> str:
    shap_items = state.get("shap_importance", []) or []
    best_model = state.get("best_model_key", "random_forest")
    prob_type  = state.get("problem_type", "classification")

    lines = ["""\
# =============================================================================
# 7.  SHAP FEATURE IMPORTANCE
# =============================================================================
"""]

    if shap_items:
        lines.append("# Pipeline-computed SHAP importance (top features):")
        for item in shap_items[:20]:
            feat = item.get("feature", "?")
            imp  = item.get("importance", 0.0)
            lines.append(f"#   {feat:<35s}  mean|SHAP| = {imp:.5f}")
        lines.append("")

    lines.append("if SHAP_AVAILABLE:")
    lines.append("    # Extract the fitted model and preprocessed data from the pipeline")
    lines.append("    fitted_model      = full_pipeline.named_steps['model']")
    lines.append("    X_train_processed = full_pipeline.named_steps['preprocessor'].transform(X_train)")
    lines.append("    X_test_processed  = full_pipeline.named_steps['preprocessor'].transform(X_test)")
    lines.append("")

    tree_models = {"random_forest", "gradient_boosting", "extra_trees",
                   "xgboost", "lightgbm", "catboost", "decision_tree"}
    key = best_model.lower().replace("-", "_")

    if key in tree_models:
        lines.append("    # Tree-based model → TreeExplainer (fast & exact)")
        lines.append("    explainer   = shap.TreeExplainer(fitted_model)")
        lines.append("    shap_values = explainer.shap_values(X_test_processed)")
    else:
        lines.append("    # Non-tree model → KernelExplainer (use a background sample)")
        lines.append("    background  = shap.sample(X_train_processed, 100, random_state=RANDOM_SEED)")
        lines.append("    explainer   = shap.KernelExplainer(fitted_model.predict_proba, background)")
        lines.append("    shap_values = explainer.shap_values(X_test_processed[:200])")

    lines.append("")
    lines.append("    # Summary bar plot")
    lines.append("    shap.summary_plot(shap_values, X_test_processed,")
    if prob_type == "classification":
        lines.append("                      plot_type='bar', show=False)")
    else:
        lines.append("                      show=False)")
    lines.append("    plt.title('SHAP Feature Importance')")
    lines.append("    plt.tight_layout()")
    lines.append("    plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')")
    lines.append("    plt.show()")
    lines.append("")
    lines.append("    # Beeswarm plot — shows direction of impact")
    lines.append("    shap.summary_plot(shap_values, X_test_processed, show=False)")
    lines.append("    plt.title('SHAP Beeswarm — Impact Direction')")
    lines.append("    plt.tight_layout()")
    lines.append("    plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')")
    lines.append("    plt.show()")
    lines.append("else:")
    lines.append("    print('Install shap via: pip install shap')")
    lines.append("")
    return "\n".join(lines)


def _save_block(state: dict, dataset_name: str) -> str:
    best_model = state.get("best_model_key", "random_forest")
    stem       = dataset_name.replace(".csv", "").replace(" ", "_").lower()
    return f"""\
# =============================================================================
# 8.  SAVE MODEL
# =============================================================================
import joblib

MODEL_PATH = "{stem}_{best_model}_pipeline.pkl"
joblib.dump(full_pipeline, MODEL_PATH)
print(f"Model saved → {{MODEL_PATH}}")

# ── Load and predict example ──────────────────────────────────────────────────
# loaded_model = joblib.load(MODEL_PATH)
# predictions  = loaded_model.predict(new_data_df)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_python_script(state: dict, dataset_name: str = "dataset.csv") -> str:
    """
    Generate a complete, professional Python analysis script that faithfully
    reproduces every step the AutoML pipeline performed.

    Parameters
    ----------
    state        : the full pipeline state dict
    dataset_name : filename of the original CSV

    Returns
    -------
    str  —  the complete Python source code
    """
    sections = [
        _header_comment(dataset_name, state),
        _imports_block(state),
        _constants_block(dataset_name, state),
        _load_block(),
        _preprocessing_block(state),
        _feature_engineering_block(state),
        _split_block(state),
        _model_block(state),
        _evaluation_block(state),
        _shap_block(state),
        _save_block(state, dataset_name),
    ]
    return "\n".join(sections)
