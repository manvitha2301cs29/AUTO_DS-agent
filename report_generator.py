"""
utils/report_generator.py — PDF Report Generator

Generates a comprehensive PDF report for the AutoML pipeline.

Sections:
1. Pipeline Overview
2. Dataset Summary (with feature data types)
3. EDA Report & Preprocessing Decisions
4. Feature Engineering
5. Split Strategy
6. Model Selection & Loss Function used
7. Evaluation Metrics
8. SHAP Explainability
9. Agent Message Log
"""

from __future__ import annotations
import io
from datetime import datetime

from utils.serialization import b64_to_df
from utils.ml_helpers import (
    generate_eda_plots,
    plot_model_comparison,
    plot_shap_bar,
    plot_confusion_matrix_fig,
)
from utils.eda_charts import generate_numeric_distributions, generate_numeric_summary_pies

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


# ── Loss / scoring function lookup ───────────────────────────────────────────

MODEL_LOSS_MAP = {
    # Classification
    "logistic_regression_clf": "Log-loss (Binary Cross-Entropy) — optimises log-likelihood of predicted probabilities.",
    "random_forest_clf":       "Gini Impurity (per split) — CV scored with weighted F1.",
    "gradient_boosting_clf":   "Log-loss / Deviance (gradient boosting on cross-entropy) — CV scored with weighted F1.",
    "svm_clf":                 "Hinge Loss — maximises margin between classes; CV scored with weighted F1.",
    "xgboost_clf":             "Log-loss (binary:logistic / multi:softprob) — CV scored with weighted F1.",
    "lightgbm_clf":            "Log-loss (binary / multiclass cross-entropy) — CV scored with weighted F1.",
    # Regression
    "ridge_reg":               "Mean Squared Error (MSE) with L2 regularisation — CV scored with neg-RMSE.",
    "lasso_reg":               "Mean Squared Error (MSE) with L1 regularisation — CV scored with neg-RMSE.",
    "random_forest_reg":       "Mean Squared Error (MSE) — CV scored with neg-RMSE.",
    "gradient_boosting_reg":   "Mean Squared Error (MSE) / Huber loss — CV scored with neg-RMSE.",
    "svm_reg":                 "Epsilon-Insensitive Loss (SVR) — CV scored with neg-RMSE.",
    "xgboost_reg":             "Mean Squared Error (reg:squarederror) — CV scored with neg-RMSE.",
    "lightgbm_reg":            "Mean Squared Error / L2 loss — CV scored with neg-RMSE.",
}

CV_SCORING_MAP = {
    "classification": "weighted F1-score  (handles class imbalance)",
    "regression":     "neg-Root-Mean-Squared-Error  (negated so higher = better)",
}


def _loss_description(model_key: str, problem_type: str) -> str:
    if model_key in MODEL_LOSS_MAP:
        return MODEL_LOSS_MAP[model_key]
    if problem_type == "classification":
        return "Log-loss / cross-entropy (default classification objective) — CV scored with weighted F1."
    return "Mean Squared Error (default regression objective) — CV scored with neg-RMSE."


# ── Style helpers ─────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title":    ParagraphStyle("ReportTitle",  fontSize=22, spaceAfter=8,  textColor=colors.HexColor("#1a237e"), alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "h1":       ParagraphStyle("H1",            fontSize=15, spaceAfter=6,  spaceBefore=14, textColor=colors.HexColor("#283593"), fontName="Helvetica-Bold"),
        "h2":       ParagraphStyle("H2",            fontSize=12, spaceAfter=4,  spaceBefore=10, textColor=colors.HexColor("#37474f"), fontName="Helvetica-Bold"),
        "body":     ParagraphStyle("Body",          fontSize=9,  spaceAfter=4,  leading=13,     textColor=colors.HexColor("#212121"), alignment=TA_JUSTIFY),
        "caption":  ParagraphStyle("Caption",       fontSize=8,  spaceAfter=3,  textColor=colors.HexColor("#607d8b"), fontName="Helvetica-Oblique"),
        "code":     ParagraphStyle("Code",          fontSize=8,  spaceAfter=3,  leading=11,     textColor=colors.HexColor("#1b5e20"), fontName="Courier", backColor=colors.HexColor("#f1f8e9")),
        "badge_ok": ParagraphStyle("BadgeOk",       fontSize=8,  textColor=colors.HexColor("#1b5e20"), fontName="Helvetica-Bold"),
        "badge_warn":ParagraphStyle("BadgeWarn",    fontSize=8,  textColor=colors.HexColor("#b71c1c"), fontName="Helvetica-Bold"),
        "normal":   base["Normal"],
    }
    return styles


def _hr(story, color="#bdbdbd"):
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(color), spaceAfter=4))


def _embed_plot(story, b64_str: str, width_cm: float = 15.0, caption: str = "",
                max_height_cm: float = 22.0) -> None:
    """Embed a base64 PNG into the PDF story, constrained to fit within the page.

    Parameters
    ----------
    max_height_cm : float
        Maximum rendered height in cm.  Default is 22 cm (almost full A4 page
        with 2 cm margins), so tall categorical/box-plot charts are not squashed.
        Pass a smaller value (e.g. 13) for plots that sit inside a section with
        other content on the same page.
    """
    if not b64_str:
        return
    try:
        import base64
        img_data = base64.b64decode(b64_str)
        img_buf  = io.BytesIO(img_data)
        img      = Image(img_buf)

        max_width  = width_cm    * cm
        max_height = max_height_cm * cm

        native_w = img.imageWidth
        native_h = img.imageHeight
        if native_w and native_h:
            scale = min(max_width / native_w, max_height / native_h)
            img.drawWidth  = native_w * scale
            img.drawHeight = native_h * scale
        else:
            img.drawWidth  = max_width
            img.drawHeight = max_height

        img.hAlign = "CENTER"
        story.append(img)
        if caption:
            styles = _styles()
            story.append(Spacer(1, 0.1 * cm))
            story.append(Paragraph(caption, styles["caption"]))
    except Exception:
        pass  # silently skip unembeddable images


def _section(story, title, styles):
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(title, styles["h1"]))
    _hr(story, "#5c6bc0")


def _subsection(story, title, styles):
    story.append(Paragraph(title, styles["h2"]))


def _esc(text: str, max_chars: int = 2000) -> str:
    """
    Escape XML special characters so ReportLab Paragraph never chokes on
    LLM-generated text containing &, <, >, or quotes.

    Fix #20: default max_chars raised from 800 → 2000 so LLM-generated
    narrative text and reasoning is not silently truncated in reports.
    Hard cap set at 8000 chars to protect against degenerate inputs.
    """
    import html as _html
    s = str(text)
    hard_cap = 8000
    if len(s) > hard_cap:
        s = s[:hard_cap] + f"… [truncated at {hard_cap} chars]"
    elif len(s) > max_chars:
        s = s[:max_chars] + "…"
    return _html.escape(s)


def _safe_para(text: str, style, max_chars: int = 2000) -> "Paragraph":
    """Return a Paragraph with XML-escaped, length-capped text."""
    return Paragraph(_esc(text, max_chars), style)


def _kv_table(rows: list[tuple[str, str]], styles) -> Table:
    """Two-column key-value table."""
    data = [
        [
            Paragraph("<b>" + _esc(k, 120) + "</b>", styles["body"]),
            _safe_para(str(v), styles["body"], max_chars=1200),
        ]
        for k, v in rows
    ]
    t = Table(data, colWidths=[5 * cm, 11 * cm], splitByRow=True)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#283593")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _data_table(headers: list[str], rows: list[list], styles, col_widths=None) -> Table:
    """Generic multi-column table with header row."""
    header_cells = [Paragraph("<b>" + _esc(h, 80) + "</b>", styles["caption"]) for h in headers]
    body_rows = [[_safe_para(str(c), styles["body"], max_chars=400) for c in row] for row in rows]
    data = [header_cells] + body_rows

    if col_widths is None:
        total = 16 * cm
        w = total / len(headers)
        col_widths = [w] * len(headers)

    t = Table(data, colWidths=col_widths, splitByRow=True, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ede7f6")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))
    return t


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(
    state: dict,
    dataset_name: str = "dataset",
    stored_messages: list[dict] | None = None,
) -> bytes:
    """
    Generate a complete PDF report from the pipeline state.

    Parameters
    ----------
    state           : PipelineState dict from st.session_state.agent_state
    dataset_name    : Name of the uploaded CSV file
    stored_messages : Optional list of DB message dicts {role, timestamp, content}

    Returns
    -------
    bytes : Raw PDF bytes ready for st.download_button
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm,   bottomMargin=2 * cm,
    )
    styles = _styles()
    story  = []

    def _plot_to_b64(fig) -> str:
        import base64
        if fig is None:
            return ""
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=140, bbox_inches="tight")
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass
        return base64.b64encode(img_buf.getvalue()).decode()

    now          = datetime.now().strftime("%Y-%m-%d %H:%M")
    problem_type = state.get("problem_type", "unknown")
    target       = state.get("target", "—")
    best_model   = state.get("best_model_key", "—")
    metrics      = state.get("eval_metrics", {})
    cv_score     = state.get("best_cv_score", 0)

    # Rebuild plot artifacts when they are missing from session state but the
    # underlying pipeline state still has enough information to regenerate them.
    eda_plots = dict(state.get("eda_plots", {}) or {})
    try:
        df_key = "df_engineered_parquet_b64" if state.get("df_engineered_parquet_b64") else "df_parquet_b64"
        if state.get(df_key) and target and target != "â€”":
            missing_eda_keys = {
                "target_distribution",
                "numeric_distributions",
                "box_plots",
                "categorical_charts",
                "categorical_distributions",
                "correlation_heatmap",
                "missing_values",
            } - set(k for k, v in eda_plots.items() if v)
            if missing_eda_keys:
                df_full = b64_to_df(state[df_key])
                if target in df_full.columns:
                    X_full = df_full.drop(columns=[target])
                    y_full = df_full[target]
                    regenerated = generate_eda_plots(
                        X_full, y_full, target, problem_type,
                    ) or {}
                    for key, value in regenerated.items():
                        if value and not eda_plots.get(key):
                            eda_plots[key] = value
                    if eda_plots.get("categorical_distributions") and not eda_plots.get("categorical_charts"):
                        eda_plots["categorical_charts"] = eda_plots["categorical_distributions"]
    except Exception:
        eda_plots = dict(state.get("eda_plots", {}) or {})

    model_eval_plot_cache = {
        "_plt_model_cmp_b64": state.get("_plt_model_cmp_b64") or "",
        "_plt_shap_b64": state.get("_plt_shap_b64") or "",
        "_plt_cm_b64": state.get("_plt_cm_b64") or "",
        "_plt_residuals_b64": state.get("_plt_residuals_b64") or "",
    }
    try:
        if not model_eval_plot_cache["_plt_model_cmp_b64"] and state.get("tuning_results"):
            model_eval_plot_cache["_plt_model_cmp_b64"] = _plot_to_b64(
                plot_model_comparison(state.get("tuning_results", []))
            )
    except Exception:
        pass
    try:
        if not model_eval_plot_cache["_plt_shap_b64"] and state.get("shap_importance"):
            model_eval_plot_cache["_plt_shap_b64"] = _plot_to_b64(
                plot_shap_bar(state.get("shap_importance", []))
            )
    except Exception:
        pass
    try:
        cm_data = (state.get("eval_metrics") or {}).get("confusion_matrix")
        if not model_eval_plot_cache["_plt_cm_b64"] and cm_data:
            model_eval_plot_cache["_plt_cm_b64"] = _plot_to_b64(
                plot_confusion_matrix_fig(cm_data)
            )
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # COVER
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5 * cm))

    # Header colour band
    header_band_data = [[Paragraph(
        "<font color='white'><b>AutoML Pipeline — Comprehensive Analysis Report</b></font>",
        ParagraphStyle("HdrBand", fontSize=16, alignment=TA_CENTER, textColor=colors.white,
                       fontName="Helvetica-Bold"),
    )]]
    header_band = Table(header_band_data, colWidths=[16.5 * cm])
    header_band.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#1a237e")),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_band)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b>", styles["caption"]))
    story.append(Paragraph(f"Generated: {now}", styles["caption"]))
    _hr(story, "#5c6bc0")

    # Quick-glance metrics banner
    if problem_type == "classification":
        primary_metric = (f"F1 (weighted): {metrics.get('f1_weighted'):.4f}" if metrics.get('f1_weighted') is not None else "F1: N/A")
    else:
        primary_metric = (f"R²: {metrics.get('r2'):.4f}" if metrics.get('r2') is not None else "R²: N/A")

    banner_data = [[
        Paragraph(f"<b>Target</b><br/>{target}", styles["body"]),
        Paragraph(f"<b>Type</b><br/>{problem_type.capitalize()}", styles["body"]),
        Paragraph(f"<b>Best Model</b><br/>{best_model}", styles["body"]),
        Paragraph(f"<b>CV Score</b><br/>{cv_score:.4f}" if cv_score is not None else "<b>CV Score</b><br/>N/A", styles["body"]),
        Paragraph(f"<b>Test Score</b><br/>{primary_metric}", styles["body"]),
        Paragraph(f"<b>Retries</b><br/>{state.get('retry_count', 0)}", styles["body"]),
    ]]
    banner = Table(banner_data, colWidths=[2.6 * cm] * 6)
    banner.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",   (0, 0), (-1, -1), colors.HexColor("#1a237e")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#9fa8da")),
        ("TOPPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
    ]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(banner)

    # Table of Contents
    story.append(Spacer(1, 0.6 * cm))
    _hr(story, "#9fa8da")
    story.append(Paragraph("<b>Table of Contents</b>", styles["h2"]))
    story.append(Spacer(1, 0.2 * cm))
    toc_items = [
        ("1.", "Dataset Summary — rows, columns, feature inventory & data types"),
        ("2.", "Exploratory Data Analysis — EDA report, preprocessing decisions"),
        ("2.1", "Preprocessing Decisions (per column strategy & rationale)"),
        ("2.2", "EDA Visualisations — Target distribution, Numeric overview (pie charts), "
                "Histograms + KDE, Box plots, Categorical bar+pie charts, "
                "Correlation heatmap, Missing values"),
        ("3.", "Feature Engineering — proposals, selection, distributions"),
        ("4.", "Data Split Strategy — method, sizes, class balance"),
        ("5.", "Model Selection — loss function, hyperparameters, all candidate models"),
        ("6.", "Evaluation Metrics — accuracy / F1 / R² / confusion matrix"),
        ("7.", "SHAP Feature Importance — ranked table with visual bar chart"),
        ("7b.", "Model & Evaluation Visualisations — comparison chart, SHAP bar, CM heatmap"),
        ("8.", "Orchestrator Quality Assessment — verdict, thresholds, confidence"),
        ("9.", "Agent & Decision Log — full timestamped message history"),
    ]
    toc_data = [
        [
            Paragraph(f"<b>{num}</b>", styles["body"]),
            Paragraph(desc, styles["body"]),
        ]
        for num, desc in toc_items
    ]
    toc_table = Table(toc_data, colWidths=[1.4 * cm, 14.6 * cm])
    toc_table.setStyle(TableStyle([
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING",     (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. DATASET SUMMARY — with feature data types
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "1. Dataset Summary", styles)
    story.append(_kv_table([
        ("Dataset",        dataset_name),
        ("Rows",           (f"{state.get('n_rows', 0):,}" if isinstance(state.get('n_rows'), int) else '—')),
        ("Columns",        str(state.get('n_cols', '—'))),
        ("Target column",  target),
        ("Problem type",   problem_type.capitalize()),
    ], styles))
    story.append(Spacer(1, 0.3 * cm))

    # Feature data types table
    col_meta = state.get("column_meta", [])
    if col_meta:
        _subsection(story, "1.1 Feature Inventory & Data Types", styles)
        story.append(Paragraph(
            "Every column in the dataset, with its dtype, null rate, unique count, "
            "and any notable statistical properties observed during EDA.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        dtype_rows = []
        for c in col_meta:
            name   = c.get("name", "?")
            dtype  = c.get("dtype", "?")
            is_tgt = "🎯 TARGET" if c.get("is_target") else ""
            nullpct = f"{c.get('null_pct', 0):.1f}%"
            unique  = str(c.get("unique", "?"))
            card    = c.get("cardinality", "")

            # Extra stats for numeric
            extras = []
            if "skew" in c:
                extras.append(f"skew={c['skew']:.2f}")
            if c.get("zero_variance"):
                extras.append("ZERO VARIANCE")
            if "mean" in c:
                extras.append(f"mean={c['mean']:.3g}")
            extra_str = ", ".join(extras) if extras else "—"

            dtype_rows.append([name, dtype, nullpct, unique, card, extra_str, is_tgt])

        story.append(_data_table(
            ["Column", "dtype", "Null %", "Unique", "Cardinality", "Stats", "Role"],
            dtype_rows,
            styles,
            col_widths=[3.5*cm, 1.8*cm, 1.4*cm, 1.2*cm, 2*cm, 4*cm, 2*cm],
        ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. EDA REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "2. Exploratory Data Analysis", styles)

    eda_report = state.get("eda_report", "")
    if eda_report:
        story.append(_safe_para(eda_report, styles["body"], max_chars=2000))
    if state.get("global_notes"):
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("<b>Global notes:</b> " + _esc(state["global_notes"], 1000), styles["caption"]))

    # ── EDA Analysis summary block ────────────────────────────────────────────
    eda_analysis = state.get("eda_analysis", {})
    if eda_analysis:
        story.append(Spacer(1, 0.2 * cm))
        summary_rows = [
            ("Columns analysed",     str(eda_analysis.get("n_columns_analysed", "—"))),
            ("Preprocessing decisions", str(eda_analysis.get("n_decisions", "—"))),
            ("Numeric columns",      ", ".join(eda_analysis.get("numeric_columns", [])) or "—"),
            ("Categorical columns",  ", ".join(eda_analysis.get("categorical_columns", [])) or "—"),
            ("High-null columns (>20%)", ", ".join(eda_analysis.get("high_null_columns", [])) or "None"),
            ("Zero-variance columns",    ", ".join(eda_analysis.get("zero_variance_cols", [])) or "None"),
            ("High-skew columns (|skew|>1)", ", ".join(eda_analysis.get("high_skew_columns", [])) or "None"),
        ]
        story.append(_kv_table(summary_rows, styles))

    story.append(Spacer(1, 0.3 * cm))
    _subsection(story, "2.1 Preprocessing Decisions (per column)", styles)
    # Use decisions_full from eda_analysis if available (has full rationale)
    decisions = (eda_analysis.get("decisions_full") if eda_analysis
                 else state.get("preprocessing_decisions", {}))
    if not decisions:
        decisions = state.get("preprocessing_decisions", {})
    if decisions:
        dec_rows = [
            [col, d.get("strategy", "—"), d.get("rationale", "—")]
            for col, d in decisions.items()
        ]
        story.append(_data_table(
            ["Column", "Strategy", "Rationale"],
            dec_rows, styles,
            col_widths=[4*cm, 3.5*cm, 8.5*cm],
        ))
    else:
        story.append(Paragraph("No preprocessing decisions recorded.", styles["caption"]))

    # ── EDA Visualisations (train-set only, post-split) ─────────────────────────
    if eda_plots:
        story.append(Spacer(1, 0.3 * cm))
        _subsection(story, "2.2 EDA Visualisations (Train Set Only)", styles)
        story.append(Paragraph(
            "All charts below were generated from the training set only, after the train/test split, "
            "to ensure no test-set data influences any analysis.",
            styles["caption"],
        ))

        # Each key maps to: (display_title, max_height_cm)
        # Categorical charts and box plots are often very tall (many features);
        # give them nearly the full A4 content area (25 cm between margins).
        # Generate numeric summary pie overview on-the-fly
        try:
            df_key_eda = "df_engineered_parquet_b64" if state.get("df_engineered_parquet_b64") else "df_parquet_b64"
            if state.get(df_key_eda) and target and target != "—":
                from utils.serialization import b64_to_df
                df_eda = b64_to_df(state[df_key_eda])
                numeric_cols_eda = [
                    c for c in df_eda.columns
                    if c != target and df_eda[c].dtype.kind in "biufc"
                ]
                if numeric_cols_eda:
                    eda_plots["numeric_summary_pies"] = generate_numeric_summary_pies(
                        df_eda[numeric_cols_eda], numeric_cols_eda,
                        title="Numeric Feature Overview (Data Types, Skewness & Null Rates)",
                    )
        except Exception:
            pass

        PLOT_REGISTRY_EDA = [
            ("target_distribution",   "Target Column Distribution",                       14.0),
            ("numeric_summary_pies",  "Numeric Feature Overview (Type / Skew / Nulls)",   10.0),
            ("numeric_distributions", "Numerical Feature Distributions",                   22.0),
            ("box_plots",             "Box Plots (per class)",                             22.0),
            ("categorical_charts",    "Categorical Feature Charts",                        25.0),
            ("correlation_heatmap",   "Feature Correlation Heatmap",                       20.0),
            ("missing_values",        "Missing Values",                                    14.0),
        ]
        for key, title, max_h in PLOT_REGISTRY_EDA:
            b64 = eda_plots.get(key)
            if not b64 and key == "categorical_charts":
                b64 = eda_plots.get("categorical_distributions")
            if b64:
                story.append(PageBreak())
                story.append(Paragraph(f"<b>{title}</b>", styles["h2"]))
                story.append(Spacer(1, 0.2 * cm))
                _embed_plot(story, b64, width_cm=16.5, max_height_cm=max_h)
                story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "3. Feature Engineering", styles)
    selected_features = state.get("selected_features", [])
    proposals         = state.get("feature_proposals", [])

    feature_analysis = state.get("feature_analysis", {})
    strategy_summary = state.get("feature_strategy_summary", "") or (
        feature_analysis.get("strategy_summary", ""))

    if not selected_features and not proposals:
        story.append(Paragraph("No features were engineered in this run.", styles["body"]))
    else:
        # Summary block from feature_analysis
        if feature_analysis:
            fa_rows = [
                ("Features proposed",    str(feature_analysis.get("n_proposed", len(proposals)))),
                ("Features computable",  str(feature_analysis.get("n_computable", "—"))),
                ("Features selected",    str(len(selected_features))),
                ("User-kept (retry)",    str(feature_analysis.get("n_user_kept", 0))),
                ("User-custom (retry)",  str(feature_analysis.get("n_user_custom", 0))),
                ("Agent-proposed",       str(feature_analysis.get("n_agent_proposed", "—"))),
                ("High leakage risk",    ", ".join(feature_analysis.get("high_leakage_risk", [])) or "None"),
                ("Uncomputable features", ", ".join(feature_analysis.get("dropped_features", [])) or "None"),
            ]
            story.append(_kv_table(fa_rows, styles))

        if strategy_summary:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("<b>Agent strategy:</b> " + _esc(strategy_summary, 1200), styles["body"]))

        if feature_analysis.get("user_instructions"):
            story.append(Paragraph(
                f"<b>User instructions (retry):</b> {feature_analysis['user_instructions']}",
                styles["caption"],
            ))

        story.append(Spacer(1, 0.2 * cm))

        if proposals:
            feat_rows = []
            for p in proposals:
                selected_mark = "✅" if p.get("name") in selected_features else "➖"
                computable    = "Yes" if p.get("_computable") else "No"
                corr          = f"{p['_corr']:.3f}" if p.get("_corr") is not None else "—"
                vif           = f"{p['_vif']:.2f}"  if p.get("_vif")  is not None else "—"
                benefit       = (p.get("benefit","") or "")[:60]
                feat_rows.append([
                    selected_mark, p.get("name","?"), p.get("formula","?"),
                    p.get("leakage_risk","?"), computable, corr, vif, benefit,
                ])
            story.append(_data_table(
                ["Sel", "Name", "Formula", "Leak Risk", "Computable", "Corr", "VIF", "Benefit"],
                feat_rows, styles,
                col_widths=[1*cm, 2.5*cm, 4*cm, 2*cm, 2*cm, 1.5*cm, 1.5*cm, 3*cm],
            ))

            # Benefit / rationale sub-table for selected features
            selected_props = [p for p in proposals if p.get("name") in selected_features]
            if selected_props:
                story.append(Spacer(1, 0.2*cm))
                _subsection(story, "3.1 Selected Feature Rationale", styles)
                ben_rows = [
                    [p.get("name","?"), p.get("benefit","—"), p.get("leakage_reason","—")]
                    for p in selected_props
                ]
                story.append(_data_table(
                    ["Feature", "Why it helps", "Leakage note"],
                    ben_rows, styles,
                    col_widths=[3*cm, 7*cm, 6*cm],
                ))

        engineered_numeric_plot = ""
        try:
            if state.get("df_engineered_parquet_b64") and selected_features:
                df_eng = b64_to_df(state["df_engineered_parquet_b64"])
                engineered_numeric = [
                    col for col in selected_features
                    if col in df_eng.columns and df_eng[col].dtype.kind in "biufc"
                ]
                if engineered_numeric:
                    engineered_numeric_plot = generate_numeric_distributions(
                        df_eng[engineered_numeric],
                        engineered_numeric,
                        title="Engineered Numeric Feature Distributions",
                    )
        except Exception:
            engineered_numeric_plot = ""

        if engineered_numeric_plot:
            story.append(PageBreak())
            _subsection(story, "3.2 Numerical Feature Distributions (Engineered Features Only)", styles)
            story.append(Paragraph(
                "This page shows distributions for engineered numeric features only, after feature engineering and before model fitting.",
                styles["caption"],
            ))
            story.append(Spacer(1, 0.2 * cm))
            _embed_plot(story, engineered_numeric_plot, width_cm=16.5, max_height_cm=22.0)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. SPLIT STRATEGY
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "4. Data Split Strategy", styles)

    split_analysis = state.get("split_analysis", {})
    split_rows = [
        ("Strategy",        state.get("split_strategy", "—")),
        ("Test size",       f"{state.get('test_size', 0.2):.0%}"),
        ("Datetime column", state.get("datetime_column") or "N/A"),
        ("Group column",    state.get("group_column") or "N/A"),
    ]
    if split_analysis:
        split_rows += [
            ("Train rows (approx.)", str(int(split_analysis.get("n_rows",0) *
                                         (1 - split_analysis.get("test_size",0.2))))),
            ("Test rows (approx.)",  str(int(split_analysis.get("n_rows",0) *
                                         split_analysis.get("test_size",0.2)))),
        ]
        cb = split_analysis.get("class_balance")
        if cb:
            split_rows.append(("Class balance",
                ", ".join(f"{k}: {v:.1%}" for k, v in cb.items())))
        snap = split_analysis.get("dataset_snapshot", {})
        if snap.get("high_null_cols"):
            split_rows.append(("High-null cols (excluded from split key)",
                ", ".join(snap["high_null_cols"])))
    story.append(_kv_table(split_rows, styles))

    if state.get("split_rationale"):
        story.append(Spacer(1, 0.2 * cm))
        story.append(_safe_para(state["split_rationale"], styles["body"], max_chars=1200))
    warnings = state.get("split_warnings", [])
    for w in warnings:
        story.append(Paragraph(f"⚠️ {w}", styles["badge_warn"]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. MODEL SELECTION & LOSS FUNCTION
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "5. Model Selection", styles)

    loss_desc    = _loss_description(best_model, problem_type)
    cv_scoring   = CV_SCORING_MAP.get(problem_type, "custom scoring")
    best_params  = state.get("best_params", {})

    model_analysis = state.get("model_analysis", {})
    model_rows = [
        ("Best model",           best_model),
        ("CV score",             f"{cv_score:.4f}" if cv_score is not None else "N/A"),
        ("CV scoring metric",    cv_scoring),
        ("Loss / objective",     loss_desc),
        ("Agent recommendation", state.get("model_recommendation", "—")),
        ("Agent retries",        str(state.get("retry_count", 0))),
    ]
    if model_analysis:
        model_rows += [
            ("Train samples",   str(model_analysis.get("n_train_samples", "—"))),
            ("Features used",   str(model_analysis.get("n_features", "—"))),
            ("CV folds",        str(model_analysis.get("cv_folds", "—"))),
            ("Optuna trials",   str(model_analysis.get("optuna_trials", "—"))),
            ("Models tuned",    str(model_analysis.get("n_tuned_successfully", "—"))),
        ]
        if model_analysis.get("user_instructions"):
            model_rows.append(("User instructions (retry)", model_analysis["user_instructions"]))
    story.append(_kv_table(model_rows, styles))

    story.append(Spacer(1, 0.3 * cm))
    _subsection(story, "5.1 Loss Function Explained", styles)
    story.append(Paragraph(
        f"The model <b>{best_model}</b> was trained using the following objective/loss function: "
        f"<b>{loss_desc}</b>  "
        f"During hyperparameter optimisation (Optuna), CV performance was measured using "
        f"<b>{cv_scoring}</b>. A higher value is better; Optuna maximised this score.",
        styles["body"],
    ))

    story.append(Spacer(1, 0.3 * cm))
    _subsection(story, "5.2 Best Hyperparameters", styles)
    if best_params:
        param_rows = [[k, str(v)] for k, v in best_params.items()]
        story.append(_data_table(
            ["Parameter", "Value"],
            param_rows,
            styles,
            col_widths=[7 * cm, 9 * cm],
        ))
    else:
        story.append(Paragraph("No hyperparameters recorded.", styles["caption"]))

    story.append(Spacer(1, 0.3 * cm))
    _subsection(story, "5.3 All Candidate Models & Their Parameters", styles)
    tuning_results   = state.get("tuning_results", [])
    all_model_params = state.get("all_model_params", {})
    if tuning_results:
        # Summary table
        tr_rows = [
            [
                r.get("model_key", "?"),
                (f"{r.get('best_score'):.4f}" if r.get("best_score") is not None else "N/A"),
                str(r.get("n_trials_completed", "?")),
                "★ BEST" if r.get("model_key") == best_model else "",
            ]
            for r in sorted(tuning_results, key=lambda x: (x.get("best_score") is not None, x.get("best_score") or 0), reverse=True)
        ]
        story.append(_data_table(
            ["Model", "Best CV Score", "Optuna Trials", ""],
            tr_rows, styles,
            col_widths=[6*cm, 3*cm, 3*cm, 4*cm],
        ))

        # Per-model hyperparameter tables
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Hyperparameters found by Optuna for each tuned model:",
            styles["body"],
        ))
        for r in sorted(tuning_results, key=lambda x: (x.get("best_score") is not None, x.get("best_score") or 0), reverse=True):
            mk     = r.get("model_key", "?")
            score  = r.get("best_score")
            params = all_model_params.get(mk) or r.get("best_params", {})
            is_best = mk == best_model
            score_str = f"{score:.4f}" if score is not None else "N/A"
            title   = f"{'★ Best: ' if is_best else ''}{mk}  (CV = {score_str})"
            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph(f"<b>{title}</b>", styles["h2"]))
            if params:
                param_rows = [[k, str(v)] for k, v in params.items()]
                story.append(_data_table(
                    ["Parameter", "Value"], param_rows, styles,
                    col_widths=[7*cm, 9*cm],
                ))
            else:
                story.append(Paragraph("No parameters recorded.", styles["caption"]))

    # ── Candidate rationales from model_analysis ─────────────────────────────
    candidates_detail = (state.get("model_analysis") or {}).get("candidates_detail", [])
    if candidates_detail:
        story.append(Spacer(1, 0.3 * cm))
        _subsection(story, "5.4 Why Each Model Was Chosen as a Candidate", styles)
        story.append(Paragraph(
            "The Model Agent selected these candidates based on the dataset characteristics. "
            "Each rationale explains the agent's reasoning.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.15 * cm))
        rat_rows = [
            [
                "★ " + c["model_key"] if c.get("is_best") else c["model_key"],
                c.get("rationale", "—"),
                (f"{c.get('best_score'):.4f}" if c.get("best_score") is not None else "N/A"),
            ]
            for c in candidates_detail
        ]
        story.append(_data_table(
            ["Model", "Why selected", "CV Score"],
            rat_rows, styles,
            col_widths=[4*cm, 9*cm, 3*cm],
        ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. EVALUATION METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "6. Evaluation Metrics (Test Set)", styles)

    eval_analysis = state.get("eval_analysis", {})

    if not metrics:
        story.append(Paragraph("No evaluation metrics available.", styles["caption"]))
    else:
        if problem_type == "classification":
            metric_rows = [
                ("Accuracy",       f"{metrics['accuracy']:.4f}" if metrics.get("accuracy") is not None else "N/A"),
                ("F1 (weighted)",  f"{metrics['f1_weighted']:.4f}" if metrics.get("f1_weighted") is not None else "N/A"),
                ("ROC-AUC",        f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "N/A"),
            ]
        else:
            metric_rows = [
                ("RMSE", f"{metrics['rmse']:.4f}" if metrics.get("rmse") is not None else "N/A"),
                ("MAE",  f"{metrics['mae']:.4f}"  if metrics.get("mae")  is not None else "N/A"),
                ("R²",   f"{metrics['r2']:.4f}"   if metrics.get("r2")   is not None else "N/A"),
            ]
        if eval_analysis:
            metric_rows += [
                ("Test samples",  str(eval_analysis.get("n_test_samples", "—"))),
                ("Features used", str(eval_analysis.get("n_features", "—"))),
                ("CV score (train)", f"{eval_analysis.get('cv_score_at_training', 0):.4f}"
                                     if eval_analysis.get('cv_score_at_training') else "—"),
            ]
        story.append(_kv_table(metric_rows, styles))

        cm_data = metrics.get("confusion_matrix")
        if cm_data:
            story.append(Spacer(1, 0.3 * cm))
            _subsection(story, "6.1 Confusion Matrix", styles)
            story.append(Paragraph(
                "Rows = actual class, Columns = predicted class.", styles["caption"]
            ))
            n = len(cm_data)
            cm_table_data = [[str(v) for v in row] for row in cm_data]
            # Cap column width so table never exceeds usable page width (16 cm)
            max_total_w = 16.0 * cm
            col_w = min(2 * cm, max_total_w / max(n, 1))
            cm_table = Table(cm_table_data, colWidths=[col_w] * n)
            cm_table.setStyle(TableStyle([
                ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE",  (0, 0), (-1, -1), 9),
                ("GRID",      (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ("BACKGROUND",(0, 0), (-1, -1), colors.HexColor("#e3f2fd")),
                ("FONTNAME",  (0, 0), (-1, -1), "Helvetica-Bold"),
            ]))
            story.append(cm_table)

    if state.get("eval_report"):
        story.append(Spacer(1, 0.3 * cm))
        _subsection(story, "6.2 Evaluation Agent Narrative", styles)
        story.append(_safe_para(state["eval_report"], styles["body"], max_chars=2000))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. EXPLAINABILITY — SHAP Feature Importance
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "7. Explainability — SHAP Feature Importance", styles)

    shap_importance = state.get("shap_importance", [])
    if not shap_importance:
        story.append(Paragraph(
            "SHAP feature importance was not computed in this run. "
            "This may occur when the model type is not supported by the SHAP explainer or "
            "when evaluation was skipped.",
            styles["body"],
        ))
    else:
        story.append(Paragraph(
            "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution "
            "to individual predictions. The values below show the mean absolute SHAP value across "
            "the test set — higher values indicate a feature has a larger overall impact on the model's output.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        # Use full shap list from eval_analysis if available
        eval_analysis_shap = state.get("eval_analysis", {})
        shap_full = eval_analysis_shap.get("shap_importance_full", shap_importance)

        # ── Robust feature name resolution ────────────────────────────────────
        # eval_analysis["feature_names"] is the list the preprocessor output —
        # the order matches the column indices used when SHAP was computed.
        # SHAP entries store the name as-computed (may be "f0", "f1" when sklearn
        # ColumnTransformer names weren't extracted, or the full preprocessor name
        # like "num_median_impute__age"). We map by index position, not key lookup.
        fn_list = eval_analysis_shap.get("feature_names", [])

        import re as _re
        _FN_RE   = _re.compile(r'^f(\d+)$')          # "f0", "f1" — NOT "fare", "features"
        _FEAT_RE = _re.compile(r'^feature_(\d+)$')    # "feature_0" — NOT "feature_count"

        def _resolve_feat(shap_entry: dict, rank_0idx: int) -> str:
            raw = shap_entry.get("feature", "")
            if not raw:
                if rank_0idx < len(fn_list):
                    n = fn_list[rank_0idx]
                    return n.split("__", 1)[1] if "__" in n else n
                return raw

            # Resolve strict "fN" index placeholder
            m = _FN_RE.match(raw)
            if m:
                idx = int(m.group(1))
                if idx < len(fn_list):
                    n = fn_list[idx]
                    return n.split("__", 1)[1] if "__" in n else n

            # Resolve strict "feature_N" index placeholder
            m2 = _FEAT_RE.match(raw)
            if m2:
                idx = int(m2.group(1))
                if idx < len(fn_list):
                    n = fn_list[idx]
                    return n.split("__", 1)[1] if "__" in n else n

            # Strip sklearn ColumnTransformer prefix "num_xyz__colname" → "colname"
            if "__" in raw:
                return raw.split("__", 1)[1]

            # Already a real column name — return as-is
            return raw

        max_imp = max((s.get("importance", 0) for s in shap_full), default=1) or 1

        # Build table rows with a ReportLab Drawing bar instead of ASCII blocks
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.lib.colors import HexColor

        BAR_W     = 4.5 * cm   # total bar cell width in points (reduced to fit page)
        BAR_H_PT  = 10         # bar height in points
        BAR_MAX_W = BAR_W * 0.92

        def _make_bar(imp: float) -> Drawing:
            pct = imp / max_imp
            fill_w = max(1, BAR_MAX_W * pct)
            # Colour: coral for top feature, steel-blue gradient otherwise
            if pct > 0.85:
                fill_col = HexColor("#e53935")
            elif pct > 0.5:
                fill_col = HexColor("#1976d2")
            else:
                fill_col = HexColor("#64b5f6")
            d = Drawing(BAR_W, BAR_H_PT + 4)
            # Background track
            d.add(Rect(0, 2, BAR_MAX_W, BAR_H_PT,
                       fillColor=HexColor("#e0e0e0"), strokeColor=None))
            # Filled bar
            d.add(Rect(0, 2, fill_w, BAR_H_PT,
                       fillColor=fill_col, strokeColor=None))
            # Percentage label
            d.add(String(BAR_MAX_W + 3, 4, f"{pct*100:.0f}%",
                         fontSize=7, fillColor=HexColor("#424242")))
            return d

        shap_rows = []
        for i, s in enumerate(shap_full[:20]):
            imp   = s.get("importance", 0)
            fname = _resolve_feat(s, i)
            shap_rows.append([str(i + 1), fname, f"{imp:.4f}", _make_bar(imp)])

        # Use a plain _data_table-style table but with Drawing in last col
        header_cells = [
            Paragraph("<b>Rank</b>",               styles["caption"]),
            Paragraph("<b>Feature</b>",             styles["caption"]),
            Paragraph("<b>Mean |SHAP|</b>",         styles["caption"]),
            Paragraph("<b>Relative importance</b>", styles["caption"]),
        ]
        col_widths_shap = [1.2*cm, 5.3*cm, 2.5*cm, BAR_W + 0.5*cm]
        table_data = [header_cells] + shap_rows
        shap_table = Table(table_data, colWidths=col_widths_shap, splitByRow=True)
        shap_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#283593")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ede7f6")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#e0e0e0")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(shap_table)

        story.append(Spacer(1, 0.3 * cm))
        _subsection(story, "7.1 How to interpret SHAP values", styles)
        story.append(Paragraph(
            "A high Mean |SHAP| for a feature means it consistently shifts the model's prediction "
            "either up or down from the baseline (average prediction). The top features listed above "
            "are the primary drivers of your model's decisions. "
            "Features ranked near the bottom have minimal predictive signal and could potentially "
            "be pruned in a future iteration to reduce model complexity.",
            styles["body"],
        ))

        # Dominant feature commentary — use real name
        if shap_importance:
            top_feat  = _resolve_feat({"feature": shap_importance[0].get("feature", "?")}, 0)
            top_val   = shap_importance[0].get("importance", 0)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                f"<b>Key insight:</b> The most influential feature is <b>'{top_feat}'</b> "
                f"with a mean |SHAP| of {top_val:.4f}. "
                "This feature should receive particular attention during domain validation — "
                "verify it does not represent data leakage and is available at inference time.",
                styles["body"],
            ))

        # Feature names legend (shown only if preprocessor renamed features)
        fn_list = state.get("eval_analysis", {}).get("feature_names", [])
        if fn_list and any(n.startswith("num_") or n.startswith("cat_") for n in fn_list[:3]):
            story.append(Spacer(1, 0.15 * cm))
            _subsection(story, "7.2 Feature Name Legend (preprocessor output)", styles)
            story.append(Paragraph(
                "The preprocessor may rename features with pipeline prefixes. "
                "The table below maps preprocessor output names to original column names.",
                styles["caption"],
            ))
            legend_rows = [[str(i), name] for i, name in enumerate(fn_list[:30])]
            story.append(_data_table(
                ["Index", "Feature name (as used by model)"],
                legend_rows, styles, col_widths=[2*cm, 14*cm],
            ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 7b. MODEL & EVALUATION VISUALISATIONS (non-EDA plots only)
    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: EDA plots (distributions, box plots, categorical charts, correlation,
    # missing values) are already shown in Section 2.2 and are NOT repeated here.
    _section(story, "7b. Model & Evaluation Visualisations", styles)
    story.append(Paragraph(
        "Charts generated during model training and evaluation. "
        "EDA charts are shown earlier in Section 2.2.",
        styles["body"],
    ))

    # Only model/eval plots — not eda_plots which are already in Section 2.2
    MODEL_EVAL_PLOTS = [
        ("_plt_model_cmp_b64",  "Model Comparison (CV Scores)",      14.0),
        ("_plt_shap_b64",       "SHAP Feature Importance Bar Chart",  16.0),
        ("_plt_cm_b64",         "Confusion Matrix Heatmap",           14.0),
        ("_plt_residuals_b64",  "Residuals Plot (Regression)",        14.0),
    ]

    any_model_plot = False
    for key, label, max_h in MODEL_EVAL_PLOTS:
        b64 = model_eval_plot_cache.get(key) or state.get(key)
        if b64:
            any_model_plot = True
            story.append(PageBreak())
            story.append(Paragraph(f"<b>{label}</b>", styles["h2"]))
            story.append(Spacer(1, 0.2 * cm))
            _embed_plot(story, b64, width_cm=16.5, max_height_cm=max_h)
            story.append(Spacer(1, 0.3 * cm))

    if not any_model_plot:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "No model/evaluation plot images were captured in the current session state. "
            "SHAP bar charts, confusion matrix, and model comparison plots are generated "
            "during the evaluation phase — re-run the pipeline if missing.",
            styles["caption"],
        ))

    story.append(PageBreak())


    _section(story, "8. Orchestrator Quality Assessment", styles)

    orch_analysis = state.get("orchestrator_analysis", {})
    orch_decision = state.get("orchestrator_decision", {})

    if orch_analysis:
        verdict    = orch_analysis.get("verdict", state.get("loop_verdict", "—"))
        confidence = orch_analysis.get("confidence", "—")
        reasoning  = orch_analysis.get("reasoning", state.get("loop_reasoning", "—"))
        assessment = orch_analysis.get("score_assessment", "")

        orch_rows = [
            ("Final verdict",   verdict.upper()),
            ("Confidence",      confidence.upper()),
            ("Score assessment", assessment),
            ("Reasoning",       reasoning),
            ("Retries completed", str(orch_analysis.get("retry_count", 0))),
        ]
        feat_strat  = orch_analysis.get("suggested_feature_strategy", "") or state.get("loop_feature_strategy","")
        model_strat = orch_analysis.get("suggested_model_strategy", "") or state.get("loop_model_strategy","")
        if feat_strat:
            orch_rows.append(("Feature strategy suggestion", feat_strat))
        if model_strat:
            orch_rows.append(("Model strategy suggestion", model_strat))
        story.append(_kv_table(orch_rows, styles))

        # Threshold breach table
        breaches = orch_analysis.get("threshold_breaches", [])
        if breaches:
            story.append(Spacer(1, 0.2 * cm))
            _subsection(story, "8.1 Quality Threshold Check", styles)
            br_rows = [
                [
                    b["metric"],
                    (f"{b['threshold']:.4f}" if b.get('threshold') is not None else 'N/A'),
                    (f"{b['actual']:.4f}" if b.get('actual') is not None else 'N/A'),
                    "✅ PASS" if b["passed"] else "❌ FAIL",
                ]
                for b in breaches
            ]
            story.append(_data_table(
                ["Metric", "Threshold", "Actual", "Result"],
                br_rows, styles,
                col_widths=[4*cm, 3.5*cm, 3.5*cm, 5*cm],
            ))
    elif orch_decision:
        story.append(_safe_para(
            "Verdict: " + str(orch_decision.get("verdict","—")) + ". " +
            str(orch_decision.get("reasoning","")), styles["body"], max_chars=2000))
    else:
        story.append(Paragraph("No orchestrator analysis recorded.", styles["caption"]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. AGENT LOG
    # ═══════════════════════════════════════════════════════════════════════════
    _section(story, "9. Agent & Decision Log", styles)
    story.append(Paragraph(
        "Complete record of every decision made by AI agents and human reviewers during this pipeline run.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    if stored_messages:
        log_rows = []
        for m in stored_messages:
            ts    = (m.get("timestamp") or "")[:16].replace("T", " ")
            role  = "🤖 Agent" if m.get("role") == "agent" else "👤 User"
            content = str(m.get("content", ""))[:300]
            log_rows.append([ts, role, content])
        story.append(_data_table(
            ["Timestamp", "Actor", "Message"],
            log_rows,
            styles,
            col_widths=[3 * cm, 2.5 * cm, 10.5 * cm],
        ))
    else:
        # Fall back to agent_messages list
        agent_msgs = state.get("agent_messages", [])
        if agent_msgs:
            log_rows = [["", "🤖 Agent", msg[:300]] for msg in agent_msgs]
            story.append(_data_table(
                ["Time", "Actor", "Message"],
                log_rows,
                styles,
                col_widths=[2 * cm, 2.5 * cm, 11.5 * cm],
            ))
        else:
            story.append(Paragraph("No messages were recorded in this session.", styles["caption"]))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    _hr(story)
    story.append(Paragraph(
        f"AutoML LangGraph Report — Generated {now} — {dataset_name}",
        styles["caption"],
    ))

    doc.build(story)
    return buf.getvalue()
