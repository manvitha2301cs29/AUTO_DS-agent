"""
ui/pages/fusion.py — Phase 0: Data Fusion

Lets users start with either one CSV file or multiple related CSV files.
Single-file uploads continue directly into the normal dataset setup flow.
Multi-file uploads can be joined first, then continue with the merged dataset.
"""
from __future__ import annotations

import io
import re
import json

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import alert, badge, metrics_row
from utils.schema_fusion import analyse_schema, execute_joins, execute_sql_query, relationship_diagnostics
from utils.column_intelligence import column_descriptions, _fallback_descriptions


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_name(filename: str) -> str:
    """Strip .csv and make safe for use as a table/variable name."""
    name = re.sub(r"\.csv$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _join_color(join_type: str) -> str:
    return {"inner": "#59a14f", "left": "#7ec8fa", "outer": "#f28e2b"}.get(join_type, "#aaa")


def _conf_badge(conf: float) -> str:
    color = "#59a14f" if conf >= 0.8 else "#edc948" if conf >= 0.5 else "#e15759"
    return (
        f'<span style="background:#12132a;color:{color};padding:1px 7px;'
        f'border-radius:4px;font-size:0.75rem;border:1px solid {color}55;">'
        f'{conf:.0%}</span>'
    )


def _quality_color(delta: int, total_before: int) -> str:
    if total_before == 0:
        return "#aaa"
    ratio = abs(delta) / total_before
    if ratio > 0.5:
        return "#e15759"
    if ratio > 0.1:
        return "#edc948"
    return "#59a14f"


# ── ER diagram ────────────────────────────────────────────────────────────────

def _render_er_diagram(tables: dict, relationships: list):
    """Simple SVG-based ER diagram showing tables and detected relationships."""
    import math

    n      = len(tables)
    radius = 160
    cx, cy = 300, 220
    w, h   = 620, 460

    table_names = list(tables.keys())
    positions   = {}
    for i, name in enumerate(table_names):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        positions[name] = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )

    svg_lines = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#12132a;border-radius:8px;">',
        f'<text x="{w//2}" y="22" text-anchor="middle" fill="#7ec8fa" '
        f'font-size="13" font-weight="bold" font-family="monospace">'
        f'Detected Schema Relationships</text>',
    ]

    # Draw relationship lines
    for rel in relationships:
        ta, tb = rel.get("table_a"), rel.get("table_b")
        if ta not in positions or tb not in positions:
            continue
        x1, y1 = positions[ta]
        x2, y2 = positions[tb]
        conf   = rel.get("confidence", 0.5)
        color  = "#59a14f" if conf >= 0.8 else "#edc948" if conf >= 0.5 else "#e15759"
        jt     = rel.get("join_type", "left").upper()
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        svg_lines.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="6,3" opacity="0.8"/>'
        )
        svg_lines.append(
            f'<rect x="{mx-22:.0f}" y="{my-9:.0f}" width="44" height="18" '
            f'rx="4" fill="#1a1a2e" stroke="{color}" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{mx:.0f}" y="{my+5:.0f}" text-anchor="middle" '
            f'fill="{color}" font-size="9" font-family="monospace">{jt}</text>'
        )

        ka, kb = rel.get("key_a", ""), rel.get("key_b", "")
        label  = f"{ka} → {kb}"
        svg_lines.append(
            f'<text x="{mx:.0f}" y="{my-13:.0f}" text-anchor="middle" '
            f'fill="#8888aa" font-size="8" font-family="monospace">{label[:22]}</text>'
        )

    # Draw table boxes
    BOX_W, BOX_H = 100, 32
    for name, (x, y) in positions.items():
        df     = tables[name]
        n_rows = len(df)
        n_cols = len(df.columns)
        svg_lines += [
            f'<rect x="{x - BOX_W//2:.0f}" y="{y - BOX_H//2:.0f}" '
            f'width="{BOX_W}" height="{BOX_H}" rx="6" '
            f'fill="#1a1b30" stroke="#7ec8fa" stroke-width="1.5"/>',
            f'<text x="{x:.0f}" y="{y - 2:.0f}" text-anchor="middle" '
            f'fill="#e8e8f5" font-size="11" font-weight="bold" '
            f'font-family="monospace">{name[:14]}</text>',
            f'<text x="{x:.0f}" y="{y + 13:.0f}" text-anchor="middle" '
            f'fill="#7ec8fa" font-size="8.5" font-family="monospace">'
            f'{n_rows:,} rows · {n_cols} cols</text>',
        ]

    svg_lines.append("</svg>")
    st.markdown("\n".join(svg_lines), unsafe_allow_html=True)


# ── quality report ────────────────────────────────────────────────────────────

def _render_quality_report(report: list[dict]):
    st.markdown("#### 📊 Join Quality Report")
    for step in report:
        delta = step["row_delta"]
        before = step["rows_before"]
        color  = _quality_color(delta, before)
        sign   = "+" if delta >= 0 else ""
        warn   = ""
        if step["unmatched_left"] > 0:
            warn += f" ⚠️ {step['unmatched_left']:,} left-side keys had no match."
        if step["unmatched_right"] > 0:
            warn += f" ⚠️ {step['unmatched_right']:,} right-side keys had no match."
        if abs(delta) / max(before, 1) > 0.5:
            warn += " 🔴 Large row count change — check join keys and type."

        st.markdown(
            f'<div style="background:#16172a;border-left:4px solid {color};'
            f'border-radius:6px;padding:10px 14px;margin:6px 0;">'
            f'<div style="color:#e8e8f5;font-weight:700;">Step {step["step"]}: '
            f'{step["description"]}</div>'
            f'<div style="color:#aaa;font-size:0.85rem;margin:4px 0;">'
            f'Join type: <strong>{step["join_type"].upper()}</strong> · '
            f'Keys: <code>{step["left_key"]} → {step["right_key"]}</code>'
            f'</div>'
            f'<div style="color:{color};font-size:0.88rem;">'
            f'Rows: {before:,} → {step["rows_after"]:,} '
            f'(<strong>{sign}{delta:,}</strong>)'
            f'</div>'
            f'{"<div style=\"color:#fcd34d;font-size:0.82rem;margin-top:4px;\">" + warn + "</div>" if warn else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_relationship_diagnostics(diagnostics: list[dict]):
    if not diagnostics:
        return

    st.markdown("#### Relationship Diagnostics")
    st.caption(
        "These checks verify cardinality and inferred PK/FK behavior from CSV values. "
        "CSV files do not contain actual database constraints."
    )

    rows = []
    for d in diagnostics:
        rows.append({
            "relationship": d.get("relationship"),
            "cardinality": d.get("cardinality"),
            "pk_candidates": d.get("pk_candidates"),
            "fk_check": d.get("fk_check"),
            "A values in B %": d.get("a_values_found_in_b_pct"),
            "B values in A %": d.get("b_values_found_in_a_pct"),
            "constraint": d.get("constraint_type"),
            "status": d.get("status"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for d in diagnostics:
        status = d.get("status")
        if status == "error":
            alert(
                f"Many-to-many risk in <strong>{d.get('relationship')}</strong>. "
                "Do not join this directly unless you aggregate or deduplicate one side first.",
                "error",
            )
        elif status == "warning":
            alert(
                f"Low FK coverage for <strong>{d.get('relationship')}</strong>. "
                "Verify this key before using it in the final dataset.",
                "warning",
            )


def _table_signature(tables: dict[str, pd.DataFrame]) -> tuple:
    return tuple(
        (name, int(len(df)), tuple(str(c) for c in df.columns))
        for name, df in sorted(tables.items())
    )


def _build_column_review_rows(table_name: str, df: pd.DataFrame, descs: dict) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        examples = descs.get(col, {}).get("examples")
        if not examples:
            examples = s.dropna().astype(str).unique()[:3].tolist()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "unique": int(s.nunique()),
            "unique_ratio": round(float(s.nunique() / max(len(df), 1)), 3),
            "null_pct": round(float(s.isna().mean() * 100), 1),
            "type_label": descs.get(col, {}).get("type_label", "Unknown"),
            "description": descs.get(col, {}).get("meaning", ""),
            "examples": ", ".join(str(x) for x in examples[:3]),
        })
    return rows


def _guess_base_table(tables: dict[str, pd.DataFrame]) -> str:
    priority = [
        "claims",
        "claim",
        "transactions",
        "transaction",
        "orders",
        "order",
        "events",
        "event",
        "visits",
        "visit",
        "encounters",
        "encounter",
        "payments",
        "payment",
    ]
    names = list(tables.keys())
    for token in priority:
        for name in names:
            if token in name.lower():
                return name
    return max(names, key=lambda n: len(tables[n])) if names else ""


def _validate_join_goal(plan: list[dict], join_goal: dict, tables: dict[str, pd.DataFrame]) -> list[str]:
    warnings = []
    if not plan or not join_goal.get("preserve_base_rows"):
        return warnings

    base_table = join_goal.get("base_table")
    if base_table and plan[0].get("left") != base_table:
        warnings.append(
            f"First join starts from `{plan[0].get('left')}` but the selected final grain is `{base_table}`."
        )

    for i, step in enumerate(plan):
        join_type = step.get("join_type", "left")
        if join_type == "inner":
            warnings.append(
                f"Step {i+1} uses INNER join. That can drop `{base_table}` rows and violate the selected grain."
            )
        if i > 0 and step.get("left") != "merged_so_far":
            warnings.append(
                f"Step {i+1} does not continue from `merged_so_far`; this can change the final dataset grain."
            )

    return warnings


def _render_column_dictionary(tables: dict[str, pd.DataFrame], descs_by_table: dict) -> dict:
    st.markdown("### 🧠 Review Table Schemas")
    st.caption(
        "AI describes every column first. Review or edit these descriptions before generating the join plan."
    )

    type_options = [
        "Unique ID",
        "Numeric (continuous)",
        "Numeric (count/integer)",
        "Categorical (binary)",
        "Categorical (ordinal)",
        "Categorical (nominal)",
        "Text / Name",
        "Date / Time",
        "Boolean",
        "Unknown",
    ]

    reviewed = {}
    tabs = st.tabs(list(tables.keys()))
    for tab, (table_name, df) in zip(tabs, tables.items()):
        with tab:
            rows = _build_column_review_rows(table_name, df, descs_by_table.get(table_name, {}))
            edited = st.data_editor(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                disabled=["column", "dtype", "unique", "unique_ratio", "null_pct", "examples"],
                column_config={
                    "type_label": st.column_config.SelectboxColumn("type_label", options=type_options),
                    "description": st.column_config.TextColumn("description", width="large"),
                },
                key=f"_fusion_col_editor_{table_name}",
            )

            table_descs = {}
            for _, row in edited.iterrows():
                col = row["column"]
                prior = descs_by_table.get(table_name, {}).get(col, {})
                table_descs[col] = {
                    "meaning": str(row.get("description") or ""),
                    "type_label": str(row.get("type_label") or "Unknown"),
                    "examples": prior.get("examples", []),
                }
            reviewed[table_name] = table_descs

    return reviewed


# ── main page ─────────────────────────────────────────────────────────────────

def page_fusion():
    st.markdown(
        f'{badge("Phase 0")} <h1 style="display:inline;margin-left:.5rem;">'
        f'Data Fusion</h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Choose whether you have one CSV file or multiple related CSV files, then prepare one dataset for the pipeline."
    )

    api_key = st.session_state.get("openai_key", "")
    if not api_key:
        alert("⚠️ Enter your OpenAI API key in the sidebar to enable AI schema analysis.", "warning")

    tid = st.session_state.get("tid")
    ps = st.session_state.get("pipeline_state", {})
    if tid and ps.get("df_parquet_b64"):
        from ui.pages.ingest import _render_existing_session
        _render_existing_session(ps, api_key, tid)
        return

    # ── State keys ────────────────────────────────────────────────────────────
    _tables_key  = "_fusion_tables"        # {safe_name: df}
    _schema_key  = "_fusion_schema"        # analyse_schema result dict
    _plan_key    = "_fusion_plan"          # user-confirmed join plan list
    _result_key  = "_fusion_result"        # merged df
    _report_key  = "_fusion_report"        # quality report list
    _desc_key    = "_fusion_column_descriptions"
    _sig_key     = "_fusion_tables_signature"
    _goal_key    = "_fusion_join_goal"
    _mode_key    = "_fusion_mode"
    _source_key  = "_fusion_source_count"

    join_mode_options = ["🤖 AI auto-detect", "✍️ Manual / SQL"]
    if st.session_state.get(_mode_key) not in join_mode_options:
        st.session_state[_mode_key] = join_mode_options[0]

    source_choice = st.radio(
        "How many CSV files do you have?",
        ["1 CSV file", "Multiple CSV files"],
        horizontal=True,
        key=_source_key,
    )

    if source_choice == "1 CSV file":
        for k in (_tables_key, _schema_key, _plan_key, _result_key, _report_key, _desc_key, _sig_key, _goal_key):
            st.session_state.pop(k, None)

        st.markdown("### 📁 Upload CSV File")
        uploaded = st.file_uploader(
            "Upload one CSV file",
            type=["csv"],
            key="_fusion_single_uploader",
        )
        if not uploaded:
            return

        content = uploaded.read()
        if len(content) > 100 * 1_048_576:
            alert("File too large (max 100 MB)", "error")
            return

        _df_cache_key = f"_df_{uploaded.name}_{len(content)}"
        for k in [k for k in st.session_state if k.startswith("_df_") and k != _df_cache_key]:
            del st.session_state[k]

        if _df_cache_key not in st.session_state:
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                alert(f"Cannot parse CSV: {e}", "error")
                return
            st.session_state[_df_cache_key] = df
        else:
            df = st.session_state[_df_cache_key]

        from ui.pages.ingest import _render_upload_ui
        _render_upload_ui(df, uploaded.name, api_key)
        return

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload CSV Files")
    uploaded_files = st.file_uploader(
        "Upload 2–10 related CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="_fusion_uploader",
    )

    # Parse and cache uploaded files
    if uploaded_files:
        tables: dict[str, pd.DataFrame] = {}
        parse_errors = []
        for f in uploaded_files:
            content = f.read()
            try:
                df = pd.read_csv(io.BytesIO(content))
                safe = _safe_name(f.name)
                tables[safe] = df
            except Exception as e:
                parse_errors.append(f"{f.name}: {e}")

        if parse_errors:
            for err in parse_errors:
                alert(f"Parse error — {err}", "error")

        if tables:
            st.session_state[_tables_key] = tables
            sig = _table_signature(tables)
            if st.session_state.get(_sig_key) != sig:
                st.session_state[_sig_key] = sig
                for k in (_schema_key, _plan_key, _result_key, _report_key, _desc_key, _goal_key):
                    st.session_state.pop(k, None)
    else:
        # Clear state when files are removed
        for k in (_tables_key, _schema_key, _plan_key, _result_key, _report_key, _desc_key, _sig_key, _goal_key):
            st.session_state.pop(k, None)

    tables: dict = st.session_state.get(_tables_key, {})
    if len(tables) < 2:
        if tables:
            alert("Upload at least 2 CSV files to perform a join.", "warning")
        return

    # ── Table previews ────────────────────────────────────────────────────────
    st.markdown("### 🗂 Uploaded Tables")
    metrics_row([(name, f"{len(df):,} rows × {len(df.columns)} cols")
                 for name, df in tables.items()])

    with st.expander("📋 Preview tables", expanded=False):
        tabs = st.tabs(list(tables.keys()))
        for tab, (name, df) in zip(tabs, tables.items()):
            with tab:
                st.dataframe(df.head(5), use_container_width=True)
                st.caption(f"Columns: {', '.join(df.columns.tolist())}")

    # ── Column dictionary review ──────────────────────────────────────────────
    descs_by_table = st.session_state.get(_desc_key)
    if descs_by_table is None:
        st.markdown("### 🧠 Analyze Table Schemas")
        st.caption(
            "First, AI will describe each column in every CSV. These reviewed descriptions are then passed to the join planner."
        )
        if not api_key:
            alert("Enter your OpenAI API key to analyze column descriptions before AI join planning.", "warning")
        if st.button(
            "🧠 Analyze columns in all tables",
            key="_fusion_analyse_columns",
            disabled=not api_key,
        ):
            descs_by_table = {}
            with st.spinner("AI is describing columns across all uploaded tables..."):
                for table_name, df in tables.items():
                    try:
                        descs_by_table[table_name] = column_descriptions(df, table_name, api_key)
                    except Exception:
                        descs_by_table[table_name] = _fallback_descriptions(df)
            st.session_state[_desc_key] = descs_by_table
            st.session_state.pop(_schema_key, None)
            st.session_state.pop(_plan_key, None)
            st.session_state.pop(_result_key, None)
            st.session_state.pop(_report_key, None)
            st.rerun()
    else:
        reviewed_descs = _render_column_dictionary(tables, descs_by_table)
        if reviewed_descs != descs_by_table:
            st.session_state[_desc_key] = reviewed_descs
            descs_by_table = reviewed_descs
            st.session_state.pop(_schema_key, None)
            st.session_state.pop(_plan_key, None)
            st.session_state.pop(_result_key, None)
            st.session_state.pop(_report_key, None)
        if st.button("🔄 Re-analyze column descriptions", key="_fusion_reanalyse_columns"):
            st.session_state.pop(_desc_key, None)
            st.session_state.pop(_schema_key, None)
            st.session_state.pop(_plan_key, None)
            st.session_state.pop(_result_key, None)
            st.session_state.pop(_report_key, None)
            st.rerun()

    # ── Mode selector ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Join Mode")
    mode = st.radio(
        "How would you like to define the joins?",
        join_mode_options,
        horizontal=True,
        key=_mode_key,
    )
    use_sql = (mode == join_mode_options[1])

    st.markdown("### Human Review: Final Dataset Grain")
    table_names = list(tables.keys())
    saved_goal = st.session_state.get(_goal_key, {})
    default_base = saved_goal.get("base_table") or _guess_base_table(tables)
    base_idx = table_names.index(default_base) if default_base in table_names else 0
    grain_col1, grain_col2 = st.columns([1.4, 1])
    with grain_col1:
        base_table = st.selectbox(
            "One row in final dataset should represent",
            table_names,
            index=base_idx,
            key="_fusion_base_table",
            help="For healthcare fraud, choose claims so the final dataset has one row per claim.",
        )
    with grain_col2:
        preserve_base_rows = st.checkbox(
            f"Preserve {base_table} row count",
            value=saved_goal.get("preserve_base_rows", True),
            key="_fusion_preserve_rows",
        )
    extra_instructions = st.text_area(
        "Instructions for AI join planning",
        value=saved_goal.get(
            "instructions",
            f"Use {base_table} as the base table. The final dataset should have {len(tables[base_table]):,} rows, one row per {base_table} record.",
        ),
        key="_fusion_join_instructions",
        height=90,
    )
    join_goal = {
        "base_table": base_table,
        "base_row_count": int(len(tables[base_table])),
        "preserve_base_rows": bool(preserve_base_rows),
        "instructions": extra_instructions,
    }
    if join_goal != saved_goal:
        st.session_state[_goal_key] = join_goal
        st.session_state.pop(_schema_key, None)
        st.session_state.pop(_plan_key, None)
        st.session_state.pop(_result_key, None)
        st.session_state.pop(_report_key, None)
    if preserve_base_rows:
        st.info(
            f"Join planner will preserve `{base_table}` as the final dataset grain "
            f"({len(tables[base_table]):,} rows expected)."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # AUTO MODE — LLM schema analysis + editable plan
    # ══════════════════════════════════════════════════════════════════════════
    if not use_sql:
        st.markdown("### 🤖 Optimal Join Planning")

        schema = st.session_state.get(_schema_key)
        if schema is None:
            if not api_key:
                alert("API key required for AI schema analysis. Switch to Manual/SQL mode or enter your key.", "warning")
                return
            if not descs_by_table:
                alert("Analyze and review the table schemas first, then generate the optimal joins.", "warning")
                return
            if st.button("🔍 Generate optimal joins", key="_fusion_analyse"):
                with st.spinner("AI is using the reviewed schemas to choose optimal joins..."):
                    schema = analyse_schema(tables, api_key, descs_by_table, join_goal)
                    st.session_state[_schema_key] = schema
                    # Pre-populate plan from AI suggestion
                    st.session_state[_plan_key] = schema.get("join_order", [])
                st.rerun()
            return

        # Show summary
        if schema.get("summary"):
            st.info(f"🧠 {schema['summary']}")

        if schema.get("error"):
            alert(f"Schema analysis error: {schema['error']}", "error")

        # ── ER Diagram ────────────────────────────────────────────────────────
        rels = schema.get("relationships", [])
        if rels:
            with st.expander("🗺 Schema diagram", expanded=True):
                _render_er_diagram(tables, rels)
            with st.expander("✅ Verify PK/FK and cardinality", expanded=True):
                _render_relationship_diagnostics(relationship_diagnostics(tables, rels))

        # ── Detected relationships ────────────────────────────────────────────
        st.markdown("#### 🔗 Detected Relationships")
        if not rels:
            alert("No relationships detected. Try Manual/SQL mode.", "warning")
        else:
            for i, rel in enumerate(rels):
                conf  = rel.get("confidence", 0)
                jtype = rel.get("join_type", "left")
                st.markdown(
                    f'<div style="background:#16172a;border-radius:6px;'
                    f'padding:10px 14px;margin:5px 0;border:1px solid #2a2b45;">'
                    f'<span style="color:#e8e8f5;font-weight:700;">'
                    f'<code>{rel.get("table_a")}.{rel.get("key_a")}</code>'
                    f' ──── <code>{rel.get("table_b")}.{rel.get("key_b")}</code>'
                    f'</span>&nbsp;&nbsp;'
                    f'<span style="background:#12132a;color:{_join_color(jtype)};'
                    f'padding:2px 7px;border-radius:4px;font-size:0.78rem;'
                    f'border:1px solid {_join_color(jtype)}55;">{jtype.upper()}</span>'
                    f'&nbsp;{_conf_badge(conf)}<br>'
                    f'<span style="color:#8888aa;font-size:0.82rem;">'
                    f'{rel.get("reasoning","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Editable join plan ────────────────────────────────────────────────
        st.markdown("#### ✏️ Review & Edit Join Plan")
        st.caption("Adjust join types, keys, or order before executing.")

        plan: list[dict] = st.session_state.get(_plan_key, [])
        table_options = list(tables.keys()) + ["merged_so_far"]
        all_cols      = {name: list(df.columns) for name, df in tables.items()}

        updated_plan = []
        for i, step in enumerate(plan):
            with st.expander(
                f"Step {step.get('step', i+1)}: {step.get('description', '')}",
                expanded=True,
            ):
                c1, c2, c3 = st.columns(3)
                left = c1.selectbox(
                    "Left table", table_options,
                    index=table_options.index(step.get("left", table_options[0]))
                          if step.get("left") in table_options else 0,
                    key=f"_fp_left_{i}",
                )
                right = c2.selectbox(
                    "Right table",
                    [t for t in table_options if t != "merged_so_far"],
                    index=max(0, [t for t in table_options if t != "merged_so_far"]
                              .index(step.get("right", "")))
                    if step.get("right") in table_options else 0,
                    key=f"_fp_right_{i}",
                )
                join_type = c3.selectbox(
                    "Join type", ["left", "inner", "outer"],
                    index=["left", "inner", "outer"].index(step.get("join_type", "left")),
                    key=f"_fp_jt_{i}",
                )

                left_cols  = all_cols.get(left, []) if left != "merged_so_far" else []
                right_cols = all_cols.get(right, [])

                c4, c5 = st.columns(2)
                # For left key, allow freetext if left is merged_so_far
                if left == "merged_so_far" or not left_cols:
                    left_key = c4.text_input(
                        "Left key (merged result column)",
                        value=step.get("left_key", ""),
                        key=f"_fp_lk_{i}",
                    )
                else:
                    lk_idx = left_cols.index(step["left_key"]) if step.get("left_key") in left_cols else 0
                    left_key = c4.selectbox("Left key", left_cols, index=lk_idx, key=f"_fp_lk_{i}")

                rk_idx   = right_cols.index(step["right_key"]) if step.get("right_key") in right_cols else 0
                right_key = c5.selectbox("Right key", right_cols, index=rk_idx, key=f"_fp_rk_{i}")

                updated_plan.append({
                    "step":        i + 1,
                    "left":        left,
                    "right":       right,
                    "left_key":    left_key,
                    "right_key":   right_key,
                    "join_type":   join_type,
                    "description": step.get("description", f"Merge {left} ← {right}"),
                })

        # Add step button
        if st.button("➕ Add join step", key="_fusion_add_step"):
            plan.append({
                "step":        len(plan) + 1,
                "left":        "merged_so_far",
                "right":       list(tables.keys())[-1],
                "left_key":    "",
                "right_key":   "",
                "join_type":   "left",
                "description": "New join step",
            })
            st.session_state[_plan_key] = plan
            st.rerun()

        st.session_state[_plan_key] = updated_plan

        grain_warnings = _validate_join_goal(updated_plan, join_goal, tables)
        if grain_warnings:
            alert(
                "HITL grain check needs attention:<br>"
                + "<br>".join(f"• {w}" for w in grain_warnings),
                "warning",
            )

        # Re-analyse button
        if st.button("🔄 Re-analyse schema", key="_fusion_reanalyse"):
            st.session_state.pop(_schema_key, None)
            st.session_state.pop(_plan_key, None)
            st.session_state.pop(_result_key, None)
            st.session_state.pop(_report_key, None)
            st.rerun()

        # ── Execute ───────────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("⚡ Execute join plan →", key="_fusion_execute",
                     disabled=not updated_plan):
            with st.spinner("Executing joins…"):
                merged, report, err = execute_joins(
                    tables,
                    updated_plan,
                    expected_rows=join_goal["base_row_count"] if join_goal.get("preserve_base_rows") else None,
                    expected_rows_label=f"`{join_goal['base_table']}`",
                )
            if err:
                alert(f"Join error: {err}", "error")
            else:
                st.session_state[_result_key] = merged
                st.session_state[_report_key] = report
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # MANUAL / SQL MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### ✍️ SQL Query")

        table_list = "\n".join(
            f"  • **{name}** — columns: {', '.join(df.columns[:8].tolist())}"
            + ("…" if len(df.columns) > 8 else "")
            for name, df in tables.items()
        )
        st.info(
            f"Available tables (use these names in your SQL):\n{table_list}\n\n"
            "Standard SQL SELECT/JOIN/WHERE/GROUP BY supported via DuckDB."
        )

        default_sql = "SELECT *\nFROM " + list(tables.keys())[0]
        if len(tables) >= 2:
            t1, t2   = list(tables.keys())[:2]
            df1, df2 = tables[t1], tables[t2]
            # Guess a likely join key
            shared = set(df1.columns) & set(df2.columns)
            if shared:
                key = next(iter(shared))
                default_sql = (
                    f"SELECT *\nFROM {t1}\nLEFT JOIN {t2}\n"
                    f"  ON {t1}.{key} = {t2}.{key}"
                )

        sql = st.text_area(
            "Write your SQL query",
            value=st.session_state.get("_fusion_sql", default_sql),
            height=160,
            key="_fusion_sql_input",
        )
        st.session_state["_fusion_sql"] = sql

        if st.button("▶ Run query", key="_fusion_run_sql"):
            with st.spinner("Running SQL query…"):
                result, err = execute_sql_query(tables, sql)
            if err:
                alert(f"SQL error: {err}", "error")
            elif join_goal.get("preserve_base_rows") and len(result) != join_goal["base_row_count"]:
                alert(
                    f"SQL result has {len(result):,} rows, but the HITL grain constraint expects "
                    f"{join_goal['base_row_count']:,} rows from `{join_goal['base_table']}`.",
                    "error",
                )
            else:
                st.session_state[_result_key] = result
                st.session_state[_report_key] = []
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RESULT
    # ══════════════════════════════════════════════════════════════════════════
    merged: pd.DataFrame | None = st.session_state.get(_result_key)
    if merged is None:
        return

    st.markdown("---")
    st.markdown("### ✅ Merged Dataset")
    metrics_row([
        ("Rows",         f"{len(merged):,}"),
        ("Columns",      len(merged.columns)),
        ("Missing cells", int(merged.isna().sum().sum())),
        ("Null %",       f"{merged.isna().mean().mean()*100:.1f}%"),
    ])
    st.dataframe(merged.head(8), use_container_width=True)

    report = st.session_state.get(_report_key, [])
    if report:
        with st.expander("📊 Join quality report", expanded=True):
            _render_quality_report(report)

    # ── Continue with merged dataset ──────────────────────────────────────────
    st.markdown("---")
    _, col_download = st.columns([2, 1])

    with col_download:
        csv_bytes = merged.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download merged CSV",
            data=csv_bytes,
            file_name="merged_dataset.csv",
            mime="text/csv",
            key="_fusion_download",
        )

    from ui.pages.ingest import _render_upload_ui
    _render_upload_ui(merged, "merged_dataset.csv", api_key)
