"""ui/pages/ingest.py — Data Fusion single-dataset setup helpers.

New in this version:
  * Column Intelligence  — LLM explains every column in plain English
                           (meaning, human-readable type label, example values)
  * Target Validation   — warns/blocks on:
      - categorical target + regression task   (warning)
      - binary target + regression             (warning)
      - continuous numeric + classification    (warning)
      - target is a name / email / description (error — blocks pipeline)
      - target is a unique ID / row number     (error — blocks pipeline)
      - target is a date/time column           (error — blocks pipeline)
      - high-cardinality text target           (error — blocks pipeline)
      - constant / single-value target         (error — blocks pipeline)
"""
from __future__ import annotations
import io
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from db.session_store import upsert_session
from utils.serialization import df_to_b64
from ui.components import alert, badge, metrics_row
from ui.graph_helpers import run_graph_sync, safe_state, persist
from ui.state_store import update_store
from utils.column_intelligence import column_descriptions, validate_target, _fallback_descriptions


# ── design tokens (match dark theme) ─────────────────────────────────────────
_TYPE_LABEL_COLORS = {
    "Unique ID":               ("#e15759", "#2e1010"),
    "Text / Name":             ("#f28e2b", "#2e1e08"),
    "Date / Time":             ("#76b7b2", "#0d1e1e"),
    "Boolean":                 ("#b07aa1", "#1e0d1e"),
    "Categorical (binary)":    ("#59a14f", "#0d2e18"),
    "Categorical (ordinal)":   ("#7ec8fa", "#0d1e2e"),
    "Categorical (nominal)":   ("#7ec8fa", "#0d1e2e"),
    "Numeric (continuous)":    ("#c6f135", "#1a2e08"),
    "Numeric (count/integer)": ("#edc948", "#2e2408"),
    "Unknown":                 ("#aaa",    "#1a1a2e"),
}

_SEVERITY_ICON = {"ok": "✅", "warning": "⚠️", "error": "❌"}


def _type_badge(label: str) -> str:
    fg, bg = _TYPE_LABEL_COLORS.get(label, ("#aaa", "#1a1a2e"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:0.72rem;font-weight:600;'
        f'border:1px solid {fg}33;">{label}</span>'
    )


def _render_column_intelligence(df: pd.DataFrame, descs: dict, ci_cache_key: str, api_key: str = ""):
    """Three-step column description flow:

    Step 1 — User input  : for each column, user can type their own description
                           or leave blank to skip.
    Step 2 — AI compares : for any column where the user provided a description,
                           the AI compares it against its own and recommends the better one.
    Step 3 — User chooses: per column, user sees both options and picks one
                           (or edits freely). Final choice is written into descs.
    """
    from utils.column_intelligence import compare_descriptions, _build_col_payload

    st.markdown("### 🧠 Column Intelligence")

    # ── State keys ────────────────────────────────────────────────────────────
    _user_inputs_key  = f"{ci_cache_key}__user_inputs"   # {col: str}
    _comparisons_key  = f"{ci_cache_key}__comparisons"   # {col: compare result dict}
    _final_key        = f"{ci_cache_key}__final"          # {col: chosen str}
    _step_key         = f"{ci_cache_key}__step"           # 1 | 2 | 3

    if _step_key not in st.session_state:
        st.session_state[_step_key] = 1
    if _user_inputs_key not in st.session_state:
        st.session_state[_user_inputs_key] = {}
    if _comparisons_key not in st.session_state:
        st.session_state[_comparisons_key] = {}
    if _final_key not in st.session_state:
        # Initialise final choices to AI descriptions
        st.session_state[_final_key] = {
            col: descs.get(col, {}).get("meaning", "—") for col in df.columns
        }

    step         = st.session_state[_step_key]
    user_inputs  = st.session_state[_user_inputs_key]
    comparisons  = st.session_state[_comparisons_key]
    final_picks  = st.session_state[_final_key]

    # ── Step indicator ────────────────────────────────────────────────────────
    steps_html = ""
    for i, (label, desc_s) in enumerate([
        ("1 · Your descriptions", "Optionally describe each column"),
        ("2 · AI comparison",     "AI picks the better description"),
        ("3 · Final choice",      "You confirm or edit each column"),
    ], 1):
        active   = step == i
        done     = step > i
        fg       = "#c6f135" if active else ("#59a14f" if done else "#555")
        bg       = "#1a2e08" if active else ("#0d2e18" if done else "#12132a")
        border   = "#c6f135" if active else ("#59a14f" if done else "#2a2b45")
        steps_html += (
            f'<div style="background:{bg};border:1px solid {border};border-radius:6px;'
            f'padding:8px 14px;flex:1;margin:0 4px;">'
            f'<div style="color:{fg};font-weight:700;font-size:0.85rem;">{"✅ " if done else ""}{label}</div>'
            f'<div style="color:#8888aa;font-size:0.76rem;">{desc_s}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:4px;margin-bottom:16px;">{steps_html}</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — User input
    # ══════════════════════════════════════════════════════════════════════════
    if step == 1:
        st.caption(
            "Optionally type your own description for any column below. "
            "Leave blank to skip — AI will use its own description for that column."
        )

        col_payload = {r["name"]: r for r in _build_col_payload(df)}

        for col in df.columns:
            ai_meaning  = descs.get(col, {}).get("meaning", "—")
            type_label  = descs.get(col, {}).get("type_label", "Unknown")
            examples    = descs.get(col, {}).get("examples", [])
            ex_str      = " · ".join(str(e) for e in examples[:3]) if examples else "—"
            null_pct    = df[col].isna().mean() * 100

            with st.container():
                h1, h2, h3 = st.columns([2, 1.2, 0.8])
                h1.markdown(
                    f'<span style="font-weight:700;color:#e8e8f5;">{col}</span> '
                    f'{_type_badge(type_label)}',
                    unsafe_allow_html=True,
                )
                h2.markdown(
                    f'<span style="color:#8888aa;font-size:0.8rem;font-style:italic;">{ex_str}</span>',
                    unsafe_allow_html=True,
                )
                h3.markdown(
                    f'<span style="color:{"#e15759" if null_pct > 20 else "#555"};'
                    f'font-size:0.8rem;">null: {null_pct:.1f}%</span>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="color:#7ec8fa;font-size:0.82rem;margin-bottom:2px;">'
                    f'🤖 AI: <em>{ai_meaning}</em></div>',
                    unsafe_allow_html=True,
                )

                user_val = st.text_input(
                    f"Your description for **{col}** (optional — press Enter to save)",
                    value=user_inputs.get(col, ""),
                    key=f"{ci_cache_key}__inp_{col}",
                    placeholder="Leave blank to keep AI description…",
                    label_visibility="collapsed",
                )
                user_inputs[col] = user_val.strip()

                st.markdown(
                    '<hr style="border:none;border-top:1px solid #1e1f35;margin:6px 0;">',
                    unsafe_allow_html=True,
                )

        # Save inputs to session_state
        st.session_state[_user_inputs_key] = user_inputs

        filled   = sum(1 for v in user_inputs.values() if v)
        skipped  = len(df.columns) - filled

        c1, c2 = st.columns([3, 1])
        c1.caption(
            f"**{filled}** column(s) with your description · "
            f"**{skipped}** will keep the AI description."
        )

        btn_label = "🤖 Get AI recommendations →" if filled > 0 else "✅ Use AI descriptions →"
        if c2.button(btn_label, key=f"{ci_cache_key}__step1_next", use_container_width=True):
            if filled == 0:
                # No user input — skip straight to step 3 with AI descriptions
                st.session_state[_step_key] = 3
            else:
                st.session_state[_step_key] = 2
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — AI comparison (only for columns where user gave input)
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 2:
        cols_to_compare = [c for c, v in user_inputs.items() if v]
        already_done    = set(comparisons.keys())
        todo            = [c for c in cols_to_compare if c not in already_done]

        if todo:
            col_payload = {r["name"]: r for r in _build_col_payload(df)}
            progress    = st.progress(0, text="AI is comparing descriptions…")

            for i, col in enumerate(todo):
                progress.progress(
                    (i + 1) / len(todo),
                    text=f"Comparing descriptions for **{col}**… ({i+1}/{len(todo)})",
                )
                ai_desc   = descs.get(col, {}).get("meaning", "—")
                user_desc = user_inputs[col]
                result    = compare_descriptions(
                    col, ai_desc, user_desc,
                    col_payload.get(col, {}),
                    api_key,
                )
                comparisons[col] = result

            progress.empty()
            st.session_state[_comparisons_key] = comparisons

        st.session_state[_step_key] = 3
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — User makes final choice per column
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3:
        st.caption(
            "Review each column. For columns the AI compared, its recommendation "
            "is pre-selected — click the other option or edit freely."
        )

        for col in df.columns:
            ai_meaning  = descs.get(col, {}).get("meaning", "—")
            type_label  = descs.get(col, {}).get("type_label", "Unknown")
            user_desc   = user_inputs.get(col, "")
            comp        = comparisons.get(col)     # None if user skipped this col

            with st.container():
                st.markdown(
                    f'<div style="font-weight:700;color:#e8e8f5;margin-bottom:4px;">'
                    f'{col} {_type_badge(type_label)}</div>',
                    unsafe_allow_html=True,
                )

                if comp:
                    # Show both options with AI recommendation pre-selected
                    rec    = comp["recommendation"]   # "ai" | "user" | "synthesised"
                    rec_txt= comp["recommended_text"]
                    reason = comp["reasoning"]

                    st.markdown(
                        f'<div style="background:#12132a;border-left:3px solid #7ec8fa;'
                        f'border-radius:4px;padding:6px 10px;margin-bottom:6px;font-size:0.82rem;'
                        f'color:#7ec8fa;">🤖 AI recommends: <strong>{rec}</strong> — {reason}</div>',
                        unsafe_allow_html=True,
                    )

                    options = {}
                    options[f"🤖 AI: {ai_meaning}"] = ai_meaning
                    options[f"👤 You: {user_desc}"] = user_desc
                    if rec == "synthesised" and rec_txt not in (ai_meaning, user_desc):
                        options[f"✨ Synthesised: {rec_txt}"] = rec_txt

                    # Default selection = recommended option
                    if rec == "user":
                        default_label = f"👤 You: {user_desc}"
                    elif rec == "synthesised" and f"✨ Synthesised: {rec_txt}" in options:
                        default_label = f"✨ Synthesised: {rec_txt}"
                    else:
                        default_label = f"🤖 AI: {ai_meaning}"

                    option_labels = list(options.keys())
                    default_idx   = option_labels.index(default_label) if default_label in option_labels else 0

                    chosen_label = st.radio(
                        f"_choose_{col}",
                        option_labels,
                        index=default_idx,
                        key=f"{ci_cache_key}__radio_{col}",
                        label_visibility="collapsed",
                        horizontal=True,
                    )
                    chosen_text = options[chosen_label]

                else:
                    # No user input for this col — show AI description, allow free edit
                    chosen_text = ai_meaning

                # Free-edit box (pre-filled with chosen/AI text)
                final_text = st.text_input(
                    f"Final description for {col}",
                    value=final_picks.get(col, chosen_text),
                    key=f"{ci_cache_key}__final_{col}",
                    label_visibility="collapsed",
                )
                final_picks[col] = final_text.strip() or chosen_text

                st.markdown(
                    '<hr style="border:none;border-top:1px solid #1e1f35;margin:6px 0;">',
                    unsafe_allow_html=True,
                )

        st.session_state[_final_key] = final_picks

        # Apply finals back into descs + ci_cache
        for col, text in final_picks.items():
            if col in descs and text:
                descs[col]["meaning"] = text
                cached = st.session_state.get(ci_cache_key)
                if isinstance(cached, dict) and col in cached:
                    cached[col]["meaning"] = text

        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
        if nav_col1.button("← Back", key=f"{ci_cache_key}__back", use_container_width=True):
            st.session_state[_step_key] = 1
            st.session_state[_comparisons_key] = {}
            st.rerun()

        if nav_col2.button(
            "🔄 Reset",
            key=f"{ci_cache_key}__reset",
            help="Clear all inputs and restart from Step 1.",
            use_container_width=True,
        ):
            for k in (_user_inputs_key, _comparisons_key, _final_key, _step_key,
                      ci_cache_key, f"{ci_cache_key}__done"):
                st.session_state.pop(k, None)
            st.rerun()

        nav_col3.success("✅ Descriptions finalised — scroll down to set your target.")


def _render_column_intelligence_v2(
    df: pd.DataFrame,
    descs: dict,
    ci_cache_key: str,
    api_key: str = "",
    dataset_name: str = "Uploaded dataset",
):
    """User-first column description flow.

    Step 1 collects optional user descriptions without calling AI.
    Step 2 generates AI descriptions once.
    Step 3 lets the user choose/edit the final description per column.
    """
    st.markdown("### Column Intelligence")

    live_descs = st.session_state.get(ci_cache_key)
    if isinstance(live_descs, dict):
        descs = live_descs
    else:
        st.session_state[ci_cache_key] = descs

    _user_inputs_key = f"{ci_cache_key}__user_inputs"
    _ai_descs_key = f"{ci_cache_key}__ai_descs"
    _comparisons_key = f"{ci_cache_key}__comparisons"
    _final_key = f"{ci_cache_key}__final"
    _step_key = f"{ci_cache_key}__step"
    _ai_error_key = f"{ci_cache_key}__ai_error"
    _choice_key = f"{ci_cache_key}__choices"

    if _user_inputs_key not in st.session_state:
        st.session_state[_user_inputs_key] = {}
    if _comparisons_key not in st.session_state:
        st.session_state[_comparisons_key] = {}
    if _final_key not in st.session_state:
        st.session_state[_final_key] = {}
    if _choice_key not in st.session_state:
        st.session_state[_choice_key] = {}
    if _step_key not in st.session_state:
        st.session_state[_step_key] = 1

    if _ai_descs_key in st.session_state:
        descs.clear()
        descs.update(st.session_state[_ai_descs_key])

    step = st.session_state[_step_key]
    user_inputs = st.session_state[_user_inputs_key]
    comparisons = st.session_state[_comparisons_key]
    final_picks = st.session_state[_final_key]
    choices = st.session_state[_choice_key]

    def _queue_ai_generation() -> None:
        for c in df.columns:
            widget_key = f"{ci_cache_key}__inp_{c}"
            if widget_key in st.session_state:
                user_inputs[c] = str(st.session_state.get(widget_key, "")).strip()
        st.session_state[_user_inputs_key] = user_inputs
        st.session_state[_step_key] = 2

    def _generate_ai_descriptions_now() -> dict:
        """Generate AI descriptions immediately and store them in session state.

        All session_state writes happen before any st.spinner() call so that
        st.rerun() fired *after* this function returns is safe from any
        spinner/expander nesting context (which can silently swallow reruns in
        some Streamlit versions).
        """
        # Flush widget values into session state before the spinner hides them
        for c in df.columns:
            widget_key = f"{ci_cache_key}__inp_{c}"
            if widget_key in st.session_state:
                user_inputs[c] = str(st.session_state.get(widget_key, "")).strip()
        st.session_state[_user_inputs_key] = user_inputs

        # Run the LLM call inside the spinner but keep all state writes outside
        generated = None
        error_msg = None
        with st.spinner("AI is analysing column meanings..."):
            try:
                generated = column_descriptions(df, dataset_name, api_key)
            except Exception as exc:
                error_msg = f"AI description generation failed ({exc}). Fallback descriptions were used."

        if generated is None:
            generated = _fallback_descriptions(df)

        # All state writes happen AFTER the spinner context exits
        if error_msg:
            st.session_state[_ai_error_key] = error_msg
        else:
            st.session_state.pop(_ai_error_key, None)

        st.session_state[_ai_descs_key] = generated
        st.session_state[ci_cache_key] = generated
        descs.clear()
        descs.update(generated)

        next_finals = {
            c: generated.get(c, {}).get("meaning", "") or user_inputs.get(c, "")
            for c in df.columns
        }
        st.session_state[_final_key] = next_finals
        st.session_state[_step_key] = 3
        return generated

    def _render_ai_preview(generated: dict):
        if st.session_state.get(_ai_error_key):
            st.warning(st.session_state[_ai_error_key])
        else:
            st.success("AI descriptions generated. Review and choose the final text below.")
        st.dataframe(
            pd.DataFrame([
                {
                    "column": col,
                    "type_label": generated.get(col, {}).get("type_label", "Unknown"),
                    "ai_description": generated.get(col, {}).get("meaning", ""),
                }
                for col in df.columns
            ]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Continue to final choices →", key=f"{ci_cache_key}__continue_after_ai"):
            # Explicitly set step=3 before rerun so it is guaranteed in session state
            st.session_state[_step_key] = 3
            st.rerun()

    steps_html = ""
    for i, (label, desc_s) in enumerate([
        ("1 · Your descriptions", "Optionally describe each column"),
        ("2 · AI descriptions", "AI describes columns once"),
        ("3 · Final choice", "Choose or edit final text"),
    ], 1):
        active = step == i
        done = step > i
        fg = "#c6f135" if active else ("#59a14f" if done else "#555")
        bg = "#1a2e08" if active else ("#0d2e18" if done else "#12132a")
        border = "#c6f135" if active else ("#59a14f" if done else "#2a2b45")
        steps_html += (
            f'<div style="background:{bg};border:1px solid {border};border-radius:6px;'
            f'padding:8px 14px;flex:1;margin:0 4px;">'
            f'<div style="color:{fg};font-weight:700;font-size:0.85rem;">{label}</div>'
            f'<div style="color:#8888aa;font-size:0.76rem;">{desc_s}</div>'
            f'</div>'
        )
    st.markdown(f'<div style="display:flex;gap:4px;margin-bottom:16px;">{steps_html}</div>', unsafe_allow_html=True)

    if step == 1:
        st.caption("Add your own description for any column, or leave it blank. AI runs only in the next step.")
        top_col1, top_col2 = st.columns([3, 1])
        if not api_key:
            top_col1.warning("Enter your OpenAI API key in the sidebar before generating AI descriptions.")
        top_col2.button(
            "Generate AI descriptions",
            key=f"{ci_cache_key}__step1_next_top",
            use_container_width=True,
            disabled=not api_key,
            help="Enter your OpenAI API key first." if not api_key else "",
            on_click=_queue_ai_generation,
        )

        for col in df.columns:
            type_label = descs.get(col, {}).get("type_label", "Unknown")
            examples = descs.get(col, {}).get("examples", [])
            ex_str = " · ".join(str(e) for e in examples[:3]) if examples else "—"
            null_pct = df[col].isna().mean() * 100

            h1, h2, h3 = st.columns([2, 1.2, 0.8])
            h1.markdown(
                f'<span style="font-weight:700;color:#e8e8f5;">{col}</span> {_type_badge(type_label)}',
                unsafe_allow_html=True,
            )
            h2.markdown(
                f'<span style="color:#8888aa;font-size:0.8rem;font-style:italic;">{ex_str}</span>',
                unsafe_allow_html=True,
            )
            h3.markdown(
                f'<span style="color:{"#e15759" if null_pct > 20 else "#555"};font-size:0.8rem;">null: {null_pct:.1f}%</span>',
                unsafe_allow_html=True,
            )
            user_val = st.text_input(
                f"Your description for {col}",
                value=user_inputs.get(col, ""),
                key=f"{ci_cache_key}__inp_{col}",
                placeholder="Optional: describe this column in your own words...",
                label_visibility="collapsed",
            )
            user_inputs[col] = user_val.strip()
            st.markdown('<hr style="border:none;border-top:1px solid #1e1f35;margin:6px 0;">', unsafe_allow_html=True)

        st.session_state[_user_inputs_key] = user_inputs
        filled = sum(1 for v in user_inputs.values() if v)
        skipped = len(df.columns) - filled
        c1, c2 = st.columns([3, 1])
        c1.caption(f"**{filled}** column(s) with your description · **{skipped}** left blank.")
        c2.button(
            "Generate AI descriptions ->",
            key=f"{ci_cache_key}__step1_next",
            use_container_width=True,
            disabled=not api_key,
            help="Enter your OpenAI API key first." if not api_key else "",
            on_click=_queue_ai_generation,
        )
        return st.session_state.get(ci_cache_key, descs)

    if step == 2:
        if _ai_descs_key not in st.session_state:
            _generate_ai_descriptions_now()
        else:
            st.session_state[_step_key] = 3
        step = 3

    if step == 3:
        ai_descs = st.session_state.get(_ai_descs_key, descs)
        if not ai_descs:
            # Guard: if ai_descs somehow empty, fall back and show a clear error
            st.error(
                "AI descriptions are missing — this can happen if the AI call "
                "failed silently. Fallback descriptions are shown below. "
                "Use the Reset button to try again."
            )
            ai_descs = descs or _fallback_descriptions(df)
            st.session_state[_ai_descs_key] = ai_descs
        if st.session_state.get(_ai_error_key):
            st.warning(st.session_state[_ai_error_key])
        st.caption("Choose AI text, your text, or edit the final description directly.")
        for col in df.columns:
            ai_meaning = ai_descs.get(col, {}).get("meaning", "")
            type_label = ai_descs.get(col, {}).get("type_label", "Unknown")
            user_desc = user_inputs.get(col, "")
            comp = comparisons.get(col)

            st.markdown(
                f'<div style="font-weight:700;color:#e8e8f5;margin-bottom:4px;">{col} {_type_badge(type_label)}</div>',
                unsafe_allow_html=True,
            )
            if comp:
                st.markdown(
                    f'<div style="background:#12132a;border-left:3px solid #7ec8fa;border-radius:4px;'
                    f'padding:6px 10px;margin-bottom:6px;font-size:0.82rem;color:#7ec8fa;">'
                    f'AI recommends: <strong>{comp.get("recommendation", "ai")}</strong> - {comp.get("reasoning", "")}</div>',
                    unsafe_allow_html=True,
                )

            options = {f"AI: {ai_meaning}": ai_meaning}
            if user_desc:
                options[f"You: {user_desc}"] = user_desc
            rec_txt = comp.get("recommended_text") if comp else None
            if rec_txt and rec_txt not in options.values():
                options[f"Synthesised: {rec_txt}"] = rec_txt

            option_labels = list(options.keys())
            default_value = final_picks.get(col)
            default_idx = 0
            if default_value in options.values():
                default_idx = list(options.values()).index(default_value)

            current_choice = choices.get(col)
            default_idx = option_labels.index(current_choice) if current_choice in option_labels else default_idx

            chosen_label = st.radio(
                f"_choose_{col}",
                option_labels,
                index=default_idx,
                key=f"{ci_cache_key}__radio_{col}",
                label_visibility="collapsed",
                horizontal=True,
            )
            chosen_text = options[chosen_label]
            if choices.get(col) != chosen_label:
                choices[col] = chosen_label
                final_picks[col] = chosen_text
                st.session_state[f"{ci_cache_key}__final_{col}"] = chosen_text
            final_text = st.text_input(
                f"Final description for {col}",
                value=st.session_state.get(f"{ci_cache_key}__final_{col}", final_picks.get(col, chosen_text)),
                key=f"{ci_cache_key}__final_{col}",
                label_visibility="collapsed",
            )
            final_picks[col] = final_text.strip() or chosen_text
            st.markdown('<hr style="border:none;border-top:1px solid #1e1f35;margin:6px 0;">', unsafe_allow_html=True)

        final_descs = dict(ai_descs)
        for col, text in final_picks.items():
            if col in final_descs and text:
                final_descs[col] = {**final_descs[col], "meaning": text}
        descs.clear()
        descs.update(final_descs)
        st.session_state[ci_cache_key] = final_descs
        st.session_state[_final_key] = final_picks
        st.session_state[_choice_key] = choices

        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
        if nav_col1.button("<- Back", key=f"{ci_cache_key}__back", use_container_width=True):
            st.session_state[_step_key] = 1
            st.session_state[_comparisons_key] = {}
            st.rerun()
        if nav_col2.button("Reset", key=f"{ci_cache_key}__reset", use_container_width=True):
            for k in (_user_inputs_key, _ai_descs_key, _ai_error_key, _comparisons_key, _final_key, _choice_key, _step_key, ci_cache_key):
                st.session_state.pop(k, None)
            for col in df.columns:
                st.session_state.pop(f"{ci_cache_key}__final_{col}", None)
                st.session_state.pop(f"{ci_cache_key}__radio_{col}", None)
            st.rerun()
        nav_col3.success("Descriptions finalised - scroll down to set your target.")
        return st.session_state.get(ci_cache_key, descs)

    return st.session_state.get(ci_cache_key, descs)


def _render_target_issues(validation):
    for issue in validation.issues:
        sev  = issue["severity"]
        icon = _SEVERITY_ICON.get(sev, "ℹ️")
        border_color = {"error": "#e15759", "warning": "#f28e2b", "ok": "#59a14f"}.get(sev, "#aaa")
        bg_color     = {"error": "#2e1010", "warning": "#2e1e08", "ok": "#0d2e18"}.get(sev, "#1a1a2e")
        text_color   = {"error": "#fca5a5", "warning": "#fcd34d", "ok": "#6ee7a0"}.get(sev, "#e8e8f5")
        st.markdown(
            f"""<div style="background:{bg_color};border-left:4px solid {border_color};
                            border-radius:6px;padding:12px 16px;margin:8px 0;">
              <div style="font-weight:700;color:{text_color};">{icon} {issue['title']}</div>
              <div style="color:#c8c8e8;margin:4px 0 6px 0;font-size:0.9rem;">{issue['detail']}</div>
              <div style="color:#7ec8fa;font-size:0.85rem;">
                💡 <strong>How to fix:</strong> {issue['suggestion']}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def page_ingest():
    st.markdown(
        f'{badge("Phase 0")} <h1 style="display:inline;margin-left:.5rem;">Data Fusion</h1>',
        unsafe_allow_html=True,
    )

    api_key  = st.session_state.get("openai_key", "")
    tid      = st.session_state.get("tid")
    ps       = st.session_state.get("pipeline_state", {})

    # ── RETURNING MODE: pipeline already started ──────────────────────────────
    # When the user navigates away and comes back to Data Fusion, we restore
    # everything from pipeline_state (graph store) — no re-upload, no LLM call.
    if tid and ps.get("df_parquet_b64"):
        _render_existing_session(ps, api_key, tid)
        return

    # ── FRESH UPLOAD MODE ─────────────────────────────────────────────────────
    st.caption("Upload your CSV — AI will explain every column, then analyse your dataset automatically.")

    if not api_key:
        alert("⚠️ Enter your OpenAI API key in the sidebar before starting.", "warning")

    # ── Fusion handoff: merged dataset from Phase 0 ───────────────────────────
    # If the user came from the Data Fusion page, we already have the merged df
    # in session_state — skip the file uploader entirely.
    _fusion_b64  = st.session_state.get("_fusion_merged_b64")
    _fusion_name = st.session_state.get("_fusion_merged_name", "merged_dataset.csv")

    if _fusion_b64:
        from utils.serialization import b64_to_df
        try:
            df = b64_to_df(_fusion_b64)
            st.info(
                f"📦 Using merged dataset from **Data Fusion** — "
                f"{len(df):,} rows × {len(df.columns)} columns. "
                "You can upload a different file below to override."
            )
            _render_upload_ui(df, _fusion_name, api_key)
            # Offer to go back to fusion
            if st.button("← Back to Data Fusion", key="_ingest_back_fusion"):
                st.session_state.pop("_fusion_merged_b64", None)
                st.session_state["phase"] = "fusion"
                from ui.state_store import update_store
                update_store(phase="fusion")
                st.rerun()
            return
        except Exception:
            st.session_state.pop("_fusion_merged_b64", None)

    uploaded = st.file_uploader("Upload CSV (max 100 MB)", type=["csv"])
    if not uploaded:
        return

    # getvalue() is stable across reruns; read() may return empty bytes after
    # the first pass, which would change cache keys and reset the step flow.
    content = uploaded.getvalue()
    if len(content) > 100 * 1_048_576:
        alert("File too large (max 100 MB)", "error")
        return

    # Cache parsed df keyed by filename+bytesize — stable across reruns.
    # Using getvalue() keeps the upload signature fixed so the column
    # intelligence step state survives each rerun.
    _df_cache_key = f"_df_{uploaded.name}_{len(content)}"

    # Evict df cache from any previous (different) upload
    for k in [k for k in st.session_state if k.startswith("_df_") and k != _df_cache_key]:
        del st.session_state[k]

    current_ci_cache_key = None

    if _df_cache_key not in st.session_state:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            alert(f"Cannot parse CSV: {e}", "error")
            return
        st.session_state[_df_cache_key] = df
        current_ci_cache_key = f"_col_intel_{uploaded.name}_{len(df)}_{len(df.columns)}"
    else:
        df = st.session_state[_df_cache_key]
        current_ci_cache_key = f"_col_intel_{uploaded.name}_{len(df)}_{len(df.columns)}"

    # Drop column-intelligence state for older uploads so a new file never
    # inherits fallback or AI descriptions from a previous dataset.
    for k in [k for k in list(st.session_state.keys()) if k.startswith("_col_intel_") and not k.startswith(current_ci_cache_key)]:
        del st.session_state[k]

    _render_upload_ui(df, uploaded.name, api_key)


def _render_existing_session(ps: dict, api_key: str, tid: str):
    """Render Data Fusion when returning after the pipeline has started.

    Restores df, column descriptions, target, and problem type from
    pipeline_state — no re-upload prompt, no LLM re-call.
    The user can still edit column descriptions (edits are written back to
    pipeline_state) and change target/problem_type before re-running.
    """
    from utils.serialization import b64_to_df
    from ui.graph_helpers import update_graph_state
    from ui.state_store import update_store

    ds_name   = st.session_state.get("dataset_name", "Uploaded dataset")
    st.caption(f"Showing saved state for **{ds_name}**. Start a new session from the sidebar to upload a different file.")

    try:
        df = b64_to_df(ps["df_parquet_b64"])
    except Exception as e:
        alert(f"Could not restore dataset: {e}", "error")
        return

    cols      = df.columns.tolist()
    dupes     = int(df.duplicated().sum())
    high_null = [c for c in cols if df[c].isna().mean() > 0.5]

    if dupes:
        alert(f"⚠️ {dupes:,} duplicate rows detected", "warning")
    if high_null:
        alert(f"⚠️ High-null columns (>50%): {', '.join(high_null)}", "warning")

    metrics_row([
        ("Rows",           f"{len(df):,}"),
        ("Columns",        len(cols)),
        ("Duplicate rows", dupes),
        ("Missing cells",  int(df.isna().sum().sum())),
    ])
    st.dataframe(df.head(6), use_container_width=True)

    # ── Column Intelligence — restored from pipeline_state, no LLM call ───────
    # Descriptions are stored in pipeline_state["column_descriptions"] and
    # persist across navigation, page refresh, and session restore.
    descs = ps.get("column_descriptions") or {}
    if not descs:
        descs = _fallback_descriptions(df)

    # Use a stable cache key so data_editor edits don't cause LLM re-fires
    _ci_cache_key = f"_col_intel_{ds_name}_{len(df)}_{len(cols)}"
    if _ci_cache_key not in st.session_state:
        st.session_state[_ci_cache_key] = descs
    else:
        descs = st.session_state[_ci_cache_key]

    with st.expander("🧠 Column Intelligence — click to expand / collapse", expanded=True):
        descs = _render_column_intelligence_v2(df, descs, _ci_cache_key, api_key=api_key, dataset_name=ds_name)

    descs = st.session_state.get(_ci_cache_key, descs)

    # Persist any description edits back into pipeline_state
    if descs != ps.get("column_descriptions"):
        update_graph_state({"column_descriptions": descs}, tid)
        st.session_state["pipeline_state"] = {**ps, "column_descriptions": descs}

    # ── Target & Task ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Target & Task Configuration")

    saved_target   = ps.get("target", cols[-1])
    saved_probtype = ps.get("problem_type", "classification")
    target_idx     = cols.index(saved_target) if saved_target in cols else len(cols) - 1

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox(
            "🎯 Target column", cols, index=target_idx,
            help="The column your model should learn to predict.",
        )
        if target in descs:
            m = descs[target]
            st.markdown(
                f'<div style="background:#12132a;border-radius:6px;padding:10px 14px;'
                f'margin-top:6px;border:1px solid #2a2b45;">'
                f'<span style="font-size:0.78rem;color:#7ec8fa;">AI understanding:</span><br>'
                f'<span style="color:#c8c8e8;font-size:0.9rem;">{m.get("meaning","—")}</span><br>'
                f'<span style="margin-top:4px;display:inline-block;">'
                f'{_type_badge(m.get("type_label","Unknown"))}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with col2:
        s_target  = df[target] if target else df.iloc[:, -1]
        is_num    = pd.api.types.is_numeric_dtype(s_target)
        guess_clf = (not is_num) or (int(s_target.nunique()) <= 20)
        default_pt_idx = ["classification", "regression"].index(saved_probtype) if saved_probtype in ["classification", "regression"] else (0 if guess_clf else 1)
        prob_type = st.radio(
            "Problem type", ["classification", "regression"],
            index=default_pt_idx, horizontal=True,
        )

    validation = validate_target(df, target, prob_type, descs)
    if validation.issues:
        st.markdown("#### ⚡ Target / Task Validation")
        _render_target_issues(validation)
    else:
        st.success(f"✅ **`{target}`** is a valid target for **{prob_type}**. Ready to continue.")

    # ── Update target/problem_type if changed ────────────────────────────────
    if target != ps.get("target") or prob_type != ps.get("problem_type"):
        patch = {"target": target, "problem_type": prob_type}
        update_graph_state(patch, tid)
        st.session_state["pipeline_state"] = {**ps, **patch}
        update_store(pipeline_state={**ps, **patch})


def _render_upload_ui(df: pd.DataFrame, filename: str, api_key: str):
    """Fresh-upload flow: show df preview, column intelligence, target picker, start button."""
    cols      = df.columns.tolist()
    dupes     = int(df.duplicated().sum())
    high_null = [c for c in cols if df[c].isna().mean() > 0.5]

    if dupes:
        alert(f"⚠️ {dupes:,} duplicate rows detected", "warning")
    if high_null:
        alert(f"⚠️ High-null columns (>50%): {', '.join(high_null)}", "warning")

    metrics_row([
        ("Rows",           f"{len(df):,}"),
        ("Columns",        len(cols)),
        ("Duplicate rows", dupes),
        ("Missing cells",  int(df.isna().sum().sum())),
    ])
    st.dataframe(df.head(6), use_container_width=True)

    # ── Column Intelligence ───────────────────────────────────────────────────
    # Cache key is stable: based on filename + df shape (both fixed after parse).
    _ci_cache_key = f"_col_intel_{filename}_{len(df)}_{len(cols)}"

    if _ci_cache_key not in st.session_state:
        st.session_state[_ci_cache_key] = _fallback_descriptions(df)
    descs = st.session_state[_ci_cache_key]

    with st.expander("🧠 Column Intelligence — click to expand / collapse", expanded=True):
        descs = _render_column_intelligence_v2(df, descs, _ci_cache_key, api_key=api_key, dataset_name=filename)

    descs = st.session_state.get(_ci_cache_key, descs)

    # ── Target & Task selection ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Target & Task Configuration")

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox(
            "🎯 Target column", cols, index=len(cols) - 1,
            help="The column your model should learn to predict.",
        )
        if target and target in descs:
            m = descs[target]
            st.markdown(
                f'<div style="background:#12132a;border-radius:6px;padding:10px 14px;'
                f'margin-top:6px;border:1px solid #2a2b45;">'
                f'<span style="font-size:0.78rem;color:#7ec8fa;">AI understanding:</span><br>'
                f'<span style="color:#c8c8e8;font-size:0.9rem;">{m.get("meaning","—")}</span><br>'
                f'<span style="margin-top:4px;display:inline-block;">'
                f'{_type_badge(m.get("type_label","Unknown"))}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with col2:
        s_target  = df[target] if target else df.iloc[:, -1]
        n_unique  = int(s_target.nunique())
        is_num    = pd.api.types.is_numeric_dtype(s_target)
        guess_clf = (not is_num) or (n_unique <= 20)
        prob_type = st.radio(
            "Problem type", ["classification", "regression"],
            index=0 if guess_clf else 1, horizontal=True,
            help="Classification predicts categories; Regression predicts numbers.",
        )

    # ── Live target validation ────────────────────────────────────────────────
    validation = validate_target(df, target, prob_type, descs)
    if validation.issues:
        st.markdown("#### ⚡ Target / Task Validation")
        _render_target_issues(validation)
    else:
        st.success(f"✅ **`{target}`** is a valid target for **{prob_type}**. Ready to start.")

    # ── Start button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if validation.blocking:
        alert(
            "❌ **Cannot start pipeline** — the target column has critical issues. "
            "Choose a different target or correct the problem type.",
            "error",
        )

    start = st.button(
        "⚡ Start multi-agent pipeline →",
        disabled=(not api_key) or validation.blocking,
        help=(
            "Fix the ❌ errors above first." if validation.blocking else
            "Enter your OpenAI API key first." if not api_key else ""
        ),
    )

    if start:
        with st.spinner("EDA Agent is analysing your dataset…"):
            try:
                tid = str(uuid.uuid4())
                upsert_session(tid, phase="fusion")
                init_state = {
                    "df_parquet_b64":      df_to_b64(df),
                    "target":              target,
                    "problem_type":        prob_type,
                    "n_rows":              int(len(df)),
                    "n_cols":              int(len(cols)),
                    "retry_count":         0,
                    "agent_messages":      [],
                    "openai_api_key":      api_key,
                    # Persist descriptions so returning to this page never re-calls LLM
                    "column_descriptions": descs,
                }
                state = run_graph_sync(init_state, tid)

                if state.get("loop_verdict") == "error":
                    alert(
                        f"❌ Pipeline error in EDA agent: {state.get('loop_reasoning', '')}",
                        "error",
                    )
                    return

                persist(tid, state, dataset_name=filename, phase="eda")
                st.session_state.tid            = tid
                st.session_state.pipeline_state = safe_state(state)
                st.session_state.dataset_name   = filename
                st.session_state.phase          = "eda"
                update_store(
                    tid=tid,
                    pipeline_state=safe_state(state),
                    dataset_name=filename,
                    phase="eda",
                )
                st.success("✅ EDA Agent complete!")
                st.rerun()
            except Exception as e:
                alert(f"EDA agent error: {e}", "error")
