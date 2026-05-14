"""
ui/nl_query.py - Natural Language Query Interface (v5)
"""

from __future__ import annotations

import json
import os

import streamlit as st

from ui.components import alert
from ui.state_store import add_nl_result, get_store
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_NL_SYSTEM = """You are an AutoML assistant that maps natural language queries to pipeline actions.

Given the user's query and current pipeline state, respond with a JSON object:
{
  "intent": "<one of: explain_results | run_pipeline | compare_models | show_features |
              export_model | check_metrics | suggest_improvements | answer_question | unsupported>",
  "action": "<specific action to take>",
  "response": "<friendly natural language response to show the user>",
  "params": {}
}

Always respond with valid JSON only, no markdown.
"""


def _score_or_neg_inf(row: dict) -> float:
    score = row.get("best_score")
    return float(score) if isinstance(score, (int, float)) else float("-inf")


def _parse_nl_intent(query: str, state: dict, api_key: str) -> dict:
    try:
        from utils.agent_utils import call_llm_json

        context = {
            "current_phase": state.get("phase", "ingest"),
            "problem_type": state.get("pipeline_state", {}).get("problem_type", "unknown"),
            "best_model": state.get("pipeline_state", {}).get("best_model_key", "none"),
            "eval_metrics": {
                k: v
                for k, v in (state.get("pipeline_state", {}).get("eval_metrics") or {}).items()
                if k != "confusion_matrix"
            },
            "has_data": bool(state.get("pipeline_state", {}).get("target")),
            "n_features": len(state.get("pipeline_state", {}).get("selected_features") or []),
            "retry_count": state.get("pipeline_state", {}).get("retry_count", 0),
        }
        user_content = f"User query: {query}\n\nContext: {json.dumps(context, indent=2)}"
        result, _err = call_llm_json(
            api_key=api_key,
            model_name=cfg("llm.default_model", default="gpt-4o-mini"),
            system_prompt=_NL_SYSTEM,
            user_content=user_content,
            temperature=0.1,
            max_tokens=500,
        )
        if result:
            return result
    except Exception as e:
        log.warning(f"NL intent parse failed: {e}")

    return {
        "intent": "answer_question",
        "action": "explain",
        "response": "I understood your query but couldn't determine a specific action. Try: 'show metrics', 'which features matter most', or 'compare models'.",
        "params": {},
    }


def _execute_intent(intent_result: dict, store) -> str:
    intent = intent_result.get("intent", "unsupported")
    response = intent_result.get("response", "")
    state = store.pipeline_state

    if intent == "check_metrics":
        metrics = state.get("eval_metrics", {})
        if metrics:
            lines = []
            for k, v in metrics.items():
                if k == "confusion_matrix":
                    continue
                if isinstance(v, (int, float)):
                    lines.append(f"- **{k}**: {float(v):.4f}")
            response += "\n\n" + "\n".join(lines) if lines else "\n\nNo metrics available yet."

    elif intent == "show_features":
        shap = state.get("shap_importance", [])
        if shap:
            top5 = shap[:5]
            lines = [
                f"- **{s['feature']}**: {s.get('importance', s.get('mean_abs_shap', 0)):.4f}"
                for s in top5
            ]
            response += "\n\nTop 5 features:\n" + "\n".join(lines)

    elif intent == "run_pipeline":
        if not state.get("target"):
            response = "Please upload a dataset and select a target column first."
        else:
            response += "\n\nUse the **Approve** buttons in each pipeline phase to advance, or navigate via the sidebar."

    elif intent == "compare_models":
        tuning = state.get("tuning_results", [])
        if tuning:
            sorted_t = sorted(tuning, key=_score_or_neg_inf, reverse=True)
            lines = []
            for r in sorted_t[:5]:
                score = r.get("best_score")
                if isinstance(score, (int, float)):
                    lines.append(f"- **{r['model_key']}**: {score:.4f}")
                else:
                    lines.append(f"- **{r['model_key']}**: failed")
            response += "\n\n" + "\n".join(lines)

    elif intent == "suggest_improvements":
        suggestions = []
        metrics = state.get("eval_metrics", {})
        if metrics.get("f1_weighted", 1) < 0.75:
            suggestions.append("F1 score is below 0.75 - try more Optuna trials or add feature interactions.")
        shap = state.get("shap_importance", [])
        if shap and len(shap) < 5:
            suggestions.append("Few important features found - consider adding domain-specific features.")
        drift = (state.get("drift_report") or {}).get("overall_severity", "low")
        if drift in ("medium", "high"):
            suggestions.append(f"Data drift detected ({drift}) - consider retraining with fresh data.")
        if not suggestions:
            suggestions.append("Model looks solid. Consider exporting to production.")
        response += "\n\n" + "\n".join(f"- {s}" for s in suggestions)

    return response


def render_nl_query_panel() -> None:
    st.markdown("### Ask AutoML")
    st.caption("Natural language queries: 'What is my model accuracy?', 'Which features matter most?', etc.")

    store = get_store()
    api_key = st.session_state.get("openai_key") or os.getenv("OPENAI_API_KEY", "")

    suggestions = [
        "What is my model accuracy?",
        "Which features matter most?",
        "Suggest improvements",
        "Compare models",
    ]
    cols = st.columns(2)
    for i, sug in enumerate(suggestions):
        if cols[i % 2].button(sug, key=f"nl_chip_{i}", width="stretch"):
            st.session_state["nl_query_input"] = sug

    query = st.text_input(
        "Query",
        value=st.session_state.get("nl_query_input", ""),
        placeholder="Ask anything about your AutoML run...",
        key="nl_query_input",
        label_visibility="collapsed",
    )

    if st.button("Ask ✨", key="nl_ask_btn", type="primary") and query.strip():
        if not api_key:
            alert("Set your OpenAI API key in the sidebar first.", "warning")
            return
        with st.spinner("Thinking..."):
            intent_result = _parse_nl_intent(
                query,
                {"phase": store.phase, "pipeline_state": store.pipeline_state},
                api_key,
            )
            response = _execute_intent(intent_result, store)
            add_nl_result(query, response)
            st.session_state["_nl_last_result"] = {"query": query, "result": response}

    last_result = st.session_state.get("_nl_last_result")
    if last_result:
        st.markdown("**Latest answer**")
        st.markdown(last_result["result"])

    history = store.nl_history[-5:]
    if history:
        for entry in reversed(history):
            with st.expander(f"💬 {entry['query'][:60]}...", expanded=False):
                st.markdown(entry["result"])
