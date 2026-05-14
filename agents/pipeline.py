"""
agents/pipeline.py — LangGraph Pipeline Definition (v3)

Improvements over v2:
  #2  Agent error routing: every agent node is followed by an error-check
      conditional edge that diverts to a terminal "pipeline_error" node
      instead of silently forwarding corrupt state to the next agent.

Pipeline flow:
  ingest → eda_agent →[err?]→ hitl_eda
         → leakage_agent →[err?]→ hitl_leakage      (FIX 3: now interrupts)
         → feature_agent →[err?]→ hitl_features
         → split_agent →[err?]→ hitl_split
         → hitl_model_selection                       (FIX 12: user reviews LLM picks)
         → model_agent →[err?]→ hitl_models
         → eval_agent →[err?]→ ensemble_agent →[err?]→ hitl_ensemble  (FIX 3)
         → orchestrator →[err?]→ hitl_loop
         → (loop or export)
"""

from __future__ import annotations
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.state import PipelineState
from agents.eda_agent       import eda_agent
from agents.leakage_agent   import leakage_agent
from agents.feature_agent   import feature_agent
from agents.split_agent     import split_agent
from agents.model_agent     import model_agent
from agents.eval_agent      import eval_agent
from agents.ensemble_agent  import ensemble_agent
from agents.orchestrator    import orchestrator_agent
from utils.agent_utils      import has_agent_error


# ─────────────────────────────────────────────────────────────────────────────
# Fix #2: Error terminal node
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_error_node(state: PipelineState) -> dict:
    """
    Terminal node reached when any agent sets _agent_error=True.
    Writes a clear message to the log and surfaces the source agent.
    The Streamlit UI checks loop_verdict == "error" to show the error banner.
    """
    source = state.get("_agent_error_source", "Unknown agent")
    msgs   = state.get("agent_messages", [])
    # Find the error message from the failing agent
    error_detail = next(
        (m for m in reversed(msgs) if "❌" in str(m)),
        f"Agent '{source}' failed — check logs for details.",
    )
    return {
        "loop_verdict":   "error",
        "loop_reasoning": error_detail,
        "agent_messages": [f"[Pipeline] ❌ Pipeline halted due to error in {source}."],
    }


def _make_error_router(next_node: str):
    """
    Returns a conditional-edge function that routes to pipeline_error_node
    if _agent_error is set, otherwise continues to next_node.
    """
    def _router(state: PipelineState) -> str:
        if has_agent_error(state):
            return "pipeline_error"
        return next_node
    return _router


def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    # ── Agent nodes ───────────────────────────────────────────────────────────
    builder.add_node("eda_agent",          eda_agent)
    builder.add_node("leakage_agent",      leakage_agent)
    builder.add_node("feature_agent",      feature_agent)
    builder.add_node("split_agent",        split_agent)
    builder.add_node("model_agent",        model_agent)
    builder.add_node("eval_agent",         eval_agent)
    builder.add_node("ensemble_agent",     ensemble_agent)
    builder.add_node("orchestrator_agent", orchestrator_agent)

    # ── Fix #2: Error terminal node ───────────────────────────────────────────
    builder.add_node("pipeline_error", pipeline_error_node)

    # ── HITL gate nodes (pass-through — app pauses here) ─────────────────────
    def hitl_eda(state):             return {}
    def hitl_leakage(state):         return {}
    def hitl_features(state):        return {}
    def hitl_split(state):           return {}
    def hitl_model_selection(state): return {}   # FIX 12: explicit gate for model-family review
    def hitl_models(state):          return {}
    def hitl_ensemble(state):        return {}
    def hitl_loop(state):            return {}

    builder.add_node("hitl_eda",              hitl_eda)
    builder.add_node("hitl_leakage",          hitl_leakage)   # FIX 3: now in interrupt_before
    builder.add_node("hitl_features",         hitl_features)
    builder.add_node("hitl_split",            hitl_split)
    builder.add_node("hitl_model_selection",  hitl_model_selection)
    builder.add_node("hitl_models",           hitl_models)
    builder.add_node("hitl_ensemble",         hitl_ensemble)  # FIX 3: now in interrupt_before
    builder.add_node("hitl_loop",             hitl_loop)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("eda_agent")

    # ── Edges with error-check after every agent node (Fix #2) ───────────────
    # FIX 12: hitl_split → hitl_model_selection → model_agent
    # User reviews LLM model-family recommendations BEFORE training starts,
    # then confirms (Deploy) to launch Optuna tuning.
    _agent_sequence = [
        ("eda_agent",             "hitl_eda"),
        ("hitl_eda",              "leakage_agent"),
        ("leakage_agent",         "hitl_leakage"),
        ("hitl_leakage",          "feature_agent"),
        ("feature_agent",         "hitl_features"),
        ("hitl_features",         "split_agent"),
        ("split_agent",           "hitl_split"),
        ("hitl_split",            "hitl_model_selection"),  # FIX 12: pause for LLM review
        ("hitl_model_selection",  "model_agent"),           # then train
        ("model_agent",           "hitl_models"),
        ("hitl_models",           "eval_agent"),
        ("eval_agent",            "ensemble_agent"),
        ("ensemble_agent",        "hitl_ensemble"),
        ("hitl_ensemble",         "orchestrator_agent"),
        ("orchestrator_agent",    "hitl_loop"),
    ]

    for source, dest in _agent_sequence:
        # HITL nodes are pass-throughs (return {}), they never set _agent_error,
        # so we use simple edges for them. Agent nodes get conditional routing.
        if source.startswith("hitl_"):
            builder.add_edge(source, dest)
        else:
            builder.add_conditional_edges(
                source,
                _make_error_router(dest),
                {"pipeline_error": "pipeline_error", dest: dest},
            )

    # pipeline_error is a terminal — route to END
    builder.add_edge("pipeline_error", END)

    # ── Conditional loop-back edge from hitl_loop ─────────────────────────────
    def loop_router(state: PipelineState) -> str:
        verdict = state.get("loop_verdict", "accept")
        if verdict in ("accept", "error"):
            return END
        elif verdict in ("retry_features", "retry_both"):
            return "feature_agent"
        else:  # retry_models
            return "model_agent"

    builder.add_conditional_edges(
        "hitl_loop",
        loop_router,
        {
            END:             END,
            "feature_agent": "feature_agent",
            "model_agent":   "model_agent",
        },
    )

    return builder


def compile_graph(db_path: str | None = None):
    """
    Compile with SQLite checkpointing.
    """
    import sqlite3
    from db.session_store import resolve_writable_db_path

    _db_path = resolve_writable_db_path(db_path or os.getenv("AUTOML_DB_PATH", "db/automl_history.db"))
    conn = sqlite3.connect(str(_db_path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    checkpointer = SqliteSaver(conn)

    graph = build_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "hitl_eda",
            "hitl_leakage",          # FIX 3: pause so user sees leakage report
            "hitl_features",
            "hitl_split",
            "hitl_model_selection",  # FIX 12: pause for model-family review
            "hitl_models",
            "hitl_ensemble",         # FIX 3: pause so user sees ensemble results
            "hitl_loop",
        ],
    )
    return graph
