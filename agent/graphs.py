"""LangGraph construction: should_continue routing, shared graph wiring, and compiled graphs."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import SCORE_THRESHOLD, MAX_ITERATIONS, CHECKPOINTER_DB_PATH
from .state import AgentState
from .tracing import TRACE_CALLBACKS
from .nodes import (
    loader_node, scanner_node, improver_node, reviewer_node,
    ats_check_node, cover_letter_node, interview_prep_node,
    pdf_exporter_node,
)
from .helpers import _safe_print


def should_continue(state: AgentState) -> str | list[str]:
    """Gate after ats_check. Returning a list fans out to both branches
    concurrently (LangGraph native parallelism)."""
    if state.get('review_failed', False):
        _safe_print("Reviewer failed - proceeding with current draft (no retry).")
        return ["cover_letter", "interview_prep"]
    if state['score'] >= SCORE_THRESHOLD:
        _safe_print(f"Success! Composite score {state['score']} >= {SCORE_THRESHOLD}. Generating Docs...")
        return ["cover_letter", "interview_prep"]
    elif state['iteration'] >= MAX_ITERATIONS:
        _safe_print(f"Max iterations ({MAX_ITERATIONS}) reached. Generating Docs anyway...")
        return ["cover_letter", "interview_prep"]
    else:
        _safe_print(f"Composite score {state['score']} < {SCORE_THRESHOLD}. Retrying...")
        return "improver"


def _add_optimization_core(g: StateGraph):
    """Shared improver -> reviewer -> ats_check loop + fan-out + export wiring."""
    g.add_node("improver", improver_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("ats_check", ats_check_node)
    g.add_node("cover_letter", cover_letter_node)
    g.add_node("interview_prep", interview_prep_node)
    g.add_node("pdf_exporter", pdf_exporter_node)

    g.add_edge("improver", "reviewer")
    g.add_edge("reviewer", "ats_check")

    # Native fan-out: returning two destinations runs cover_letter and
    # interview_prep concurrently. LangSmith traces each as a sibling run
    # under this gate (the old ThreadPoolExecutor version orphaned them
    # into separate root traces).
    g.add_conditional_edges("ats_check", should_continue)

    # Join: wait for BOTH branches before exporting
    g.add_edge(["cover_letter", "interview_prep"], "pdf_exporter")
    g.add_edge("pdf_exporter", END)


# --- GRAPH A: FULL (no interrupt; used by CLI and tests) ---
workflow = StateGraph(AgentState)
workflow.add_node("loader", loader_node)
_add_optimization_core(workflow)
workflow.set_entry_point("loader")
workflow.add_edge("loader", "improver")
full_app = workflow.compile()


# --- GRAPH B: INTERACTIVE (checkpointer + human-review interrupt) ---
# One graph for the Gradio UI. Step 1 runs loader -> scanner and INTERRUPTS
# before the Improver (first-class human-in-the-loop). Step 2 injects the
# user's edited notes via update_state and resumes from the checkpoint.
# The loader therefore runs exactly once per session (no double PDF parsing),
# and the whole session survives in the in-memory checkpointer.
def _create_checkpointer():
    """Returns a checkpointer instance.

    If CHECKPOINTER_DB_PATH is configured, attempts to initialize a persistent
    SQLite checkpointer for multi-worker / crash-resilient deployments.
    Otherwise, defaults to MemorySaver() for zero-configuration local use.
    """
    if CHECKPOINTER_DB_PATH:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(CHECKPOINTER_DB_PATH, check_same_thread=False)
            _safe_print(f"[Checkpointer] Using persistent SQLite storage at: {CHECKPOINTER_DB_PATH}")
            return SqliteSaver(conn)
        except Exception as e:
            _safe_print(f"[Checkpointer] Warning: Could not initialize SQLite checkpointer ({e}). Falling back to MemorySaver.")
    return MemorySaver()


interactive_workflow = StateGraph(AgentState)
interactive_workflow.add_node("loader", loader_node)
interactive_workflow.add_node("scanner", scanner_node)
_add_optimization_core(interactive_workflow)
interactive_workflow.set_entry_point("loader")
interactive_workflow.add_edge("loader", "scanner")
interactive_workflow.add_edge("scanner", "improver")
agent_app = interactive_workflow.compile(
    checkpointer=_create_checkpointer(),
    interrupt_before=["improver"],
)
