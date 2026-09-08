"""LangGraph construction: should_continue routing, shared graph wiring, and compiled graphs."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import SCORE_THRESHOLD, MAX_ITERATIONS, init_output_dirs
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
# Upgrade path: swap MemorySaver for SqliteSaver to persist across restarts.
interactive_workflow = StateGraph(AgentState)
interactive_workflow.add_node("loader", loader_node)
interactive_workflow.add_node("scanner", scanner_node)
_add_optimization_core(interactive_workflow)
interactive_workflow.set_entry_point("loader")
interactive_workflow.add_edge("loader", "scanner")
interactive_workflow.add_edge("scanner", "improver")
agent_app = interactive_workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["improver"],
)


# --- CLI entry point ---
if __name__ == "__main__":
    init_output_dirs()

    sample_job = "Looking for a Python Developer with Data Science skills."
    sample_resume = "I am a Python developer."
    sample_notes = "I have 2 years of experience."

    print("Starting Advanced Resume Agent (CLI Mode)...")
    try:
        final_state = full_app.invoke({
            "job_description": sample_job,
            "original_resume": sample_resume,
            "human_notes": sample_notes,
            "resume_text": "", "job_text": "", "optimized_resume": "",
            "feedback": "", "missing_skills": "", "score": 0, "iteration": 0,
            "llm_quality_score": 0, "ats_percentage": 0.0, "review_failed": False,
            "cover_letter": "", "interview_questions": "", "ats_result": "",
            "resume_pdf_path": "", "cover_letter_pdf_path": "",
            "resume_docx_path": "", "resume_tex_path": ""
        }, config={"callbacks": TRACE_CALLBACKS})
        print("Done! Files saved.")
    except Exception as e:
        print(f"Error: {e}")
