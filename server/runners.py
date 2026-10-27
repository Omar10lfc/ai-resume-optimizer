"""Two-step pipeline runners ported from the legacy Gradio app (app.py).

Framework-agnostic sync generators that drive the interactive LangGraph
graph (``agent_app``) with identical semantics to the Gradio handlers, so
the FastAPI SSE layer and the Next.js frontend share the same flow:

- Step 1 runs loader -> scanner, then the graph INTERRUPTS before the
  Improver (first-class human-in-the-loop checkpoint). The scanner's
  missing-skills output is returned for the review/edit box.
- Step 2 resumes the interrupted thread: if Step 1 was skipped it first
  runs loader -> scanner up to the interrupt, injects the user's edited
  notes via ``update_state``, then resumes with a PINNED root run id so
  pipeline feedback records attach to the exact LangSmith trace.
"""

import traceback
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from agent import TRACE_CALLBACKS, agent_app
from agent import config as agent_config
from agent.tracing import (
    candidate_name_hint,
    log_pipeline_feedback,
    register_known_name,
)

# Redacted-tracing callbacks go with every invoke (empty list = tracing off)
_INVOKE_CONFIG_BASE = {"callbacks": TRACE_CALLBACKS}

# Per-node status messages for the streaming pipeline view
NODE_STATUS = {
    "loader": "Loading documents...",
    "scanner": "Scanning for skill gaps...",
    "improver": "Improving resume...",
    "reviewer": "Reviewer is grading the draft...",
    "ats_check": "Computing ATS keyword match...",
    "cover_letter": "Writing cover letter...",
    "interview_prep": "Preparing interview questions...",
    "pdf_exporter": "Exporting documents...",
}

# Artifact kind -> state key (only kinds whose file exists are reported)
_ARTIFACT_KINDS = {
    "resume-pdf": "resume_pdf_path",
    "cover-pdf": "cover_letter_pdf_path",
    "resume-docx": "resume_docx_path",
    "resume-tex": "resume_tex_path",
}


def _thread_config(thread_id: str, run_id: str | None = None) -> dict:
    """RunnableConfig for one graph segment.

    run_id (optional) pins the ROOT run id of the segment's trace: LangGraph
    reads config['run_id'] for the root run, so pipeline feedback records can
    target a known trace root instead of scraping the tracer's run tree.
    """
    cfg = dict(_INVOKE_CONFIG_BASE)
    cfg["configurable"] = {"thread_id": thread_id}
    if run_id:
        cfg["run_id"] = run_id
    return cfg


def _initial_inputs(job_input: str, resume_source: str) -> dict:
    return {
        "job_description": job_input.strip(),
        "original_resume": resume_source,
        "human_notes": "", "missing_skills": "",
        "resume_text": "", "job_text": "", "optimized_resume": "",
        "feedback": "", "score": 0, "iteration": 0,
        "llm_quality_score": 0, "ats_percentage": 0.0, "review_failed": False,
        "cover_letter": "", "interview_questions": "", "ats_result": "",
        "resume_pdf_path": "", "cover_letter_pdf_path": "",
        "resume_docx_path": "", "resume_tex_path": ""
    }


def resolve_resume_source(resume_path: str | None, resume_text: str | None):
    """File takes priority over pasted text (same policy as the Gradio app)."""
    if resume_path:
        return resume_path
    if resume_text and resume_text.strip():
        return resume_text.strip()
    return None


def premask_resume_name(resume_source) -> None:
    """Registers the candidate name parsed from a PASTED resume before any
    graph run is traced, so even the loader run's own input mask is complete.
    (PDF/DOCX uploads pass a file path here - no content, nothing to mask;
    the loader registers their names after parsing.)"""
    try:
        register_known_name(candidate_name_hint(resume_source))
    except Exception as exc:  # masking must never block the pipeline
        print(f"[PII] name pre-registration skipped: {exc}")


def _status(message: str) -> dict:
    return {"type": "status", "message": message}


def stream_step1(job_description: str, resume_source: str,
                 session: str | None) -> Iterator[dict]:
    """Step 1: loader -> scanner, interrupt before the improver.

    Yields dict events; the final one is
    ``{"type": "done", "session", "missing_skills"}``.
    """
    if not job_description or not job_description.strip():
        yield {"type": "error", "message": "Please provide a Job Description (text or URL)."}
        return
    if not resume_source:
        yield {"type": "error", "message": "Please upload a resume PDF or paste your resume text."}
        return

    premask_resume_name(resume_source)
    session = session or str(uuid4())
    config = _thread_config(session)

    yield _status("Loading documents...")
    try:
        for update in agent_app.stream(_initial_inputs(job_description, resume_source),
                                       config=config, stream_mode="updates",
                                       subgraphs=False):
            for node, delta in update.items():
                if node and delta:
                    yield _status(
                        f"{NODE_STATUS.get(node, node + '...')} (step 1 of 2)")
        snapshot = agent_app.get_state(config)
        missing = (snapshot.values or {}).get("missing_skills", "")
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"Scan failed: {e}"}
        return

    yield {"type": "done", "session": session, "missing_skills": missing,
           "status": "Analysis complete - review & edit the gaps below, then run Step 2."}


def _extract_results(final_state: dict) -> dict:
    """Maps final graph state into the frontend result payload, mirroring the
    Gradio extraction (safe fallbacks, existence-checked artifact files)."""
    artifacts = []
    for kind, key in _ARTIFACT_KINDS.items():
        path = final_state.get(key) or ""
        if path:
            file_path = Path(path)
            # PDFs/DOCX/TEX live in the per-request dir (loader's request_id,
            # independent of the thread session id): the URL segment is the
            # file's actual parent name, validated as a UUID by the endpoint.
            if file_path.exists():
                artifacts.append({
                    "kind": kind,
                    "filename": file_path.name,
                    "url": f"api/v1/artifacts/{file_path.parent.name}/{kind}"
                           f"?file={file_path.name}",
                })
            else:
                print(f"[Warning] Artifact path does not exist: {path}")

    score_val = final_state.get("score", 0)
    ats_pct = final_state.get("ats_percentage", 0)
    score_class = ("good" if score_val >= agent_config.SCORE_THRESHOLD
                   else "warn" if score_val >= 60 else "bad")
    return {
        "optimized_resume": final_state.get("optimized_resume") or "No resume generated.",
        "score": score_val,
        "score_class": score_class,
        "feedback": final_state.get("feedback") or "No feedback provided.",
        "cover_letter": final_state.get("cover_letter") or "No cover letter generated.",
        "ats_result": final_state.get("ats_result") or "ATS analysis not available.",
        "interview_questions": final_state.get("interview_questions")
        or "No interview questions generated.",
        "ats_percentage": ats_pct,
        "artifacts": artifacts,
        "weights": {
            "ats": agent_config.ATS_SCORE_WEIGHT,
            "llm": agent_config.LLM_SCORE_WEIGHT,
        },
        "status": (f"Done - composite score {score_val}/100 "
                   f"(ATS {ats_pct:.0f}% x {agent_config.ATS_SCORE_WEIGHT:g} "
                   f"+ LLM quality x {agent_config.LLM_SCORE_WEIGHT:g})."),
    }


def stream_step2(job_description: str, resume_source: str, notes: str,
                 session: str | None) -> Iterator[dict]:
    """Step 2: resumes the interrupted session (or starts fresh).

    Final event: ``{"type": "result", **payload}``.
    """
    if not job_description or not job_description.strip():
        yield {"type": "error", "message": "Please provide a Job Description (text or URL)."}
        return
    if not resume_source:
        yield {"type": "error", "message": "Please upload a resume PDF or paste your resume text."}
        return

    premask_resume_name(resume_source)

    if not notes or not notes.strip():
        yield {"type": "warning",
               "message": "No missing skills / context provided. Running with auto-detected gaps only."}

    session = session or str(uuid4())
    config = _thread_config(session)

    yield _status("Starting pipeline...")
    final_state: dict = {}
    try:
        state = agent_app.get_state(config)
        if not state.next:
            if state.values:
                # previous run completed in this session -> start a fresh one
                session = str(uuid4())
                config = _thread_config(session)
                yield {"type": "session", "session": session}
                yield _status("Starting a fresh run...")
            # Step 1 was skipped: run loader -> scanner up to the interrupt
            for update in agent_app.stream(_initial_inputs(job_description, resume_source),
                                           config=config, stream_mode="updates"):
                for node, delta in update.items():
                    if node and delta:
                        yield _status(f"{NODE_STATUS.get(node, node + '...')} (1/2)")

        # Inject the user's edited gaps/context at the human-review checkpoint
        agent_app.update_state(config, {
            "human_notes": notes or "",
            "missing_skills": notes or "",
        })

        # Resume: improver <-> reviewer -> ats_check -> fan-out -> pdf_exporter.
        # A pinned root run id lets the post-run feedback records attach to
        # the exact LangSmith trace (threshold/root of THIS segment).
        resume_run_id = str(uuid4())
        for update in agent_app.stream(None, config=_thread_config(session, resume_run_id),
                                       stream_mode="updates"):
            for node, delta in update.items():
                if node and delta:
                    final_state.update(delta)
                    yield _status(f"{NODE_STATUS.get(node, node + '...')} (2/2)")
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"Optimization failed: {e}"}
        return

    if not final_state:
        snapshot = agent_app.get_state(config)
        final_state = snapshot.values or {}

    # Score + failure feedback on the pinned trace root. Degraded runs become
    # queryable on the LangSmith dashboard (composite_score low / review_failed
    # = 1) even though no run inside the trace ever carried error status.
    # (Only successful full runs reach this point - pipeline errors returned
    # above, mirroring the original Gradio semantics.)
    log_pipeline_feedback(final_state, resume_run_id)

    yield {"type": "result", "session": session,
           **_extract_results(final_state)}
