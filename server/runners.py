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

import re
import traceback
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from agent import TRACE_CALLBACKS, agent_app
from agent import config as agent_config
from agent.llms import groq_key_override
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
    "polisher": "Polishing draft with review feedback...",
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


def _status(message: str, node: str | None = None) -> dict:
    info = {"type": "status", "message": message}
    if node:
        info["node"] = node
        info["node_state"] = "running"
    return info


def stream_step1(job_description: str, resume_source: str,
                 session: str | None, groq_api_key: str | None = None) -> Iterator[dict]:
    """Step 1: loader -> scanner, interrupt before the improver.

    groq_api_key (optional) honors a client-supplied Groq key (browser
    extension BYOK): every LLM call inside the segment binds to it instead of
    the server's GROQ_API_KEY.

    Yields dict events; the final one is
    ``{"type": "done", "session", "missing_skills"}``.
    """
    if not job_description or not job_description.strip():
        yield {"type": "error", "message": "Please provide a Job Description (text or URL)."}
        return
    if not resume_source:
        yield {"type": "error", "message": "Please upload a resume PDF or paste your resume text."}
        return

    with groq_key_override(groq_api_key):
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
                            f"{NODE_STATUS.get(node, node + '...')} (step 1 of 2)", node)
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
        # QW3: the integrity channels the results UI renders as a dedicated
        # view (the SSE "integrity check" warning stays as the live banner);
        # QW1/QW2: forensic prompt stamps + the writer's declined requests.
        "removals": final_state.get("scrub_removals") or [],
        "unevidenced_metrics": final_state.get("unevidenced_metrics") or [],
        "declined_requests": final_state.get("declined_requests") or [],
        "prompt_versions": final_state.get("prompt_versions") or {},
        # MR2: deterministic requirement-level fit table (covered/partial/gap
        # rows with evidence terms and human reasons).
        "requirement_matches": final_state.get("requirement_matches") or [],
        "weights": {
            "ats": agent_config.ATS_SCORE_WEIGHT,
            "llm": agent_config.LLM_SCORE_WEIGHT,
        },
        "status": (f"Done - composite score {score_val}/100 "
                   f"(ATS {ats_pct:.0f}% x {agent_config.ATS_SCORE_WEIGHT:g} "
                   f"+ LLM quality x {agent_config.LLM_SCORE_WEIGHT:g})."),
    }


def _norm_gap_line(s: str) -> str:
    """Normalization for matching a composed notes line against the step-1
    scanner chips: strip bullet markers, case, and repeated whitespace."""
    t = (s or "").strip().lower()
    t = re.sub(r"^[-*+•]\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _split_composed_notes(notes: str, pristine_missing: str) -> tuple[str, str]:
    """Splits the composed step-2 notes into (missing_skills, human_notes).

    The frontend composes ONE string (frontend/src/lib/gaps.ts): typed free
    text first, then kept chip lines ("- Label" / "- Label: note"), then a
    final "Do NOT feature: X, Y" line. A line belongs to the GAP channel
    when it matches a line of the step-1 scanner snapshot (kept-chip
    matching tolerates a trailing ": user note" annotation), or when it is
    the "Do NOT feature" directive. Everything else is the user's typed
    evidence and goes to human_notes (live run 868e406d: the chips were
    stamped into the evidence channel under the pasted project text, which
    made the model emit them as user-confirmed Aegis bullets).

    Stale-chip hardening (live request 539f7c1d, Nov 8 2026): the composed
    string can carry chip lines from a PREVIOUS session whose stale wording
    no longer matches the current scanner snapshot. They then fell into
    evidence_lines unchanged - the writer read those GAP descriptions
    ("Experience with n8n", "MCP servers") as user-confirmed evidence and
    shipped them verbatim into Summary + Technical Skills; the scrubber
    (which treats user context as evidence) found nothing to remove, ATS
    matched the JD's own vocabulary at 100%, and the reviewer capped
    quality at ~38 - the run-868e406d laundering mechanism re-armed by
    stale data. Any line the extension serialized as an ASCII bullet
    therefore stays a GAP line ("research topic, never claim") when it
    does not match the snapshot either: misclassifying a genuine typed
    note as a gap topic is the safe failure direction (the writer merely
    omits it), while classifying a gap chip as evidence is fabrication
    laundering. Real typed evidence in this app uses "•" bullets or prose
    paragraphs - the pasted Aegis block of the same run proves the
    distinction.
    """
    chip_keys: set[str] = set()
    for line in (pristine_missing or "").splitlines():
        t = _norm_gap_line(line)
        if not t:
            continue
        chip_keys.add(t)
        # chips the user annotated ("- Docker: ran containers in coursework"):
        # also accept the label-only match before the first colon
        label = t.split(":", 1)[0].strip()
        if label and len(label) >= 3:
            chip_keys.add(label)

    gap_lines: list[str] = []
    evidence_lines: list[str] = []
    for line in (notes or "").splitlines():
        s = line.strip()
        if not s:
            continue
        n = _norm_gap_line(s)
        if (n.startswith("do not feature") or n.startswith("don't feature")
                or n.startswith("dont feature")):
            gap_lines.append(s)
            continue
        if n in chip_keys or n.split(":", 1)[0].strip() in chip_keys:
            gap_lines.append(s)
            continue
        if s.startswith("- ") or s.startswith("* "):
            # ASCII bullets are the extension's chip serialization; they are
            # gap topics whatever the snapshot says (stale-session chips).
            gap_lines.append(s)
            continue
        evidence_lines.append(s)
    return "\n".join(gap_lines), "\n".join(evidence_lines)


def stream_step2(job_description: str, resume_source: str, notes: str,
                 session: str | None, groq_api_key: str | None = None) -> Iterator[dict]:
    """Step 2: resumes the interrupted session (or starts fresh).

    groq_api_key (optional) honors a client-supplied Groq key - see stream_step1.

    Final event: ``{"type": "result", **payload}``.
    """
    if not job_description or not job_description.strip():
        yield {"type": "error", "message": "Please provide a Job Description (text or URL)."}
        return
    if not resume_source:
        yield {"type": "error", "message": "Please upload a resume PDF or paste your resume text."}
        return

    with groq_key_override(groq_api_key):
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
                            yield _status(f"{NODE_STATUS.get(node, node + '...')} (1/2)", node)
                # refresh the snapshot so the pristine scanner chips below
                # come from THIS run, not a previous thread's state
                state = agent_app.get_state(config)

            # Inject the user's edited gaps/context at the human-review
            # checkpoint. SPLIT (live runs a32f30fc / 868e406d, Nov 6 2026):
            # stamping the whole composed notes string into BOTH fields
            # caused the model to fold the kept chips (which sit directly
            # under the user's pasted project text in the composed string)
            # into fake "user-confirmed" bullets ("Gained experience with
            # n8n (or similar ...)"). missing_skills receives ONLY the gap
            # lines - chips it can treat as research topics and "Do NOT
            # feature" directives - while human_notes receives the typed
            # evidence the improver is allowed to cite.
            pristine_missing = str((state.values or {}).get("missing_skills") or "")
            gap_lines, evidence_lines = _split_composed_notes(
                notes or "", pristine_missing)
            agent_app.update_state(config, {
                "human_notes": evidence_lines,
                "missing_skills": gap_lines,
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
                        yield _status(f"{NODE_STATUS.get(node, node + '...')} (2/2)", node)
        except Exception as e:
            traceback.print_exc()
            yield {"type": "error", "message": f"Optimization failed: {e}"}
            return

        # Reducer channels (prompt_versions) merge across concurrent node
        # writes - the in-loop delta accumulation above is last-writer-wins,
        # so those keys must be pulled from the authoritative final snapshot
        # or the parallel cover_letter / interview_prep stamps would clobber
        # each other (and the reviewer/improver entries).
        snapshot = agent_app.get_state(config)
        merged_versions = (snapshot.values or {}).get("prompt_versions")
        if merged_versions:
            final_state["prompt_versions"] = merged_versions

        if not final_state:
            final_state = snapshot.values or {}

        # Score + failure feedback on the pinned trace root. Degraded runs become
        # queryable on the LangSmith dashboard (composite_score low / review_failed
        # = 1) even though no run inside the trace ever carried error status.
        # (Only successful full runs reach this point - pipeline errors returned
        # above, mirroring the original Gradio semantics.)
        log_pipeline_feedback(final_state, resume_run_id)

        # Integrity tripwire: surface scrubbed + unevidenced content BEFORE
        # the result lands. The client turns this into its warning banner, so
        # the user sees what was deterministically removed and which numeric
        # claims still need review (live-run evidence Nov 6 2026: "cut ... by
        # 40%" with an empty user context; fabricated n8n/Claude Code/MCP
        # bullets in runs a32f30fc / 868e406d).
        integrity = final_state.get("guardrail_warnings") or []
        if integrity:
            yield {"type": "warning",
                   "message": "Integrity check: " + "; ".join(integrity)
                              + ". Verify the resume before applying."}

        yield {"type": "result", "session": session,
               **_extract_results(final_state)}
