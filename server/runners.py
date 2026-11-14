"""Two-step pipeline runners ported from the legacy Gradio app (app.py).

Framework-agnostic sync generators that drive the interactive LangGraph
graph (``agent_app``) with identical semantics to the Gradio handlers, so
the FastAPI SSE layer and the Next.js frontend share the same flow:

- Step 1 runs loader -> scanner, then the graph INTERRUPTS before the
  Improver (first-class human-in-the-loop checkpoint). The scanner's
  missing-skills output is returned for the review/edit box.
- Step 2 resumes the interrupted thread: if Step 1 was skipped it first
  runs loader -> scanner up to the interrupt, injects the user's edited
  notes via ``update_state`` (probe-calibrated Nov 11: providing the notes
  as graph INPUT re-enters the graph from the entry point - one wasted
  fast-model call, and the re-run scanner clobbers the user's gap lines),
  then resumes with a PINNED root run id so pipeline feedback records
  attach to the exact LangSmith trace. Trace visibility of the injection
  is restored by the ``notes_injection`` side run (MR3) whose input is the
  split notes payload.

MR3 (durable, reattachable runs): both runners are wrapped by
``_recorded`` so every attempt is tracked in the ``run_store`` ledger
(status / per-step progress / step-1 chips / errors / done-run result
payloads). Mid-pipeline leftovers from a crashed attempt are never
resumed where they stopped - they are rewound to the human-review
checkpoint with every per-run channel reset, which fixes the stale-ATS
snapshot (shipped ATS 56.2 while the shipped resume recomputed to
37.5/43.8) and the 7.4s band-aid stitch on live runs 29afd91b /
62725081.
"""

import functools
import inspect
import re
import traceback
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from langchain_core.runnables import RunnableLambda

from agent import TRACE_CALLBACKS, agent_app
from agent import config as agent_config
from agent.helpers import _safe_truncate
from agent.llms import groq_key_override
from agent.provider import TokenUsageTracker
from agent.tracing import (
    candidate_name_hint,
    log_pipeline_feedback,
    register_known_name,
)
from server import profile_store, run_store

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


def _thread_config(thread_id: str, run_id: str | None = None,
                   callbacks: list | None = None) -> dict:
    """RunnableConfig for one graph segment.

    run_id (optional) pins the ROOT run id of the segment's trace: LangGraph
    reads config['run_id'] for the root run, so pipeline feedback records can
    target a known trace root instead of scraping the tracer's run tree.
    callbacks (optional) EXTENDS the trace callbacks with per-run handlers
    (MR5: the TokenUsageTracker); the shared TRACE_CALLBACKS list is never
    mutated - a fresh list is built.
    """
    cfg = dict(_INVOKE_CONFIG_BASE)
    cfg["configurable"] = {"thread_id": thread_id}
    if run_id:
        cfg["run_id"] = run_id
    if callbacks:
        cfg["callbacks"] = [*TRACE_CALLBACKS, *callbacks]
    return cfg


def _recovery_reset() -> dict:
    """Per-run channels cleared for a mid-pipeline recovery (MR3).

    A crashed attempt leaves its scoring snapshot (best_result / score /
    ats_*) in the checkpoint; resuming from there ships the stale numbers
    with whatever the recovery produced. Recovery rewinds to the human
    checkpoint and resets every per-run channel so the redone segment
    recomputes scoring on ITS OWN drafts. Loader outputs (resume_text /
    job_text / output_dir / request_id) are deliberately kept: the
    uploaded temp file is already deleted when recovery runs, so the
    checkpointed parsed text is the only durable copy. prompt_versions is
    not reset - it is reducer-merged (cannot be cleared) and its stamps
    are re-emitted identically by the re-run nodes.
    """
    return {
        "optimized_resume": "", "feedback": "", "score": 0,
        "llm_quality_score": 0, "ats_percentage": 0.0,
        "review_failed": False, "iteration": 0, "ats_result": "",
        "best_result": {}, "cover_letter": "", "interview_questions": "",
        "resume_pdf_path": "", "cover_letter_pdf_path": "",
        "resume_docx_path": "", "resume_tex_path": "",
        "resume_pages": 0, "resume_compacted": False,
        "page_overflow_warning": "", "guardrail_warnings": [],
        "render_engine": "",
        "scrub_removals": [], "unevidenced_metrics": [],
        "declined_requests": [], "requirement_matches": [],
        "feedback_sections": [], "polish_sections_targeted": [],
        "cover_letter_metrics": [],
        "human_notes": "", "missing_skills": "",
    }


def _trace_notes_injection(session: str, human_notes: str,
                           missing_skills: str) -> None:
    """MR3: a real traced side run whose INPUT is the injected notes payload.

    ``update_state`` is not a runnable invocation, so the state write never
    reaches LangSmith - which blinded post-run forensics on every resumed
    invoke (runs 29afd91b / 62725081: "what did the user actually ask
    for?" was unanswerable from the trace). This side run closes that gap:
    it carries the split notes (evidence + gap lines) as its input,
    PII-redacted by the tracer's ``hide_inputs`` hook like every other run,
    and is named for project-level filtering. Tracing disabled (tests,
    air-gapped) -> silent no-op; any failure is non-fatal by contract.
    """
    if not TRACE_CALLBACKS:
        return
    try:
        RunnableLambda(lambda _: {"written": True},
                       name="notes_injection").invoke(
            {"session": session,
             "human_notes": human_notes,
             "missing_skills": missing_skills},
            config={"callbacks": list(TRACE_CALLBACKS)},
        )
    except Exception as exc:
        print(f"[Tracing] notes-injection side run failed (non-fatal): {exc}")


def _recorded(runner):
    """MR3: records a runner's lifecycle into the durable runs ledger.

    The session id is normalized here (generated when absent) so the row
    exists from the first event; the runner's fresh-session switch
    (completed-thread reuse) re-points the recording and marks the
    superseded row. Events pass through untouched; errors, crashes, and
    client disconnects are RECORDED, never masked. Ledger failures are
    non-fatal by contract (run_store prints and continues)."""
    @functools.wraps(runner)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(runner).bind(*args, **kwargs)
        bound.apply_defaults()
        session = str(bound.arguments.get("session") or uuid4())
        bound.arguments["session"] = session
        run_store.start(session, "step1" if runner is _stream_step1 else "step2")
        gen = runner(*bound.args, **bound.kwargs)
        try:
            for event in gen:
                etype = event.get("type")
                if etype == "session" and event.get("session"):
                    run_store.supersede(
                        session,
                        "Superseded - a fresh session was started because"
                        " this one already held a completed run.")
                    session = event["session"]
                    run_store.start(session, "step2")
                elif etype == "status" and event.get("node"):
                    run_store.step(session, event["node"])
                elif etype == "done":
                    run_store.finish_scan(
                        session, str(event.get("missing_skills") or ""))
                elif etype == "result":
                    run_store.finish(session, event)
                elif etype == "error":
                    run_store.fail(session, str(event.get("message") or "error"))
                yield event
        except GeneratorExit:
            # Client disconnected mid-stream: the LLM work stops at the next
            # checkpoint and the row must not linger as 'running' (it would
            # only be fixed by the next backend restart).
            run_store.fail(session,
                           "Client disconnected mid-run - run Step 2 again.")
            raise
        except Exception as exc:  # backstop: record, never mask
            run_store.fail(session, f"Pipeline crashed: {exc}")
            raise
    return wrapper


def _capped_job_description(job_input: str) -> str:
    """Input cap (the Dossier study's MAX_POSTING_CHARACTERS pattern): URL
    fetches are already truncated in ``agent.url_fetch``; PASTED job
    descriptions were not. One deterministic cap at the stream entry so both
    steps, the ledgers, and the profile store see the same text."""
    return _safe_truncate(job_input, agent_config.MAX_JOB_CHARS,
                          "Job description")


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


def _stream_step1(job_description: str, resume_source: str,
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
    job_description = _capped_job_description(job_description)

    with groq_key_override(groq_api_key):
        premask_resume_name(resume_source)
        session = session or str(uuid4())
        usage = TokenUsageTracker()
        config = _thread_config(session, callbacks=[usage])

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
               "token_usage": usage.totals(),
               "task_usage": usage.task_totals(),
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
        # Cover-letter integrity scan (live run 22e943c6): the letter's own
        # unevidenced numbers, rendered by the results Integrity tab.
        "cover_letter_metrics": final_state.get("cover_letter_metrics") or [],
        "prompt_versions": final_state.get("prompt_versions") or {},
        # BB2: which engine produced the resume PDF (html-chromium@1 /
        # latex@1 / xhtml2pdf@1) - the renderer stamp beside the prompt
        # stamps, one lookup for "which engine produced this run".
        "render_engine": final_state.get("render_engine") or "",
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


# Gap-chip opener phrasings the scanner's missing-skills list uses (live
# chips: "Experience with n8n (or similar workflow automation platform)",
# "Hands-on work with Claude Code, Cursor, or Codex", "Experience working
# with MCP servers (FortyGuard Temperature MCP)", "Direct experience with
# MCP servers (...)"). Matched against the NORMALIZED line start, so
# bullet markers and case do not matter.
_CHIP_OPENERS = (
    "experience working with", "direct experience", "hands-on",
    "hands on", "experience with", "experience in", "experience of",
    "familiarity with", "familiar with", "working knowledge",
    "knowledge of", "comfortable with", "proficient in", "proficient with",
    "proficiency in", "proficiency with", "exposure to", "understanding of",
    "awareness of",
)


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

    Chip-opener hardening (round 7, runs 25ff4ae5 / 1a8125f0, Nov 12
    2026): the pasted context carried the scanner's MCP gap phrasings
    BARE (no "- " bullet). Each run's snapshot matched one variant
    ("Experience working with MCP ..." / "Direct experience with MCP ..."
    - the scanner words chips slightly differently per run), so that one
    went to the gap channel and was declined on-record, while the OTHER
    variant missed the snapshot, carried no ASCII bullet, and fell into
    the evidence channel as "user-confirmed experience" - the 868e406d
    laundering shape, dash-free edition. A line OPENING with the scanner's
    gap phrasing (_CHIP_OPENERS) is a gap topic whatever the snapshot
    says, same safe-failure direction as the ASCII rule.
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
        if any(n.startswith(op) for op in _CHIP_OPENERS):
            # Bare chip-shaped lines (round 7 runs): the scanner's gap
            # phrasing without a bullet marker and without a snapshot
            # match. Gap topic, never evidence - see the docstring.
            gap_lines.append(s)
            continue
        evidence_lines.append(s)

    # Exact-normalized dedupe of the gap channel (live run 22e943c6, Nov 11
    # 2026: a retry after a mid-pipeline crash re-carried chip lines the
    # crashed attempt had already injected into the checkpoint, doubling the
    # gap channel the improver reads as research topics). First occurrence
    # wins; only IDENTICAL lines (after _norm_gap_line) collapse - wording
    # variants are composed client-side (frontend/src/lib/gaps.ts serializes
    # fresh per submit) and are unknowable server-side.
    seen: set[str] = set()
    deduped: list[str] = []
    for line in gap_lines:
        key = _norm_gap_line(line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return "\n".join(deduped), "\n".join(evidence_lines)


def _stream_step2(job_description: str, resume_source: str, notes: str,
                  session: str | None, groq_api_key: str | None = None,
                  include_original: bool = False) -> Iterator[dict]:
    """Step 2: resumes the interrupted session (or starts fresh).

    groq_api_key (optional) honors a client-supplied Groq key - see stream_step1.
    include_original (BB1 Phase B / PB3): echoes the PARSED original resume
    text in the final result payload so the SPA's before/after diff works
    for uploads too (default off - paste mode already holds the text).

    Session-entry semantics (MR3):
    - ``next == ("improver",)`` - the durable human-review checkpoint: the
      normal path.
    - empty next + values - the thread already completed: a FRESH session
      is started (surfaced as a ``session`` event).
    - anything else - mid-pipeline leftovers of a crashed attempt: never
      resumed where they stopped; rewound to the human checkpoint with
      per-run channels reset (``_recovery_reset``), then the model segment
      is redone from the improver. A crashed SCANNER is re-run first (one
      fast-model call) so the fresh chips are available to the notes split.

    Final event: ``{"type": "result", **payload}``.
    """
    if not job_description or not job_description.strip():
        yield {"type": "error", "message": "Please provide a Job Description (text or URL)."}
        return
    if not resume_source:
        yield {"type": "error", "message": "Please upload a resume PDF or paste your resume text."}
        return
    job_description = _capped_job_description(job_description)

    with groq_key_override(groq_api_key):
        premask_resume_name(resume_source)

        if not notes or not notes.strip():
            yield {"type": "warning",
                   "message": "No missing skills / context provided. Running with auto-detected gaps only."}

        session = session or str(uuid4())
        # MR5: one usage tracker for the WHOLE step-2 attempt (any recovery
        # re-runs + the resume segment all count toward this task).
        usage = TokenUsageTracker()
        config = _thread_config(session, callbacks=[usage])

        yield _status("Starting pipeline...")
        final_state: dict = {}
        rewound = False
        try:
            state = agent_app.get_state(config)
            next_nodes = tuple(state.next or ())
            if next_nodes and next_nodes != ("improver",):
                # MR3 recovery: mid-pipeline leftovers of a crashed attempt.
                # The 7.4s score-59 band-aid run and the stale-ATS snapshot
                # (shipped 56.2, shipped-resume recompute 37.5/43.8 on live
                # runs 29afd91b / 62725081) both came from resuming such a
                # checkpoint where it stopped. Rewind to the human-review
                # checkpoint and redo the model segment from the improver.
                yield {"type": "warning",
                       "message": "Recovered an interrupted run - restarting the optimization from your review checkpoint."}
                if next_nodes == ("scanner",):
                    # The loader finished but the scanner never did: re-run
                    # it (one fast-model call) so fresh chips are available
                    # to the notes split below.
                    for _ in agent_app.stream(None, config=config,
                                              stream_mode="updates"):
                        pass
                    state = agent_app.get_state(config)
                    next_nodes = tuple(state.next or ())
                if next_nodes and next_nodes != ("improver",):
                    # The reset itself is DEFERRED to the notes injection
                    # below: reset + notes must land as ONE update_state
                    # anchored with as_node="scanner" (live run 22e943c6 -
                    # see the injection site for the mechanism).
                    rewound = True
            elif not next_nodes:
                if state.values:
                    # previous run completed in this session -> start a fresh one
                    session = str(uuid4())
                    config = _thread_config(session, callbacks=[usage])
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
                next_nodes = tuple(state.next or ())

            # Pristine scanner chips for the notes split: the checkpoint's
            # own copy - EXCEPT on the rewind path, where that copy holds
            # the crashed attempt's injected gap lines; the ledger's step-1
            # snapshot is the durable pristine source there.
            if rewound:
                row = run_store.get(session)
                pristine_missing = str((row or {}).get("missing_skills") or "")
            else:
                pristine_missing = str((state.values or {}).get("missing_skills") or "")

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
            gap_lines, evidence_lines = _split_composed_notes(
                notes or "", pristine_missing)
            # MR3: the write is ``update_state`` - probe-calibrated Nov 11:
            # providing the notes as graph INPUT re-enters the graph from the
            # entry point (the apply-run re-ran loader+scanner: one extra
            # fast-model call per step-2, and the fresh scanner chips CLOBBER
            # the user's edited gap lines - caught by the MR4 wire tests).
            # Trace visibility of the injected notes - lost because
            # ``update_state`` is not a runnable invocation - is restored by
            # the ``notes_injection`` side run below, whose INPUT is the
            # split notes payload (PII-redacted by the tracer's hide_inputs
            # hook like every other run; live runs 29afd91b / 62725081).
            #
            # Rewound path (live run 22e943c6, Nov 11 2026): reset + notes
            # MUST land as ONE update_state anchored with as_node="scanner".
            # An UNANCHORED second write lets LangGraph's as_node inference
            # (no as_node -> the node with the highest versions_seen in the
            # checkpoint, pregel/main.py _infer_as_node) attribute it to the
            # INTERRUPTED polisher task - it had consumed the newest channel
            # versions before crashing - so the notes write advanced `next`
            # past the improver to the fan-out: the "recovered" segment only
            # re-ran cover_letter / interview_prep / pdf_exporter and the
            # run shipped score 0 with "No resume generated." An explicit
            # as_node is immune to the inference; the notes keys come AFTER
            # the spread so they win over the reset's empty strings.
            if rewound:
                agent_app.update_state(config, {
                    **_recovery_reset(),
                    "human_notes": evidence_lines,
                    "missing_skills": gap_lines,
                }, as_node="scanner")
            else:
                agent_app.update_state(config, {
                    "human_notes": evidence_lines,
                    "missing_skills": gap_lines,
                })
            _trace_notes_injection(session, evidence_lines, gap_lines)

            # Resume: improver <-> reviewer -> ats_check -> fan-out -> pdf_exporter.
            # A pinned root run id lets the post-run feedback records attach to
            # the exact LangSmith trace (threshold/root of THIS segment).
            resume_run_id = str(uuid4())
            for update in agent_app.stream(
                    None,
                    config=_thread_config(session, resume_run_id,
                                          callbacks=[usage]),
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

        # Cover-letter integrity (live run 22e943c6, Nov 11 2026): the letter
        # gets its own scan + warning - the resume path's tripwire never
        # watched the letter, which fabricated "70%"/"45%" from nothing.
        # Its own event (not folded into the resume integrity list) because
        # the artifact to verify is different - and cover_letter writes the
        # channel from the concurrent fan-out, so guardrail_warnings
        # (last-writer-wins) would clobber the resume's warnings.
        letter_metrics = final_state.get("cover_letter_metrics") or []
        if letter_metrics:
            yield {"type": "warning",
                   "message": "Integrity check (cover letter): unevidenced "
                              "number(s) " + ", ".join(letter_metrics)
                              + ". Verify the cover letter before sending."}

        results_payload = _extract_results(final_state)
        # PB3: opt-in echo of the PARSED original resume. It lives in the
        # authoritative snapshot (loader output - final_state only carries
        # this segment's deltas on the step1->step2 resume path). The
        # profile store filters this key back out of result_json: the
        # resume text is already stored at application level.
        if include_original:
            results_payload["original_resume"] = str(
                (snapshot.values or {}).get("resume_text") or "")
        yield {"type": "result", "session": session,
               # MR5: per-task token accounting rides the result payload (and
               # lands in the runs ledger for free, since done-run payloads
               # are persisted wholesale).
               "token_usage": usage.totals(),
               "task_usage": usage.task_totals(),
               **results_payload}

        # BB1: record the completed run into the persistent profile store
        # (auto-record, zero UI). The PARSED evidence texts live in the
        # authoritative SNAPSHOT (loader outputs - `final_state` only
        # accumulates this segment's deltas, so on the step1->step2 resume
        # path it never held them); the wrapper only sees the input, which
        # may be a file path. Non-fatal by contract; disabled by
        # PROFILE_DB_PATH=off.
        try:
            snapshot_values = snapshot.values or {}
            profile_store.record_run(
                str(snapshot_values.get("job_text") or job_description or ""),
                str(snapshot_values.get("resume_text") or ""),
                session, results_payload)
        except Exception as exc:  # belt & braces: record_run never raises
            print(f"[Profile] recording skipped (non-fatal): {exc}")


# MR3: the module-level public names are the RECORDING wrappers (the SSE
# layer, tests, and monkeypatchers target these); the raw generators keep
# underscore names.
stream_step1 = _recorded(_stream_step1)
stream_step2 = _recorded(_stream_step2)
