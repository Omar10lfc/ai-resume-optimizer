"""NODE 3: Reviewer - LLM quality judgment."""
import re

from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted, ReviewOutput
from .. import llms as _llms
from ..provider import invoke_structured_with_repair, extract_failed_generation
from ..helpers import _safe_print, _safe_truncate, _strip_markdown_for_review
from langchain_core.messages import HumanMessage, SystemMessage

# Bump when the review prompt or its output schema changes materially.
# review@2: adds the structured "sections" answer for the polisher's
# section-targeted mode (MR1).
REVIEWER_PROMPT_VERSION = "reviewer@2"

# Trace evidence (Oct 2026, export runs-all-20261021): the Reviewer's structured
# output folds with a "Tool choice is required, but model did not call a tool"
# BadRequestError (code tool_use_failed). Trace 01a15f6d proved the failure is
# content/protocol-shaped, NOT a transient blip: attempt 1 and the identical
# same-payload retry folded within 3 seconds. Crucially, the error payload
# carries 'failed_generation' - the model's prose answer (score + feedback) that
# it produced but failed to package as a tool call. MR5 therefore routes the
# call through the provider ladder (invoke_structured_with_repair): attempt 2
# is a schema REPAIR (the model sees its own answer + the schema and
# re-packages - a different payload, so the identical-fold objection does not
# apply), and the prose salvage below remains the free deterministic fallback
# when even the repair folds. The cap stays at TWO LLM calls, then give up.

# Unicode space variants (narrow no-break U+202F, NBSP, thin spaces, minus
# signs) that Groq's prose answers inject between score tokens: '**\u2248 80 /
# 100**'. Normalize before parsing. Also strip markdown emphasis (**).
_NORM_SPACE_RE = re.compile("[\\u00a0\\u2000-\\u200b\\u202f\\u2212]")
_SCORE_FRACTION_RE = re.compile(r"[~≈]?\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100")
_SCORE_LABELLED_RE = re.compile(r"score[^:\d]{0,12}[:：]\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"feedback\s*:[:\s]*(.+?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
_SALVAGE_DEFAULT_FEEDBACK = ("Score recovered from a failed structured-output "
                             "response; no specific feedback was parsed.")


def _parse_salvaged_score(text: str) -> int | None:
    """Extract a 0-100 score from normalized prose.

    Prefers a fraction on a line that names the combined/weighted quality
    score (trace evidence: '**Combined weighted quality score:** **≈ 80 /
    100**'); falls back to the first N/100 fraction anywhere, then to a
    'score: N' label. Clamps to [0, 100]."""
    candidates = []
    for line in text.splitlines():
        m = _SCORE_FRACTION_RE.search(line)
        if m:
            named = bool(re.search(r"combin|weigh|quality|total|overall|score", line, re.IGNORECASE))
            candidates.append((named, m))
    chosen = None
    for named, m in candidates:  # first keyword-named fraction wins
        if named:
            chosen = m
            break
    if chosen is None and candidates:
        chosen = candidates[0][1]
    score = None
    if chosen is not None:
        score = float(chosen.group(1))
    else:
        m = _SCORE_LABELLED_RE.search(text)
        if m:
            score = float(m.group(1))
    if score is None:
        return None
    return max(0, min(100, int(round(score))))


def _parse_salvaged_feedback(text: str) -> str:
    """Extract the one-sentence feedback after a 'Feedback:' marker (case
    insensitive), or the first substantial line that is not the score line."""
    m = _FEEDBACK_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) >= 12 and not _SCORE_FRACTION_RE.search(line) \
                and not re.match(r"feedback\b", line, re.IGNORECASE):
            return line
    return _SALVAGE_DEFAULT_FEEDBACK


def _salvage_review_output(exc: BaseException) -> ReviewOutput | None:
    """Parse score + feedback from the model's prose answer inside a
    structured-output failure. Returns None when the failure offers no
    salvageable generation (the ladder then raises into the caller)."""
    generation = extract_failed_generation(exc)
    if not generation:
        return None
    text = _NORM_SPACE_RE.sub(" ", generation.replace("**", "")).strip()
    score = _parse_salvaged_score(text)
    if score is None:
        return None
    return ReviewOutput(score=score, feedback=_parse_salvaged_feedback(text))


def reviewer_node(state: AgentState):
    _safe_print(f"\n--- NODE 3: REVIEWING DRAFT ---")

    prompt = f"""
    You are a senior hiring manager judging the QUALITY of a resume's presentation.
    Do NOT judge keyword coverage - that is measured separately.

    You are reading the FLATTENED TEXT of the rendered document: markdown
    syntax has been removed so what you see matches what a human reads in
    the generated PDF. Grade the content and structure that text conveys,
    not any markup.

    Rate each dimension 0-100 and combine them:
    - Relevance & Tailoring (40%): Does the summary mirror the exact target role
      and its top requirements? Are skills ordered most-relevant-first? Does the
      resume use the JD's terminology for experience the candidate truly has?
    - Quantified Achievements (30%): Do bullets follow "Accomplished [X] as
      measured by [Y] by doing [Z]" - strong action-verb openers plus at least
      one true metric (money, %, time, scale, before -> after)? Penalize
      duty-only bullets and passive openers ("Responsible for", "Helped with",
      "Worked on", "Assisted with").
    - Formatting & Clarity (30%): Standard section headers, scannable single
      structure, bullets kept to 1-2 lines, natural keyword use (no stuffing),
      free of filler?

    TRUTHFULNESS GATE (overrides the anchors above): the candidate's evidence
    is the ORIGINAL RESUME plus explicit user context ONLY. If the rewritten
    resume claims tools, projects, integrations, employers, or NUMBERS that
    neither of those shows, that is fabrication: cap the score at 40 and name
    the invented claims in the feedback. Do not reward good style for
    invented content.

    FEEDBACK SCOPE (live-run evidence Nov 7 2026, requests d8756ddf /
    a9da3660: salvaged reviewer prose told the writer to "explicitly mention
    experience with n8n / Claude Code / MCP" - advice that can only be
    followed by inventing experience, which the pipeline then scrubs at
    wasted cost): grade PRESENTATION QUALITY ONLY. Never recommend adding
    tools, technologies, employers, projects, or metrics the candidate's
    evidence does not show - missing technologies are surfaced separately
    as keyword gaps, and a truthful resume cannot add them. Direct feedback
    at what can be improved WITHOUT new unverifiable claims: stronger
    action-verb openers, sharper quantified phrasing for existing true
    achievements, tighter structure, role-aligned wording of existing
    experience.

    Calibration anchors - anchor the combined score to these reference points:
    - 85-100: Every bullet opens with a strong action verb and carries a true,
      verifiable metric (numbers, percentages, or scale); the summary names the
      exact target role; skills are ordered most-relevant-first; zero filler.
    - 70-84: Most bullets are quantified but 1-2 are duty-only; summary is close
      to the target role but not an exact mirror; occasional wording padding.
    - 50-69: Mixed - several duty-only bullets lacking metrics, a generic
      summary that could apply to any role, uneven structure.
    - Below 50: Mostly duty descriptions without measurable outcomes; content
      is generic (not tailored to this JD); weak or verbose phrasing.

    {_untrusted('JOB DESCRIPTION', _safe_truncate(state['job_text'], 2500, 'Job text (reviewer)'))}

    {_untrusted('ORIGINAL RESUME (the only source of true claims beside user context)', _safe_truncate(state['resume_text'], 6000, 'Original resume (reviewer)'))}

    {_untrusted('RESUME (flattened plain-text view of the rendered document)', _strip_markdown_for_review(state['optimized_resume']))}

    Provide the combined weighted quality score and one specific sentence of feedback to improve it.

    SECTION TARGETING (mechanical follow-up, code assembles the document): in
    "sections", list which resume sections your feedback addresses, choosing
    ONLY from this fixed vocabulary (as many as apply):
    Professional Summary, Technical Skills, Experience, Projects, Education,
    Certifications. Return nothing when the feedback is general, about
    formatting/length, or spans every section - the code then rewrites the
    whole document, and names outside the vocabulary have the same effect.
    """

    messages = [
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ]

    try:
        result, method = invoke_structured_with_repair(
            _llms.llm_strict, ReviewOutput, messages,
            salvage=_salvage_review_output, label="reviewer")
        _safe_print(f"-> LLM quality score: {result.score}/100")
        _safe_print(f"-> Feedback: {result.feedback}")
        if method != "direct" and result.sections:
            _safe_print(f"-> Feedback targets: {', '.join(result.sections)}")
        elif method == "salvage":
            _safe_print("-> (salvaged prose carries no section targets - the"
                        " polisher will consider the whole document)")
        return {"llm_quality_score": result.score, "feedback": result.feedback,
                "feedback_sections": [str(s) for s in result.sections],
                "review_failed": False,
                "prompt_versions": {"reviewer": REVIEWER_PROMPT_VERSION}}
    except Exception:
        _safe_print("   [Warning] Review failed after 2 attempts. Flagging review as failed.")
        return {"review_failed": True,
                "feedback": "Review could not be completed. Proceeding with current draft."}