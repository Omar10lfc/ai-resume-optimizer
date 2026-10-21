"""NODE 3: Reviewer - LLM quality judgment."""
import re

from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted, ReviewOutput
from .. import llms as _llms
from ..helpers import _safe_print, _safe_truncate
from langchain_core.messages import HumanMessage, SystemMessage

# Trace evidence (Oct 2026, export runs-all-20261021): the Reviewer's structured
# output folds with a "Tool choice is required, but model did not call a tool"
# BadRequestError (code tool_use_failed). Trace 01a15f6d proved the failure is
# content/protocol-shaped, NOT a transient blip: attempt 1 and the identical
# same-payload retry folded within 3 seconds. Crucially, the error payload
# carries 'failed_generation' - the model's prose answer (score + feedback) that
# it produced but failed to package as a tool call. So on this failure class we
# PARSE the prose straight out of the exception instead of retrying identical
# bytes; the generic retry below only remains for true transient errors.
_REVIEW_ATTEMPTS = 2

# Unicode space variants (narrow no-break U+202F, NBSP, thin spaces, minus
# signs) that Groq's prose answers inject between score tokens: '**\u2248 80 /
# 100**'. Normalize before parsing. Also strip markdown emphasis (**).
_NORM_SPACE_RE = re.compile("[\\u00a0\\u2000-\\u200b\\u202f\\u2212]")
_SCORE_FRACTION_RE = re.compile(r"[~≈]?\s*(\d{1,3}(?:\.\d+)?)\s*/\s*100")
_SCORE_LABELLED_RE = re.compile(r"score[^:\d]{0,12}[:：]\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"feedback\s*:[:\s]*(.+?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
_SALVAGE_DEFAULT_FEEDBACK = ("Score recovered from a failed structured-output "
                             "response; no specific feedback was parsed.")
# Fallback when the exception object is wrapped and only str(exc) survives,
# matching how the trace export recorded the error string. The repr carries
# escaped unicode (\u202f becomes six ASCII chars) that must be decoded first.
_FAILED_GENERATION_STR_RE = re.compile(r"failed_generation[\"']?\s*:?\s*'(.+?)'\s*\}\}", re.DOTALL)
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _extract_failed_generation(exc: BaseException) -> str | None:
    """Best-effort extraction of the model's prose answer from a structured
    output failure. Groq/OpenAI BadRequestError (tool_use_failed) carries the
    generated text in body['error']['failed_generation']; some SDK versions
    expose the same envelope on .response.json(), and wrapped errors may only
    keep the repr string."""
    candidates: list = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            candidates.append(err.get("failed_generation"))
        candidates.append(body.get("failed_generation"))
    if not any(isinstance(c, str) and c.strip() for c in candidates):
        try:
            resp = getattr(exc, "response", None)
            payload = resp.json() if resp is not None else None
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                candidates.append(err.get("failed_generation"))
        except Exception:
            pass
    if not any(isinstance(c, str) and c.strip() for c in candidates):
        try:
            raw = _FAILED_GENERATION_STR_RE.search(str(exc) or "")[1]
            raw = _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m[1], 16)), raw)
            raw = raw.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            candidates.append(raw)
        except (AttributeError, IndexError, TypeError):
            pass
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


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
    salvageable generation (caller then re-raises into the retry loop)."""
    generation = _extract_failed_generation(exc)
    if not generation:
        return None
    text = _NORM_SPACE_RE.sub(" ", generation.replace("**", "")).strip()
    score = _parse_salvaged_score(text)
    if score is None:
        return None
    return ReviewOutput(score=score, feedback=_parse_salvaged_feedback(text))


def reviewer_node(state: AgentState):
    _safe_print(f"\n--- NODE 3: REVIEWING DRAFT ---")

    structured_llm = _llms.llm_strict.with_structured_output(ReviewOutput)

    prompt = f"""
    You are a senior hiring manager judging the QUALITY of a resume's presentation.
    Do NOT judge keyword coverage - that is measured separately.

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

    {_untrusted('JOB DESCRIPTION', _safe_truncate(state['job_text'], 2500, 'Job text (reviewer)'))}

    {_untrusted('RESUME', state['optimized_resume'])}

    Provide the combined weighted quality score and one specific sentence of feedback to improve it.
    """

    messages = [
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ]

    last_error = None
    for attempt in range(1, _REVIEW_ATTEMPTS + 1):
        try:
            result = structured_llm.invoke(messages)
            _safe_print(f"-> LLM quality score: {result.score}/100")
            _safe_print(f"-> Feedback: {result.feedback}")
            if attempt > 1:
                _safe_print(f"-> (recovered on attempt {attempt}/{_REVIEW_ATTEMPTS})")
            return {"llm_quality_score": result.score, "feedback": result.feedback,
                    "review_failed": False}
        except Exception as e:
            last_error = e
            # Content-shaped failure (tool call folded, prose exists): salvage
            # instead of retrying identical bytes - trace 01a15f6d showed the
            # same-payload retry folding identically 3s apart.
            salvaged = _salvage_review_output(e)
            if salvaged is not None:
                _safe_print(f"-> Salvaged score from failed generation: {salvaged.score}/100")
                return {"llm_quality_score": salvaged.score, "feedback": salvaged.feedback,
                        "review_failed": False}
            _safe_print(f"   [Warning] Structured output attempt {attempt}/{_REVIEW_ATTEMPTS} failed: {e}")

    _safe_print(f"   [Warning] Review failed after {_REVIEW_ATTEMPTS} attempts. Flagging review as failed.")
    return {"review_failed": True,
            "feedback": "Review could not be completed. Proceeding with current draft."}
