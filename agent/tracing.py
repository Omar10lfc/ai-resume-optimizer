"""LangSmith tracing with client-side PII redaction + score feedback."""
import os
import re as _re

_EMAIL_RE = _re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_PHONE_RE = _re.compile(r'(?<![\w.-])(?:\+?\d[\d\s().-]{7,}\d)(?![\w.-])')

# Feedback keys used by app.py after a completed pipeline run. They turn the
# silent-degradation gap (trace history: review_failed runs showed status
# "success" server-side, and not a single feedback record existed project-wide)
# into queryable dashboard signals.
COMPOSITE_SCORE_FEEDBACK_KEY = "composite_score"
REVIEW_FAILED_FEEDBACK_KEY = "review_failed"

# --- Name redaction (production checklist Item 4) ---
# Emails and phone numbers mask by pattern; a candidate NAME cannot be
# pattern-detected reliably, so it is registered explicitly and then masked
# everywhere it appears in traced payloads - inputs AND outputs (outputs were
# previously not masked at all). Sources:
#   * loader.py / app.py call register_known_name(candidate_name_hint(text))
#     with the name parsed from the resume at runtime.
#   * REDACT_NAMES env (comma-separated) supplies manual values, e.g. for
#     re-exporting runs made before name masking existed.
# Full names mask before standalone tokens (>= 4 chars) to limit collateral;
# over-masking is acceptable for trace/debug data.
_KNOWN_NAMES: list[str] = []
_NAME_MASKS: list = None
_NAME_TOKEN_MIN_CHARS = 4
_NAME_MAX_CHARS = 60

_NAME_LABEL_RE = _re.compile(r"(?i)^\s*name\s*[:：]\s*(.+)$")
_NAME_CHAR_RE = _re.compile(r"^[A-Za-z\u00C0-\u024F'’.\- ]{2,60}$")
# Common resume heading/role words that must NOT count as a person name.
_PERSON_NAME_EXCLUDE_RE = _re.compile(
    r"^(resume|curriculum|vitae|summary|profile|objective|experience|"
    r"education|skills|contact|details|engineer|developer|manager|designer|"
    r"analyst|architect|scientist|student|intern|specialist|consultant|"
    r"expert|lead|senior|junior)$", _re.IGNORECASE)


def _looks_like_person_name(line: str) -> bool:
    """2-4 letter-only capitalized tokens ('John Doe'/'John A. Doe'), no
    digits, no heading/role words like 'SOFTWARE ENGINEER'."""
    if not _NAME_CHAR_RE.match(line or ""):
        return False
    tokens = [t for t in line.split() if t]
    if not (2 <= len(tokens) <= 4):
        return False
    return all(t[0].isalpha() and t[0].isupper()
               and not _PERSON_NAME_EXCLUDE_RE.match(t.strip("."))
               for t in tokens)


def candidate_name_hint(text: str) -> str | None:
    """Heuristic candidate-name extraction from resume text.

    An explicit 'Name: <value>' line wins; otherwise the first non-empty line
    is used when it is name-shaped. Returns None when unsure (conservative:
    better an unmasked debug name in local logs than a masked address heading).
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines[:8]:
        m = _NAME_LABEL_RE.match(line)
        if m:
            raw = " ".join(m.group(1).split())
            # Labeled values often carry a role suffix ('Name: John Doe - CEO');
            # the part before the first separator is the name itself.
            head = _re.split(r"\s+[-|,/•]\s*", raw)[0].strip()
            hint = head if head else raw
            if _looks_like_person_name(hint) \
                    or (4 <= len(hint) <= _NAME_MAX_CHARS and any(c.isalpha() for c in hint)):
                # The label explicitly declares it a name; trust it even when
                # the shape check would fail (single-token surnames, accents).
                return hint
    for line in lines[:3]:
        if _looks_like_person_name(line):
            return line
    return None


def register_known_name(name: str) -> bool:
    """Adds a candidate name to the redaction register.

    Returns True if accepted; junk (paths, headings, empty strings) is
    silently ignored so callers never need to pre-validate. Idempotent.
    """
    clean = " ".join((name or "").split())
    if not (2 <= len(clean) <= _NAME_MAX_CHARS) \
            or not any(c.isalpha() for c in clean):
        return False
    if clean not in _KNOWN_NAMES:
        _KNOWN_NAMES.append(clean)
        _reset_name_masks()
    return True


def _reset_name_masks() -> None:
    """Forces the mask cache to rebuild (after a new name or an env change)."""
    global _NAME_MASKS
    _NAME_MASKS = None


def _name_masks() -> list:
    """Compiled name masks, longest-first (full names before tokens). Built
    lazily so REDACT_NAMES set after import (e.g. by _load_dotenv) counts."""
    global _NAME_MASKS
    if _NAME_MASKS is None:
        env_names = [n.strip() for n in os.environ.get("REDACT_NAMES", "").split(",") if n.strip()]
        names = list(dict.fromkeys(_KNOWN_NAMES + env_names))
        masks = []
        for name in names:
            escaped = _re.escape(name).replace(r"\ ", r"\s+")
            masks.append(_re.compile(r"\b" + escaped + r"\b", _re.IGNORECASE))
        tokens = sorted({tok.lower() for name in names for tok in name.split()
                         if len(tok) >= _NAME_TOKEN_MIN_CHARS}, key=len, reverse=True)
        for token in tokens:
            masks.append(_re.compile(r"\b" + _re.escape(token) + r"\b", _re.IGNORECASE))
        _NAME_MASKS = masks
    return _NAME_MASKS


def _redact_pii(obj):
    """Recursively mask emails, phone numbers and any registered candidate
    name in traced payloads."""
    if isinstance(obj, str):
        for mask in _name_masks():
            obj = mask.sub("[redacted-name]", obj)
        obj = _EMAIL_RE.sub("[redacted-email]", obj)
        obj = _PHONE_RE.sub("[redacted-phone]", obj)
        return obj
    if isinstance(obj, dict):
        return {k: _redact_pii(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_pii(v) for v in obj]
    return obj


# Shared LangSmith client (None when tracing is disabled). Used by both the
# tracer above and log_feedback() below.
_CLIENT = None

# Callbacks passed to every graph invoke; empty when tracing is disabled.
TRACE_CALLBACKS: list = []

if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    if os.environ.get("LANGCHAIN_API_KEY"):
        _project = os.environ.setdefault("LANGCHAIN_PROJECT", "resume-optimizer")
        try:
            from langsmith import Client
            from langchain_core.tracers.langchain import LangChainTracer
            _CLIENT = Client(api_key=os.environ["LANGCHAIN_API_KEY"],
                             hide_inputs=_redact_pii,
                             hide_outputs=_redact_pii)
            TRACE_CALLBACKS = [LangChainTracer(client=_CLIENT, project_name=_project)]
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            print("[LangSmith] Tracing enabled with PII redaction (inputs + outputs) - project:", _project)
        except Exception as _e:
            print(f"[LangSmith] Tracing disabled (client init failed: {_e})")
            _CLIENT = None
    else:
        print("[LangSmith] LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. Tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def log_feedback(run_id, key, *, score=None, value=None, comment=None) -> bool:
    """Attach a feedback record to a LangSmith run (usually the trace root).

    Returns True if a record was emitted; a silent no-op when tracing is
    disabled - runs, tests, and air-gapped deployments never fail on feedback.
    """
    if _CLIENT is None or not run_id:
        return False
    try:
        _CLIENT.create_feedback(run_id, key=key, score=score, value=value,
                                comment=comment)
        return True
    except Exception as exc:
        print(f"[LangSmith] feedback '{key}' failed: {exc}")
        return False


def log_pipeline_feedback(final_state, run_id) -> bool:
    """Record the composite score and the review-failure flag for a completed
    pipeline trace root.

    Degraded runs are then directly alertable on the LangSmith dashboard:
    filter feedback `composite_score < N` or `review_failed = 1`. The composite
    deliberately reflects degradation (review_failed runs score ATS-only), so
    both signals fire on the same incident.
    """
    if not isinstance(final_state, dict):
        return False
    emitted = False
    composite = final_state.get("score")
    if composite is not None:
        emitted = log_feedback(run_id, COMPOSITE_SCORE_FEEDBACK_KEY,
                               score=round(float(composite), 2)) or emitted
    failed = bool(final_state.get("review_failed"))
    emitted = log_feedback(
        run_id, REVIEW_FAILED_FEEDBACK_KEY,
        value=1 if failed else 0,
        comment=("Reviewer structured output failed - composite fell back to "
                 "ATS-only score") if failed else None,
    ) or emitted
    return emitted
