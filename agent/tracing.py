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


def _redact_pii(obj):
    """Recursively mask emails and phone numbers in traced run inputs."""
    if isinstance(obj, str):
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
                             hide_inputs=_redact_pii)
            TRACE_CALLBACKS = [LangChainTracer(client=_CLIENT, project_name=_project)]
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            print("[LangSmith] Tracing enabled with PII redaction - project:", _project)
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
