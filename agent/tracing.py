"""LangSmith tracing with client-side PII redaction."""
import os
import re as _re

_EMAIL_RE = _re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_PHONE_RE = _re.compile(r'(?<![\w.-])(?:\+?\d[\d\s().-]{7,}\d)(?![\w.-])')


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


# Callbacks passed to every graph invoke; empty when tracing is disabled.
TRACE_CALLBACKS: list = []

if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    if os.environ.get("LANGCHAIN_API_KEY"):
        _project = os.environ.setdefault("LANGCHAIN_PROJECT", "resume-optimizer")
        try:
            from langsmith import Client
            from langchain_core.tracers.langchain import LangChainTracer
            _client = Client(api_key=os.environ["LANGCHAIN_API_KEY"],
                             hide_inputs=_redact_pii)
            TRACE_CALLBACKS = [LangChainTracer(client=_client, project_name=_project)]
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            print("[LangSmith] Tracing enabled with PII redaction - project:", _project)
        except Exception as _e:
            print(f"[LangSmith] Tracing disabled (client init failed: {_e})")
    else:
        print("[LangSmith] LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. Tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
