"""LLM factory functions and pre-configured instances."""
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from .config import _PRIMARY_MODEL, _FALLBACK_MODEL, _EMERGENCY_MODEL

try:
    from groq import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError
    _RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
except ImportError:
    _RETRYABLE_EXCEPTIONS = (Exception,)


def _create_llm(temperature: float, reasoning_effort: str | None = None):
    """Creates an LLM with automatic fallback for transient provider failures."""
    def _build(model: str) -> ChatGroq:
        kwargs = dict(model=model, temperature=temperature, max_retries=2)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return ChatGroq(**kwargs)

    return _build(_PRIMARY_MODEL).with_fallbacks(
        [_build(_FALLBACK_MODEL), ChatGroq(model=_EMERGENCY_MODEL,
                                           temperature=temperature, max_retries=2)],
        exceptions_to_handle=_RETRYABLE_EXCEPTIONS,
    )


def _create_llm_fast(temperature: float):
    """Creates a faster LLM for non-critical tasks."""
    fast = ChatGroq(model=_FALLBACK_MODEL, temperature=temperature, max_retries=2,
                    reasoning_effort="low")
    fallback = ChatGroq(model=_EMERGENCY_MODEL, temperature=temperature, max_retries=2)
    return fast.with_fallbacks([fallback], exceptions_to_handle=_RETRYABLE_EXCEPTIONS)


# Three LLM profiles:
#   strict: scoring/analysis (primary, temp=0, low reasoning)
#   creative: resume writing (primary, temp=0.3, full reasoning)
#   fast: cover letter & interview prep (faster model, temp=0.3, low reasoning)
llm_strict = _create_llm(temperature=0, reasoning_effort="low")
llm_creative = _create_llm(temperature=0.3)
llm_fast = _create_llm_fast(temperature=0.3)

print(f"[Models] Primary: {_PRIMARY_MODEL} | Fast: {_FALLBACK_MODEL} | Emergency: {_EMERGENCY_MODEL}")
