"""LLM factory functions and pre-configured instances."""
import logging
import os
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from .config import _PRIMARY_MODEL, _FALLBACK_MODEL, _EMERGENCY_MODEL

logger = logging.getLogger(__name__)


def _groq_models_status(key: str, timeout_seconds: float = 6.0) -> tuple[bool, str]:
    """One cheap REST call (GET /models) - no generation tokens are spent.

    Returns (ok, message). Auth failures and missing keys are hard problems;
    network failures and odd statuses are soft warnings.
    """
    import json
    import os
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status == 200:
                ids = [m.get("id", "") for m in json.load(response).get("data", [])]
                if ids and _PRIMARY_MODEL not in ids:
                    return True, (f"API reachable ({len(ids)} models), but primary model "
                                  f"'{_PRIMARY_MODEL}' is not among them - set PRIMARY_MODEL "
                                  f"to a model your account can access.")
                return True, f"API reachable ({len(ids)} models)."
            return True, f"API reachable (HTTP {response.status})."
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"GROQ_API_KEY rejected by the API (HTTP {e.code}) - check the key in .env."
        if e.code == 429:
            return True, "API rate-limited (HTTP 429) - the key is valid; free-tier quota resets shortly."
        return True, f"API reachable but returned HTTP {e.code} - key looks valid."
    except Exception as e:
        return False, f"Could not reach the Groq API ({e}) - check connectivity. Nothing was spent (models list call, zero generation tokens)."


def check_provider_health(timeout_seconds: float = 6.0) -> tuple[bool, str]:
    """Startup health check for the LLM provider.

    Fails quickly with an actionable message instead of letting the user hit a
    confusing error deep in Step 2. Returns (ok, message); never raises.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or "your_groq_key" in api_key.lower():
        return False, "GROQ_API_KEY is not set. Add it to .env (get a free key: https://console.groq.com/keys)."
    return _groq_models_status(api_key, timeout_seconds)

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

logger.info(f"[Models] Primary: {_PRIMARY_MODEL} | Fast: {_FALLBACK_MODEL} | Emergency: {_EMERGENCY_MODEL}")
