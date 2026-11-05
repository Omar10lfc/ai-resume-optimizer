"""LLM factory functions and pre-configured instances."""
import logging
import os
from contextlib import contextmanager
from typing import Iterator

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from .config import _PRIMARY_MODEL, _FALLBACK_MODEL, _EMERGENCY_MODEL

logger = logging.getLogger(__name__)


def _groq_models_status(key: str, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """One cheap REST call (GET /models) - no generation tokens are spent.

    Returns (ok, message). Auth failures and missing keys are hard problems;
    network failures and odd statuses are soft warnings.
    """
    import json
    import os
    import urllib.error
    import urllib.request

    # If urllib.request.urlopen has been monkeypatched (e.g. by unit tests), use the mock
    if getattr(urllib.request.urlopen, "__module__", "") != "urllib.request":
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "JobApplicationOptimizer/1.0",
            })
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

    # Production path: httpx handles modern TLS/ALPN and avoids Cloudflare 1010 signature blocks
    try:
        import httpx
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {key}",
                    "User-Agent": "JobApplicationOptimizer/1.0",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                ids = [m.get("id", "") for m in data.get("data", [])]
                if ids and _PRIMARY_MODEL not in ids:
                    return True, (f"API reachable ({len(ids)} models), but primary model "
                                  f"'{_PRIMARY_MODEL}' is not among them - set PRIMARY_MODEL "
                                  f"to a model your account can access.")
                return True, f"API reachable ({len(ids)} models)."
            if resp.status_code in (401, 403):
                return False, f"GROQ_API_KEY rejected by the API (HTTP {resp.status_code}) - check the key in .env."
            if resp.status_code == 429:
                return True, "API rate-limited (HTTP 429) - the key is valid; free-tier quota resets shortly."
            return True, f"API reachable but returned HTTP {resp.status_code} - key looks valid."
    except Exception as e:
        return False, f"Could not reach the Groq API ({e}) - check connectivity. Nothing was spent (models list call, zero generation tokens)."


def check_provider_health(timeout_seconds: float = 8.0) -> tuple[bool, str]:
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


def _create_llm(temperature: float, reasoning_effort: str | None = None,
                api_key: str | None = None):
    """Creates an LLM with automatic fallback for transient provider failures.

    api_key (optional) pins a concrete Groq key instead of the GROQ_API_KEY
    environment variable - used by the per-request BYO-key override below.
    """
    def _build(model: str) -> ChatGroq:
        kwargs = dict(model=model, temperature=temperature, max_retries=2)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if api_key:
            kwargs["groq_api_key"] = api_key
        return ChatGroq(**kwargs)

    emergency_kwargs = dict(model=_EMERGENCY_MODEL, temperature=temperature, max_retries=2)
    if api_key:
        emergency_kwargs["groq_api_key"] = api_key

    return _build(_PRIMARY_MODEL).with_fallbacks(
        [_build(_FALLBACK_MODEL), ChatGroq(**emergency_kwargs)],
        exceptions_to_handle=_RETRYABLE_EXCEPTIONS,
    )


def _create_llm_fast(temperature: float, api_key: str | None = None):
    """Creates a faster LLM for non-critical tasks."""
    # NOTE: never pass groq_api_key=None explicitly - that would override the
    # env-var default factory and silently drop the server's own key.
    fast_kwargs = dict(model=_FALLBACK_MODEL, temperature=temperature, max_retries=2,
                       reasoning_effort="low")
    if api_key:
        fast_kwargs["groq_api_key"] = api_key
    fast = ChatGroq(**fast_kwargs)
    fallback = ChatGroq(model=_EMERGENCY_MODEL, temperature=temperature, max_retries=2)
    return fast.with_fallbacks([fallback], exceptions_to_handle=_RETRYABLE_EXCEPTIONS)


# ---------------------------------------------------------------------------
# Per-request API-key override (browser-extension BYO-key support)
# ---------------------------------------------------------------------------
# The frontend can attach a client's own Groq key via the X-Groq-Api-Key
# header. The server passes it into the runners, which bind every llm_* chain
# to that key for the duration of the pipeline. Safe because the server
# serializes pipelines (server.app._RUN_LOCK): at most one pipeline streams at
# a time, so a single module-level "active key" holder has no cross-request
# races.

# User-key chains cached while a keyed pipeline runs (see _KeyScopedLLM).
_OVERRIDDEN_CHAINS: dict[str, tuple] = {}

_ACTIVE_GROQ_KEY: str | None = None


def active_groq_key() -> str | None:
    """The Groq key every llm_* call currently resolves to (None = env key)."""
    return _ACTIVE_GROQ_KEY


@contextmanager
def groq_key_override(api_key: str | None) -> Iterator[None]:
    """Binds all pre-configured LLM chains to ``api_key`` while active.

    Nested overrides restore the previously active key; the finally block
    guarantees restoration so a client disconnect mid-stream can never leave
    the holder pointing at a user key.
    """
    global _ACTIVE_GROQ_KEY
    previous = _ACTIVE_GROQ_KEY
    _ACTIVE_GROQ_KEY = api_key
    try:
        yield
    finally:
        _ACTIVE_GROQ_KEY = previous


def _build_scoped_chains(api_key: str | None) -> tuple:
    """Builds (strict, creative, fast) for one concrete Groq API key."""
    return (
        _create_llm(temperature=0, reasoning_effort="low", api_key=api_key),
        _create_llm(temperature=0.3, api_key=api_key),
        _create_llm_fast(temperature=0.3, api_key=api_key),
    )


class _KeyScopedLLM:
    """Attribute-delegating proxy resolving the LLM chain at call time.

    Singleton LLMs stay module-level (nodes already access them that way),
    but every attribute access (invoke / with_structured_output / ...) is
    delegated to the chain built for the currently active key.
    """

    __slots__ = ("_spec", "_default")

    def __init__(self, spec: int, default):
        self._spec = spec
        self._default = default

    def _current(self):
        key = _ACTIVE_GROQ_KEY
        if not key:
            return self._default
        chains = _OVERRIDDEN_CHAINS.get(key)
        if chains is None:
            # Replace (instead of growing) the cache: with the server's
            # one-pipeline-at-a-time serialization, only the active key's
            # chains can ever be in use.
            _OVERRIDDEN_CHAINS.clear()
            chains = _build_scoped_chains(key)
            _OVERRIDDEN_CHAINS[key] = chains
        return chains[self._spec]

    def __getattr__(self, name: str):
        return getattr(self._current(), name)


# --- LLM profiles (strict: scoring | creative: writing | fast: parallel docs) ---

# Three LLM profiles:
#   strict: scoring/analysis (primary, temp=0, low reasoning)
#   creative: resume writing (primary, temp=0.3, full reasoning)
#   fast: cover letter & interview prep (faster model, temp=0.3, low reasoning)
_DEFAULT_CHAINS = (
    _create_llm(temperature=0, reasoning_effort="low"),
    _create_llm(temperature=0.3),
    _create_llm_fast(temperature=0.3),
)
llm_strict = _KeyScopedLLM(0, _DEFAULT_CHAINS[0])
llm_creative = _KeyScopedLLM(1, _DEFAULT_CHAINS[1])
llm_fast = _KeyScopedLLM(2, _DEFAULT_CHAINS[2])

logger.info(f"[Models] Primary: {_PRIMARY_MODEL} | Fast: {_FALLBACK_MODEL} | Emergency: {_EMERGENCY_MODEL}")
