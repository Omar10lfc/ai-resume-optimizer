"""LLM factory functions and pre-configured instances."""
import logging
import os
from contextlib import contextmanager
from typing import Iterator

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from .config import (_PRIMARY_MODEL, _FALLBACK_MODEL, _EMERGENCY_MODEL,
                     LLM_BASE_URL, LLM_TIMEOUT_SECONDS)

logger = logging.getLogger(__name__)


def _chat_kwargs(model: str, temperature: float, *,
                 reasoning_effort: str | None = None,
                 api_key: str | None = None) -> dict:
    """MR5: the single construction point for every chat model.

    - ``request_timeout`` caps one call (the SDK default is 10 minutes - a
      hung call stalled the serialized pipeline for everyone).
    - ``base_url`` rides the MR4-proven OpenAI-compatible seam: any endpoint
      speaking the OpenAI wire protocol (Ollama, LM Studio, OpenRouter, the
      MR4 fake server) works behind the same ChatGroq client.
    - ``reasoning_effort`` is Groq-specific - omitted for custom endpoints,
      which may reject unknown params.
    - ``api_key`` (optional) pins a concrete key instead of the
      GROQ_API_KEY environment variable - used by the per-request BYO-key
      override below. NOTE: never pass ``api_key=None`` explicitly into the
      ChatGroq kwargs - that would override the env-var default factory and
      silently drop the server's own key (only ADDED when truthy).
    """
    kwargs = dict(model=model, temperature=temperature, max_retries=2,
                  request_timeout=LLM_TIMEOUT_SECONDS)
    if reasoning_effort and not LLM_BASE_URL:
        kwargs["reasoning_effort"] = reasoning_effort
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    if api_key:
        kwargs["groq_api_key"] = api_key
    return kwargs


def _catalog_gap_warning(ids: list[str]) -> str | None:
    """Soft warning listing every configured tier absent from the live catalog.

    All three tiers are validated, not just the primary: the emergency tier is
    only reached after two upstream failures, so a decommissioned emergency
    model used to surface as a mid-run 404 instead of this actionable startup
    message (Nov 10 2026 incident: 'llama-3.1-8b-instant' was pulled from the
    catalog while still the EMERGENCY default).
    """
    if not ids:
        return None
    missing = [
        f"{label} '{model}' (set {env_name})"
        for label, env_name, model in (
            ("primary", "PRIMARY_MODEL", _PRIMARY_MODEL),
            ("fallback", "FALLBACK_MODEL", _FALLBACK_MODEL),
            ("emergency", "EMERGENCY_MODEL", _EMERGENCY_MODEL),
        )
        if model not in ids
    ]
    if not missing:
        return None
    return (f"API reachable ({len(ids)} models), but missing from this account's "
            f"catalog: {', '.join(missing)}. Update the *_MODEL overrides in .env "
            f"to models the account can access.")


def _models_endpoint() -> tuple[str, bool]:
    """MR5: (models URL, is_groq_official).

    Custom ``LLM_BASE_URL`` endpoints get ``<base>/models`` and skip the
    Groq-catalog tier-gap warning (tier names are Groq-catalog-specific).
    """
    if not LLM_BASE_URL:
        return "https://api.groq.com/openai/v1/models", True
    return LLM_BASE_URL.rstrip("/") + "/models", False


def _groq_models_status(key: str, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """One cheap REST call (GET <provider>/models) - no generation tokens are
    spent. Works against Groq's official API or any configured OpenAI-compatible
    ``LLM_BASE_URL`` (no auth header sent when the key is empty - local Ollama).

    Returns (ok, message). Auth failures and missing keys are hard problems;
    network failures and odd statuses are soft warnings.
    """
    import json
    import os
    import urllib.error
    import urllib.request

    url, is_groq = _models_endpoint()
    key_label = "GROQ_API_KEY" if is_groq else "API key"
    provider_label = "the Groq API" if is_groq else "the provider"

    def _finish(ok: bool, ids: list[int | str] | None, status: int | None,
                err: str | None) -> tuple[bool, str]:
        if ids is not None:
            if is_groq:
                gap = _catalog_gap_warning(ids)
                if gap:
                    return True, gap
            return True, f"API reachable ({len(ids)} models)."
        if err is not None:
            return False, f"Could not reach {provider_label} ({err}) - check connectivity. Nothing was spent (models list call, zero generation tokens)."
        if status in (401, 403):
            return False, f"{key_label} rejected by the API (HTTP {status}) - check the key in .env."
        if status == 429:
            return True, "API rate-limited (HTTP 429) - the key is valid; free-tier quota resets shortly."
        return True, f"API reachable but returned HTTP {status} - key looks valid."

    # If urllib.request.urlopen has been monkeypatched (e.g. by unit tests), use the mock
    if getattr(urllib.request.urlopen, "__module__", "") != "urllib.request":
        headers = {"User-Agent": "JobApplicationOptimizer/1.0"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status == 200:
                    return _finish(True, [m.get("id", "") for m in json.load(response).get("data", [])], None, None)
                return _finish(True, None, response.status, None)
        except urllib.error.HTTPError as e:
            return _finish(False, None, e.code, None)
        except Exception as e:
            return _finish(False, None, None, str(e))

    # Production path: httpx handles modern TLS/ALPN and avoids Cloudflare 1010 signature blocks
    try:
        import httpx
        headers = {"User-Agent": "JobApplicationOptimizer/1.0"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                ids = [m.get("id", "") for m in resp.json().get("data", [])]
                return _finish(True, ids, None, None)
            return _finish(False, None, resp.status_code, None)
    except Exception as e:
        return _finish(False, None, None, str(e))


def check_provider_health(timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """Startup health check for the LLM provider.

    Fails quickly with an actionable message instead of letting the user hit a
    confusing error deep in Step 2. Returns (ok, message); never raises.
    With a custom ``LLM_BASE_URL`` the key is optional (a local Ollama/LM
    Studio server needs none), so the missing-key guidance only applies to
    Groq's official API.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not LLM_BASE_URL and (not api_key or "your_groq_key" in api_key.lower()):
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
    emergency = ChatGroq(**_chat_kwargs(_EMERGENCY_MODEL, temperature,
                                        api_key=api_key))
    return ChatGroq(**_chat_kwargs(_PRIMARY_MODEL, temperature,
                                   reasoning_effort=reasoning_effort,
                                   api_key=api_key)).with_fallbacks(
        [ChatGroq(**_chat_kwargs(_FALLBACK_MODEL, temperature,
                                 api_key=api_key)),
         emergency],
        exceptions_to_handle=_RETRYABLE_EXCEPTIONS,
    )


def _create_llm_fast(temperature: float, api_key: str | None = None):
    """Creates a faster LLM for non-critical tasks."""
    # Same BYO-key passthrough as _create_llm's tiers: without it a pinned
    # client key would silently launder fallback spend onto the server key.
    return ChatGroq(**_chat_kwargs(_FALLBACK_MODEL, temperature,
                                   reasoning_effort="low",
                                   api_key=api_key)).with_fallbacks(
        [ChatGroq(**_chat_kwargs(_EMERGENCY_MODEL, temperature,
                                 api_key=api_key))],
        exceptions_to_handle=_RETRYABLE_EXCEPTIONS,
    )


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
