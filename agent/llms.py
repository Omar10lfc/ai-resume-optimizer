"""LLM factory functions and pre-configured instances.

Round 15 (roadmap item 13): the factory dispatches on ``LLM_PROVIDER``
(groq | openai | anthropic | openai-compatible) through
``init_chat_model`` - the 3-profile structure (strict/creative/fast) and
the three-tier fallback chain (primary -> fallback -> emergency, all
within ONE provider) are unchanged. ``GROQ_API_KEY`` stays the universal
API-key slot (historical name); ``reasoning_effort`` is Groq-GPT-OSS
vocabulary and is only sent to the official Groq API.
"""
import logging
import os
from contextlib import contextmanager
from typing import Iterator

# pyrefly: ignore [missing-import]
from langchain.chat_models import init_chat_model

from .config import (_EMERGENCY_MODEL, _FALLBACK_MODEL, _PRIMARY_MODEL,
                     LLM_BASE_URL, LLM_MAX_TOKENS, LLM_PROVIDER,
                     LLM_TIMEOUT_SECONDS, POLISHER_REASONING_EFFORT)

logger = logging.getLogger(__name__)

# --- Provider construction table -------------------------------------------
# model_provider is the string init_chat_model dispatches on. The
# OpenAI-compatible seam (local Ollama / LM Studio / vLLM) deliberately rides
# the ChatGroq CLASS: it speaks the OpenAI wire protocol and has carried the
# MR4 fake-server seam + every custom LLM_BASE_URL since MR5.
_PROVIDER_MODEL_IDS = {"groq": "groq", "openai": "openai",
                       "anthropic": "anthropic", "openai-compatible": "groq"}
# Per-class API-key field names (all truthy-added only - never pass None,
# which would override the env-var default factory).
_PROVIDER_KEY_FIELDS = {"groq": "groq_api_key", "openai": "openai_api_key",
                        "anthropic": "anthropic_api_key",
                        "openai-compatible": "groq_api_key"}
# Per-class request-timeout field names (ChatAnthropic rejects
# request_timeout - its field is default_request_timeout).
_PROVIDER_TIMEOUT_FIELDS = {"groq": "request_timeout",
                            "openai": "request_timeout",
                            "anthropic": "default_request_timeout",
                            "openai-compatible": "request_timeout"}
# pip package each provider needs (import name = pip name with dashes
# replaced; surfaced verbatim in the actionable startup/health errors).
_PROVIDER_PACKAGES = {"groq": "langchain-groq", "openai": "langchain-openai",
                      "anthropic": "langchain-anthropic",
                      "openai-compatible": "langchain-groq"}


def _provider_kwargs(provider: str, model: str, temperature: float, *,
                     reasoning_effort: str | None = None,
                     api_key: str | None = None) -> dict:
    """Round 15: the single construction-point kwargs builder, per provider.

    - The provider's request-timeout field caps one call (the SDK default is
      10 minutes - a hung call stalled the serialized pipeline for everyone).
    - ``base_url`` rides the MR4-proven OpenAI-compatible seam for every
      provider (anthropic-compatible endpoints included).
    - ``reasoning_effort`` is Groq-GPT-OSS vocabulary: sent ONLY to the
      official Groq API. Custom OpenAI-wire endpoints may reject unknown
      params; openai/anthropic gate it per model, so it is never sent there.
    - ``api_key`` (optional) pins a concrete key instead of the
      GROQ_API_KEY environment variable - used by the per-request BYO-key
      override below, written to the provider's own field name. NOTE: never
      pass the key field with a None value - that would override the
      env-var default factory and silently drop the server's own key
      (only ADDED when truthy).
    - Anthropic requires an explicit ``max_tokens`` (langchain-anthropic
      leaves it None and the API then truncates at a small default).
    """
    kwargs = dict(model=model, temperature=temperature, max_retries=2)
    kwargs[_PROVIDER_TIMEOUT_FIELDS[provider]] = LLM_TIMEOUT_SECONDS
    if reasoning_effort and provider == "groq" and not LLM_BASE_URL:
        kwargs["reasoning_effort"] = reasoning_effort
    if provider == "anthropic":
        kwargs["max_tokens"] = LLM_MAX_TOKENS
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    if api_key:
        kwargs[_PROVIDER_KEY_FIELDS[provider]] = api_key
    return kwargs


def _build_chat_model(provider: str, model: str, temperature: float, *,
                      reasoning_effort: str | None = None,
                      api_key: str | None = None):
    """One chat model of ``provider`` via ``init_chat_model`` - a missing
    provider package surfaces as an actionable runtime error (naming the
    pip install) instead of a bare ImportError mid-pipeline."""
    try:
        kwargs = _provider_kwargs(provider, model, temperature,
                                  reasoning_effort=reasoning_effort,
                                  api_key=api_key)
        # init_chat_model takes the model positionally; the kwargs builder
        # keeps it for a complete description of one model (asserted in the
        # construction tests), so it is popped here, not dropped there.
        model_name = kwargs.pop("model")
        return init_chat_model(model_name, model_provider=_PROVIDER_MODEL_IDS[provider],
                               **kwargs)
    except ImportError as exc:
        package = _PROVIDER_PACKAGES[provider]
        raise RuntimeError(
            f"LLM_PROVIDER={provider} needs the {package} package - "
            f"pip install {package} ({exc})") from exc


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


def _models_endpoint(provider: str | None = None) -> tuple[str, bool]:
    """Round 15: (models URL, is_groq_catalog) per provider.

    A custom ``LLM_BASE_URL`` wins for every provider and skips the
    Groq-catalog tier-gap warning (tier names are Groq-catalog-specific);
    the official endpoints default to each vendor's models list. The
    Groq-catalog flag stays True ONLY for the official Groq API.
    """
    provider = provider or LLM_PROVIDER
    if LLM_BASE_URL:
        return LLM_BASE_URL.rstrip("/") + "/models", False
    if provider == "openai":
        return "https://api.openai.com/v1/models", False
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/models", False
    return "https://api.groq.com/openai/v1/models", True


def _models_headers(provider: str, key: str) -> dict[str, str]:
    """Auth headers per provider: Bearer for the OpenAI-wire families,
    ``x-api-key`` + the required ``anthropic-version`` for Anthropic."""
    if provider == "anthropic":
        headers = {"User-Agent": "JobApplicationOptimizer/1.0",
                   "anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        return headers
    headers = {"User-Agent": "JobApplicationOptimizer/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _groq_models_status(key: str, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """One cheap REST call (GET <provider>/models) - no generation tokens are
    spent. Provider-aware since round 15 (the name stays for the tests and
    the app.py import): Groq / OpenAI / Anthropic official APIs or any
    configured ``LLM_BASE_URL`` (no auth header sent when the key is empty -
    local Ollama).

    Returns (ok, message). Auth failures and missing keys are hard problems;
    network failures and odd statuses are soft warnings.
    """
    import json
    import urllib.error
    import urllib.request

    provider = LLM_PROVIDER
    url, is_groq = _models_endpoint(provider)
    key_label = "GROQ_API_KEY"  # the universal API-key slot (historical name)
    provider_label = {"groq": "the Groq API", "openai": "the OpenAI API",
                      "anthropic": "the Anthropic API",
                      "openai-compatible": "the provider"}.get(
        provider, "the provider")

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

    headers = _models_headers(provider, key)

    # If urllib.request.urlopen has been monkeypatched (e.g. by unit tests), use the mock
    if getattr(urllib.request.urlopen, "__module__", "") != "urllib.request":
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
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                ids = [m.get("id", "") for m in resp.json().get("data", [])]
                return _finish(True, ids, None, None)
            return _finish(False, None, resp.status_code, None)
    except Exception as e:
        return _finish(False, None, None, str(e))


def _provider_package_ok(provider: str) -> tuple[bool, str]:
    """Round 15: is the provider's langchain package importable? Surfaced by
    the health check BEFORE the first run so a hand-edited provider fails
    with the pip install command, not a mid-pipeline crash (the import-time
    chain build already fails loudly for a broken config)."""
    import importlib
    try:
        importlib.import_module(_PROVIDER_PACKAGES[provider].replace("-", "_"))
        return True, ""
    except ImportError as exc:
        package = _PROVIDER_PACKAGES[provider]
        return False, (f"LLM_PROVIDER={provider} needs the {package} package - "
                       f"pip install {package} ({exc}).")


def check_provider_health(timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """Startup health check for the LLM provider.

    Fails quickly with an actionable message instead of letting the user hit a
    confusing error deep in Step 2. Returns (ok, message); never raises.
    ``GROQ_API_KEY`` is the UNIVERSAL API-key slot (historical name) - its
    guidance names the right console per provider. A custom
    ``LLM_BASE_URL`` (and the keyless local Ollama posture) skips the
    missing-key guidance entirely.
    """
    provider = LLM_PROVIDER
    api_key = os.environ.get("GROQ_API_KEY", "")
    if provider == "openai-compatible" and not LLM_BASE_URL:
        return False, ("LLM_PROVIDER=openai-compatible needs LLM_BASE_URL - "
                       "point it at an OpenAI-wire endpoint, e.g. "
                       "http://localhost:11434/v1 for a local Ollama.")
    if LLM_BASE_URL or provider == "openai-compatible":
        pass  # custom endpoint / local server: the key is optional
    elif not api_key or "your_groq_key" in api_key.lower():
        hint = {
            "groq": "get a free key: https://console.groq.com/keys",
            "openai": "get a key: https://platform.openai.com/api-keys",
            "anthropic": "get a key: https://console.anthropic.com",
        }.get(provider, "set it to your provider's key")
        return False, (f"GROQ_API_KEY is not set (the universal API-key slot - "
                       f"historical name). {hint}.")
    package_ok, package_message = _provider_package_ok(provider)
    if not package_ok:
        return False, package_message
    return _groq_models_status(api_key, timeout_seconds)


try:
    from groq import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError
    _RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
except ImportError:
    _RETRYABLE_EXCEPTIONS = (Exception,)


def _retryable_exceptions(provider: str) -> tuple:
    """Round 15: per-provider retryable SDK errors for ``with_fallbacks`` -
    the same four classes (rate limit / internal / connection / timeout)
    from each vendor's SDK. Unknown providers (package missing) inherit the
    Groq table rather than the catch-all default, so a KeyError-style bug
    can never be silently retried as a transient failure."""
    if provider == "openai":
        try:
            from openai import (APIConnectionError, APITimeoutError,
                                InternalServerError, RateLimitError)
            return (RateLimitError, InternalServerError,
                    APIConnectionError, APITimeoutError)
        except ImportError:
            return _RETRYABLE_EXCEPTIONS
    if provider == "anthropic":
        try:
            from anthropic import (APIConnectionError, APITimeoutError,
                                   InternalServerError, RateLimitError)
            return (RateLimitError, InternalServerError,
                    APIConnectionError, APITimeoutError)
        except ImportError:
            return _RETRYABLE_EXCEPTIONS
    return _RETRYABLE_EXCEPTIONS


def _create_llm(temperature: float, reasoning_effort: str | None = None,
                api_key: str | None = None):
    """Creates an LLM with automatic fallback for transient provider failures.

    All three tiers stay WITHIN the configured provider (round 15): a
    degraded run falls to a weaker model of the same vendor, never a silent
    cross-vendor hop.
    api_key (optional) pins a concrete key instead of the GROQ_API_KEY
    environment variable - used by the per-request BYO-key override below.
    """
    provider = LLM_PROVIDER
    retryable = _retryable_exceptions(provider)
    emergency = _build_chat_model(provider, _EMERGENCY_MODEL, temperature,
                                  api_key=api_key)
    return _build_chat_model(provider, _PRIMARY_MODEL, temperature,
                             reasoning_effort=reasoning_effort,
                             api_key=api_key).with_fallbacks(
        [_build_chat_model(provider, _FALLBACK_MODEL, temperature,
                           api_key=api_key),
         emergency],
        exceptions_to_handle=retryable,
    )


def _create_llm_fast(temperature: float, api_key: str | None = None):
    """Creates a faster LLM for non-critical tasks."""
    # Same BYO-key passthrough as _create_llm's tiers: without it a pinned
    # client key would silently launder fallback spend onto the server key.
    provider = LLM_PROVIDER
    return _build_chat_model(provider, _FALLBACK_MODEL, temperature,
                             reasoning_effort="low",
                             api_key=api_key).with_fallbacks(
        [_build_chat_model(provider, _EMERGENCY_MODEL, temperature,
                           api_key=api_key)],
        exceptions_to_handle=_retryable_exceptions(provider),
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

# Three primary LLM profiles (+ lazy reasoning-tuned polisher proxy):
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

_POLISHER_DEFAULT_BY_EFFORT: dict[str, object] = {}
_POLISHER_KEY_CACHE: dict[tuple[str, str], object] = {}


def polisher_model():
    """Returns a POLISHER_REASONING_EFFORT-configured chain in production,
    or monkeypatched llm_creative in unit tests."""
    if not isinstance(llm_creative, _KeyScopedLLM):
        return llm_creative
    effort = (POLISHER_REASONING_EFFORT or "").strip().lower()
    if not effort:
        return llm_creative
    key = _ACTIVE_GROQ_KEY
    if key:
        ck = (key, effort)
        chain = _POLISHER_KEY_CACHE.get(ck)
        if chain is None:
            _POLISHER_KEY_CACHE.clear()
            chain = _create_llm(temperature=0.3, reasoning_effort=effort, api_key=key)
            _POLISHER_KEY_CACHE[ck] = chain
        return chain
    chain = _POLISHER_DEFAULT_BY_EFFORT.get(effort)
    if chain is None:
        chain = _create_llm(temperature=0.3, reasoning_effort=effort)
        _POLISHER_DEFAULT_BY_EFFORT[effort] = chain
    return chain

logger.info(f"[Models] Primary: {_PRIMARY_MODEL} | Fast: {_FALLBACK_MODEL} | Emergency: {_EMERGENCY_MODEL}")
