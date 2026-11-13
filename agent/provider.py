"""MR5 provider seam: the uniform structured-output ladder + usage accounting.

Everything here is provider-agnostic (it wraps whatever chat model the nodes
already resolve through ``agent.llms``), so LangSmith trees keep their shape:
each attempt is a normal structured-output invoke on the same chain, and the
ladder only ADDS attempts on the failure path.

- ``invoke_structured_with_repair`` — the Dossier study's schema-repair retry,
  made uniform for every structured call (reviewer + ATS semantic check).
  Attempt order per failure:
    1. direct invoke;
    2. extract rung (FREE, no LLM call) — when the failure carries the
       model's own prose (``failed_generation``) and that prose itself parses
       as JSON and validates against the schema, it IS the answer the model
       meant to package; return it directly (the Dossier study's
       ``extractJson`` pre-parse: whole text, fenced ```json blocks, then the
       first complete object via bracket matching);
    3. ONE attempt 2, in exactly one of three shapes:
       - SCHEMA_REFUSED — the endpoint refused the tool-calling CONTRACT
         itself ("Tool choice is required, but model did not call a tool",
         the live Groq flake; "tool calling is not supported", the local
         OpenAI-compatible class): a PROMPT-SCHEMA rung — a plain ``invoke``
         with NO tool binding and the JSON schema delivered in a code-owned
         HumanMessage (the Dossier study's ``withJsonSchema`` fallback);
         the answer is parsed against the schema in code. A repair retry
         through the tool layer cannot fix a contract the endpoint itself
         refused (this is what makes local Ollama/LM Studio endpoints work
         end-to-end);
       - the schema-repair retry — when the failure carries the model's own
         prose (``failed_generation``), attempt 2 shows the model its answer
         plus the required schema so it re-packages (this recovers the
         STRUCTURED result incl. ``sections``, which prose salvage loses);
       - without prose (transient class) it stays the pre-MR5 plain
         same-payload retry;
    4. prose salvage (caller-supplied) — the SCHEMA_REFUSED downgrade, free
       and deterministic; without one, raise. A prompt-schema answer that
       fails to parse rides to salvage in the ``failed_generation`` envelope
       (``_JsonSchemaAnswerFailed``), so the fresh answer still gets a real
       chance.
  Contract preserved from the pre-MR5 reviewer: at most TWO LLM calls, then
  give up (one attempt 2, never more; the extract rung costs zero);
  ``extract_failed_generation`` moved here from ``nodes/reviewer.py``
  (re-exported there for compatibility).
- ``TokenUsageTracker`` — per-task token accounting (a LangChain callback the
  runners attach to one pipeline segment). Thread-safe: the fan-out runs the
  cover-letter and interview-prep siblings concurrently.
"""

import json
import re
import threading

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

from .helpers import _safe_print


def extract_failed_generation(exc: BaseException) -> str | None:
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
        import re

        m = re.search(r"failed_generation[\"']?\s*:?\s*'(.+?)'\s*\}\}", str(exc) or "",
                      re.DOTALL)
        if m:
            raw = m.group(1)
            raw = re.sub(r"\\u([0-9a-fA-F]{4})", lambda x: chr(int(x.group(1), 16)), raw)
            raw = raw.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
            candidates.append(raw)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


# --- MR5b: the Dossier study's extractJson pre-parse (provider.ts) ----------
# The failed generation often IS the JSON the model meant to package - it
# just never reached the tool call (prose around it, a fenced block). Parsing
# it BEFORE spending the repair retry is free (zero LLM calls): a candidate
# that survives the schema is returned whole, so the structured result (incl.
# ``sections``, which prose salvage loses) is recovered one attempt earlier.

_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n?([\s\S]*?)```", re.IGNORECASE)


def _first_json_span(text: str) -> str | None:
    """The first complete {...} or [...] in ``text`` — bracket matching that
    skips string literals (a brace inside a value never closes the span) —
    or None when nothing closes."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return text[start:index + 1]
    return None


def _json_candidates(text: str) -> list[str]:
    """Parse candidates for a failed generation, best first: the whole text
    (a bare JSON answer), each fenced ```json block, then the first complete
    object/array in the prose."""
    trimmed = (text or "").strip()
    if not trimmed:
        return []
    candidates = [trimmed]
    candidates.extend(match.group(1).strip()
                      for match in _JSON_FENCE_RE.finditer(trimmed))
    span = _first_json_span(trimmed)
    if span is not None:
        candidates.append(span.strip())
    return candidates


def extract_structured_candidate(schema, text: str):
    """The first candidate that parses as JSON AND validates against
    ``schema`` (the ladder's schemas are pydantic models), else None."""
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return schema(**data)
        except Exception:
            continue
    return None


# --- MR5c: the SCHEMA_REFUSED flake class -> the prompt-schema rung ---------
# Some endpoints fold not on the answer's SHAPE but on the tool-calling
# CONTRACT itself: "Tool choice is required, but model did not call a tool"
# (the live Groq flake), "tool calling is not supported" and variants (local
# OpenAI-compatible servers via LLM_BASE_URL). No repair retry can fix those
# - the model never sees the schema through the tool layer. The marker list
# is deliberately tight: a match reroutes attempt 2 away from the tool
# binding, so a false positive would silently drop structured enforcement
# for that call.
_SCHEMA_REFUSED_MARKERS = (
    "tool choice is required",         # the live Groq flake (Nov 2026)
    "tool calling is not supported",   # conservative local-server class
    "tool use is not supported",
    "function calling is not supported",
)


def _is_schema_refused(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return any(marker in text for marker in _SCHEMA_REFUSED_MARKERS)


def _json_schema_messages(messages: list, schema) -> list:
    """The prompt-schema rung's messages: the original conversation plus ONE
    code-owned instruction carrying the JSON schema itself (the Dossier
    study's ``withJsonSchema`` fallback - the schema travels in the prompt,
    never through a tool binding)."""
    spec = json.dumps(schema.model_json_schema())
    return list(messages) + [HumanMessage(content=(
        "Respond with ONLY a JSON object that matches this JSON schema -"
        " no prose, no markdown fence:\n" + spec))]


def _message_text(answer) -> str:
    """Coerces a plain-invoke answer to text: chat models return a message
    whose ``content`` is a string (or provider-style content blocks)."""
    content = getattr(answer, "content", answer)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content)
    return str(content)


class _JsonSchemaAnswerFailed(Exception):
    """The prompt-schema rung's answer failed to parse/validate. The fresh
    answer rides the standard ``failed_generation`` envelope so caller
    salvage (which reads that envelope) gets a real chance on it."""

    def __init__(self, answer: str):
        super().__init__("json-schema answer failed to parse/validate: "
                         + (answer or "")[:160])
        self.body = {"error": {"failed_generation": answer or ""}}


def _repair_messages(messages: list, schema_name: str, answer: str) -> list:
    """The repair retry's messages: the original conversation plus the model's
    own failed answer (as its prior turn) and the fixed repair instruction.
    The model's prose rides as an AIMessage - it is the model's own output,
    never pasted into instructions - and the instruction text is code-owned."""
    return list(messages) + [
        AIMessage(content=answer),
        HumanMessage(content=(
            "Your previous response could not be parsed as the required"
            f" structured output ({schema_name} tool call). Respond again:"
            " call the tool EXACTLY once with arguments matching the"
            " required schema. Do not write prose outside the tool call.")),
    ]


def invoke_structured_with_repair(llm, schema, messages, *,
                                  salvage=None, label: str = ""):
    """Runs one structured-output call through the repair ladder.

    Returns ``(result, method)`` with ``method`` in ``{"direct", "extract",
    "repair", "retry", "json_schema", "salvage"}``; raises when every rung
    fails (caller decides the degradation - reviewer flags review_failed,
    the ATS semantic check keeps keywords as missing). ``salvage(exc)``
    maps the LAST failure to a result, or returns None.
    """
    schema_name = getattr(schema, "__name__", None) or str(schema)
    try:
        return llm.with_structured_output(schema).invoke(messages), "direct"
    except Exception as exc:
        _safe_print(f"   [Warning] [{label or 'provider'}] structured"
                    f" attempt 1/2 failed: {exc}")
        answer = extract_failed_generation(exc)
        # Attempt 1.5 - the extract rung (free, no LLM call): when the failed
        # generation itself parses as JSON and validates against the schema,
        # it IS the packaged answer; the repair retry is not spent on it.
        if answer:
            extracted = extract_structured_candidate(schema, answer)
            if extracted is not None:
                _safe_print(f"-> [{label or 'provider'}] recovered the"
                            " structured result from the failed generation"
                            " (extract; no repair call needed)")
                return extracted, "extract"
        # Attempt 2 - one LLM call, in exactly one of three shapes:
        #   * SCHEMA_REFUSED -> the prompt-schema rung (the endpoint refused
        #     the tool-calling CONTRACT itself; repairing through the tool
        #     layer cannot fix that, so the schema travels in the prompt);
        #   * the model's own prose is available -> the schema-repair retry
        #     (show it the answer + schema so it re-packages);
        #   * no prose (transient class) -> the pre-MR5 plain same-payload
        #     retry.
        exc2: Exception
        if _is_schema_refused(exc):
            try:
                fresh = _message_text(
                    llm.invoke(_json_schema_messages(messages, schema)))
                result = extract_structured_candidate(schema, fresh)
                if result is not None:
                    _safe_print(f"-> [{label or 'provider'}] recovered on"
                                " attempt 2/2 (prompt schema; no tool"
                                " binding)")
                    return result, "json_schema"
                # The fresh answer is unusable AS the structured result, but
                # it is still the model's best answer: carry it to salvage
                # in the standard failed_generation envelope.
                exc2 = _JsonSchemaAnswerFailed(fresh)
            except Exception as e2:
                exc2 = e2
        else:
            second = (_repair_messages(messages, schema_name, answer)
                      if answer else messages)
            mode = "repair" if answer else "retry"
            try:
                result = llm.with_structured_output(schema).invoke(second)
                _safe_print(f"-> [{label or 'provider'}] recovered on attempt 2/2"
                            f" ({'schema repair' if answer else 'retry'})")
                return result, mode
            except Exception as e2:
                exc2 = e2
        _safe_print(f"   [Warning] [{label or 'provider'}] structured"
                    f" attempt 2/2 failed: {exc2}")
        if salvage is not None:
            salvaged = salvage(exc2)
            if salvaged is not None:
                _safe_print(f"-> [{label or 'provider'}] salvaged a result"
                            " from the failed generation")
                return salvaged, "salvage"
        raise exc2


class TokenUsageTracker(BaseCallbackHandler):
    """Sums token usage for one pipeline segment, split per task.

    Reads the OpenAI-shaped ``token_usage`` from the response's llm_output
    when the provider supplies it, else langchain-core's ``usage_metadata``
    (input_tokens/output_tokens). One instance per runner attempt; the
    totals ride the SSE ``result`` payload and the runs ledger.

    Task attribution needs no node-code changes: a callback inside a
    LangGraph node sees ``metadata['langgraph_node']`` at START but
    ``metadata=None`` at END (probed on langgraph 1.2.16) - so the tracker
    captures the task into a run_id map at ``on_chat_model_start`` /
    ``on_llm_start`` and attributes at ``on_llm_end`` via the shared
    run_id. Calls with no captured start (fakes that replace whole chains)
    bucket as ``unknown``. Thread-safe: the fan-out runs the cover-letter
    and interview-prep siblings concurrently.
    """

    # LangGraph node ids use underscores; the prompt_versions stamps use
    # hyphens. task_usage keys match the stamp vocabulary so the forensic
    # loop closes (scanner/improver/reviewer/polisher already match).
    _TASK_ALIASES = {"ats_check": "ats-semantic",
                     "cover_letter": "cover-letter",
                     "interview_prep": "interview-prep"}

    def __init__(self):
        self._lock = threading.Lock()
        self._prompt = 0
        self._completion = 0
        self._calls = 0
        self._tasks: dict[str, dict[str, int]] = {}
        self._run_task: dict = {}

    # --- task capture (START) / attribution (END) --------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id=None,
                            parent_run_id=None, metadata=None, tags=None,
                            **kwargs) -> None:
        self._capture(run_id, metadata)

    def on_llm_start(self, serialized, prompts, *, run_id=None,
                     parent_run_id=None, metadata=None, tags=None,
                     **kwargs) -> None:
        self._capture(run_id, metadata)

    def _capture(self, run_id, metadata) -> None:
        if run_id is None:
            return
        node = (metadata.get("langgraph_node")
                if isinstance(metadata, dict) else None)
        name = str(node) if node else "unknown"
        with self._lock:
            self._run_task[run_id] = self._TASK_ALIASES.get(name, name)

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None,
                   **kwargs) -> None:
        usage = None
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage")
        if not usage:
            try:
                usage = response.generations[0][0].message.usage_metadata
            except (AttributeError, IndexError, TypeError):
                usage = None
        if not isinstance(usage, dict):
            if run_id is not None:
                with self._lock:
                    self._run_task.pop(run_id, None)
            return
        prompt = int(usage.get("prompt_tokens",
                               usage.get("input_tokens", 0)) or 0)
        completion = int(usage.get("completion_tokens",
                                   usage.get("output_tokens", 0)) or 0)
        with self._lock:
            self._prompt += prompt
            self._completion += completion
            self._calls += 1
            task = (self._run_task.pop(run_id, None)
                    if run_id is not None else None)
            bucket = self._tasks.setdefault(task or "unknown", {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "llm_calls": 0})
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["total_tokens"] += prompt + completion
            bucket["llm_calls"] += 1

    def on_llm_error(self, error, *, run_id=None, parent_run_id=None,
                     **kwargs) -> None:
        # A failed call produces no usage; drop its START-time mapping so
        # the map cannot leak across the segment.
        if run_id is None:
            return
        with self._lock:
            self._run_task.pop(run_id, None)

    def totals(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self._prompt,
                "completion_tokens": self._completion,
                "total_tokens": self._prompt + self._completion,
                "llm_calls": self._calls,
            }

    def task_totals(self) -> dict:
        with self._lock:
            return {task: dict(stats)
                    for task, stats in sorted(self._tasks.items())}
