"""Zero-dependency fake of the Groq REST API for over-the-wire server tests.

Serves the exact wire protocol langchain-groq speaks (the groq SDK posts to
``<base_url>/openai/v1/chat/completions`` and GETs ``<base_url>/openai/v1/models``),
so a pipeline can run through the REAL client stack - httpx -> groq SDK ->
ChatGroq -> LangGraph node - with actual HTTP requests on 127.0.0.1 instead of
the monkeypatched invoke() fakes of test_server.py. Only the model's
"intelligence" is faked; everything below the chain boundary is real.

Response branching mirrors the per-profile fakes of test_server.py, keyed on
what each node's prompt contains (text branches) or on the bound tool (the
structured-output branches, immune to prompt text):

| Node (profile)              | Wire key                       | Tool call / reply                       |
|-----------------------------|--------------------------------|-----------------------------------------|
| scanner (llm_strict)        | "MISSING SKILLS" in prompt     | plain: gap chips (Kubernetes/Terraform) |
| improver + polisher-full    | tool bound: "ImproverOutput"   | sections by id (whole-document unit     |
|                             |                                | when the prompt offers "[id=Resume]")   |
| polisher-target (llm_creat) | "SURGICAL REVISION" in prompt  | plain: one rewritten "## Skills"        |
| cover_letter  (llm_fast)    | "Cover Letter" in prompt       | plain: prose cover letter               |
| interview_prep (llm_fast)   | "Preparation Tips" in prompt   | plain: questions + tips markdown        |
| reviewer      (llm_strict)  | tool bound: "ReviewOutput"     | score 85 + feedback, sections=[]        |
| ats_check     (llm_fast)    | tool bound: "SemanticReview"   | verdicts=[] (no semantic rescues)       |

Key notes:
- Tool branches take priority: the reviewer/semantic/improver prompts contain job
  descriptions could accidentally match a text key, so the bound tool decides.
- "SURGICAL REVISION" is checked before the plain text keys: the polisher's
  full-rewrite fallback now rides the improver's STRUCTURED path (bound
  tool), so the plain "Rewrite the resume" branch only survives as a
  safety net for an unbound writer answer.
- Every accepted completion request is appended to ``requests`` in arrival
  order so tests can assert the exact call sequence/arity of a pipeline.
"""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FAKE_MODEL = "fake-groq/wire-model"

# Exported literals so tests can assert payload contents without retyping them.
FAKE_REVIEW_SCORE = 85
FAKE_REVIEW_FEEDBACK = "Add metrics to the bullets."
FAKE_MISSING_SKILLS = "- Kubernetes\n- Terraform"
FAKE_RESUME = "## Skills\nPython, Docker\n## Experience\nBuilt ML pipelines."
FAKE_SECTION = "## Skills\nPython, Docker, Kubernetes"
FAKE_COVER_LETTER = "Dear Hiring Team,\nI am excited to apply.\nSincerely, Candidate"
FAKE_INTERVIEW_PREP = "1. Sample question?\n\n## Preparation Tips\n- Tip one\n- Tip two"


def _prompt_text(body: dict) -> str:
    """Concatenates the string content of every message in the request."""
    parts: list[str] = []
    for message in body.get("messages") or []:
        content = (message or {}).get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts)


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{name.lower().replace(':', '')}{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _classify(body: dict) -> tuple[str | None, dict | None, str]:
    """Returns (plain_content, tool_call, request_key) for one completion.

    ``body`` is the parsed chat-completions payload (OpenAI-compatible).
    """
    tools = [((tool or {}).get("function") or {}).get("name", "")
             for tool in (body.get("tools") or [])]
    bound_tool = next((name for name in tools if name), "")

    # --- structured outputs: keyed on the bound tool, not the prompt text ---
    if bound_tool == "ReviewOutput":
        return None, _tool_call("ReviewOutput", {
            "score": FAKE_REVIEW_SCORE,
            "feedback": FAKE_REVIEW_FEEDBACK,
            "sections": [],
        }), "tool:ReviewOutput"
    if bound_tool == "SemanticReview":
        return None, _tool_call("SemanticReview", {"verdicts": []}), "tool:SemanticReview"
    if bound_tool == "ImproverOutput":
        # BB1 Phase B: the writer returns section replacements by id.
        # Whole-document mode (the prompt offers "[id=\"Resume\"]" for a
        # heading-less source) -> one Resume replacement whose body is the
        # full FAKE_RESUME, so the assembled output is byte-identical to
        # the plain-answer era. By-section mode (the polisher fallback
        # rewriting a marked draft) -> replacements for its sections.
        prompt = _prompt_text(body)
        if "DROPPED CONTEXT TERMS" in prompt:
            # Round 11 context weave-in: the fake answers with an EMPTY
            # sections list, so the deterministic acceptance never adopts
            # on the wire. Keyed separately ("context_weave", not
            # "tool:ImproverOutput") so wire call sequences stay
            # deterministic and inspectable.
            return None, _tool_call("ImproverOutput",
                                    {"sections": [], "declined": []}), \
                "context_weave"
        if '[id="Resume"]' in prompt:
            arguments = {"sections": [{"id": "Resume", "body": FAKE_RESUME}],
                         "declined": []}
        else:
            arguments = {"sections": [
                {"id": "Skills", "body": "Python, Docker"},
                {"id": "Experience", "body": "Built ML pipelines."},
            ], "declined": []}
        return None, _tool_call("ImproverOutput", arguments), "tool:ImproverOutput"
    if bound_tool:
        # Unknown bound tool: keep the structured flow loud and visible.
        return None, _tool_call(bound_tool, {}), f"tool:{bound_tool}"

    # --- plain-text branches, most specific key first ---
    prompt = _prompt_text(body)
    if "SURGICAL REVISION" in prompt:        # polisher targeted rewrites
        return FAKE_SECTION, None, "polish_target"
    if "Rewrite the resume" in prompt:       # unbound writer answer (safety net)
        return FAKE_RESUME, None, "rewrite"
    if "MISSING SKILLS" in prompt:           # scanner
        return FAKE_MISSING_SKILLS, None, "missing_skills"
    if "Cover Letter" in prompt:             # cover letter
        return FAKE_COVER_LETTER, None, "cover"
    if "Preparation Tips" in prompt:         # interview prep
        return FAKE_INTERVIEW_PREP, None, "prep"
    return "OK.", None, "default"


def _completion_payload(body: dict, content: str | None, tool_call: dict | None,
                        model: str) -> dict:
    """OpenAI-compatible chat.completion response with stable minimal fields."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content is not None else "",
    }
    finish_reason = "stop"
    if tool_call:
        message["tool_calls"] = [tool_call]
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or model,
        "system_fingerprint": "fp_fake_groq",
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "queue_time": 0.0,
            "prompt_time": 0.0,
            "prompt_tokens": 1,
            "completion_time": 0.0,
            "completion_tokens": 1,
            "total_tokens": 2,
            "total_time": 0.0,
        },
    }


def _error_payload(message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": "404"}}


class _FakeGroqHandler(BaseHTTPRequestHandler):
    """Serves GET /models and POST /chat/completions, 404s anything else."""

    protocol_version = "HTTP/1.1"  # keep-alive: httpx pools connections

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return  # silence per-request stderr noise under pytest

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.endswith("/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.server.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fake-groq",
                    "active": True,
                    "context_window": 131072,
                    "max_completion_tokens": 8192,
                    "public_apps": None,
                }],
            })
        else:
            self._send_json(404, _error_payload(f"unknown path: {path}"))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send_json(400, _error_payload(f"bad request body: {exc}"))
            return
        path = self.path.split("?", 1)[0]
        if not path.endswith("chat/completions"):
            self._send_json(404, _error_payload(f"unknown path: {path}"))
            return
        content, tool_call, key = _classify(body)
        self.server.requests.append({
            "path": path,
            "model": body.get("model") or self.server.model,
            "tool": (((tool_call or {}).get("function")) or {}).get("name"),
            "key": key,
        })
        self._send_json(200, _completion_payload(body, content, tool_call,
                                                 self.server.model))

    def _send_json(self, status: int, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class FakeGroqServer:
    """Context-managed ThreadingHTTPServer on an ephemeral 127.0.0.1 port.

    - ``base_url``: hand straight to ChatGroq(base_url=...).
    - ``requests``: every POST /chat/completions in arrival order as
      ``{"path", "model", "tool", "key"}`` (thread-safe via list.append).
    - ``request_keys()``: just the branch keys, for call-arity assertions.
    """

    def __init__(self, model: str = FAKE_MODEL):
        self.model = model
        self.requests: list[dict] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request_keys(self) -> list[str]:
        return [entry["key"] for entry in self.requests]

    def start(self) -> "FakeGroqServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGroqHandler)
        self._httpd.daemon_threads = True
        # The handler reaches shared state via handler.server (the HTTPd).
        self._httpd.requests = self.requests
        self._httpd.model = self.model
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="fake-groq-server", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "FakeGroqServer":
        return self.start()

    def __exit__(self, *args):
        self.stop()
