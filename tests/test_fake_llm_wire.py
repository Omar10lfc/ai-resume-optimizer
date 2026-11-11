"""MR4: over-the-wire end-to-end tests for BOTH pipeline steps.

Unlike test_server.py (whose fakes monkeypatch the pre-configured chains'
invoke()), these tests run the REAL graphs with REAL ChatGroq chains whose
``base_url`` points at the local fake Groq REST server (tests/fake_groq_server.py):
every LLM call leaves the process as an actual HTTP request the groq SDK
builds, so the whole httpx -> groq SDK -> ChatGroq wire is exercised and only
the model intelligence is faked.

Asserted contracts (things the fake-invoke layer cannot see):
- LLM wire call arity/onset: the exact per-pipeline call sequences recorded by
  the fake server (incl. tool-call payloads for the structured outputs).
- SSE event kinds/arity/order of both steps over the same wire events.
- Artifact emission + download links for a wire-driven run.
- The merged prompt-version stamp payload (QW1) a real LangGraph run produces.
- Prompt/payload camouflage: BYO key NOTE - no chaining monkeypatch is needed;
  new ChatGroq instances are only swapped in place of the module-level chains
  (same seam the existing fake_llms fixture uses).

Run with: venv\\Scripts\\python.exe -m pytest tests/test_fake_llm_wire.py -v
"""

import json
import os
import sys
import threading
import uuid
from pathlib import Path

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

import agent  # noqa: E402
import agent.config  # noqa: E402
import agent.llms  # noqa: E402
import server.app as server_app  # noqa: E402
import server.runners as runners  # noqa: E402
from tests.fake_groq_server import (  # noqa: E402
    FAKE_COVER_LETTER, FAKE_INTERVIEW_PREP, FAKE_MISSING_SKILLS, FAKE_MODEL,
    FAKE_REVIEW_FEEDBACK, FAKE_REVIEW_SCORE, FAKE_RESUME, FakeGroqServer,
)

JD = "Python engineer with Docker and Kubernetes experience."
RESUME_TEXT = "Jane Roe\nPython developer with cloud exposure."

# One entry per LLM-writing node; all seven stamps must reach the result
# payload merged (QW1's snapshot-pull handles the concurrent cover/prep fan-out).
EXPECTED_PROMPT_VERSIONS = {
    "scanner": "scanner@1",
    "improver": "improver@2",
    "polisher": "polisher-full@1",
    "reviewer": "reviewer@2",
    "ats-semantic": "ats-semantic@1",
    "cover-letter": "cover-letter@1",
    "interview-prep": "interview-prep@1",
}

# Deterministic first segment of a FRESH step-2 run (no session): the gate sees
# a passing score WITH feedback (polisher full-rewrite, sections=[]), so both
# ATS passes fit exactly two (reviewer -> semantic check) rounds before the
# iteration cap fans out. Only the fan-out tail's ORDER is nondeterministic
# (cover_letter vs interview_prep run concurrently).
FRESH_STEP2_WIRES = [
    "missing_skills",       # scanner
    "rewrite",              # improver
    "tool:ReviewOutput",    # reviewer pass 1
    "tool:SemanticReview",  # ats_check pass 1 (paragraph = iteration cap -> polisher)
    "rewrite",              # polisher full-rewrite fallback
    "tool:ReviewOutput",    # reviewer pass 2
    "tool:SemanticReview",  # ats_check pass 2 (iteration cap -> fan-out)
]


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture(autouse=True)
def runs_db(tmp_path, monkeypatch):
    """Hermetic runs ledger (MR3 records both steps' lifecycles)."""
    monkeypatch.setattr(agent.config, "RUNS_DB_PATH",
                        str(tmp_path / "runs.sqlite3"))


@pytest.fixture
def fake_groq():
    with FakeGroqServer() as server:
        yield server


@pytest.fixture
def wire_llms(fake_groq, monkeypatch):
    """REAL ChatGroq chains rebuilt against the fake server (base_url seam).

    Same injection point as test_server's fake_llms fixture; no pipeline code
    must know whether a chain is real or fake.
    """

    def _chain(temperature: float) -> ChatGroq:
        return ChatGroq(model=FAKE_MODEL, temperature=temperature,
                        groq_api_key="gsk_fake_wire_key", max_retries=0,
                        base_url=fake_groq.base_url)

    monkeypatch.setattr(agent.llms, "llm_strict", _chain(0))
    monkeypatch.setattr(agent.llms, "llm_creative", _chain(0.3))
    monkeypatch.setattr(agent.llms, "llm_fast", _chain(0.3))
    return fake_groq


@pytest.fixture
def client():
    # Without a context manager, lifespan does NOT run: no startup health check
    # and no cleanup daemon - same semantics as the fake-invoke tests.
    return TestClient(server_app.app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parses the text/event-stream protocol into (event, payload) pairs."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip("\r\n")
        if not block:
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event and data is not None:
            events.append((event, data))
    return events


def _statuses(events) -> list[str]:
    return [e.get("message", "") for kind, e in events if kind == "status"]


def _spy_feedback(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(runners, "log_pipeline_feedback",
                        lambda state, run_id: calls.append((state, run_id)))
    return calls


# ==========================================
# Fake server mechanics (no pipeline)
# ==========================================

def test_fake_groq_server_serves_models_and_404s_unknown_paths(fake_groq):
    import urllib.error
    import urllib.request

    with urllib.request.urlopen(f"{fake_groq.base_url}/openai/v1/models") as resp:
        data = json.load(resp)
    assert data["data"][0]["id"] == FAKE_MODEL

    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{fake_groq.base_url}/openai/v1/nope")
    assert err.value.code == 404


def test_fake_groq_server_handles_concurrent_completion_requests(fake_groq):
    """The fan-out siblings race: parallel completions must never cross branches
    (each response is built from its own request body, never global request
    state). Workers only collect; the assertions run in the test thread."""
    import urllib.request

    def _post(prompt: str) -> str:
        req = urllib.request.Request(
            f"{fake_groq.base_url}/openai/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": FAKE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }).encode())
        with urllib.request.urlopen(req) as resp:
            body = json.load(resp)
        assert body["choices"][0]["finish_reason"] == "stop"
        return body["choices"][0]["message"]["content"]

    results: dict[str, str] = {}

    def _worker(name: str, prompt: str):
        results[name] = _post(prompt)

    threads = [
        threading.Thread(target=_worker, args=("cover", "Write a Cover Letter")),
        threading.Thread(target=_worker, args=("prep", "## Preparation Tips asked here")),
        threading.Thread(target=_worker, args=("scan", "Identify MISSING SKILLS")),
        threading.Thread(target=_worker, args=("noop", "nothing special")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results["cover"] == FAKE_COVER_LETTER
    assert results["prep"] == FAKE_INTERVIEW_PREP
    assert results["scan"] == FAKE_MISSING_SKILLS
    assert results["noop"] == "OK."
    assert fake_groq.request_keys() == ["cover", "prep", "missing_skills", "default"]


# ==========================================
# Step 1 over the wire
# ==========================================

def test_step1_over_the_wire_returns_gaps_after_single_llm_call(client, wire_llms):
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": RESUME_TEXT})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done"
    assert "error" not in kinds

    done = events[-1][1]
    uuid.UUID(done["session"])  # well-formed uuid
    assert done["missing_skills"] == "- Kubernetes\n- Terraform"
    assert "Analysis complete" in done["status"]
    assert any("(step 1 of 2)" in s for s in _statuses(events))
    assert any("Loading documents..." in s for s in _statuses(events))

    # Wire contract: the scanner was the ONLY LLM call, and it was the fake
    # model id echoed back (an HTTP request actually crossed the stack).
    assert [e["model"] for e in wire_llms.requests] == [FAKE_MODEL]
    assert wire_llms.request_keys() == ["missing_skills"]
    assert [entry["path"] for entry in wire_llms.requests] == \
        ["/openai/v1/chat/completions"]


# ==========================================
# Step 2 over the wire
# ==========================================

def test_step2_over_the_wire_end_to_end(client, wire_llms, monkeypatch):
    feedback_calls = _spy_feedback(monkeypatch)
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT, "notes": ""})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result"
    assert "error" not in kinds
    warnings = [e for kind, e in events if kind == "warning"]
    assert warnings and "auto-detected" in warnings[0]["message"]

    result = events[-1][1]
    uuid.UUID(result["session"])
    assert result["optimized_resume"].strip() == FAKE_RESUME
    assert result["cover_letter"] == FAKE_COVER_LETTER
    assert result["interview_questions"] == FAKE_INTERVIEW_PREP
    assert result["feedback"] == FAKE_REVIEW_FEEDBACK
    assert result["ats_result"].strip()

    expected_score = round(
        agent.config.ATS_SCORE_WEIGHT * result["ats_percentage"]
        + agent.config.LLM_SCORE_WEIGHT * FAKE_REVIEW_SCORE)
    assert result["score"] == expected_score
    assert result["score_class"] == ("good" if expected_score >= agent.config.SCORE_THRESHOLD
                                      else "warn" if expected_score >= 60 else "bad")
    assert result["weights"] == {"ats": agent.config.ATS_SCORE_WEIGHT,
                                  "llm": agent.config.LLM_SCORE_WEIGHT}
    assert "composite score" in result["status"]

    # Artifacts: a wire-driven run still exports real files with download URLs.
    artifact_kinds = {a["kind"] for a in result["artifacts"]}
    assert {"resume-pdf", "cover-pdf"} <= artifact_kinds
    for artifact in result["artifacts"]:
        download = client.get("/" + artifact["url"])
        assert download.status_code == 200
        assert len(download.content) > 0

    # The seven prompt-version stamps survive the concurrent fan-out merge.
    assert result["prompt_versions"] == EXPECTED_PROMPT_VERSIONS

    # MR2 rides the same payload: the deterministic requirement-fit table for
    # the FAKE_RESUME draft (no semantic rescues in the wire fake) - classified
    # in code, so 'kubernetes' honestly stays missing here.
    assert result["requirement_matches"] == [{
        "requirement": "Python engineer with Docker and Kubernetes experience.",
        "status": "partial",
        "evidence": ["python", "docker"],
        "missing": ["kubernetes"],
        "reason": "Covers python, docker; missing kubernetes.",
    }]

    # Wire call sequence: exact until the concurrent fan-out tail, whose order
    # is decided by the two sibling threads.
    keys = wire_llms.request_keys()
    assert keys[:-2] == FRESH_STEP2_WIRES
    assert sorted(keys[-2:]) == ["cover", "prep"]
    assert [entry["tool"] for entry in wire_llms.requests] == [
        None, None, "ReviewOutput", "SemanticReview", None, "ReviewOutput",
        "SemanticReview", None, None,
    ]

    assert len(feedback_calls) == 1
    state_sent, run_id_sent = feedback_calls[0]
    uuid.UUID(run_id_sent)
    assert state_sent.get("score") == expected_score


def test_step2_resumes_step1_session_over_the_wire(client, wire_llms, monkeypatch):
    feedback_calls = _spy_feedback(monkeypatch)
    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    assert r2.status_code == 200
    events = parse_sse(r2.text)
    assert events[-1][0] == "result"

    result = events[-1][1]
    assert result["session"] == session  # same thread continued, no fresh uuid
    statuses = _statuses(events)
    # The checkpoint interrupted BEFORE the improver: segment 1 is never re-run
    # (its scanner LLM call must not reappear on the wire either).
    assert not any("(1/2)" in s for s in statuses)
    assert any("(2/2)" in s for s in statuses)
    assert "improver" in {e["node"] for kind, e in events
                          if kind == "status" and e.get("node")}

    keys = wire_llms.request_keys()
    # index 0 is step 1's scanner call (same test fixture, same server record);
    # from the resume point onwards the shape matches a fresh run minus scanner.
    assert keys[:-2] == ["missing_skills", *FRESH_STEP2_WIRES[1:]]
    assert sorted(keys[-2:]) == ["cover", "prep"]

    assert result["prompt_versions"] == EXPECTED_PROMPT_VERSIONS
    assert len(feedback_calls) == 1


def test_step2_over_the_wire_survives_full_rewrite_fallback(client, wire_llms, monkeypatch):
    """The reviewer fake returns sections=[] on every pass, so EVERY polish pass
    routes to the improver's full-rewrite chain on the wire (the fallback that
    MR1 made structurally safe). The stamps must prove which polisher mode ran."""
    _spy_feedback(monkeypatch)
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "I used k8s at work"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[-1][0] == "result"
    result = events[-1][1]
    assert result["prompt_versions"]["polisher"] == "polisher-full@1"
    assert "polisher-target@1" not in result["prompt_versions"].values()
    # ...and the fake server saw the polisher fall through "Rewrite..." prompts
    assert wire_llms.request_keys().count("rewrite") == 2
    assert wire_llms.request_keys().count("polish_target") == 0
