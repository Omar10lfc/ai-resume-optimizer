"""Tests for the FastAPI server layer (server/) - the Gradio replacement.

Covers input validation (400/413), the SSE streaming contract of both pipeline
steps against the REAL graphs wired with fake LLMs (no network, no API cost),
pipeline queue/serialization semantics (Gradio queue port), artifact download
guards, app-wide basic auth, and health reporting.

Run with: venv\\Scripts\\python.exe -m pytest tests/test_server.py -v
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
from langchain_core.messages import AIMessage  # noqa: E402

import agent  # noqa: E402
import agent.config  # noqa: E402
import agent.llms  # noqa: E402
import agent.tracing  # noqa: E402
import server.app as server_app  # noqa: E402
import server.runners as runners  # noqa: E402
from server import run_store  # noqa: E402

JD = "Python engineer with Docker and Kubernetes experience."
RESUME_TEXT = "Jane Roe\nPython developer with cloud exposure."


# ==========================================
# Shared fakes (same shape as test_resume_agent)
# ==========================================

class FakeStrictLLM:
    def invoke(self, messages):
        return AIMessage(content="- Kubernetes\n- Terraform")

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                return agent.ReviewOutput(score=85, feedback="Add metrics to bullets.")
        return Structured()


class FakeCreativeLLM:
    def invoke(self, messages):
        return AIMessage(content="## Skills\nPython, Docker\n## Experience\nBuilt ML pipelines.")


class FakeFastLLM:
    def invoke(self, messages):
        prompt = messages[-1].content
        if "Cover Letter" in prompt:
            content = "Dear Hiring Team,\nI am excited to apply.\nSincerely, Candidate"
        else:
            content = "1. Sample question?\n\n## Preparation Tips\n- Tip one\n- Tip two"
        return AIMessage(content=content)

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                return agent.SemanticReview(verdicts=[])
        return Structured()


@pytest.fixture
def fake_llms(monkeypatch):
    monkeypatch.setattr(agent.llms, "llm_strict", FakeStrictLLM())
    monkeypatch.setattr(agent.llms, "llm_creative", FakeCreativeLLM())
    monkeypatch.setattr(agent.llms, "llm_fast", FakeFastLLM())


@pytest.fixture
def clean_name_register(monkeypatch):
    monkeypatch.setattr(agent.tracing, "_KNOWN_NAMES", [])
    monkeypatch.setattr(agent.tracing, "_NAME_MASKS", None)


@pytest.fixture
def client():
    # Without a context manager, lifespan does NOT run: app.state.health stays
    # untouched and the cleanup daemon and startup health check are skipped.
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


# ==========================================
# Health
# ==========================================

def test_health_reports_skipped_without_startup_check(client):
    body = client.get("/api/v1/health").json()
    assert body["ok"] is None
    assert "skipped" in body["message"]


def test_health_reflects_startup_check_when_lifespan_ran(monkeypatch):
    monkeypatch.setattr(agent.llms, "check_provider_health",
                        lambda: (False, "groq is down"))
    with TestClient(server_app.app) as c:
        body = c.get("/api/v1/health").json()
    assert body == {"ok": False, "message": "groq is down"}


# ==========================================
# Runners: config + resolution primitives
# ==========================================

def test_thread_config_pins_thread_and_run_id():
    cfg = runners._thread_config("t1", run_id="rid-123")
    assert cfg["configurable"]["thread_id"] == "t1"
    assert cfg["run_id"] == "rid-123"
    assert "callbacks" in cfg

    cfg = runners._thread_config("t2")
    assert "run_id" not in cfg


def test_resolve_resume_source_file_wins_over_text():
    assert runners.resolve_resume_source("p.pdf", "text") == "p.pdf"
    assert runners.resolve_resume_source(None, "  text  ") == "text"
    assert runners.resolve_resume_source(None, "   ") is None


# ==========================================
# Validation (HTTP level, before streaming)
# ==========================================

def test_step1_requires_job_description(client):
    r = client.post("/api/v1/step1",
                    data={"job_description": "   ", "resume_text": RESUME_TEXT})
    assert r.status_code == 400
    assert "Job Description" in r.json()["detail"]


def test_step1_requires_resume(client):
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": "  "})
    assert r.status_code == 400
    assert "resume" in r.json()["detail"].lower()


def test_step1_rejects_unsupported_file_type(client):
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": ""},
                    files={"resume_file": ("resume.txt", b"plain text", "text/plain")})
    assert r.status_code == 400
    assert "PDF or DOCX" in r.json()["detail"]


def test_step1_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(agent.config, "MAX_PDF_BYTES", 64)
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": ""},
                    files={"resume_file": ("resume.pdf", b"A" * 100, "application/pdf")})
    assert r.status_code == 413


def test_step2_rejects_invalid_session(client):
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "session": "not-a-uuid"})
    assert r.status_code == 400
    assert "session" in r.json()["detail"].lower()


# ==========================================
# Step 1: SSE stream over the real graph
# ==========================================

def test_step1_stream_premasks_name_and_returns_gaps(client, fake_llms, clean_name_register):
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": RESUME_TEXT})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done"
    assert "error" not in kinds

    done = events[-1][1]
    uuid.UUID(done["session"])  # well-formed uuid
    assert "Kubernetes" in done["missing_skills"]
    assert any("(step 1 of 2)" in s for s in _statuses(events))
    assert any("Loading documents..." in s for s in _statuses(events))
    # node-aware statuses: every tagged event carries a node id + running state
    tagged = [e for kind, e in events if kind == "status" and e.get("node")]
    assert tagged and all(e["node_state"] == "running" for e in tagged)
    tagged_nodes = {e["node"] for e in tagged}
    assert {"loader", "scanner"} <= tagged_nodes or tagged_nodes == {"loader"}
    assert "Jane Roe" in agent.tracing._KNOWN_NAMES  # pasted resume pre-masked


def test_step1_accepts_docx_upload(client, fake_llms, clean_name_register):
    import io
    from docx import Document

    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Jane Roe")
    doc.add_paragraph("Python developer")
    doc.save(buf)
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": ""},
                    files={"resume_file": ("resume.docx", buf.getvalue(),
                                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[-1][0] == "done"
    # the loader registers the candidate name parsed from the document
    assert "Jane Roe" in agent.tracing._KNOWN_NAMES


# ==========================================
# Step 2: SSE stream - fresh / resume / reuse flows
# ==========================================

def _spy_feedback(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(runners, "log_pipeline_feedback",
                        lambda state, run_id: calls.append((state, run_id)))
    return calls


def test_step2_from_scratch_warns_and_returns_full_result(client, fake_llms, monkeypatch):
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
    assert result["optimized_resume"].strip()
    assert result["cover_letter"].startswith("Dear")
    assert "Preparation Tips" in result["interview_questions"]
    assert result["ats_result"].strip()

    expected = round(agent.config.ATS_SCORE_WEIGHT * result["ats_percentage"]
                     + agent.config.LLM_SCORE_WEIGHT * 85)
    assert result["score"] == expected
    assert result["score_class"] == ("good" if expected >= agent.config.SCORE_THRESHOLD
                                     else "warn" if expected >= 60 else "bad")
    assert result["weights"] == {"ats": agent.config.ATS_SCORE_WEIGHT,
                                 "llm": agent.config.LLM_SCORE_WEIGHT}
    assert "composite score" in result["status"]

    artifact_kinds = {a["kind"] for a in result["artifacts"]}
    assert {"resume-pdf", "cover-pdf"} <= artifact_kinds
    for artifact in result["artifacts"]:
        dl = client.get("/" + artifact["url"])
        assert dl.status_code == 200
        assert len(dl.content) > 0

    assert len(feedback_calls) == 1
    state_sent, run_id_sent = feedback_calls[0]
    uuid.UUID(run_id_sent)
    assert state_sent.get("score") == expected

    statuses = _statuses(events)
    assert any("(1/2)" in s for s in statuses)
    assert any("(2/2)" in s for s in statuses)

    # node ids accompany the per-node statuses across the whole run incl. fan-out
    tagged = [e for kind, e in events if kind == "status" and e.get("node")]
    assert tagged and all(e["node_state"] == "running" for e in tagged)
    tagged_nodes = {e["node"] for e in tagged}
    # "polisher": the reviewer's default fake (85 + feedback) now earns one
    # polish pass even on a passing composite (Nov 6 2026 directive).
    assert {"loader", "scanner", "improver", "reviewer", "ats_check", "polisher",
            "cover_letter", "interview_prep", "pdf_exporter"} <= tagged_nodes


def test_step2_resumes_interrupted_session_from_step1(client, fake_llms, clean_name_register):
    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    events = parse_sse(r2.text)
    assert events[-1][0] == "result"
    result = events[-1][1]
    assert result["session"] == session
    statuses = _statuses(events)
    # checkpoint interruption: segment 1 (loader/scanner) is NOT re-run
    assert not any("(1/2)" in s for s in statuses)
    assert any("(2/2)" in s for s in statuses)
    tagged_nodes = {e["node"] for kind, e in events
                    if kind == "status" and e.get("node")}
    assert "improver" in tagged_nodes
    assert not {"loader", "scanner"} & tagged_nodes

    # the user's edited notes were injected at the human-review checkpoint
    snapshot = agent.agent_app.get_state({"configurable": {"thread_id": session}})
    assert snapshot.values.get("human_notes") == "I used k8s at work"


# ==========================================
# Composed-notes split (runs a32f30fc / 868e406d)
# ==========================================
# The frontend composes ONE string: typed free text, then kept chip lines,
# then "Do NOT feature: ...". Stamping that whole string into BOTH
# human_notes and missing_skills made the model fold kept chips under the
# pasted project text into fake "user-confirmed" bullets ("Gained
# experience with n8n (or similar ...)").

def test_split_composed_notes_separates_chips_and_typed_evidence():
    composed = (
        "Aegis: built a 4-node LangGraph agent with FastAPI and OpenRouter.\n"
        "- Experience with n8n (or similar workflow automation platform)\n"
        "- Hands-on work with Claude Code, Cursor, or Codex\n"
        "- Direct experience working with MCP servers (FortyGuard Temperature MCP)\n"
        "Do NOT feature: Terraform, Airflow.\n"
    )
    pristine = (
        "- Experience with n8n (or similar workflow automation platform)\n"
        "- Hands-on work with Claude Code, Cursor, or Codex\n"
        "- Direct experience working with MCP servers (FortyGuard Temperature MCP)\n"
        "- Terraform\nAirflow\n"
    )
    gaps, evidence = runners._split_composed_notes(composed, pristine)

    # GAP channel: chips (kept as annotated) + the removal directive
    assert "Experience with n8n (or similar workflow automation platform)" in gaps
    assert "Hands-on work with Claude Code, Cursor, or Codex" in gaps
    assert "Direct experience working with MCP servers (FortyGuard Temperature MCP)" in gaps
    assert "Do NOT feature: Terraform, Airflow." in gaps

    # EVIDENCE channel: ONLY the typed project text
    assert evidence == "Aegis: built a 4-node LangGraph agent with FastAPI and OpenRouter."

    # chips must never leak into the evidence channel again
    assert "n8n" not in evidence and "Claude Code" not in evidence


def test_split_composed_notes_ascii_chip_without_snapshot_routes_to_gap():
    """Stale-chip hardening (request 539f7c1d, Nov 8 2026): without a step-1
    snapshot AND with one, an ASCII-bullet line is chip serialization, not
    typed evidence - routing it to the gap channel means the writer merely
    omits that topic (safe), while routing it to evidence launders stale
    gap chips into user-confirmed claims (the 75/100% incident). Prose and
    '•'-bulleted lines are still typed evidence even without a snapshot."""
    composed = "Built dashboards with Grafana.\n- Experience with n8n\n"
    gaps, evidence = runners._split_composed_notes(composed, "")
    assert gaps == "- Experience with n8n"
    assert evidence == "Built dashboards with Grafana."


def test_split_composed_notes_stale_chips_under_pasted_evidence():
    """The exact 539f7c1d composition: the user pasted Aegis evidence with
    '•' bullets, then STALE chip lines from the previous session (wording
    drifted from the fresh scanner snapshot). None of the chip lines match
    the pristine chips, yet every ASCII-bullet line must land in the GAP
    channel while the '•'-bulleted project evidence stays in human_notes."""
    composed = (
        "here is a project that i built:\n"
        "Aegis: Autonomous Heat-Risk Operations Agent Aug-Sep 2026\n"
        "• Engineered a FastAPI service around a 4-node LangGraph agent\n"
        "  with bounded LLM timeouts and deterministic fallbacks.\n"
        "- Experience with n8n (or similar workflow-automation platform) and building automations using it\n"
        "- Hands-on work with Claude Code, Cursor, Codex or comparable LLM-code generation tools\n"
        "- Direct experience working with MCP servers (FortyGuard Temperature MCP)\n"
    )
    # TODAY's snapshot wording (deliberately different from the stale lines)
    pristine = (
        "- Experience with n8n (or similar workflow automation platform)\n"
        "- Hands-on work with Claude Code, Cursor, or Codex for building AI agents and demos\n"
        "- Experience working with MCP servers (FortyGuard Temperature MCP)\n"
    )
    gaps, evidence = runners._split_composed_notes(composed, pristine)
    assert gaps.count("n8n") == 1 and gaps.count("FortyGuard") == 1
    assert "workflow-automation" in gaps          # stale wording kept verbatim
    assert "here is a project that i built:" in evidence
    assert "Aegis: Autonomous Heat-Risk Operations Agent" in evidence
    assert "• Engineered a FastAPI service around a 4-node LangGraph agent" in evidence
    # nothing chip-shaped leaks into the evidence channel
    for tool in ("n8n", "Claude Code", "Codex", "MCP"):
        assert tool not in evidence


def test_split_composed_notes_bare_chip_openers_route_to_gap():
    """Round 7 (runs 25ff4ae5 / 1a8125f0, Nov 12 2026): the pasted context
    carried the scanner's MCP gap phrasings BARE (no '- ' bullet). Each
    run's snapshot matched one variant - the scanner words chips slightly
    differently per run - so that one went to the gap channel and was
    declined on-record, while the OTHER variant missed the snapshot, had
    no ASCII bullet, and fell into the evidence channel as user-confirmed
    experience. Opener-shaped lines are gap topics whatever the snapshot
    says; prose and '•' evidence still survive."""
    composed = (
        "Aegis: built a heat-risk agent with FastAPI.\n"
        "Experience working with MCP servers (FortyGuard Temperature MCP)\n"
        "Direct experience with MCP servers (FortyGuard Temperature MCP)\n"
        "• Built dashboards for the ops team\n"
    )
    pristine = ("- Experience with n8n\n"
                "- Experience working with MCP servers "
                "(FortyGuard Temperature MCP)\n")
    gaps, evidence = runners._split_composed_notes(composed, pristine)
    assert "Experience working with MCP servers" in gaps
    assert "Direct experience with MCP servers" in gaps
    assert evidence == ("Aegis: built a heat-risk agent with FastAPI.\n"
                        "• Built dashboards for the ops team")
    assert "MCP" not in evidence and "FortyGuard" not in evidence


def test_split_composed_notes_tolerates_chip_annotations_and_blank_lines():
    """'- Docker: ran containers in coursework' matches the pristine chip by
    its LABEL (text before the first colon)."""
    composed = "My project text about CI.\n\n- Docker: ran containers in coursework\n"
    pristine = "- Docker\n"
    gaps, evidence = runners._split_composed_notes(composed, pristine)
    assert gaps == "- Docker: ran containers in coursework"
    assert evidence == "My project text about CI."


def test_split_composed_notes_without_pristine_typed_evidence_survives():
    """No step-1 snapshot (fresh step-2 entry): prose and '•'-bulleted
    context must still survive verbatim - the no-vanish spirit of the old
    fallback now applies to typed evidence only, while ASCII-bullet lines
    (the extension's chip serialization) still route to the gap channel
    regardless of the snapshot (stale-chip hardening, request 539f7c1d)."""
    composed = ("Refactored the deploy pipeline on a home lab.\n"
                "• Ran container basics on a home lab cluster\n"
                "- Docker\n")
    gaps, evidence = runners._split_composed_notes(composed, "")
    assert gaps == "- Docker"
    assert evidence == ("Refactored the deploy pipeline on a home lab.\n"
                        "• Ran container basics on a home lab cluster")


def test_split_composed_notes_bullet_marker_equivalence():
    """Pristine chips may come with or without '- ' bullets; labels match
    either way (front-end compose always uses '- ')."""
    gaps, evidence = runners._split_composed_notes(
        "Typed context line.\n- Kubernetes basics",
        "Kubernetes basics\nTerraform")
    assert gaps == "- Kubernetes basics"
    assert evidence == "Typed context line."


def test_step2_reuses_completed_session_by_starting_fresh(client, fake_llms, monkeypatch):
    _spy_feedback(monkeypatch)
    r1 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "some notes"})
    first_session = parse_sse(r1.text)[-1][1]["session"]

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "some notes", "session": first_session})
    events = parse_sse(r2.text)
    fresh_events = [e for kind, e in events if kind == "session"]
    assert fresh_events and fresh_events[0]["session"] != first_session
    assert any("fresh run" in s for s in _statuses(events))
    assert events[-1][1]["session"] != first_session


def test_step2_pipeline_error_becomes_error_event(client, fake_llms, monkeypatch, clean_name_register):
    class BrokenImproverLLM:
        def invoke(self, messages):
            raise RuntimeError("provider outage")

    # the improver is the first node after the interrupt -> fails the resume
    monkeypatch.setattr(agent.llms, "llm_creative", BrokenImproverLLM())
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "notes"})
    events = parse_sse(r.text)
    assert events[-1][0] == "error"
    assert "provider outage" in events[-1][1]["message"]


def test_step2_emits_integrity_warning_before_result(client, fake_llms, monkeypatch):
    """Run 14751ff9 (Nov 6 2026): with EMPTY user notes the model invented
    'cut manual data-ingestion effort by 40%'. The integrity tripwire must
    now surface such claims as a warning event immediately before the
    result, so the user reviews the drafted numbers."""
    class InventingCreativeLLM(FakeCreativeLLM):
        def invoke(self, messages):
            return AIMessage(content="## Skills\nPython, Docker\n## Experience\n"
                                     "Built ML pipelines. Cut manual data-ingestion effort by 40%.")

    monkeypatch.setattr(agent.llms, "llm_creative", InventingCreativeLLM())
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "- I used k8s at work"})
    events = parse_sse(r.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result"
    assert "error" not in kinds

    warnings = [e for kind, e in events if kind == "warning"]
    # Non-empty notes suppress the auto-detected-gaps warning, so the only
    # warning in this stream must be the integrity one.
    assert len(warnings) == 1
    assert "Integrity check" in warnings[0]["message"]
    assert "40%" in warnings[0]["message"]
    assert kinds.index("warning") == len(kinds) - 2      # directly precedes result


def test_step2_result_payload_carries_integrity_and_versions(client, fake_llms, monkeypatch):
    """QW1/QW2/QW3: the final result payload carries the forensic prompt
    stamps, the scrubber's removal list, and the writer's declined requests
    (the ResultsDashboard renders them as the Integrity tab)."""
    from agent.nodes.improver import IMPROVER_PROMPT_VERSION
    from agent.nodes.cover_letter import COVER_LETTER_PROMPT_VERSION
    from agent.nodes.interview_prep import INTERVIEW_PREP_PROMPT_VERSION

    class InventingDecliningLLM(FakeCreativeLLM):
        def invoke(self, messages):
            return AIMessage(content="## Skills\nPython, Docker\n## Experience\n"
                                     "Built ML pipelines. Cut manual data-ingestion effort by 40%.\n\n"
                                     "DECLINED: Experience with n8n")

    monkeypatch.setattr(agent.llms, "llm_creative", InventingDecliningLLM())
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "- I used k8s at work"})
    events = parse_sse(r.text)
    assert events[-1][0] == "result"
    payload = events[-1][1]

    assert payload["declined_requests"] == ["Experience with n8n"]
    # The fabricated 40% was deterministically scrubbed -> it lands in the
    # removals list ("what we removed & why"), not silently in the resume.
    assert any("40%" in m for m in payload["removals"])
    assert "40%" not in payload["optimized_resume"]

    pv = payload["prompt_versions"]
    assert pv["improver"] == IMPROVER_PROMPT_VERSION
    assert pv["cover-letter"] == COVER_LETTER_PROMPT_VERSION
    assert pv["interview-prep"] == INTERVIEW_PREP_PROMPT_VERSION
    # Both fan-out siblings ship their own stamp (reducer merge, not clobber).
    assert pv["polisher"]

    # MR5: per-task token accounting always rides the result payload. The
    # fake-invoke harness bypasses the callback layer (fakes replace whole
    # chains), so the totals are zeroed here - the exact numbers are
    # asserted in the WIRE tests, which run real chat-model calls.
    usage = payload["token_usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens",
                          "total_tokens", "llm_calls"}
    assert usage["llm_calls"] == 0 and usage["total_tokens"] == 0


def test_cover_letter_fabricated_metrics_are_flagged(client, fake_llms,
                                                    monkeypatch):
    """Live run 22e943c6 (Nov 11 2026): the cover letter fabricated "70%"
    and "45%" - numbers absent from every input - because the resume
    path's unevidenced-metric tripwire never watched the letter. The
    letter now gets the same scan (evidence = source resume + user notes)
    surfaced as its own warning event + payload key."""
    class FabricatingLetterLLM:
        def invoke(self, messages):
            return AIMessage(content=(
                "Dear Hiring Manager,\n\n"
                "I reduced manual processing time by 70% for a climate-data "
                "client and increased user engagement by 45%.\n\n"
                "Sincerely, Jane Roe"))

    monkeypatch.setattr(agent.llms, "llm_fast", FabricatingLetterLLM())
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "- I used k8s at work"})
    events = parse_sse(r.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result"

    payload = events[-1][1]
    assert payload["cover_letter_metrics"] == ["70%", "45%"]
    # no cross-contamination: the letter's findings live in their own
    # channel - the resume path's accumulator is untouched
    assert payload["unevidenced_metrics"] == []

    warnings = [e for kind, e in events if kind == "warning"]
    # clean resume draft + non-empty notes -> the letter's warning is the
    # only one, and it lands directly before the result like the resume's
    assert len(warnings) == 1
    assert "cover letter" in warnings[0]["message"]
    assert "70%" in warnings[0]["message"] and "45%" in warnings[0]["message"]
    assert kinds.index("warning") == len(kinds) - 2


def test_cover_letter_evidenced_metrics_pass_clean(client, fake_llms,
                                                  monkeypatch):
    """The scan's positive control: numbers the USER NOTES evidence are
    legitimate letter content and raise no cover-letter warning."""
    class EvidencedLetterLLM:
        def invoke(self, messages):
            return AIMessage(content=(
                "Dear Hiring Manager,\n\n"
                "At work I reduced manual processing time by 70% and would "
                "bring the same rigor to your team.\n\n"
                "Sincerely, Jane Roe"))

    monkeypatch.setattr(agent.llms, "llm_fast", EvidencedLetterLLM())
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "I reduced manual processing time by 70% at work"})
    events = parse_sse(r.text)
    assert events[-1][0] == "result"

    payload = events[-1][1]
    assert payload["cover_letter_metrics"] == []
    warnings = [e for kind, e in events if kind == "warning"]
    assert not any("cover letter" in w["message"] for w in warnings)


def test_step2_result_payload_carries_requirement_table(client, fake_llms):
    """MR2: the SSE result payload ships the deterministic requirement-fit
    table (covered/partial/gap rows with evidence terms and human reasons),
    which the ResultsDashboard renders as its Fit tab."""
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": ""})
    events = parse_sse(r.text)
    assert events[-1][0] == "result"
    payload = events[-1][1]

    rows = payload["requirement_matches"]
    assert len(rows) == 1
    row = rows[0]
    assert row["requirement"] == "Python engineer with Docker and Kubernetes experience."
    assert row["status"] == "partial"
    assert row["evidence"] == ["python", "docker"]
    assert row["missing"] == ["kubernetes"]
    assert row["reason"] == "Covers python, docker; missing kubernetes."


# ==========================================
# Queue semantics (Gradio queue port)
# ==========================================

def test_queue_full_emits_error_event_not_http(client, fake_llms, monkeypatch):
    monkeypatch.setattr(server_app, "_QUEUE_MAX", 0)
    r = client.post("/api/v1/step1",
                    data={"job_description": JD, "resume_text": RESUME_TEXT})
    events = parse_sse(r.text)
    assert len(events) == 1
    kind, payload = events[0]
    assert kind == "error"
    assert "queue is full" in payload["message"]


def test_run_waits_for_free_slot_when_pipeline_busy(client, fake_llms):
    assert server_app._RUN_LOCK.acquire(blocking=False)
    timer = threading.Timer(0.6, server_app._RUN_LOCK.release)
    timer.start()
    try:
        r = client.post("/api/v1/step1",
                        data={"job_description": JD, "resume_text": RESUME_TEXT})
    finally:
        timer.cancel()
    events = parse_sse(r.text)
    assert any(s.startswith("Waiting for a free pipeline slot")
               for s in _statuses(events))
    assert events[-1][0] == "done"  # completed after the slot freed


# ==========================================
# Artifact downloads
# ==========================================

def test_artifact_download_roundtrip(client, monkeypatch, tmp_path):
    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    session = str(uuid.uuid4())
    session_dir = tmp_path / session
    session_dir.mkdir()
    (session_dir / "resume_optimized.pdf").write_bytes(b"%PDF-1.4 resume")
    (session_dir / "cover_letter.pdf").write_bytes(b"%PDF-1.4 cover")
    (session_dir / "resume.docx").write_bytes(b"PK\x03\x04 docx bytes")
    (session_dir / "resume.tex").write_text("\\documentclass{article}")

    pdf = client.get(f"/api/v1/artifacts/{session}/resume-pdf",
                     params={"file": "resume_optimized.pdf"})
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")
    assert "resume_optimized.pdf" in pdf.headers["content-disposition"]
    assert pdf.content.startswith(b"%PDF")

    cover = client.get(f"/api/v1/artifacts/{session}/cover-pdf",
                       params={"file": "cover_letter.pdf"})
    assert cover.status_code == 200

    docx = client.get(f"/api/v1/artifacts/{session}/resume-docx",
                      params={"file": "resume.docx"})
    assert docx.status_code == 200
    assert docx.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    tex = client.get(f"/api/v1/artifacts/{session}/resume-tex",
                     params={"file": "resume.tex"})
    assert tex.status_code == 200
    assert tex.headers["content-type"].startswith("text/x-tex")


def test_artifact_unknown_session_or_kind_is_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    session = str(uuid.uuid4())
    (tmp_path / session).mkdir()

    assert client.get(f"/api/v1/artifacts/{session}/resume-pdf",
                      params={"file": "missing.pdf"}).status_code == 404
    assert client.get(f"/api/v1/artifacts/not-a-uuid/resume-pdf",
                      params={"file": "x.pdf"}).status_code == 404
    assert client.get(f"/api/v1/artifacts/{session}/resume-xls",
                      params={"file": "resume.pdf"}).status_code == 404


def test_artifact_traversal_is_contained(client, monkeypatch, tmp_path):
    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    session = str(uuid.uuid4())
    (tmp_path / session).mkdir()
    (tmp_path / "outside_secret.pdf").write_bytes(b"%PDF")

    # Path.name strips the '../' component, so the probe looks for a
    # non-existent file inside the session dir and is contained by construction.
    r = client.get(f"/api/v1/artifacts/{session}/resume-pdf",
                   params={"file": "../outside_secret.pdf"})
    assert r.status_code == 404


# ==========================================
# App-wide basic auth
# ==========================================

def test_auth_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(server_app, "AUTH_CREDENTIALS", ("admin", "s3cret"))
    r = client.get("/api/v1/health")
    assert r.status_code == 401
    assert "basic" in r.headers["www-authenticate"].lower()

    r_wrong = client.get("/api/v1/health", auth=("admin", "nope"))
    assert r_wrong.status_code == 401

    r_ok = client.get("/api/v1/health", auth=("admin", "s3cret"))
    assert r_ok.status_code == 200


def test_api_documentation_not_exposed(client, monkeypatch):
    # FastAPI's built-in docs don't inherit app-level auth dependencies, so
    # the schema routes are disabled entirely (internal API; the BFF owns
    # the public contract).
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404


def test_open_access_when_auth_not_configured(client):
    assert client.get("/api/v1/health").status_code == 200


# ==========================================
# BYO Groq key (browser-extension BYOK support)
# ==========================================

BYO_KEY = "gsk_client_supplied_for_tests"


class _FakeChain:
    def invoke(self, messages):  # duck-typed delegate for Easy stubbing
        return "invoked"


def test_groq_key_override_rebinds_and_restores(monkeypatch):
    recorded: list[tuple[str, str | None]] = []

    def fake_create_llm(temperature, reasoning_effort=None, api_key=None):
        recorded.append(("llm", api_key))
        return _FakeChain()

    def fake_create_llm_fast(temperature, api_key=None):
        recorded.append(("fast", api_key))
        return _FakeChain()

    monkeypatch.setattr(agent.llms, "_create_llm", fake_create_llm)
    monkeypatch.setattr(agent.llms, "_create_llm_fast", fake_create_llm_fast)
    monkeypatch.setattr(agent.llms, "_OVERRIDDEN_CHAINS", {})
    monkeypatch.setattr(agent.llms, "_ACTIVE_GROQ_KEY", None)

    assert agent.llms.active_groq_key() is None
    with agent.llms.groq_key_override(BYO_KEY):
        assert agent.llms.active_groq_key() == BYO_KEY

        # First attribute resolution builds ALL three scoped chains for the key
        assert agent.llms.llm_strict.invoke([]) == "invoked"
        assert [(kind, key) for kind, key in recorded] == \
            [("llm", BYO_KEY), ("llm", BYO_KEY), ("fast", BYO_KEY)]
        assert BYO_KEY in agent.llms._OVERRIDDEN_CHAINS

        # Subsequent resolutions reuse the cached chains without rebuilding
        assert agent.llms.llm_fast.invoke([]) == "invoked"
        assert len(recorded) == 3

    with pytest.raises(RuntimeError):
        with agent.llms.groq_key_override("gsk_nested_key"):
            raise RuntimeError("boom")
    assert agent.llms.active_groq_key() is None


def test_stream_step1_clears_groq_key_after_run(fake_llms, clean_name_register):
    events = list(runners.stream_step1(JD, RESUME_TEXT, None, groq_api_key=BYO_KEY))
    assert agent.llms.active_groq_key() is None
    assert events[-1]["type"] == "done"


def test_step_passes_client_groq_key_header_to_runner(client, fake_llms,
                                                      clean_name_register, monkeypatch):
    captured: dict[str, str | None] = {}
    original_step1, original_step2 = runners.stream_step1, runners.stream_step2

    def spy_step1(job_description, resume_source, session, groq_api_key=None):
        captured["step1"] = groq_api_key
        return original_step1(job_description, resume_source, session, groq_api_key)

    def spy_step2(job_description, resume_source, notes, session,
                  groq_api_key=None, **kwargs):
        captured["step2"] = groq_api_key
        return original_step2(job_description, resume_source, notes, session,
                              groq_api_key, **kwargs)

    monkeypatch.setattr(runners, "stream_step1", spy_step1)
    monkeypatch.setattr(runners, "stream_step2", spy_step2)

    # step1 carries the header; step2 intentionally does not
    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT},
                     headers={server_app._CLIENT_GROQ_KEY_HEADER: BYO_KEY})
    _ = parse_sse(r1.text)
    assert captured["step1"] == BYO_KEY

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    _ = parse_sse(r2.text)
    assert captured["step2"] is None


def test_health_with_user_key_validates_that_key(client, monkeypatch):
    seen: dict[str, str] = {}

    def fake_status(key, *args, **kwargs):
        seen["key"] = key
        return True, "API reachable (99 models)."

    monkeypatch.setattr(server_app, "_groq_models_status", fake_status)
    body = client.get("/api/v1/health",
                      headers={server_app._CLIENT_GROQ_KEY_HEADER: BYO_KEY}).json()
    assert seen["key"] == BYO_KEY
    assert body == {"ok": True, "message": "API reachable (99 models)."}

    # no header -> unchanged behavior (pre-parse comes from startup state)
    assert client.get("/api/v1/health").status_code == 200


def test_client_groq_key_header_is_sanitized():
    from fastapi import Request as _Request  # local import keeps the test lint-clean

    req = _Request({"type": "http", "query_string": b"", "headers": [(b"x-groq-api-key", b"  gsk_x  ")]})
    assert server_app._client_groq_key(req) == "gsk_x"

    req_empty = _Request({"type": "http", "query_string": b"", "headers": []})
    assert server_app._client_groq_key(req_empty) is None


# ==========================================
# MR3: durable, reattachable runs (run_store + recovery)
# ==========================================
# Live defects this round closes (Nov 11 2026): a crashed attempt's scoring
# snapshot (best_result / score / ats) leaked into the next attempt on the
# same thread - live runs 29afd91b / 62725081 shipped ATS 56.2 while the
# SHIPPED resume recomputed to 37.5 / 43.8, and the 7.4s score-59
# "recovery" was a band-aid stitch of half-completed checkpoints. The
# notes injection was also invisible on every resumed invoke's trace
# (untraced update_state).

@pytest.fixture(autouse=True)
def runs_db(tmp_path, monkeypatch):
    """Hermetic runs ledger + profile store per test (both read their
    config paths dynamically, so the monkeypatch is enough)."""
    path = tmp_path / "runs.sqlite3"
    monkeypatch.setattr(agent.config, "RUNS_DB_PATH", str(path))
    monkeypatch.setattr(agent.config, "PROFILE_DB_PATH",
                        str(tmp_path / "profile.sqlite3"))
    monkeypatch.setattr(agent.config, "PROFILE_DB_ENABLED", True)
    return path


def test_runs_endpoint_roundtrips_step1_and_step2(client, fake_llms):
    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]

    row = client.get(f"/api/v1/runs/{session}").json()
    assert row["status"] == "awaiting_notes"   # durable review checkpoint
    assert row["step"] == "step1"
    assert "Kubernetes" in row["missing_skills"]
    assert row["result"] is None

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    assert parse_sse(r2.text)[-1][0] == "result"

    row = client.get(f"/api/v1/runs/{session}").json()
    assert row["status"] == "done"
    assert row["result"]["optimized_resume"].strip()
    assert row["session"] == session


def test_runs_ledger_records_failures_with_reason(client, fake_llms,
                                                  monkeypatch):
    class BrokenImproverLLM:
        def invoke(self, messages):
            raise RuntimeError("provider exploded")

    # the improver is the first node after the interrupt -> fails the run
    monkeypatch.setattr(agent.llms, "llm_creative", BrokenImproverLLM())
    known = "66666666-6666-6666-6666-666666666666"
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": "", "session": known})
    events = parse_sse(r.text)
    assert events[-1][0] == "error"

    row = client.get(f"/api/v1/runs/{known}").json()
    assert row["status"] == "failed"
    assert "provider exploded" in row["error"]


def test_runs_endpoint_unknown_or_invalid_session_is_404(client):
    assert client.get(
        f"/api/v1/runs/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/v1/runs/not-a-uuid").status_code == 404


def test_recover_interrupted_marks_running_rows_failed(runs_db):
    import sqlite3

    run_store.start("11111111-1111-1111-1111-111111111111", "step2")
    run_store.finish_scan("22222222-2222-2222-2222-222222222222",
                          "- Kubernetes")
    run_store.supersede("33333333-3333-3333-3333-333333333333", "done elsewhere")

    recovered = run_store.recover_interrupted()
    assert recovered == 1
    dead = run_store.get("11111111-1111-1111-1111-111111111111")
    assert dead["status"] == "failed"
    assert "backend restart" in dead["error"]

    # the durable review checkpoint survives a restart UNtouched
    alive = run_store.get("22222222-2222-2222-2222-222222222222")
    assert alive["status"] == "awaiting_notes"
    assert alive["missing_skills"] == "- Kubernetes"
    # superseded rows are never mistaken for interrupted work
    assert run_store.get("33333333-3333-3333-3333-333333333333")["status"] \
        == "superseded"

    # second recovery is a no-op
    assert run_store.recover_interrupted() == 0
    with sqlite3.connect(str(runs_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE status='running'"
                            ).fetchone()[0] == 0


def test_run_store_purges_expired_rows(runs_db):
    import sqlite3

    run_store.start("44444444-4444-4444-4444-444444444444", "step1")
    with sqlite3.connect(str(runs_db)) as conn:
        conn.execute("UPDATE runs SET created_at = 0, updated_at = 0"
                     " WHERE session = '44444444-4444-4444-4444-444444444444'")

    run_store.start("55555555-5555-5555-5555-555555555555", "step2")

    assert run_store.get("44444444-4444-4444-4444-444444444444") is None
    assert run_store.get("55555555-5555-5555-5555-555555555555") is not None


def test_step2_recovery_recomputes_stale_ats_snapshot(client, fake_llms,
                                                      clean_name_register):
    """THE defect: mid-pipeline leftovers + a stale best_result snapshot on
    the thread must never leak into the recovery attempt."""
    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]
    config = {"configurable": {"thread_id": session}}

    # Fabricate the crashed-attempt shape: past ats_check (best snapshot
    # written), died in the polisher/reviewer loop.
    agent.agent_app.update_state(
        config,
        {"optimized_resume": "STALE DRAFT", "score": 59,
         "ats_percentage": 56.2, "iteration": 1, "ats_result": "STALE",
         "llm_quality_score": 85,
         "best_result": {"score": 59, "optimized_resume": "STALE DRAFT",
                         "ats_percentage": 56.2, "ats_result": "STALE",
                         "llm_quality_score": 85, "requirement_matches": []}},
        as_node="polisher")
    assert agent.agent_app.get_state(config).next == ("reviewer",)

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    events = parse_sse(r2.text)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result"
    warnings = [e for kind, e in events if kind == "warning"]
    assert any("Recovered an interrupted run" in w["message"] for w in warnings)

    result = events[-1][1]
    # the stale snapshot is gone: fresh score from the fresh draft
    assert result["optimized_resume"] != "STALE DRAFT"
    assert result["ats_percentage"] != 56.2
    expected = round(agent.config.ATS_SCORE_WEIGHT * result["ats_percentage"]
                     + agent.config.LLM_SCORE_WEIGHT * 85)
    assert result["score"] == expected

    # the recovery redid the model segment WITHOUT re-running the loader
    statuses = _statuses(events)
    assert not any("(1/2)" in s for s in statuses)
    tagged_nodes = {e["node"] for kind, e in events
                    if kind == "status" and e.get("node")}
    assert {"improver", "reviewer", "ats_check", "cover_letter",
            "pdf_exporter"} <= tagged_nodes
    assert not {"loader", "scanner"} & tagged_nodes

    snapshot = agent.agent_app.get_state(config)
    values = snapshot.values or {}
    assert values.get("human_notes") == "I used k8s at work"
    assert values.get("best_result", {}).get("score") == expected

    # and the ledger shows the recovery attempt as done
    row = client.get(f"/api/v1/runs/{session}").json()
    assert row["status"] == "done"


def test_step2_recovery_after_real_midnode_crash_redoes_model_segment(
        client, clean_name_register, monkeypatch):
    """THE regression (live run 22e943c6, Nov 11 2026): a REAL mid-node
    crash - the polisher's 3rd targeted rewrite raising, not a fabricated
    update_state - leaves an INTERRUPTED polisher task in the checkpoint.
    The recovery's notes injection (``update_state`` without as_node) was
    then re-anchored onto that task by LangGraph's versions_seen inference,
    which advanced ``next`` past the improver to the fan-out: the
    "recovered" segment only re-ran cover_letter / interview_prep /
    pdf_exporter and shipped score 0 with "No resume generated." The
    rewound path must write reset + notes as ONE ``update_state`` anchored
    with ``as_node="scanner"`` (probe-reproduced and probe-verified)."""

    class CrashyCreative:
        """Mirrors the live incident: invoke 1 = the improver's full
        draft, invokes 2-4 = the polisher's targeted section rewrites,
        invoke 4 RAISES mid-node (the live APIConnectionError)."""

        DRAFT = ("## Professional Summary\n"
                 "Python developer with cloud exposure.\n\n"
                 "## Technical Skills\n- Python, Docker\n\n"
                 "## Experience\n- Built ML pipelines.\n")

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError(
                    "Simulated APIConnectionError('Connection error.')")
            prompt = messages[-1].content
            if "SURGICAL REVISION" in prompt:
                lines = prompt.splitlines()
                header = "## Technical Skills"
                for i, line in enumerate(lines):
                    if line.strip().startswith(
                            "REVISION TARGET (return the full replacement"):
                        header = lines[i + 1].strip()
                        break
                return AIMessage(content=f"{header}\n- Rewritten: {header}")
            return AIMessage(content=self.DRAFT)

    class StrictWithSections:
        def invoke(self, messages):
            return AIMessage(content="- Kubernetes\n- Terraform")

        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ReviewOutput(
                        score=62,
                        feedback="Rewrite the summary to target the role "
                                 "and reorder skills.",
                        sections=["Professional Summary", "Technical Skills",
                                  "Experience"])
            return Structured()

    crashy = CrashyCreative()
    monkeypatch.setattr(agent.llms, "llm_strict", StrictWithSections())
    monkeypatch.setattr(agent.llms, "llm_creative", crashy)
    monkeypatch.setattr(agent.llms, "llm_fast", FakeFastLLM())

    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]
    config = {"configurable": {"thread_id": session}}

    # attempt 1 dies mid-polisher (creative invoke #4)
    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    assert parse_sse(r2.text)[-1][0] == "error"
    # the crash left an INTERRUPTED task - not a fabricated snapshot
    assert agent.agent_app.get_state(config).next == ("polisher",)

    r3 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work", "session": session})
    events = parse_sse(r3.text)
    assert [kind for kind, _ in events][-1] == "result"
    warnings = [e for kind, e in events if kind == "warning"]
    assert any("Recovered an interrupted run" in w["message"] for w in warnings)

    # the model segment redid its work: the improver re-ran (calls 5+,
    # one per recovery improver entry) and every loop node re-executed
    assert crashy.calls >= 5
    tagged_nodes = {e["node"] for kind, e in events
                    if kind == "status" and e.get("node")}
    assert {"improver", "reviewer", "ats_check", "polisher",
            "cover_letter", "pdf_exporter"} <= tagged_nodes

    result = events[-1][1]
    assert result["optimized_resume"].strip()
    assert "No resume generated" not in result["optimized_resume"]
    assert result["score"] > 0

    snapshot = agent.agent_app.get_state(config)
    values = snapshot.values or {}
    assert values.get("human_notes") == "I used k8s at work"
    assert snapshot.next == ()

    # and the ledger shows the recovery attempt as done with a real score
    row = client.get(f"/api/v1/runs/{session}").json()
    assert row["status"] == "done"
    assert row["result"]["score"] > 0


def test_split_composed_notes_dedupes_exact_gap_lines():
    """Live run 22e943c6: a crash retry re-carried chip lines the crashed
    attempt had already injected into the checkpoint, doubling the gap
    channel the improver reads as research topics. Identical (normalized)
    gap lines collapse - first occurrence wins - while wording variants
    and distinct chip annotations stay; user evidence is never deduped."""
    gaps, evidence = runners._split_composed_notes(
        "- Kubernetes\n"
        "- Kubernetes\n"
        "- Docker: used it in coursework\n"
        "- Kubernetes: used k8s at work\n"
        "I also led the migration\n"
        "I also led the migration",
        "- Kubernetes\n- Terraform")
    assert gaps == ("- Kubernetes\n"
                    "- Docker: used it in coursework\n"
                    "- Kubernetes: used k8s at work")
    # evidence is user prose - untouched by the gap-channel dedupe
    assert evidence == "I also led the migration\nI also led the migration"


def test_step2_resume_writes_notes_without_reentering_the_graph(client,
                                                                fake_llms,
                                                                monkeypatch,
                                                                clean_name_register):
    """MR3 + probe-calibrated Nov 11: providing the notes as graph INPUT
    re-enters the graph from the entry point (loader+scanner re-run: one
    extra fast-model call, and the fresh scanner chips CLOBBER the user's
    edited gap lines - caught by the MR4 wire tests). The write must be
    ``update_state``, with the ``notes_injection`` side run restoring trace
    visibility of the injected payload."""
    streams = []
    updates = []
    original_stream = agent.agent_app.stream
    original_update = agent.agent_app.update_state

    def spy_stream(graph_input, config, **kwargs):
        streams.append(graph_input)
        return original_stream(graph_input, config, **kwargs)

    def spy_update(config, values, as_node=None):
        updates.append(dict(values))
        return original_update(config, values, as_node=as_node)

    monkeypatch.setattr(agent.agent_app, "stream", spy_stream)
    monkeypatch.setattr(agent.agent_app, "update_state", spy_update)

    traced = []
    monkeypatch.setattr(runners, "_trace_notes_injection",
                        lambda session, notes, gaps: traced.append((notes, gaps)))

    r1 = client.post("/api/v1/step1",
                     data={"job_description": JD, "resume_text": RESUME_TEXT})
    session = parse_sse(r1.text)[-1][1]["session"]
    streams.clear()  # only the step-2 segment is under test

    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "- Kubernetes: used at work\nI also led the migration",
                           "session": session})
    assert parse_sse(r2.text)[-1][0] == "result"

    # the ONLY graph stream in the resume segment is the None-resume: an
    # input-provided stream would re-run loader/scanner (re-entry bug)
    assert streams == [None]
    # the write is update_state, exactly once, with the SPLIT payloads:
    # the kept chip line (matches the step-1 snapshot) -> gap channel, the
    # typed prose -> evidence channel
    assert len(updates) == 1
    assert updates[0]["human_notes"] == "I also led the migration"
    assert updates[0]["missing_skills"] == "- Kubernetes: used at work"
    # the traced side run carries the same split payload for LangSmith
    assert traced == [("I also led the migration", "- Kubernetes: used at work")]

    snapshot = agent.agent_app.get_state({"configurable": {"thread_id": session}})
    values = snapshot.values or {}
    assert values.get("human_notes") == "I also led the migration"
    assert values.get("missing_skills") == "- Kubernetes: used at work"


# ==========================================
# BB1 Phase A: persistent profile store (application history)
# ==========================================

def test_profile_store_records_and_dedupes_by_pair(runs_db):
    import server.profile_store as profile_store

    rid1 = profile_store.record_run(JD, RESUME_TEXT, "sess-1",
                                    {"score": 71, "ats_percentage": 44.0,
                                     "llm_quality_score": 85,
                                     "review_failed": False,
                                     "optimized_resume": "R1",
                                     "requirement_matches": []})
    assert rid1 is not None
    # Same (JD, resume) pair -> the SAME application, a second attempt.
    rid2 = profile_store.record_run(JD, RESUME_TEXT, "sess-2",
                                    {"score": 76, "ats_percentage": 50.0,
                                     "llm_quality_score": 88,
                                     "review_failed": False,
                                     "optimized_resume": "R2"})
    apps = profile_store.list_applications()
    assert len(apps) == 1
    assert apps[0]["attempts"] == 2
    assert apps[0]["best_score"] == 76
    assert apps[0]["title"].startswith("Python engineer")
    assert "resume_text" not in apps[0]  # summaries stay cheap

    detail = profile_store.get_application(apps[0]["id"])
    assert detail["job_text"] == JD
    assert detail["resume_text"] == RESUME_TEXT
    assert [r["id"] for r in detail["runs"]] == [rid2, rid1]  # newest first
    assert detail["runs"][0]["result"]["optimized_resume"] == "R2"
    assert detail["runs"][-1]["result"]["optimized_resume"] == "R1"

    # A different resume is a different application (same posting).
    profile_store.record_run(JD, RESUME_TEXT + "\nMore experience.", "sess-3",
                             {"score": 60})
    assert len(profile_store.list_applications()) == 2


def test_profile_store_nonfatal_and_disabled(runs_db, monkeypatch, capsys):
    import server.profile_store as profile_store

    # Store failure is non-fatal: warn + continue, never raise.
    def broken_connect():
        raise RuntimeError("disk full")

    monkeypatch.setattr(profile_store, "_connect", broken_connect)
    assert profile_store.record_run(JD, RESUME_TEXT, "s", {"score": 1}) is None
    assert profile_store.list_applications() == []
    assert profile_store.get_application(1) is None
    assert profile_store.delete_application(1) is False
    assert "recording failed" in capsys.readouterr().out

    # Disabled posture (PROFILE_DB_PATH=off): nothing recorded, empty reads.
    monkeypatch.setattr(agent.config, "PROFILE_DB_ENABLED", False)
    monkeypatch.setattr(profile_store, "_connect",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("must not open when disabled")))
    assert profile_store.record_run(JD, RESUME_TEXT, "s", {"score": 1}) is None
    assert profile_store.list_applications() == []
    assert profile_store.get_application(1) is None
    assert profile_store.delete_application(1) is False


def test_applications_endpoints_roundtrip(client, fake_llms):
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": ""})
    assert parse_sse(r.text)[-1][0] == "result"

    listing = client.get("/api/v1/applications").json()["applications"]
    assert len(listing) == 1
    row = listing[0]
    assert row["attempts"] == 1
    assert row["best_score"] is not None
    assert "Python engineer" in row["title"]
    assert "resume_text" not in row

    detail = client.get(f"/api/v1/applications/{row['id']}").json()
    assert detail["job_text"].strip() == JD
    assert detail["runs"][0]["session"]
    assert detail["runs"][0]["result"]["optimized_resume"].strip()

    # Re-running the same pair attaches a second attempt - no duplicate app.
    r2 = client.post("/api/v1/step2",
                     data={"job_description": JD, "resume_text": RESUME_TEXT,
                           "notes": "I used k8s at work"})
    assert parse_sse(r2.text)[-1][0] == "result"
    listing = client.get("/api/v1/applications").json()["applications"]
    assert len(listing) == 1
    assert listing[0]["attempts"] == 2

    assert client.delete(f"/api/v1/applications/{row['id']}").status_code == 200
    assert client.get("/api/v1/applications").json()["applications"] == []
    assert client.get(f"/api/v1/applications/{row['id']}").status_code == 404


def test_applications_endpoint_validation(client):
    assert client.get("/api/v1/applications/999999").status_code == 404
    assert client.delete("/api/v1/applications/999999").status_code == 404
    assert client.get("/api/v1/applications?limit=0").status_code == 422
    assert client.get("/api/v1/applications?limit=500").status_code == 422


# ==========================================
# BB1 Phase B: workspace artifacts + history re-download + server-side diff
# ==========================================

def test_output_root_default_is_project_local():
    """PB1: artifacts default into the workspace's gitignored data/ dir
    (was %TEMP%/resume_optimizer_output - OS temp cleaners could vanish
    files mid-retention-window). The OUTPUT_ROOT env knob stays available
    for overrides; tests monkeypatch the attribute directly."""
    if os.environ.get("OUTPUT_ROOT", "").strip():
        pytest.skip("OUTPUT_ROOT explicitly set in this environment")
    project_root = Path(agent.__file__).resolve().parents[1]
    assert agent.config.OUTPUT_ROOT == project_root / "data" / "artifacts"


def test_regenerate_run_artifacts_from_stored_text(client, fake_llms, monkeypatch, tmp_path):
    """PB2: a past run's artifacts are re-created from the STORED text after
    the originals hit the 24h sweep. Regeneration goes through the CURRENT
    template; with LaTeX absent the resume still degrades to HTML->PDF."""
    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT, "notes": ""})
    assert parse_sse(r.text)[-1][0] == "result"

    detail = client.get("/api/v1/applications/1").json()
    run_id = detail["runs"][0]["id"]

    def regen(kind):
        return client.get(f"/api/v1/applications/1/runs/{run_id}/artifacts/{kind}")

    resume_pdf = regen("resume-pdf")
    assert resume_pdf.status_code == 200
    assert resume_pdf.headers["content-type"].startswith("application/pdf")
    assert resume_pdf.content[:4] == b"%PDF"

    cover_pdf = regen("cover-pdf")
    assert cover_pdf.status_code == 200
    assert cover_pdf.headers["content-type"].startswith("application/pdf")
    assert cover_pdf.content[:4] == b"%PDF"

    # The .tex source is written before compilation (pure Python render),
    # so it regenerates regardless of pdflatex availability.
    tex = regen("resume-tex")
    assert tex.status_code == 200
    assert tex.headers["content-type"].startswith("text/x-tex")
    assert b"documentclass" in tex.content

    # DOCX needs pandoc: served when present, honest 404 when the renderer
    # is unavailable (endpoint-only surface - the UI exposes PDFs only).
    assert regen("resume-docx").status_code in (200, 404)

    # Regeneration writes into FRESH request dirs under OUTPUT_ROOT, left
    # behind for the regular 24h sweep (pipeline dir + regenerated dirs).
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) >= 2


def test_regenerate_run_artifact_validation(client, monkeypatch, tmp_path):
    import server.profile_store as profile_store

    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    assert client.get(
        "/api/v1/applications/1/runs/1/artifacts/resume-xls").status_code == 404
    assert client.get(
        "/api/v1/applications/999/runs/1/artifacts/resume-pdf").status_code == 404

    rid = profile_store.record_run(
        JD, RESUME_TEXT, "sess-x",
        {"score": 60, "ats_percentage": 40.0, "llm_quality_score": 70,
         "review_failed": False,
         "optimized_resume": "No resume generated.",
         "cover_letter": "No cover letter generated."})
    assert rid is not None
    # _extract_results' placeholder strings are not exportable content.
    assert client.get(
        f"/api/v1/applications/1/runs/{rid}/artifacts/resume-pdf").status_code == 404
    assert client.get(
        f"/api/v1/applications/1/runs/{rid}/artifacts/cover-pdf").status_code == 404

    # A run id from a DIFFERENT application is unknown (no cross-app leaks).
    profile_store.record_run(JD, RESUME_TEXT + "\nMore experience.", "sess-y",
                             {"score": 50})
    assert client.get(
        f"/api/v1/applications/2/runs/{rid}/artifacts/resume-pdf").status_code == 404


def test_step2_include_diff_original_echoes_parsed_resume(client, fake_llms):
    """PB3: include_diff=original echoes the PARSED original resume so
    DiffView works for uploads too (the SPA never parses PDFs client-side)."""
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": ""},
                    params={"include_diff": "original"})
    assert parse_sse(r.text)[-1][0] == "result"
    result = parse_sse(r.text)[-1][1]
    # Pasted text passes through the loader unmodified -> exact echo.
    assert result["original_resume"] == RESUME_TEXT

    # Stored history: the echo is NOT duplicated at rest - resume_text is
    # already the application's evidence column.
    detail = client.get("/api/v1/applications/1").json()
    assert "original_resume" not in detail["runs"][0]["result"]
    assert detail["resume_text"] == RESUME_TEXT


def test_step2_without_include_diff_omits_original(client, fake_llms):
    r = client.post("/api/v1/step2",
                    data={"job_description": JD, "resume_text": RESUME_TEXT,
                          "notes": ""})
    assert parse_sse(r.text)[-1][0] == "result"
    assert "original_resume" not in parse_sse(r.text)[-1][1]
