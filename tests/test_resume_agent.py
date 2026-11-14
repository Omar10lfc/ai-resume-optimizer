"""Unit and integration tests for the AI Resume Optimizer agent.

Run with: venv\\Scripts\\python.exe -m pytest tests -v

Unit tests cover the deterministic ATS/keyword engine, Unicode PDF cleaning,
the hallucination post-filter, URL validation, and hybrid score computation.
Integration tests run both LangGraph graphs end-to-end with fake chat models
(no network, no API cost).
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
import agent.config  # noqa: E402
import agent.llms  # noqa: E402
import agent.tracing  # noqa: E402
import agent.url_fetch  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402


# ==========================================
# _simple_stem
# ==========================================

@pytest.mark.parametrize("word,expected", [
    ("models", "model"),
    ("deployment", "deploy"),
    ("optimization", "optim"),
    ("libraries", "librar"),
    ("testing", "test"),
    ("trained", "train"),
])
def test_simple_stem_known_cases(word, expected):
    assert agent._simple_stem(word) == expected


def test_simple_stem_short_words_untouched():
    assert agent._simple_stem("go") == "go"
    assert agent._simple_stem("ai") == "ai"


# ==========================================
# Keyword matching (boundary / synonym / stem)
# ==========================================

def test_short_keyword_no_false_positive_inside_word():
    text = "category theory and logarithms of the googleplex".lower()
    assert agent._keyword_found_in_text("go", text)[0] is False
    assert agent._keyword_found_in_text("r", text)[0] is False


def test_short_keyword_matches_as_whole_word():
    text = "experience with r, go and sql databases".lower()
    assert agent._keyword_found_in_text("go", text)[0] is True
    assert agent._keyword_found_in_text("r", text)[0] is True


def test_synonym_match_kubernetes_k8s():
    found, method = agent._keyword_found_in_text(
        "kubernetes", "orchestrated workloads on k8s clusters")
    assert found and method == "synonym"


def test_stem_match_deployment_deploying():
    found, method = agent._keyword_found_in_text(
        "deployment", "responsible for deploying services")
    assert found and method == "stem"


def test_multiword_phrase_exact_match():
    found, method = agent._keyword_found_in_text(
        "machine learning", "strong background in machine learning systems")
    assert found and method == "exact"


# ==========================================
# Keyword extraction
# ==========================================

def test_extract_keywords_excludes_jd_boilerplate():
    jd = ("You have:\nYour responsibilities\n"
          "Support engineering teams in adopting AI tools.\n"
          "About Acme: equal opportunity employer.")
    kws = agent._extract_keywords(jd, top_n=20)
    garbage = {"have", "adopting", "responsibilities", "acme"}
    assert not [k for k in kws if k.rstrip(":").lower() in garbage]


def test_extract_keywords_dedupes_compound_variants():
    jd = "Strong CI/CD experience required. Familiarity with ci cd pipelines too."
    kws = agent._extract_keywords(jd, top_n=20)
    variants = sum(1 for k in kws if k in ("ci/cd", "ci cd"))
    assert variants == 1


def test_extract_keywords_low_signal_requires_frequency():
    jd = "Support the team. We value support. Support builds support."
    kws = agent._extract_keywords(jd, top_n=50)
    assert "support" in kws


def test_low_signal_single_mention_suppressed():
    jd = "Occasional testing of features. Python development."
    kws = agent._extract_keywords(jd, top_n=50)
    assert "testing" not in kws
    jd2 = "Support the team. We value support. Support matters."
    kws2 = agent._extract_keywords(jd2, top_n=50)
    assert "support" not in kws2


def test_action_verbs_never_keywords():
    jd = ("Design and implement Agentic AI workflows. "
          "Contribute to LLM-based applications. "
          "Collaborate to enhance developer productivity.")
    kws = agent._extract_keywords(jd, top_n=20)
    for verb in ("implement", "contribute", "design", "collaborate", "enhance"):
        assert verb not in kws


def test_extract_keywords_skips_perk_and_apply_sections():
    """Perk/legal/application sections never yield keywords (live-run
    evidence Nov 5 2026: 'tools', 'basic' leaked via such content)."""
    jd = (
        "Temperature data agents need Python, Kafka and strong APIs.\n"
        "## What we offer\nFree snacks, gym membership and a commuting budget.\n"
        "## How to apply\nSend your CV with a short note to jobs@fortyguard.example\n"
    )
    kws = agent._extract_keywords(jd)
    assert "python" in kws and "kafka" in kws
    junk = [k for k in ("snacks", "gym", "commuting", "cv") if k in kws]
    assert junk == []


def test_extract_keywords_stops_timeline_and_logistics_noise():
    jd = (
        "## Requirements\nStrong Python engineering.\n"
        "Start date: January 2027. Duration: 6 months, flexible hours.\n"
        "Eligibility: basic scheduling. Relocation support available.\n"
    )
    kws = agent._extract_keywords(jd)
    assert "python" in kws
    leaked = [k for k in ("january", "start", "date", "duration", "basic",
                          "scheduling", "flexible", "hours", "relocation") if k in kws]
    assert leaked == []


def test_extract_keywords_drops_posting_header_fields():
    """Nov 7 2026 run pair (FortyGuard AI Automation Intern post, traces
    01a1b4cb/09cf51fc): a tiny scraped JD carried a plain-text header block
    ('Location: Remote / Duration: 3-6 months / Type: Paid internship')
    plus prose verbs - 5 of the 20 weighted keywords were words NO resume
    can contain, which depresses every honest match. Header fields and
    prose verbs are stopped; real gap terms ('mcp', 'temperature') stay."""
    jd = ("We're hiring an AI Automation Intern at FortyGuard!\n\n"
          "Start: January 2027\nLocation: Remote\nDuration: 3\u20136 months\n"
          "Type: Paid internship\n\n"
          "You'll work with the FortyGuard Temperature MCP, which enables AI agents "
          "to interact with our temperature data.\n\n"
          "What you'll be building:\n\n"
          "\u2022 AI agents, workflows, and demos using tools like Claude Code, "
          "Cursor, and Codex.\n\n"
          "\u2022 n8n automations that help our internal team work more efficiently.\n\n"
          "Who we're looking for:\n\n"
          "Someone who has built projects using LLM tools, AI agents, or "
          "workflow automation.\n\n"
          "You should understand APIs, webhooks, and JSON, and have experience "
          "using n8n or a similar automation platform.\n\n"
          "Experience working with MCP servers is a plus.")
    kws = agent._extract_keywords(jd, top_n=20)
    junk = [k for k in ("location", "months", "paid", "type", "interact",
                        "understand", "someone", "built", "efficiently",
                        "internal", "help") if k in kws]
    assert junk == []
    assert "mcp" in kws
    assert "temperature" in kws
    assert "claude" in kws
    assert "cursor" in kws


def test_extract_keywords_strips_camelcase_brand_but_keeps_tech():
    jd = ("FortyGuard builds temperature-data agents. Requirements: Python "
          "and OpenCV. We also use MySQL.")
    kws = agent._extract_keywords(jd)
    assert "fortyguard" not in kws
    assert "python" in kws
    assert "opencv" in kws   # CamelCase tech term protected via tier list
    assert "mysql" in kws    # CamelCase tech term protected via allowlist


def test_company_filter_never_eats_skill_before_suffix_verb():
    """Nov 10 2026 probe: the company heuristic 'X develops/builds...' had no
    trailing word boundary, so 'develop' prefix-matched 'development' in
    "3+ years of Python development experience" - 'python' was captured as a
    company name and left the entire ATS vocabulary (surfaced by the MR2
    requirement table, which lost its Python requirement card)."""
    jd = "TechCorp builds data platforms. 3+ years of Python development experience."
    kws = agent._extract_keywords(jd, top_n=20)
    assert "python" in kws
    assert "techcorp" not in kws


def test_camelcase_brand_filter_never_eats_devsecops():
    """Golden JD regression (Nov 5 2026): the camelCase brand filter matched
    Dev|Sec|Ops and dropped the required 'devsecops' keyword because it is
    absent from the tier lists — it must survive alongside 'devops'."""
    jd = "Support DevOps, DevSecOps, CI/CD automation, testing."
    kws = agent._extract_keywords(jd, top_n=20)
    assert "devsecops" in kws
    assert "devops" in kws


def test_extract_keywords_passes_plain_text_without_headings():
    """No recognized heading → nothing is stripped (evidence, not guessing)."""
    jd = "Python, Docker, Kafka, and API experience needed. Great growth path."
    kws = agent._extract_keywords(jd)
    assert "python" in kws and "docker" in kws and "kafka" in kws


def test_extract_keywords_normalizes_literal_escaped_newlines():
    jd = ("Your responsibilities\\nSupport engineering teams.\\n"
          "Design Agentic AI workflows using LangGraph.\\n"
          "Contribute to RAG pipelines.\\nWork on MCP integrations.\\n"
          "Configure IDEs and engineering tools.")
    kws = agent._extract_keywords(jd, top_n=20)
    # Action verbs with literal \n should NOT become nsupport, ndesign, ncontribute, nwork, nconfigure
    assert "nsupport" not in kws
    assert "ndesign" not in kws
    assert "ncontribute" not in kws
    assert "nwork" not in kws
    assert "nconfigure" not in kws
    # Real keywords should be present
    assert "langgraph" in kws
    assert "rag" in kws
    assert "mcp" in kws


def test_hyphenated_fragments_suppressed():
    jd = "Contribute to LLM-based applications and AI-native tooling. Use front-end frameworks."
    kws = agent._extract_keywords(jd, top_n=20)
    assert "llm-based" not in kws
    assert "ai-native" not in kws
    assert "front-end" in kws or "front end" in kws


# ==========================================
# Unicode / PDF cleaning
# ==========================================

def test_force_ascii_replaces_dashes_and_emoji():
    text = "Led team \u2013 achieved \u2705 results \u2192 done \U0001F680"
    out = agent._force_ascii(text)
    assert all(ord(c) < 128 for c in out)
    assert "--" in out or "-" in out
    assert "->" in out


def test_force_ascii_transliterates_accents():
    assert agent._force_ascii("café") == "cafe"


def test_strip_code_fences():
    fenced = "```markdown\n## Header\nBody\n```"
    assert agent._strip_code_fences(fenced) == "## Header\nBody"


# ==========================================
# Hallucination filter
# ==========================================

def test_hallucinated_section_removed():
    original = "## Experience\nDid things.\n## Skills\nPython"
    optimized = original + "\n## Certifications\nAWS Certified - fabricated!"
    cleaned = agent._filter_hallucinated_sections(optimized, original)
    assert "fabricated" not in cleaned.lower()
    assert "Certifications" not in cleaned


def test_legitimate_section_preserved():
    original = "## Certifications\nAWS Certified.\n## Skills\nPython"
    optimized = "## Certifications\nAWS Certified.\n## Skills\nPython, Go"
    cleaned = agent._filter_hallucinated_sections(optimized, original)
    assert "AWS Certified" in cleaned


# ==========================================
# Hybrid composite scoring
# ==========================================

def test_composite_score_formula():
    composite = round(agent.ATS_SCORE_WEIGHT * 90 + agent.LLM_SCORE_WEIGHT * 70)
    assert composite == round(54 + 28) == 82


def test_score_threshold_and_cap_constants():
    assert agent.SCORE_THRESHOLD == 80
    # 2 = one feedback-driven polish pass beyond the first draft; the
    # ats_check best-of guard keeps regression impossible.
    assert agent.MAX_ITERATIONS == 2
    assert abs(agent.ATS_SCORE_WEIGHT + agent.LLM_SCORE_WEIGHT - 1.0) < 1e-9


# ==========================================
# Job URL validation (SSRF protections)
# ==========================================

def test_job_url_rejects_non_https(monkeypatch):
    with pytest.raises(ValueError, match="HTTPS"):
        agent._validate_job_url("http://boards.greenhouse.io/job")


def test_job_url_rejects_disallowed_domain(monkeypatch):
    monkeypatch.setattr(agent.url_fetch, "ALLOWED_JOB_URL_DOMAINS", {"boards.greenhouse.io"})
    with pytest.raises(ValueError, match="allowlist"):
        agent._validate_job_url("https://evil.example.com/job")


def test_job_url_rejects_private_ip(monkeypatch):
    import agent.url_fetch as uf
    monkeypatch.setattr(uf, "ALLOWED_JOB_URL_DOMAINS", {"boards.greenhouse.io"})

    def fake_getaddrinfo(host, port, type=None):
        return [(None, None, None, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(uf.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="public IP"):
        agent._validate_job_url("https://boards.greenhouse.io/job")


def test_job_url_accepts_public_ip(monkeypatch):
    import agent.url_fetch as uf
    monkeypatch.setattr(uf, "ALLOWED_JOB_URL_DOMAINS", {"boards.greenhouse.io"})

    def fake_getaddrinfo(host, port, type=None):
        return [(None, None, None, "", ("140.82.121.4", 0))]

    monkeypatch.setattr(uf.socket, "getaddrinfo", fake_getaddrinfo)
    url = agent._validate_job_url("https://boards.greenhouse.io/job/123")
    assert url.endswith("/job/123")


# ==========================================
# PII redaction & retention cleanup
# ==========================================

def test_pii_redaction_masks_emails_and_phones():
    from agent.tracing import _redact_pii
    data = {
        "resume_text": "Contact omar.example@gmail.com or +20 100 170 1821 today.",
        "messages": [{"content": "email: a.b@company.org"}],
        "nested": {"score": 90},
    }
    out = _redact_pii(data)
    assert "omar.example@gmail.com" not in out["resume_text"]
    assert "[redacted-email]" in out["resume_text"]
    assert "[redacted-phone]" in out["resume_text"]
    assert "[redacted-email]" in out["messages"][0]["content"]
    assert out["nested"]["score"] == 90


def test_pii_redaction_leaves_technical_numbers_alone():
    from agent.tracing import _redact_pii
    text = "Score 82/100 with 3 years of GPT-4 experience"
    out = _redact_pii(text)
    assert "82/100" in out
    assert "GPT-4" in out


def test_stale_request_dirs_purged(tmp_path, monkeypatch):
    import os
    import time
    old_dir = tmp_path / "old-request"
    new_dir = tmp_path / "new-request"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "resume.pdf").write_text("x")
    two_days_ago = time.time() - 48 * 3600
    os.utime(old_dir, (two_days_ago, two_days_ago))

    monkeypatch.setattr(agent.config, "OUTPUT_ROOT", tmp_path)
    removed = agent._cleanup_stale_request_dirs(max_age_hours=24)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()


# ==========================================
# Integration: graphs with fake LLMs
# ==========================================

class FakeStrictLLM:
    def invoke(self, messages):
        return AIMessage(content="- Kubernetes\n- Terraform")

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                return agent.ReviewOutput(score=85, feedback="Add metrics to bullets.")
        return Structured()


class _CreativeStructuredMixin:
    """BB1 Phase B: the improver (and the polisher's full-rewrite fallback)
    now call with_structured_output(ImproverOutput) through the MR5 ladder.
    The fakes answer from their own invoke() markdown - the same seam as
    before - switching to the whole-document unit when the prompt offers
    only "[id=\"Resume\"]" (a heading-less source)."""

    def with_structured_output(self, schema):
        if schema is not agent.ImproverOutput:
            raise NotImplementedError(
                "creative-profile fakes only serve ImproverOutput")
        outer = self

        class Structured:
            def invoke(self, messages):
                prompt = messages[-1].content
                markdown = outer.invoke(messages).content
                return agent.improver_output_from_markdown(
                    markdown, whole_document='[id="Resume"]' in prompt)

        return Structured()


class FakeCreativeLLM(_CreativeStructuredMixin):
    def invoke(self, messages):
        return AIMessage(content="## Skills\nPython, Docker\n## Experience\nBuilt ML pipelines.")


class FakeFastLLM:
    def invoke(self, messages):
        prompt = messages[-1].content
        if "Cover Letter" in prompt:
            content = "Dear Hiring Team,\nI am excited to apply.\nSincerely, Candidate"
        else:
            content = "1. Sample question?\n\n## Preparation Tips\n- Tip one\n- Tip two\n- Tip three"
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


def _base_inputs():
    return {
        "job_description": "Python engineer with Docker and Kubernetes experience.",
        "original_resume": "Python developer.",
        "human_notes": "",
        "missing_skills": "- Kubernetes",
        "resume_text": "", "job_text": "", "optimized_resume": "",
        "feedback": "", "score": 0, "iteration": 0,
        "llm_quality_score": 0, "ats_percentage": 0.0, "review_failed": False,
        "cover_letter": "", "interview_questions": "", "ats_result": "",
        "resume_pdf_path": "", "cover_letter_pdf_path": "",
        "resume_docx_path": "", "resume_tex_path": ""
    }


def test_agent_graph_interrupts_for_human_review(fake_llms):
    """Step 1: the interactive graph runs loader -> scanner and interrupts
    before the Improver -- the first-class human-in-the-loop checkpoint."""
    config = {"configurable": {"thread_id": "test-session-1"}}
    result = agent.agent_app.invoke(_base_inputs(), config=config)
    snapshot = agent.agent_app.get_state(config)
    assert snapshot.next == ("improver",)
    assert "Kubernetes" in (snapshot.values or {}).get("missing_skills", "")
    assert snapshot.values.get("resume_text") == "Python developer."
    assert snapshot.values.get("request_id")

    agent.agent_app.update_state(config, {"human_notes": "used k8s at work",
                                         "missing_skills": "- Kubernetes"})
    final = agent.agent_app.invoke(None, config=config)
    assert "Kubernetes" in final["missing_skills"]
    assert final["human_notes"] == "used k8s at work"
    assert final["optimized_resume"]
    assert final["cover_letter"].startswith("Dear")
    assert "Preparation Tips" in final["interview_questions"]
    assert final["ats_result"]
    expected = round(agent.ATS_SCORE_WEIGHT * final["ats_percentage"]
                     + agent.LLM_SCORE_WEIGHT * 85)
    assert final["score"] == expected
    req_dir = Path(final["output_dir"])
    assert req_dir.parent == agent.OUTPUT_ROOT
    assert Path(final["resume_pdf_path"]).exists()
    assert Path(final["cover_letter_pdf_path"]).exists()


def test_full_graph_end_to_end(fake_llms):
    """The non-interactive graph runs straight through (CLI/test path)."""
    result = agent.full_app.invoke(_base_inputs())
    assert result["optimized_resume"]
    assert result["cover_letter"].startswith("Dear")
    assert "Preparation Tips" in result["interview_questions"]
    assert result["ats_result"]
    expected = round(agent.ATS_SCORE_WEIGHT * result["ats_percentage"]
                     + agent.LLM_SCORE_WEIGHT * 85)
    assert result["score"] == expected
    req_dir = Path(result["output_dir"])
    assert req_dir.parent == agent.OUTPUT_ROOT
    assert Path(result["resume_pdf_path"]).exists()
    assert Path(result["cover_letter_pdf_path"]).exists()
    assert Path(result["resume_pdf_path"]).parent == req_dir


def test_optimizer_reviewer_failure_skips_retry(fake_llms, monkeypatch):
    class BrokenReviewer(FakeStrictLLM):
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    raise RuntimeError("structured output unavailable")
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", BrokenReviewer())
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is True
    assert result["iteration"] == 1


# ==========================================
# Best-of guard: extra polish iteration (MAX_ITERATIONS=2)
# ==========================================

class SeqCreativeLLM(_CreativeStructuredMixin):
    """Returns queued drafts in call order; tracks invocations."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self._contents) - 1)
        self.calls += 1
        return AIMessage(content=self._contents[idx])


def _run_with_drafts(drafts, monkeypatch, reviewer_score=60, feedback="Add metrics.",
                     sections=()):
    """Reviewer 60 keeps the composite under the 80 threshold so the loop
    actually runs a second iteration under the new default cap. Generic
    feedback keeps the rewriter grounded (improver embeds it as guidance).
    The evidence text mirrors the fixture drafts' skills: since the
    deterministic fabrication scrubber (runs a32f30fc/868e406d) removes
    gap-chip echoes the evidence lacks, these routing-only tests would lose
    their fixture content otherwise.

    sections (optional) feeds the reviewer's structured section targeting
    (MR1): the polisher picks its section-targeted or full-rewrite mode
    from it.
    """

    class FixedReviewer(FakeStrictLLM):
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ReviewOutput(score=reviewer_score,
                                              feedback=feedback,
                                              sections=list(sections))
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FixedReviewer())
    seq = SeqCreativeLLM(drafts)
    monkeypatch.setattr(agent.llms, "llm_creative", seq)
    inputs = _base_inputs()
    inputs["original_resume"] = "Python developer experienced with Docker and Kubernetes."
    return agent.full_app.invoke(inputs), seq


def test_iteration_two_can_never_regress_below_draft_one(fake_llms, monkeypatch):
    """A weaker second draft loses: the first draft is restored wholesale
    (resume, ATS result, percentage, composite, quality)."""
    good = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Python services on Kubernetes with Docker."
    worse = "## Skills\nPython\n## Experience\nWrote some scripts."
    result, seq = _run_with_drafts([good, worse], monkeypatch)

    assert seq.calls == 2
    assert result["optimized_resume"] == good
    expected_pct = agent.compute_ats_match(_base_inputs()["job_description"], good)["percentage"]
    assert result["ats_percentage"] == expected_pct
    assert result["score"] == round(agent.ATS_SCORE_WEIGHT * expected_pct
                                    + agent.LLM_SCORE_WEIGHT * 60)
    assert result["llm_quality_score"] == 60


def test_iteration_two_wins_when_stronger(fake_llms, monkeypatch):
    """A stronger second draft is adopted (feedback actually improves fit)."""
    weaker = "## Skills\nPython\n## Experience\nWrote some scripts."
    better = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Kubernetes pipelines with Docker and Python."
    result, seq = _run_with_drafts([weaker, better], monkeypatch)

    assert seq.calls == 2
    assert result["optimized_resume"] == better
    expected_pct = agent.compute_ats_match(_base_inputs()["job_description"], better)["percentage"]
    assert result["ats_percentage"] == expected_pct


def test_max_iterations_cap_still_respected(fake_llms, monkeypatch):
    """MAX_ITERATIONS=1 keeps the single-pass behavior opt-in."""
    good = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Kubernetes services with Python and Docker."
    monkeypatch.setattr(agent.graphs, "MAX_ITERATIONS", 1)
    result, seq = _run_with_drafts([good, "## Skills\nPython"], monkeypatch)

    assert seq.calls == 1
    assert result["optimized_resume"] == good


def test_passing_review_with_feedback_still_gets_one_polish_pass(fake_llms, monkeypatch):
    """Live run 14751ff9 (Nov 6 2026): a PASSING reviewer (composite 86 >= 80)
    still left concrete feedback ('adjust the summary to target the role')
    which was dropped because passing scores bailed straight to fan-out.
    Directive: non-empty feedback earns exactly one polish pass whenever the
    iteration cap allows — passing or failing."""
    good = "## Skills\nPython\n## Experience\nWrote some scripts."
    polished = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Kubernetes pipelines with Docker and Python."
    result, seq = _run_with_drafts([good, polished], monkeypatch, reviewer_score=85)

    assert seq.calls == 2                      # first draft + one polish pass
    assert result["iteration"] == 2
    assert result["review_failed"] is False
    assert result["optimized_resume"] == polished   # best-of adopts the better pass


def test_passing_review_without_feedback_skips_polish(fake_llms, monkeypatch):
    """A clean pass (no feedback) must still finish in a single pass — the
    polish route keys on feedback, never on marking time."""
    result, seq = _run_with_drafts(["## Skills\nPython"], monkeypatch,
                                   reviewer_score=85, feedback="")

    assert seq.calls == 1
    assert result["iteration"] == 1


def test_polish_pass_cannot_regress_below_first_draft(fake_llms, monkeypatch):
    """A weaker polish draft loses through the same best-of guard that
    protects failing-score retries."""
    good = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Kubernetes pipelines with Docker and Python."
    worse = "## Skills\nPython\n## Experience\nWrote some scripts."
    result, seq = _run_with_drafts([good, worse], monkeypatch, reviewer_score=85)

    assert seq.calls == 2
    assert result["optimized_resume"] == good


# ==========================================
# QW1: prompt version stamps
# ==========================================

def test_prompt_versions_stamped_across_full_graph(fake_llms):
    """Every LLM node contributes exactly one stamp (QW1), merged through
    the reducer channel: 'which prompt produced this run' is one lookup."""
    from agent.nodes.improver import (IMPROVER_PROMPT_VERSION,
                                      CONTEXT_WEAVE_PROMPT_VERSION)
    from agent.nodes.polisher import POLISHER_FULL_VERSION
    from agent.nodes.reviewer import REVIEWER_PROMPT_VERSION
    from agent.nodes.scanner import SCANNER_PROMPT_VERSION
    from agent.nodes.ats_check import ATS_SEMANTIC_PROMPT_VERSION
    from agent.nodes.cover_letter import COVER_LETTER_PROMPT_VERSION
    from agent.nodes.interview_prep import INTERVIEW_PREP_PROMPT_VERSION

    # The interactive graph carries the scanner; step 1 interrupts before
    # the improver, so the reviewer feedback drives the polish from there.
    config = {"configurable": {"thread_id": "pv-stamps-1"}}
    agent.agent_app.invoke(_base_inputs(), config=config)
    agent.agent_app.update_state(config, {"human_notes": "used k8s at work",
                                          "missing_skills": "- Kubernetes"})
    result = agent.agent_app.invoke(None, config=config)
    pv = result["prompt_versions"]
    assert pv["scanner"] == SCANNER_PROMPT_VERSION
    assert pv["improver"] == IMPROVER_PROMPT_VERSION
    assert pv["reviewer"] == REVIEWER_PROMPT_VERSION
    # The fake reviewer's feedback triggers exactly one polish pass; without
    # structured section targets the polisher runs its full-rewrite mode.
    assert pv["polisher"] == POLISHER_FULL_VERSION
    assert pv["ats-semantic"] == ATS_SEMANTIC_PROMPT_VERSION
    # Fan-out siblings must BOTH survive the concurrent writes.
    assert pv["cover-letter"] == COVER_LETTER_PROMPT_VERSION
    assert pv["interview-prep"] == INTERVIEW_PREP_PROMPT_VERSION
    # The dropped 'k8s' note triggers exactly ONE context weave attempt
    # (round 11: stamped on attempt; the attempt cap keeps the polisher's
    # full-rewrite pass from re-running it - merged beside the writer stamps).
    assert pv["context-weave"] == CONTEXT_WEAVE_PROMPT_VERSION


# ==========================================
# MR1: section utilities + section-targeted polish
# ==========================================

def test_split_markdown_sections_returns_canonical_blocks():
    md = ("# Sam\nCity | 555 | x@c.io\n\n"
          "## Professional Summary\nGeneric text.\n"
          "## Experience\nBuilt pipelines with Python.")
    blocks = agent.split_markdown_sections(md)
    assert [b[0] for b in blocks] == ["Professional Summary", "Experience"]
    assert blocks[0][1] == "## Professional Summary"
    assert blocks[0][2] == "Generic text."
    assert blocks[1][2] == "Built pipelines with Python."


def test_split_markdown_sections_flags_unknown_headers_and_skips_preamble():
    md = "## Projects\nDemo for ML.\n## Something Obscure\nKept as-is."
    blocks = agent.split_markdown_sections(md)
    assert [b[0] for b in blocks] == ["Projects", None]


def test_canonical_feedback_section_maps_aliases_and_rejects_crosscutting():
    assert agent.canonical_feedback_section(" Summary ") == "Professional Summary"
    assert agent.canonical_feedback_section("work experience") == "Experience"
    assert agent.canonical_feedback_section("Technical Skills") == "Technical Skills"
    assert agent.canonical_feedback_section("certificates") == "Certifications"
    assert agent.canonical_feedback_section("Formatting") is None
    assert agent.canonical_feedback_section("overall tone") is None
    assert agent.canonical_feedback_section("") is None


def test_merge_rewritten_sections_keeps_rest_byte_identical():
    md = ("# Sam\nCity | 555 | x@c.io\n"
          "## Professional Summary\nGeneric text.\n\n"
          "## Experience\nBuilt pipelines with Python.\n\n"
          "## Education\nUni | Degree.\n")
    summary = "Machine Learning engineer shipping Python services with Docker."
    merged, replaced = agent.merge_rewritten_sections(md, {"Professional Summary": summary})
    assert replaced == ["Professional Summary"]
    # The identity block, untouched sections and their separators survive
    # byte-identical; only the summary body changed.
    assert merged == (md.replace("Generic text.", summary))
    # Original header line survives (project title/link consistency).
    assert "## Professional Summary" in merged


def test_merge_rewritten_sections_preserves_header_and_separator():
    md = "## Projects\n**Demo** [Live Demo](https://x) | Tech | 2024\nBody bullet."
    merged, replaced = agent.merge_rewritten_sections(
        md, {"Projects": "Rewritten body only."})
    assert replaced == ["Projects"]
    # The merge keeps the ORIGINAL "## Projects" header line (byte-identical)
    # and swaps only the body; the caller owns what the new body contains.
    assert merged == "## Projects\nRewritten body only."
    assert "[Live Demo](https://x)" not in merged   # old body is gone with the rewrite


def test_polisher_section_gate_rejects_rewrites_that_drop_links():
    from agent.nodes.polisher import _rewrite_kept_links
    old = "**Demo** [Live Demo](https://x) [GitHub](https://g) | Tech\n- Bullet"
    assert _rewrite_kept_links(old, old)          # identity passes
    assert _rewrite_kept_links(
        old, "**Demo** [Live Demo](https://x) [GitHub](https://g) | Tech\n- Sharper phrasing")
    assert not _rewrite_kept_links(old, "Plain bullet with no links at all.")
    assert _rewrite_kept_links("", "New [link](https://n)")   # vacuous on empty original


def test_merge_rewritten_sections_refuses_partial_match():
    md = "## Skills\nPython.\n## Experience\nWork."
    merged, replaced = agent.merge_rewritten_sections(md, {"Education": "Uni."})
    assert merged == md
    assert replaced == []


def test_polisher_targets_reviewer_sections_and_merges_deterministically(fake_llms, monkeypatch):
    """Reviewer targets 'Summary' -> the polisher makes ONE small section
    call and code merges it; every other byte of the draft is unchanged."""
    draft = ("# Sam\nCity | x@c.io\n"
             "## Professional Summary\nGeneric summary text.\n"
             "## Experience\nBuilt Python services on Kubernetes with Docker.")
    reworked_summary = ("## Professional Summary\n"
                        "Machine Learning engineer shipping Python services "
                        "with Docker and Kubernetes.")
    result, seq = _run_with_drafts(
        [draft, reworked_summary], monkeypatch, reviewer_score=85,
        sections=["Summary"])

    assert seq.calls == 2                          # draft 1 + one section call
    assert result["optimized_resume"].startswith("# Sam\nCity | x@c.io\n")
    assert "Machine Learning engineer shipping Python services" in result["optimized_resume"]
    # Untouched section survives byte-identical.
    assert "## Experience\nBuilt Python services on Kubernetes with Docker." \
        in result["optimized_resume"]
    assert result["polish_sections_targeted"] == ["Professional Summary"]


def test_polisher_falls_back_to_full_rewrite_without_addressable_sections(fake_llms, monkeypatch):
    """Cross-cutting reviewer feedback ('Formatting') cannot be merged
    surgically -> the polisher reuses the improver's full-rewrite chain."""
    good = ("## Professional Summary\nFine.\n## Skills\nPython, Docker, Kubernetes\n"
            "## Experience\nBuilt Python services on Kubernetes with Docker.")
    polished = ("## Professional Summary\nSharper.\n## Skills\nPython, Docker, Kubernetes\n"
                "## Experience\nRebuilt Python services on Kubernetes with Docker.")
    result, seq = _run_with_drafts(
        [good, polished], monkeypatch, reviewer_score=85, sections=["Formatting"])

    assert seq.calls == 2                          # full rewrite, not a section call
    assert result["optimized_resume"] == polished


def test_polisher_full_rewrite_mode_is_stamped(fake_llms, monkeypatch):
    good = "## Skills\nPython\n## Experience\nBuilt Python services with Kubernetes."
    result, seq = _run_with_drafts(
        [good, "## Skills\nPython, Docker\n## Experience\nBuilt Python services with Docker."],
        monkeypatch)
    assert seq.calls == 2
    assert result["prompt_versions"]["polisher"] == "polisher-full@1"


def test_polisher_refused_merge_falls_back_to_full_rewrite(fake_llms, monkeypatch):
    """An unusable section answer (blank) refuses the merge; the pass is
    then completed by a full rewrite instead of being lost."""
    draft = "# Sam\n## Professional Summary\nGeneric.\n## Skills\nPython."
    polished = "# Sam\n## Professional Summary\nSharper.\n## Skills\nPython."
    result, seq = _run_with_drafts(
        [draft, "   ", polished], monkeypatch, reviewer_score=85,
        sections=["Summary"])

    assert seq.calls == 3                          # draft + failed target + full rewrite
    assert result["optimized_resume"] == polished
    assert result["prompt_versions"]["polisher"] == "polisher-full@1"


# ==========================================
# Integrity tripwire: unevidenced metric claims
# ==========================================

def test_find_unevidenced_metric_flags_invented_numbers():
    """Run 14751ff9: empty user context still produced 'cut ... by 40%'.
    Only metric tokens absent from the evidence are flagged; real ones pass."""
    evidence = "Built YOLO pipelines reaching 94.8% mAP. Cut latency by 40%."
    optimized = "Cut ingestion effort by 55%. Trained YOLO to 94.8%. Latency down 40%."
    assert agent.find_unevidenced_metric_claims(optimized, evidence) == ["55%"]


def test_find_unevidenced_metric_matches_spacing_variants():
    """LLM drafts spell '94.8 %' with a soft space and expect it to count as
    the same claim as the evidence's '94.8%' (compact comparison)."""
    evidence = "accuracy of 94.8%"
    optimized = "Reached 94.8 % accuracy. Achieved 3x throughput."
    assert agent.find_unevidenced_metric_claims(optimized, evidence) == ["3x"]


def test_find_unevidenced_metric_matches_exotic_spaces():
    """Run 868e406d: the draft spelled real WER numbers as '42.69 %' with a
    U+202F narrow no-break space while the source resume has plain '42.69%'.
    Only-ASCII-space compaction flagged the user's REAL numbers. Compaction
    must strip every whitespace class from the token."""
    evidence = "WER on Common Voice Arabic (WER 42.69%\u219220.61%,\u221251.7%)"
    optimized = ("\u202f42.69\u202f%\u202f \u2192 20.61\u202f%, reduced WER by 51.7\u202f%. "
                 "Invented 36% gain.")
    assert agent.find_unevidenced_metric_claims(optimized, evidence) == ["36%"]


def test_find_unevidenced_metric_ignores_years_and_versions():
    """Plain numerals without a metric unit (GPT-4, Python 3.12, years) are
    not metric claims and must never trigger a warning."""
    evidence = "Python since 2019."
    optimized = "GPT-4, Python 3.12, on-device since 2026, 1.7 ms inference."
    assert agent.find_unevidenced_metric_claims(optimized, evidence) == ["1.7 ms"]


def test_find_unevidenced_metric_empty_inputs():
    assert agent.find_unevidenced_metric_claims("", "94.8%") == []
    assert agent.find_unevidenced_metric_claims("Cut by 55%", "") == ["55%"]


def test_find_unevidenced_metric_rounding_tolerance():
    """Nov 14 2026 ('96 %' survivor): typed numbers compared as text only
    flagged the cover letter's legitimate rounding - the letter cited
    '96 %' where the resume says '96.2%'. A claim that is the rounding or
    truncation of an evidence value at the claim's precision is evidenced;
    drift, fabrication, wrong values, and ratios still flag."""
    evidence = ("Cut latency 96.2% via caching. Trimmed spend by 44.6%. "
                "Speed 2.6x faster. p95 latency 380 ms.")
    # legitimate roundings -> silent
    assert agent.find_unevidenced_metric_claims(
        "Cut latency 96 % via caching.", evidence) == []
    assert agent.find_unevidenced_metric_claims(
        "Trimmed spend by 45%.", evidence) == []
    assert agent.find_unevidenced_metric_claims(
        "Speed 3x faster.", evidence) == []
    # drift / fabrication / wrong value / ratio -> still flagged
    assert agent.find_unevidenced_metric_claims(
        "p95 latency 400 ms.", evidence) == ["400 ms"]
    assert agent.find_unevidenced_metric_claims(
        "Reduced manual processing by 70%.", evidence) == ["70%"]
    assert agent.find_unevidenced_metric_claims(
        "Trimmed spend by 98%.", evidence) == ["98%"]
    assert agent.find_unevidenced_metric_claims(
        "Passed 150/150 benchmark tests.", "Passed 149/150 benchmark tests."
    ) == ["150/150"]


def test_find_unevidenced_metric_rounding_tolerance_unit_must_match():
    """96 % tolerates 96.2%; it does not tolerate a different-unit metric
    that happens to share the number."""
    assert agent.find_unevidenced_metric_claims(
        "Cut 96% of waste.", "Measured 96.2 ms per request.") == ["96%"]


def test_unevidenced_metric_tokens_rounding_tolerance():
    """The scrubber's per-line scan shares the tolerance: a rounded claim
    survives the scrub, a drifted one does not."""
    assert agent.helpers._unevidenced_metric_tokens(
        "improved throughput 96 %", ["improved throughput 96.2%"]) == []
    assert agent.helpers._unevidenced_metric_tokens(
        "improved latency 400 ms", ["improved latency 380 ms"]) == ["400ms"]


def test_find_unevidenced_metric_catches_fraction_pass_rates():
    """Nov 7 run pair (1d3e1550): '150/150 benchmark passes' is a claim a
    model can coin without a %/x/ms unit - it must be flagged when the
    evidence lacks it, and must stay quiet when the evidence carries it."""
    idle = ("- Implemented a scope filter achieving 150/150 benchmark passes. "
            "We ranked 5/47 overall.")
    assert agent.find_unevidenced_metric_claims(idle, "") == ["150/150"]
    evidenced = "- Achieved 150/150 benchmark passes across 189 tests."
    assert agent.find_unevidenced_metric_claims(
        evidenced, "user context: achieved 150/150 benchmark passes") == []


def test_improver_scrubs_and_reports_via_guardrail_warnings(monkeypatch):
    """Runs 14751ff9/a32f30fc/868e406d made 'detect only' insufficient: the
    reviewer's removal feedback was ignored twice. The improver now removes
    the unsupported content itself and reports what it removed (plus any
    metric leftovers) via guardrail_warnings, which the server relays as a
    warning SSE event before the result."""
    from agent.nodes.improver import improver_node

    state = {
        "iteration": 0,
        "optimized_resume": "",          # empty -> base_content = resume_text
        "resume_text": "# Omar\n## Experience\nBuilt pipelines, cutting latency by 40%.",
        "job_text": "ML platform engineer role.",
        "human_notes": "",
        "missing_skills": "- Kubernetes",
        "feedback": None,
    }

    class InventingLLM(FakeCreativeLLM):
        def invoke(self, messages):
            return AIMessage(content="# Omar\n## Experience\n"
                                     "- Wrote Kubernetes operators for nightly jobs.\n"
                                     "- Cut latency by 40%.\n"
                                     "- Cut manual ingestion effort by 55%.")

    monkeypatch.setattr(agent.llms, "llm_creative", InventingLLM())
    out = improver_node(state)
    assert out["iteration"] == 1
    joined = "; ".join(out["guardrail_warnings"])
    # the fabricated Kubernetes echo bullet and the 55% claim are REMOVED
    assert "kubernetes" in joined.lower()
    assert "55%" in joined
    assert "Kubernetes" not in out["optimized_resume"]
    assert "55%" not in out["optimized_resume"]
    assert "40%" in out["optimized_resume"]          # evidenced number stays


def test_improver_never_ships_fully_scrubbed_empty_draft(monkeypatch):
    """Pipeline continuity: when every removal would empty the draft, the
    pre-scrub draft is kept and the user is warned (never silently dropped,
    never silently shipped as clean)."""
    from agent.nodes.improver import improver_node

    state = {
        "iteration": 0,
        "optimized_resume": "",
        "resume_text": "# Omar\n## Experience",
        "job_text": "ML platform engineer role.",
        "human_notes": "",
        "missing_skills": "- Kubernetes",
        "feedback": None,
    }

    class FullyFabricatedLLM(FakeCreativeLLM):
        def invoke(self, messages):
            return AIMessage(content="# Omar\n## Experience\n"
                                     "- Cut manual ingestion effort by 55%.")

    monkeypatch.setattr(agent.llms, "llm_creative", FullyFabricatedLLM())
    out = improver_node(state)
    assert out["optimized_resume"].strip()
    assert any("40%" in w or "55%" in w or "kept draft" in w
               for w in out["guardrail_warnings"])


def test_improver_decline_protocol_records_and_strips(monkeypatch):
    """QW2: the writer answers unsupported requests with DECLINED: lines that
    are stripped from the draft and surfaced as declined_requests, so an
    omission is never silent (and never fabricated)."""
    from agent.nodes.improver import improver_node, IMPROVER_PROMPT_VERSION

    state = {
        "iteration": 0,
        "optimized_resume": "",
        "resume_text": "# Omar\n## Experience\nBuilt pipelines, cutting latency by 40%.",
        "job_text": "ML platform engineer role.",
        "human_notes": "",
        "missing_skills": "- Kubernetes",
        "feedback": None,
    }

    class DecliningLLM(FakeCreativeLLM):
        def invoke(self, messages):
            prompt = messages[-1].content
            assert "DECLINE PROTOCOL" in prompt     # the clause ships with the prompt
            return AIMessage(content="# Omar\n## Experience\n"
                                     "- Built pipelines, cutting latency by 40%.\n\n"
                                     "DECLINED: Experience with n8n\n"
                                     "DECLINED: 3x throughput claims\n")

    monkeypatch.setattr(agent.llms, "llm_creative", DecliningLLM())
    out = improver_node(state)
    assert "DECLINED" not in out["optimized_resume"]
    assert "40%" in out["optimized_resume"]
    assert out["declined_requests"] == ["Experience with n8n", "3x throughput claims"]
    assert out["prompt_versions"] == {"improver": IMPROVER_PROMPT_VERSION}


# ==========================================
# BB1 Phase B: typed resume units + code-assembled delivery
# ==========================================

def test_parse_resume_units_pins_header_and_canonicalizes_marks():
    units = agent.parse_resume_units(
        "# Jane Roe\nCity | j@x.io\n\nSUMMARY\nOld summary.\n\n"
        "EXPERIENCE\nBuilt things.")
    assert not units.whole_document
    assert units.header_block == "# Jane Roe\nCity | j@x.io"
    assert [u.id for u in units.sections] == ["Professional Summary",
                                              "Experience"]
    assert units.sections[1].heading_line == "EXPERIENCE"
    assert units.sections[1].body == "Built things."


def test_parse_resume_units_markdown_draft_keeps_headings_byte_identical():
    units = agent.parse_resume_units(
        "## Skills\nPython, Docker\n## Experience\nBuilt ML pipelines.")
    assert units.header_block == ""
    assert [u.id for u in units.sections] == ["Technical Skills", "Experience"]
    assert units.sections[0].heading_line == "## Skills"


def test_parse_resume_units_whole_document_when_source_has_no_marks():
    units = agent.parse_resume_units("Jane Roe\nPython developer.")
    assert units.whole_document
    assert units.sections[0].id == "Resume"
    assert units.sections[0].body == "Jane Roe\nPython developer."


def test_assemble_rewrites_in_place_skips_verbatim_adds_missing():
    units = agent.parse_resume_units(
        "# Jane\nCity | j@x.io\nSUMMARY\nOld summary.\nEXPERIENCE\nBuilt things.")
    answer = agent.ImproverOutput(sections=[
        agent.ImproverSection(id="Professional Summary", body="New summary.")])
    text, warns, declined = agent.assemble_resume_units(units, answer)
    # The identity header is pinned verbatim and the skipped unit ships
    # byte-stable from the source (normalized heading for a raw ALL-CAPS one).
    assert text.startswith("# Jane\nCity | j@x.io")
    assert "## Professional Summary\nNew summary." in text
    assert "## Experience\nBuilt things." in text
    assert warns == [] and declined == []


def test_assemble_drops_unknown_ids_with_guardrail_note():
    units = agent.parse_resume_units("## Skills\nPython")
    answer = agent.ImproverOutput(sections=[
        agent.ImproverSection(id="Awards", body="Made-up award")])
    text, warns, _ = agent.assemble_resume_units(units, answer)
    assert "Made-up award" not in text and "## Awards" not in text
    # A markdown source heading ships byte-verbatim (byte-stability for
    # untouched units); only ALL-CAPS raw headings normalize to '## <id>'.
    assert text == "## Skills\nPython"
    assert any("unknown section id 'Awards'" in w for w in warns)


def test_assemble_refuses_rewrites_that_drop_links_or_ship_empty():
    units = agent.parse_resume_units(
        "## Projects\n**Aegis** [GitHub](https://x.io/aegis)\n- Built it.")
    empty = agent.ImproverOutput(sections=[
        agent.ImproverSection(id="Projects", body="")])
    text, warns, _ = agent.assemble_resume_units(units, empty)
    assert "https://x.io/aegis" in text
    assert any("empty rewrite for 'Projects'" in w for w in warns)

    linkless = agent.ImproverOutput(sections=[
        agent.ImproverSection(id="Projects", body="- Built it without links.")])
    text2, warns2, _ = agent.assemble_resume_units(units, linkless)
    assert "https://x.io/aegis" in text2
    assert any("dropped a source link" in w for w in warns2)


def test_assemble_whole_document_replacement_and_outsider_drop():
    units = agent.parse_resume_units("Jane Roe\nPython developer.")
    answer = agent.ImproverOutput(sections=[
        agent.ImproverSection(
            id="Resume", body="# Jane Roe\nj@x.io\n## Skills\nPython")])
    text, warns, _ = agent.assemble_resume_units(units, answer)
    assert text.startswith("# Jane Roe")
    assert "Python developer." not in text       # replaced wholesale
    assert warns == []

    outsider = agent.ImproverOutput(sections=[
        agent.ImproverSection(id="Skills", body="Python")])
    text2, warns2, _ = agent.assemble_resume_units(units, outsider)
    assert text2 == "Jane Roe\nPython developer."
    assert any("outside the 'Resume' whole-document unit" in w for w in warns2)


def test_improver_output_from_markdown_modes_and_declined():
    out = agent.improver_output_from_markdown(
        "# Jane\n## Skills\nPython\n\nDECLINED: n8n")
    assert [s.id for s in out.sections] == ["Skills"]
    assert out.sections[0].body == "Python"
    assert out.declined == ["n8n"]

    out = agent.improver_output_from_markdown(
        "# Jane\n## Skills\nPython", whole_document=True)
    assert [s.id for s in out.sections] == ["Resume"]
    assert out.sections[0].body == "# Jane\n## Skills\nPython"

    out = agent.improver_output_from_markdown("plain prose, no headings")
    assert [s.id for s in out.sections] == ["Resume"]
    assert out.sections[0].body == "plain prose, no headings"


def _improver_state(resume_text: str, **overrides) -> dict:
    state = {
        "iteration": 0,
        "optimized_resume": "",          # empty -> base_content = resume_text
        "resume_text": resume_text,
        "job_text": "ML platform engineer role.",
        "human_notes": "",
        "missing_skills": "- Kubernetes",
        "feedback": None,
    }
    state.update(overrides)
    return state


def test_improver_writer_outage_falls_back_to_source_verbatim(monkeypatch):
    """BB1 Phase B: a total writer failure degrades to the source resume
    verbatim with a loud integrity warning instead of killing the run."""
    from agent.nodes.improver import improver_node

    class BrokenCreative:
        def invoke(self, messages):
            raise RuntimeError("provider outage")

        def with_structured_output(self, schema):
            raise RuntimeError("provider outage")

    monkeypatch.setattr(agent.llms, "llm_creative", BrokenCreative())
    out = improver_node(_improver_state(
        "Jane Roe\nPython developer with cloud exposure."))
    assert out["iteration"] == 1
    assert out["optimized_resume"].strip() == \
        "Jane Roe\nPython developer with cloud exposure."
    warns = "; ".join(out["guardrail_warnings"])
    assert "provider outage" in warns
    assert "shipped the source resume verbatim" in warns


def test_improver_unknown_unit_ids_are_dropped_and_noted(monkeypatch):
    """The trusting-nothing guarantee: the writer cannot invent a section
    by naming it - unknown ids are dropped and surfaced in the integrity
    warning, while the source units ship verbatim."""
    from agent.nodes.improver import improver_node

    class UnknownIdLLM(FakeCreativeLLM):
        def invoke(self, messages):
            return AIMessage(content="## Awards\nMade-up award")

    monkeypatch.setattr(agent.llms, "llm_creative", UnknownIdLLM())
    # MARKED source: the converter then emits id "Awards" for the fake's
    # markdown. (A heading-less source would whole-document it into one
    # Resume replacement, which the hallucination filter then empties -
    # that defense-in-depth belt has this assembly drop as its suspenders.)
    out = improver_node(_improver_state(
        "# Jane\n## Experience\nBuilt pipelines, cutting latency by 40%."))
    final = out["optimized_resume"]
    assert "## Experience\nBuilt pipelines, cutting latency by 40%." in final
    assert "Made-up award" not in final and "Awards" not in final
    assert any("unknown section id 'Awards'" in w
               for w in out["guardrail_warnings"])


def test_improver_structured_declined_list_feeds_the_channel(monkeypatch):
    """QW2 in the structured shape: the writer's `declined` entries ride
    the same declined_requests channel the DECLINED: lines used."""
    from agent.nodes.improver import improver_node, IMPROVER_PROMPT_VERSION

    class StructuredDecliner:
        def invoke(self, messages):
            raise AssertionError("plain invoke must not be used")

        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ImproverOutput(
                        sections=[agent.ImproverSection(
                            id="Experience",
                            body="- Built pipelines, cutting latency by 40%.")],
                        declined=["Experience with n8n"])

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_creative", StructuredDecliner())
    out = improver_node(_improver_state(
        "# Jane\n## Experience\nBuilt pipelines, cutting latency by 40%."))
    assert out["declined_requests"] == ["Experience with n8n"]
    assert "n8n" not in out["optimized_resume"]
    assert "40%" in out["optimized_resume"]
    assert out["prompt_versions"] == {"improver": IMPROVER_PROMPT_VERSION}


def test_extract_declined_requests_variants():
    text, declined = agent.extract_declined_requests(
        "# Name\n## Skills\nPython\n\nDECLINED: Terraform basics\nDECLINED: Terraform basics")
    assert declined == ["Terraform basics"]          # deduplicated
    assert text.strip() == "# Name\n## Skills\nPython"
    untouched, empty = agent.extract_declined_requests("# Name\n## Skills\nPython")
    assert empty == [] and untouched == "# Name\n## Skills\nPython"

def test_extract_declined_requests_strips_mid_document_lines():
    cleaned, declined = agent.extract_declined_requests(
        "## Experience\n- Job one.\nDECLINED: Docker certification\n- Job two.")
    assert declined == ["Docker certification"]
    assert cleaned == "## Experience\n- Job one.\n- Job two."


# ==========================================
# Deterministic fabrication scrubber
# ==========================================

def test_scrubber_drops_unevidenced_chip_echo_bullets():
    """Run 868e406d: kept chips composed under pasted project text became
    fake Aegis bullets. Bullets whose chip terms the evidence lacks go."""
    draft = (
        "## Projects\n"
        "**Aegis** | FastAPI | Aug - Sep 2026\n"
        "- Engineered a FastAPI service around a 4-node LangGraph agent.\n"
        "- Gained experience with n8n (or similar workflow automation platform).\n"
        "- Hands-on work using Claude Code, Cursor, and Codex for building AI agents and demos.\n"
    )
    scrubbed, removals = agent.scrub_unsupported_content(
        draft, resume_text="A FastAPI + LangGraph service.", human_notes="",
        missing_skills_text="- Experience with n8n (or similar workflow automation platform)\n"
                            "- Hands-on work with Claude Code, Cursor, or Codex for building AI agents and demos")
    assert "n8n" not in scrubbed
    assert "Claude Code" not in scrubbed
    assert "FastAPI service around a 4-node LangGraph agent" in scrubbed
    assert any("n8n" in r for r in removals)


def test_scrubber_removes_chip_claims_from_skills_rows_and_prose():
    """Skills rows lose only the offending ITEM; prose loses the offending
    SENTENCE; '(or similar)' hedges never survive."""
    row = "**Automation & Workflow**: n8n (or similar), Webhooks, JSON-based API integrations"
    scrubbed, removals = agent.scrub_unsupported_content(
        row, resume_text="Python, SQL, webhooks.", human_notes="",
        missing_skills_text="- Experience with n8n (or similar workflow automation platform)")
    assert "n8n" not in scrubbed
    assert "(or similar)" not in scrubbed
    assert "Webhooks" in scrubbed and "JSON-based API integrations" in scrubbed
    assert any("skills item" in r for r in removals)

    prose = "AI engineer with FastAPI experience. Acquired production proficiency with n8n."
    scrubbed2, removals2 = agent.scrub_unsupported_content(
        prose, resume_text="FastAPI services.", human_notes="", missing_skills_text="- n8n")
    assert "Acquired production proficiency" not in scrubbed2
    assert "FastAPI experience" in scrubbed2
    assert any("prose" in r for r in removals2)


def test_scrubber_clauses_out_unevidenced_numbers_but_keeps_evidence():
    """Run 14751ff9: 'cut manual data-ingestion effort by 40%' with an empty
    context. Parenthesized and comma-segment claims are excised without
    losing the surrounding evidenced text; unexcisable bullets go whole."""
    bullet_paren = ("- Implemented a pre-LLM injection filter (< 15 ms), achieving "
                    "189 passing tests.\n")
    scrubbed, _ = agent.scrub_unsupported_content(
        bullet_paren, resume_text="189 tests passing in CI.", human_notes="",
        missing_skills_text="")
    assert "< 15 ms" not in scrubbed
    assert "injection filter" in scrubbed and "189 passing tests" in scrubbed

    bullet_whole = "- Cut manual data-ingestion effort by 40%.\n"
    scrubbed2, removals2 = agent.scrub_unsupported_content(
        bullet_whole, resume_text="", human_notes="", missing_skills_text="")
    assert "40%" not in scrubbed2
    assert any("40%" in r for r in removals2)


def test_scrubber_honors_do_not_feature_and_evidenced_chips():
    """Removed chips die even when evidenced; kept chips survive exactly
    when the evidence demonstrates them."""
    removed_chip = "Do NOT feature: Docker."
    dropped, removals = agent.scrub_unsupported_content(
        "- Deployed Docker containers for the demo.\n",
        resume_text="Deployed Docker containers for the demo.",
        human_notes="", missing_skills_text=removed_chip)
    assert "Docker" not in dropped
    assert removals  # the user's explicit suppression is reported

    kept_evidenced = "- Experience demonstrated with FastAPI service work.\n"
    kept, removals2 = agent.scrub_unsupported_content(
        kept_evidenced, resume_text="", human_notes="Used FastAPI for two services.",
        missing_skills_text="- Experience with FastAPI (or similar)")
    assert "FastAPI" in kept
    assert not removals2


def test_scrubber_stale_chip_lines_in_notes_are_not_evidence():
    """Request 539f7c1d, Nov 8 2026: stale chips from a PREVIOUS session were
    stamped into human_notes under the pasted Aegis evidence. Notes are
    EVIDENCE to this scrubber by contract, so the unresolved '- '-bulleted
    chip lines 'verified' their own fabricated echoes -> shipped verbatim
    (ATS then matched the JD's own vocabulary at 100%). Belt-and-braces:
    the notes-evidence corpus drops ASCII-bullet lines (the extension's
    chip serialization; typed evidence uses '•' bullets or prose), so the
    poisoned notes behave exactly like clean ones."""
    chips = ("- Experience with n8n (or similar workflow automation platform)\n"
             "- Hands-on work with Claude Code, Cursor, or Codex for building AI agents and demos\n"
             "- Experience working with MCP servers (FortyGuard Temperature MCP)\n")
    stale = ("- Experience with n8n (or similar workflow-automation platform) and building automations using it\n"
             "- Hands-on work with Claude Code, Cursor, Codex or comparable LLM-code generation tools\n"
             "- Direct experience working with MCP servers (FortyGuard Temperature MCP) or similar server environments\n")
    typed = ("here is a project that i built:\n"
             "Aegis: Autonomous Heat-Risk Operations Agent Aug-Sep 2026\n"
             "• Engineered a FastAPI service with bounded LLM timeouts.\n")
    draft = (
        "# Jane Doe\n"
        "## Technical Skills\n"
        "- n8n (workflow automation), Webhooks\n"
        "- Claude Code, Cursor, Codex\n"
        "- MCP servers (FortyGuard Temperature MCP)\n"
        "- Built real FastAPI services.\n"
    )
    poisoned = agent.scrub_unsupported_content(
        draft, resume_text="Built real FastAPI services.",
        human_notes=typed + stale,
        missing_skills_text=chips)
    clean = agent.scrub_unsupported_content(
        draft, resume_text="Built real FastAPI services.",
        human_notes=typed,
        missing_skills_text=chips)
    for scrubbed, removals, label in ((poisoned[0], poisoned[1], "poisoned"),
                                      (clean[0], clean[1], "clean")):
        assert "n8n" not in scrubbed, label
        assert "Claude Code" not in scrubbed, label
        assert "FortyGuard" not in scrubbed, label
        assert "Built real FastAPI services." in scrubbed, label
        assert removals, f"{label}: expected removals"
    # the poisoned notes must NOT make the fabricated echoes look evidenced:
    # same outcome as clean notes (chip lines in notes = no protective power)
    assert sorted(poisoned[1]) == sorted(clean[1])


def test_scrubber_keeps_evidenced_numbers():
    """Narrow-space copies of real numbers (run 868e406d: '42.69 %' vs the
    resume's '42.69%') must survive."""
    draft = "- WER on Common Voice Arabic improved from 42.69 % to 20.61 %.\n"
    scrubbed, removals = agent.scrub_unsupported_content(
        draft, resume_text="WER on Common Voice Arabic: 42.69%->20.61%",
        human_notes="", missing_skills_text="")
    assert "42.69" in scrubbed and "20.61" in scrubbed
    assert not removals


def test_scrubber_removes_fully_fabricated_draft():
    """A draft made ONLY of an unsupported claim comes back empty, with the
    removal reported."""
    scrubbed, removals = agent.scrub_unsupported_content(
        "- Cut ingestion effort by 40%.\n", resume_text="", human_notes="",
        missing_skills_text="")
    assert "40%" not in scrubbed
    assert any("40%" in r for r in removals)


def test_scrubber_dedups_blanks_and_reports_cap():
    many = ("- Built real FastAPI endpoints.\n"
            + "".join(f"- Gained experience with n8n item{i}.\n" for i in range(12)))
    scrubbed, removals = agent.scrub_unsupported_content(
        many, resume_text="- Built real FastAPI endpoints.", human_notes="",
        missing_skills_text="- Experience with n8n (or similar)")
    assert "Built real FastAPI endpoints" in scrubbed
    assert "\n\n\n" not in scrubbed
    assert len(removals) <= 11


# ==========================================
# Chip-term precision (Nov 7 run pair d8756ddf/a9da3660)
# ==========================================
# The Nov 7 run pair's real scanner chips carried plain-word filler ('using',
# 'agent', 'development', 'workflow-automation', 'environments') that the
# old term heuristic turned into removal triggers: run 09cf51fc culled a
# draft bullet on the term 'using' alone. Chip terms are now TECHNOLOGY
# NAMES - capitalized or digit-bearing tokens; consecutive names inside one
# segment join into a single multi-word term ('Claude Code') so pieces like
# 'code' cannot hit innocent bullets ("code coverage").

def test_chip_terms_extract_technology_names_from_real_run_chips():
    import agent.helpers as helpers
    a1 = "- Experience with n8n (or similar workflow-automation platform) and building n8n automations"
    a2 = ("- Hands-on work using Claude Code, Cursor, Codex, or comparable "
          "LLM-code generation tools for AI agent development")
    b3 = ("- Direct experience working with MCP servers (FortyGuard Temperature MCP) "
          "or similar server environments\u200b")
    assert helpers._significant_terms(a1) == ["n8n"]
    assert helpers._significant_terms(a2) == ["claude code", "cursor", "codex", "llm-code"]
    assert helpers._significant_terms(b3) == ["mcp", "fortyguard temperature mcp"]
    # 'apis' is effectively generic; 'JSON' is a name.
    assert helpers._significant_terms(
        "- Practical experience with APIs, webhooks, and JSON data handling") == ["json"]
    # lowercase-only chips yield no terms (missing a cull beats a wrong one)
    assert helpers._significant_terms("- docker and kubernetes experience") == []


def test_scrubber_spares_innocent_bullets_on_chip_filler_terms():
    """Run 09cf51fc dropped a draft bullet on the term 'using'. Bullets that
    merely say 'using X' or read like chip filler must survive."""
    chips = ("- Hands-on work using Claude Code, Cursor, Codex, or comparable "
             "LLM-code generation tools for AI agent development\n"
             "- Experience with n8n (or similar workflow-automation platform) "
             "and building n8n automations")
    innocent = ("- Reduced manual payroll export effort using OR-Tools scheduling.\n"
                "- Increased code coverage on the ingest pipeline.\n")
    scrubbed, removals = agent.scrub_unsupported_content(
        innocent, resume_text="OR-Tools scheduling and coverage analysis.",
        human_notes="", missing_skills_text=chips)
    assert "OR-Tools scheduling" in scrubbed
    assert "code coverage" in scrubbed
    assert not removals


def test_scrubber_still_culls_true_multiname_chip_echoes():
    chips = ("- Hands-on work using Claude Code, Cursor, Codex, or comparable "
             "LLM-code generation tools for AI agent development\n"
             "- Direct experience working with MCP servers (FortyGuard "
             "Temperature MCP) or similar server environments")
    guilty = ("- Integrated FortyGuard Temperature MCP into the nightly automation.\n"
              "- Practiced MCP server responses against mock data.\n"
              "- Used Claude Code to scaffold the CLI agent.\n")
    scrubbed, removals = agent.scrub_unsupported_content(
        guilty, resume_text="", human_notes="", missing_skills_text=chips)
    assert "FortyGuard" not in scrubbed
    assert "MCP" not in scrubbed
    assert "Claude Code" not in scrubbed
    assert len(removals) >= 2


# ==========================================
# Bullet Unicode hygiene (live run 14751ff9)
# ==========================================

def test_tame_unicode_bullets_normalizes_separators():
    """Scanner bullets from web-sourced JD text carried U+202F and U+2011
    ('Claude\u202fCode', 'Hands\u2011on') which broke chip parsing downstream."""
    raw = "- Claude\u202fCode\r\n- Hands\u2011on MCP  \n- NBSP\u00a0item"
    assert agent._tame_unicode_bullets(raw) == "- Claude Code\n- Hands-on MCP\n- NBSP item"


def test_tame_unicode_bullets_untouched_input():
    plain = "- Kubernetes\n- Terraform "
    assert agent._tame_unicode_bullets(plain) == "- Kubernetes\n- Terraform"


def test_tame_unicode_bullets_canonizes_percent_spacing():
    """Run 28e2e56f (Nov 12 2026): one resume printed '62 %', '70 % to
    97.2 %' AND '47%', '98.6%' - LLM source-text variance. Digit-adjacent
    '62%' is the dominant form, and whitespace before % is never
    meaningful in resume prose."""
    assert agent._tame_unicode_bullets(
        "cut costs by 62 % versus a 70 % baseline") == \
        "cut costs by 62% versus a 70% baseline"
    assert agent._tame_unicode_bullets(
        "from 70 % to 97.2 % and 47% accuracy") == \
        "from 70% to 97.2% and 47% accuracy"
    assert agent._tame_unicode_bullets("share of % growth") == "share of % growth"


def test_preserve_contact_links_restores_links_the_rewrite_dropped():
    """Run 124f40cb (Nov 12 2026): the source header carried the HF profile
    only as a bare label inside a markdown link whose href pointed at the
    LinkedIn URL; the improver's rewrite dropped it (3 of 4 stored runs
    kept it) and shipped a dangling trailing pipe. Contact links are
    identity - the deterministic guard re-injects any source-header link
    the draft's header lost, cleans the separator, and is idempotent."""
    source = (
        "Omar Mashaly\n"
        "AI/ML Engineer\n"
        "[Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com |]"
        "(https://linkedin.com/in/omar-mashaly)\n"
        "[linkedin.com/in/omar-mashaly | github.com/Omar10lfc | "
        "huggingface.co/Omar10lfc](https://linkedin.com/in/omar-mashaly)\n"
        "SUMMARY\n"
        "Experienced AI/ML engineer.\n"
    )
    draft = (
        "# Omar Mashaly\n"
        "Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com | "
        "https://linkedin.com/in/omar-mashaly | https://github.com/Omar10lfc |\n"
        "\n"
        "## Professional Summary\n"
        "Experienced AI/ML engineer.\n"
    )
    fixed, restored, placeholders, normalized = agent.preserve_contact_links(
        draft, source)
    assert restored == ["huggingface.co/Omar10lfc"]
    assert placeholders == [] and normalized == []
    assert "https://huggingface.co/Omar10lfc" in fixed
    # the dangling separator is gone and the link sits at the line's end
    assert "github.com/Omar10lfc | https://huggingface.co/Omar10lfc" in fixed
    assert not fixed.splitlines()[1].rstrip().endswith("|")
    # idempotent: a second pass restores nothing and changes nothing
    fixed2, restored2, placeholders2, normalized2 = \
        agent.preserve_contact_links(fixed, source)
    assert restored2 == [] and placeholders2 == [] and fixed2 == fixed
    assert normalized2 == []
    # a complete draft is untouched
    complete = draft.replace(
        "https://github.com/Omar10lfc |",
        "https://github.com/Omar10lfc | https://huggingface.co/Omar10lfc")
    same, restored3, placeholders3, normalized3 = agent.preserve_contact_links(
        complete, source)
    assert restored3 == [] and placeholders3 == [] and same == complete
    assert normalized3 == []
    # an email domain inside the source header can never be re-injected
    assert "| gmail.com" not in fixed and "https://gmail.com" not in fixed


def test_preserve_contact_links_strips_placeholder_tokens():
    """Run bf0f7941 (Nov 12 2026): the writer's draft emitted a literal
    "N/A" placeholder in the contact line where a field had no value, and
    the link restoration then appended right after it - the PDF printed
    "...github.com/Omar10lfc | N/A | huggingface.co/Omar10lfc". Placeholder
    tokens are fake fields: stripped before restoration, surfaced in the
    return, idempotent, and a line keeps at least two tokens."""
    source = (
        "Omar Mashaly\n"
        "[linkedin.com/in/omar | github.com/omar | huggingface.co/omar]"
        "(https://linkedin.com/in/omar)\n"
        "SUMMARY\n"
        "AI engineer.\n"
    )
    draft = (
        "# Omar Mashaly\n"
        "Alexandria, Egypt | omarmashaly86@gmail.com | "
        "https://linkedin.com/in/omar | https://github.com/omar | N/A |\n"
        "\n"
        "## Professional Summary\n"
        "AI engineer.\n"
    )
    fixed, restored, removed, normalized = agent.preserve_contact_links(
        draft, source)
    assert removed == ["N/A"]
    assert restored == ["huggingface.co/omar"]
    assert normalized == []
    assert " | N/A" not in fixed and "N/A |" not in fixed
    # the restored link lands directly after the last real token
    assert "https://github.com/omar | https://huggingface.co/omar" in fixed
    # idempotent
    fixed2, restored2, removed2, normalized2 = agent.preserve_contact_links(
        fixed, source)
    assert fixed2 == fixed and restored2 == [] and removed2 == []
    assert normalized2 == []
    # a degenerate line ("Name | N/A") is never stripped below two tokens
    thin = "# Jane Roe\nJane Roe | N/A\n\n## Summary\nx\n"
    same, restored4, removed4, normalized4 = agent.preserve_contact_links(
        thin, "Jane Roe\nSUMMARY\nx\n")
    assert same == thin and restored4 == [] and removed4 == []
    assert normalized4 == []


def test_preserve_contact_links_normalizes_label_links_to_bare_urls():
    """Round 7 (runs 25ff4ae5 / 1a8125f0, Nov 12 2026): both drafts emitted
    label-form markdown links in the contact line and the renderer printed
    the LABELS - the PDF showed the word "Portfolio" where every accepted
    run displays huggingface.co/Omar10lfc. The contact line's [label](url)
    tokens are unwrapped to bare URLs deterministically; body links
    (project [Live Demo](url)) are content and stay untouched."""
    source = (
        "Omar Mashaly\n"
        "[linkedin.com/in/omar | github.com/omar | huggingface.co/omar]"
        "(https://linkedin.com/in/omar)\n"
        "SUMMARY\n"
        "AI engineer.\n"
    )
    draft = (
        "# Omar Mashaly\n"
        "Alexandria, Egypt | omarmashaly86@gmail.com | "
        "[LinkedIn](https://linkedin.com/in/omar) | "
        "[GitHub](https://github.com/omar) | "
        "[Portfolio](https://huggingface.co/omar)\n"
        "\n"
        "## Professional Summary\n"
        "**AI Resume Optimizer** [Live Demo](https://demo.example) | x\n"
    )
    fixed, restored, removed, normalized = agent.preserve_contact_links(
        draft, source)
    assert normalized == ["linkedin.com/in/omar", "github.com/omar",
                          "huggingface.co/omar"]
    line = fixed.splitlines()[1]
    assert "[LinkedIn]" not in line and "[Portfolio]" not in line
    assert line.endswith("https://huggingface.co/omar")
    assert ("https://linkedin.com/in/omar | https://github.com/omar"
            in line)
    # body markdown links are content and survive untouched
    assert "[Live Demo](https://demo.example)" in fixed
    assert restored == [] and removed == []
    # idempotent: a bare-URL contact line normalizes nothing
    fixed2, restored2, removed2, normalized2 = agent.preserve_contact_links(
        fixed, source)
    assert fixed2 == fixed and normalized2 == []
    assert restored2 == [] and removed2 == []


def test_improver_postprocess_restores_dropped_contact_links():
    """The guard rides the shared writer tail (_postprocess_draft), so both
    the improver and the polisher passes restore identity links, strip
    placeholder tokens, and surface dropped source evidence in the
    guardrail channel."""
    source = (
        "Jane Roe\n"
        "[linkedin.com/in/jane | github.com/jane | huggingface.co/jane]"
        "(https://linkedin.com/in/jane)\n"
        "SUMMARY\n"
        "Python developer.\n"
        "- Cut inference cost 40% with response caching\n"
    )
    state = {"resume_text": source, "human_notes": "", "missing_skills": "",
             "iteration": 1, "scrub_removals": [], "unevidenced_metrics": [],
             "declined_requests": []}
    draft = ("# Jane Roe\n"
             "https://linkedin.com/in/jane | https://github.com/jane | N/A\n"
             "\n"
             "## Summary\n"
             "Python developer.\n")
    out = agent.nodes.improver._postprocess_draft(state, draft, 1)
    body = out["optimized_resume"]
    assert "https://huggingface.co/jane" in body
    assert " | N/A" not in body and "N/A" not in body.splitlines()[1]
    assert "40%" not in body  # the draft dropped the source metric
    assert any("restored dropped contact link: huggingface.co/jane" in w
               for w in out["guardrail_warnings"])
    assert any("removed placeholder token from the contact line: N/A" in w
               for w in out["guardrail_warnings"])
    assert any("source evidence metric(s) 40% no longer in the resume" in w
               for w in out["guardrail_warnings"])


def test_find_dropped_source_metrics_flags_the_wer_claim_loss():
    """Run bf0f7941 (Nov 12 2026): the Smart Lecture WER/transcription
    accomplishment vanished from the rewrite while its ROUGE-L half
    survived as a fragment - removals were [], so nothing recorded the
    loss. The tripwire flags only metric evidence that is genuinely gone
    from the whole final text, and stays silent when the rewrite keeps
    every quantified claim (the with-context run of the same pair)."""
    source = (
        "## Projects\n"
        "**Smart Lecture Assistant** | Whisper | Mar 2026 - May 2026\n"
        "- Cut Arabic transcription error rate 51.7% (WER 42.69 to 20.61) "
        "fine-tuning Whisper-small\n"
        "- Boosted semantic search precision 34 points (P@1 0.64 to 0.86)\n"
    )
    dropped = ("## Projects\n"
               "**Smart Lecture Assistant** | Whisper | Mar 2026 - May 2026\n"
               "- Increased semantic search precision by 34 points "
               "(P@1 0.64 -> 0.86)\n")
    kept = ("## Projects\n"
            "**Smart Lecture Assistant** | Whisper | Mar 2026 - May 2026\n"
            "- Reduced Arabic transcription error rate 51.7% "
            "(WER 42.69 -> 20.61) by fine-tuning Whisper-small\n"
            "- Boosted semantic search precision 34 points\n")
    findings = agent.find_dropped_source_metrics(dropped, source)
    assert len(findings) == 1
    assert "51.7%" in findings[0]
    assert "Cut Arabic transcription" in findings[0]
    assert agent.find_dropped_source_metrics(kept, source) == []
    # spaced variants never read as absent (same contract as the
    # added-metric tripwire: the LLM drafts carry U+202F the source lacks)
    spaced = ("## Projects\n"
              "**Smart Lecture Assistant** | Whisper | Mar 2026 - May 2026\n"
              "- Cut Arabic transcription error rate 51.7\u202f% (WER 42.69 "
              "to 20.61) fine-tuning Whisper-small\n")
    assert agent.find_dropped_source_metrics(spaced, source) == []


def test_find_unused_context_terms_flags_missing_context_terms():
    """Round 7 (runs 25ff4ae5 / 1a8125f0, Nov 12 2026): both runs lost the
    user's pasted Aegis project and nothing recorded it. The notes-side
    mirror of find_dropped_source_metrics flags distinctive notes terms
    that appear in NEITHER the source resume NOR the final resume;
    function words, resume-covered terms, and metrics stay silent."""
    source = "Jane Roe\nSUMMARY\nPython developer.\n"
    final = "## Summary\nPython developer with cloud experience.\n"
    notes = ("Aegis: built a heat-risk agent with FastAPI and LangGraph.\n"
             "• Ran the deploy pipeline on a home lab\n")
    out = agent.find_unused_context_terms(notes, final, source)
    lowers = [t.lower() for t in out]
    assert "aegis" in lowers and "fastapi" in lowers
    assert "heat-risk" in lowers and "langgraph" in lowers
    # terms the source resume or the final text already cover never warn
    assert "python" not in lowers and "cloud" not in lowers
    assert "developer" not in lowers and "experience" not in lowers
    # metric-shaped and verb noise stays out
    assert not any(t[0].isdigit() for t in out)
    assert "ran" not in lowers and "lab" not in lowers
    # proper-noun findings rank first (most actionable for the user)
    assert out[0][:1].isupper()
    # empty notes -> silent
    assert agent.find_unused_context_terms("", final, source) == []


def test_find_unused_context_terms_filters_deploy_verbs():
    """Live smoke (session 17f5cfff, Nov 14 2026): the notes' "deployed on a
    4-node home lab" was flagged as an unused context term even though the
    substantive context (Aegis/Kubernetes) was already woven - the one-shot
    weave attempt was spent on a generic verb. Deployment verbs and their
    nominalizations are noise, like ran/run before them."""
    notes = ("Aegis: built a heat-risk agent, deployed on Kubernetes.\n"
             "Ran the deploy pipeline at work.")
    source = "Jane Roe\nSUMMARY\nPython developer.\n"
    final = ("Jane Roe\nSUMMARY\nPython developer.\nPROJECTS\n"
             "Built the Aegis heat-risk agent and its deploy pipeline "
             "on Kubernetes.\n")
    # the substantive terms are covered by the final text -> fully silent
    assert agent.find_unused_context_terms(notes, final, source) == []
    # negative control: the filter is term-level, not a blanket silencer -
    # with a draft that really drops the context, the distinctive terms
    # still surface while the deployment verbs stay out
    out = agent.find_unused_context_terms(notes, "Jane Roe\nPython.\n", source)
    lowers = [t.lower() for t in out]
    assert "aegis" in lowers and "heat-risk" in lowers
    assert "kubernetes" in lowers
    for noise in ("deployed", "deploy", "deployment", "deploying",
                  "deployments", "ran", "lab"):
        assert noise not in lowers


def test_improver_postprocess_flags_unused_context_terms():
    """The tripwire rides the shared writer tail, so both the improver and
    the polisher passes surface notes-sourced terms the final draft never
    references - the round-7 runs shipped no Aegis and no warning."""
    source = (
        "Jane Roe\n"
        "[linkedin.com/in/jane | github.com/jane | huggingface.co/jane]"
        "(https://linkedin.com/in/jane)\n"
        "SUMMARY\n"
        "Python developer.\n"
    )
    state = {"resume_text": source,
             "human_notes": "Aegis: Autonomous Heat-Risk Operations Agent - "
                            "built a FastAPI service around a 4-node "
                            "LangGraph agent with bounded timeouts.",
             "missing_skills": "", "iteration": 1,
             "scrub_removals": [], "unevidenced_metrics": [],
             "declined_requests": [],
             # Round 11: pins the weave attempt cap - without this the
             # dropped-context trigger would invoke the (unpatched, real)
             # creative LLM from a unit test.
             "context_weave_attempted": True}
    draft = ("# Jane Roe\n"
             "https://linkedin.com/in/jane | https://github.com/jane | "
             "https://huggingface.co/jane\n"
             "\n"
             "## Summary\n"
             "Python developer with cloud experience.\n")
    out = agent.nodes.improver._postprocess_draft(state, draft, 1)
    warns = out["guardrail_warnings"]
    assert any("user-context term not in the final resume: Aegis" in w
               for w in warns)
    assert any("user-context term not in the final resume: Heat-Risk" in w
               for w in warns)
    # the tripwire is detection-only: no content change (the tail may
    # normalize the draft's trailing newline)
    assert out["optimized_resume"].rstrip("\n") == draft.rstrip("\n")


# ==========================================
# Round 11: the deterministic context weave-in
# ==========================================
# The dropped-context trigger offers ONE extra structured pass that re-weaves
# user notes the writer dropped (Option A). Adoption is decided in code: the
# woven draft re-enters the SAME patch stack and is kept only if the
# unused-term count strictly decreases.

_WEAVE_NOTES = "Aegis: built a FastAPI + LangGraph heat-risk agent."
_WEAVE_SOURCE = ("Jane Roe\n"
                 "SUMMARY\n"
                 "Python developer.\n"
                 "PROJECTS\n"
                 "Deploy tooling.\n")
_WEAVE_DRAFT = ("Jane Roe\n"
                "SUMMARY\n"
                "Python developer with cloud experience.\n"
                "PROJECTS\n"
                "Deploy tooling.\n")
_WEAVE_WOVEN_PROJECTS = ("**Aegis** - heat-risk agent\n"
                         "- Built the FastAPI + LangGraph heat-risk service.")


def _weave_state(**extra):
    state = {"resume_text": _WEAVE_SOURCE, "human_notes": _WEAVE_NOTES,
             "missing_skills": "", "iteration": 1, "scrub_removals": [],
             "unevidenced_metrics": [], "declined_requests": []}
    state.update(extra)
    return state


def test_context_weave_adopts_when_terms_land_in_the_draft(monkeypatch):
    """A faithful weave passes the deterministic acceptance: the woven draft
    re-enters the same patch stack and ships with the adoption note; the
    stale unused-context warnings for the woven terms are gone."""
    captured = []

    class WeaveLLM:
        def with_structured_output(self, schema):
            assert schema is agent.ImproverOutput

            class Structured:
                def invoke(self, messages):
                    prompt = "\n".join(str(m.content) for m in messages)
                    captured.append(prompt)
                    assert "DROPPED CONTEXT TERMS" in prompt
                    return agent.ImproverOutput(
                        sections=[{"id": "Projects",
                                   "body": _WEAVE_WOVEN_PROJECTS}])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_creative", WeaveLLM())
    out = agent.nodes.improver._postprocess_draft(_weave_state(),
                                                  _WEAVE_DRAFT, 1)
    body = out["optimized_resume"]
    assert "Aegis" in body and "FastAPI" in body and "LangGraph" in body
    assert "user context woven via targeted pass" in out["guardrail_warnings"]
    assert not any("user-context term not in the final resume" in w
                   for w in out["guardrail_warnings"])
    assert out["context_weave_attempted"] is True
    assert out["prompt_versions"] == {
        "context-weave": agent.nodes.improver.CONTEXT_WEAVE_PROMPT_VERSION}
    # exactly one call: the weave pass (this helper bypasses the writer)
    assert len(captured) == 1
    assert "USER NOTES" in captured[0] and "Aegis" in captured[0]


def test_context_weave_not_adopted_keeps_draft_and_warning(monkeypatch):
    """A weave answer that changes nothing (empty sections) cannot pass the
    deterministic acceptance: the draft and the original warning stay, and
    no adoption note is added - only the attempt stamp marks that it ran."""

    class EchoLLM:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ImproverOutput(sections=[], declined=[])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_creative", EchoLLM())
    out = agent.nodes.improver._postprocess_draft(_weave_state(),
                                                  _WEAVE_DRAFT, 1)
    assert out["optimized_resume"].rstrip("\n") == _WEAVE_DRAFT.rstrip("\n")
    warns = out["guardrail_warnings"]
    assert any("user-context term not in the final resume: Aegis" in w
               for w in warns)
    assert not any("woven" in w for w in warns)
    assert out["context_weave_attempted"] is True
    assert out["prompt_versions"] == {
        "context-weave": agent.nodes.improver.CONTEXT_WEAVE_PROMPT_VERSION}


def test_context_weave_runs_at_most_once_per_run(monkeypatch):
    """The attempt cap: once an earlier pass set context_weave_attempted,
    later passes keep the detection-only tripwire and make NO LLM call,
    with no attempt stamp of their own."""

    class MustNotRunLLM:
        def with_structured_output(self, schema):
            raise AssertionError("the weave must not run twice in one run")

    monkeypatch.setattr(agent.llms, "llm_creative", MustNotRunLLM())
    out = agent.nodes.improver._postprocess_draft(
        _weave_state(context_weave_attempted=True), _WEAVE_DRAFT, 2)
    assert out["optimized_resume"].rstrip("\n") == _WEAVE_DRAFT.rstrip("\n")
    assert any("user-context term not in the final resume: Aegis" in w
               for w in out["guardrail_warnings"])
    assert "context_weave_attempted" not in out
    assert "prompt_versions" not in out


def test_context_weave_unavailable_degrades_to_detection_only(monkeypatch):
    """The weave rides the repair ladder; when every rung fails, the draft
    and the original warning stay untouched - and the attempt is still
    stamped, so a dead weave pass is visible in the forensics."""

    class DownLLM:
        def invoke(self, messages):
            raise RuntimeError("ladder down")

        def with_structured_output(self, schema):
            raise RuntimeError("ladder down")

    monkeypatch.setattr(agent.llms, "llm_creative", DownLLM())
    out = agent.nodes.improver._postprocess_draft(_weave_state(),
                                                  _WEAVE_DRAFT, 1)
    assert out["optimized_resume"].rstrip("\n") == _WEAVE_DRAFT.rstrip("\n")
    warns = out["guardrail_warnings"]
    assert any("user-context term not in the final resume: Aegis" in w
               for w in warns)
    assert not any("woven" in w for w in warns)
    assert out["context_weave_attempted"] is True
    assert out["prompt_versions"] == {
        "context-weave": agent.nodes.improver.CONTEXT_WEAVE_PROMPT_VERSION}


def test_full_rewrite_writer_down_skips_the_context_weave(monkeypatch):
    """When the structured writer itself fails, the source ships verbatim
    and the one-shot weave attempt is NOT spent on it: an outage cannot
    weave anything, so no weave prompt ever goes out and no attempt stamp
    lands."""
    seen = []

    class DownWriterLLM:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    seen.append("\n".join(str(m.content) for m in messages))
                    raise RuntimeError("writer down")
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_creative", DownWriterLLM())
    state = {"resume_text": _WEAVE_SOURCE, "optimized_resume": "",
             "human_notes": _WEAVE_NOTES,
             "job_text": "Python engineer with Docker experience.",
             "missing_skills": "", "feedback": "", "iteration": 0,
             "scrub_removals": [], "unevidenced_metrics": [],
             "declined_requests": []}
    out = agent.nodes.improver._full_rewrite(
        state, node_label="improver", version_key="improver",
        version=agent.nodes.improver.IMPROVER_PROMPT_VERSION)
    assert any("writer unavailable" in w for w in out["guardrail_warnings"])
    assert out["optimized_resume"].rstrip("\n") == _WEAVE_SOURCE.rstrip("\n")
    # the dropped context is still surfaced - detection-only
    assert any("user-context term not in the final resume: Aegis" in w
               for w in out["guardrail_warnings"])
    # writer prompts went out; the weave prompt never did
    assert seen and all("Rewrite the resume" in p for p in seen)
    assert not any("DROPPED CONTEXT TERMS" in p for p in seen)
    assert out.get("context_weave_attempted") is None
    assert out["prompt_versions"] == {
        "improver": agent.nodes.improver.IMPROVER_PROMPT_VERSION}


def test_full_rewrite_merges_the_context_weave_stamp(monkeypatch):
    """The stamp merge: _postprocess_draft stamps the weave attempt and the
    caller's own version stamp must survive it (merge, never clobber)."""
    prompts = []

    class ModeKeyedLLM:
        def with_structured_output(self, schema):
            assert schema is agent.ImproverOutput

            class Structured:
                def invoke(self, messages):
                    prompt = "\n".join(str(m.content) for m in messages)
                    prompts.append(prompt)
                    if "DROPPED CONTEXT TERMS" in prompt:
                        return agent.ImproverOutput(
                            sections=[{"id": "Projects",
                                       "body": _WEAVE_WOVEN_PROJECTS}])
                    # the main writer adds nothing: the source ships verbatim
                    return agent.ImproverOutput(sections=[], declined=[])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_creative", ModeKeyedLLM())
    state = {"resume_text": _WEAVE_SOURCE, "optimized_resume": "",
             "human_notes": _WEAVE_NOTES,
             "job_text": "Python engineer with Docker experience.",
             "missing_skills": "", "feedback": "", "iteration": 0,
             "scrub_removals": [], "unevidenced_metrics": [],
             "declined_requests": []}
    out = agent.nodes.improver._full_rewrite(
        state, node_label="improver", version_key="improver",
        version=agent.nodes.improver.IMPROVER_PROMPT_VERSION)
    assert out["context_weave_attempted"] is True
    assert out["prompt_versions"] == {
        "improver": agent.nodes.improver.IMPROVER_PROMPT_VERSION,
        "context-weave": agent.nodes.improver.CONTEXT_WEAVE_PROMPT_VERSION}
    body = out["optimized_resume"]
    assert "Aegis" in body and "FastAPI" in body and "LangGraph" in body
    assert "user context woven via targeted pass" in out["guardrail_warnings"]
    # the writer prompt and the weave prompt are distinguishable on the wire
    assert any("DROPPED CONTEXT TERMS" in p for p in prompts)
    assert any("Rewrite the resume" in p for p in prompts)


def test_scanner_output_is_tamed_for_consumers(monkeypatch):
    class ExoticScannerLLM:
        def invoke(self, messages):
            return AIMessage(content="- Claude\u202fCode workflows\r\n- Hands\u2011on Kubernetes")

    monkeypatch.setattr(agent.llms, "llm_strict", ExoticScannerLLM())
    result = agent.agent_app.invoke(_base_inputs(),
                                    config={"configurable": {"thread_id": "tame-scan"}})
    ms = result["missing_skills"]
    assert "- Claude Code workflows" in ms
    assert "- Hands-on Kubernetes" in ms
    assert "\u202f" not in ms and "\u2011" not in ms


def test_semantic_verification_reclassifies_expressed_keywords(fake_llms, monkeypatch):
    """A keyword absent by string match but clearly expressed under different
    wording is reclassified as a semantic match by the LLM pass."""
    class SemanticFastLLM(FakeFastLLM):
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.SemanticReview(verdicts=[
                        agent.SemanticVerdict(keyword="kubernetes", genuinely_missing=False),
                    ])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_fast", SemanticFastLLM())
    jd = "Engineer with Kubernetes and Docker experience."
    resume = "Used Docker and managed container clusters daily."
    plain = agent.compute_ats_match(jd, resume)
    boosted = agent.compute_ats_match(jd, resume, semantic_matches={"kubernetes"})
    assert boosted["percentage"] > plain["percentage"]
    assert any(r["keyword"] == "kubernetes" and r["method"] == "semantic"
               for r in boosted["keywords"])
    assert "matched semantically" in boosted["formatted"]

    # In-place semantic re-scoring
    in_place = agent.apply_semantic_matches(plain, {"kubernetes"})
    assert in_place["percentage"] == boosted["percentage"]
    assert in_place["match_count"] == boosted["match_count"]
    assert any(r["keyword"] == "kubernetes" and r["method"] == "semantic"
               for r in in_place["keywords"])


# ==========================================
# _strip_emoji_from_html
# ==========================================

def test_strip_emoji_from_html_entities():
    html = "<p>Hello&nbsp;world&mdash;test&hellip;</p>"
    out = agent._strip_emoji_from_html(html)
    assert "&nbsp;" not in out
    assert "Hello world" in out
    assert "--" in out

def test_strip_emoji_from_html_non_ascii_in_text():
    html = "<p>Caf\u00e9 na\u00efve</p>"
    out = agent._strip_emoji_from_html(html)
    assert "Cafe" in out
    assert "naive" in out
    assert all(ord(c) < 128 for c in out)

def test_strip_emoji_from_html_preserves_tags():
    html = "<h1>Title</h1><p>Body</p>"
    out = agent._strip_emoji_from_html(html)
    assert "<h1>" in out
    assert "</p>" in out


# ==========================================
# _sanitize_html_for_pdf
# ==========================================

def test_sanitize_html_removes_script_tags():
    html = "<p>Hello</p><script>alert('xss')</script><p>World</p>"
    out = agent._sanitize_html_for_pdf(html)
    assert "script" not in out.lower()
    assert "Hello" in out
    assert "World" in out

def test_sanitize_html_removes_iframe():
    html = '<p>Text</p><iframe src="https://evil.com"></iframe>'
    out = agent._sanitize_html_for_pdf(html)
    assert "iframe" not in out.lower()

def test_sanitize_html_strips_file_urls():
    html = '<img src="file:///etc/passwd">'
    out = agent._sanitize_html_for_pdf(html)
    assert "file://" not in out

def test_sanitize_html_strips_event_handlers():
    html = '<p onclick="alert(1)">Hello</p>'
    out = agent._sanitize_html_for_pdf(html)
    assert "onclick" not in out

def test_sanitize_html_allows_http_urls():
    html = '<a href="https://example.com">link</a>'
    out = agent._sanitize_html_for_pdf(html)
    assert "https://example.com" in out


# ==========================================
# _html_to_pdf edge cases
# ==========================================

def test_html_to_pdf_empty_markdown(tmp_path):
    result = agent._html_to_pdf("", "test.pdf", "Empty", tmp_path)
    assert Path(result).exists()

def test_html_to_pdf_headers_only(tmp_path):
    result = agent._html_to_pdf("# Title\n## Subtitle", "test.pdf", "Headers", tmp_path)
    assert Path(result).exists()

def test_html_to_pdf_pisa_failure(tmp_path, monkeypatch):
    import agent.helpers as h
    def fake_pisa_create(html, dest=None):
        class FakeResult:
            err = True
        return FakeResult()
    monkeypatch.setattr(h.pisa, "CreatePDF", fake_pisa_create)
    with pytest.raises(RuntimeError, match="Could not create"):
        agent._html_to_pdf("content", "test.pdf", "Fail", tmp_path)


# ==========================================
# _job_brief
# ==========================================

def test_job_brief_includes_keywords():
    jd = "Looking for Python and Docker experience with CI/CD pipelines."
    brief = agent._job_brief(jd, max_chars=200)
    assert "Key requirements/keywords:" in brief
    assert len(brief) > 0

def test_job_brief_respects_max_chars():
    jd = "A" * 5000
    brief = agent._job_brief(jd, max_chars=100)
    assert "Role context:" in brief
    # The role context part should be truncated
    role_line = brief.split("\n")[0]
    assert len(role_line) < 200


# ==========================================
# _fetch_job_description (mocked HTTP)
# ==========================================

def test_fetch_job_description_follows_redirects(monkeypatch):
    import agent.url_fetch as uf
    from email.message import Message

    def _make_headers(**kwargs):
        h = Message()
        for k, v in kwargs.items():
            h[k] = v
        return h

    responses = [
        # First: redirect
        type('Response', (), {
            'status': 302,
            'headers': _make_headers(Location='https://example.com/job/final'),
            'read': lambda self, n: b'',
            'get_content_charset': lambda self: 'utf-8',
        })(),
        # Second: final content
        type('Response', (), {
            'status': 200,
            'headers': _make_headers(),
            'read': lambda self, n: b'Python developer job posting',
            'get_content_charset': lambda self: 'utf-8',
            'close': lambda self: None,
        })(),
    ]
    call_count = [0]
    def fake_open(request, timeout=None):
        r = responses[call_count[0]]
        call_count[0] += 1
        return r

    monkeypatch.setattr(uf, "_validate_job_url", lambda url: url)
    monkeypatch.setattr(uf, "_is_public_host", lambda h: True)
    monkeypatch.setattr(uf, "build_opener", lambda *a: type('Opener', (), {'open': staticmethod(fake_open)})())

    result = uf._fetch_job_description("https://example.com/job")
    assert "Python developer" in result


def test_fetch_job_description_rejects_html_file(monkeypatch):
    import agent.url_fetch as uf
    from email.message import Message

    def _make_headers(**kwargs):
        h = Message()
        for k, v in kwargs.items():
            h[k] = v
        return h

    monkeypatch.setattr(uf, "_validate_job_url", lambda url: url)
    monkeypatch.setattr(uf, "_is_public_host", lambda h: True)

    class FakeResponse:
        status = 200
        headers = _make_headers(**{'Content-Type': 'application/octet-stream'})
        def read(self, n): return b'data'
        def get_content_charset(self): return 'utf-8'

    monkeypatch.setattr(uf, "build_opener", lambda *a: type('Opener', (), {'open': staticmethod(lambda *a, **kw: FakeResponse())})())

    with pytest.raises(ValueError, match="HTML or plain text"):
        uf._fetch_job_description("https://example.com/job")


# ==========================================
# Scanner failure mode
# ==========================================

class FailingStrictLLM:
    """LLM that fails on structured output (reviewer path) AND plain invoke (scanner path)."""
    def invoke(self, messages):
        raise RuntimeError("provider down")
    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                raise RuntimeError("structured output unavailable")
        return Structured()

class FakeCreativeLLM2:
    def invoke(self, messages):
        return AIMessage(content="## Skills\nPython")

class FakeFastLLM2:
    def invoke(self, messages):
        return AIMessage(content="Dear Team,\nSincerely")
    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, messages):
                return agent.SemanticReview(verdicts=[])
        return Structured()


def test_scanner_failure_propagates(monkeypatch):
    """When the scanner LLM fails, the error should propagate up."""
    class FailingScannerLLM:
        """Fails on plain invoke (scanner path) but succeeds on structured output (reviewer)."""
        def invoke(self, messages):
            raise RuntimeError("provider down")
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ReviewOutput(score=85, feedback="ok")
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FailingScannerLLM())
    monkeypatch.setattr(agent.llms, "llm_creative", FakeCreativeLLM2())
    monkeypatch.setattr(agent.llms, "llm_fast", FakeFastLLM2())
    config = {"configurable": {"thread_id": "test-scanner-fail"}}
    with pytest.raises(RuntimeError, match="provider down"):
        agent.agent_app.invoke(_base_inputs(), config=config)


# ==========================================
# Versioned terms
# ==========================================

def test_versioned_terms_extract_base_form():
    jd = "Need Python3.11 and Node20 and React18 experience."
    kws = agent._extract_keywords(jd, top_n=20)
    assert "python" in kws
    assert "node" in kws
    assert "react" in kws


# ==========================================
# Config env overrides
# ==========================================

def test_config_env_overrides(monkeypatch):
    import agent.config as cfg
    monkeypatch.setenv("PRIMARY_MODEL", "custom/model")
    monkeypatch.setenv("SCORE_THRESHOLD", "90")
    monkeypatch.setenv("ATS_SCORE_WEIGHT", "0.7")
    # Re-read from env (the module already loaded, so we test the mechanism)
    assert cfg.ATS_SCORE_WEIGHT == 0.6  # loaded at import, not re-read
    # But the env var IS set
    assert os.environ.get("SCORE_THRESHOLD") == "90"


# ==========================================
# Background disk cleanup & Checkpointer factory
# ==========================================

def test_start_background_cleanup_thread():
    thread = agent.start_background_cleanup(interval_seconds=3600)
    assert thread.is_alive()
    assert thread.daemon is True
    assert thread.name == "DiskCleanupDaemon"


def test_create_checkpointer_default():
    from agent.graphs import _create_checkpointer
    from langgraph.checkpoint.memory import MemorySaver
    cp = _create_checkpointer()
    assert isinstance(cp, MemorySaver)


# ==========================================
# Reviewer structured-output retry
# ==========================================

def test_reviewer_retry_recovers_from_transient_failure(fake_llms, monkeypatch):
    """Trace evidence (Oct 2026): the Reviewer's structured output folds
    transiently (~1 in 3 on that day), zeroing llm_quality_score. One same-payload
    retry must recover instead of flagging review_failed."""
    class FlakyReviewer:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema):
            outer = self

            class Structured:
                def invoke(self, messages):
                    outer.calls += 1
                    if outer.calls == 1:
                        raise RuntimeError("transient structured-output failure")
                    return agent.ReviewOutput(score=85, feedback="Add metrics to bullets.")

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FlakyReviewer())
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is False
    assert result["llm_quality_score"] == 85
    expected = round(agent.ATS_SCORE_WEIGHT * result["ats_percentage"] + agent.LLM_SCORE_WEIGHT * 85)
    assert result["score"] == expected


def test_reviewer_retry_gives_up_after_two_attempts(fake_llms, monkeypatch):
    """When both attempts fail, review_failed still flags (no endless retries)."""
    probe = {"llm_calls": 0}

    class AlwaysBroken(FakeStrictLLM):
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    probe["llm_calls"] += 1
                    raise RuntimeError("structured output unavailable")
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", AlwaysBroken())
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is True
    assert probe["llm_calls"] == 2  # exactly one retry, never more
    assert result["iteration"] == 1


# ==========================================
# Reviewer failed_generation salvage
# ==========================================

# Verbatim prose shape recorded in the Oct 21 trace export (traces 01a15e3a /
# 01a15f6d): unicode narrow no-break spaces and markdown emphasis around the
# score, feedback after a bold marker. '\u2248' = ~, '\u202f' = narrow NBSP.
_SALVAGE_GENERATION = (
    "**Combined weighted quality score:** **\u2248\u202f80\u202f/\u202f100**\n\n"
    "**Feedback:**\u202fHighlight any direct experience you have supporting or "
    "integrating AI coding assistants (e.g., GitHub\u202fCopilot, Cursor, Claude Code) "
    "and explicitly tie those projects to the tooling responsibilities."
)


def _tool_use_failed_exc(generation: str, *, with_body: bool = True):
    """Mimics the Groq BadRequestError shape recorded in the trace export.

    with_body=False reproduces a wrapped error whose str() keeps the repr
    form only (unicode escaped as '\\u202f' literals), since real exceptions
    carry body/response objects."""
    err = {"message": "Tool choice is required, but model did not call a tool",
           "type": "invalid_request_error", "code": "tool_use_failed",
           "failed_generation": generation}
    if with_body:
        class _Exc(Exception):
            pass

        exc = _Exc(f"Error code: 400 - {{'error': {err!r}}}")
        exc.body = {"error": err}
        return exc
    return Exception(f"Error code: 400 - {{'error': {err!r}}}")


def _parse_failed_exc(generation: str):
    """The NON-refused structured-output failure class: the tool call never
    parsed (message differs from the 'Tool choice is required' refusal), so
    the ladder's attempt 2 stays the schema-REPAIR retry instead of the
    prompt-schema rung. Same body/failed_generation shape."""
    err = {"message": "Failed to parse tool arguments for ReviewOutput",
           "type": "invalid_request_error", "code": "invalid_tool_usage",
           "failed_generation": generation}

    class _Exc(Exception):
        pass

    exc = _Exc(f"Error code: 400 - {{'error': {err!r}}}")
    exc.body = {"error": err}
    return exc


def test_salvage_parses_score_from_failed_generation(fake_llms, monkeypatch):
    """Traces 01a15e3a/01a15f6d: on a structured-output failure the model's
    prose answer (score + feedback) sits in body.error.failed_generation.
    MR5 ladder order: attempt 2 is the schema REPAIR (the model sees its own
    answer + the schema - a different payload, so the identical-bytes
    objection from trace 01a15f6d does not apply), and the prose salvage
    below runs only when even the repair folds. This fake folds EVERY
    attempt, so the salvage must still recover the score and avoid
    review_failed at the 2-call cap."""
    probe = {"calls": 0}

    class FoldedReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    probe["calls"] += 1
                    raise _parse_failed_exc(_SALVAGE_GENERATION)

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FoldedReviewer())
    # Pin the loop cap so `calls` counts reviewer invocations of ONE iteration
    # (the default MAX_ITERATIONS=2 runs the reviewer once per pass and the
    # intent of this test is the salvage shortcut, not the loop length).
    monkeypatch.setattr(agent.graphs, "MAX_ITERATIONS", 1)
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is False
    assert result["llm_quality_score"] == 80
    assert "Copilot" in result["feedback"]
    expected = round(agent.ATS_SCORE_WEIGHT * result["ats_percentage"]
                     + agent.LLM_SCORE_WEIGHT * 80)
    assert result["score"] == expected
    assert probe["calls"] == 2  # repair precedes salvage: 2 calls, then salvage


def test_salvage_skipped_when_generation_unparseable(fake_llms, monkeypatch):
    """A structured-output failure whose prose has NO extractable score: the
    MR5 ladder still spends its one repair attempt (the prose exists, so
    attempt 2 shows it back to the model with the schema), the salvage
    returns None on the second failure, and the reviewer degrades to
    review_failed at the 2-call cap."""
    probe = {"calls": 0}

    class FoldedReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    probe["calls"] += 1
                    raise _parse_failed_exc("I could not grade this resume.")

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FoldedReviewer())
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is True
    assert probe["calls"] == 2
    assert result["llm_quality_score"] == 0


def test_salvage_handles_payload_variants():
    salvage = agent._salvage_review_output

    # Subscore lines present: the combined/weighted line must win, not line 1
    combined = _tool_use_failed_exc(
        "Relevance & Tailoring: 70/100\nQuantified Achievements: 90/100\n"
        "Combined weighted quality score: 82/100\n\nFeedback: tighten bullets.")
    assert salvage(combined).score == 82
    assert salvage(combined).feedback == "tighten bullets."

    plain = _tool_use_failed_exc("Quality score: 87/100. Feedback: measure impact.")
    assert salvage(plain).score == 87
    assert salvage(plain).feedback == "measure impact."

    labelled = _tool_use_failed_exc("Score: 87\nFeedback: strong summary.")
    assert salvage(labelled).score == 87

    clamped = _tool_use_failed_exc("Combined weighted quality score: ~130/100")
    assert salvage(clamped).score == 100

    decimal = _tool_use_failed_exc("Quality score: 98.4/100. Feedback: ok.")
    assert salvage(decimal).score == 98

    from agent.nodes.reviewer import _SALVAGE_DEFAULT_FEEDBACK
    silent = _tool_use_failed_exc("Combined weighted quality score: 76/100")
    assert salvage(silent).feedback == _SALVAGE_DEFAULT_FEEDBACK

    assert salvage(_tool_use_failed_exc("Sorry, I cannot grade without context.")) is None
    assert salvage(RuntimeError("connection reset")) is None


def test_salvage_reads_wrapped_repr_errors():
    """When the exception has no .body/.response, the repr string in str(exc)
    is parsed (restoring escaped unicode like the trace export format)."""
    wrapped = _tool_use_failed_exc(_SALVAGE_GENERATION, with_body=False)
    assert getattr(wrapped, "body", None) is None
    salvaged = agent._salvage_review_output(wrapped)
    assert salvaged.score == 80
    assert "Copilot" in salvaged.feedback


# ==========================================
# MR5 provider ladder: uniform schema-repair retry
# ==========================================

def _foldable_llm(first_error, success_after, *, always_fail=False,
                  plain_answers=None):
    """Fake strict LLM for direct ladder tests: raises ``first_error`` on
    attempt 1 (pass None to succeed immediately), returns ``success_after``
    on the next attempt (or keeps raising when ``always_fail``). Records
    every attempt's message list. ``plain_answers`` scripts the prompt-schema
    rung's PLAIN ``llm.invoke`` calls (the json_schema attempt): each call
    pops the next item - a content string (wrapped in an AIMessage) or a
    BaseException to raise - and its message lists land on
    ``Fake.plain_calls``."""
    calls = []
    plain_calls: list = []
    plain = list(plain_answers or [])

    class Fake:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    calls.append(list(messages))
                    if first_error is not None and (len(calls) == 1 or always_fail):
                        raise first_error
                    return success_after

            return Structured()

        def invoke(self, messages):
            plain_calls.append(list(messages))
            answer = plain.pop(0) if plain else RuntimeError(
                "no plain answer scripted")
            if isinstance(answer, BaseException):
                raise answer
            return AIMessage(content=answer)

    Fake.plain_calls = plain_calls
    return Fake, calls


def test_ladder_direct_success_is_one_call():
    good = agent.ReviewOutput(score=90, feedback="ok")
    Fake, calls = _foldable_llm(None, good)
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "direct"
    assert result is good
    assert len(calls) == 1


def test_ladder_repair_shows_model_its_own_answer_and_schema():
    exc = _parse_failed_exc(_SALVAGE_GENERATION)
    good = agent.ReviewOutput(score=85, feedback="ok", sections=["projects"])
    Fake, calls = _foldable_llm(exc, good)
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput,
        [SystemMessage(content="sys"), HumanMessage(content="grade")], label="t")
    assert method == "repair"
    assert result is good
    assert len(calls) == 2
    first, second = calls
    # Attempt 2 = attempt 1's conversation + the model's own prose (as its
    # prior turn) + ONE code-owned repair instruction naming the schema.
    assert second[:len(first)] == first
    assert isinstance(second[-2], AIMessage)
    assert "Combined weighted quality score" in second[-2].content
    assert "ReviewOutput" in second[-1].content
    assert "tool" in second[-1].content.lower()
    assert second[-1].content != first[-1].content


def test_ladder_plain_retry_when_failure_has_no_prose():
    """Transient class (no failed_generation anywhere): attempt 2 stays the
    pre-MR5 plain same-payload retry - messages untouched."""
    Fake, calls = _foldable_llm(
        RuntimeError("connection reset"),
        agent.ReviewOutput(score=70, feedback="ok"))
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "retry"
    assert calls[1] == calls[0]


def test_ladder_salvage_runs_only_after_both_attempts_fail():
    exc = _parse_failed_exc(_SALVAGE_GENERATION)
    Fake, calls = _foldable_llm(exc, None, always_fail=True)
    salvage_seen = []

    def salvage(e):
        salvage_seen.append(e)
        return agent.ReviewOutput(score=80, feedback="salvaged")

    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")],
        salvage=salvage, label="t")
    assert method == "salvage"
    assert result.score == 80
    assert len(calls) == 2          # the 2-LLM-call cap holds
    assert len(salvage_seen) == 1   # salvage sees only the LAST failure


def test_ladder_raises_when_everything_fails():
    Fake, calls = _foldable_llm(RuntimeError("provider down"), None,
                                always_fail=True)
    with pytest.raises(RuntimeError, match="provider down"):
        agent.invoke_structured_with_repair(
            Fake(), agent.ReviewOutput, [HumanMessage(content="grade")],
            label="t")
    assert len(calls) == 2


# ==========================================
# MR5b: extractJson pre-parse rung (free, no LLM call)
# ==========================================

def test_extract_rung_recovers_wrapped_json_without_a_second_call():
    """The failed generation IS the JSON (prose around a fenced block): the
    extract rung returns the schema-valid instance and the repair retry (an
    LLM call) is never spent."""
    generation = ("Sure, here is the review:\n```json\n"
                  '{"score": 85, "feedback": "Tighten bullets.",'
                  ' "sections": ["projects"]}\n```\n'
                  "Let me know if you need changes.")
    exc = _tool_use_failed_exc(generation)
    Fake, calls = _foldable_llm(exc, None, always_fail=True)
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "extract"
    assert result.score == 85
    assert result.sections == ["projects"]
    assert len(calls) == 1          # zero repair calls spent


def test_extract_rung_skips_prose_that_was_never_json():
    """Reviewer prose (score sentences, no JSON) does not trigger the extract
    rung - the ladder still spends the schema-repair retry (the pre-round-8
    behavior is preserved exactly)."""
    exc = _parse_failed_exc(_SALVAGE_GENERATION)
    good = agent.ReviewOutput(score=85, feedback="ok")
    Fake, calls = _foldable_llm(exc, good)
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "repair"
    assert result is good
    assert len(calls) == 2


def test_extract_rung_rejects_json_that_fails_the_schema():
    """A candidate that parses but violates the schema is not returned - the
    ladder falls through to the repair retry."""
    exc = _parse_failed_exc(
        'Here you go:\n{"score": "very high", "feedback": "ok"}')
    good = agent.ReviewOutput(score=70, feedback="ok")
    Fake, calls = _foldable_llm(exc, good)
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "repair"
    assert result is good
    assert len(calls) == 2


# ==========================================
# MR5c: SCHEMA_REFUSED -> the prompt-schema rung
# ==========================================

def test_schema_refused_markers_stay_tight():
    """A match reroutes attempt 2 away from the tool binding, so the marker
    list must not match ordinary parse/transient failures."""
    from agent.provider import _is_schema_refused

    assert _is_schema_refused(RuntimeError(
        "Error 400: Tool choice is required, but model did not call a tool"))
    assert _is_schema_refused(RuntimeError("tool calling is not supported"))
    assert _is_schema_refused(RuntimeError("Tool use is not supported"))
    assert _is_schema_refused(RuntimeError("Function calling is not supported"))
    assert not _is_schema_refused(RuntimeError("connection reset"))
    assert not _is_schema_refused(RuntimeError("Failed to parse tool arguments"))
    assert not _is_schema_refused(_parse_failed_exc("Quality score: 87/100"))


def test_ladder_schema_refused_switches_to_prompt_schema_rung():
    """The 'Tool choice is required' flake class: the endpoint refused the
    tool-calling contract itself, so attempt 2 does NOT retry through the
    tool layer - it re-asks with the schema in the PROMPT (plain invoke, no
    binding) and the code parses the answer against the schema."""
    import json as _json

    exc = _tool_use_failed_exc(
        "Error: Tool choice is required, but model did not call a tool")
    good_json = _json.dumps({"score": 88, "feedback": "ok",
                             "sections": ["experience"]})
    Fake, calls = _foldable_llm(exc, None, always_fail=True,
                                plain_answers=[good_json])
    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")], label="t")
    assert method == "json_schema"
    assert result.score == 88
    assert result.sections == ["experience"]
    assert len(calls) == 1          # only attempt 1 used the tool binding
    plain = Fake.plain_calls
    assert len(plain) == 1
    # The original conversation is preserved and ONE code-owned instruction
    # carrying the JSON schema is appended - the schema travels in the
    # prompt, never through a tool binding.
    assert len(plain[0]) == 2
    assert plain[0][0].content == "grade"
    assert '"score"' in plain[0][1].content
    assert '"sections"' in plain[0][1].content
    assert "JSON" in plain[0][1].content


def test_ladder_schema_refused_rejected_answer_rides_to_salvage():
    """A prompt-schema answer that fails to parse is carried in the
    failed_generation envelope to caller salvage - the fresh answer gets a
    real chance (the reviewer's prose salvage parses a score out of it)."""
    exc = _tool_use_failed_exc(
        "Error: Tool choice is required, but model did not call a tool")
    Fake, calls = _foldable_llm(
        exc, None, always_fail=True,
        plain_answers=["Combined weighted quality score: 82/100\n"
                       "Feedback: tightened the summary."])
    salvage_seen = []

    def salvage(e):
        salvage_seen.append(e)
        return agent._salvage_review_output(e)

    result, method = agent.invoke_structured_with_repair(
        Fake(), agent.ReviewOutput, [HumanMessage(content="grade")],
        salvage=salvage, label="t")
    assert method == "salvage"
    assert result.score == 82
    assert len(salvage_seen) == 1
    body = getattr(salvage_seen[0], "body", None)
    assert body["error"]["failed_generation"].startswith(
        "Combined weighted quality score")
    assert len(calls) == 1          # the 2-LLM-call cap holds


def test_ladder_schema_refused_invoke_failure_goes_to_salvage():
    """If the prompt-schema invoke itself raises, that exception (not the
    original refusal) is what salvage sees - and what is raised after."""
    exc = _tool_use_failed_exc("tool calling is not supported on this endpoint")
    Fake, calls = _foldable_llm(exc, None, always_fail=True,
                                plain_answers=[RuntimeError("endpoint dropped")])
    salvage_seen = []

    def salvage(e):
        salvage_seen.append(e)
        return None

    with pytest.raises(RuntimeError, match="endpoint dropped"):
        agent.invoke_structured_with_repair(
            Fake(), agent.ReviewOutput, [HumanMessage(content="grade")],
            salvage=salvage, label="t")
    assert len(salvage_seen) == 1
    assert str(salvage_seen[0]) == "endpoint dropped"
    assert len(calls) == 1          # the 2-LLM-call cap holds


def test_extract_candidate_finds_first_object_and_skips_string_braces():
    """The Dossier-study candidate order: whole text, fenced blocks, then the
    first complete object - bracket matching that skips string literals (a
    brace inside a value never closes the span early)."""
    noisy = 'Preface: { "score": 88, "feedback": "has } brace" }'
    got = agent.extract_structured_candidate(agent.ReviewOutput, noisy)
    assert got is not None and got.score == 88
    got = agent.extract_structured_candidate(
        agent.ReviewOutput, '{"score": 90, "feedback": "ok"}')
    assert got is not None and got.score == 90
    # No JSON / empty / schema-violating -> None.
    assert agent.extract_structured_candidate(
        agent.ReviewOutput, "no json here") is None
    assert agent.extract_structured_candidate(agent.ReviewOutput, "") is None
    assert agent.extract_structured_candidate(
        agent.ReviewOutput, '{"score": "ninety", "feedback": "ok"}') is None


# ==========================================
# Round 8: durable checkpoint store (Dossier steal - SqliteSaver default)
# ==========================================

def test_checkpointer_resolver_escapes_and_default():
    from pathlib import Path
    resolve = agent.config._resolve_checkpointer_db_path
    assert resolve("off") == ""
    assert resolve("none") == "" and resolve("memory") == ""
    assert resolve(str(Path("somewhere") / "cp.sqlite3")) == str(
        Path("somewhere") / "cp.sqlite3")
    # Unset / whitespace-only -> the durable workspace default.
    expected = (Path(agent.__file__).resolve().parents[1] / "data"
                / "checkpoints" / "checkpoints.sqlite3")
    assert Path(resolve("")) == expected
    assert Path(resolve("  ")) == expected


def test_create_checkpointer_opens_sqlite_and_creates_dirs(monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    db = tmp_path / "nested" / "cp.sqlite3"
    monkeypatch.setattr(agent.graphs, "CHECKPOINTER_DB_PATH", str(db))
    saver = agent.graphs._create_checkpointer()
    assert isinstance(saver, SqliteSaver)
    assert db.exists()              # parent dirs created, file opened


def test_create_checkpointer_memory_when_disabled(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    monkeypatch.setattr(agent.graphs, "CHECKPOINTER_DB_PATH", "")
    assert isinstance(agent.graphs._create_checkpointer(), MemorySaver)


def test_create_checkpointer_falls_back_when_store_unavailable(monkeypatch, tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    # The db's parent is a FILE -> mkdir raises -> graceful MemorySaver fallback.
    monkeypatch.setattr(agent.graphs, "CHECKPOINTER_DB_PATH",
                        str(blocker / "cp.sqlite3"))
    assert isinstance(agent.graphs._create_checkpointer(), MemorySaver)


def test_repair_recovers_structured_review_with_sections(fake_llms, monkeypatch):
    """The point of MR5: when the reviewer's structured output folds but the
    model's prose exists, the REPAIR attempt recovers the full structured
    result - including `sections`, which prose salvage loses."""
    calls = []

    class FoldThenRepair:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    calls.append(1)
                    if len(calls) == 1:
                        raise _parse_failed_exc(_SALVAGE_GENERATION)
                    return agent.ReviewOutput(score=85, feedback="Fix bullets.",
                                              sections=["projects"])

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FoldThenRepair())
    monkeypatch.setattr(agent.graphs, "MAX_ITERATIONS", 1)
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is False
    assert result["llm_quality_score"] == 85
    assert result["feedback_sections"] == ["projects"]
    assert len(calls) == 2


def test_ats_semantic_repair_recovers_verdicts(monkeypatch):
    """The non-refused parse-failure class (the refusal class now takes the
    prompt-schema rung): attempt 2 (repair) recovers the verdicts instead of
    degrading keywords to missing."""
    calls = []

    class FoldThenRepair:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    calls.append(1)
                    if len(calls) == 1:
                        raise _parse_failed_exc(
                            "Docker: the resume mentions building images.")
                    return agent.SemanticReview(verdicts=[
                        agent.SemanticVerdict(keyword="docker",
                                              genuinely_missing=False),
                        agent.SemanticVerdict(keyword="kubernetes",
                                              genuinely_missing=True),
                    ])

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_fast", FoldThenRepair())
    from agent.nodes.ats_check import _semantic_verify_missing
    expressed = _semantic_verify_missing(
        "Jane Roe built docker images", ["docker", "kubernetes"])
    assert expressed == {"docker"}
    assert len(calls) == 2


def test_ats_semantic_degrades_after_ladder_exhausted(monkeypatch):
    """Both attempts fold with NO prose: the safe failure direction keeps
    every keyword as missing (never a fabricated semantic rescue)."""
    calls = []

    class AlwaysFolded:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    calls.append(1)
                    raise RuntimeError("no structured output today")

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_fast", AlwaysFolded())
    from agent.nodes.ats_check import _semantic_verify_missing
    expressed = _semantic_verify_missing("resume text", ["docker"])
    assert expressed == set()
    assert len(calls) == 2


def test_token_usage_tracker_reads_llm_output_and_metadata_fallback():
    from types import SimpleNamespace

    # OpenAI-shaped llm_output (the wire path) wins when present...
    tracker = agent.TokenUsageTracker()
    tracker.on_llm_end(SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        generations=[[SimpleNamespace(message=SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 100}))]]))
    assert tracker.totals() == {"prompt_tokens": 5, "completion_tokens": 2,
                                "total_tokens": 7, "llm_calls": 1}

    # ...else langchain-core's usage_metadata on the generated message.
    tracker2 = agent.TokenUsageTracker()
    tracker2.on_llm_end(SimpleNamespace(
        llm_output=None,
        generations=[[SimpleNamespace(message=SimpleNamespace(
            usage_metadata={"input_tokens": 7, "output_tokens": 3}))]]))
    assert tracker2.totals() == {"prompt_tokens": 7, "completion_tokens": 3,
                                 "total_tokens": 10, "llm_calls": 1}

    # No usage anywhere (some fakes): zeroed totals, never raises.
    tracker3 = agent.TokenUsageTracker()
    tracker3.on_llm_end(SimpleNamespace(llm_output=None, generations=[]))
    tracker3.on_llm_end(SimpleNamespace(llm_output="not a dict", generations=[]))
    assert tracker3.totals() == {"prompt_tokens": 0, "completion_tokens": 0,
                                 "total_tokens": 0, "llm_calls": 0}

    # Concurrent accumulation (the cover/prep fan-out runs in threads).
    import threading
    tracker4 = agent.TokenUsageTracker()
    resp = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    threads = [threading.Thread(target=tracker4.on_llm_end, args=(resp,))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker4.totals() == {"prompt_tokens": 4, "completion_tokens": 4,
                                 "total_tokens": 8, "llm_calls": 4}


def test_token_usage_tracker_attributes_tasks_via_langgraph_node():
    """MR5: task attribution with zero node-code changes. A callback inside
    a LangGraph node sees metadata['langgraph_node'] at START (metadata is
    None at END - probed on langgraph 1.2.16), so the tracker captures the
    task into a run_id map at start and attributes at on_llm_end via the
    shared run_id. Node ids with underscores alias to the prompt_versions
    stamp vocabulary."""
    import uuid as _uuid
    from types import SimpleNamespace

    tracker = agent.TokenUsageTracker()
    resp = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        generations=[])

    for node, expected in (("ats_check", "ats-semantic"),
                           ("scanner", "scanner"),
                           ("improver", "improver")):
        run = _uuid.uuid4()
        tracker.on_chat_model_start(None, [], run_id=run,
                                    metadata={"langgraph_node": node})
        tracker.on_llm_end(resp, run_id=run)
        assert tracker.task_totals()[expected]["llm_calls"] == 1

    assert tracker.totals() == {"prompt_tokens": 6, "completion_tokens": 3,
                                "total_tokens": 9, "llm_calls": 3}
    assert tracker.task_totals() == {
        "ats-semantic": {"prompt_tokens": 2, "completion_tokens": 1,
                         "total_tokens": 3, "llm_calls": 1},
        "improver": {"prompt_tokens": 2, "completion_tokens": 1,
                     "total_tokens": 3, "llm_calls": 1},
        "scanner": {"prompt_tokens": 2, "completion_tokens": 1,
                    "total_tokens": 3, "llm_calls": 1},
    }


def test_token_usage_tracker_unknown_bucket_and_garbage_metadata():
    """No captured start (run_id=None, unusable metadata) buckets as
    'unknown' and never raises."""
    import uuid as _uuid
    from types import SimpleNamespace

    tracker = agent.TokenUsageTracker()
    resp = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        generations=[])
    # Existing direct on_llm_end calls (fakes that replace whole chains)
    # carry no run_id at all.
    tracker.on_llm_end(resp)
    # Garbage metadata: wrong type, then an empty dict.
    run = _uuid.uuid4()
    tracker.on_chat_model_start(None, [], run_id=run, metadata="not-a-dict")
    tracker.on_llm_end(resp, run_id=run)
    run = _uuid.uuid4()
    tracker.on_llm_start(None, ["p"], run_id=run, metadata={})
    tracker.on_llm_end(resp, run_id=run)
    totals = tracker.task_totals()
    assert set(totals) == {"unknown"}
    assert totals["unknown"] == {"prompt_tokens": 3, "completion_tokens": 3,
                                 "total_tokens": 6, "llm_calls": 3}


def test_token_usage_tracker_on_llm_error_drops_the_run_mapping():
    """A failed call produces no usage; the START-time mapping is popped so
    the map cannot leak across the segment."""
    import uuid as _uuid

    tracker = agent.TokenUsageTracker()
    run = _uuid.uuid4()
    tracker.on_chat_model_start(None, [], run_id=run,
                                metadata={"langgraph_node": "reviewer"})
    tracker.on_llm_error(RuntimeError("boom"), run_id=run)
    assert tracker.task_totals() == {}
    assert tracker.totals()["llm_calls"] == 0


def test_token_usage_tracker_task_attribution_is_thread_safe():
    """The fan-out (cover letter + prep) attributes concurrently."""
    import threading
    import uuid as _uuid
    from types import SimpleNamespace

    tracker = agent.TokenUsageTracker()
    resp = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        generations=[])

    def work():
        run = _uuid.uuid4()
        tracker.on_chat_model_start(None, [], run_id=run,
                                    metadata={"langgraph_node": "cover_letter"})
        tracker.on_llm_end(resp, run_id=run)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.task_totals() == {
        "cover-letter": {"prompt_tokens": 4, "completion_tokens": 4,
                         "total_tokens": 8, "llm_calls": 4}}


# ==========================================
# LangSmith feedback logging
# ==========================================

class _FakeFeedbackClient:
    def __init__(self):
        self.calls = []

    def create_feedback(self, run_id, **kwargs):
        self.calls.append((run_id, kwargs))


def test_log_feedback_noops_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(agent.tracing, "_CLIENT", None)
    assert agent.log_feedback("run-1", "composite_score", score=90) is False


def test_log_feedback_forwards_to_client_and_swallows_errors(monkeypatch, capsys):
    fake = _FakeFeedbackClient()
    monkeypatch.setattr(agent.tracing, "_CLIENT", fake)
    assert agent.log_feedback("run-x", "composite_score", score=67.0) is True
    assert agent.log_feedback("run-x", "review_failed", value=1) is True
    assert fake.calls[0] == ("run-x", {"key": "composite_score", "score": 67.0,
                                       "value": None, "comment": None})
    assert fake.calls[1][1] == {"key": "review_failed", "score": None,
                                "value": 1, "comment": None}

    class BrokenClient:
        def create_feedback(self, *args, **kwargs):
            raise RuntimeError("langsmith down")

    monkeypatch.setattr(agent.tracing, "_CLIENT", BrokenClient())
    assert agent.log_feedback("run-x", "composite_score", score=10) is False
    assert "feedback" in capsys.readouterr().out  # graceful warning, no raise


def test_log_pipeline_feedback_records_score_and_failure_flag(monkeypatch):
    fake = _FakeFeedbackClient()
    monkeypatch.setattr(agent.tracing, "_CLIENT", fake)
    assert agent.log_pipeline_feedback(
        {"score": 41, "ats_percentage": 12.3, "llm_quality_score": 0,
         "review_failed": True}, "run-42") is True
    records = [(c[1]["key"], c[1]["score"], c[1]["value"], c[1]["comment"])
               for c in fake.calls]
    assert ("composite_score", 41.0, None, None) in records
    failed = [r for r in records if r[0] == "review_failed"][0]
    assert failed[2] == 1
    assert "ATS-only" in failed[3]

    healthy_client = _FakeFeedbackClient()
    monkeypatch.setattr(agent.tracing, "_CLIENT", healthy_client)
    agent.log_pipeline_feedback(
        {"score": 82, "ats_percentage": 74.0, "llm_quality_score": 88,
         "review_failed": False}, "run-43")
    healthy = [c for c in healthy_client.calls if c[1]["key"] == "review_failed"][0]
    assert healthy[1]["value"] == 0
    assert healthy[1]["comment"] is None


def test_log_pipeline_feedback_guardrails_without_client_or_run_id(monkeypatch):
    assert agent.log_pipeline_feedback(None, "run-9") is False

    monkeypatch.setattr(agent.tracing, "_CLIENT", _FakeFeedbackClient())
    assert agent.log_pipeline_feedback({"score": 41}, None) is False

    monkeypatch.setattr(agent.tracing, "_CLIENT", None)
    assert agent.log_pipeline_feedback({"score": 41}, "run-9") is False


# ==========================================
# Name-based PII masking (traced payloads)
# ==========================================

@pytest.fixture
def clean_name_register(monkeypatch):
    """Isolates the global name register/mask cache for masking tests."""
    monkeypatch.setattr(agent.tracing, "_KNOWN_NAMES", [])
    monkeypatch.setattr(agent.tracing, "_NAME_MASKS", None)
    monkeypatch.delenv("REDACT_NAMES", raising=False)
    yield


def test_redact_masks_registered_candidate_name(clean_name_register):
    agent.register_known_name("John Doe")
    payload = {"resume": "John Doe built Python pipelines.",
               "nested": ["Contact for John Doe at 0812 123456"],
               "letter": "Sincerely,\nJOHN DOE"}
    masked = agent._redact_pii(payload)
    assert "John Doe" not in str(masked)
    assert "[redacted-name]" in masked["resume"]
    assert "[redacted-name]" in masked["nested"][0]
    assert "[redacted-name]" in masked["letter"]
    assert "[redacted-phone]" in masked["nested"][0]


def test_name_masking_is_case_insensitive_with_word_boundaries(clean_name_register):
    agent.register_known_name("John Doe")
    masked = agent._redact_pii("JOHN DOE joined; john dough stayed; JohnDoe merged.")
    assert "JOHN DOE" not in masked
    assert "dough stayed" in masked   # 'Doe' token too short to mask; no bleed into 'Dough'
    assert "JohnDoe" in masked        # full-name mask requires a separator (\s+)


def test_name_token_masks_cover_longer_names(clean_name_register):
    agent.register_known_name("Alexander Wang")
    masked = agent._redact_pii("Alexander Wang; alexander alone; Wang alone; Wang2 stays")
    assert "Alexander" not in masked
    assert "alexander alone" not in masked
    assert "[redacted-name] alone" in masked.split(";")[1]
    assert "Wang2" in masked  # boundary: token regex must not split off 2


def test_redact_names_env_feeds_masking(clean_name_register, monkeypatch):
    monkeypatch.setenv("REDACT_NAMES", " Omar Ali , Mixed Case Name ")
    masked = agent._redact_pii("hello Omar Ali and MIXED CASE NAME")
    assert "Omar Ali" not in masked
    assert "MIXED CASE NAME" not in masked


def test_candidate_name_hint_accepts_and_rejects():
    hint = agent.candidate_name_hint
    assert hint("Name: John Doe - CEO\nExperience") == "John Doe"
    assert hint("John A. Doe\nPython developer.") == "John A. Doe"
    assert hint("John Doe | Data Engineer avatar line") is None
    assert hint("SOFTWARE ENGINEER\nToledo, OH") is None
    assert hint("Python developer with 5 years of experience.") is None
    assert hint("John Doe2\nSkills") is None  # digits disqualify
    assert hint("") is None


def test_loader_registers_resume_name_for_masking(clean_name_register, fake_llms):
    result = agent.full_app.invoke({**_base_inputs(),
                                    "original_resume": "John Doe\nPython developer."})
    assert result["resume_text"].startswith("John Doe")
    assert "John Doe" in agent.tracing._KNOWN_NAMES
    masked = agent._redact_pii(result["optimized_resume"] and "John Doe wrote Go.")
    assert "John Doe" not in masked


def test_loader_without_name_registers_nothing(clean_name_register, fake_llms):
    agent.full_app.invoke(_base_inputs())
    assert agent.tracing._KNOWN_NAMES == []


# ==========================================
# DOCX resume ingestion
# ==========================================

def _add_hyperlink(paragraph, text, url):
    """Create a real w:hyperlink run with the relationship registered on the
    document part (matches what Word writes for hyperlinked resume text)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    r_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    run = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _make_docx(path, paragraphs, tables=()):
    # paragraph/cell values may be ("text", "url") -> hyperlinked text.
    from docx import Document
    doc = Document()
    for paragraph in paragraphs:
        if isinstance(paragraph, tuple):
            text, url = paragraph
            p = doc.add_paragraph()
            _add_hyperlink(p, text, url)
        else:
            doc.add_paragraph(paragraph)
    for rows in tables:
        table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if isinstance(cell, tuple):
                    text, url = cell
                    _add_hyperlink(table.rows[i].cells[j].paragraphs[0], text, url)
                else:
                    table.rows[i].cells[j].text = cell
    doc.save(str(path))


def test_load_resume_docx_extracts_paragraphs_and_tables(tmp_path):
    doc = tmp_path / "resume.docx"
    _make_docx(doc, ["# Jane Doe", "## Skills: Python, Docker"], tables=[[["AWS", "Docker", "Git"]]])
    text = agent._load_resume_docx(str(doc))
    assert "# Jane Doe" in text
    assert "Python, Docker" in text
    assert "AWS | Docker | Git" in text


def test_load_resume_docx_paragraph_hyperlinks_preserved(tmp_path):
    """Hyperlinked paragraph text must come through as [Text](URL), not bare text."""
    doc = tmp_path / "resume.docx"
    _make_docx(doc, [
        "# Jane Doe",
        ("jane.dev portfolio", "https://jane.dev"),
        "## Skills: Python, Docker",
    ])
    text = agent._load_resume_docx(str(doc))
    assert "[jane.dev portfolio](https://jane.dev)" in text


def test_load_resume_docx_table_cell_hyperlinks_preserved(tmp_path):
    """Audit P1: table-grid resumes lose every link when cells are read as raw
    text. Cell paragraphs must run through the same hyperlink-aware reader
    as body paragraphs, so grid links survive as markdown links."""
    doc = tmp_path / "resume.docx"
    _make_docx(doc, ["# Jane Doe"], tables=[[
        ["Project", ("Live Demo", "https://jane.dev/streamer"), "Go, Kafka"],
        ["Skills", "Docker", ("GitHub", "https://github.com/jane/streamer")],
    ]])
    text = agent._load_resume_docx(str(doc))
    assert "Project | [Live Demo](https://jane.dev/streamer) | Go, Kafka" in text
    assert "Skills | Docker | [GitHub](https://github.com/jane/streamer)" in text


def test_join_visitor_fragments_gap_aware():
    """Audit P1: pypdf visitor fragments joined with ''.join() glued words
    together. Separator decisions must come from run geometry."""
    join = agent.url_fetch._join_visitor_fragments
    # Real inter-word gap -> space
    assert join([("Python", 100.0, 200.0, 10.0),
                 ("Developer", 132.5, 200.0, 10.0)]) == "Python Developer"
    # Mid-word run split (next run starts at the estimated right edge of the
    # previous run) -> must glue back together, no space inserted
    assert join([("Pyth", 100.0, 200.0, 10.0),
                 ("onistic", 119.3, 200.0, 10.0)]) == "Pythonistic"
    # Baseline y move -> newline
    assert join([("Machine", 100.0, 200.0, 10.0),
                 ("Learning", 100.0, 212.0, 10.0)]) == "Machine\nLearning"
    # Runs carrying their own whitespace are attached verbatim
    assert join([("Hello ", 100.0, 200.0, 10.0),
                 ("World", 135.0, 200.0, 10.0)]) == "Hello World"
    # Whitespace-only fragment passes through without an invented separator
    assert join([("Text", 100.0, 200.0, 10.0),
                 ("  ", 140.0, 200.0, 10.0),
                 ("more", 142.0, 200.0, 10.0)]) == "Text  more"
    # Same position (overlap) -> direct concatenation
    assert join([("line1", 100.0, 200.0, 10.0),
                 ("line2", 100.0, 200.0, 10.0)]) == "line1line2"
    assert join([]) == ""
    # Markdown-wrapped link labels don't count unrendered URL glyphs toward
    # the width estimate (label-only visibility)
    out = join([(" [Live Demo](https://jane.dev)", 100.0, 200.0, 10.0),
                ("2026", 158.0, 200.0, 10.0)])
    assert out.endswith(" 2026"), out


def test_load_resume_docx_rejects_non_docx_bytes(tmp_path):
    doc = tmp_path / "resume.docx"
    doc.write_bytes(b"%%PDF-not-a-word-file")
    with pytest.raises(ValueError, match="not a valid Word document"):
        agent._load_resume_docx(str(doc))


def test_load_resume_docx_rejects_oversize(tmp_path, monkeypatch):
    doc = tmp_path / "resume.docx"
    doc.write_bytes(b"PK\x03\x04small")
    import agent.url_fetch as uf
    monkeypatch.setattr(uf, "MAX_RESUME_FILE_BYTES", 2)
    with pytest.raises(ValueError, match="5 MB or smaller"):
        agent._load_resume_docx(str(doc))


def test_full_graph_accepts_docx_resume(fake_llms, tmp_path):
    """A .docx upload flows through loader -> improver -> ... like a PDF."""
    doc = tmp_path / "resume.docx"
    _make_docx(doc, ["Python developer.", "## Skills", "Docker and Kubernetes"])
    inputs = _base_inputs()
    inputs["original_resume"] = str(doc)
    result = agent.full_app.invoke(inputs)
    assert "Kubernetes" in result["resume_text"]
    assert result["optimized_resume"]
    assert Path(result["resume_pdf_path"]).exists()


# ==========================================
# Startup provider health check
# ==========================================

def test_check_provider_health_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ok, message = agent.llms.check_provider_health()
    assert ok is False
    assert "GROQ_API_KEY is not set" in message


def test_check_provider_health_placeholder_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_key_here")
    ok, message = agent.llms.check_provider_health()
    assert ok is False


def test_check_provider_health_delegates_to_models_status(monkeypatch):
    called = {}
    monkeypatch.setattr(agent.llms, "_groq_models_status",
                        lambda key, t: (True, f"API reachable ({key[:3]}...)"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_live_123")
    ok, message = agent.llms.check_provider_health()
    assert ok is True
    assert "API reachable" in message


def test_groq_models_status_auth_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("https://api.groq.com/openai/v1/models", 401,
                                     "Unauthorized", None, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, message = agent.llms._groq_models_status("gsk_bad")
    assert ok is False
    assert "401" in message


def test_groq_models_status_rate_limit_counts_as_ok(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("https://api.groq.com/openai/v1/models", 429,
                                     "Too Many Requests", None, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is True
    assert "429" in message


def test_groq_models_status_unreachable_soft_fails(monkeypatch):
    from urllib.error import URLError

    def fake_urlopen(request, timeout=None):
        raise URLError("dns failure")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is False
    assert "Could not reach the Groq API" in message


def test_groq_models_status_success_reports_models(monkeypatch):
    import json

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [
                {"id": agent.llms._PRIMARY_MODEL},
                {"id": agent.llms._FALLBACK_MODEL},
                {"id": agent.llms._EMERGENCY_MODEL},
            ]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse())
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is True
    assert "API reachable" in message
    assert "missing from this account's catalog" not in message


def test_groq_models_status_flags_missing_catalog_tiers(monkeypatch):
    """A 200 lacking any configured tier (here: all three) names each with its env override."""
    import json

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "llama-3.1-8b-instant"}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse())
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is True
    assert agent.llms._PRIMARY_MODEL in message
    assert agent.llms._FALLBACK_MODEL in message
    assert agent.llms._EMERGENCY_MODEL in message
    assert "missing from this account's catalog" in message
    assert "PRIMARY_MODEL" in message
    assert "EMERGENCY_MODEL" in message


def test_groq_models_status_missing_emergency_only_warns(monkeypatch):
    """Regression (Nov 10 2026 404 incident): only the emergency tier missing.

    The catalog holds the primary and fallback models but the emergency default
    was decommissioned - the warning must name it at startup instead of letting
    a run 404 after falling through two tiers.
    """
    import json

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [
                {"id": agent.llms._PRIMARY_MODEL},
                {"id": agent.llms._FALLBACK_MODEL},
            ]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse())
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is True
    assert "missing from this account's catalog" in message
    assert agent.llms._EMERGENCY_MODEL in message
    assert "EMERGENCY_MODEL" in message
    assert agent.llms._PRIMARY_MODEL not in message


# ==========================================
# MR5 provider knobs: LLM_BASE_URL / LLM_TIMEOUT_SECONDS
# ==========================================

def test_models_endpoint_official_and_custom(monkeypatch):
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "")
    assert agent.llms._models_endpoint() == ("https://api.groq.com/openai/v1/models", True)
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "http://localhost:11434/v1/")
    url, is_groq = agent.llms._models_endpoint()
    assert url == "http://localhost:11434/v1/models"
    assert is_groq is False


def test_chat_kwargs_official_groq(monkeypatch):
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "")
    kwargs = agent.llms._chat_kwargs("m1", 0.2, reasoning_effort="low",
                                     api_key="k")
    assert kwargs["model"] == "m1"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_retries"] == 2
    assert kwargs["request_timeout"] == agent.llms.LLM_TIMEOUT_SECONDS
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["groq_api_key"] == "k"
    assert "base_url" not in kwargs


def test_chat_kwargs_never_injects_none_key_or_reasoning(monkeypatch):
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "")
    kwargs = agent.llms._chat_kwargs("m", 0.0)
    # api_key=None must never override ChatGroq's env-var default factory
    assert "groq_api_key" not in kwargs
    # reasoning_effort is set only by the Groq-tuned tiers that pass it
    assert "reasoning_effort" not in kwargs


def test_chat_kwargs_custom_endpoint(monkeypatch):
    """Any OpenAI-compatible base rides the MR4-proven ChatGroq seam; the
    Groq-specific reasoning_effort is omitted (custom endpoints may reject
    unknown params)."""
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "http://localhost:1234/v1")
    kwargs = agent.llms._chat_kwargs("m2", 0.2, reasoning_effort="low")
    assert kwargs["base_url"] == "http://localhost:1234/v1"
    assert "reasoning_effort" not in kwargs


def test_check_provider_health_custom_endpoint_needs_no_key(monkeypatch):
    """A local Ollama/LM Studio endpoint needs no GROQ_API_KEY (config.py
    relaxed its import-time check the same way); the delegated status call
    receives the empty key (no Authorization header)."""
    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = {}

    def fake_status(key, timeout_seconds=8.0):
        seen["key"] = key
        return True, "API reachable (1 models)."

    monkeypatch.setattr(agent.llms, "_groq_models_status", fake_status)
    ok, message = agent.llms.check_provider_health()
    assert ok is True
    assert seen["key"] == ""


def test_groq_models_status_custom_endpoint_skips_tier_gap_warning(monkeypatch):
    """Tier names are Groq-catalog vocabulary: with a custom LLM_BASE_URL the
    missing-tier warning must NOT fire even when the catalog lacks every
    configured model."""
    import json

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "local-model"}]}).encode()

    monkeypatch.setattr(agent.llms, "LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda request, timeout=None: FakeResponse())
    ok, message = agent.llms._groq_models_status("")
    assert ok is True
    assert "API reachable" in message
    assert "missing from this account's catalog" not in message


def test_create_llm_fast_fallback_carries_api_key():
    """Regression: the fast chain's emergency fallback honors a pinned BYO key.

    Without the passthrough, a client key forced the first two tiers to bill the
    client but silently dropped back onto the server's own key on emergency.
    """
    byo_key = "gsk_byo_client_key"
    chain = agent.llms._create_llm_fast(temperature=0.3, api_key=byo_key)

    def _secret(model):
        key = getattr(model, "groq_api_key", None)
        return getattr(key, "get_secret_value", lambda: key)()

    fallbacks = getattr(chain, "fallbacks", [])
    assert fallbacks, "expected an emergency fallback attached to the fast chain"

    fast, fallback = chain, fallbacks[-1]
    assert _secret(fast) == byo_key
    assert _secret(fallback) == byo_key
    fallback_model = getattr(fallback, "model_name", None) or getattr(fallback, "model", "")
    assert fallback_model == agent.llms._EMERGENCY_MODEL


# ==========================================
# MR2: deterministic requirement match table (agent/match.py)
# ==========================================

MR2_JD = (
    "TechCorp builds data platforms.\n"
    "## Requirements\n"
    "- 3+ years of Python development experience\n"
    "- Production experience with Docker and Kubernetes\n"
    "- Hands-on work with Apache Kafka for event streaming\n"
    "## What we offer\n"
    "- Free meals, gym membership, relocation support\n"
)
MR2_RESUME = "## Skills\nPython\n## Experience\nUsed Docker for local builds."


def test_requirement_table_classifies_covered_partial_gap():
    """The one-table answer to 'which stated requirements does this resume
    cover': classified in code with human reasons, sharing the ATS matcher's
    semantics - the calibrated ATS percentage is never touched by it."""
    rows = agent.compute_requirement_match_table(MR2_JD, MR2_RESUME)
    assert [row["status"] for row in rows] == ["covered", "partial", "gap"]
    assert rows[0]["requirement"] == "3+ years of Python development experience"
    assert rows[0]["evidence"] == ["python"]
    assert rows[0]["missing"] == []
    assert rows[0]["reason"] == "Covers all key terms (python)."
    assert rows[1]["requirement"] == "Production experience with Docker and Kubernetes"
    assert rows[1]["evidence"] == ["docker"]
    assert rows[1]["missing"] == ["kubernetes"]
    assert rows[1]["reason"] == "Covers docker; missing kubernetes."
    assert rows[2]["requirement"] == "Hands-on work with Apache Kafka for event streaming"
    assert rows[2]["evidence"] == []
    assert rows[2]["missing"] == ["kafka"]
    assert rows[2]["reason"] == "Missing from the resume: kafka."
    keys = {"requirement", "status", "evidence", "missing", "reason"}
    assert all(set(row) == keys for row in rows)


def test_requirement_table_role_filler_never_evidence():
    """'engineer'/'developer' name the role, not something a resume can carry
    evidence for (Nov 10 probe: they otherwise dominated reasons and told the
    user to 'add engineer')."""
    rows = agent.compute_requirement_match_table(
        "Python engineer with Docker and Kubernetes experience.", MR2_RESUME)
    assert len(rows) == 1
    assert rows[0]["missing"] == ["kubernetes"]
    joined = " ".join(term for row in rows for term in row["evidence"] + row["missing"])
    assert "engineer" not in joined and "developer" not in joined


def test_requirement_table_synonyms_and_semantic_matches_count():
    """Same expression-in-different-words semantics as the ATS score: 'k8s on
    containers' reads as docker/kubernetes, and the semantic verifier's
    reclassification set upgrades rows without a second LLM call."""
    rows = agent.compute_requirement_match_table(
        MR2_JD, "## Skills\nk8s on containers\n## Experience\n")
    docker_row = next(row for row in rows if "Docker" in row["requirement"])
    assert docker_row["status"] == "covered"
    assert "kubernetes" in docker_row["evidence"]
    assert not docker_row["missing"]

    rows = agent.compute_requirement_match_table(MR2_JD, MR2_RESUME, ["Kafka"])
    kafka_row = next(row for row in rows if "Kafka" in row["requirement"])
    assert kafka_row["status"] == "covered"
    assert kafka_row["evidence"] == ["kafka"]
    assert kafka_row["reason"] == "Covers all key terms (kafka)."


def test_requirement_table_skips_prose_and_perk_sections():
    """Marketing prose without Tool/Concept terms and perk strips never become
    rows: a front-page 'TechCorp builds data platforms' cannot pose as a
    requirement, and empty or no-vocabulary JDs yield zero rows."""
    rows = agent.compute_requirement_match_table(MR2_JD, MR2_RESUME)
    assert not any("TechCorp" in row["requirement"] for row in rows)
    assert not any("meals" in row["requirement"] or "gym" in row["requirement"]
                   for row in rows)
    assert agent.compute_requirement_match_table(
        "We value passion and teamwork and grow together.", "Python, Docker") == []
    assert agent.compute_requirement_match_table("", MR2_RESUME) == []


def test_requirement_table_caps_at_twelve_in_jd_order():
    """Requirement-dense postings truncate by first appearance (top-of-posting
    requirements lead any JD) and keep the JD's own order."""
    tools = ["redis", "nginx", "react", "vue", "tensorflow", "pytorch",
             "airflow", "spark", "hive", "presto", "kafka", "docker",
             "s3", "git"]
    jd = "## Requirements\n" + "\n".join(
        f"- Python experience with {tool}" for tool in tools) + "\n"
    rows = agent.compute_requirement_match_table(jd, "## Skills\nPython")
    assert len(rows) == 12
    assert [row["requirement"] for row in rows] == [
        f"Python experience with {tool}" for tool in tools[:12]]
    assert all(row["status"] == "partial" for row in rows)


def test_requirement_table_dedupes_repeated_lines():
    jd = ("## Requirements\n"
          "- Python services with Docker\n"
          "- Python services with Docker\n")
    rows = agent.compute_requirement_match_table(jd, "## Skills\nPython")
    assert len(rows) == 1


def test_requirement_table_segmentation_bullets_numbers_and_lines():
    """Segmentation contract: bullets and numbered items are single cards,
    heading markers never are, and plain non-bulleted skill lines don't bleed
    into their neighbors (golden-JD live evidence)."""
    jd = ("Requirements:\n"
          "- Python services with Docker\n"
          "1. Kubernetes operations and platform reliability\n"
          "## Duties\n"
          "Support data workflows and agent teams\n")
    rows = agent.compute_requirement_match_table(jd, "## Skills\nPython")
    requirements = [row["requirement"] for row in rows]
    assert "Python services with Docker" in requirements
    assert "Kubernetes operations and platform reliability" in requirements
    assert not any(row.startswith("Requirements") or row.startswith("Duties")
                   for row in requirements)
    assert not any("Support data workflows" in row for row in requirements)


def test_requirement_table_normalizes_literal_escaped_newlines():
    """The Sep 2026 escaped-\\n incident contract extends to the table:
    literal escape payloads classify identically to real newlines."""
    real = agent.compute_requirement_match_table(MR2_JD, MR2_RESUME)
    escaped = agent.compute_requirement_match_table(
        MR2_JD.replace("\n", "\\n"), MR2_RESUME)
    assert real and real == escaped


def test_graph_writes_requirement_match_table(fake_llms, monkeypatch):
    """MR2 wiring: ats_check computes the table every pass and the best-of
    guard keeps the WINNER's table - the shipped state always describes the
    shipped resume (a regressed second pass cannot ship a 'kubernetes missing'
    table for the good first draft)."""
    good = "## Skills\nPython, Docker, Kubernetes\n## Experience\nBuilt Python services on Kubernetes with Docker."
    worse = "## Skills\nPython\n## Experience\nWrote some scripts."
    result, seq = _run_with_drafts([good, worse], monkeypatch)

    rows = result.get("requirement_matches")
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert row["requirement"] == "Python engineer with Docker and Kubernetes experience."
    assert row["evidence"] == ["python", "docker", "kubernetes"]
    assert row["missing"] == []
    assert row["status"] == "covered"
    assert row["reason"] == "Covers all key terms (python, docker, kubernetes)."


# ==========================================
# Scoring truth: reviewer flattening + ATS link-target alignment
# ==========================================

def test_strip_markdown_for_review_flattens_all_syntax():
    f = agent.helpers._strip_markdown_for_review
    md = ("# Jane Doe\n"
          "jane@example.com | github.com/janedoe\n\n"
          "## Projects\n"
          "**Event Streamer** [Live Demo](https://jane.dev/stream) | Go | 2025\n"
          "- Built pipeline processing 10K events/sec with **true** metrics\n"
          "- Text with *italics* and `code` and - markers\n")
    flat = f(md)
    assert "#" not in flat
    assert "**" not in flat and "`" not in flat
    # Markdown link TARGETS are invisible to external parsers (the text layer
    # holds only the visible label) - targets must be dropped.
    assert "jane.dev/stream" not in flat
    # Bare-text URLs are rendered text: external parsers extract them too, so
    # they stay visible in the flattened view (only link targets disappear).
    assert "github.com/janedoe" in flat
    assert "Live Demo" in flat            # label text survives
    assert "Event Streamer" in flat
    assert "Jane Doe" in flat and "60 1c" not in flat


def test_reviewer_grades_flattened_text_with_anchors(monkeypatch):
    """Audit root cause of the 42/100: the reviewer graded RAW markdown.
    It must now read the flattened rendered-text view and anchor its grade
    to the fixed rubric reference points."""
    captured = []

    class CapturingReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    captured.append(" ".join(getattr(m, "content", "") for m in messages))
                    return agent.ReviewOutput(score=80, feedback="Tighten bullets.")
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", CapturingReviewer())
    from agent.nodes.reviewer import reviewer_node
    state = {
        "optimized_resume": (
            "# Jane Doe\njane@example.com | github.com/janedoe\n\n"
            "## Projects\n"
            "**Event Streamer** [Live Demo](https://github.com/jane/stream) | Go, Kafka | 2025\n"
            "- Processed 10K events/sec with sub-second latency\n"),
        "resume_text": ("Jane Doe\njane@example.com | github.com/janedoe\n\n"
                        "Event Streamer | Go, Kafka | 2025 (Live Demo)\n"
                        "- Processed 10K events/sec with sub-second latency\n"),
        "job_text": "Senior platform engineer with Kafka and Go experience.",
    }
    out = reviewer_node(state)
    assert out["llm_quality_score"] == 80
    prompt = captured[0]
    assert "## Projects" not in prompt          # markup must not reach the reviewer
    assert "**Event Streamer**" not in prompt
    assert "https://github.com/jane/stream" not in prompt   # link TARGET hidden
    assert "github.com/janedoe" in prompt       # bare contact URL is rendered text
    assert "Live Demo" in prompt and "Event Streamer" in prompt
    assert "Calibration anchors" in prompt
    # Run 14751ff9 guard: fabrication is graded, not just style.
    assert "TRUTHFULNESS GATE" in prompt
    assert "ORIGINAL RESUME" in prompt          # evidence block between JD and draft
    # Nov 7 run pair (d8756ddf/a9da3660): salvaged prose advised adding n8n /
    # Claude Code / MCP experience - advice only fabrication can follow, and
    # it fed the polish loop straight into the scrubber's removals.
    assert "FEEDBACK SCOPE" in prompt
    assert "Never recommend adding" in prompt


def test_ats_contact_line_links_still_visible_as_labels():
    flatten = agent.ats._flatten_link_targets
    assert flatten("[Live Demo](https://github.com/jane/docker-pipeline)") == "Live Demo"
    assert flatten("Contact [Email](mailto:jane@x.com) | [GitHub](https://gh.com/)") == \
        "Contact Email | GitHub"


def test_ats_no_keyword_credit_to_url_targets():
    """Audit: internal scoring credited keywords inside LINK TARGETS (URLs)
    that external parsers (Workday/Greenhouse/Lever) can never extract from
    the rendered PDF. Keywords must be visible in LABEL/bullet text only."""
    from agent.ats import compute_ats_match
    job = "We use docker for our CI/CD pipelines."

    url_only = "# Jane Doe\n## Projects\n- [Live Demo](https://github.com/jane/docker-pipeline)\n"
    r1 = compute_ats_match(job, url_only)
    docker1 = next(k for k in r1["keywords"] if k["keyword"] == "docker")
    assert docker1["found"] is False, r1["keywords"]

    labeled = "# Jane Doe\n## Projects\n- [Docker](https://github.com/jane/stream)\n"
    r2 = compute_ats_match(job, labeled)
    docker2 = next(k for k in r2["keywords"] if k["keyword"] == "docker")
    assert docker2["found"] is True
