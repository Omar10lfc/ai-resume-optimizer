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
from langchain_core.messages import AIMessage  # noqa: E402


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
    assert agent.MAX_ITERATIONS == 1
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


class FakeCreativeLLM:
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


def test_salvage_parses_score_from_failed_generation(fake_llms, monkeypatch):
    """Traces 01a15e3a/01a15f6d: on tool_use_failed the model's prose answer
    (score + feedback) sits in body.error.failed_generation. The reviewer must
    salvage it instead of flagging review_failed - and NOT spend a retry on
    identical bytes (trace 01a15f6d: retry folded identically 3s apart)."""
    probe = {"calls": 0}

    class FoldedReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    probe["calls"] += 1
                    raise _tool_use_failed_exc(_SALVAGE_GENERATION)

            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", FoldedReviewer())
    result = agent.full_app.invoke(_base_inputs())
    assert result["review_failed"] is False
    assert result["llm_quality_score"] == 80
    assert "Copilot" in result["feedback"]
    expected = round(agent.ATS_SCORE_WEIGHT * result["ats_percentage"]
                     + agent.LLM_SCORE_WEIGHT * 80)
    assert result["score"] == expected
    assert probe["calls"] == 1  # salvage short-circuits the retry


def test_salvage_skipped_when_generation_unparseable(fake_llms, monkeypatch):
    """A tool_use_failed payload WITHOUT an extractable score stays on the
    regular path: one same-payload retry, then review_failed."""
    probe = {"calls": 0}

    class FoldedReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    probe["calls"] += 1
                    raise _tool_use_failed_exc("I could not grade this resume.")

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

def _make_docx(path, paragraphs, tables=()):
    from docx import Document
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    for rows in tables:
        table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                table.rows[i].cells[j].text = cell
    doc.save(str(path))


def test_load_resume_docx_extracts_paragraphs_and_tables(tmp_path):
    doc = tmp_path / "resume.docx"
    _make_docx(doc, ["# Jane Doe", "## Skills: Python, Docker"], tables=[[["AWS", "Docker", "Git"]]])
    text = agent._load_resume_docx(str(doc))
    assert "# Jane Doe" in text
    assert "Python, Docker" in text
    assert "AWS | Docker | Git" in text


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
                {"id": "llama-3.1-8b-instant"},
            ]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse())
    ok, message = agent.llms._groq_models_status("gsk_ok")
    assert ok is True
    assert "API reachable" in message


def test_groq_models_status_flags_unavailable_primary_model(monkeypatch):
    """A 200 whose model list lacks the configured primary model is actionable."""
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
    assert "not among them" in message
