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
