"""url_fetch: robots.txt enforcement and redirect validation on the job-fetch
path (adapted from the MIT ai-job-search repo; see NOTICE).

No network: ``robots_check._fetch`` serves literal robots.txt bodies (the
gate itself stays real), ``build_opener`` returns a scripted fake opener, and
``_validate_job_url`` is patched only where DNS would otherwise be needed.
"""
import sys
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import robots_check, url_fetch  # noqa: E402

PAGE_UA = robots_check.USER_AGENT


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200,
                 location: str | None = None):
        self.status = status
        self._body = body
        headers = Message()
        headers["Content-Type"] = "text/html"
        if location:
            headers["Location"] = location
        self.headers = headers

    def read(self, n: int = -1) -> bytes:
        return self._body[:n] if n >= 0 else self._body


class _FakeOpener:
    """Scripted opener: raises the next item in `script` if it is an
    exception, otherwise returns it as the response."""

    def __init__(self, script):
        self._script = list(script)
        self.uas = []

    def open(self, request, timeout=None):
        self.uas.append(request.get_header("User-agent"))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_fetch(monkeypatch, script=(), robots=("", 404),
                 patch_validate=True):
    """Patch page transport and robots.txt transport; the gate stays real."""
    opener = _FakeOpener(script)
    robots_calls = []

    def fake_robots_fetch(url, ua):
        robots_calls.append((url, ua))
        return robots(url) if callable(robots) else robots

    if patch_validate:
        monkeypatch.setattr(url_fetch, "_validate_job_url",
                            lambda raw: raw.strip())
    monkeypatch.setattr(robots_check, "_fetch", fake_robots_fetch)
    monkeypatch.setattr(url_fetch, "build_opener", lambda *a: opener)
    return opener, robots_calls


def _real_validator(monkeypatch, private_hosts=()):
    """Keep the real validator; fake DNS and allowlist for example.com."""
    monkeypatch.setattr(url_fetch, "ALLOWED_JOB_URL_DOMAINS", {"example.com"})
    monkeypatch.setattr(
        url_fetch, "_is_public_host",
        lambda host: host not in set(private_hosts),
    )
    real = url_fetch._validate_job_url
    seen = []

    def spy(url):
        seen.append(url)
        return real(url)

    monkeypatch.setattr(url_fetch, "_validate_job_url", spy)
    return seen


def _redirect_error(url: str, location: str) -> Exception:
    from urllib.error import HTTPError
    headers = Message()
    headers["Location"] = location
    return HTTPError(url, 302, "Found", headers, None)


# ==========================================
# robots.txt gate (real gate, fake transport)
# ==========================================

def test_robots_allow_fetches_page_with_one_honest_user_agent(monkeypatch):
    opener, robots_calls = _patch_fetch(
        monkeypatch,
        script=[_FakeResponse(b"<html>JOB POSTING</html>")],
        robots=("User-agent: *\nAllow: /\n", 200),
    )
    body = url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "JOB POSTING" in body
    assert opener.uas == [PAGE_UA]
    assert robots_calls == [("https://jobs.example.com/robots.txt", PAGE_UA)]


def test_robots_disallow_blocks_before_the_page_is_fetched(monkeypatch):
    opener, _ = _patch_fetch(
        monkeypatch,
        robots=("User-agent: *\nDisallow: /\n", 200),
    )
    with pytest.raises(ValueError, match="robots.txt policy") as excinfo:
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "Paste the job description" in str(excinfo.value)
    assert opener.uas == []


def test_unreadable_robots_is_refused(monkeypatch):
    opener, _ = _patch_fetch(
        monkeypatch, robots=("<html>404 not found</html>", 200))
    with pytest.raises(ValueError, match="robots.txt policy"):
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert opener.uas == []


# ==========================================
# Status handling
# ==========================================

def test_plain_200_returns_body(monkeypatch):
    opener, _ = _patch_fetch(
        monkeypatch, script=[_FakeResponse(b"<html>JOB POSTING</html>")])
    body = url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "<html>JOB POSTING</html>" in body
    assert opener.uas == [PAGE_UA]


def test_403_raises_actionable_error_without_retry(monkeypatch):
    from urllib.error import HTTPError
    opener, _ = _patch_fetch(
        monkeypatch,
        script=[HTTPError("https://jobs.example.com/page", 403, "Forbidden",
                          Message(), None)],
    )
    with pytest.raises(ValueError, match="blocked automated access") as excinfo:
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "paste the description" in str(excinfo.value).lower()
    assert opener.uas == [PAGE_UA]  # one request, no browser-UA retry


def test_500_is_not_retried(monkeypatch):
    from urllib.error import HTTPError
    opener, _ = _patch_fetch(
        monkeypatch,
        script=[HTTPError("https://jobs.example.com/page", 500, "Server",
                          Message(), None)],
    )
    with pytest.raises(HTTPError):
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert opener.uas == [PAGE_UA]


# ==========================================
# Redirects
# ==========================================

def test_page_opener_disables_implicit_redirects(monkeypatch):
    captured = {}
    opener = _FakeOpener([_FakeResponse(b"<html>JOB</html>")])

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return opener

    monkeypatch.setattr(url_fetch, "_validate_job_url", lambda raw: raw.strip())
    monkeypatch.setattr(robots_check, "_fetch", lambda url, ua: ("", 404))
    monkeypatch.setattr(url_fetch, "build_opener", fake_build_opener)

    url_fetch._fetch_job_description("https://jobs.example.com/page")

    handlers = captured["handlers"]
    assert len(handlers) == 1
    assert isinstance(handlers[0], url_fetch._NoRedirect)
    assert url_fetch._NoRedirect().redirect_request(
        None, None, 302, "Found", None, "https://elsewhere.example/") is None


def test_every_redirect_hop_is_validated_and_gated(monkeypatch):
    seen = _real_validator(monkeypatch)
    opener, robots_calls = _patch_fetch(
        monkeypatch,
        script=[
            _FakeResponse(b"", status=302,
                          location="https://example.com/job/step2"),
            _redirect_error("https://example.com/job/step2",
                            "https://example.com/job/final"),
            _FakeResponse(b"<html>JOB POSTING</html>"),
        ],
        robots=("", 404),
        patch_validate=False,
    )

    body = url_fetch._fetch_job_description("https://example.com/job")

    assert "JOB POSTING" in body
    assert seen == [
        "https://example.com/job",
        "https://example.com/job/step2",
        "https://example.com/job/final",
    ]
    # The robots gate ran for the initial URL and for both redirect hops.
    assert [url for url, _ in robots_calls] == [
        "https://example.com/robots.txt",
        "https://example.com/robots.txt",
        "https://example.com/robots.txt",
    ]
    assert opener.uas == [PAGE_UA, PAGE_UA, PAGE_UA]


def test_redirect_to_private_address_is_rejected_before_fetch(monkeypatch):
    seen = _real_validator(monkeypatch,
                           private_hosts={"internal.example.com"})
    opener, _ = _patch_fetch(
        monkeypatch,
        script=[
            _FakeResponse(b"", status=302,
                          location="https://internal.example.com/private"),
        ],
        robots=("", 404),
        patch_validate=False,
    )

    with pytest.raises(ValueError, match="public IP"):
        url_fetch._fetch_job_description("https://example.com/job")

    assert seen == [
        "https://example.com/job",
        "https://internal.example.com/private",
    ]
    assert opener.uas == [PAGE_UA]  # the private hop was never fetched
