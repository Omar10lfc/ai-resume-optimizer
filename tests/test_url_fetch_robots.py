"""url_fetch: robots.txt enforcement and the browser-header retry on the
job-fetch path (adapted from the MIT ai-job-search repo; see NOTICE).

No network: ``_validate_job_url`` and ``robots_check.gate`` are patched, and
``build_opener`` returns a scripted fake opener.
"""
import sys
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import url_fetch  # noqa: E402
from agent.robots_check import BROWSER  # noqa: E402


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self._body = body
        headers = Message()
        headers["Content-Type"] = "text/html"
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


def _patch_fetch(monkeypatch, gate_code=0, script=()):
    opener = _FakeOpener(script)
    monkeypatch.setattr(url_fetch, "_validate_job_url",
                        lambda raw: raw.strip())
    monkeypatch.setattr(url_fetch.robots_check, "gate",
                        lambda url: (gate_code, "ALLOWED" if gate_code == 0 else "DISALLOWED"))
    monkeypatch.setattr(url_fetch, "build_opener", lambda *a: opener)
    return opener


def test_robots_disallow_raises(monkeypatch):
    opener = _patch_fetch(monkeypatch, gate_code=1)
    with pytest.raises(ValueError, match="robots.txt policy"):
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert opener.uas == []  # the page itself was never fetched


def test_plain_200_returns_body(monkeypatch):
    opener = _patch_fetch(
        monkeypatch, script=[_FakeResponse(b"<html>JOB POSTING</html>")])
    body = url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "<html>JOB POSTING</html>" in body
    assert opener.uas == ["ResumeOptimizer/1.0"]


def test_403_retries_with_browser_ua(monkeypatch):
    from urllib.error import HTTPError
    opener = _patch_fetch(
        monkeypatch,
        script=[
            HTTPError("https://jobs.example.com/page", 403, "Forbidden",
                      Message(), None),
            _FakeResponse(b"<html>JOB POSTING</html>"),
        ],
    )
    body = url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert "<html>JOB POSTING</html>" in body
    assert opener.uas == ["ResumeOptimizer/1.0", BROWSER]


def test_500_is_not_retried(monkeypatch):
    from urllib.error import HTTPError
    opener = _patch_fetch(
        monkeypatch,
        script=[HTTPError("https://jobs.example.com/page", 500, "Server",
                          Message(), None)],
    )
    with pytest.raises(HTTPError):
        url_fetch._fetch_job_description("https://jobs.example.com/page")
    assert opener.uas == ["ResumeOptimizer/1.0"]
