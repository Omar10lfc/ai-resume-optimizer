"""Tracing is off by default in the shipped template; the in-app note covers it."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_env_example_tracing_off_by_default():
    for line in _read(".env.example").splitlines():
        stripped = line.strip()
        if stripped.startswith("LANGCHAIN_TRACING_V2="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'").lower()
            assert value in {"false", "0", "no", "off", ""}, stripped
            return
    raise AssertionError("LANGCHAIN_TRACING_V2 missing from .env.example")


def test_tracing_on_note_documents_storage_and_privacy():
    src = _read("agent/tracing.py")
    assert "Tracing enabled" in src
    assert "resume/JD content is stored in local trace copies" in src
    assert "disable tracing for strict privacy" in src
