"""Golden ATS fixtures: checked-in JD/resume pairs replayed through the keyword
engine so extraction regressions are caught by tests before they show up as
bad traces or collapsed scores.

The JD is the real posting from the incident where literal `\n` escape
sequences reached the ATS engine and collapsed the score to 0. The fixtures
assert: (1) real-newline and literal-escaped extraction are identical, (2) the
required skills survive extraction while boilerplate/action-verb garbage does
not, and (3) a resume that demonstrably covers the requirements clears a
minimum match floor.

Run with: venv\\Scripts\\python.exe -m pytest tests/test_golden_ats.py -v
"""

import json
import sys
from pathlib import Path

import pytest

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "golden_ats.json").read_text(encoding="utf-8"))
JD_REAL = FIXTURE["job_description_real_newlines"]


def _escaped(jd: str) -> str:
    """Builds the bug-window payload: literal backslash-n sequences."""
    return jd.replace("\n", "\\n")


def test_golden_escaped_newlines_extract_identically():
    """THE incident: literal \n sequences previously swallowed the following
    word into nsupport/ndesign tokens. Extraction must be payload-shape blind."""
    real = agent._extract_keywords(JD_REAL, top_n=20)
    escaped = agent._extract_keywords(_escaped(JD_REAL), top_n=20)
    assert sorted(real) == sorted(escaped)


def test_golden_required_keywords_survive_extraction():
    real = agent._extract_keywords(JD_REAL, top_n=20)
    missing = sorted(set(FIXTURE["expect"]["keywords_present"]) - set(real))
    assert not missing, f"real keywords lost by the extractor: {missing}"


def test_golden_garbage_never_extracted():
    real = agent._extract_keywords(JD_REAL, top_n=50)
    leaked = sorted(set(FIXTURE["expect"]["keywords_absent"]) & set(real))
    assert not leaked, f"boilerplate/garbage leaked into keywords: {leaked}"


def test_golden_minimum_match_floor():
    """A resume covering the requirements must clear a floor under the weighted
    matcher (catches matcher regressions that damage every real run)."""
    result = agent.compute_ats_match(JD_REAL, FIXTURE["synthetic_resume"])
    assert result["percentage"] >= FIXTURE["minimum_ats_percentage"], (
        f"ATS match collapsed below floor: {result['percentage']:.1f}% "
        f"(matched={result['match_count']}/{result['total']})")


def test_golden_hard_requirements_count_matched():
    """Reviews keywords that must register as found (the skills the synthetic
    resume demonstrably covers)."""
    result = agent.compute_ats_match(JD_REAL, FIXTURE["synthetic_resume"])
    found = {r["keyword"] for r in result["keywords"] if r["found"]}
    must_count = [k for k in FIXTURE["resume_keywords_that_must_count"]
                  if k in {r["keyword"] for r in result["keywords"]}]
    lost = sorted(set(must_count) - found)
    assert not lost, f"skills the resume demonstrably has, scored missing: {lost}"


def test_golden_corporate_filler_and_verbs_never_keywords():
    """Verifies that non-skills (translate, stakeholders, datasets, fresh, graduate,
    decisions, organizations, business, solutions) are never extracted as keywords,
    and technical skills (.NET, Python, SQL, Docker) are prioritized."""
    jd = (
        "Lamdax develops scalable digital solutions that streamline business processes "
        "and help organizations turn data into actionable decisions. "
        "Collaborate with engineers and stakeholders to translate requirements into AI solutions. "
        "Prepare, clean, and analyze datasets for AI model development. "
        "Fresh graduate with a bachelor's degree in Computer Science. "
        "Working knowledge of Python, SQL, and .NET. Docker and APIs experience."
    )
    kws = agent._extract_keywords(jd, top_n=20)
    for garbage in ("translate", "stakeholders", "datasets", "fresh", "graduate",
                    "organizations", "decisions", "business", "solutions"):
        assert garbage not in kws, f"Garbage '{garbage}' leaked into extracted keywords"
    for expected in ("python", "sql", ".net", "docker"):
        assert expected in kws, f"Core skill '{expected}' missing from extracted keywords"

