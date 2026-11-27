"""Tests for Issue 3: stop improver/polisher from emitting placeholders and
strip any surviving placeholders ('Location not specified', 'not specified',
'N/A', 'TBD', 'unknown', '[...]') before final PDF rendering across CORPUS_13.
"""
import re
from pathlib import Path

import pdfplumber
import pytest

from agent.latex_render import HAVE_PDFLATEX, latex_to_pdf, render_resume_tex
from agent.nodes.improver import _build_improver_prompt, _patch_stack
from agent.nodes.polisher import _build_section_prompt
from agent.resume_parser import parse_markdown_resume
from agent.resume_units import parse_resume_units
from tests.test_layout_parity import CORPUS_13


PLACEHOLDER_RES = [
    re.compile(r"\blocation\s+not\s+specified\b", re.IGNORECASE),
    re.compile(r"\bnot\s+specified\b", re.IGNORECASE),
    re.compile(r"(?<![a-z0-9/])n\s*/\s*a(?![a-z0-9/])", re.IGNORECASE),
    re.compile(r"\btbd\b", re.IGNORECASE),
    re.compile(r"\bunknown\b", re.IGNORECASE),
    re.compile(r"\[\s*\.\.\.\s*\]"),
]


def _extract_pdf_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _assert_no_placeholders(text: str, label: str = "output") -> None:
    for pat in PLACEHOLDER_RES:
        m = pat.search(text)
        assert not m, f"Placeholder {pat.pattern!r} matched {m.group(0)!r} in {label}"


def test_prompts_forbid_placeholder_fields():
    """Improver and polisher prompts must explicitly instruct the LLM to omit
    absent optional fields rather than emitting placeholder phrases."""
    sample_md = CORPUS_13[0][2]
    units = parse_resume_units(sample_md)
    state = {
        "resume_text": sample_md,
        "job_text": "Python Backend Engineer",
        "human_notes": "",
        "missing_skills": "",
        "feedback": "Tighten wording",
    }
    imp_prompt = _build_improver_prompt(state, units).lower()
    pol_prompt = _build_section_prompt(
        state, "## Experience", "Software Engineer | Acme | Jan 2024", "", "Tighten"
    ).lower()
    for prompt in (imp_prompt, pol_prompt):
        assert "not specified" in prompt
        assert "n/a" in prompt
        assert "tbd" in prompt


def test_patch_stack_strips_surviving_placeholders_without_breaking_links():
    """Post-processing removes placeholder phrases across contact, headings,
    certifications, and inline fields while preserving URLs and real fields."""
    source_md = """# Jordan Rivera
jordan@example.com | [GitHub](https://github.com/jrivera)

## Experience
AI Engineer | Horizon Labs | Jun 2024 -- Present
- Built retrieval API handling 500 req/s in Python.

## Projects
DocPipeline | [Live Demo](https://example.com/unknown-path) | Jan 2025
- Automated ingestion with FastAPI and Redis.

## Education
State Tech University | B.S. in CS | 2020 -- 2024

## Certifications
Cloud Architect - CloudOrg
ML Foundations - Academy
"""
    draft_with_placeholders = """# Jordan Rivera
jordan@example.com | Location not specified | N/A | TBD | github.com/jrivera

## Experience
**AI Engineer** | Horizon Labs | Location not specified | Jun 2024 -- Present
- Built retrieval API handling 500 req/s in Python [unknown].

## Projects
**DocPipeline** | [Live Demo](https://example.com/unknown-path) | [...] | Jan 2025
- Automated ingestion with FastAPI and Redis (N/A).

## Education
**State Tech University** | Not Specified | B.S. in CS | 2020 -- 2024

## Certifications
Cloud Architect - CloudOrg (TBD)
ML Foundations - Academy (not specified)
"""
    state = {
        "resume_text": source_md,
        "optimized_resume": source_md,
        "job_text": "Python engineer",
        "human_notes": "",
        "missing_skills": "",
    }
    cleaned, *_ = _patch_stack(state, draft_with_placeholders)
    # Remove real URL targets before placeholder assertion so URL path check stays accurate.
    cleaned_visible = re.sub(r"https?://\S+", "", cleaned)
    _assert_no_placeholders(cleaned_visible, "cleaned draft")
    assert "Horizon Labs" in cleaned
    assert "Jun 2024 -- Present" in cleaned
    assert "https://example.com/unknown-path" in cleaned
    assert "State Tech University" in cleaned
    assert "Cloud Architect" in cleaned


def _inject_placeholders_into_corpus_md(md: str, idx: int) -> str:
    """Inject realistic LLM-emitted placeholder fields into a corpus resume."""
    phrases = [
        ("Location not specified", "TBD"),
        ("Not Specified", "N/A"),
        ("Unknown", "[...]"),
    ]
    loc_ph, tag_ph = phrases[idx % len(phrases)]
    lines = md.splitlines()
    out = []
    injected_contact = False
    injected_exp = False
    for line in lines:
        if not injected_contact and "@" in line and "|" in line:
            line = f"{line} | {loc_ph} | {tag_ph}"
            injected_contact = True
        elif not injected_exp and "|" in line and not line.startswith("#") and not line.startswith("-"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                parts.insert(2, loc_ph)
                line = " | ".join(parts)
                injected_exp = True
        out.append(line)
    out.append(f"\n## Certifications\nCore Practitioner - Institute ({tag_ph})\n")
    return "\n".join(out)


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_corpus_13_no_placeholders_in_final_pdf(tmp_path: Path):
    """Across all 13 corpus resumes (clean + placeholder-injected through
    _patch_stack and rendered to PDF), zero placeholder phrases ship in PDF text."""
    for idx, (slug, _pages, md) in enumerate(CORPUS_13):
        # 1) Clean corpus resume -> PDF has no placeholders
        clean_data = parse_markdown_resume(md)
        clean_tex = render_resume_tex(clean_data)
        clean_dir = tmp_path / f"{slug}_clean"
        clean_pdf = latex_to_pdf(clean_tex, clean_dir, "clean.pdf")
        assert clean_pdf, f"pdflatex failed on clean {slug}"
        _assert_no_placeholders(_extract_pdf_text(clean_pdf), f"clean PDF {slug}")

        # 2) Placeholder-injected draft -> _patch_stack + parse/render -> PDF has no placeholders
        dirty_md = _inject_placeholders_into_corpus_md(md, idx)
        state = {
            "resume_text": md,
            "optimized_resume": md,
            "job_text": "Software Engineer Python",
            "human_notes": "",
            "missing_skills": "",
        }
        cleaned_md, *_ = _patch_stack(state, dirty_md)
        data = parse_markdown_resume(cleaned_md)
        tex = render_resume_tex(data)
        dirty_dir = tmp_path / f"{slug}_patched"
        pdf_path = latex_to_pdf(tex, dirty_dir, "Optimized_Resume.pdf")
        assert pdf_path, f"pdflatex failed on patched {slug}"
        _assert_no_placeholders(_extract_pdf_text(pdf_path), f"final PDF {slug}")
