"""PDF text-layer verification + ASCII date normalization.

Pure logic tests run everywhere (the reader is faked). The LaTeX checks skip
when pdflatex is absent, mirroring test_latex_render.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import pdf_verify  # noqa: E402
from agent.latex_render import (  # noqa: E402
    HAVE_PDFLATEX,
    latex_to_pdf,
    parse_markdown_resume,
    render_resume_tex,
)
from agent.nodes import pdf_exporter as _exporter  # noqa: E402
from agent.pdf_verify import normalize_text, verify_pdf  # noqa: E402


# ==========================================
# Fake reader helpers
# ==========================================

class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, texts):
        self.pages = [_FakePage(text) for text in texts]


def _fake_reader(monkeypatch, texts):
    monkeypatch.setattr(pdf_verify, "PdfReader", lambda _path: _FakeReader(texts))


def _by_name(result):
    return {check["name"]: check for check in result["checks"]}


def _write_blank_pdf(path, pages=1):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as handle:
        writer.write(handle)
    return str(path)


# ==========================================
# verify_pdf - structural checks
# ==========================================

def test_unreadable_file_is_a_failed_check_not_an_exception(tmp_path):
    result = verify_pdf(str(tmp_path / "missing.pdf"))

    assert result["ok"] is False
    assert result["pages"] == 0
    assert len(result["checks"]) == 1
    assert result["checks"][0]["name"] == "readable"
    assert result["checks"][0]["ok"] is False


def test_blank_pdf_has_no_text_layer(tmp_path):
    path = _write_blank_pdf(tmp_path / "blank.pdf")

    result = verify_pdf(path, expected_pages=1)
    checks = _by_name(result)

    assert result["pages"] == 1
    assert checks["readable"]["ok"] is True
    assert checks["text_layer"]["ok"] is False
    assert "page(s) 1" in checks["text_layer"]["detail"]
    assert checks["page_count"]["ok"] is True
    assert result["ok"] is False


def test_page_count_contract(tmp_path, monkeypatch):
    _fake_reader(monkeypatch, ["one", "two"])

    strict = verify_pdf("ignored.pdf", expected_pages=1)
    loose = verify_pdf("ignored.pdf", expected_pages=None)

    assert _by_name(strict)["page_count"]["ok"] is False
    assert "expected 1" in _by_name(strict)["page_count"]["detail"]
    assert _by_name(loose)["page_count"]["ok"] is True
    assert loose["ok"] is True


def test_cid_and_replacement_damage_flagged(monkeypatch):
    _fake_reader(monkeypatch, ["Jake Ryan", "(cid:12) (cid:34)", "ok \ufffd"])

    result = verify_pdf("ignored.pdf")
    integrity = _by_name(result)["text_integrity"]

    assert integrity["ok"] is False
    assert "CID" in integrity["detail"]
    assert "U+FFFD" in integrity["detail"]
    assert result["ok"] is False


# ==========================================
# ASCII dates - raw layer, no folding
# ==========================================

@pytest.mark.parametrize("date_text", [
    "Sep 2015 \u2013 Jun 2019",
    "2016\u20142024",
    "2016\u20142024",
    "Aug 2020 \u2212 Present",
    "\u2013 2021",
])
def test_non_ascii_date_separators_flagged(monkeypatch, date_text):
    _fake_reader(monkeypatch, ["Header", date_text])

    checks = _by_name(verify_pdf("ignored.pdf"))

    assert checks["ascii_dates"]["ok"] is False
    assert "non-ASCII date separator" in checks["ascii_dates"]["detail"]


def test_ascii_dates_pass(monkeypatch):
    _fake_reader(monkeypatch, ["Experience", "2016-2024", "Sep 2025 - Jun 2026"])

    checks = _by_name(verify_pdf("ignored.pdf"))

    assert checks["ascii_dates"]["ok"] is True


def test_adjacent_year_and_number_with_hyphen_not_mistaken_for_range(monkeypatch):
    # A plain hyphen between a year and a number is the correct ASCII form.
    _fake_reader(monkeypatch, ["reduced 2024-25 costs by 12-15%"])

    assert _by_name(verify_pdf("ignored.pdf"))["ascii_dates"]["ok"] is True


# ==========================================
# Required literals - conservative fold
# ==========================================

def test_required_literals_pass_through_typographic_fold(monkeypatch):
    _fake_reader(monkeypatch, [
        "Jake Ryan\nJake \ufb01 Ryan \u2014 Pro\ufb01cient in Python\njake@su.edu"
    ])

    result = verify_pdf("ignored.pdf", required_literals=[
        "Jake Ryan", "Proficient in Python", "jake@su.edu"])

    checks = _by_name(result)
    assert checks["required_text"]["ok"] is True
    assert result["ok"] is True


def test_missing_required_literal_reported(monkeypatch):
    _fake_reader(monkeypatch, ["Jake Ryan", "Python"])

    result = verify_pdf("ignored.pdf", required_literals=["Jake Ryan", "Kubernetes"])

    required = _by_name(result)["required_text"]
    assert required["ok"] is False
    assert "Kubernetes" in required["detail"]
    assert "Jake Ryan" not in required["detail"]


def test_normalize_text_folds_whitespace_and_punctuation():
    assert normalize_text("  Jake\u00a0Ryan \u2014 Pro\ufb01cient ") == \
        "jake ryan - proficient"


# ==========================================
# Date normalization in the renderer
# ==========================================

def _entry_fields(entry):
    return [entry.header or "", entry.right or "",
            entry.sub or "", entry.sub_right or ""]


@pytest.mark.parametrize("separator", ["\u2013", "\u2014", "--", "\u2212"])
def test_parsed_experience_date_is_ascii(separator):
    md = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Experience\n"
        f"**Software Engineer** | Acme Corp | 2016{separator}2024\n"
        "- Built things\n"
    )

    data = parse_markdown_resume(md)
    fields = _entry_fields(data.sections[-1].entries[0])

    assert "2016-2024" in fields
    for field in fields:
        assert not any(dash in field for dash in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212")


def test_parsed_standalone_month_range_is_ascii():
    md = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Projects\n"
        "**Live Dashboard** | Python, React | Aug \u2013 Sep 2026\n"
        "- Shipped it\n"
    )

    data = parse_markdown_resume(md)
    fields = _entry_fields(data.sections[-1].entries[0])

    assert "Aug - Sep 2026" in fields


def test_rendered_tex_carries_no_typographic_date_dash():
    md = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Experience\n"
        "**Software Engineer** | Acme Corp | 2016\u20142024\n"
        "- Built things\n"
    )

    tex = render_resume_tex(parse_markdown_resume(md))

    assert "2016-2024" in tex
    assert "2016\u20142024" not in tex


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_compiled_resume_text_layer_passes_and_dates_are_ascii(tmp_path):
    md = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Experience\n"
        "**Software Engineer** | Acme Corp | 2016\u20142024\n"
        "- Built things with Python\n"
    )

    pdf = latex_to_pdf(render_resume_tex(parse_markdown_resume(md)),
                       tmp_path, "dates.pdf")
    assert pdf is not None

    result = verify_pdf(pdf, expected_pages=1,
                        required_literals=["Jake Ryan", "jake@su.edu"])

    checks = _by_name(result)
    assert checks["text_layer"]["ok"] is True
    assert checks["ascii_dates"]["ok"] is True
    assert checks["required_text"]["ok"] is True
    assert result["ok"] is True


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_compiled_en_dash_is_detected(tmp_path):
    # `--` in LaTeX source is the en dash ATS imports choke on. The renderer
    # no longer emits it for dates; this proves the verifier would catch it.
    tex = "\\documentclass{article}\\begin{document}2016--2024\\end{document}"

    pdf = latex_to_pdf(tex, tmp_path, "en_dash.pdf")
    assert pdf is not None

    result = verify_pdf(pdf, expected_pages=1)

    assert _by_name(result)["ascii_dates"]["ok"] is False
    assert "2016" in _by_name(result)["ascii_dates"]["detail"]


# ==========================================
# Exporter wiring
# ==========================================

def test_resume_literals_are_name_email_and_literal_keywords(monkeypatch):
    resume = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Skills\n"
        "Python, FastAPI\n"
    )
    monkeypatch.setattr(_exporter, "compute_ats_match", lambda _j, _r: {
        "keywords": [
            {"keyword": "Python", "found": True, "method": "exact"},
            {"keyword": "Kubernetes", "found": True, "method": "synonym"},
            {"keyword": "FastAPI", "found": True, "method": "stem"},
        ],
    })

    literals = _exporter._resume_required_literals(resume, "job text")

    assert literals[:2] == ["Jake Ryan", "jake@su.edu"]
    assert "Python" in literals
    assert "FastAPI" in literals
    # Credited only through a synonym - the PDF cannot be required to spell it.
    assert "Kubernetes" not in literals


def test_verify_exported_pdfs_targets_resume_and_cover(monkeypatch, tmp_path):
    resume_pdf = tmp_path / "resume.pdf"
    cover_pdf = tmp_path / "cover.pdf"
    resume_pdf.write_bytes(b"%PDF-1.4")
    cover_pdf.write_bytes(b"%PDF-1.4")
    calls = []

    def fake_verify(path, expected_pages=1, required_literals=()):
        calls.append((Path(path).name, expected_pages, tuple(required_literals)))
        if Path(path).name == "resume.pdf":
            return {"pages": 1, "ok": True, "checks": [
                {"name": "text_layer", "ok": True, "detail": "ok"}]}
        return {"pages": 1, "ok": False, "checks": [
            {"name": "ascii_dates", "ok": False, "detail": "2016\u20132024"}]}

    monkeypatch.setattr(_exporter, "verify_pdf", fake_verify)

    checks = _exporter._verify_exported_pdfs(
        str(resume_pdf), str(cover_pdf), ["Jake Ryan"])

    assert calls == [("resume.pdf", 1, ("Jake Ryan",)), ("cover.pdf", 1, ())]
    assert [c["target"] for c in checks] == ["resume", "cover"]
    assert checks[0]["ok"] is True
    assert checks[1]["ok"] is False


def test_verify_exported_pdfs_skips_missing_files(monkeypatch, tmp_path):
    cover_pdf = tmp_path / "cover.pdf"
    cover_pdf.write_bytes(b"%PDF-1.4")
    calls = []

    def fake_verify(path, expected_pages=1, required_literals=()):
        calls.append(path)
        return {"pages": 1, "ok": True, "checks": []}

    monkeypatch.setattr(_exporter, "verify_pdf", fake_verify)

    checks = _exporter._verify_exported_pdfs(str(tmp_path / "gone.pdf"),
                                             str(cover_pdf), [])

    assert calls == [str(cover_pdf)]
    assert checks == []


def test_export_documents_passes_job_text_to_verification(monkeypatch, tmp_path):
    """The export core calls the verifier with the job text for literals."""
    seen = {}

    def fake_export_resume(resume_text, request_dir):
        return {"resume_pdf_path": str(tmp_path / "r.pdf"), "resume_pages": 1}

    monkeypatch.setattr(_exporter, "_export_resume_latex", fake_export_resume)
    monkeypatch.setattr(_exporter, "_export_resume_html", fake_export_resume)
    monkeypatch.setattr(_exporter, "_html_to_pdf",
                        lambda *a, **k: str(tmp_path / "c.pdf"))
    (tmp_path / "r.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")

    def fake_literals(resume_text, job_text=""):
        seen["job_text"] = job_text
        return ["Jake Ryan"]

    monkeypatch.setattr(_exporter, "_resume_required_literals", fake_literals)
    monkeypatch.setattr(_exporter, "verify_pdf", lambda *a, **k: {
        "pages": 1, "ok": True, "checks": []})

    got = _exporter.export_documents(
        "# Jake Ryan\njake@su.edu\n", "Dear team,\n", tmp_path,
        job_text="We require Python")

    assert seen["job_text"] == "We require Python"
    assert got["pdf_checks"] == []
