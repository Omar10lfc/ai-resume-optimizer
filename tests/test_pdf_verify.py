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
from agent.pdf_verify import (find_non_ascii_date_ranges, normalize_text,
                              verify_pdf)  # noqa: E402


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
# find_non_ascii_date_ranges (quantity carve-outs)
# ==========================================
# Ported from tests/test_verify_pdf.py in the MIT ai-job-search repo
# (FindNonAsciiDateRangesTests): the date-range rule reads the raw layer;
# `normalize_text` folds and cannot detect.


def test_en_dash_between_years_is_reported_with_its_code_point():
    hits = find_non_ascii_date_ranges("Six Sigma Green Belt, 2016–2024.\n")
    assert hits == [("Six Sigma Green Belt, 2016–2024.", "\u2013")]


def test_ascii_hyphen_range_is_clean():
    assert find_non_ascii_date_ranges("2016-2024\nMar 2016 - Jul 2016\n") == []


@pytest.mark.parametrize("text", ["Mar 2016 – Jul 2016", "2016 – Present", "\u2013 2024"])
def test_a_year_on_either_side_of_the_dash_is_enough(text):
    # Month-qualified and open-ended ranges: the year is only on one side.
    assert len(find_non_ascii_date_ranges(text)) == 1


@pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2012", "\u2014", "\u2015", "\u2212"])
def test_other_unicode_dashes_and_the_minus_sign_are_caught(dash):
    assert find_non_ascii_date_ranges(f"2016{dash}2024")[0][1] == dash


def test_numeric_range_without_a_year_is_not_a_date():
    # 05-cv-templates.md keeps `--` in prose ranges like EUR 600k--1M.
    assert find_non_ascii_date_ranges("EUR 600k\u20131M, 12\u201315 people") == []


@pytest.mark.parametrize("text", ["Grew budget DKK 120000\u2013200000",
                                  "Scaled ingestion from 12000\u201315000 events/s"])
def test_year_like_digits_inside_a_longer_number_are_not_a_year(text):
    # `2000` inside `120000`/`12000` is not a year-shaped boundary.
    assert find_non_ascii_date_ranges(text) == []


@pytest.mark.parametrize("text", [
    "Salary EUR 1500\u20132000 per month",
    "Scaled the API to serve 2000\u20135000 requests per second",
    "Grew users from 2000\u20135000",
    "Budget 1.5\u20132.0M, headcount 20\u201325",
])
def test_numeric_range_with_year_shaped_ends_is_a_quantity_not_a_date(text):
    assert find_non_ascii_date_ranges(text) == []


@pytest.mark.parametrize("symbol", ["\u20ac", "\u00a3", "$", "\u00a5"])
@pytest.mark.parametrize("text", [
    "Budget {s}2000\u2013{s}5000 per month",
    "Budget 1500{s}\u20132000{s} per month",
    "Budget {s}2000\u2013{s}2024 per month",
])
def test_attached_currency_symbols_make_numeric_ranges_quantities(text, symbol):
    assert find_non_ascii_date_ranges(text.format(s=symbol)) == []


@pytest.mark.parametrize("text", [
    "03/2016\u201307/2024",
    "01.2016 \u2013 12.2024",
    "2016-03\u20132024-07",
    "2016/17\u20132018/19",
    "2016/03\u20132024/07",
    "01.03.2016\u201331.07.2024",
    "2016\u201307/2024",
])
def test_numerically_formatted_date_ranges_are_still_caught(text):
    # A composite token is not a bare number, so the quantity rule does not
    # apply and the year rule fires - these are the Workday-drops-the-date
    # shape the check exists for.
    assert len(find_non_ascii_date_ranges(text)) == 1


@pytest.mark.parametrize("text", ["MSc Physics, 2016\u201317", "Sep 2016\u201317",
                                  "2019\u201320 season", "2016 \u2013 17"])
def test_year_and_two_digit_year_is_a_date_range(text):
    # The academic-year/season form: `17` is a bare non-year number, so the
    # quantity rule would otherwise suppress the flag.
    assert len(find_non_ascii_date_ranges(text)) == 1


def test_two_digit_number_before_a_year_is_still_a_quantity():
    # Only year-first is the date form; `17-2016` is not a date anyone writes.
    assert find_non_ascii_date_ranges("items 17\u20132016") == []


@pytest.mark.parametrize("text", ["2016\u20132024", "1990\u20132010 programme",
                                  "Roles 2000\u20132024"])
def test_two_years_joined_by_a_dash_are_still_a_date_range(text):
    assert len(find_non_ascii_date_ranges(text)) == 1


def test_a_standalone_year_next_to_a_longer_number_is_still_caught():
    assert find_non_ascii_date_ranges("2016\u20132024 (120000 users)")[0][1] == "\u2013"


def test_year_ending_a_line_is_not_joined_to_the_next_lines_dash():
    text = "Heading 1988-1994\n\u2013 note\n"
    assert find_non_ascii_date_ranges(text) == []


def test_dash_ending_a_line_is_not_joined_to_the_next_lines_year():
    text = "Six Sigma \u2013\n2016 onwards\n"
    assert find_non_ascii_date_ranges(text) == []


def test_same_line_spaces_around_the_dash_are_still_caught():
    assert len(find_non_ascii_date_ranges("1988 \u2013 1994")) == 1


def test_hits_are_reported_in_document_order_one_per_range():
    text = "2016\u20132024 role\nmore text\nJan 2010 \u2013 Dec 2012 degree\n"
    assert [line for line, _ in find_non_ascii_date_ranges(text)] == [
        "2016\u20132024 role", "Jan 2010 \u2013 Dec 2012 degree"]


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
@pytest.mark.pdf_render
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
@pytest.mark.pdf_render
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


# ==========================================
# Geometric bounds & accent-folded literals
# ==========================================

def test_normalize_text_preserves_accents_by_default_and_folds_when_requested():
    assert normalize_text("Jos\u00e9 \u00c1lvarez-Mu\u00f1oz") == "jos\u00e9 \u00e1lvarez-mu\u00f1oz"
    assert normalize_text("Jos\u00e9 \u00c1lvarez-Mu\u00f1oz", fold_accents=True) == "jose alvarez-munoz"


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
@pytest.mark.pdf_render
def test_non_ascii_name_passes_required_text_in_latex_pdf(tmp_path):
    md = (
        "# Jos\u00e9 \u00c1lvarez-Mu\u00f1oz\n"
        "jose.alvarez@example.ch\n\n"
        "## Experience\n"
        "Engineer | M\u00fcller-Stra\u00dfe GmbH | Jan 2022 - Present\n"
        "- Built Python pipelines.\n"
    )
    pdf = latex_to_pdf(render_resume_tex(parse_markdown_resume(md)), tmp_path, "intl.pdf")
    assert pdf is not None
    res = verify_pdf(pdf, expected_pages=1, required_literals=["Jos\u00e9 \u00c1lvarez-Mu\u00f1oz", "jose.alvarez@example.ch"])
    checks = _by_name(res)
    assert checks["required_text"]["ok"] is True
    assert checks["geometry"]["ok"] is True
    assert res["ok"] is True


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
@pytest.mark.pdf_render
def test_geometry_catches_horizontal_and_bottom_overflow(tmp_path):
    # Right-margin overflow past 612pt physical sheet edge
    tex_right = (
        "\\documentclass[letterpaper,10pt]{article}"
        "\\begin{document}"
        "\\noindent\\makebox[720pt][l]{Start \\hfill OverflowedRightEdgeWord}"
        "\\end{document}"
    )
    pdf_r = latex_to_pdf(tex_right, tmp_path, "ovf_r.pdf")
    assert pdf_r is not None
    checks_r = _by_name(verify_pdf(pdf_r, expected_pages=1))
    assert checks_r["geometry"]["ok"] is False
    assert "edge" in checks_r["geometry"]["detail"] or "margin" in checks_r["geometry"]["detail"]

    # Bottom physical clip overflow past 792pt
    tex_bot = (
        "\\documentclass[letterpaper,10pt]{article}"
        "\\begin{document}"
        "\\vspace*{690pt} BottomClippedLine"
        "\\end{document}"
    )
    pdf_b = latex_to_pdf(tex_bot, tmp_path, "ovf_b.pdf")
    assert pdf_b is not None
    checks_b = _by_name(verify_pdf(pdf_b, expected_pages=1))
    assert checks_b["geometry"]["ok"] is False
    assert "bottom clip" in checks_b["geometry"]["detail"]


def test_export_documents_retries_then_falls_back_to_html_on_geometry_failure(monkeypatch, tmp_path):
    latex_pdf = tmp_path / "latex.pdf"
    compact_pdf = tmp_path / "compact.pdf"
    html_pdf = tmp_path / "html.pdf"
    cover_pdf = tmp_path / "cover.pdf"
    for p in (latex_pdf, compact_pdf, html_pdf, cover_pdf):
        p.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(_exporter, "_export_resume", lambda *a, **k: {
        "resume_pdf_path": str(latex_pdf),
        "resume_pages": 1,
        "resume_compacted": False,
        "render_engine": "latex@1",
    })
    monkeypatch.setattr(_exporter._latex, "render_compact_tiers", lambda *a, **k: {
        "pdf_path": str(compact_pdf),
        "tex": "% compact",
        "tier": 1,
        "pages": 1,
    })
    monkeypatch.setattr(_exporter._latex, "latex_to_docx", lambda *a, **k: None)
    monkeypatch.setattr(_exporter._html, "HAVE_BROWSER", True)
    import agent.config as _cfg
    monkeypatch.setattr(_cfg, "RESUME_RENDERER", "auto")
    monkeypatch.setattr(_exporter, "_export_resume_html", lambda *a, **k: {
        "resume_pdf_path": str(html_pdf),
        "resume_pages": 1,
        "resume_compacted": False,
        "render_engine": "html-chromium@1",
    })
    monkeypatch.setattr(_exporter, "_html_to_pdf", lambda *a, **k: str(cover_pdf))
    monkeypatch.setattr(_exporter, "_resume_required_literals", lambda *a, **k: [])

    def fake_verify(path, expected_pages=1, required_literals=()):
        name = Path(path).name
        if name in ("latex.pdf", "compact.pdf"):
            return {"pages": 1, "ok": False, "checks": [
                {"name": "geometry", "ok": False, "detail": "p1 off right sheet edge (734.9>612pt)"}]}
        return {"pages": 1, "ok": True, "checks": [
            {"name": "geometry", "ok": True, "detail": "ok"}]}

    monkeypatch.setattr(_exporter, "verify_pdf", fake_verify)

    got = _exporter.export_documents("# A\na@b.com\n## Skills\nPython\n", "Dear,\n", tmp_path)
    assert got["render_engine"] == "html-chromium@1"
    names = [c["name"] for c in got["pdf_checks"]]
    assert "render_retry" in names
    assert "engine_fallback" in names


def test_export_documents_blocks_and_unlinks_when_both_engines_fail_geometry(monkeypatch, tmp_path):
    latex_pdf = tmp_path / "bad.pdf"
    cover_pdf = tmp_path / "cover.pdf"
    latex_pdf.write_bytes(b"%PDF-1.4")
    cover_pdf.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(_exporter, "_export_resume", lambda *a, **k: {
        "resume_pdf_path": str(latex_pdf),
        "resume_pages": 1,
        "resume_compacted": True,
        "render_engine": "latex@1",
    })
    monkeypatch.setattr(_exporter._html, "HAVE_BROWSER", False)
    monkeypatch.setattr(_exporter, "_html_to_pdf", lambda *a, **k: str(cover_pdf))
    monkeypatch.setattr(_exporter, "_resume_required_literals", lambda *a, **k: [])
    monkeypatch.setattr(_exporter, "verify_pdf", lambda path, **k: {
        "pages": 1, "ok": False, "checks": [
            {"name": "geometry", "ok": False, "detail": "p1 off right sheet edge"}]
    } if Path(path).name == "bad.pdf" else {"pages": 1, "ok": True, "checks": []})

    with pytest.raises(_exporter.PdfGeometryError, match="failed geometric layout bounds"):
        _exporter.export_documents("# A\na@b.com\n", "Dear,\n", tmp_path)
    assert not latex_pdf.exists()
