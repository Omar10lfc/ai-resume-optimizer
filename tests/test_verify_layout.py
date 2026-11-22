"""Offline layout measurement tests for agent/verify_layout.py.

Mirrors tools/verify_layout.py in the MIT ai-job-search repo (adapted).
Geometry cases are built from synthetic Page/Line structures so the suite
needs neither Poppler nor a compiled PDF; integration tests on real LaTeX
output run only when HAVE_PDFLATEX (like tests/test_pdf_verify.py).
"""
import os
import subprocess
from pathlib import Path

import pytest

from agent import verify_layout as vl
from agent.latex_render import HAVE_PDFLATEX, latex_to_pdf, parse_markdown_resume, render_resume_tex
from agent.verify_layout import Line, Page, find_orphans, parse_pdf, report, text_start

A4_HEIGHT = 842.0


def line(top, left=50.0, height=10.0, text="x"):
    return Line(top=top, bottom=top + height, left=left, height=height, text=text)


def at(top, left, text, text_left=None, bottom=None):
    """A bbox line; text_left is where its text starts after a merged marker."""
    bottom = top + 14.0 if bottom is None else bottom
    return Line(top=top, bottom=bottom, left=left, height=bottom - top, text=text,
                text_left=text_left)


@pytest.fixture(autouse=True)
def _clear_pdftotext_cache():
    vl.find_pdftotext.cache_clear()
    yield
    vl.find_pdftotext.cache_clear()


@pytest.fixture(autouse=True)
def _enable_layout_measurement(monkeypatch):
    """Measurement tests opt in; the default-off gate has its own tests."""
    monkeypatch.setattr(vl, "LAYOUT_VERIFY_ENABLED", True)


# ==========================================
# Gap + bottom-space measurement (the "internal hole")
# ==========================================

class TestGapAndBottomSpace:
    def setup_method(self):
        # Three body lines, then a 322pt jump, then the page-number footer.
        self.holed = Page(
            A4_HEIGHT,
            [line(50), line(64), line(78), line(400), line(770, text="1/2")],
        )

    def test_largest_gap_reports_size_and_position(self):
        gap, y = self.holed.largest_gap()
        assert round(gap) == 322 and round(y) == 78

    def test_bottom_space_ignores_the_footer_band(self):
        # Measured to the last body line (y410), not the footer at y770.
        assert round(self.holed.bottom_space) == 432

    def test_page_with_no_body_lines_is_empty(self):
        assert Page(A4_HEIGHT, []).empty

    def test_report_flags_the_hole(self):
        rows = report(Path("synthetic"), [self.holed])
        assert any(row["kind"] == "hole" for row in rows), rows


# ==========================================
# Footer band
# ==========================================

class TestFooterBand:
    def test_single_line_in_band_is_just_the_page_number(self):
        assert not Page(A4_HEIGHT, [line(50), line(800, text="2/2")]).footer_crowded

    def test_two_lines_in_band_means_body_text_spilled_in(self):
        # \\enlargethispage pushing body text over the footer.
        assert Page(A4_HEIGHT, [line(50), line(780), line(800)]).footer_crowded


# ==========================================
# Heading + indent detection (document margin)
# ==========================================

class TestHeadingAndIndentDetection:
    def setup_method(self):
        self.page = Page(
            A4_HEIGHT,
            [
                line(50, height=16.0, text="Professional Experience"),
                line(80, left=50.0),
                line(94, left=70.0),
            ],
        )

    def test_taller_line_is_a_heading(self):
        assert self.page.is_heading(self.page.body[0])
        assert not self.page.is_heading(self.page.body[1])

    def test_left_edge_separates_bullets_from_headers(self):
        assert self.page.is_indented(self.page.body[2])
        assert not self.page.is_indented(self.page.body[1])

    def test_indent_is_judged_against_the_document_margin(self):
        # A page that OPENS with bullets must not mistake their indent for its margin.
        page = Page(A4_HEIGHT, [line(50, left=50.0), line(700, left=50.0, text="Data Analyst")])
        assert page.is_indented(line(60, left=70.0))


# ==========================================
# find_orphans (entry headers split across page breaks)
# ==========================================

class TestOrphansBasics:
    def test_page_ending_on_a_section_heading(self):
        # a heading stranded at the bottom, content overleaf
        p1 = Page(A4_HEIGHT, [line(50), line(64), line(700, height=16.0, text="Education")])
        p2 = Page(A4_HEIGHT, [line(60, text="Example University"), line(74, left=70.0)])
        assert any(kind == "stranded_heading" for kind, _ in find_orphans([p1, p2]))

    def test_entry_header_orphaned_from_its_bullets(self):
        # the job-entry title on one page, its bullets on the next
        q1 = Page(A4_HEIGHT, [line(50), line(700, left=50.0, text="Software Engineer")])
        q2 = Page(A4_HEIGHT, [line(60, left=70.0, text="- built the thing")])
        assert any(kind == "orphaned_entry_header" for kind, _ in find_orphans([q1, q2]))

    def test_lone_list_marker_is_a_split_bullet_not_an_orphaned_header(self):
        # moderncv gives the itemize marker its own bbox line: different defect, different fix
        m1 = Page(A4_HEIGHT, [line(50), line(700, left=50.0, text="\u25cf")])
        m2 = Page(A4_HEIGHT, [line(60, left=70.0, text="continued item text here")])
        assert any(kind == "split_list_marker" for kind, _ in find_orphans([m1, m2]))

    def test_clean_break_reports_nothing(self):
        r1 = Page(A4_HEIGHT, [line(50), line(700, left=50.0)])
        r2 = Page(A4_HEIGHT, [line(60, left=50.0)])
        assert find_orphans([r1, r2]) == []


class TestOrphansReadTextX:
    """The orphan rule compares where the TEXT starts, not where the line starts."""

    def test_bullet_wrapping_across_the_break_is_not_an_orphaned_header(self):
        p1 = Page(A4_HEIGHT, [
            at(50, 59.5, "Plant Grower Example Garden 1981-1990"),
            at(700, 58.9, "\u2022 Replanted the long border with plants raised", 72.1),
        ])
        p2 = Page(A4_HEIGHT, [at(60, 72.1, "from seed, lifting the old clumps")])
        assert find_orphans([p1, p2]) == []

    def test_bullet_whose_text_opens_with_a_symbol_is_still_a_wrapped_bullet(self):
        p1 = Page(A4_HEIGHT, [
            at(50, 59.5, "Plant Grower Example Garden 1981-1990"),
            at(700, 58.9, "\u2022 (in winter) relaid the gravel paths along", 72.1),
        ])
        p2 = Page(A4_HEIGHT, [at(60, 72.1, "the old orchard wall and round the pond")])
        assert find_orphans([p1, p2]) == []

    def test_header_level_with_merged_bullets_is_an_orphan(self):
        # the bullet line starts at x58.9 (left of the header), its text at x72.1
        p1 = Page(A4_HEIGHT, [at(700, 59.5, "Kitchen Gardener 1975-1980")])
        p2 = Page(A4_HEIGHT, [at(60, 58.9, "\u2022 Worked in the kitchen garden", 72.1)])
        assert any(kind == "orphaned_entry_header" for kind, _ in find_orphans([p1, p2]))


# ==========================================
# text_start
# ==========================================

class TestTextStart:
    def test_a_leading_list_marker_is_set_aside(self):
        assert text_start([(58.9, 0, 1, "\u2022"), (72.1, 0, 1, "Laid")]) == 72.1

    def test_no_marker_means_no_separate_text_start(self):
        assert text_start([(59.5, 0, 1, "Kitchen"), (94.9, 0, 1, "Gardener")]) is None

    def test_ascii_dashes_and_other_glyphs_are_text(self):
        for glyph in ("-", "*", "\u2013", "\u2014", "\u25cb"):
            assert text_start([(69.4, 0, 1, glyph), (77.7, 0, 1, "[Event]")]) is None

    def test_a_bare_marker_has_no_text(self):
        assert text_start([(59.0, 0, 1, "\u2022")]) is None


# ==========================================
# parse_pdf on synthetic -bbox output
# ==========================================

SYNTHETIC_BBOX = (
    '<page width="612" height="792">'
    '<word xMin="50" yMin="100" xMax="200" yMax="112">Hello</word>'
    '<word xMin="200" yMin="100" xMax="300" yMax="112">world</word>'
    '</page>'
    '<page width="612" height="792">'
    '<word xMin="50" yMin="50" xMax="150" yMax="62">Second</word>'
    '<word xMin="51" yMin="62" xMax="160" yMax="74">page</word>'
    '</page>'
)


def _fake_run(cmd, *args, **kwargs):
    if "-h" in cmd:
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Usage: pdftotext -bbox", stderr="")
    return subprocess.CompletedProcess(cmd, 0, stdout=SYNTHETIC_BBOX, stderr="")


class TestParsePdfSynthetic:
    def test_parses_lines_from_synthetic_output(self, monkeypatch):
        monkeypatch.setattr(vl.shutil, "which", lambda cmd, path=None: "/usr/bin/pdftotext")
        monkeypatch.setattr(vl.subprocess, "run", _fake_run)
        pages = parse_pdf(Path("synthetic.pdf"))
        assert len(pages) == 2
        first = pages[0]
        assert len(first.lines) == 1
        line0 = first.lines[0]
        assert line0.text == "Hello world"
        assert round(line0.top) == 100 and round(line0.bottom) == 112
        assert round(line0.left) == 50
        assert round(line0.height) == 12
        # No list marker merged: text starts at the line's left.
        assert line0.text_left is None
        assert round(first.height) == 792


def _fake_bbox_with_marker():
    return (
        '<page width="612" height="792">'
        '<word xMin="50" yMin="100" xMax="200" yMax="112">\u2022</word>'
        '<word xMin="72.1" yMin="100" xMax="220" yMax="112">Body text starts here</word>'
        '</page>'
    )


class TestFindPdfText:
    def test_first_candidate_lacks_bbox_and_second_has_it(self, monkeypatch):
        """Walks PATH, probes with -h, runs the working candidate (#655)."""
        def fake_which(cmd, path=None):
            if path == "/git/bin":
                return "/git/bin/pdftotext"
            if path == "/miktex/bin":
                return "/miktex/bin/pdftotext"
            return None

        def fake_run(cmd, *args, **kwargs):
            binary = cmd[0]
            if "-h" in cmd:
                if binary == "/git/bin/pdftotext":
                    return subprocess.CompletedProcess(cmd, 99, stdout="", stderr="Usage: pdftotext ...")
                if binary == "/miktex/bin/pdftotext":
                    return subprocess.CompletedProcess(
                        cmd, 0, stdout="Usage: pdftotext ...\n  -bbox : output bounding box", stderr="")
            if "-bbox" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout=SYNTHETIC_BBOX, stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(os, "get_exec_path", lambda: ["/git/bin", "/miktex/bin"])
        monkeypatch.setattr(vl.shutil, "which", fake_which)
        monkeypatch.setattr(vl.subprocess, "run", fake_run)
        pages = parse_pdf(Path("x.pdf"))
        assert len(pages) == 2

    def test_missing_poppler_raises_skippable(self, monkeypatch):
        monkeypatch.setattr(vl.shutil, "which", lambda cmd, path=None: None)
        with pytest.raises(RuntimeError, match="not found"):
            parse_pdf(Path("x.pdf"))


class TestMeasurePdfLayout:
    def test_clean_pages_report_nothing(self, monkeypatch, tmp_path):
        pdf = tmp_path / "synthetic.pdf"
        pdf.write_bytes(b"%PDF-fake")
        pages = [Page(A4_HEIGHT, [line(50), line(64), line(78)])]
        monkeypatch.setattr(vl, "find_pdftotext", lambda: "pdftotext")
        monkeypatch.setattr(vl, "parse_pdf", lambda path: pages)
        assert vl.measure_pdf_layout(pdf) == []

    def test_skips_when_pdftotext_is_unavailable(self, monkeypatch, tmp_path):
        pdf = tmp_path / "synthetic.pdf"
        pdf.write_bytes(b"%PDF-fake")
        monkeypatch.setattr(vl, "find_pdftotext", lambda: None)
        assert vl.measure_pdf_layout(pdf) == []

    def test_flags_hole(self, monkeypatch, tmp_path):
        holed = Page(A4_HEIGHT, [line(50), line(64), line(78), line(400), line(770, text="1/2")])
        pdf = tmp_path / "synthetic.pdf"
        pdf.write_bytes(b"%PDF-fake")
        monkeypatch.setattr(vl, "find_pdftotext", lambda: "pdftotext")
        monkeypatch.setattr(vl, "parse_pdf", lambda path: [holed])
        rows = vl.measure_pdf_layout(pdf)
        assert any(row["kind"] == "hole" for row in rows), rows


class TestEnvironmentGate:
    def test_disabled_returns_empty_without_touching_poppler(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(vl, "LAYOUT_VERIFY_ENABLED", False)
        pdf = tmp_path / "synthetic.pdf"
        pdf.write_bytes(b"%PDF-fake")
        monkeypatch.setattr(
            vl, "parse_pdf",
            lambda path: pytest.fail("parse_pdf must not run when disabled"))
        assert vl.measure_pdf_layout(pdf) == []

    def test_enabled_with_missing_poppler_never_raises(
            self, monkeypatch, tmp_path):
        pdf = tmp_path / "synthetic.pdf"
        pdf.write_bytes(b"%PDF-fake")
        monkeypatch.setattr(vl, "find_pdftotext", lambda: None)
        assert vl.measure_pdf_layout(pdf) == []


class TestMeasureOnDeliberatelyBadPdf:
    def test_large_internal_gap_is_flagged_on_a_real_one_page_pdf(
            self, tmp_path):
        if vl.find_pdftotext() is None:
            pytest.skip("Poppler pdftotext not installed")
        canvas = pytest.importorskip("reportlab.pdfgen.canvas")
        pdf = tmp_path / "bad_layout.pdf"
        c = canvas.Canvas(str(pdf), pagesize=(612, 792))
        c.drawString(72, 750, "Alex Candidate")
        c.drawString(72, 730, "Experience")
        c.drawString(72, 710, "Built Python services")
        c.drawString(72, 120, "Education")  # ~590pt hole above this line
        c.showPage()
        c.save()

        rows = vl.measure_pdf_layout(pdf)

        assert [row["kind"] for row in rows] == ["hole"], rows


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
class TestMeasureOnRealResume:
    """Integration: measuring a real rendered resume never raises, rows have keys."""

    def test_measure_compiled_resume(self, tmp_path):
        data = parse_markdown_resume("# Test Resume\n\nTest University, Degree")
        tex = render_resume_tex(data)
        pdf = latex_to_pdf(tex, tmp_path, "Optimized_Resume.pdf")
        assert pdf is not None
        rows = vl.measure_pdf_layout(pdf)
        assert isinstance(rows, list)
        for row in rows:
            assert "kind" in row and "detail" in row
