"""Regression tests for PDF hyperlink extraction, layout/parsing repairs, and
polisher/cover-letter attribution guards (Nov 24 2026 run-23 findings).
"""
from pathlib import Path

from agent.helpers import normalize_markdown_links, preserve_contact_links
from agent.html_render import _simple_rows
from agent.latex_render import (
    _is_plain_label_line,
    parse_markdown_resume,
    render_resume_tex,
)
from agent.nodes.polisher import _build_section_prompt
from agent.url_fetch import _load_resume_pdf


def _link_substr(c, x: float, y: float, full: str, target: str, url: str, font="Helvetica", size=10) -> None:
    start = full.index(target)
    x0 = x + c.stringWidth(full[:start], font, size)
    x1 = x0 + c.stringWidth(target, font, size)
    c.linkURL(url, (x0, y - 2, x1, y + size), relative=0)


def _build_pdf_with_right_aligned_project_links(dest: Path) -> None:
    """Create a 1-page PDF using reportlab with left-aligned titles and
    right-aligned clickable 'Live Demo' links + multi-link contact header."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(dest), pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawString(240, 750, "Omar Example")
    hdr = "linkedin.com/in/omar | github.com/omar | huggingface.co/omar"
    c.drawString(120, 735, hdr)
    _link_substr(c, 120, 735, hdr, "linkedin.com/in/omar", "https://linkedin.com/in/omar")
    _link_substr(c, 120, 735, hdr, "github.com/omar", "https://github.com/omar")
    _link_substr(c, 120, 735, hdr, "huggingface.co/omar", "https://huggingface.co/omar")

    c.drawString(36, 705, "PROJECTS")
    p1 = "Port Twin (Grad Project) Sep 2025 - Jun 2026 | Live Demo |"
    c.drawString(36, 688, p1)
    _link_substr(c, 36, 688, p1, "Live Demo", "https://demo.example.com/port")
    c.drawString(36, 675, "GitHub")
    _link_substr(c, 36, 675, "GitHub", "GitHub", "https://github.com/omar/port")
    c.drawString(36, 662, "- Built OR-Tools CP-SAT solver cutting wait time by 47%.")

    p2 = "Fil-RAG-Goal: Arabic Football Q&A Feb 2026 - Jun 2026 | Live Demo"
    c.drawString(36, 642, p2)
    _link_substr(c, 36, 642, p2, "Live Demo", "https://hf.co/spaces/omar/fil-rag")
    c.drawString(36, 629, "- Boosted MRR 7% with hybrid dense/BM25 retrieval.")

    c.drawString(36, 600, "CERTIFICATIONS")
    cert = "McKinsey Forward Program - Dec 2024"
    c.drawString(36, 583, cert)
    _link_substr(c, 36, 583, cert, "McKinsey Forward Program", "https://drive.google.com/drive/folders/abc_123")
    c.save()


def test_load_resume_pdf_preserves_right_aligned_and_contact_links(tmp_path):
    pdf_path = tmp_path / "sample_resume.pdf"
    _build_pdf_with_right_aligned_project_links(pdf_path)

    extracted = _load_resume_pdf(str(pdf_path))
    assert "https://linkedin.com/in/omar" in extracted
    assert "https://github.com/omar" in extracted
    assert "https://huggingface.co/omar" in extracted
    assert "[Live Demo](https://demo.example.com/port)" in extracted
    assert "[GitHub](https://github.com/omar/port)" in extracted
    assert "[Live Demo](https://hf.co/spaces/omar/fil-rag)" in extracted
    assert "https://drive.google.com/drive/folders/abc_123" in extracted

    # Ensure downstream normalization keeps every distinct project & cert link.
    norm = normalize_markdown_links(extracted)
    norm, *_ = preserve_contact_links(norm, extracted)
    assert "[Live Demo](https://demo.example.com/port)" in norm
    assert "[GitHub](https://github.com/omar/port)" in norm
    assert "[Live Demo](https://hf.co/spaces/omar/fil-rag)" in norm

    parsed = parse_markdown_resume(norm)
    tex = render_resume_tex(parsed)
    assert "https://demo.example.com/port" in tex
    assert "https://github.com/omar/port" in tex
    assert "https://hf.co/spaces/omar/fil-rag" in tex
    assert r"https://drive.google.com/drive/folders/abc\%5F123" in tex
    assert "https}{:" not in tex


def test_normalize_joins_wrapped_skill_rows():
    raw = (
        "SKILLS\n"
        "Machine Learning & Deep Learning: Python, SQL, Model Fine-Tuning,\n"
        "Neural Networks\n"
        "Cloud & MLOps: AWS, Docker\n"
    )
    norm = normalize_markdown_links(raw)
    assert "Model Fine-Tuning, Neural Networks\n" in norm
    assert "\nNeural Networks\n" not in norm


def test_plain_label_line_and_simple_rows_ignore_markdown_url_schemes():
    short_cert = "[McKinsey Forward Program](https://drive.google.com/folders/1) - Dec 2024"
    assert not _is_plain_label_line("[GitHub](https://github.com/omar/repo)")
    assert not _is_plain_label_line("[Mail](mailto:test@example.com)")
    assert not _is_plain_label_line(short_cert)

    parsed = parse_markdown_resume(f"# Candidate\n\n## Certifications\n{short_cert}\n")
    tex = render_resume_tex(parsed)
    assert r"\href{https://drive.google.com/folders/1}{\underline{McKinsey Forward Program}}" in tex
    assert "https}{:" not in tex

    rows = _simple_rows([short_cert])
    assert len(rows) == 1
    is_label, html = rows[0]
    assert is_label is False
    assert 'href="https://drive.google.com/folders/1"' in html


def test_polisher_prompt_guards_summary_length_and_cross_project_conflation():
    prompt = _build_section_prompt(
        state={"resume_text": "Original", "human_notes": ""},
        header_line="## Professional Summary",
        section_body="AI Engineer.",
        rest_of_draft="## Projects\n**Proj A**\n- Cut costs 70%.",
        feedback="Quantify achievements.",
    )
    assert "max" in prompt.lower() and "55 words" in prompt
    assert "NEVER transfer or merge one project's metrics" in prompt
