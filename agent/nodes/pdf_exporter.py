"""NODE 8: PDF Exporter - saves resume and cover letter as styled PDFs/DOCX.

BB2 (Nov 14 2026): the resume renders through the engine ``RESUME_RENDERER``
selects - the HTML/CSS theme printed by headless Chromium (``agent/
html_render.py``, the Dossier steal) when one is available, the bundled
LaTeX template otherwise (the classic theme, unchanged). Both paths keep
the full artifact contract (PDF + DOCX + .tex source); the HTML path still
writes the .tex source (Overleaf/advanced editing) and the DOCX (pandoc)
from the same parsed data. The cover letter keeps the simpler HTML->PDF
path. If every renderer fails, exports still work via the xhtml2pdf
fallback. The chosen engine is stamped ``render_engine`` for forensics.

BB1 Phase B (PB2): the export core is factored into ``export_documents`` so
the history panel can REGENERATE artifacts from a past run's stored text
(the pipeline node and the server endpoint share one implementation).
"""
from pathlib import Path
from uuid import uuid4

from ..config import OUTPUT_ROOT
from ..state import AgentState
from ..helpers import _safe_print, _html_to_pdf
from .. import latex_render as _latex
from .. import html_render as _html


def _export_resume_latex(resume_text: str, request_dir: Path) -> dict:
    """The classic LaTeX path (historical behavior, byte-for-byte)."""
    resume_pdf_path = None
    resume_docx_path = None
    resume_tex_path = None
    resume_pages = None
    resume_compacted = False
    page_overflow_warning = ""
    try:
        data = _latex.parse_markdown_resume(resume_text)
        tex_content = _latex.render_resume_tex(data)
        # Save the .tex source alongside the compiled outputs (useful for
        # advanced editing in Overleaf/TeX editors).
        tex_file = None
        if tex_content:
            request_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            tex_file = request_dir / "Optimized_Resume.tex"
            tex_file.write_text(tex_content, encoding="utf-8")
            resume_tex_path = str(tex_file)

        resume_pdf_path = _latex.latex_to_pdf(
            tex_content, request_dir, "Optimized_Resume.pdf")
        resume_docx_path = _latex.latex_to_docx(
            tex_content, request_dir, "Optimized_Resume.docx")

        # --- Enforced 1-page containment ---
        # The internal bar is ONE page: 2+ page resumes silently lose the
        # bottom of the document and recruiters only scan page 1. Count the
        # produced PDF and, on overflow, re-render with the spacing-only
        # auto-compact pass before accepting the output.
        if resume_pdf_path:
            resume_pages = _latex.pdf_page_count(resume_pdf_path)
            if resume_pages and resume_pages > 1:
                _safe_print(
                    f"   [LaTeX] Compiled {resume_pages} pages - "
                    "trying auto-compact tiers (airy first).")
                tiered = _latex.render_compact_tiers(
                    data, request_dir, "Optimized_Resume.pdf")
                if tiered:
                    # Adopt the winning tier everywhere (tex source, DOCX).
                    if tex_file is not None:
                        tex_file.write_text(tiered["tex"], encoding="utf-8")
                    compact_docx = _latex.latex_to_docx(
                        tiered["tex"], request_dir, "Optimized_Resume.docx")
                    if compact_docx:
                        resume_docx_path = compact_docx
                    resume_pdf_path = tiered["pdf_path"]
                    resume_compacted = True
                    resume_pages = 1
                    _safe_print(
                        f"   [LaTeX] Adopted auto-compact tier {tiered['tier']} "
                        f"({'airy' if tiered['tier'] == 1 else 'tight'} spacing).")
                else:
                    # No tier reclaimed one page: keep the first
                    # (good-compiling) render and surface the overflow.
                    page_overflow_warning = (
                        f"Optimized resume is {resume_pages} pages after compaction attempt. "
                        "Internal target is 1 page - trim content in Step 2 review or hide low-value sections.")
                    _safe_print(f"   [WARN] {page_overflow_warning}")
    except Exception as e:
        _safe_print(f"   [LaTeX Render] Error: {e}. Falling back to HTML→PDF for resume.")

    return {
        "resume_pdf_path": resume_pdf_path or "",
        "resume_docx_path": resume_docx_path or "",
        "resume_tex_path": resume_tex_path or "",
        "resume_pages": resume_pages or 0,
        "resume_compacted": resume_compacted,
        "page_overflow_warning": page_overflow_warning,
        "render_engine": "latex@1",
    }


def _export_resume_html(resume_text: str, request_dir: Path) -> dict | None:
    """The BB2 HTML/CSS path (the Dossier steal).

    Renders the resume through the CSS-density theme printed by headless
    Chromium; the .tex source + DOCX still ride the LaTeX tooling (no
    pdflatex needed) so the artifact contract stays intact. Returns None
    when the browser is unavailable or the print fails - the caller falls
    back to the LaTeX path, so an export never fails because of the theme.
    """
    try:
        data = _latex.parse_markdown_resume(resume_text)
        page = _html.render_resume_html(data)
        resume_pdf_path = _html.html_to_pdf(
            page, request_dir, "Optimized_Resume.pdf")
        if not resume_pdf_path:
            return None

        resume_pages = _latex.pdf_page_count(resume_pdf_path)
        resume_compacted = False
        page_overflow_warning = ""
        density_used = 0
        # Same enforced 1-page containment as the LaTeX path, measured with
        # the SAME pdfplumber convention: on overflow the density-tier
        # ladder probes tighter CSS and adopts the first tier that fits one
        # page with the geometry invariants held.
        if resume_pages and resume_pages > 1:
            _safe_print(
                f"   [HTML] Rendered {resume_pages} pages - "
                "trying density tiers (airy-tight first).")
            tiered = _html.render_density_tiers(
                data, request_dir, "Optimized_Resume.pdf")
            if tiered:
                resume_pdf_path = tiered["pdf_path"]
                resume_compacted = True
                resume_pages = 1
                density_used = tiered["density"]
                _safe_print(
                    f"   [HTML] Adopted density tier {tiered['density']} "
                    "(tighter typography).")
            else:
                page_overflow_warning = (
                    f"Optimized resume is {resume_pages} pages after compaction attempt. "
                    "Internal target is 1 page - trim content in Step 2 review or hide low-value sections.")
                _safe_print(f"   [WARN] {page_overflow_warning}")

        # Artifact contract: .tex source (advanced editing/Overleaf) + DOCX
        # (pandoc - no pdflatex needed) still generated from the same data.
        resume_tex_path = ""
        resume_docx_path = None
        try:
            tex_content = _latex.render_resume_tex(data)
            if tex_content:
                request_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                tex_file = request_dir / "Optimized_Resume.tex"
                tex_file.write_text(tex_content, encoding="utf-8")
                resume_tex_path = str(tex_file)
                resume_docx_path = _latex.latex_to_docx(
                    tex_content, request_dir, "Optimized_Resume.docx")
        except Exception as e:
            _safe_print(f"   [HTML] .tex/DOCX side artifacts skipped: {e}")

        _safe_print(f"   [Renderer] HTML theme (Chromium), density tier "
                    f"{density_used} (0 = base).")
        return {
            "resume_pdf_path": resume_pdf_path,
            "resume_docx_path": resume_docx_path or "",
            "resume_tex_path": resume_tex_path,
            "resume_pages": resume_pages or 0,
            "resume_compacted": resume_compacted,
            "page_overflow_warning": page_overflow_warning,
            "render_engine": _html.HTML_RENDERER_VERSION,
        }
    except Exception as e:
        _safe_print(f"   [HTML Render] Error: {e}. Falling back to LaTeX.")
        return None


def _export_resume(resume_text: str, request_dir: Path) -> dict:
    """Engine selection: the config knob routes the resume to the HTML/CSS
    theme (when a browser is available and the print succeeds) or the
    classic LaTeX path. ``latex`` pins the historical behavior exactly;
    ``html`` forces the theme with the LaTeX fallback; ``auto`` prefers the
    theme whenever a browser was found."""
    from .. import config
    mode = config.RESUME_RENDERER
    if mode in ("html", "auto"):
        if mode == "auto" and not _html.HAVE_BROWSER:
            result = None  # no browser: straight to the classic theme
        else:
            result = _export_resume_html(resume_text, request_dir)
        if result is not None:
            return result
    return _export_resume_latex(resume_text, request_dir)


def export_documents(resume_text: str, cover_text: str, request_dir: Path) -> dict:
    """Renders the optimized resume + cover letter into ``request_dir``.

    Returns exactly the keys the exporter node writes back into the graph
    state (file paths as strings, "" when an export failed; the trailing
    keys carry the 1-page containment outcome and the engine stamp).
    Shared by the pipeline node and the PB2 history re-download endpoint.
    """
    resume_out = _export_resume(resume_text, request_dir)

    # If BOTH engines failed to produce a PDF, fall back to the xhtml2pdf
    # path so the resume download always works.
    if not resume_out.get("resume_pdf_path"):
        resume_out["resume_pdf_path"] = _html_to_pdf(
            resume_text, "Optimized_Resume.pdf", "Resume", request_dir)
        resume_out["render_engine"] = "xhtml2pdf@1"
        if resume_out["resume_pdf_path"] and not resume_out.get("resume_pages"):
            resume_out["resume_pages"] = _latex.pdf_page_count(
                resume_out["resume_pdf_path"])

    # --- Cover letter: HTML→PDF (unchanged) ---
    cover_path = _html_to_pdf(cover_text, "Cover_Letter.pdf", "Cover Letter", request_dir)

    return {
        "resume_pdf_path": resume_out.get("resume_pdf_path") or "",
        "cover_letter_pdf_path": cover_path,
        "resume_docx_path": resume_out.get("resume_docx_path") or "",
        "resume_tex_path": resume_out.get("resume_tex_path") or "",
        "resume_pages": resume_out.get("resume_pages") or 0,
        "resume_compacted": resume_out.get("resume_compacted") or False,
        "page_overflow_warning": resume_out.get("page_overflow_warning") or "",
        "render_engine": resume_out.get("render_engine") or "",
    }


def pdf_exporter_node(state: AgentState):
    _safe_print(f"\n--- PDF EXPORT ---")

    request_dir = Path(state.get("output_dir") or str(OUTPUT_ROOT / uuid4()))
    exported = export_documents(
        state['optimized_resume'], state['cover_letter'], request_dir)

    _safe_print(f"-> Saved resume PDF:  {exported['resume_pdf_path']}")
    _safe_print(f"-> Saved resume DOCX: {exported['resume_docx_path']}")
    _safe_print(f"-> Saved resume TEX:  {exported['resume_tex_path']}")
    _safe_print(f"-> Saved cover:       {exported['cover_letter_pdf_path']}")

    return exported
