"""NODE 8: PDF Exporter - saves resume and cover letter as styled PDFs/DOCX.

The optimized resume is rendered through a bundled LaTeX template (PDF + DOCX
+ source .tex), while the cover letter keeps the simpler HTML→PDF path.
If LaTeX/pandoc are unavailable, the resume gracefully falls back to HTML→PDF
so exports still work everywhere.
"""
from pathlib import Path
from uuid import uuid4

from ..config import OUTPUT_ROOT
from ..state import AgentState
from ..helpers import _safe_print, _html_to_pdf
from .. import latex_render as _latex


def pdf_exporter_node(state: AgentState):
    _safe_print(f"\n--- PDF EXPORT ---")

    request_dir = Path(state.get("output_dir") or str(OUTPUT_ROOT / uuid4()))

    # --- Resume: LaTeX template → PDF + DOCX (+ .tex source) ---
    resume_pdf_path = None
    resume_docx_path = None
    resume_tex_path = None
    optimized_resume = state['optimized_resume']
    try:
        data = _latex.parse_markdown_resume(optimized_resume)
        tex_content = _latex.render_resume_tex(data)
        # Save the .tex source alongside the compiled outputs (useful for
        # advanced editing in Overleaf/TeX editors).
        if tex_content:
            request_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            tex_file = request_dir / "Optimized_Resume.tex"
            tex_file.write_text(tex_content, encoding="utf-8")
            resume_tex_path = str(tex_file)

        resume_pdf_path = _latex.latex_to_pdf(
            tex_content, request_dir, "Optimized_Resume.pdf")
        resume_docx_path = _latex.latex_to_docx(
            tex_content, request_dir, "Optimized_Resume.docx")
    except Exception as e:
        _safe_print(f"   [LaTeX Render] Error: {e}. Falling back to HTML→PDF for resume.")

    # If LaTeX PDF export was unavailable/failed, fall back to HTML→PDF so the
    # resume download always works.
    if not resume_pdf_path:
        resume_pdf_path = _html_to_pdf(
            optimized_resume, "Optimized_Resume.pdf", "Resume", request_dir)

    # --- Cover letter: HTML→PDF (unchanged) ---
    cover_path = _html_to_pdf(state['cover_letter'], "Cover_Letter.pdf", "Cover Letter", request_dir)

    _safe_print(f"-> Saved resume PDF:  {resume_pdf_path}")
    _safe_print(f"-> Saved resume DOCX: {resume_docx_path}")
    _safe_print(f"-> Saved resume TEX:  {resume_tex_path}")
    _safe_print(f"-> Saved cover:       {cover_path}")

    return {
        "resume_pdf_path": resume_pdf_path or "",
        "cover_letter_pdf_path": cover_path,
        "resume_docx_path": resume_docx_path or "",
        "resume_tex_path": resume_tex_path or "",
    }
