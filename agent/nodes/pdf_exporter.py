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

Revised Nov 15 2026 (the user's call after the first Chromium-rendered
download read as a regression): under the default ``auto``, the CLASSIC
look prints through the bundled LaTeX template again - the historical
behavior; the Chromium theme prints only for the serif/modern looks
(or always under ``RESUME_RENDERER=html``). Engine stamps unchanged.

BB1 Phase B (PB2): the export core is factored into ``export_documents`` so
the history panel can REGENERATE artifacts from a past run's stored text
(the pipeline node and the server endpoint share one implementation).
"""
from pathlib import Path
from uuid import uuid4
import re

from ..ats import compute_ats_match
from ..config import OUTPUT_ROOT
from ..state import AgentState
from ..helpers import _safe_print, _html_to_pdf
from ..pdf_verify import normalize_text, verify_pdf
from ..trim import suggest_trims
from .. import latex_render as _latex
from .. import html_render as _html
from .. import verify_layout as _layout


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
        "resume_html_path": "",
        "resume_pages": resume_pages or 0,
        "resume_compacted": resume_compacted,
        "page_overflow_warning": page_overflow_warning,
        "render_engine": "latex@1",
    }


def _export_resume_html(resume_text: str, request_dir: Path,
                        theme: str = "") -> dict | None:
    """The BB2 HTML/CSS path (the Dossier steal).

    Renders the resume through the CSS-density theme printed by headless
    Chromium; the .tex source + DOCX still ride the LaTeX tooling (no
    pdflatex needed) so the artifact contract stays intact. ``theme``
    selects the look (html_render.THEMES); unknown values fall back to the
    default here - the runner already validated against the env knob, this
    is just defense in depth. Returns None when the browser is unavailable
    or the print fails - the caller falls back to the LaTeX path, so an
    export never fails because of the theme.
    """
    if theme not in _html.THEMES:
        theme = _html.DEFAULT_THEME
    try:
        data = _latex.parse_markdown_resume(resume_text)
        page = _html.render_resume_html(data, theme=theme)
        resume_pdf_path = _html.html_to_pdf(
            page, request_dir, "Optimized_Resume.pdf")
        if not resume_pdf_path:
            return None

        resume_pages = _latex.pdf_page_count(resume_pdf_path)
        resume_compacted = False
        page_overflow_warning = ""
        density_used = 0
        adopted_html = page
        # Same enforced 1-page containment as the LaTeX path, measured with
        # the SAME pdfplumber convention: on overflow the density-tier
        # ladder probes tighter CSS and adopts the first tier that fits one
        # page with the geometry invariants held.
        if resume_pages and resume_pages > 1:
            _safe_print(
                f"   [HTML] Rendered {resume_pages} pages - "
                "trying density tiers (airy-tight first).")
            tiered = _html.render_density_tiers(
                data, request_dir, "Optimized_Resume.pdf", theme=theme)
            if tiered:
                resume_pdf_path = tiered["pdf_path"]
                resume_compacted = True
                resume_pages = 1
                density_used = tiered["density"]
                adopted_html = tiered["html"]
                _safe_print(
                    f"   [HTML] Adopted density tier {tiered['density']} "
                    "(tighter typography).")
            else:
                page_overflow_warning = (
                    f"Optimized resume is {resume_pages} pages after compaction attempt. "
                    "Internal target is 1 page - trim content in Step 2 review or hide low-value sections.")
                _safe_print(f"   [WARN] {page_overflow_warning}")

        # Round 14 preview surface: persist the ADOPTED page html beside the
        # PDF. The preview IS the printed document (the Dossier's render
        # architecture): it shows the exact fonts/colors/density the PDF
        # carries, served later as the "resume-html" artifact kind.
        resume_html_path = ""
        try:
            preview_file = request_dir / "Optimized_Resume.html"
            preview_file.write_text(adopted_html, encoding="utf-8")
            resume_html_path = str(preview_file)
        except Exception as e:
            _safe_print(f"   [HTML] Preview artifact skipped: {e}")

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

        _safe_print(f"   [Renderer] HTML theme '{theme}' (Chromium), "
                    f"density tier {density_used} (0 = base).")
        return {
            "resume_pdf_path": resume_pdf_path,
            "resume_docx_path": resume_docx_path or "",
            "resume_tex_path": resume_tex_path,
            "resume_html_path": resume_html_path,
            "resume_pages": resume_pages or 0,
            "resume_compacted": resume_compacted,
            "page_overflow_warning": page_overflow_warning,
            "render_engine": _html.HTML_RENDERER_VERSION,
        }
    except Exception as e:
        _safe_print(f"   [HTML Render] Error: {e}. Falling back to LaTeX.")
        return None


def _export_resume(resume_text: str, request_dir: Path,
                   theme: str = "") -> dict:
    """Engine selection (Nov 15 2026 revision - LaTeX is the default print
    again): ``auto`` (the default) is LOOK-driven - the classic look IS
    the bundled LaTeX template, so it prints through LaTeX exactly as
    before; a serif/modern pick needs the CSS engine and routes to the
    Chromium theme when a browser is found (LaTeX classic otherwise).
    ``html`` forces the theme for every look (LaTeX fallback on print
    failure); ``latex`` pins the historical behavior for every look.
    ``theme`` selects the look (classic | serif | modern; the LaTeX
    engine prints the classic template regardless)."""
    from .. import config
    mode = config.RESUME_RENDERER
    if mode == "latex":
        return _export_resume_latex(resume_text, request_dir)
    # html always tries the theme; auto only for a non-classic look (the
    # classic look IS the LaTeX template - no browser roundtrip for it).
    if mode == "html" or (mode == "auto" and theme not in ("", "classic")):
        if mode == "html" or _html.HAVE_BROWSER:
            result = _export_resume_html(resume_text, request_dir,
                                         theme=theme)
            if result is not None:
                return result
    return _export_resume_latex(resume_text, request_dir)


def _resume_required_literals(resume_text: str, job_text: str = "") -> list[str]:
    """Name, email, and up to three literal ATS-credited keywords.

    The keyword sample is filtered to terms the DRAFT carries literally:
    a synonym/stem the ATS credited ("kubernetes" for "k8s") is the right
    semantic match but the wrong text-layer assertion - requiring it would
    raise a false alarm on a perfectly extracted PDF.
    """
    literals: list[str] = []
    try:
        data = _latex.parse_markdown_resume(resume_text)
        name = (data.contact.name or "").strip()
        if name:
            literals.append(name)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+",
                          data.contact.details or resume_text or "")
        if email:
            literals.append(email.group(0))
    except Exception:
        pass
    if job_text:
        try:
            match = compute_ats_match(job_text, resume_text)
            haystack = normalize_text(resume_text)
            for row in match.get("keywords", []):
                keyword = str(row.get("keyword") or "")
                if (row.get("found")
                        and row.get("method") in {"exact", "stem", "synonym"}
                        and len(keyword) >= 3
                        and normalize_text(keyword) in haystack):
                    literals.append(keyword)
                if len(literals) >= 5:
                    break
        except Exception:
            pass
    return literals


def _verify_exported_pdfs(resume_pdf: str, cover_pdf: str,
                          literals: list[str]) -> list[dict]:
    """Read both shipped PDFs back; one flat check list for the UI.

    Missing files are skipped: a failed render already has its own fallback
    path and warning, and a verification row for a file that was never
    written would only duplicate it. Every real failure is printed for the
    server log and returned for the payload.
    """
    checks: list[dict] = []
    for target, path, expected, required in (
            ("resume", resume_pdf, 1, literals),
            ("cover", cover_pdf, 1, ())):
        if not path or not Path(path).exists():
            continue
        result = verify_pdf(path, expected_pages=expected,
                            required_literals=required)
        failed = [c for c in result["checks"] if not c["ok"]]
        if failed:
            _safe_print(f"   [PDF verify] {target}: " + "; ".join(
                f"{c['name']}: {c['detail']}" for c in failed))
        checks.extend({"target": target, **check}
                      for check in result["checks"])
    return checks


def export_documents(resume_text: str, cover_text: str, request_dir: Path,
                     theme: str = "", job_text: str = "") -> dict:
    """Renders the optimized resume + cover letter into ``request_dir``.

    ``theme`` selects the resume look (classic | serif | modern) on the
    HTML path; the LaTeX/xhtml2pdf paths ignore it. Returns exactly the
    keys the exporter node writes back into the graph state (file paths as
    strings, "" when an export failed; the trailing keys carry the 1-page
    containment outcome and the engine stamp). Shared by the pipeline node
    and the PB2 history re-download endpoint (which passes no theme).
    """
    resume_out = _export_resume(resume_text, request_dir, theme=theme)

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

    # Layout measurement (adapted from tools/verify_layout.py in the
    # MIT ai-job-search repo, see NOTICE): advisory rows from Poppler
    # word bounding boxes (holes, orphaned entry headers, footer collisions,
    # thin final page, ...) for the Integrity tab. Advisory only - never
    # blocks, never edits. Empty when pdftotext is unavailable.
    layout_findings: list[dict] = []
    for target, path in (("resume", resume_out.get("resume_pdf_path") or ""),
                         ("cover", cover_path)):
        if not path:
            continue
        findings = _layout.measure_pdf_layout(path)
        for finding in findings:
            _safe_print(f"   [PDF layout] {target} {finding['kind']}: {finding['detail']}")
            layout_findings.append({"target": target, **finding})

    # Text-layer gate (Nov 19 2026): the PDF an ATS parses, read back. Name
    # and email are the required literals every engine must preserve; the
    # ATS keyword sample proves the keyword leg of the score is not measuring
    # text that never made it into the download.
    pdf_checks = _verify_exported_pdfs(
        resume_out.get("resume_pdf_path") or "", cover_path,
        _resume_required_literals(resume_text, job_text))

    return {
        "resume_pdf_path": resume_out.get("resume_pdf_path") or "",
        "cover_letter_pdf_path": cover_path,
        "resume_docx_path": resume_out.get("resume_docx_path") or "",
        "resume_tex_path": resume_out.get("resume_tex_path") or "",
        "resume_html_path": resume_out.get("resume_html_path") or "",
        "resume_pages": resume_out.get("resume_pages") or 0,
        "resume_compacted": resume_out.get("resume_compacted") or False,
        "page_overflow_warning": resume_out.get("page_overflow_warning") or "",
        "render_engine": resume_out.get("render_engine") or "",
        "pdf_checks": pdf_checks,
        "layout_findings": layout_findings,
    }


def pdf_exporter_node(state: AgentState):
    _safe_print(f"\n--- PDF EXPORT ---")

    request_dir = Path(state.get("output_dir") or str(OUTPUT_ROOT / uuid4()))
    exported = export_documents(
        state['optimized_resume'], state['cover_letter'], request_dir,
        theme=state.get("resume_theme") or "",
        job_text=state.get("job_text") or "")

    # One-page containment already tried every density tier: when it still
    # overflows, rank the cheapest content cuts from the same deterministic
    # signals the Fit/Bullet-impact tabs show. Empty on a normal one-page run.
    exported["trim_suggestions"] = suggest_trims(
        state.get("optimized_resume") or "",
        job_text=state.get("job_text") or "",
        cover_text=state.get("cover_letter") or "",
        requirement_matches=state.get("requirement_matches") or [],
        bullet_impact=state.get("bullet_impact") or [],
        overflowed=bool(exported.get("page_overflow_warning")),
    )

    _safe_print(f"-> Saved resume PDF:  {exported['resume_pdf_path']}")
    _safe_print(f"-> Saved resume DOCX: {exported['resume_docx_path']}")
    _safe_print(f"-> Saved resume TEX:  {exported['resume_tex_path']}")
    _safe_print(f"-> Saved cover:       {exported['cover_letter_pdf_path']}")

    return exported
