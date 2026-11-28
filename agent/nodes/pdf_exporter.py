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


class PdfGeometryError(RuntimeError):
    """Raised when no render engine can produce a PDF passing geometric bounds."""

from ..ats import compute_ats_match
from ..config import OUTPUT_ROOT
from ..state import AgentState
from ..helpers import _safe_print, _html_to_pdf
from ..pdf_verify import normalize_text, verify_pdf
from ..trim import suggest_trims
from .. import latex_render as _latex
from .. import html_render as _html
from .. import verify_layout as _layout


def _export_resume_latex(resume_text: str, request_dir: Path,
                         target_pages: int = 1) -> dict:
    """The classic LaTeX path (historical behavior, byte-for-byte)."""
    resume_pdf_path = None
    resume_docx_path = None
    resume_tex_path = None
    resume_pages = None
    resume_compacted = False
    page_overflow_warning = ""
    target_pages = max(1, min(2, int(target_pages or 1)))
    from ..helpers import strip_placeholder_text
    resume_text, _ = strip_placeholder_text(resume_text or "")
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

        # --- Enforced target-page containment (1 default, 2 max) ---
        if resume_pdf_path:
            resume_pages = _latex.pdf_page_count(resume_pdf_path)
            if resume_pages and resume_pages > target_pages:
                _safe_print(
                    f"   [LaTeX] Compiled {resume_pages} pages (target={target_pages}) - "
                    "trying auto-compact tiers (airy first).")
                tiered = (
                    _latex.render_compact_tiers(data, request_dir, "Optimized_Resume.pdf")
                    if target_pages == 1 else
                    _latex.render_compact_tiers(data, request_dir, "Optimized_Resume.pdf", target_pages=target_pages)
                )
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
                    resume_pages = tiered["pages"]
                    _safe_print(
                        f"   [LaTeX] Adopted auto-compact tier {tiered['tier']} "
                        f"({'airy' if tiered['tier'] == 1 else 'tight'} spacing).")
                else:
                    page_overflow_warning = (
                        f"Optimized resume is {resume_pages} pages after compaction attempt. "
                        f"Target is {target_pages} page(s) - trim content in Step 2 review or hide low-value sections.")
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
                        theme: str = "", target_pages: int = 1) -> dict | None:
    """The BB2 HTML/CSS path (the Dossier steal)."""
    if theme not in _html.THEMES:
        theme = _html.DEFAULT_THEME
    target_pages = max(1, min(2, int(target_pages or 1)))
    from ..helpers import strip_placeholder_text
    resume_text, _ = strip_placeholder_text(resume_text or "")
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
        if resume_pages and resume_pages > target_pages:
            _safe_print(
                f"   [HTML] Rendered {resume_pages} pages (target={target_pages}) - "
                "trying density tiers (airy-tight first).")
            tiered = (
                _html.render_density_tiers(data, request_dir, "Optimized_Resume.pdf", theme=theme)
                if target_pages == 1 else
                _html.render_density_tiers(data, request_dir, "Optimized_Resume.pdf", theme=theme, target_pages=target_pages)
            )
            if tiered:
                resume_pdf_path = tiered["pdf_path"]
                resume_compacted = True
                resume_pages = tiered["pages"]
                density_used = tiered["density"]
                adopted_html = tiered["html"]
                _safe_print(
                    f"   [HTML] Adopted density tier {tiered['density']} "
                    "(tighter typography).")
            else:
                page_overflow_warning = (
                    f"Optimized resume is {resume_pages} pages after compaction attempt. "
                    f"Target is {target_pages} page(s) - trim content in Step 2 review or hide low-value sections.")
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
                   theme: str = "", target_pages: int = 1) -> dict:
    """Engine selection (Nov 15 2026 revision - LaTeX is the default print
    again)."""
    from .. import config
    mode = config.RESUME_RENDERER
    if mode == "latex":
        return (_export_resume_latex(resume_text, request_dir)
                if target_pages == 1 else
                _export_resume_latex(resume_text, request_dir, target_pages=target_pages))
    if mode == "html" or (mode == "auto" and theme not in ("", "classic")):
        if mode == "html" or _html.HAVE_BROWSER:
            result = (_export_resume_html(resume_text, request_dir, theme=theme)
                      if target_pages == 1 else
                      _export_resume_html(resume_text, request_dir, theme=theme, target_pages=target_pages))
            if result is not None:
                return result
    return (_export_resume_latex(resume_text, request_dir)
            if target_pages == 1 else
            _export_resume_latex(resume_text, request_dir, target_pages=target_pages))


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
                          literals: list[str],
                          expected_resume_pages: int = 1) -> list[dict]:
    """Read both shipped PDFs back; one flat check list for the UI.

    Missing files are skipped: a failed render already has its own fallback
    path and warning, and a verification row for a file that was never
    written would only duplicate it. Every real failure is printed for the
    server log and returned for the payload.
    """
    checks: list[dict] = []
    for target, path, expected, required in (
            ("resume", resume_pdf, expected_resume_pages, literals),
            ("cover", cover_pdf, 1, ())):
        if not path or not Path(path).exists():
            continue
        result = verify_pdf(path, expected_pages=expected,
                            required_literals=required)
        failed = [c for c in result["checks"] if not c["ok"]]
        if failed:
            from ..tracing import _redact_pii
            _safe_print(f"   [PDF verify] {target}: " + "; ".join(
                f"{c['name']}: {_redact_pii(c['detail'])}" for c in failed))  # Item 10
        checks.extend({"target": target, **check}
                      for check in result["checks"])
    return checks


def export_documents(resume_text: str, cover_text: str, request_dir: Path,
                     theme: str = "", job_text: str = "",
                     target_pages: int = 1, input_pages: int = 0,
                     has_human_notes: bool = False) -> dict:
    """Renders the optimized resume + cover letter into ``request_dir``.

    ``theme`` selects the resume look (classic | serif | modern) on the
    HTML path; the LaTeX/xhtml2pdf paths ignore it. ``target_pages`` enforces
    1- or 2-page containment (with +1 page allowance up to hard cap 2 when
    ``has_human_notes`` added content).
    """
    target_pages = max(1, min(2, int(target_pages or 1)))
    from ..helpers import strip_placeholder_text
    resume_text, _ = strip_placeholder_text(resume_text or "")
    resume_out = (
        _export_resume(resume_text, request_dir, theme=theme)
        if target_pages == 1 else
        _export_resume(resume_text, request_dir, theme=theme, target_pages=target_pages)
    )

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

    layout_findings: list[dict] = []
    for target, path in (("resume", resume_out.get("resume_pdf_path") or ""),
                         ("cover", cover_path)):
        if not path:
            continue
        findings = _layout.measure_pdf_layout(path)
        from ..tracing import _redact_pii
        for finding in findings:
            _safe_print(f"   [PDF layout] {target} {finding['kind']}: {_redact_pii(finding['detail'])}")  # Item 10
            layout_findings.append({"target": target, **finding})

    shipped_pages = int(resume_out.get("resume_pages") or 0)
    effective_expected = target_pages
    expanded_from_notes = False
    if shipped_pages > target_pages:
        if target_pages == 1 and has_human_notes and shipped_pages == 2:
            effective_expected = 2
            expanded_from_notes = True
            resume_out["page_overflow_warning"] = "Added content moved this to 2 pages."
        elif shipped_pages > 2:
            resume_out["page_overflow_warning"] = (
                f"Optimized resume is {shipped_pages} pages (exceeds hard 2-page cap). "
                "Trim content before exporting."
            )

    # Text-layer + geometry gate (Nov 19 / Nov 26 2026): verify the shipped PDF
    # back. On resume geometry failure, retry compact LaTeX (if uncompacted),
    # then fall back to Chromium HTML; block export if neither passes.
    literals = _resume_required_literals(resume_text, job_text)
    gate_checks: list[dict] = []
    resume_pdf = resume_out.get("resume_pdf_path") or ""

    def _run_verify(pdf_p: str, exp_p: int) -> list[dict]:
        return (
            _verify_exported_pdfs(pdf_p, cover_path, literals)
            if exp_p == 1 else
            _verify_exported_pdfs(pdf_p, cover_path, literals, expected_resume_pages=exp_p)
        )

    pdf_checks = _run_verify(resume_pdf, effective_expected)

    def _geom_failure(checks: list[dict]) -> dict | None:
        return next(
            (c for c in checks
             if c.get("target") == "resume"
             and c.get("name") == "geometry"
             and not c.get("ok")),
            None,
        )

    geom_fail = _geom_failure(pdf_checks)
    if geom_fail and resume_pdf:
        from .. import config
        engine = resume_out.get("render_engine") or ""

        # Step 1: retry compact LaTeX if not yet compacted (margins expand slightly)
        if engine == "latex@1" and not resume_out.get("resume_compacted"):
            gate_checks.append({
                "target": "resume",
                "name": "render_retry",
                "ok": True,
                "detail": f"LaTeX geometry failed ({geom_fail['detail']}); retrying compact tiers",
            })
            try:
                data = _latex.parse_markdown_resume(resume_text)
                tiered = (
                    _latex.render_compact_tiers(data, request_dir, "Optimized_Resume.pdf")
                    if target_pages == 1 else
                    _latex.render_compact_tiers(data, request_dir, "Optimized_Resume.pdf", target_pages=target_pages)
                )
                if tiered:
                    tex_file = request_dir / "Optimized_Resume.tex"
                    tex_file.write_text(tiered["tex"], encoding="utf-8")
                    compact_docx = _latex.latex_to_docx(
                        tiered["tex"], request_dir, "Optimized_Resume.docx")
                    if compact_docx:
                        resume_out["resume_docx_path"] = compact_docx
                    resume_out["resume_pdf_path"] = tiered["pdf_path"]
                    resume_out["resume_compacted"] = True
                    resume_out["resume_pages"] = tiered["pages"]
                    resume_pdf = tiered["pdf_path"]
                    pdf_checks = _run_verify(resume_pdf, effective_expected)
                    geom_fail = _geom_failure(pdf_checks)
            except Exception as exc:
                gate_checks.append({
                    "target": "resume",
                    "name": "render_retry",
                    "ok": False,
                    "detail": f"LaTeX compact retry failed: {exc}",
                })

        # Step 2: if geometry still fails on LaTeX, fall back to HTML classic/theme
        if (geom_fail and config.RESUME_RENDERER != "latex"
                and (resume_out.get("render_engine") or "") == "latex@1"
                and _html.HAVE_BROWSER):
            fb_theme = theme or _html.DEFAULT_THEME
            html_out = (
                _export_resume_html(resume_text, request_dir, theme=fb_theme)
                if target_pages == 1 else
                _export_resume_html(resume_text, request_dir, theme=fb_theme, target_pages=target_pages)
            )
            if html_out and html_out.get("resume_pdf_path"):
                gate_checks.append({
                    "target": "resume",
                    "name": "engine_fallback",
                    "ok": True,
                    "detail": f"LaTeX failed geometry ({geom_fail['detail']}); fell back to HTML ({fb_theme})",
                })
                resume_out = html_out
                resume_pdf = resume_out["resume_pdf_path"]
                shipped_pages = int(resume_out.get("resume_pages") or 0)
                pdf_checks = _run_verify(resume_pdf, effective_expected)
                geom_fail = _geom_failure(pdf_checks)
            else:
                gate_checks.append({
                    "target": "resume",
                    "name": "engine_fallback",
                    "ok": False,
                    "detail": f"LaTeX failed geometry ({geom_fail['detail']}) and HTML fallback produced no PDF",
                })

        # Step 3: hard block if neither engine passed geometric bounds
        if geom_fail:
            Path(resume_pdf).unlink(missing_ok=True)
            resume_out["resume_pdf_path"] = ""
            raise PdfGeometryError(
                f"Resume PDF failed geometric layout bounds and could not be exported cleanly: {geom_fail['detail']}"
            )

    if target_pages != 1 or expanded_from_notes or shipped_pages > target_pages:
        target_ok = (shipped_pages == 0) or (shipped_pages <= 2 and shipped_pages <= effective_expected)
        detail = f"input={input_pages or 'text/unknown'}p, target={target_pages}p, output={shipped_pages}p"
        if expanded_from_notes:
            detail += " — Added content moved this to 2 pages."
        gate_checks.append({
            "target": "resume",
            "name": "page_target",
            "ok": target_ok,
            "detail": detail,
        })

    pdf_checks = [*gate_checks, *pdf_checks]

    return {
        "resume_pdf_path": resume_out.get("resume_pdf_path") or "",
        "cover_letter_pdf_path": cover_path,
        "resume_docx_path": resume_out.get("resume_docx_path") or "",
        "resume_tex_path": resume_out.get("resume_tex_path") or "",
        "resume_html_path": resume_out.get("resume_html_path") or "",
        "resume_pages": resume_out.get("resume_pages") or 0,
        "target_pages": effective_expected,
        "input_pages": input_pages,
        "page_target_expanded": expanded_from_notes,
        "resume_compacted": resume_out.get("resume_compacted") or False,
        "page_overflow_warning": resume_out.get("page_overflow_warning") or "",
        "render_engine": resume_out.get("render_engine") or "",
        "pdf_checks": pdf_checks,
        "layout_findings": layout_findings,
    }


def pdf_exporter_node(state: AgentState):
    import inspect
    _safe_print(f"\n--- PDF EXPORT ---")

    request_dir = Path(state.get("output_dir") or str(OUTPUT_ROOT / uuid4()))
    target_pages = int(state.get("target_pages") or 1)
    input_pages = int(state.get("input_pages") or 0)
    has_notes = bool((state.get("human_notes") or "").strip())
    kwargs = {}
    if "target_pages" in inspect.signature(export_documents).parameters:
        kwargs = {
            "target_pages": target_pages,
            "input_pages": input_pages,
            "has_human_notes": has_notes,
        }
    exported = export_documents(
        state['optimized_resume'], state['cover_letter'], request_dir,
        theme=state.get("resume_theme") or "",
        job_text=state.get("job_text") or "",
        **kwargs)

    still_overflows = bool(exported.get("page_overflow_warning")) and not exported.get("page_target_expanded")
    exported["trim_suggestions"] = suggest_trims(
        state.get("optimized_resume") or "",
        job_text=state.get("job_text") or "",
        cover_text=state.get("cover_letter") or "",
        requirement_matches=state.get("requirement_matches") or [],
        bullet_impact=state.get("bullet_impact") or [],
        overflowed=still_overflows,
    )
    decision_checks = []
    if state.get("polish_decision"):
        decision_checks.append({
            "target": "resume",
            "name": "polish_gate",
            "ok": True,
            "detail": str(state["polish_decision"]),
        })
    if state.get("best_of_decision"):
        decision_checks.append({
            "target": "resume",
            "name": "best_of_guard",
            "ok": True,
            "detail": str(state["best_of_decision"]),
        })
    for leak in (state.get("prompt_prevention_failures") or []):
        if isinstance(leak, dict) and leak.get("terms"):
            stage_name = str(leak.get("stage") or "pipeline")
            decision_checks.append({
                "target": "cover" if stage_name == "cover_letter" else "resume",
                "name": "prompt_prevention_failure",
                "ok": True,
                "detail": f"{stage_name}: stripped leaked term(s) {', '.join(leak['terms'])}",
            })
    if decision_checks:
        exported["pdf_checks"] = [*(exported.get("pdf_checks") or []), *decision_checks]

    _safe_print(f"-> Saved resume PDF:  {exported['resume_pdf_path']}")
    _safe_print(f"-> Saved resume DOCX: {exported['resume_docx_path']}")
    _safe_print(f"-> Saved resume TEX:  {exported['resume_tex_path']}")
    _safe_print(f"-> Saved cover:       {exported['cover_letter_pdf_path']}")

    return exported
