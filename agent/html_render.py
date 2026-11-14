"""BB2 (Dossier steal): HTML/CSS resume renderer printed by headless Chromium.

The Dossier study's render architecture (``src/lib/pdf/render.ts``): the PDF
is printed from the app's OWN HTML page, so the preview IS the PDF, printed
by pooled headless Chromium with a network-isolated context. This module is
the Python mapping, replacing the LaTeX ``_COMPACT_OVERRIDES`` spacing
surgery (the whole 7pt-overprint / boundary / hanging-indent defect lineage)
with a CSS density knob and a MEASURED 1-page loop:

- :func:`render_resume_html` - the SAME battle-tested markdown parser as
  LaTeX (``parse_markdown_resume``) feeds one self-contained HTML page
  (inline CSS only). Network isolation is enforced three ways: the markup
  carries no external fetchable URL by construction (system font stacks, no
  ``<img>``/``<link>``/``@import``/remote fonts), a strict
  ``Content-Security-Policy`` meta (``default-src 'none'``) makes any
  injected or leaked fetch fail closed, and the renderer prints a local
  ``file://`` page. Fonts come from the OS - nothing is downloaded, nothing
  leaves the server.
- :func:`html_to_pdf` - one headless Chromium/Edge subprocess per render
  (``--headless --print-to-pdf``), honoring the CSS ``@page`` size/margins.
  The Dossier's pooled browser (max 2 concurrent renders, queue beyond,
  5-min idle close, relaunch on crash) maps onto a bounded-concurrency
  semaphore: a one-shot process has no idle state to close and cannot crash
  a shared browser, so the pool's guarantees (never more than 2 renders at
  once, queued beyond) hold WITHOUT the ~250 MB Playwright browser download
  - the system's own Chrome or Edge (Windows ships Edge) is reused.
- :func:`render_density_tiers` - the measured loop, mirroring the LaTeX
  tier contract (:func:`latex_render.render_compact_tiers`): probe renders
  to ``__density_tier{n}.pdf``, adopt the first tier that fits ONE page
  with the geometry invariants held (tier 1 additionally gated by the
  bottom-margin stop), winners copied onto the shipped filename, losers
  unlinked. A one-page render that crosses the stop is kept as a
  last-resort fallback; when no tier fits one page the caller keeps the
  base render and surfaces the overflow warning.

Env knobs (agent/config.py): ``RESUME_RENDERER`` = auto (default) | html |
latex; ``RENDER_BROWSER_PATH`` forces a browser binary. The LaTeX path
stays the classic/conservative theme; Unicode is preserved verbatim (the
LaTeX renderer must ASCII-fold it - HTML does not, a rendering-quality win).

Note: unit placement is "ATS-safe by construction" - real text (no tables,
no columns, no images) survives extraction, pinned by the acceptance tests
against the SAME fixture family the LaTeX path is validated with.
"""
from __future__ import annotations

import html as _html_mod
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .latex_render import (
    Entry,
    ResumeData,
    _BOLD_LABEL_RE,
    _DATE_RANGE_RE,
    _display_url_text,
    _extract_project_links_and_title,
    _strip_inline,
    pdf_last_baseline,
    pdf_page_count,
)

logger = logging.getLogger(__name__)

HTML_RENDERER_VERSION = "html-chromium@1"

# ---------------------------------------------------------------------------
# Browser discovery (the print engine)
# ---------------------------------------------------------------------------


def _find_browser() -> Optional[str]:
    """Locates a Chromium-family binary: explicit env override first, then
    PATH, then the standard Windows/macOS/Linux install locations, then a
    Playwright browser cache. Returns the absolute path or None."""
    override = os.environ.get("RENDER_BROWSER_PATH", "").strip()
    if override and Path(override).exists():
        return override
    for name in ("chrome", "msedge", "chromium", "google-chrome", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    program_files = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application"),
    ]
    candidates: list[Path] = []
    for root in program_files:
        candidates += [root / "Google" / "Chrome" / "Application" / "chrome.exe",
                       root / "Microsoft" / "Edge" / "Application" / "msedge.exe"]
    candidates += [Path(p) for p in (
        "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable", "/usr/bin/microsoft-edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    )]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    # Playwright's cached Chromium (if the user ever installed the package)
    pw_root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    if pw_root.is_dir():
        for probe in sorted(pw_root.glob("chromium*/chrome-win*/chrome.exe")):
            if probe.is_file():
                return str(probe)
    return None


BROWSER_PATH = _find_browser()
HAVE_BROWSER = BROWSER_PATH is not None
if not HAVE_BROWSER:
    logger.info("HTML renderer: no Chromium-family browser found - the HTML "
                "theme is unavailable (RESUME_RENDERER falls back to LaTeX).")

# The Dossier pool's MAX_CONCURRENT_RENDERS: a bounded semaphore the print
# call waits on, so parallel runs queue instead of stampeding the browser.
_RENDER_SLOTS = threading.BoundedSemaphore(2)


# ---------------------------------------------------------------------------
# Inline text: markdown links, bare URLs, escaping
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BARE_URL_RE = re.compile(r"https?://\S+")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_CODE_RE = re.compile(r"`(.+?)`")


def _clean_href(raw_url: str) -> Optional[str]:
    """Sanitizes a link target for an href attribute. Only http(s)/mailto
    schemes pass (the same scheme allowlist the LaTeX linkifier uses);
    anything else - javascript:, data:, or schemeless oddities - returns
    None and the caller renders plain text instead."""
    url = raw_url.strip().lstrip("<").rstrip(">")
    url = url.split()[0].strip().rstrip(")").rstrip(".,;")
    if not url:
        return None
    if not url.startswith(("http://", "https://", "mailto:")):
        url = ("mailto:" if "@" in url else "https://") + url
    if not url.startswith(("http://", "https://", "mailto:")):
        return None
    return url


def _esc(text: str) -> str:
    return _html_mod.escape(text or "", quote=True)


def _inline(text: str) -> str:
    """Inline text for HTML: markdown links and bare URLs become <a> (the
    displayed text keeps the LaTeX path's scheme-stripping convention - the
    FULL URL stays in the href), bold/italic/code markers are stripped,
    everything is HTML-escaped. Untrusted resume content can never inject
    markup."""
    if not text:
        return ""
    links: list[tuple[str, Optional[str]]] = []

    def _save_link(match: "re.Match[str]") -> str:
        label = match.group(1).strip()
        idx = len(links)
        links.append((label, _clean_href(match.group(2))))
        return f"\x00{idx}\x00"

    work = _MD_LINK_RE.sub(_save_link, text)
    work = _MD_BOLD_RE.sub(r"\1", work)
    work = _MD_ITALIC_RE.sub(r"\1", work)
    work = _MD_CODE_RE.sub(r"\1", work)

    parts: list[str] = []
    token_re = re.compile(r"(\x00\d+\x00)")
    for piece in token_re.split(work):
        if not piece:
            continue
        m_link = re.fullmatch(r"\x00(\d+)\x00", piece)
        if m_link:
            label, url = links[int(m_link.group(1))]
            display = _esc(_display_url_text(label))
            if url:
                parts.append(f'<a href="{_esc(url)}">{display}</a>')
            else:  # unparsable target: render the label as plain text
                parts.append(display)
            continue
        pos = 0
        for m_url in _BARE_URL_RE.finditer(piece):
            parts.append(_esc(piece[pos:m_url.start()]))
            raw = m_url.group(0)
            url = _clean_href(raw)
            display = _esc(_display_url_text(raw))
            if url:
                parts.append(f'<a href="{_esc(url)}">{display}</a>')
            else:
                parts.append(display)
            pos = m_url.end()
        parts.append(_esc(piece[pos:]))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Markdown -> structured data (reuses the LaTeX parser verbatim)
# ---------------------------------------------------------------------------


def _clean_field(s: str) -> str:
    return re.sub(r"\s*[\|—–\-:]\s*$", "", (s or "").strip()).strip()


# Experience-role vocabulary (ported from latex_render._render_section):
# decides whether a sub line is the ROLE and the header line the COMPANY.
_ROLE_KW = (
    "engineer", "developer", "trainee", "intern", "specialist", "manager",
    "lead", "analyst", "scientist", "consultant", "architect", "officer",
    "assistant", "director", "associate", "fellow", "head", "coordinator",
    "researcher", "instructor", "designer", "administrator", "technician",
)


def _entry_head_rows(entry: Entry, title_lower: str) -> list[tuple[str, str]]:
    """Classifies an entry's header fields into rendered (left, right) rows.

    Ports the LaTeX renderer's per-kind semantic corrections - the
    project date/tech-stack disambiguation with the pipe-glue defense (a
    date column carrying '| more fields' once printed the whole string
    right-aligned and ran off the sheet), the experience role/company swap
    (the role keyword decides which line is which), and the education
    two-row mapping - into layout-neutral tuples. The HTML side renders
    them as flex rows with a right-aligned date cell, so the width
    estimation gymnastics LaTeX needs (the <=85-chars branch) are not
    required here: CSS wraps the left cell naturally.
    """
    is_project = "project" in title_lower
    is_exp = any(k in title_lower for k in
                 ("experience", "employment", "work", "history", "career"))
    is_edu = any(k in title_lower for k in
                 ("education", "academic", "university", "college", "school"))

    rows: list[tuple[str, str]] = []
    if is_project:
        if _DATE_RANGE_RE.search(entry.right) and not entry.sub_right:
            date_field = entry.right
            tech_raw = entry.sub
        else:
            date_field = entry.sub_right or (
                _clean_field(entry.right) if _DATE_RANGE_RE.search(entry.right)
                else "")
            tech_raw = (entry.right
                        if not _DATE_RANGE_RE.search(entry.right)
                        else entry.sub)
        m_glue = re.match(r"^(.*\d{4})\s*\|\s*(\S.*)$", date_field or "")
        if m_glue:
            date_field = m_glue.group(1).strip()
            if not tech_raw:
                tech_raw = m_glue.group(2).strip()
        other_fields = [f for f in (entry.right, entry.sub)
                        if f and f != date_field]
        clean_title, _links = _extract_project_links_and_title(
            raw_title=entry.header, other_fields=other_fields,
            bullet_links=entry.links)
        md_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        clean_tech = md_re.sub("", tech_raw or "")
        clean_tech = re.sub(r"https?://\S+", "", clean_tech)
        clean_tech = re.sub(r"^\s*[\|—–\-:]\s*", "", clean_tech).strip()
        clean_tech = re.sub(r"\s*[\|—–\-:]\s*$", "", clean_tech).strip()
        left = _inline(clean_title)
        if _links:
            link_html = "".join(
                ' <span class="psep">|</span> <a href="{}">{}</a>'.format(
                    _esc(u), _esc(_display_url_text(lbl)))
                for lbl, u in _links)
            left += link_html
        if clean_tech:
            left += (f' <span class="psep">|</span> '
                     f"<em>{_inline(clean_tech)}</em>")
        rows.append((left, _esc(date_field)))
    elif is_exp:
        role = entry.header
        company = entry.sub or entry.right
        location = entry.right if entry.sub else ""
        date = entry.sub_right
        if (entry.sub
                and any(k in entry.sub.lower() for k in _ROLE_KW)
                and not any(k in entry.header.lower() for k in _ROLE_KW)):
            role, company, location = entry.sub, entry.header, entry.right
        left = f"<b>{_inline(role)}</b>"
        if company:
            left += (f' <span class="psep">|</span> '
                     f"<em>{_inline(company)}</em>")
            if location:
                left += f' <span class="loc">({_esc(location)})</span>'
        rows.append((left, _esc(date)))
    elif is_edu:
        # Education: institution & location on row 1, degree & dates on row 2
        rows.append((f"<b>{_inline(entry.header)}</b>",
                     _inline(entry.right)))
        rows.append((_inline(entry.sub), _inline(entry.sub_right)))
    else:
        sub_l = _inline(entry.sub)
        sub_r = _inline(entry.sub_right)
        if not sub_l and not sub_r:
            rows.append((f"<b>{_inline(entry.header)}</b>",
                         _esc(entry.right)))
        else:
            rows.append((f"<b>{_inline(entry.header)}</b>",
                         _esc(entry.right)))
            rows.append((sub_l, sub_r))
    return rows


def _simple_rows(lines: list[str]) -> list[tuple[bool, str]]:
    """Renders a section's simple_lines as label-value rows (the layout
    spec A/B semantics: bold labels, one row per line, hanging indent for
    wrapped tails - CSS does in one declaration what LaTeX needed the
    parshape surgery for). Returns [(is_label_row, html), ...]."""
    rows: list[tuple[bool, str]] = []
    for ln in lines:
        m = _BOLD_LABEL_RE.match(ln)
        if not m:
            cm = re.match(r"^(.*?):\s*(.*)$", ln)
            if cm and len(cm.group(1)) <= 35:
                m = cm
            else:
                rows.append((False, _inline(ln)))
                continue
        label = _strip_inline(m.group(1))
        value = m.group(2)
        rows.append((True, f'<span class="lbl">{_esc(label)}:</span> '
                     f"{_inline(value)}"))
    return rows


# ---------------------------------------------------------------------------
# The page: self-contained HTML with print CSS + strict CSP
# ---------------------------------------------------------------------------

_CSP_META = ('<meta http-equiv="Content-Security-Policy" '
             "content=\"default-src 'none'; style-src 'unsafe-inline'\">")

# Density tiers (0 = base). Tier 1 is the "airy-tight" analog of the LaTeX
# tier ladder's tier-1: adopted only when the render stays one page AND the
# last baseline stays at or above _HTML_BOTTOM_STOP; tiers 2-3 adopt
# whenever they reclaim one page (one dense page beats two airy ones).
_DENSITY_VARS: dict[int, dict[str, str]] = {
    0: {"fs": "10pt", "lh": "1.35", "secgap": "10pt", "itemgap": "2pt",
        "entrygap": "5pt", "rowsfs": "9.2pt", "namefs": "20pt",
        "contactfs": "9.5pt", "h2fs": "10.5pt"},
    1: {"fs": "9.75pt", "lh": "1.3", "secgap": "8pt", "itemgap": "1.5pt",
        "entrygap": "4pt", "rowsfs": "9pt", "namefs": "19pt",
        "contactfs": "9.25pt", "h2fs": "10.25pt"},
    2: {"fs": "9.5pt", "lh": "1.25", "secgap": "6.5pt", "itemgap": "1pt",
        "entrygap": "3pt", "rowsfs": "8.75pt", "namefs": "18pt",
        "contactfs": "9pt", "h2fs": "10pt"},
    3: {"fs": "9pt", "lh": "1.18", "secgap": "5pt", "itemgap": "0.5pt",
        "entrygap": "2pt", "rowsfs": "8.25pt", "namefs": "17pt",
        "contactfs": "8.75pt", "h2fs": "9.5pt"},
}


def _density_css(density: int) -> str:
    v = _DENSITY_VARS[density]
    return (f"  :root{{--fs:{v['fs']};--lh:{v['lh']};--secgap:{v['secgap']};"
            f"--itemgap:{v['itemgap']};--entrygap:{v['entrygap']};"
            f"--rowsfs:{v['rowsfs']};--namefs:{v['namefs']};"
            f"--contactfs:{v['contactfs']};--h2fs:{v['h2fs']};}}")


_PAGE_CSS = """
  @page { size: letter; margin: 36pt 40pt 36pt 40pt; }
  * { box-sizing: border-box; }
  body {
    margin: 0; color: #111;
    font-family: "Segoe UI", "Helvetica Neue", Arial, "DejaVu Sans", sans-serif;
    font-size: var(--fs); line-height: var(--lh);
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  header { text-align: center; margin: 0 0 2pt; }
  h1 { font-size: var(--namefs); margin: 0; letter-spacing: 0.3pt; }
  .contact { font-size: var(--contactfs); }
  .contact a { color: inherit; }
  section { margin-top: var(--secgap); }
  h2 {
    font-size: var(--h2fs); margin: 0 0 2.5pt;
    text-transform: uppercase; letter-spacing: 0.6pt;
    border-bottom: 0.8pt solid #444; padding-bottom: 1.5pt;
  }
  .entry { margin-top: var(--entrygap); }
  .entry:first-of-type { margin-top: 0; }
  .eh { display: flex; justify-content: space-between; align-items: baseline;
        gap: 12pt; }
  .eh .left { min-width: 0; }
  .eh .right { white-space: nowrap; flex-shrink: 0; color: #333;
               font-size: calc(var(--fs) - 0.5pt); }
  .sub-row { display: flex; justify-content: space-between;
             align-items: baseline; gap: 12pt; color: #222; }
  .sub-row .right { white-space: nowrap; flex-shrink: 0; }
  .psep, .loc { color: #444; }
  ul.bullets { margin: 2pt 0 0 0; padding-left: 14pt; }
  ul.bullets li { margin-bottom: var(--itemgap); }
  ul.bullets li:last-child { margin-bottom: 0; }
  .rows { font-size: var(--rowsfs); }
  /* Hanging indent for wrapped row tails: first line flush, every
     continuation +12pt (the layout-spec fix, CSS-native here). */
  .row { padding-left: 12pt; text-indent: -12pt; margin-bottom: 1pt; }
  .row:last-child { margin-bottom: 0; }
  .row .lbl { font-weight: 600; }
  a { color: inherit; text-decoration: none; }
"""


def render_resume_html(data: ResumeData, density: int = 0) -> str:
    """Renders the structured resume into ONE self-contained HTML page.

    ``density`` selects the tier's CSS variables (0 base, 1-3 progressively
    tighter - the same airy-to-floor ladder the LaTeX compact tiers walk,
    expressed as typography instead of spacing surgery). Content is never
    altered; Unicode is preserved verbatim.
    """
    if density not in _DENSITY_VARS:
        density = 0
    out: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        _CSP_META,
        "<title>Resume</title>",
        "<style>",
        _density_css(density),
        _PAGE_CSS,
        "</style></head><body>",
    ]
    # Header: centered name + contact line (same part-classification as
    # LaTeX: markdown links, email, linkedin/github, web domains, plain).
    name = (data.contact.name or "").strip()
    if not name and not data.contact.details:
        name = "Candidate"
    out.append("<header>")
    out.append(f"<h1>{_esc(name)}</h1>")
    if data.contact.details:
        parts = [p.strip() for p in
                 re.split(r"\s*(?:[\|•·;]|\s+-\s+)\s*", data.contact.details)
                 if p.strip()]
        rendered: list[str] = []
        for p in parts:
            m_md = _MD_LINK_RE.fullmatch(p)
            if m_md:
                url = _clean_href(m_md.group(2))
                display = _esc(_display_url_text(m_md.group(1)))
                rendered.append(f'<a href="{_esc(url or "")}">{display}</a>'
                                if url else display)
                continue
            if "@" in p and "." in p and " " not in p:
                email = p.lstrip("<").rstrip(">").strip()
                rendered.append(
                    f'<a href="mailto:{_esc(email)}">{_esc(email)}</a>')
                continue
            if "linkedin.com" in p.lower() or "github.com" in p.lower():
                url = p if p.startswith(("http://", "https://")) else \
                    "https://" + p
                rendered.append(f'<a href="{_esc(url)}">'
                                f"{_esc(_display_url_text(p))}</a>")
                continue
            if re.match(r"^(?:https?://|www\.)", p) or \
                    re.search(r"\.(?:io|dev|ai|app|me|tech|com|org)(?:/|$)", p):
                url = p if p.startswith(("http://", "https://")) else \
                    "https://" + p
                rendered.append(f'<a href="{_esc(url)}">'
                                f"{_esc(_display_url_text(p))}</a>")
                continue
            rendered.append(_esc(p))
        out.append(f'<div class="contact">{"  |  ".join(rendered)}</div>')
    out.append("</header>")

    for section in data.sections:
        title = (section.title or "").strip()
        title_lower = title.lower()
        populated = [e for e in section.entries
                     if (e.header or e.right or e.sub or e.sub_right
                         or e.items or e.links)]
        body: list[str] = []
        if populated:
            for e in populated:
                entry_parts = []
                for left, right in _entry_head_rows(e, title_lower):
                    if not left and not right:
                        continue
                    entry_parts.append(
                        f'<div class="eh"><div class="left">{left}</div>'
                        f'<div class="right">{right}</div></div>'
                        if right else
                        f'<div class="eh"><div class="left">{left}</div></div>')
                if e.items:
                    lis = "".join(f"<li>{_inline(item)}</li>"
                                  for item in e.items if _inline(item))
                    if lis:
                        entry_parts.append(f'<ul class="bullets">{lis}</ul>')
                if entry_parts:
                    body.append(f'<div class="entry">'
                                f'{"".join(entry_parts)}</div>')
        if section.simple_lines:
            rows_html = []
            for _is_label, html_row in _simple_rows(section.simple_lines):
                rows_html.append(f'<div class="row">{html_row}</div>')
            if rows_html:
                body.append('<div class="rows">'
                            + "".join(rows_html) + "</div>")
        if not body:
            continue  # empty section would render a bare heading - drop it
        out.append(f"<section><h2>{_esc(title)}</h2>"
                   + "".join(body) + "</section>")

    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The print engine: one headless Chromium subprocess per render
# ---------------------------------------------------------------------------


def html_to_pdf(html_content: str, output_dir, filename: str = "Resume.pdf",
                timeout: int = 90) -> Optional[str]:
    """Prints the HTML page to PDF with the system's headless Chromium/Edge.

    The page is written next to the output as a temp ``__print_source.html``
    and unlinked afterwards. Returns the PDF path, or None when no browser
    is available or the print failed (the caller falls back - an export
    never fails because of the renderer). Concurrency is bounded by
    ``_RENDER_SLOTS`` (the Dossier pool's MAX_CONCURRENT_RENDERS = 2; calls
    beyond that queue).
    """
    if not HAVE_BROWSER:
        logger.info("HTML renderer: no browser available - skipping.")
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pdf_path = output_dir / filename
    src_path = output_dir / "__print_source.html"
    try:
        src_path.write_text(html_content, encoding="utf-8")
    except OSError as e:
        logger.warning("HTML renderer: could not write the print source: %s", e)
        return None
    args = [
        BROWSER_PATH, "--headless", "--disable-gpu", "--disable-extensions",
        "--no-first-run", "--no-default-browser-check",
        "--disable-crash-reporter", "--disable-remote-fonts",
        "--no-pdf-header-footer", "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        src_path.resolve().as_uri(),
    ]
    try:
        with _RENDER_SLOTS:
            proc = subprocess.run(args, capture_output=True, text=True,
                                  timeout=timeout, cwd=str(output_dir))
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            logger.warning("HTML renderer: print produced no PDF (rc=%s): %s",
                           proc.returncode, tail)
            return None
        return str(pdf_path)
    except subprocess.TimeoutExpired:
        logger.warning("HTML renderer: print timed out after %ss.", timeout)
        return None
    except OSError as e:
        logger.warning("HTML renderer: print failed to launch: %s", e)
        return None
    finally:
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# The measured 1-page loop (the density-tier ladder)
# ---------------------------------------------------------------------------

# Tier-1 gating stop: the letter content bottom is 792 - 36 (the @page
# margin) = 756pt; keep the same 12.8pt slack the LaTeX stop
# (_COMPACT_BOTTOM_STOP = 763.2 vs its 776pt block bottom) enforces.
_HTML_BOTTOM_STOP = 743.2


def render_density_tiers(data: ResumeData, output_dir,
                         filename: str = "Optimized_Resume.pdf"
                         ) -> Optional[dict]:
    """The measured loop: probe-renders tiers 1-3 and adopts the first that
    fits ONE page (tier 1 additionally gated by ``_HTML_BOTTOM_STOP``),
    mirroring the LaTeX tier contract - winner copied onto ``filename``,
    losing probes unlinked, one-page-crossing-the-stop tier kept as a
    last-resort fallback, None when no tier fits one page (the caller keeps
    the base render and surfaces the overflow warning).

    Returns {"pdf_path", "html", "density", "pages"} for the adopted render.
    """
    output_dir = Path(output_dir)
    probes: dict[int, tuple[str, Optional[str]]] = {}
    fallback_tier = None
    chosen_tier = None
    for tier in (1, 2, 3):
        page = render_resume_html(data, density=tier)
        probe = html_to_pdf(page, output_dir, f"__density_tier{tier}.pdf")
        probes[tier] = (page, probe)
        if not probe or (pdf_page_count(probe) or 99) != 1:
            continue
        if tier == 1:
            last = pdf_last_baseline(probe)
            if last is not None and last > _HTML_BOTTOM_STOP:
                fallback_tier = fallback_tier or tier
                continue
        chosen_tier = tier
        break
    winner = chosen_tier if chosen_tier is not None else fallback_tier
    if winner is None:
        for _, probe in probes.values():
            if probe:
                Path(probe).unlink(missing_ok=True)
        return None
    page_html, probe = probes[winner]
    dest = output_dir / filename
    shutil.copyfile(probe, dest)
    for tier_probe in (p for _, p in probes.values() if p):
        probe_path = Path(tier_probe)
        if probe_path != dest:
            probe_path.unlink(missing_ok=True)
    return {"pdf_path": str(dest), "html": page_html, "density": winner,
            "pages": 1}
