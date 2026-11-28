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

Env knobs (agent/config.py): ``RESUME_RENDERER`` = auto (default: the
classic look prints LaTeX; serif/modern print this theme) | html (this
theme for every look) | latex; ``RENDER_BROWSER_PATH`` forces a browser
binary; ``RESUME_THEME``
selects the HTML look (``THEMES`` below: classic | serif | modern - fonts,
colors, and heading decoration ONLY, never sizes, so the density ladder
stays the single size authority). The LaTeX path always prints the
classic/conservative look; Unicode is preserved verbatim (the
LaTeX renderer must ASCII-fold it - HTML does not, a rendering-quality win).

Note: unit placement is "ATS-safe by construction" - real text (no tables,
no columns, no images) survives extraction, pinned by the acceptance tests
against the SAME fixture family the LaTeX path is validated with.
"""
from __future__ import annotations

import base64
from functools import lru_cache
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
    pdf_last_baseline,
    pdf_page_count,
)
from .resume_parser import (
    Entry,
    ResumeData,
    _BOLD_LABEL_RE,
    _DATE_RANGE_RE,
    _display_url_text,
    _extract_project_links_and_title,
    _strip_inline,
    mkstemp_pdf_name,
)

logger = logging.getLogger(__name__)

HTML_RENDERER_VERSION = "html-chromium@1"

# ---------------------------------------------------------------------------
# Browser discovery (the print engine)
# ---------------------------------------------------------------------------


def _standard_browser_candidates() -> list[Path]:
    """The fixed-location installs probed after PATH lookup."""
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
    return candidates


def _playwright_cache_roots() -> list[Path]:
    """Playwright's per-platform browser caches (Windows and POSIX)."""
    roots: list[Path] = []
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        roots.append(Path(local) / "ms-playwright")
    roots.append(Path.home() / ".cache" / "ms-playwright")
    return roots


def _find_browser() -> Optional[str]:
    """Locates a Chromium-family binary: explicit env override first, then
    PATH, then the standard Windows/macOS/Linux install locations, then a
    Playwright browser cache (Windows or Linux). Returns the absolute path or
    None."""
    override = os.environ.get("RENDER_BROWSER_PATH", "").strip()
    if override:
        if Path(override).is_file():
            return override
        logger.warning("HTML renderer: RENDER_BROWSER_PATH=%s does not exist - "
                       "ignoring the override and probing the defaults.", override)
    for name in ("chrome", "msedge", "chromium", "google-chrome", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    for cand in _standard_browser_candidates():
        if cand.is_file():
            return str(cand)
    # Playwright's cached Chromium (only when the user installed the package).
    # Linux uses chromium-<rev>/chrome-linux/chrome; Windows keeps chrome-win.
    for pw_root in _playwright_cache_roots():
        if not pw_root.is_dir():
            continue
        for pattern in ("chromium*/chrome-win*/chrome.exe",
                        "chromium*/chrome-linux*/chrome"):
            for probe in sorted(pw_root.glob(pattern)):
                if probe.is_file():
                    return str(probe)
    return None


BROWSER_PATH = _find_browser()
HAVE_BROWSER = BROWSER_PATH is not None
if not HAVE_BROWSER:
    logger.info("HTML renderer: no Chromium-family browser found - the HTML "
                "theme is unavailable (non-classic looks fall back to the "
                "LaTeX classic).")

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
                '{}<a href="{}">{}</a>'.format(
                    ' <span class="psep">|</span> ' if left else "",
                    _esc(u), _esc(_display_url_text(lbl)))
                for lbl, u in _links)
            left += link_html
        if clean_tech:
            left += ((f' <span class="psep">|</span> ' if left else "")
                     + f"<em>{_inline(clean_tech)}</em>")
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
            if (cm and len(cm.group(1)) <= 35
                    and "[" not in cm.group(1)
                    and not cm.group(2).startswith("//")
                    and not cm.group(1).lower().endswith("mailto")):
                m = cm
            else:
                # A verbatim-shipped section can keep raw bullet lines
                # ("- [label](url)"); strip the marker outside the link so
                # it does not print as stray text (parity with the LaTeX
                # simple-line path, Nov 15 2026).
                rows.append((False, _inline(re.sub(r"^\s*[-*•]\s+", "", ln))))
                continue
        label = _strip_inline(m.group(1))
        value = m.group(2)
        rows.append((True, f'<span class="lbl">{_esc(label)}:</span> '
                     f"{_inline(value)}"))
    return rows


# ---------------------------------------------------------------------------
# The page: self-contained HTML with print CSS + strict CSP
# ---------------------------------------------------------------------------

_CSP_META = (
    '<meta http-equiv="Content-Security-Policy" content="default-src '
    "'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "font-src data:; connect-src 'none'; object-src 'none'; frame-src "
    "'none'; base-uri 'none'; form-action 'none'\">"
)

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


# ---------------------------------------------------------------------------
# Themes (round 14): looks, never sizes
# ---------------------------------------------------------------------------
# A theme is a pure CSS-variable set over FONTS, COLORS, and heading
# decoration. It must never set a size: the density variables stay the only
# size authority, so the measured tier ladder and its pdfplumber-verified
# floors hold for every theme (the ladder re-measures per render anyway).
# "classic" reproduces the round-12 look byte-for-byte in effect.
_LM_FONT_FILES = (
    ("Latin Modern Roman", 400, "normal", "lmroman10-regular.otf"),
    ("Latin Modern Roman", 700, "normal", "lmroman10-bold.otf"),
    ("Latin Modern Roman", 400, "italic", "lmroman10-italic.otf"),
    ("Latin Modern Roman", 700, "italic", "lmroman10-bolditalic.otf"),
    ("Latin Modern Caps", 400, "normal", "lmromancaps10-regular.otf"),
)


def _find_latin_modern_dir() -> Optional[Path]:
    """Locate local Latin Modern OTF fonts (MiKTeX / TeX Live / bundled) offline."""
    candidates: list[Path] = [
        Path(__file__).resolve().parent / "templates" / "fonts",
        Path(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "fonts", "opentype", "public", "lm"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "MiKTeX", "fonts", "opentype", "public", "lm"),
        Path("/usr/share/texmf/fonts/opentype/public/lm"),
        Path("/usr/share/fonts/opentype/lm"),
    ]
    kpse = shutil.which("kpsewhich")
    if kpse:
        try:
            proc = subprocess.run([kpse, "lmroman10-regular.otf"], capture_output=True, text=True, timeout=3)
            out = (proc.stdout or "").strip()
            if out:
                candidates.insert(1, Path(out).parent)
        except Exception:
            pass
    for d in candidates:
        if d.is_dir() and (d / "lmroman10-regular.otf").is_file():
            return d
    return None


@lru_cache(maxsize=1)
def _latin_modern_font_css() -> str:
    """Build @font-face data URIs for Latin Modern Roman (classic parity, zero network)."""
    lm_dir = _find_latin_modern_dir()
    if not lm_dir:
        return ""
    rules: list[str] = []
    for family, weight, style, fname in _LM_FONT_FILES:
        fpath = lm_dir / fname
        if not fpath.is_file():
            continue
        b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
        rules.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};"
            f"font-style:{style};src:url('data:font/opentype;base64,{b64}') format('opentype');}}"
        )
    return "\n  ".join(rules)


_THEME_VARS: dict[str, dict[str, str]] = {
    "classic": {
        "font-body": '"Latin Modern Roman", "Latin Modern", "Computer Modern", Georgia, serif',
        "font-head": '"Latin Modern Caps", "Latin Modern Roman", Georgia, serif',
        "ink": "#000", "date": "#000", "muted": "#222",
        "name-tracking": "0.3pt", "name-color": "inherit",
        "h1-variant": "small-caps",
        "h2-case": "none", "h2-variant": "small-caps", "h2-tracking": "0.3pt",
        "h2-rule": "0.5pt solid #000", "h2-color": "inherit",
        "link-deco": "underline",
        "bullet-align": "justify",
    },
    "serif": {
        # Editorial contrast: serif body, sans bold headings, no rule.
        "font-body": 'Charter, Georgia, Cambria, "Times New Roman", serif',
        "font-head": '"Segoe UI", "Helvetica Neue", Arial, sans-serif',
        "ink": "#1a1a1a", "date": "#3a3a3a", "muted": "#555",
        "name-tracking": "0.5pt", "name-color": "inherit",
        "h1-variant": "normal",
        "h2-case": "none", "h2-variant": "normal", "h2-tracking": "0.2pt",
        "h2-rule": "none", "h2-color": "inherit",
        "link-deco": "none",
        "bullet-align": "left",
    },
    "modern": {
        # System sans throughout; the name and headings carry the only hue.
        "font-body": 'system-ui, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
        "font-head": 'system-ui, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
        "ink": "#14181d", "date": "#3f4a56", "muted": "#52606d",
        "name-tracking": "1.2pt", "name-color": "#155e75",
        "h1-variant": "normal",
        "h2-case": "none", "h2-variant": "normal", "h2-tracking": "0.4pt",
        "h2-rule": "none", "h2-color": "#155e75",
        "link-deco": "none",
        "bullet-align": "left",
    },
}

THEMES = frozenset(_THEME_VARS)
DEFAULT_THEME = "classic"


def _theme_css(theme: str) -> str:
    v = _THEME_VARS.get(theme, _THEME_VARS[DEFAULT_THEME])
    parts = ";".join(f"--{k}:{val}" for k, val in v.items())
    return f"  :root{{{parts};}}"


_PAGE_CSS = """
  @page { size: letter; margin: 22pt 32pt 36pt 32pt; }
  * { box-sizing: border-box; }
  body {
    margin: 0; color: var(--ink);
    font-family: var(--font-body);
    font-size: var(--fs); line-height: var(--lh);
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  header { text-align: center; margin: 0 0 2pt; }
  h1 {
    font-family: var(--font-head); font-size: var(--namefs);
    font-variant: var(--h1-variant);
    margin: 0; letter-spacing: var(--name-tracking);
    color: var(--name-color);
  }
  .contact { font-size: var(--contactfs); }
  .contact a, .eh a { color: inherit; text-decoration: var(--link-deco); }
  section { margin-top: var(--secgap); }
  h2 {
    font-family: var(--font-head);
    font-size: var(--h2fs); margin: 0 0 2.5pt;
    text-transform: var(--h2-case); font-variant: var(--h2-variant);
    letter-spacing: var(--h2-tracking);
    color: var(--h2-color);
    border-bottom: var(--h2-rule); padding-bottom: 1.5pt;
  }
  .entry { margin-top: var(--entrygap); }
  .entry:first-of-type { margin-top: 0; }
  .eh { display: flex; justify-content: space-between; align-items: baseline;
        gap: 12pt; }
  .eh .left { min-width: 0; }
  .eh .right { white-space: nowrap; flex-shrink: 0; color: var(--date);
               font-size: calc(var(--fs) - 0.5pt); }
  .sub-row { display: flex; justify-content: space-between;
             align-items: baseline; gap: 12pt; color: var(--ink); }
  .sub-row .right { white-space: nowrap; flex-shrink: 0; }
  .psep, .loc { color: var(--muted); }
  ul.bullets { margin: 2pt 0 0 0; padding-left: 14pt; text-align: var(--bullet-align); }
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


def render_resume_html(data: ResumeData, density: int = 0,
                       theme: str = DEFAULT_THEME) -> str:
    """Renders the structured resume into ONE self-contained HTML page.

    ``density`` selects the tier's CSS variables (0 base, 1-3 progressively
    tighter - the same airy-to-floor ladder the LaTeX compact tiers walk,
    expressed as typography instead of spacing surgery). ``theme`` selects
    the look (fonts/colors/decoration; never sizes - see ``_THEME_VARS``).
    An unknown theme falls back to ``DEFAULT_THEME``. Content is never
    altered; Unicode is preserved verbatim.
    """
    if density not in _DENSITY_VARS:
        density = 0
    if theme not in _THEME_VARS:
        theme = DEFAULT_THEME
    font_css = _latin_modern_font_css() if theme == "classic" else ""
    out: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        _CSP_META,
        "<title>Resume</title>",
        "<style>",
        *([font_css] if font_css else []),
        _density_css(density),
        _theme_css(theme),
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
    # The professional title on its own line under the name (Nov 16 2026,
    # user ask) - same part flow as the LaTeX header.
    if getattr(data.contact, "title", ""):
        out.append(f'<div class="contact">{_esc(data.contact.title)}</div>')
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
                    re.search(r"\.(?:io|dev|ai|app|me|tech|com|org|co|net)(?:/|$)", p):
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


# Chromium refuses to run as root with its sandbox enabled (the child exits 1
# with this marker). The app never silently disables the sandbox: the operator
# opts in with RENDER_NO_SANDBOX=1 (containers only), or runs the app as a
# non-root user. Everything else keeps the sandbox on. If the marker appears
# without the opt-in, the render logs the remedy and falls back to LaTeX.
_SANDBOX_BLOCKED_MARKERS = (
    "Running as root without --no-sandbox",
    "No usable sandbox",
)
_NO_SANDBOX_FLAGS = ("--no-sandbox", "--disable-dev-shm-usage")


def _no_sandbox_opted_in() -> bool:
    """True when the operator explicitly accepted Chromium's sandbox being off."""
    return os.environ.get("RENDER_NO_SANDBOX", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _run_print(args: list[str], output_dir: Path, timeout: int):
    """Run one browser print with stderr in a file, not a pipe.

    Chrome/Edge child processes inherit the print process's stdout/stderr
    handles and can keep them open after the direct process exits.
    ``capture_output=True`` then blocks in ``communicate()`` waiting for EOF -
    observed as a >120s hang under four concurrent renders. A file handle has
    no EOF wait, so a timeout can kill the direct process and return.
    """
    fd, err_name = tempfile.mkstemp(prefix="__print_stderr_", suffix=".log")
    os.close(fd)
    err_path = Path(err_name)
    try:
        with err_path.open("wb") as err_file:
            proc = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
                timeout=timeout,
                cwd=str(output_dir),
            )
        # Tests script stderr on the fake process; production writes the file.
        stderr_text = (proc.stderr if proc.stderr is not None
                       else err_path.read_text(encoding="utf-8",
                                               errors="replace"))
        return proc, stderr_text
    finally:
        try:
            err_path.unlink(missing_ok=True)
        except OSError:
            pass


# Chromium's CSP does not block <meta http-equiv="refresh">; an instant 0s
# refresh navigates away from the file:// source (failing DNS resolution and
# aborting --print-to-pdf with rc=0 on Linux Chrome). Strip meta-refresh tags
# whenever locking down print HTML.
_META_REFRESH_RE = re.compile(
    r"<meta\b[^>]*\bhttp-equiv\s*=\s*(?:[\"']\s*refresh\s*[\"']|refresh\b)[^>]*>",
    flags=re.IGNORECASE,
)


# Chromium has no portable CLI switch that disables JavaScript (Edge's Blink
# preference exits 0 without printing), so the print source carries a strict
# CSP instead. It is injected when the input page has none: inline styles are
# needed, everything else (scripts, remote fetches, frames, plugins) is denied.
def _lock_down_print_source(html_content: str) -> str:
    """Strips meta-refresh redirects and injects the print CSP when absent."""
    locked = _META_REFRESH_RE.sub("", html_content)
    if "content-security-policy" in locked.lower():
        return locked
    marker = re.search(r"<head(?:\s[^>]*)?>", locked, flags=re.IGNORECASE)
    if marker is None:
        return f"<head>{_CSP_META}</head>{locked}"
    return (locked[:marker.end()] + _CSP_META + locked[marker.end():])


def html_to_pdf(html_content: str, output_dir, filename: str = "Resume.pdf",
                timeout: int = 90) -> Optional[str]:
    """Prints the HTML page to PDF with the system's headless Chromium/Edge.

    The page is written next to the output as a unique temp
    ``__print_source_<random>.html`` and unlinked afterwards. Each render gets
    its own browser profile (``--user-data-dir``) and source file, so two
    concurrent renders - or a render while the user's browser is open - never
    share browser state or overwrite each other's print source. Returns the
    PDF path, or None when no browser is available or the print failed (the
    caller falls back - an export never fails because of the renderer).
    Concurrency is bounded by ``_RENDER_SLOTS`` (the Dossier pool's
    MAX_CONCURRENT_RENDERS = 2; calls beyond that queue).

    The print is locked down on the argv: JavaScript and background
    networking are disabled, DNS is blackholed, remote fonts are off, and the
    only non-flag argument is the local ``file://`` source URI. Chromium's
    sandbox stays ON unless the operator sets ``RENDER_NO_SANDBOX=1`` (the
    usual root-in-container escape hatch); a root/no-sandbox failure is logged
    with that remedy and the caller falls back to LaTeX.
    """
    if not HAVE_BROWSER:
        logger.info("HTML renderer: no browser available - skipping.")
        return None
    no_sandbox = _no_sandbox_opted_in()
    if no_sandbox:
        logger.warning(
            "HTML renderer: RENDER_NO_SANDBOX is enabled - Chromium runs "
            "with --no-sandbox (its sandbox is disabled). Use this only in "
            "a container whose threat model accepts it.")
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pdf_path = output_dir / filename
    src_path = None
    try:
        fd, src_name = tempfile.mkstemp(
            prefix="__print_source_", suffix=".html", dir=str(output_dir))
        os.close(fd)
        src_path = Path(src_name)
        src_path.write_text(_lock_down_print_source(html_content),
                            encoding="utf-8")
    except OSError as e:
        logger.warning("HTML renderer: could not write the print source: %s", e)
        if src_path is not None:
            try:
                src_path.unlink(missing_ok=True)
            except OSError:
                pass
        return None
    try:
        profile_dir = Path(tempfile.mkdtemp(prefix="resume-print-profile-"))
    except OSError as e:
        logger.warning("HTML renderer: could not create a browser profile: %s", e)
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    args = [
        BROWSER_PATH, "--headless", "--disable-gpu", "--disable-extensions",
        "--no-first-run", "--no-default-browser-check",
        "--disable-crash-reporter", "--disable-remote-fonts",
        "--disable-background-networking",
        "--host-resolver-rules=MAP * ~NOTFOUND",
        f"--user-data-dir={profile_dir}",
        "--no-pdf-header-footer", "--print-to-pdf-no-header",
    ]
    if no_sandbox:
        args += list(_NO_SANDBOX_FLAGS)
    args += [f"--print-to-pdf={pdf_path}", src_path.resolve().as_uri()]
    try:
        with _RENDER_SLOTS:
            proc, stderr_text = _run_print(args, output_dir, timeout)
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            if (not no_sandbox
                    and any(m in stderr_text for m in _SANDBOX_BLOCKED_MARKERS)):
                logger.error(
                    "HTML renderer: Chromium refused to start because its "
                    "sandbox is unavailable (usually running as root). Set "
                    "RENDER_NO_SANDBOX=1 to opt in to --no-sandbox in the "
                    "container, or run the app as a non-root user. Falling "
                    "back to the LaTeX renderer.")
            else:
                tail = stderr_text[-800:]
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
        shutil.rmtree(profile_dir, ignore_errors=True)



# ---------------------------------------------------------------------------
# The measured 1-page loop (the density-tier ladder)
# ---------------------------------------------------------------------------

# Tier-1 gating stop: the letter content bottom is 792 - 36 (the @page
# margin) = 756pt; keep the same 12.8pt slack the LaTeX stop
# (_COMPACT_BOTTOM_STOP = 763.2 vs its 776pt block bottom) enforces.
_HTML_BOTTOM_STOP = 743.2


def render_density_tiers(data: ResumeData, output_dir,
                         filename: str = "Optimized_Resume.pdf",
                         theme: str = DEFAULT_THEME,
                         target_pages: int = 1
                         ) -> Optional[dict]:
    """The measured loop: probe-renders tiers 1-3 and adopts the first that
    fits ``target_pages`` AND passes geometry bounds (tier 1 additionally gated
    by ``_HTML_BOTTOM_STOP`` on 1-page targets).

    Returns {"pdf_path", "html", "density", "pages"} for the adopted render.
    """
    from .pdf_verify import check_pdf_geometry

    target_pages = max(1, min(2, int(target_pages or 1)))
    output_dir = Path(output_dir)
    probes: dict[int, tuple[str, Optional[str], int, bool]] = {}
    fallback_tier = None
    chosen_tier = None
    for tier in (1, 2, 3):
        page = render_resume_html(data, density=tier, theme=theme)
        probe = html_to_pdf(page, output_dir,
                            mkstemp_pdf_name(output_dir, f"__density_tier{tier}_").name)
        pages = (pdf_page_count(probe) or 99) if probe else 99
        geom_res = check_pdf_geometry(probe) if probe else {"ok": False, "detail": ""}
        geom_ok = bool(
            probe and (
                geom_res["ok"]
                or str(geom_res.get("detail") or "").startswith("geometry measurement failed")
            )
        )
        probes[tier] = (page, probe, pages, geom_ok)
        if not probe or pages > target_pages:
            continue
        if not geom_ok:
            fallback_tier = fallback_tier or tier
            continue
        if tier == 1 and target_pages == 1:
            last = pdf_last_baseline(probe)
            if last is not None and last > _HTML_BOTTOM_STOP:
                fallback_tier = fallback_tier or tier
                continue
        chosen_tier = tier
        break
    winner = chosen_tier if chosen_tier is not None else fallback_tier
    if winner is None:
        for _, probe, _, _ in probes.values():
            if probe:
                Path(probe).unlink(missing_ok=True)
        return None
    page_html, probe, pages, _ = probes[winner]
    dest = output_dir / filename
    shutil.copyfile(probe, dest)
    for tier_probe in (p for _, p, _, _ in probes.values() if p):
        probe_path = Path(tier_probe)
        if probe_path != dest:
            probe_path.unlink(missing_ok=True)
    return {"pdf_path": str(dest), "html": page_html, "density": winner,
            "pages": pages}
