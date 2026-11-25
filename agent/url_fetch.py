"""URL validation, SSRF protection, and document loading."""
import ipaddress
import re
import socket
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import robots_check
from .config import (ALLOWED_JOB_URL_DOMAINS, MAX_FETCH_BYTES, MAX_PDF_BYTES,
                     MAX_PDF_PAGES, MAX_RESUME_CHARS, FETCH_TIMEOUT_SECONDS)
from .helpers import _safe_print, _safe_truncate

from pypdf import PdfReader


def _is_public_host(hostname: str) -> bool:
    """Resolve a hostname and reject loopback, private, and reserved addresses."""
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Could not resolve the job URL host.") from exc
    for address in {item[4][0] for item in addresses}:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return False
    return True


class _NoRedirect(HTTPRedirectHandler):
    """Disable implicit redirects so every destination is checked explicitly."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


URL_IMPORT_DISABLED_MESSAGE = (
    "URL import is disabled until ALLOWED_JOB_URL_DOMAINS is configured. "
    "Paste the job description instead."
)


def looks_like_job_url(value: str) -> bool:
    """True when the input is an http(s) URL rather than pasted text."""
    return (value or "").strip().lower().startswith(("http://", "https://"))


def _static_job_url_error(raw_url: str) -> str | None:
    """Configuration-only URL checks shared by the loader and the SSE
    preflight. DNS/SSRF and fetch checks stay in ``_validate_job_url`` /
    ``_fetch_job_description`` so this helper never touches the network."""
    parsed = urlparse((raw_url or "").strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return "Job URLs must use HTTPS."
    if not ALLOWED_JOB_URL_DOMAINS:
        return URL_IMPORT_DISABLED_MESSAGE
    hostname = parsed.hostname.lower()
    if not any(hostname == domain or hostname.endswith(f".{domain}")
               for domain in ALLOWED_JOB_URL_DOMAINS):
        return "This job URL domain is not on the deployment allowlist."
    return None


def job_url_preflight_error(raw_url: str) -> str | None:
    """Return the user-facing reason a URL input cannot be imported, or None.

    Static only (scheme, allowlist configuration, host allowlist): the
    runners call this before tracing/recording a run so an unconfigured URL
    import fails as a clean SSE error instead of a loader traceback. A
    non-URL input always passes.
    """
    if not looks_like_job_url(raw_url):
        return None
    return _static_job_url_error(raw_url)


def _validate_job_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    error = _static_job_url_error(raw_url)
    if error:
        raise ValueError(error)
    parsed = urlparse(raw_url)
    if not _is_public_host(parsed.hostname):
        raise ValueError("Job URL must resolve to a public IP address.")
    return parsed.geturl()


def _robots_gate_check(url: str) -> None:
    """Raise when robots.txt policy does not permit fetching ``url``.

    Adapted from ``tools/robots_check.py`` in the MIT ai-job-search repo
    (see NOTICE): the gate fails closed, so an unreadable or unconfirmed
    robots.txt also stops the fetch. There is no browser impersonation and
    no retry around a refusal - the user is asked to paste the posting.
    """
    code, msg = robots_check.gate(url)
    if code != 0:
        raise ValueError(
            f"Job URL refused by robots.txt policy: {msg}. "
            "Paste the job description text instead."
        )


def _read_page_body(response) -> str:
    """Decode the body of a fetched page (``_fetch_job_description`` contract).

    The body is returned decoded and bounded by MAX_FETCH_BYTES - NOT by the
    much smaller MAX_JOB_CHARS. The loader cleans raw HTML first (hidden
    elements, markup) and truncates the visible text afterwards, so a markup-
    heavy page does not lose half its posting to the character cap.
    """
    content_type = response.headers.get_content_type()
    if content_type not in {"text/html", "text/plain"}:
        raise ValueError("Job URL must return HTML or plain text.")
    body = response.read(MAX_FETCH_BYTES + 1)
    if len(body) > MAX_FETCH_BYTES:
        raise ValueError("Job page is too large to import.")
    return body.decode(response.headers.get_content_charset() or "utf-8",
                       errors="replace")


def _fetch_job_description(raw_url: str) -> str:
    """Fetch an allowlisted job page with bounded redirects and response size.

    robots.txt is enforced on every hop (RFC 9309, ``agent/robots_check``;
    adapted from ``tools/robots_check.py``): fetching is never done against
    a policy that says no. The opener disables implicit redirects, so every
    destination is validated (allowlist + public IP) before it is fetched.
    Requests identify honestly as the app user agent; a site that blocks
    automated access gets a clear error asking the user to paste the posting
    instead of a browser-impersonating retry.
    """
    current_url = _validate_job_url(raw_url)
    _robots_gate_check(current_url)
    opener = build_opener(_NoRedirect())
    for _ in range(4):
        try:
            request = Request(
                current_url,
                headers={"User-Agent": robots_check.USER_AGENT},
            )
            response = opener.open(request, timeout=FETCH_TIMEOUT_SECONDS)
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Job URL redirected without a destination.")
                current_url = _validate_job_url(location)
                _robots_gate_check(current_url)
                continue
            return _read_page_body(response)
        except HTTPError as exc:
            location = exc.headers.get("Location") \
                if exc.code in {301, 302, 303, 307, 308} else None
            if location:
                current_url = _validate_job_url(location)
                _robots_gate_check(current_url)
                continue
            if exc.code == 403:
                raise ValueError(
                    "The site blocked automated access (HTTP 403). "
                    "Open the job posting in your browser and paste the "
                    "description instead."
                ) from exc
            raise
    raise ValueError("Too many job URL redirects.")


def _extract_page_text_with_links(page) -> str:
    """Extract page text while converting hyperlinked text into markdown links [Text](URL)."""
    links = []
    annots = page.get('/Annots') or []
    for a in annots:
        try:
            obj = a.get_object()
            if obj.get('/Subtype') == '/Link':
                action = obj.get('/A')
                if action and action.get('/URI'):
                    uri = str(action.get('/URI')).strip()
                    rect = obj.get('/Rect')
                    if uri and rect:
                        links.append({
                            'uri': uri,
                            'rect': [float(x) for x in rect],
                        })
        except Exception:
            continue

    if not links:
        return page.extract_text() or ""

    fragments = []

    def visitor(text, cm, tm, font_dict, font_size):
        if not text:
            return
        x, y = tm[4], tm[5]
        size = font_size if font_size else 10.0
        clean_text = text.strip()
        if clean_text:
            matched_uri = None
            for lk in links:
                x0, y0, x1, y1 = lk['rect']
                if (x0 - 8 <= x <= x1 + 8) and (y0 - 5 <= y <= y1 + 5):
                    matched_uri = lk['uri']
                    break
            if matched_uri and not clean_text.startswith(('http://', 'https://', 'mailto:')):
                leading = text[:len(text) - len(text.lstrip())]
                trailing = text[len(text.rstrip()):]
                fragments.append((f"{leading}[{clean_text}]({matched_uri}){trailing}", x, y, size))
                return
        fragments.append((text, x, y, size))

    try:
        page.extract_text(visitor_text=visitor)
        extracted = _join_visitor_fragments(fragments)
        if extracted.strip():
            return extracted
    except Exception:
        pass

    return page.extract_text() or ""


# ============================================================
# Gap-aware visitor fragment joining
#
# pypdf's visitor_text hands out raw text runs WITHOUT inter-run
# spacing, so a naive "".join() glues words together ("PythonDeveloper",
# "Machine LearningSystems") on exactly the fancy-design PDFs that carry
# the most hyperlink annotations. The join below reconstructs whitespace
# from run geometry: coordinates only, no PDF internals needed.
# ============================================================

# Rough per-glyph advance widths in em (font-size units). Overestimation
# widens the previous run -> shrinksthe gap -> misses a space; that's the
# harmless failure (spaces are split-safe for keyword matching, glued
# words are not), so thresholds are biased toward inserting separators.
_NARROW_CHARS = set("ijltf.,'\":;|!()[]{}")   # ~0.3em glyph advance
_NARROW_ADVANCE_EM = 0.30
_UPPER_DIGIT_EM = 0.62
_SPACE_CHAR_EM = 0.25
_DEFAULT_EM = 0.50
# A horizontal gap beyond this fraction of the font size separates runs
# into distinct words; smaller gaps are mid-word run splits (inline
# formatting/ligature changes) and must glue back together.
_SPACE_GAP_EM = 0.15
# Baseline moves farther than this = new visual line (typical leading is
# ~1.2em; same-line jitter sits well below half an em).
_LINE_Y_TOL_EM = 0.50


def _estimate_run_width(text: str, font_size: float) -> float:
    """Estimate the rendered width of a text run from its glyphs (em-based)."""
    width = 0.0
    for ch in text:
        if ch in _NARROW_CHARS:
            width += _NARROW_ADVANCE_EM
        elif ch.isupper() or ch.isdigit():
            width += _UPPER_DIGIT_EM
        elif ch.isspace():
            width += _SPACE_CHAR_EM
        elif ord(ch) > 0x2E7F:  # CJK/fullwidth glyphs are ~1em wide
            width += 1.0
        else:
            width += _DEFAULT_EM
    return width * font_size


def _visible_text(fragment: str) -> str:
    """Rendered-text approximation: markdown-wrapped links collapse to just
    the label glyphs ([Live Demo](url) renders 'Live Demo', not the URL)."""
    return re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', fragment)


def _join_visitor_fragments(fragments: list[tuple[str, float, float, float]]) -> str:
    """Rebuild page text from (text, x, y, font_size) visitor fragments.

    Whitespace policy:
    - fragment carries leading whitespace, or the previous one carries
      trailing whitespace  -> attach directly (the space already exists);
    - baseline y changed by more than _LINE_Y_TOL_EM * font size -> newline;
    - same line, horizontal gap between the previous run's estimated right
      edge and this run's left edge exceeds _SPACE_GAP_EM * font size -> space;
    - near-zero gap -> mid-word run split, attach directly (re-glues
      words that a formatting change split mid-glyph).
    """
    out: list[str] = []
    px = py = pfs = 0.0
    prev_text = ""
    for i, (text, x, y, fs) in enumerate(fragments):
        if not text:
            continue
        if i > 0:
            if text[0].isspace() or prev_text.endswith((" ", "\t", "\n")):
                pass  # fragment boundary already carries whitespace
            else:
                same_line = abs(y - py) <= _LINE_Y_TOL_EM * max(fs, pfs)
                if not same_line:
                    out.append("" if prev_text.endswith("\n") else "\n")
                elif x > px:
                    gap = x - (px + _estimate_run_width(_visible_text(prev_text), pfs))
                    if gap > _SPACE_GAP_EM * max(fs, pfs):
                        out.append(" ")
        out.append(text)
        px, py, pfs, prev_text = x, y, fs, text
    return "".join(out)


def _extract_plumber_page_with_links(plumber_page) -> str:
    """Extract page text using per-character geometry for exact hyperlink matching.

    Unlike pypdf's visitor_text (whose tm[4] can stay anchored at the line
    origin across multiple text runs on a line, causing right-aligned links
    like 'Live Demo' to miss their annotation boxes and left-aligned links to
    bleed across entire rows), pdfplumber exposes every glyph's bounding box.
    """
    chars = getattr(plumber_page, "chars", None) or []
    if not chars:
        return ""

    links = getattr(plumber_page, "hyperlinks", None) or []
    link_boxes: list[tuple[float, float, float, float, float, str]] = []
    for h in links:
        uri = (h.get("uri") or "").strip()
        if not uri:
            continue
        x0, top = float(h["x0"]), float(h["top"])
        x1, bot = float(h["x1"]), float(h["bottom"])
        area = max(0.0, (x1 - x0) * (bot - top))
        link_boxes.append((area, x0, top, x1, bot, uri))
    # Tightest bounding box wins when two link annotations sit close together.
    link_boxes.sort(key=lambda item: item[0])

    tagged: list[tuple[dict, str | None]] = []
    for c in chars:
        ch = c.get("text", "")
        if not ch:
            continue
        cx = (float(c["x0"]) + float(c["x1"])) / 2.0
        cy = (float(c["top"]) + float(c["bottom"])) / 2.0
        matched_uri = None
        for _, x0, top, x1, bot, uri in link_boxes:
            if (x0 - 0.5 <= cx <= x1 + 0.5) and (top - 1.5 <= cy <= bot + 1.5):
                matched_uri = uri
                break
        tagged.append((c, matched_uri))

    if not tagged:
        return ""

    tagged.sort(key=lambda item: (round(float(item[0]["top"]), 1), float(item[0]["x0"])))
    lines: list[list[tuple[dict, str | None]]] = []
    cur_line: list[tuple[dict, str | None]] = []
    cur_top: float | None = None
    cur_size = 10.0
    for c, uri in tagged:
        top = float(c["top"])
        size = float(c.get("size") or 10.0)
        if cur_line and cur_top is not None and abs(top - cur_top) > 0.5 * max(size, cur_size):
            lines.append(cur_line)
            cur_line = [(c, uri)]
            cur_top = top
            cur_size = size
        else:
            cur_line.append((c, uri))
            if cur_top is None:
                cur_top = top
                cur_size = size
    if cur_line:
        lines.append(cur_line)

    out_lines: list[str] = []
    for line in lines:
        line.sort(key=lambda item: float(item[0]["x0"]))
        runs: list[list] = []
        prev_x1: float | None = None
        prev_size = 10.0
        for c, uri in line:
            ch = c["text"]
            x0 = float(c["x0"])
            x1 = float(c["x1"])
            size = float(c.get("size") or 10.0)
            need_space = (
                prev_x1 is not None
                and not ch.isspace()
                and (x0 - prev_x1) > 0.16 * max(size, prev_size)
            )
            if runs and runs[-1][0] == uri:
                if need_space and not runs[-1][1].endswith(" "):
                    runs[-1][1] += " "
                runs[-1][1] += ch
            else:
                if need_space and runs and not runs[-1][1].endswith(" "):
                    runs[-1][1] += " "
                runs.append([uri, ch])
            prev_x1 = x1
            prev_size = size

        parts: list[str] = []
        for uri, txt in runs:
            clean = txt.strip()
            if uri and clean and not clean.startswith(("http://", "https://", "mailto:")):
                lead = txt[:len(txt) - len(txt.lstrip())]
                trail = txt[len(txt.rstrip()):]
                parts.append(f"{lead}[{clean}]({uri}){trail}")
            else:
                parts.append(txt)
        out_lines.append("".join(parts).rstrip())
    return "\n".join(out_lines)


def _load_resume_pdf(file_path: str) -> str:
    path = Path(file_path)
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("Resume PDF must be 5 MB or smaller.")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError("Uploaded file is not a valid PDF.")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"Resume PDF must have {MAX_PDF_PAGES} pages or fewer.")

    plumber_pages = None
    plumber_ctx = None
    try:
        import pdfplumber
        plumber_ctx = pdfplumber.open(str(path))
        plumber_pages = plumber_ctx.pages
    except Exception:
        plumber_pages = None

    page_texts = []
    try:
        for idx, page in enumerate(reader.pages):
            txt = ""
            if plumber_pages is not None and idx < len(plumber_pages):
                try:
                    txt = _extract_plumber_page_with_links(plumber_pages[idx])
                except Exception:
                    txt = ""
            if not txt.strip():
                txt = _extract_page_text_with_links(page)
            if txt.strip():
                page_texts.append(txt)
    finally:
        if plumber_ctx is not None:
            try:
                plumber_ctx.close()
            except Exception:
                pass

    text = "\n".join(page_texts)
    return _safe_truncate(text, MAX_RESUME_CHARS, "Resume")


# Shared 5 MB upload cap for both resume formats (PDF & DOCX).
MAX_RESUME_FILE_BYTES = MAX_PDF_BYTES


def _extract_docx_paragraph_text(p) -> str:
    """Extract paragraph text while converting hyperlink runs to markdown [Text](URL)."""
    try:
        from docx.oxml.ns import qn
        t_tag = qn('w:t')
        parts = []
        for child in p._p:
            tag = child.tag.split('}')[-1]
            if tag == 'hyperlink':
                r_id = child.get(qn('r:id'))
                url = ""
                try:
                    if r_id and p.part and r_id in p.part.rels:
                        url = p.part.rels[r_id].target_ref
                except Exception:
                    pass
                txt = "".join(t.text for t in child.iter(t_tag) if t.text).strip()
                if txt and url and not txt.startswith(('http://', 'https://', 'mailto:')):
                    parts.append(f"[{txt}]({url})")
                elif txt:
                    parts.append(txt)
            elif tag == 'r':
                txt = "".join(t.text for t in child.iter(t_tag) if t.text)
                if txt:
                    parts.append(txt)
        if parts:
            res = "".join(parts).strip()
            if res:
                return res
    except Exception:
        pass
    return p.text.strip()


def _extract_docx_cell_text(cell) -> str:
    """Extract table-cell text preserving hyperlink runs as markdown links.

    ATS-styled Word resumes put entire sections inside table grids; reading
    the cell's raw text drops every hyperlink target the paragraph reader
    would have preserved (audit finding P1). Falls back to raw cell text
    when the paragraph walk yields nothing or fails.
    """
    lines: list[str] = []
    try:
        for p in cell.paragraphs:
            txt = _extract_docx_paragraph_text(p)
            if txt:
                lines.append(txt)
    except Exception:
        lines = []
    if lines:
        return "\n".join(lines)
    try:
        return cell.text.strip()
    except Exception:
        return ""


def _load_resume_docx(file_path: str) -> str:
    """Extracts text (paragraphs + table rows) from a Word (.docx) resume,
    converting hyperlinked text into markdown links [Text](URL)."""
    path = Path(file_path)
    if path.stat().st_size > MAX_RESUME_FILE_BYTES:
        raise ValueError("Resume file must be 5 MB or smaller.")
    with path.open("rb") as source:
        if source.read(2) != b"PK":
            raise ValueError("Uploaded file is not a valid Word document.")
    # Deferred import keeps python-docx optional at import time of this module.
    from docx import Document

    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        txt = _extract_docx_paragraph_text(p)
        if txt:
            parts.append(txt)
    for table in doc.tables:
        for row in table.rows:
            cells = []
            seen_tcs: list[int] = []
            for c in row.cells:
                # python-docx repeats merged cells across their span, sharing
                # the same w:tc element; dedupe by element identity so
                # genuine duplicate values in distinct cells stay visible.
                tc_id = id(getattr(c, '_tc', c))
                txt = _extract_docx_cell_text(c)
                if txt and tc_id not in seen_tcs:
                    seen_tcs.append(tc_id)
                    cells.append(txt)
            if cells:
                parts.append(" | ".join(cells))
    if not parts:
        raise ValueError("The Word document appears to be empty.")
    return _safe_truncate("\n".join(parts), MAX_RESUME_CHARS, "Resume")

