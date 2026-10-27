"""URL validation, SSRF protection, and document loading."""
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import (ALLOWED_JOB_URL_DOMAINS, MAX_FETCH_BYTES, MAX_PDF_BYTES,
                     MAX_PDF_PAGES, MAX_RESUME_CHARS, MAX_JOB_CHARS,
                     FETCH_TIMEOUT_SECONDS)
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


def _validate_job_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Job URLs must use HTTPS.")
    hostname = parsed.hostname.lower()
    if not ALLOWED_JOB_URL_DOMAINS:
        raise ValueError(
            "URL import is disabled until ALLOWED_JOB_URL_DOMAINS is configured. "
            "Paste the job description instead."
        )
    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in ALLOWED_JOB_URL_DOMAINS):
        raise ValueError("This job URL domain is not on the deployment allowlist.")
    if not _is_public_host(hostname):
        raise ValueError("Job URL must resolve to a public IP address.")
    return parsed.geturl()


def _fetch_job_description(raw_url: str) -> str:
    """Fetch an allowlisted job page with bounded redirects and response size."""
    current_url = _validate_job_url(raw_url)
    opener = build_opener(_NoRedirect())
    for _ in range(4):
        request = Request(current_url, headers={"User-Agent": "ResumeOptimizer/1.0"})
        response = opener.open(request, timeout=FETCH_TIMEOUT_SECONDS)
        if response.status in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            if not location:
                raise ValueError("Job URL redirected without a destination.")
            current_url = _validate_job_url(location)
            continue
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "text/plain"}:
            raise ValueError("Job URL must return HTML or plain text.")
        body = response.read(MAX_FETCH_BYTES + 1)
        if len(body) > MAX_FETCH_BYTES:
            raise ValueError("Job page is too large to import.")
        return _safe_truncate(body.decode(response.headers.get_content_charset() or "utf-8", errors="replace"), MAX_JOB_CHARS, "Job description")
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

    linked_parts = []

    def visitor(text, cm, tm, font_dict, font_size):
        if not text:
            return
        x, y = tm[4], tm[5]
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
                linked_parts.append(f"{leading}[{clean_text}]({matched_uri}){trailing}")
                return
        linked_parts.append(text)

    try:
        page.extract_text(visitor_text=visitor)
        extracted = "".join(linked_parts)
        if extracted.strip():
            return extracted
    except Exception:
        pass

    return page.extract_text() or ""


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
    page_texts = []
    for page in reader.pages:
        txt = _extract_page_text_with_links(page)
        if txt.strip():
            page_texts.append(txt)
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
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    if not parts:
        raise ValueError("The Word document appears to be empty.")
    return _safe_truncate("\n".join(parts), MAX_RESUME_CHARS, "Resume")

