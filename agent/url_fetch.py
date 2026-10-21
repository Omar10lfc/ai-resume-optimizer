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
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _safe_truncate(text, MAX_RESUME_CHARS, "Resume")


# Shared 5 MB upload cap for both resume formats (PDF & DOCX).
MAX_RESUME_FILE_BYTES = MAX_PDF_BYTES


def _load_resume_docx(file_path: str) -> str:
    """Extracts plain text (paragraphs + table rows) from a Word (.docx) resume."""
    path = Path(file_path)
    if path.stat().st_size > MAX_RESUME_FILE_BYTES:
        raise ValueError("Resume file must be 5 MB or smaller.")
    with path.open("rb") as source:
        if source.read(2) != b"PK":
            raise ValueError("Uploaded file is not a valid Word document.")
    # Deferred import keeps python-docx optional at import time of this module.
    from docx import Document

    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    if not parts:
        raise ValueError("The Word document appears to be empty.")
    return _safe_truncate("\n".join(parts), MAX_RESUME_CHARS, "Resume")
