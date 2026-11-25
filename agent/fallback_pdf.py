"""Last-resort xhtml2pdf fallback exporter used when Chromium and LaTeX fail."""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from xhtml2pdf import pisa


def _sanitize_html_for_pdf(html: str) -> str:
    """Strip unsafe elements, event handlers, and non-http(s) resource URLs
    before passing HTML to xhtml2pdf (mitigates SSRF via local file:// URLs).
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ("script", "style", "iframe", "object", "embed", "applet", "form"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    _URL_ATTRS = {"src", "href", "action", "poster", "data", "formaction"}
    _SAFE_SCHEMES = re.compile(r"^https?://", re.IGNORECASE)

    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr.lower().startswith("on"):
                del tag[attr]
        for attr in _URL_ATTRS & set(tag.attrs.keys()):
            val = tag[attr]
            if isinstance(val, str) and not _SAFE_SCHEMES.match(val):
                del tag[attr]

    return str(soup)


def _html_to_pdf(markdown_text: str, filename: str, title: str, output_dir: Path) -> str:
    """Convert Markdown -> HTML -> PDF with CSS styling via xhtml2pdf."""
    from .helpers import _strip_emoji_for_pdf, _strip_emoji_from_html

    css_style = """
    body { font-family: 'Helvetica', sans-serif; font-size: 12px; line-height: 1.4; color: #333; }
    h1 { font-size: 24px; border-bottom: 2px solid #333; margin-bottom: 10px; text-transform: uppercase; }
    h2 { font-size: 16px; border-bottom: 1px solid #ccc; margin-top: 20px; text-transform: uppercase; color: #555; }
    ul { padding-left: 20px; }
    li { margin-bottom: 5px; }
    strong { color: #000; }
    """
    clean_text = _strip_emoji_for_pdf(markdown_text)
    html_body = _strip_emoji_from_html(markdown.markdown(clean_text))
    full_html = f"""
    <html>
    <head>
        <style>{css_style}</style>
    </head>
    <body>
        <h1>{title}</h1>
        {html_body}
    </body>
    </html>
    """
    full_html = _sanitize_html_for_pdf(full_html)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, "wb") as pdf_file:
        result = pisa.CreatePDF(full_html, dest=pdf_file)
    if result.err:
        raise RuntimeError(f"Could not create {title} PDF.")
    return str(filepath)
