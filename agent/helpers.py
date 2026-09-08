"""Shared utility functions: printing, truncation, text cleaning, PDF export."""
import logging
import re
import unicodedata
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from xhtml2pdf import pisa

logger = logging.getLogger(__name__)


def _safe_print(msg: str):
    """Log via the standard logger (handles encoding, redirectable to files)."""
    logger.info(msg)


def _safe_truncate(text: str, max_chars: int, label: str = "text") -> str:
    """Truncate text at the nearest word boundary."""
    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        _safe_print(f"   [Warning] {label} truncated from {len(text)} to {len(truncated)} characters.")
        return truncated
    return text


def _strip_code_fences(text: str) -> str:
    """Strip wrapping markdown code fences that LLMs often add."""
    stripped = text.strip()
    for _ in range(3):
        pattern = r'^`{3,}(?:\w*)\s*\n(.*?)\n\s*`{3,}\s*$'
        match = re.match(pattern, stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
        else:
            break
    return stripped


# Explicit Unicode-to-ASCII mapping for PDF output
_UNICODE_REPLACEMENTS = {
    '\u2705': '[Y]', '\u274c': '[N]', '\U0001f7e2': '[+]', '\U0001f7e1': '[~]', '\U0001f534': '[-]',
    '\U0001f4c4': '', '\U0001f4ca': '', '\U0001f4cb': '', '\U0001f4ce': '', '\U0001f4dd': '',
    '\U0001f4e5': '', '\U0001f3a4': '', '\u2709\ufe0f': '', '\U0001f50d': '', '\U0001f4a1': '',
    '\u23f3': '', '\u2764\ufe0f': '', '\u2764': '',
    '\u2010': '-', '\u2011': '-', '\u2012': '-',
    '\u2013': '-', '\u2014': '--',
    '\u2015': '--', '\u2212': '-',
    '\u2043': '-', '\uFE58': '-', '\uFE63': '-',
    '\uFF0D': '-',
    '\u2192': '->', '\u2190': '<-', '\u2022': '-',
    '\u2023': '>', '\u25B6': '>',
    '\u2502': '|', '\u2500': '-',
    '\u25CF': '*', '\u25CB': 'o', '\u25AA': '*',
    '\u2026': '...',
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00AB': '"', '\u00BB': '"',
    '\u202f': ' ', '\xa0': ' ', '\u2009': ' ', '\u200a': ' ', '\u2003': ' ',
    '\u200b': '', '\u200c': '', '\u200d': '',
    '\ufeff': '',
    '\ufe0f': '', '\ufe0e': '',
}


def _force_ascii(text: str) -> str:
    """Force text to pure ASCII."""
    for unicode_char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)
    cleaned = []
    for ch in text:
        if ord(ch) < 128:
            cleaned.append(ch)
        else:
            decomposed = unicodedata.normalize('NFD', ch)
            ascii_parts = [c for c in decomposed if ord(c) < 128]
            if ascii_parts:
                cleaned.extend(ascii_parts)
    return ''.join(cleaned)


def _strip_emoji_for_pdf(text: str) -> str:
    """Replace emoji/Unicode with ASCII before markdown->HTML conversion."""
    return _force_ascii(text)


def _strip_emoji_from_html(html: str) -> str:
    """Clean non-ASCII from HTML output after markdown conversion."""
    html_entity_map = {
        '&ndash;': '-', '&mdash;': '--', '&lsquo;': "'", '&rsquo;': "'",
        '&ldquo;': '"', '&rdquo;': '"', '&bull;': '-', '&hellip;': '...',
        '&nbsp;': ' ', '&trade;': '(TM)', '&copy;': '(c)', '&reg;': '(R)',
        '&laquo;': '"', '&raquo;': '"', '&minus;': '-',
    }
    for entity, replacement in html_entity_map.items():
        html = html.replace(entity, replacement)
    result = []
    in_tag = False
    for ch in html:
        if ch == '<':
            in_tag = True
            result.append(ch)
        elif ch == '>':
            in_tag = False
            result.append(ch)
        elif in_tag:
            result.append(ch)
        elif ord(ch) < 128:
            result.append(ch)
        else:
            decomposed = unicodedata.normalize('NFD', ch)
            ascii_parts = [c for c in decomposed if ord(c) < 128]
            if ascii_parts:
                result.extend(ascii_parts)
    return ''.join(result)


def _filter_hallucinated_sections(optimized: str, original: str) -> str:
    """Remove sections from optimized resume that don't exist in original."""
    suspect_sections = [
        'professional affiliations', 'affiliations', 'certifications',
        'awards', 'honors', 'publications', 'memberships',
        'professional memberships', 'awards and honors',
        'certifications and licenses', 'licenses',
    ]
    original_lower = original.lower()
    original_has = set()
    for section in suspect_sections:
        if (f'## {section}' in original_lower or
            f'# {section}' in original_lower or
            f'**{section}' in original_lower or
            f'{section}:' in original_lower or
            f'{section}\n' in original_lower):
            original_has.add(section)

    lines = optimized.split('\n')
    filtered_lines = []
    skip_until_next_header = False
    skipped_section = None

    for line in lines:
        line_lower = line.lower().strip()
        is_suspect_header = False
        for section in suspect_sections:
            if (line_lower.startswith(f'## {section}') or
                line_lower.startswith(f'# {section}') or
                line_lower.startswith(f'### {section}') or
                line_lower == f'**{section}**' or
                line_lower == f'{section}:'):
                if section not in original_has:
                    is_suspect_header = True
                    skipped_section = section
                    break
        if is_suspect_header:
            skip_until_next_header = True
            _safe_print(f"   [Hallucination Filter] Removed fabricated section: '{skipped_section}'")
            continue
        if skip_until_next_header:
            if line_lower.startswith('#') or (line_lower.startswith('**') and line_lower.endswith('**')):
                skip_until_next_header = False
                filtered_lines.append(line)
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def _sanitize_html_for_pdf(html: str) -> str:
    """Strip dangerous tags and attributes from HTML before passing to xhtml2pdf.

    Mitigates SSRF via file:// URLs and script injection in LLM-generated content.
    xhtml2pdf follows arbitrary URLs in <img src>, <link href>, etc. — this
    function neutralizes that by removing non-http(s) resource references.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove tags that can execute code or load external resources
    for tag_name in ("script", "style", "iframe", "object", "embed", "applet", "form"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove event handler attributes and dangerous URL schemes
    _URL_ATTRS = {"src", "href", "action", "poster", "data", "formaction"}
    _SAFE_SCHEMES = re.compile(r'^https?://', re.IGNORECASE)

    for tag in soup.find_all(True):
        # Strip on* event handlers
        for attr in list(tag.attrs.keys()):
            if attr.lower().startswith("on"):
                del tag[attr]
        # Validate URL attributes — only allow http/https
        for attr in _URL_ATTRS & set(tag.attrs.keys()):
            val = tag[attr]
            if isinstance(val, str) and not _SAFE_SCHEMES.match(val):
                del tag[attr]

    return str(soup)


def _html_to_pdf(markdown_text: str, filename: str, title: str, output_dir: Path) -> str:
    """Convert Markdown -> HTML -> PDF with CSS Styling."""
    css_style = """
    body { font-family: 'Helvetica', sans-serif; font-size: 12px; line-height: 1.4; color: #333; }
    h1 { font-size: 24px; border-bottom: 2px solid #333; margin-bottom: 10px; text-transform: uppercase; }
    h2 { font-size: 16px; border-bottom: 1px solid #ccc; margin-top: 20px; text-transform: uppercase; color: #555; }
    ul { padding-left: 20px; }
    li { margin-bottom: 5px; }
    strong { color: #000; }
    """
    clean_text = _strip_emoji_for_pdf(markdown_text)
    html_body = markdown.markdown(clean_text)
    html_body = _strip_emoji_from_html(html_body)
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
