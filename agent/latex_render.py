"""LaTeX rendering for the optimized resume.

Converts the LLM's markdown resume output into a structured representation,
fills a bundled LaTeX template (based on the popular Jake Gutierrez resume
template), and compiles the result into both PDF and DOCX.

Toolchain
---------
- PDF: pdflatex (MiKTeX / TeX Live / Overleaf) — must be on PATH.
- DOCX: pandoc — must be on PATH.

Both tools are auto-detected at runtime. If a tool is missing, its export
fails gracefully (returns None) rather than crashing the pipeline.

The markdown parser is intentionally tolerant: it extracts whatever structure
it can recognize and falls back to plain text groups so the output is never
empty, even for oddly-formatted LLM output.
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool availability detection
# ---------------------------------------------------------------------------

def _find_on_path(*names: str) -> Optional[str]:
    """Return the absolute path of the first executable found on PATH."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


PANDOC_PATH = _find_on_path("pandoc")
PDFLATEX_PATH = _find_on_path("pdflatex")
XELATEX_PATH = _find_on_path("xelatex")

HAVE_PANDOC = PANDOC_PATH is not None
HAVE_PDFLATEX = PDFLATEX_PATH is not None
# pdflatex provides unicode-sensitive glyphs; xelatex isn't required.
HAVE_TOOLS = HAVE_PDFLATEX and HAVE_PANDOC

if not HAVE_PDFLATEX:
    logger.warning("LaTeX renderer: pdflatex not found on PATH — PDF export via "
                   "LaTeX template is unavailable (falling back to HTML→PDF).")
if not HAVE_PANDOC:
    logger.warning("LaTeX renderer: pandoc not found on PATH — DOCX export is unavailable.")


# ---------------------------------------------------------------------------
# Structured resume model
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """Parsed heading block (name + line of contact details)."""
    name: str = ""
    details: str = ""


@dataclass
class Entry:
    """A single item under a section.

    For a subheading (e.g. a role/project/education row) the ``position`` and
    ``location``/companion text live in the ``header`` fields; bullet items are
    collected in ``items``. This maps cleanly onto the template macros.
    """
    header: str = ""          # bold left text (title, company, school)
    right: str = ""           # right-aligned column (dates / second field)
    sub: str = ""             # italic left text (role, degree)
    sub_right: str = ""       # italic right text (location / second date field)
    items: list[str] = field(default_factory=list)


@dataclass
class Section:
    title: str
    entries: list[Entry] = field(default_factory=list)
    # For simple one-line sections (e.g. Technical Skills) that are not lists
    # of subheadings, treat the whole body as plain lines under the section.
    simple_lines: list[str] = field(default_factory=list)


@dataclass
class ResumeData:
    contact: Contact = field(default_factory=Contact)
    sections: list[Section] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Markdown → structured data
# ---------------------------------------------------------------------------

# Detect markdown heading levels, bold-only headings, and "Title:" pseudo-headings
# Note: a single '#' (H1) at the top of the resume is conventionally the name,
# handled separately by _extract_contact. Section headings are H2+.
_HEADING_RE = re.compile(r'^\s{0,3}#{2,6}\s+(.*?)\s*$')
_BOLD_HEADING_RE = re.compile(r'^\s{0,3}\*\*(.+?)\*\*\s*$')
_SECTION_LABEL_LINE_RE = re.compile(r'^\s{0,3}[A-Za-z][A-Za-z &/-]{2,40}:\s*(.*)$')
_BULLET_RE = re.compile(r'^\s*[-*+•]\s+(.*)$')

# Markdown horizontal rule / decorative divider (e.g. `---`, `***`, `___`).
# Never content — must be skipped, not parsed as a subheading.
_HR_RE = re.compile(r'^\s*([-*_])\s*(?:\1\s*){2,}\s*$')

# Regexes reused for contact / subheading parsing
# Matches a trailing date range like "June 2020 -- Present", "2020-2021",
# "Aug 2018 -- May 2021". Month is optional. End-anchored and searched.
_DATE_RANGE_RE = re.compile(
    r'((?:[A-Za-z]{3,9}\.?\s+)?\d{4}|\d{4})\s*[-–—]{1,3}\s*(.+?)\s*$',
    re.IGNORECASE,
)

# A bold label line like "**Languages**: Python" or "**Tools**: Git, Docker"
# — typically a leaf line under a Technical Skills (or similar) section.
_BOLD_LABEL_RE = re.compile(r'^\s{0,3}\*\*(.+?)\*\*\s*:\s*(.*)$')


def _is_plain_label_line(s: str) -> bool:
    """True for a non-bold 'label: value' leaf line (e.g. a skill row like
    'Machine Learning & Deep Learning: PyTorch, ...'). These read better as
    simple lines than being split into subheading columns by commas/colons."""
    if not s or '|' in s or '**' in s:
        return False
    if _DATE_RANGE_RE.search(s):
        return False
    colon = s.find(':')
    if colon < 2 or colon > 50:
        return False
    label = s[:colon]
    value = s[colon + 1:].strip()
    # label is a short phrase, value is non-trivial content
    if not value or len(value) < 4:
        return False
    if len(label.split()) > 6:
        return False
    return True


_STANDARD_SECTION_WORDS = {
    "summary", "objective", "profile", "about", "education", "experience",
    "employment", "history", "projects", "skills", "competencies",
    "technologies", "certifications", "certificates", "awards", "honors",
    "publications", "volunteer", "volunteering", "languages", "references",
    "interests", "affiliations", "coursework", "activities", "leadership"
}


def _is_known_section_title(title: str) -> bool:
    """True if title contains a recognizable resume section keyword."""
    t = re.sub(r'[^a-zA-Z\s]', '', title or "").lower()
    words = t.split()
    return any(w in _STANDARD_SECTION_WORDS for w in words)


def _looks_like_paragraph(s: str) -> bool:
    """True if the line is a prose paragraph rather than a structured entry.

    Entries (roles, projects, education) carry structure — a date range, a '|'
    column split, or bold emphasis. A Summary/About paragraph is a single long
    run of prose with none of those, so it must render as a simple line, not be
    split into subheading columns by commas.
    """
    if not s:
        return False
    if '|' in s:
        return False
    if '**' in s:               # bold markers → likely an entry heading/label
        return False
    if _DATE_RANGE_RE.search(s):
        return False            # trailing dates → a dated entry
    plain = _strip_inline(s)
    # A paragraph is prose: either long (>= 75 chars) or ends with period,
    # and has multiple words without short label:value format.
    if len(plain) >= 75 and len(plain.split()) >= 8:
        if ':' in plain and plain.index(':') < 35:
            return False        # likely a skill row "Label: val"
        return True
    if plain.endswith('.') and len(plain.split()) >= 7:
        return True
    return False


def _is_section_heading(line: str) -> Optional[str]:
    """Return the normalized section title if line looks like a heading, else None."""
    m = _HEADING_RE.match(line)
    if m:
        return m.group(1).strip()
    m = _BOLD_HEADING_RE.match(line)
    if m:
        return m.group(1).strip()
    # "Experience:", "Technical Skills:", "Education:" etc.
    m = _SECTION_LABEL_LINE_RE.match(line)
    if m and len(m.group(1).strip()) <= 45:
        return m.group(1).strip().rstrip(':').strip()
    return None


def _is_prose_section(title: str) -> bool:
    """True for sections that contain free-form prose or simple item lists rather
    than structured entries (Summary, Objective, Profile, About, Certifications, etc.).
    Their body must always be rendered as simple lines, never split into subheading columns."""
    t = (title or "").strip().lower().rstrip(':')
    return any(k in t for k in ("summary", "objective", "profile", "about",
                                "professional summary", "intro", "certifications",
                                "certificates", "awards", "honors"))


def _strip_inline(s: str) -> str:
    """Remove markdown bold/italic/code markers from a string, preserving markdown links [text](url)."""
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', s)
    s = re.sub(r'`(.+?)`', r'\1', s)
    return s.strip()


def _extract_contact(lines: list[str]) -> tuple[Contact, list[str]]:
    """Pull the name + contact line(s) from the head of the resume if present.

    Accepts:
      - '# Name' (H1), '## Name' (H2 if not a known section title), or '**Name**'
      - A '## Contact' or '## Personal Info' heading
      - A plain name line followed by contact details (email/phone/links)

    Returns the Contact and remaining lines (heading block removed).
    """
    name = ""
    idx = 0
    header_block = []

    # Skip leading blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx >= len(lines):
        return Contact(), lines

    first_line = lines[idx].strip()
    sec_title = _is_section_heading(first_line)

    if sec_title:
        st_lower = sec_title.lower().rstrip(':')
        if st_lower in ("contact", "contact information", "contact details",
                        "personal info", "personal information", "personal details"):
            # A '## Contact' heading: consume the heading, then collect contact lines
            idx += 1
        elif not _is_known_section_title(sec_title) and len(sec_title.split()) <= 4:
            # Candidate name in H2 or bold (e.g. '## Omar Mashaly')
            name = _strip_inline(sec_title)
            idx += 1
        else:
            # It's a real section (e.g. '## Summary', '## Education') - no contact header
            return Contact(), lines[idx:]
    else:
        # Check if first line is '# Name'
        if first_line.startswith('#'):
            name_cand = re.sub(r'^#+\s*', '', first_line).strip()
            if len(name_cand) <= 50 and ':' not in name_cand:
                name = _strip_inline(name_cand)
                idx += 1

    # Now collect remaining header lines until the first real section heading
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if _BULLET_RE.match(line) or _HR_RE.match(line):
            break
        cand_title = _is_section_heading(line)
        if cand_title and (_is_known_section_title(cand_title) or len(cand_title.split()) > 4):
            break
        # If we don't have a name yet, check if this line is a plain name
        if not name and len(line) <= 40 and not any(c in line for c in ('@', '|', '•', '://', ':')):
            name = _strip_inline(line)
            idx += 1
            continue
        header_block.append(line)
        idx += 1

    details = ' | '.join(_strip_inline(d) for d in header_block if d)
    return Contact(name=name, details=details), lines[idx:]


def _clean_field(s: str) -> str:
    """Trim a parsed header/date field of wrappers and dangling punctuation.

    Handles the common LLM pattern where a date range is wrapped in parens that
    leak into the neighbouring field, e.g. "**Title** (" ... "*Sep 2025 - Jun
    2026)*". Safely preserves legitimate balanced parentheses like "(Hybrid RAG)"
    or "(E-JUST)". Returns the cleaned field, or "" if it becomes empty.
    """
    if not s:
        return ""
    s = _strip_inline(s.strip())

    # Strip dangling opening parens/brackets at the end of the line (e.g. "Title (")
    s = re.sub(r'[\(\[\{]\s*$', '', s).strip()

    # If the ENTIRE field is wrapped in matching parens (e.g. "(Sep 2025 - Jun 2026)"), strip both
    if (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']')):
        s = s[1:-1].strip()

    # If there is a trailing dangling paren without an opener (e.g. "Sep 2025 - Jun 2026)"):
    if s.count('(') < s.count(')') and s.endswith(')'):
        s = s[:-1].strip()
    if s.count('[') < s.count(']') and s.endswith(']'):
        s = s[:-1].strip()

    # If there is a leading dangling paren without a closer (e.g. "(Sep 2025 - Jun 2026"):
    if s.count('(') > s.count(')') and s.startswith('('):
        s = s[1:].strip()
    if s.count('[') > s.count(']') and s.startswith('['):
        s = s[1:].strip()

    s = s.strip(' -–—•')
    # A lone divider like just "---" / "-" is never real content.
    if not s or _HR_RE.match(s):
        return ""
    return s


def _should_split_on_comma(s: str) -> bool:
    """True only if 's' is a short 'Role, Company' or 'City, State' pair,
    and NOT a prose sentence that happens to contain commas."""
    if not s or ',' not in s:
        return False
    if len(s) > 65:
        return False
    if s.endswith('.'):
        return False
    parts = [p.strip() for p in s.split(',')]
    if len(parts) > 3 or len(parts) < 2:
        return False
    p0 = parts[0]
    p1 = ', '.join(parts[1:])
    if len(p0) > 35 or len(p1) > 40:
        return False
    lower0 = p0.lower()
    sentence_starters = (
        'across', 'during', 'in', 'with', 'experienced', 'proven', 'passionate',
        'strong', 'skilled', 'led', 'built', 'developed', 'engineered', 'managed',
        'implemented', 'responsible', 'including', 'designed', 'achieved', 'created',
        'focused', 'demonstrated'
    )
    if any(lower0.startswith(w + ' ') or lower0 == w for w in sentence_starters):
        return False
    return True


def _split_header_and_dates(heading_line: str) -> tuple[str, str, str, str]:
    """Best-effort split of a subheading line into left/right columns.

    Handles the common patterns:
      "Company, Location"  -> company + location
      "Role | Company"     -> sub + header
      "Role, Company"      -> sub + header
      "Title 2020 -- Present" -> title + dates

    Returns (left_bold, left_italic, right_bold, right_italic).
    """
    # Strip markdown emphasis from the WHOLE line first so bold/italic markers
    # that span across '|' (e.g. "**Role** | Company | *Date*") are removed
    # before we split the columns.
    line = _strip_inline(heading_line.strip())
    # Split trailing date range like "2020 -- Present" / "2020-2021"
    dates = ""
    m = _DATE_RANGE_RE.search(line)
    left = line
    if m:
        left = line[:m.start()].strip().rstrip('|').strip()
        dates = _clean_field(m.group(0).strip())

    # "Role | Company" or "Role | Company | Location" or "Degree | University | Location"
    if '|' in left:
        parts = [p.strip() for p in left.split('|') if p.strip()]
        if len(parts) >= 3:
            p0, p1, p2 = parts[0], parts[1], parts[2]
            inst_kw = ('university', 'college', 'school', 'institute', 'academy')
            deg_kw = ('b.sc', 'm.sc', 'ph.d', 'bachelor', 'master', 'doctor', 'associate',
                      'b.a.', 'b.s.', 'm.s.', 'degree', 'major', 'minor', 'computer science',
                      'engineering')
            if any(k in p1.lower() for k in inst_kw):
                # Degree | University | Location -> University on Row 1 (left), Degree on Row 2 (left), Location on Row 1 (right)
                return (_clean_field(p1), _clean_field(p0), _clean_field(p2), dates)
            elif any(k in p0.lower() for k in inst_kw):
                # University is p0 (Institution on Row 1 left)
                # Check whether p1 or p2 is the degree vs location:
                if any(k in p2.lower() for k in deg_kw) or not any(k in p1.lower() for k in deg_kw):
                    # University | Location | Degree (e.g. E-JUST | Alexandria | B.Sc. in CS)
                    # Institution on Row 1 (left), Degree on Row 2 (left), Location on Row 1 (right)
                    return (_clean_field(p0), _clean_field(p2), _clean_field(p1), dates)
                else:
                    # University | Degree | Location (e.g. Southwestern | B.S. in CS | Georgetown, TX)
                    # Institution on Row 1 (left), Degree on Row 2 (left), Location on Row 1 (right)
                    return (_clean_field(p0), _clean_field(p1), _clean_field(p2), dates)
            else:
                # Role | Company | Location
                return (_clean_field(p0), _clean_field(p1), _clean_field(p2), dates)
        elif len(parts) == 2:
            p0, p1 = parts[0], parts[1]
            inst_kw = ('university', 'college', 'school', 'institute', 'academy')
            if any(k in p1.lower() for k in inst_kw):
                # Degree | University -> University on line 1, Degree on line 2
                return (_clean_field(p1), _clean_field(p0), "", dates)
            return (_clean_field(p0), "", _clean_field(p1), dates)
        left = _clean_field(parts[0])
        return (left, "", "", dates)

    # "Role, Company" (comma pattern) — first is role, last is org (only if short, not prose)
    if _should_split_on_comma(left):
        parts = [p.strip() for p in left.split(',')]
        if len(parts) >= 2:
            return (_clean_field(parts[0]), "", _clean_field(', '.join(parts[1:])), dates)

    return (_clean_field(left), "", "", dates)


def _parse_entry_block(lines: list[str]) -> tuple[Optional[Entry], int]:
    """Parse one subheading + its bullets from position 0 of ``lines``.

    Returns (entry, consumed). ``consumed`` is the number of leading lines
    consumed from ``lines``.

    Handles two common layouts:
      Role | Company
      Location
      2020 -- Present
      - bullets

    or a single combined line. Fields map to the template macro arguments:
    header / right (bold columns) and sub / sub_right (italic columns).
    """
    header_lines = []
    idx = 0
    # Gather the subheading (non-bullet) lines
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            break
        if _BULLET_RE.match(line):
            break
        if _BOLD_LABEL_RE.match(line):
            break  # a leaf "**Label**: value" line — not a subheading
        if _is_plain_label_line(stripped):
            break  # a non-bold "label: value" leaf — not a subheading
        if stripped.startswith('#') and _is_section_heading(stripped):
            break
        if _HR_RE.match(stripped):
            # a decorative divider (---): skip it, don't treat as content
            idx += 1
            continue
        header_lines.append(_strip_inline(stripped))
        idx += 1

    if not header_lines:
        return None, 0

    # No more than 3 header lines are meaningful for the four-column macro
    header_lines = header_lines[:3]
    entry = Entry()

    # Last header line frequently holds a date range or location/role.
    # If the date is embedded at the end of the line (m.start() > 0), strip
    # only that date substring from the line. If the line is purely a date
    # (m.start() == 0), drop the whole line.
    last = header_lines[-1]
    last_dates = _DATE_RANGE_RE.search(last)
    if last_dates:
        entry.sub_right = _clean_field(last_dates.group(0))
        if last_dates.start() > 0:
            header_lines[-1] = last[:last_dates.start()].strip()
        else:
            header_lines = header_lines[:-1]

    if not header_lines:
        return None, idx

    # If two or more lines remain, split the first into (header, right) and
    # the next into (sub, sub_right).
    if len(header_lines) >= 2:
        first = header_lines[0]
        if '|' in first:
            p = [x.strip() for x in first.split('|')]
            entry.header = _clean_field(p[0])
            entry.right = _clean_field(p[1]) if len(p) > 1 else ""
        elif _should_split_on_comma(first):
            p = [x.strip() for x in first.split(',')]
            entry.header = _clean_field(p[0])
            entry.right = _clean_field(', '.join(p[1:]))
        else:
            entry.header = _clean_field(first)
            entry.right = ""
        second = header_lines[1]
        if '|' in second:
            p = [x.strip() for x in second.split('|')]
            entry.sub = _clean_field(p[0])
            entry.sub_right = _clean_field(p[1]) if len(p) > 1 else entry.sub_right
        elif _should_split_on_comma(second):
            p = [x.strip() for x in second.split(',')]
            entry.sub = _clean_field(p[0])
            entry.sub_right = _clean_field(', '.join(p[1:]))
        else:
            entry.sub = _clean_field(second)
            entry.sub_right = entry.sub_right
        # A possible third line is a location; keep it attached to sub_right
        if len(header_lines) >= 3 and not entry.sub_right:
            entry.sub_right = _clean_field(header_lines[2])
    else:
        # Single header line — split via the generic helper (already cleaned)
        h, s, r, sr = _split_header_and_dates(header_lines[0])
        entry.header = _clean_field(h)
        entry.right = _clean_field(r)
        entry.sub = _clean_field(s)
        # Keep any date already extracted (don't overwrite with empty)
        if not entry.sub_right:
            entry.sub_right = _clean_field(sr)

    # Collect bullets
    while idx < len(lines):
        line = lines[idx]
        m = _BULLET_RE.match(line)
        if m:
            entry.items.append(_strip_inline(m.group(1)))
            idx += 1
        elif not lines[idx].strip():
            # blank line: peek ahead — if next is not a bullet, stop
            j = idx + 1
            if j < len(lines) and _BULLET_RE.match(lines[j]):
                idx += 1
                continue
            break
        else:
            break

    return entry, idx


def parse_markdown_resume(markdown_text: str) -> ResumeData:
    """Parse the LLM's markdown resume into a structured ResumeData."""
    lines = [ln.rstrip() for ln in (markdown_text or "").splitlines()]

    contact, remaining = _extract_contact(lines)
    if not contact.name and not contact.details:
        # No explicit heading block; treat everything as body
        remaining = lines

    data = ResumeData(contact=contact)
    current_section: Optional[Section] = None

    i = 0
    while i < len(remaining):
        line = remaining[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        title = _is_section_heading(line)
        if title:
            current_section = Section(title=title)
            data.sections.append(current_section)
            i += 1
            continue

        if current_section is None:
            # Lines before any section — treat as stray detail (skip or keep)
            i += 1
            continue

        if _HR_RE.match(stripped):
            # decorative divider (---) — skip silently
            i += 1
            continue

        # Prose-only sections (Summary, Objective, Profile, About...) never
        # produce structured entries — every content line is a simple line.
        if _is_prose_section(current_section.title):
            if _BULLET_RE.match(line):
                content = _strip_inline(_BULLET_RE.match(line).group(1))
            else:
                content = _strip_inline(stripped)
            if content:
                current_section.simple_lines.append(content)
            i += 1
            continue

        if _BULLET_RE.match(line):
            # A bare bullet directly under a section that has no subheading,
            # treat as a simple line (e.g. dot-point technical skills)
            current_section.simple_lines.append(_strip_inline(_BULLET_RE.match(line).group(1)))
            i += 1
            continue

        if _BOLD_LABEL_RE.match(line):
            # "**Label**: value" leaf — store as a simple line, but remember the
            # label so the renderer can bold it separately.
            current_section.simple_lines.append(stripped)
            i += 1
            continue

        # A non-bold "label: value" leaf (e.g. skill row) → simple line.
        if _is_plain_label_line(stripped):
            current_section.simple_lines.append(_strip_inline(stripped))
            i += 1
            continue

        # A long prose paragraph (e.g. a Summary) must stay a simple line —
        # don't let the entry parser split it across subheading columns.
        if _looks_like_paragraph(stripped):
            current_section.simple_lines.append(_strip_inline(stripped))
            i += 1
            continue

        # Try to parse as a subheading entry (with optional bullets)
        entry, consumed = _parse_entry_block(remaining[i:])
        if entry is not None and current_section is not None:
            current_section.entries.append(entry)
            i += consumed
            continue

        # Fallback: treat as a simple line
        current_section.simple_lines.append(_strip_inline(stripped))
        i += 1

    return data


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = str.maketrans({
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
})

# Unicode → ASCII replacements. pdflatex (unlike xelatex) fails hard on most
# non-ASCII glyphs (e.g. U+202F narrow no-break space, •, en-dash), so we map
# them to ASCII before escaping.
_UNICODE_REPLACEMENTS = {
    '\u2010': '-', '\u2011': '-', '\u2012': '-',
    '\u2013': '-', '\u2014': '--',
    '\u2015': '--', '\u2212': '-',
    '\u2043': '-', '\uFE58': '-', '\uFE63': '-',
    '\uFF0D': '-',
    '\u2192': '->', '\u2190': '<-', '\u2022': '-',
    '\u2023': '>', '\u25B6': '>',
    '\u2794': '->', '\u279C': '->', '\u21D2': '=>', '\u2194': '<->',
    '\u2502': '|', '\u2500': '-',
    '\u25CF': '*', '\u25CB': 'o', '\u25AA': '*',
    '\u2026': '...',
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00AB': '"', '\u00BB': '"',
    '\u00bf': '->',   # '¿' — mojibake artifact of a mangled '→' arrow
    '\u00a1': '!',    # '¡' inverted bang (mojibake-ish)
    '\u00b1': '+/-',  # '±' plus-minus
    '\u00d7': 'x',    # '×' multiplication sign
    '\u00f7': '/',    # '÷' division sign
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u00fc': 'u',
    '\u202f': ' ', '\xa0': ' ', '\u2009': ' ', '\u200a': ' ', '\u2003': ' ',
    '\u200b': '', '\u200c': '', '\u200d': '',
    '\ufeff': '',
    '\ufe0f': '', '\ufe0e': '',
}


def _force_ascii(text: str) -> str:
    """Replace known Unicode glyphs with ASCII, then strip any residual
    non-ASCII so pdflatex never sees a character it can't typeset."""
    import unicodedata
    if not text:
        return ""
    for uni, rep in _UNICODE_REPLACEMENTS.items():
        text = text.replace(uni, rep)
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


def _latex_escape(text: str) -> str:
    """Strip non-ASCII and escape LaTeX special characters (except in URLs,
    which are handled by hyperref)."""
    if not text:
        return ""
    text = _force_ascii(text)
    # Defensive: drop any residual markdown emphasis markers that survived
    # parsing so they never appear raw in the LaTeX (and never break pdflatex).
    text = re.sub(r'\*', '', text)
    text = re.sub(r'__', '', text)
    return text.translate(_LATEX_SPECIAL)


def _clean_latex_url(url: str) -> str:
    """Escape characters in URLs that break LaTeX hyperref when nested in macro arguments."""
    if not url:
        return ""
    url = url.strip()
    # In LaTeX macro arguments (such as \resumeSubheading and \resumeProjectHeading),
    # the URL is parsed in a context where catcodes are frozen.
    # # must be \# (otherwise: "! Illegal parameter number in definition of \Hy@tempa")
    # % must be \% (otherwise: treated as LaTeX comment)
    # & must be \& (otherwise: treated as tabular cell separator)
    url = url.replace('\\#', '#').replace('\\%', '%').replace('\\&', '&')
    url = url.replace('#', r'\#')
    url = url.replace('%', r'\%')
    url = url.replace('&', r'\&')
    return url


def _latex_link(text: str) -> str:
    """Escape a URL-ish string for inclusion in href (keep http:// ... intact)."""
    if not text:
        return ""
    if re.match(r'^(https?://|mailto:|www\.)', text):
        url = text if not text.startswith('www.') else 'http://' + text
        return r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(text) + r'}}'
    return _latex_escape(text)


def _render_text_with_links(text: str) -> str:
    """Format inline text for LaTeX, converting markdown links [label](url)
    and raw URLs into clickable \\href{url}{\\underline{label}} while properly
    escaping special LaTeX characters and Unicode."""
    if not text:
        return ""
    text = _force_ascii(text)
    links = []

    def save_link(match):
        label = match.group(1).strip()
        raw_url = match.group(2).strip().lstrip('<').rstrip('>')
        raw_url = raw_url.split()[0].strip().rstrip(')')
        if not raw_url.startswith(('http://', 'https://', 'mailto:')):
            raw_url = ('mailto:' if '@' in raw_url else 'https://') + raw_url
        idx = len(links)
        links.append((label, raw_url))
        return f"__LINK_PLACEHOLDER_{idx}__"

    # Match markdown links: [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', save_link, text)

    # Strip residual markdown bold/italic/code markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)

    token_re = re.compile(r'(__LINK_PLACEHOLDER_\d+__)')
    parts = []
    for piece in token_re.split(text):
        if not piece:
            continue
        m_link = re.match(r'^__LINK_PLACEHOLDER_(\d+)__$', piece)
        if m_link:
            idx = int(m_link.group(1))
            label, url = links[idx]
            clean_url = _clean_latex_url(url)
            parts.append(r'\href{' + clean_url + r'}{\underline{' + _latex_escape(label) + r'}}')
        else:
            parts.append(_latex_escape(piece))
    return ''.join(parts)


def _render_contact(contact: Contact) -> str:
    """Render the centered name + contact line using the template's heading style."""
    lines = []
    if not contact.name and not contact.details:
        return ""
    name_part = contact.name or "Candidate"
    lines.append(r"\begin{center}")
    # Reduced from \Huge to \LARGE (or \Large if long) so name heading is proportional
    heading_font = r"\LARGE" if len(name_part) <= 30 else r"\Large"
    lines.append(r"\textbf{" + heading_font + r" \scshape " + _latex_escape(name_part) + r"} \\ \vspace{1pt}")
    if contact.details:
        # Split on common contact separators (|, •, ·, ;) plus " - "
        details = contact.details
        parts = [p.strip() for p in
                 re.split(r'\s*(?:[\|•·;]|\s+-\s+)\s*', details) if p.strip()]
        rendered = []
        for p in parts:
            # Check for markdown link [label](url)
            m_md = re.match(r'^\[([^\]]+)\]\(([^)]+)\)$', p)
            if m_md:
                lbl = m_md.group(1).strip()
                url = m_md.group(2).strip().lstrip('<').rstrip('>').split()[0].rstrip(')')
                if not url.startswith(('http://', 'https://', 'mailto:')):
                    url = ('mailto:' if '@' in url else 'https://') + url
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(lbl) + r'}}')
                continue

            # Check for email
            if '@' in p and '.' in p and ' ' not in p:
                clean_email = p.lstrip('<').rstrip('>').strip()
                rendered.append(r'\href{mailto:' + _clean_latex_url(clean_email) + r'}{\underline{' + _latex_escape(clean_email) + r'}}')
                continue

            # Check for LinkedIn profile
            if 'linkedin.com' in p.lower():
                url = p if p.startswith(('http://', 'https://')) else 'https://' + p
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(p) + r'}}')
                continue

            # Check for GitHub profile
            if 'github.com' in p.lower():
                url = p if p.startswith(('http://', 'https://')) else 'https://' + p
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(p) + r'}}')
                continue

            # Check for web URLs or portfolio domains
            if re.match(r'^(?:https?://|www\.)', p) or re.search(r'\.(?:io|dev|ai|app|me|tech|com|org)(?:/|$)', p):
                url = p if p.startswith(('http://', 'https://')) else 'https://' + p
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(p) + r'}}')
                continue

            # Otherwise plain text (e.g. location, phone number)
            rendered.append(_latex_escape(p))

        lines.append(r"\small " + r" $|$ ".join(rendered))
    lines.append(r"\end{center}")
    lines.append("")
    return "\n".join(lines)


def _render_section(section: Section) -> str:
    """Render one section into LaTeX using the template macros."""
    parts = []
    parts.append(r"\section{" + _latex_escape(section.title) + "}")

    title_lower = section.title.lower()
    is_project_sec = "project" in title_lower
    is_exp_sec = any(k in title_lower for k in ("experience", "employment", "work", "history", "career"))
    is_edu_sec = any(k in title_lower for k in ("education", "academic", "university", "college", "school"))

    if section.entries:
        populated = [e for e in section.entries if (e.header or e.right or e.sub or e.sub_right)]
        if populated:
            parts.append(r"\resumeSubHeadingListStart")
            for e in populated:
                headers_left = e.header or ""
                headers_right = e.right or ""
                sub_left = e.sub or ""
                sub_right = e.sub_right or ""

                if is_project_sec:
                    # Projects: title + links + tech stack + date
                    date_field = sub_right
                    title_text = _render_text_with_links(headers_left)
                    tech_stack = headers_right or sub_left
                    tech_text = _render_text_with_links(tech_stack)
                    total_len = len(headers_left) + len(tech_stack) + len(date_field)
                    if total_len <= 75 and tech_text:
                        left_text = r"\textbf{" + title_text + r"} $|$ \emph{" + tech_text + r"}"
                        parts.append(r"\resumeProjectHeading{" + left_text + r"}{" +
                                     _latex_escape(date_field) + r"}")
                    elif tech_text:
                        parts.append(r"\resumeSubheading{" + title_text + "}{" +
                                     _latex_escape(date_field) + "}{" +
                                     tech_text + "}{}")
                    else:
                        parts.append(r"\resumeProjectHeading{\textbf{" + title_text + r"}}{" +
                                     _latex_escape(date_field) + r"}")
                elif is_exp_sec:
                    # Experience: Role and Company side-by-side on the left, dates on the right
                    # (e.g. \resumeProjectHeading{\textbf{Role} $|$ \emph{Company}}{Dates})
                    role_text = _render_text_with_links(headers_left)
                    date_text = _latex_escape(sub_right or "")
                    company_raw = sub_left or headers_right or ""
                    location_raw = headers_right if sub_left else ""

                    # Detect if headers_left is Company and sub_left is Role:
                    role_kw = (
                        'engineer', 'developer', 'trainee', 'intern', 'specialist', 'manager',
                        'lead', 'analyst', 'scientist', 'consultant', 'architect', 'officer',
                        'assistant', 'director', 'associate', 'fellow', 'head', 'coordinator',
                        'researcher', 'instructor', 'designer', 'administrator', 'technician'
                    )
                    if sub_left and any(k in sub_left.lower() for k in role_kw) and not any(k in headers_left.lower() for k in role_kw):
                        role_text = _render_text_with_links(sub_left)
                        company_raw = headers_left
                        location_raw = headers_right

                    company_text = _render_text_with_links(company_raw)
                    location_text = _latex_escape(location_raw)

                    if company_text:
                        left_heading = r"\textbf{" + role_text + r"} $|$ \emph{" + company_text + r"}"
                        if location_text:
                            left_heading += r" (" + location_text + r")"
                        parts.append(r"\resumeProjectHeading{" + left_heading + r"}{" + date_text + r"}")
                    else:
                        parts.append(r"\resumeProjectHeading{\textbf{" + role_text + r"}}{" + date_text + r"}")
                elif is_edu_sec:
                    # Education: University & Location on row 1, Degree & Dates on row 2
                    inst_text = _render_text_with_links(headers_left)
                    loc_text = _render_text_with_links(headers_right)
                    deg_text = _render_text_with_links(sub_left)
                    date_text = _render_text_with_links(sub_right)
                    parts.append(r"\resumeSubheading{" + inst_text + "}{" +
                                 loc_text + "}{" + deg_text + "}{" + date_text + "}")
                else:
                    # Generic entries (Awards, Leadership, Publications, Certifications, etc.)
                    h_text = _render_text_with_links(headers_left)
                    r_text = _latex_escape(headers_right)
                    s_text = _render_text_with_links(sub_left)
                    sr_text = _latex_escape(sub_right)
                    if not s_text and not sr_text:
                        parts.append(r"\resumeProjectHeading{\textbf{" + h_text + r"}}{" + r_text + r"}")
                    else:
                        parts.append(r"\resumeSubheading{" + h_text + "}{" +
                                     r_text + "}{" + s_text + "}{" + sr_text + "}")

                if e.items:
                    parts.append(r"\resumeItemListStart")
                    for item in e.items:
                        parts.append(r"\resumeItem{" + _render_text_with_links(item) + "}")
                    parts.append(r"\resumeItemListEnd")
            parts.append(r"\resumeSubHeadingListEnd")
            parts.append("")

    if section.simple_lines:
        items = []
        for ln in section.simple_lines:
            m = _BOLD_LABEL_RE.match(ln)
            if m:
                # **Label**: value → bold label, formatted value
                label = _strip_inline(m.group(1))
                value = m.group(2)
                item_text = (r"\textbf{" + _latex_escape(label) + r"}{: " +
                             _render_text_with_links(value) + r"}")
            else:
                # Generic label:value → bold label
                cm = re.match(r'^(.*?):\s*(.*)$', ln)
                if cm and len(cm.group(1)) <= 35:
                    item_text = (r"\textbf{" + _latex_escape(cm.group(1)) + r"}{: " +
                                 _render_text_with_links(cm.group(2)) + r"}")
                else:
                    item_text = _render_text_with_links(ln)
            items.append(item_text)

        # Render simple line rows inside a single item joined by \\ (Jake Gutierrez style)
        if len(items) > 1:
            joined = " \\\\\n     ".join(items)
            parts.append(r"\begin{itemize}[leftmargin=0.15in, label={}]")
            parts.append(r"    \small{\item{")
            parts.append(f"     {joined}")
            parts.append(r"    }}")
            parts.append(r"\end{itemize}")
            parts.append("")
        elif items:
            parts.append(r"\begin{itemize}[leftmargin=0.15in, label={}]")
            parts.append(r"    \small{\item{" + items[0] + r"}}")
            parts.append(r"\end{itemize}")
            parts.append("")

    # A section that ended up with no usable content (no non-empty entries and
    # no simple lines) would render an empty \section that pdflatex rejects —
    # drop the whole section.
    if not section.simple_lines and not any(
            e.header or e.right or e.sub or e.sub_right for e in section.entries):
        return ""

    return "\n".join(parts)



def render_resume_tex(data: ResumeData) -> str:
    """Fill the bundled LaTeX template with the structured resume data."""
    template_path = Path(__file__).resolve().parent / "templates" / "resume_template.tex"
    template = template_path.read_text(encoding="utf-8")

    body_parts = []
    contact_tex = _render_contact(data.contact)
    if contact_tex:
        body_parts.append(contact_tex)
    for section in data.sections:
        body_parts.append(_render_section(section))

    body = "\n".join(body_parts) if body_parts else _latex_escape(
        "No structured content could be extracted — please provide a resume.")

    # Replace the placeholder comment with the rendered body
    template = template.replace("<!--RESUME_BODY-->", body)
    return template


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> tuple[int, str]:
    """Run a subprocess, returning (returncode, combined stdout+stderr)."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output
    except FileNotFoundError:
        return -1, "Executable not found"
    except subprocess.TimeoutExpired:
        return -2, "Compilation timed out"


def latex_to_pdf(tex_content: str, output_dir: Path, filename: str = "Resume.pdf",
                 timeout: int = 300) -> Optional[str]:
    """Compile LaTeX content into a PDF using pdflatex (run in a temp dir).

    Returns the path to the produced PDF, or None on failure.
    """
    import tempfile
    if not HAVE_PDFLATEX:
        logger.warning("LaTeX PDF export skipped: pdflatex not available.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="resume_latex_") as tmp:
        tmp_dir = Path(tmp)
        teX_file = tmp_dir / "resume.tex"
        # pdflatex expects a specific encoding of some glyphs — write utf8
        teX_file.write_text(tex_content, encoding="utf-8")
        rc, out = _run([PDFLATEX_PATH, "-interaction=nonstopmode", "-halt-on-error",
                        teX_file.name], cwd=tmp_dir, timeout=timeout)
        if rc != 0:
            # pdflatex writes the log even on error — log the tail for help
            logger.error("pdflatex failed (rc=%s):\n%s", rc, out[-2000:])
            return None

        produced = tmp_dir / "resume.pdf"
        if not produced.exists():
            logger.error("pdflatex reported success but no resume.pdf produced.\n%s", out[-1000:])
            return None

        dest = output_dir / filename
        import shutil as _sh
        _sh.copyfile(produced, dest)
        return str(dest)


def latex_to_docx(tex_content: str, output_dir: Path, filename: str = "Resume.docx",
                  timeout: int = 120) -> Optional[str]:
    """Convert LaTeX content into a DOCX using pandoc.

    Returns the path to the produced DOCX, or None on failure.
    """
    if not HAVE_PANDOC:
        logger.warning("DOCX export skipped: pandoc not available.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="resume_latex_") as tmp:
        tmp_dir = Path(tmp)
        teX_file = tmp_dir / "resume.tex"
        teX_file.write_text(tex_content, encoding="utf-8")
        out_docx = tmp_dir / "resume.docx"
        rc, out = _run([PANDOC_PATH, str(teX_file), "-o", str(out_docx)],
                       cwd=tmp_dir, timeout=timeout)
        if rc != 0 or not out_docx.exists():
            logger.error("pandoc failed (rc=%s):\n%s", rc, out[-1000:])
            return None
        dest = output_dir / filename
        import shutil as _sh
        _sh.copyfile(out_docx, dest)
        return str(dest)
