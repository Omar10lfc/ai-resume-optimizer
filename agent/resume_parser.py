"""Markdown resume AST and parser shared across renderers and audit modules."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Structured resume model
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """Parsed heading block (name + line of contact details)."""
    name: str = ""
    title: str = ""
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
    links: list[str] = field(default_factory=list)


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
_BOLD_HEADING_RE = re.compile(r'^\s{0,3}\*\*([^*\n]+?)\*\*\s*$')
_SECTION_LABEL_LINE_RE = re.compile(r'^\s{0,3}([A-Za-z][A-Za-z &/-]{2,40}):\s*$')
_BULLET_RE = re.compile(r'^\s*[-*+•]\s+(.*)$')

# Markdown horizontal rule / decorative divider (e.g. `---`, `***`, `___`).
# Never content — must be skipped, not parsed as a subheading.
_HR_RE = re.compile(r'^\s*([-*_])\s*(?:\1\s*){2,}\s*$')

# Regexes reused for contact / subheading parsing
# Matches a trailing date range like "June 2020 -- Present", "2020-2021",
# "Aug 2018 -- May 2021". Month is optional. End-anchored and searched.
# The separator also accepts the worded form ("Oct 2022 to Jun 2026" -
# pypdf-7 extraction keeps it, run d946bb5f's Education row shipped
# dateless): guarded by a lookahead so prose like "grew users from 10K in
# 2021 to 50K" (no 4-digit year / Present after the word) never matches.
# Every typographic dash an author or renderer can put between dates. The
# ASCII-date gate flags all of them, so parsing must recognize all of them.
_NON_ASCII_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DATE_DASH = "[-" + _NON_ASCII_DASHES + "]"

_DATE_RANGE_RE = re.compile(
    r'((?:[A-Za-z]{3,9}\.?\s+)?\d{4}|\d{4})'
    r'(?:\s*' + _DATE_DASH + r'{1,3}\s*|\s+(?:to|through|till|until)\s+'
    r'(?=(?:[A-Za-z]{3,9}\.?\s*,?\s*)?\d{4}|\bPresent\b))'
    r'(.+?)\s*$',
    re.IGNORECASE,
)

# Month-to-month ranges without a leading year ("Aug – Sep 2026",
# "May-Aug 2026"): _DATE_RANGE_RE needs a year BEFORE the dash, so these
# were silently dropped from 4-field project headers (live run 868e406d:
# "… | Aug – Sep 2026" lost its date entirely). End-anchored to avoid
# matching month words mid-stack ("May" inside a tool list).
_MONTH_RANGE_RE = re.compile(
    r'(?:[A-Za-z]{3,9}\.?\s*)?' + _DATE_DASH + r'?\s*[A-Za-z]{3,9}\.?\s*'
    r'(?:' + _DATE_DASH + r'{1,3}|\s+(?:to|through)\s+)'
    r'\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4}\s*$', re.IGNORECASE)
# A field whose ENTIRE content is a standalone date-ish token
# ("Aug 2026 - Present", "2020-2022", "Aug – Sep 2026"). Only whole-segment
# matches pop a column off, never inline mentions inside longer text.
_STANDALONE_DATE_RE = re.compile(
    r'^\s*(?:'
    r'(?:[A-Za-z]{3,9}\.?\s+)?\d{4}\s*' + _DATE_DASH + r'{1,3}\s*(?:.+?)?\s*\d{0,4}\s*'
    r'|[A-Za-z]{3,9}\.?\s+\d{4}\s*$'
    r'|[A-Za-z]{3,9}\.?\s*' + _DATE_DASH + r'{1,3}\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4}\s*$'
    r'|(?:[A-Za-z]{3,9}\.?\s+)?\d{4}\s+(?:to|through|till|until)\s+'
    r'(?:[A-Za-z]{3,9}\.?\s*,?\s*)?(?:\d{4}|Present)\s*$'
    r'|[A-Za-z]{3,9}\.?\s*(?:' + _DATE_DASH + r'{1,3}|\s+(?:to|through)\s+)\s*'
    r'[A-Za-z]{3,9}\.?\s*,?\s*\d{4}\s*$'
    r')\s*$', re.IGNORECASE)

# Date ranges ship ASCII-only. LaTeX turns `--` into an en dash, and ATS
# imports (Workday) drop a date whose separator is non-ASCII along with the
# column; `_force_ascii` maps an em dash to `--`, so normalizing the parsed
# date field is the root-cause repair for BOTH renderers and every engine.
_ANY_DASH_RE = re.compile(f"[{_NON_ASCII_DASHES}]")


def _normalize_date_ascii(value: str) -> str:
    """Render a date/date-range with ASCII hyphens only.

    Keeps the author's spacing ("Sep 2025 - Jun 2026" stays spaced) and
    collapses repeated separators ("2016--2024" -> "2016-2024") so LaTeX
    cannot turn them into an en dash. Applied only to fields already known
    to be dates, never to titles or locations.
    """
    if not value:
        return value
    value = _ANY_DASH_RE.sub("-", value)
    return re.sub(r"-{2,}", "-", value)

# A bold label line like "**Languages**: Python" or "**Tools**: Git, Docker"
# — typically a leaf line under a Technical Skills (or similar) section.
_BOLD_LABEL_RE = re.compile(r'^\s{0,3}\*\*(.+?)\*\*\s*:\s*(.*)$')

# First alphabetic character lowercase = the line starts mid-sentence.
# Bracket-led lines ('[Live Demo](...)' headers) are excluded - they are
# header shapes, never bullet fragments.
_ORPHAN_CONTINUATION_RE = re.compile(r"^[^A-Za-z\[\(]*[a-z]")


def _is_orphan_continuation(stripped: str, min_len: int = 1) -> bool:
    """True when a non-bullet line reads as a mid-sentence fragment a writer
    pass emitted without its bullet marker (run bf0f7941, Nov 12 2026:
    'doubled summarization ROUGE-L (13.48 -> 29.56) with AraBART.' right
    under an entry header - it used to be gathered as a header field,
    take the never-rendered sub slot AND drop the entry's date column).

    Guards: no '|' (a wrapped header tail carries pipes and stays a header
    field), and - for header gathering (min_len=25) - long enough that a
    short lowercase Title-ish sub-line ('iOS Developer') is never diverted.
    """
    if len(stripped) < min_len or "|" in stripped:
        return False
    return bool(_ORPHAN_CONTINUATION_RE.match(stripped))


def _is_plain_label_line(s: str) -> bool:
    """True for a non-bold 'label: value' leaf line (e.g. a skill row like
    'Machine Learning & Deep Learning: PyTorch, ...'). These read better as
    simple lines than being split into subheading columns by commas/colons."""
    if not s or '|' in s or '**' in s or s.lstrip().startswith('['):
        return False
    if _DATE_RANGE_RE.search(s):
        return False
    colon = s.find(':')
    if colon < 2 or colon > 50:
        return False
    # Never mistake a URL scheme ('https://...', 'mailto:...') for a skill label.
    if s[colon:colon + 3] == '://' or s[:colon].lower().endswith('mailto'):
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
    # A bold-label leaf ("**Label**: value") or plain label ("Label: value") is
    # a list item (e.g. under Technical Skills), never a section heading.
    if _BOLD_LABEL_RE.match(line) or _is_plain_label_line(line):
        return None

    m = _HEADING_RE.match(line)
    if m:
        return m.group(1).strip()

    m = _BOLD_HEADING_RE.match(line)
    if m:
        cand = m.group(1).strip()
        # A bold heading must be a concise title (not a paragraph or label:value line)
        if len(cand) <= 45 and ':' not in cand and len(cand.split()) <= 6:
            return cand
        if len(cand) <= 45 and cand.endswith(':') and len(cand.split()) <= 6:
            return cand.rstrip(':').strip()

    # "Experience:", "Technical Skills:", "Education:" on a line by itself
    m = _SECTION_LABEL_LINE_RE.match(line)
    if m:
        label = m.group(1).strip()
        if _is_known_section_title(label):
            return label

    # Bare ALL-CAPS headings from raw PDF extraction ("SUMMARY", "SKILLS",
    # "PROJECTS", "EXPERIENCE", "CERTIFICATIONS" - pypdf emits section
    # titles without a colon; run d946bb5f's source parsed to ZERO sections
    # and the contact collector swallowed 21 of 57 lines). Gated on known
    # section titles so random uppercase lines ("NTI", "AI & ML") never
    # become sections. Markdown-native shapes above keep priority.
    stripped = line.strip()
    if stripped.isupper() and 2 <= len(stripped) <= 40 \
            and _is_known_section_title(stripped):
        words = stripped.split()
        return " ".join(
            w if (len(w) <= 3 and w.isalpha()) else w.title() for w in words)

    return None


def _is_prose_section(title: str) -> bool:
    """True for sections that contain free-form prose or simple item lists rather
    than structured entries (Summary, Objective, Profile, About, Certifications, etc.).
    Their body must always be rendered as simple lines, never split into subheading columns."""
    t = (title or "").strip().lower().rstrip(':')
    return any(k in t for k in ("summary", "objective", "profile", "about",
                                "professional summary", "intro", "certifications",
                                "certificates", "awards", "honors"))


_SKILLS_SECTION_RE = re.compile(
    r'(?i)\b(skill|skills|competenc|technolog|tech\s?stack|stack)\w*')


def _is_skills_section(title: str) -> bool:
    """True for flat list-style sections (Skills, Competencies, Tech Stack).
    Their plain lines are comma/pipe lists, never role+org subheadings."""
    return bool(_SKILLS_SECTION_RE.search(title or ""))


def _strip_inline(s: str) -> str:
    """Remove markdown bold/italic/code markers from a string, preserving markdown links [text](url)."""
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', s)
    s = re.sub(r'`(.+?)`', r'\1', s)
    return s.strip()


def _is_link_item(text: str) -> bool:
    """Detect if an item line is purely project links (Live Demo, GitHub, etc.)
    rather than an achievement bullet."""
    if not text:
        return False
    t = text.strip()
    # Pure markdown links and separators: [Live Demo](...) | [GitHub](...)
    if re.match(r'^(?:\[[^\]]+\]\([^)]+\)|\s*[\|\-–—/]\s*)+$', t):
        return True
    # Strip markdown links to inspect residual text
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    clean_lower = re.sub(r'\s+', ' ', clean.strip().lower())
    # Known standalone link labels/phrases
    keywords = {
        "live demo", "github", "demo", "repo", "source code", "code",
        "live demo | github", "live demo - github", "live demo -- github", "live demo — github",
        "github | live demo", "github - live demo", "github -- live demo", "github — live demo",
        "demo | github", "github | demo", "view demo", "view code", "project link"
    }
    if clean_lower in keywords:
        return True
    if re.match(r'^(?:live\s+demo|github|demo|repo|code|project\s+link|link)\s*:\s*\S+', t, re.I):
        return True
    if re.match(r'^https?://\S+$', t):
        return True
    return False


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
    under_contact_heading = False

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
            under_contact_heading = True
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
    title = ""
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if _BULLET_RE.match(line) or _HR_RE.match(line):
            break
        cand_title = _is_section_heading(line)
        if cand_title:
            # Any further markdown heading ends the header block. Known
            # sections obviously, but custom short headings too: the old
            # check (known OR >4 words) absorbed custom brief sections
            # like '## Tech Stack' into the contact details line, which
            # then deleted the entire section from the document.
            st_lower = cand_title.lower().rstrip(':')
            if st_lower not in ("contact", "contact information", "contact details",
                                "personal info", "personal information",
                                "personal details"):
                break
        # If we don't have a name yet, check if this line is a plain name
        if not name and len(line) <= 40 and not any(c in line for c in ('@', '|', '•', '://', ':')):
            name = _strip_inline(line)
            idx += 1
            continue
        # The professional title rides the FIRST header line of the
        # conventional resume shape ("Omar Mashaly" / "AI/ML Engineer" /
        # "Alexandria, Egypt | ..."). It renders on its own line under the
        # name (Nov 16 2026, user ask) instead of glued into the contact
        # row - recognized when the line is separator-free, so a first
        # line that is already contact info keeps the old single-row
        # layout.
        if (not title and not header_block and not under_contact_heading
                and len(line) <= 60
                and not any(c in line for c in ('@', '|', '•', '://', ';'))
                and len(line.split()) <= 6):
            title = _strip_inline(line)
            idx += 1
            continue
        header_block.append(line)
        idx += 1

    details = ' | '.join(_strip_inline(d) for d in header_block if d)
    return Contact(name=name, title=title, details=details), lines[idx:]


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


def _date_match_to_field(m: "re.Match") -> tuple[str, str]:
    """Splits a date-range match into (clean date, post-date field tail).

    _DATE_RANGE_RE's tail is ``(.+?)\\s*$`` - it swallows everything up to the
    end of the line. On 'Title | Feb 2026 - Jun 2026 | BM25, ...' that glued
    the tech stack INTO the date column (live run 28e2e56f, Nov 12 2026:
    three project entries printed the stack right-aligned inside the date
    argument, running 123pt past the physical page edge at x1=735 on a
    612pt sheet). A pipe inside the match is a field boundary, never date
    content, so the tail is handed back for the pipe-split logic to route.
    """
    matched = m.group(0)
    if "|" in matched:
        date, _, tail = matched.partition("|")
        return _normalize_date_ascii(_clean_field(date)), tail.strip()
    return _normalize_date_ascii(_clean_field(matched)), ""


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
    # Split trailing date range like "2020 -- Present" / "2020-2021", or a
    # month-to-month range with no leading year ("Aug – Sep 2026").
    dates = ""
    m = _DATE_RANGE_RE.search(line)
    if not m:
        m = _MONTH_RANGE_RE.search(line)
    left = line
    if m:
        left = line[:m.start()].strip().rstrip('|').strip()
        dates, tail = _date_match_to_field(m)
        if tail:
            # "Title | <date> | more fields" - the date is a mid-line column,
            # not a trailing annotation. Rejoin the post-date fields so the
            # pipe-split below routes every one of them instead of the tail
            # being swallowed into (or dropped after) the date.
            left = f"{left} | {tail}" if left else tail

    # "Role | Company" or "Role | Company | Location" or "Degree | University | Location"
    if '|' in left:
        parts = [p.strip() for p in left.split('|') if p.strip()]

        def _pop_standalone_date(segments: list[str]) -> tuple[list[str], str]:
            """When no inline date was captured, a WHOLE trailing segment that
            is itself just a date ("Aug – Sep 2026") is the date column, not
            a fourth field. 4-field project/role headers otherwise dropped
            their date (live run 868e406d)."""
            nonlocal dates
            if dates:
                return segments, dates
            for idx in (len(segments) - 1, len(segments) - 2):
                if 0 <= idx < len(segments):
                    seg = _clean_field(segments[idx])
                    if seg and _STANDALONE_DATE_RE.match(seg):
                        tail = segments.copy()
                        del tail[idx]
                        return tail, _normalize_date_ascii(seg)
            return segments, dates

        parts, dates = _pop_standalone_date(parts)
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

    Orphan salvage (run bf0f7941, Nov 12 2026): a writer pass can emit a
    bullet FRAGMENT without its "- " marker (a hard-wrapped tail or a
    half-merged accomplishment). Such a line must never be gathered as a
    header field: it used to take the sub slot (which the project emission
    drops) AND trigger the >=2-header-line path that dropped the entry's
    date column - content silently vanished from the PDF. Mid-sentence
    lines are salvaged as entry content instead: joined onto the previous
    bullet, or kept as their own item when no bullet exists yet.
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
        if _is_section_heading(stripped):
            break  # a bare heading ('## X' / ALL-CAPS / 'Label:') starts a
                   # new section - never a header field of the entry above
                   # (run d946bb5f: 'PROJECTS' took the education subheading
                   # slot and the first two project entries were absorbed).
        if _HR_RE.match(stripped):
            # a decorative divider (---): skip it, don't treat as content
            idx += 1
            continue
        if _is_orphan_continuation(stripped, min_len=25):
            break  # a mid-sentence fragment - never a header field
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
        date_str, tail = _date_match_to_field(last_dates)
        if tail:
            # Mid-line date column ("Title | <date> | stack"): leave the line
            # whole so the split below routes every field; the date is
            # re-extracted there instead of being glued with the tail into
            # sub_right (run 28e2e56f: that glue overflowed the page).
            pass
        else:
            entry.sub_right = date_str
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
            if len(p) > 2 and not entry.sub_right and _STANDALONE_DATE_RE.match(
                    _clean_field(p[-1])):
                # A trailing standalone DATE column on the first header line
                # ("Title | stack | Mar 2026 - May 2026"): the ad-hoc
                # p[0]/p[1] extraction used to drop it (any >=2-header-line
                # entry lost its date - run bf0f7941's Smart Lecture row
                # shipped dateless).
                entry.sub_right = _normalize_date_ascii(_clean_field(p[-1]))
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
            bullet_raw = m.group(1).strip()
            if _is_link_item(bullet_raw):
                entry.links.append(bullet_raw)
            else:
                entry.items.append(_strip_inline(bullet_raw))
            idx += 1
        elif not lines[idx].strip():
            # blank line: peek ahead — if next is not a bullet, stop
            j = idx + 1
            if j < len(lines) and _BULLET_RE.match(lines[j]):
                idx += 1
                continue
            break
        elif _is_orphan_continuation(lines[idx].strip()):
            # Salvage (run bf0f7941): a writer pass emitted a bullet
            # fragment without its "- " marker. Join it onto the previous
            # bullet when one exists (a hard-wrapped tail), otherwise keep
            # it as its own item (a broken head) - either way it renders
            # instead of vanishing.
            text = _strip_inline(lines[idx].strip())
            if entry.items:
                entry.items[-1] = f"{entry.items[-1]} {text}"
            else:
                entry.items.append(text)
            idx += 1
        else:
            break

    # Date-shaped columns are normalized regardless of which branch filled
    # them: two-line headers can put the range in `right`, and the compact/
    # HTML/project paths all read these fields later.
    if entry.sub_right and (
            _DATE_RANGE_RE.search(entry.sub_right)
            or _MONTH_RANGE_RE.search(entry.sub_right)
            or _STANDALONE_DATE_RE.match(entry.sub_right)):
        entry.sub_right = _normalize_date_ascii(entry.sub_right)
    if entry.right and (
            _DATE_RANGE_RE.search(entry.right)
            or _STANDALONE_DATE_RE.match(entry.right)):
        entry.right = _normalize_date_ascii(entry.right)

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

        # Skills-family sections hold flat lists and label rows - plain
        # lines are never subheading entries. A plain 'Python, Docker,
        # Kafka' list line used to fall through to the entry parser and
        # get comma-split into header/right columns, lopping the list
        # into a broken two-column subheading row.
        if _is_skills_section(current_section.title):
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



def _display_url_text(s: str) -> str:
    """Displayed URL text without the scheme prefix (layout spec A, Nov 11
    2026): LinkedIn/GitHub links print as linkedin.com/... instead of
    https://linkedin.com/... so the contact row fits one line and link-heavy
    project headers wrap one line less. The FULL URL stays in the href -
    only the displayed characters change, which the spec's acceptance diff
    already whitelists ('text layer unchanged except whitespace and the
    removed https://')."""
    return re.sub(r'^https?://', '', s.strip(), flags=re.IGNORECASE)



def _extract_project_links_and_title(
    raw_title: str,
    other_fields: list[str],
    bullet_links: list[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Extract project title and links for ANY candidate resume, verbatim.

    Pulls ONLY links that genuinely exist in the source document:
      1. Embedded markdown links in raw_title: `**Project Name** [Live Demo](url) [GitHub](url)`
      2. Pipe-separated header fields: `**Project Name** | [Live Demo](url) | Tech Stack`
      3. Bullet links: `- [Live Demo](url)` or `- Live Demo: https://...` or raw URLs

    ANTI-FABRICATION RULE (audit finding P0): no link is ever constructed
    from a title slug, a candidate handle, or any other inference. A guessed
    repo/demo URL is a dead 404 on a recruiter's screen - worse than an
    honest absence. If the source has no link, none is rendered.

    Returns (clean_title, [(label, url), ...]) with deduplicated URLs.
    """
    links: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    seen_labeled: set[tuple[str, str]] = set()

    def add_link(lbl: str, url: str) -> None:
        if not url:
            return
        lbl = lbl.strip() or "Link"
        url = url.strip().rstrip(").,;")
        norm = url.lower().rstrip("/")
        # A repeated URL with a NEW label is real content, not a duplicate:
        # the source PDF links the SAME featured-project URL from the title,
        # a "(Graduation Project)" badge, and a "GitHub" row - dropping the
        # later labels deleted the user's own labels (run d946bb5f audit,
        # Nov 16). Only a repeated (label, url) PAIR is a true duplicate.
        key = (lbl.lower(), norm)
        if norm in seen_urls and key in seen_labeled:
            return
        seen_urls.add(norm)
        seen_labeled.add(key)
        links.append((lbl, url))

    # 1. Extract markdown links from raw_title
    md_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    for m in md_re.finditer(raw_title):
        add_link(m.group(1), m.group(2))
    clean_title = md_re.sub('', raw_title)

    # Clean raw URLs if any embedded in title
    url_re = re.compile(r'https?://\S+')
    for m in url_re.finditer(clean_title):
        u = m.group(0)
        lbl = "GitHub" if "github.com" in u.lower() else "Live Demo"
        add_link(lbl, u)
    clean_title = url_re.sub('', clean_title)

    # Strip formatting and stray pipe/dash/colon separators
    clean_title = _strip_inline(clean_title)
    # Bare link labels the model emitted WITHOUT a target ("**Title** Live
    # Demo | GitHub | …"): the URL is unknown, so no link can be rendered
    # (anti-fabrication rule), and a trailing "Live Demo"/"GitHub" glued to
    # the title is presentation leakage, not title text. Drop TRAILING
    # labels only - a title just containing those words elsewhere stays.
    clean_title = re.sub(
        r'\s*[|\-–—:,.]?\s*\b(?:live\s+demo|demo|github|git\s+repo(?:sitory)?|'
        r'repo(?:sitory)?|portfolio|website|web\s+app|playground|url|link)\b\.?\s*$',
        '', clean_title, count=0, flags=re.IGNORECASE)
    clean_title = re.sub(r'\s*[\|—–\-:]\s*$', '', clean_title).strip()
    clean_title = re.sub(r'^\s*[\|—–\-:]\s*', '', clean_title).strip()

    # 2. Check other header fields (e.g. if link was a separate pipe segment)
    for field in other_fields:
        for m in md_re.finditer(field):
            add_link(m.group(1), m.group(2))
        for m in url_re.finditer(field):
            u = m.group(0)
            lbl = "GitHub" if "github.com" in u.lower() else "Live Demo"
            add_link(lbl, u)

    # 3. Process bullet links
    for bl in bullet_links:
        sub_items = [s.strip() for s in re.split(r'\s*[\|—]\s*|\s+-\s+', bl) if s.strip()]
        for item in sub_items:
            m_md = md_re.search(item)
            if m_md:
                add_link(m_md.group(1), m_md.group(2))
                continue
            m_pre = re.search(r'^(?:live\s+demo|github|demo|repo|code|website|docs)\s*:\s*(https?://\S+)', item, re.IGNORECASE)
            if m_pre:
                lbl = item.split(':')[0].strip().title()
                add_link(lbl, m_pre.group(1))
                continue
            m_raw = url_re.search(item)
            if m_raw:
                u = m_raw.group(0)
                lbl = "GitHub" if "github.com" in u.lower() else "Live Demo"
                add_link(lbl, u)
                continue
            # No URL present: the link label alone (e.g. bare "- GitHub") is
            # NOT renderable. Fabricating a URL would violate the
            # anti-hallucination contract; the label is dropped entirely so
            # the heading only carries verifiably present links.

    return clean_title, links


def iter_achievement_bullets(
    resume_text: str,
    skip_section_re: Optional[re.Pattern] = None,
) -> list[tuple[str, str]]:
    """Return ``(section_title, bullet_text)`` pairs for achievement bullets."""
    try:
        data = parse_markdown_resume(resume_text or "")
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for section in data.sections:
        title = (section.title or "").strip()
        if not title:
            continue
        if skip_section_re is not None and skip_section_re.search(title):
            continue
        for entry in section.entries:
            for item in entry.items:
                text = _strip_inline(item or "")
                if not text or _is_link_item(text):
                    continue
                out.append((title, text))
    return out


def mkstemp_pdf_name(output_dir: Path, prefix: str) -> Path:
    """Per-render unique probe name from ``mkstemp`` (request-dir scoped)."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".pdf", dir=str(output_dir))
    os.close(fd)
    return Path(name)
