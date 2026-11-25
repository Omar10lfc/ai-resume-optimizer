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
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .helpers import _force_ascii

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

HAVE_PANDOC = PANDOC_PATH is not None
HAVE_PDFLATEX = PDFLATEX_PATH is not None

if not HAVE_PDFLATEX:
    logger.warning("LaTeX renderer: pdflatex not found on PATH — PDF export via "
                   "LaTeX template is unavailable (falling back to HTML→PDF).")
if not HAVE_PANDOC:
    logger.warning("LaTeX renderer: pandoc not found on PATH — DOCX export is unavailable.")


# ---------------------------------------------------------------------------
from .resume_parser import (  # noqa: F401
    Contact,
    Entry,
    ResumeData,
    Section,
    _BOLD_LABEL_RE,
    _DATE_RANGE_RE,
    _MONTH_RANGE_RE,
    _STANDALONE_DATE_RE,
    _clean_field,
    _display_url_text,
    _extract_contact,
    _extract_project_links_and_title,
    _is_link_item,
    _is_plain_label_line,
    _is_section_heading,
    _normalize_date_ascii,
    _should_split_on_comma,
    _split_header_and_dates,
    _strip_inline,
    mkstemp_pdf_name as _mkstemp_pdf_name,
    parse_markdown_resume,
)


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


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters after mapping Unicode characters to ASCII."""
    if not text:
        return ""
    text = _force_ascii(text)
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

    # Characters that terminate or break the \href{...} argument. Percent-encode
    # these; % and & are still escaped below, preserving the existing shape.
    url = ''.join(
        "%5F" if ch == "_"
        else "%7E" if ch == "~"
        else quote(ch, safe="") if ch in "\\{}$^"
        else ch
        for ch in url
    )

    url = url.replace(' ', '%20')
    url = url.replace('#', r'\#')
    url = url.replace('%', r'\%')
    url = url.replace('&', r'\&')
    return url


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
            # Display text hides the scheme (href keeps the full URL).
            parts.append(r'\href{' + clean_url + r'}{\underline{' + _latex_escape(_display_url_text(label)) + r'}}')
        else:
            parts.append(_latex_escape(piece))
    return ''.join(parts)


def _render_contact(contact: Contact) -> str:
    """Render the centered name + contact line using the template's heading style."""
    lines = []
    if not contact.name and not contact.details:
        return ""
    name_part = contact.name or "Candidate"
    # Name top offset (layout spec B item 5): the name printed 7.3pt from
    # the page's top edge vs ~34pt of bottom margin - a print-clip risk.
    # This offset is the LAST vertical dollar spent: every point here
    # pushes the whole page down.
    # RIGID GLUE GUARD: LaTeX's center environment is a trivlist whose
    # default \topsep/\partopsep are RUBBER (plus/minus stretch). The page
    # builder exercises that rubber differently depending on how much
    # content follows, which moved the name's top offset by +-2pt between
    # resumes (probe-calibrated Nov 11 2026: partopsep alone accounted for
    # exactly 2.0pt). Zeroing both inside a group pins the name to one
    # deterministic position for ANY body content; the group closes before
    # the sections, so no list elsewhere is affected.
    # -3.3pt lands the WORST case (accent-free names, glyph height = cap
    # height, probe floor 2.2pt under the old -15.3) at 14.2pt >= the
    # 14pt floor; names with accents/descenders sit even lower (measured
    # 16.1pt). Compact stop: run-B's last baseline ends at 760.6pt <= the
    # 763.2pt bottom-margin line (2.6pt margin).
    lines.append(r"\vspace*{-3.3pt}")
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\topsep}{0pt}\setlength{\partopsep}{0pt}")
    lines.append(r"\begin{center}")
    # Reduced from \Huge to \LARGE (or \Large if long) so name heading is proportional
    heading_font = r"\LARGE" if len(name_part) <= 30 else r"\Large"
    lines.append(r"\textbf{" + heading_font + r" \scshape " + _latex_escape(name_part) + r"} \\ \vspace{1pt}")
    # The professional title on its OWN line under the name (Nov 16 2026,
    # user ask: run d946bb5f's header glued "AI/ML Engineer" into the
    # contact row; the conventional layout puts it under the name).
    if contact.title:
        lines.append(r"\small " + _latex_escape(contact.title) + r" \\ \vspace{1pt}")
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
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(_display_url_text(p)) + r'}}')
                continue

            # Check for GitHub profile
            if 'github.com' in p.lower():
                url = p if p.startswith(('http://', 'https://')) else 'https://' + p
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(_display_url_text(p)) + r'}}')
                continue

            # Check for web URLs or portfolio domains (.co covers
            # huggingface.co - run d946bb5f's header shipped the HF handle
            # as plain text while linkedin/github linked; net joins the
            # common portfolio set)
            if re.match(r'^(?:https?://|www\.)', p) or re.search(r'\.(?:io|dev|ai|app|me|tech|com|org|co|net)(?:/|$)', p):
                url = p if p.startswith(('http://', 'https://')) else 'https://' + p
                rendered.append(r'\href{' + _clean_latex_url(url) + r'}{\underline{' + _latex_escape(_display_url_text(p)) + r'}}')
                continue

            # Otherwise plain text (e.g. location, phone number)
            rendered.append(_latex_escape(p))

        lines.append(r"\small " + r" $|$ ".join(rendered))
    lines.append(r"\end{center}")
    lines.append(r"\endgroup")
    lines.append("")
    return "\n".join(lines)


def _render_section(section: Section, compact: bool = False) -> str:
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
            # Layout spec B (Nov 11 2026) - section-tail normalization: when a
            # section ENDS on a header row (Education-style subheading tabular
            # or a header-only ProjectHeading), the next section heading's
            # gap above it is much tighter than after an entry that ends on a
            # bullet list (measured 4.7pt vs 9.3pt on the shipped PDF). The
            # emitted tail token gets replaced in render_resume_tex with a
            # per-mode \vspace that re-uniforms every heading gap.
            last_header_kind = None
            last_is_bulleted = False
            parts.append(r"\resumeSubHeadingListStart")
            for e in populated:
                headers_left = e.header or ""
                headers_right = e.right or ""
                sub_left = e.sub or ""
                sub_right = e.sub_right or ""

                if is_project_sec:
                    # Projects: title + links + tech stack + date
                    if _DATE_RANGE_RE.search(headers_right) and not sub_right:
                        date_field = headers_right
                        tech_stack_raw = sub_left
                    else:
                        date_field = sub_right or (_clean_field(headers_right) if _DATE_RANGE_RE.search(headers_right) else "")
                        tech_stack_raw = headers_right if not _DATE_RANGE_RE.search(headers_right) else sub_left

                    # Last-line defense (run 28e2e56f): a date column that
                    # still carries "| more fields" would print the whole
                    # string right-aligned and run off the page (measured
                    # x1=735 on a 612pt sheet). Split it: the date is left of
                    # the pipe; the rest becomes the tech stack when none was
                    # parsed.
                    m_glue = re.match(r"^(.*\d{4})\s*\|\s*(\S.*)$", date_field)
                    if m_glue:
                        date_field = m_glue.group(1).strip()
                        if not tech_stack_raw:
                            tech_stack_raw = m_glue.group(2).strip()

                    # Universally extract project title and links
                    other_fields = [f for f in (headers_right, sub_left) if f and f != date_field]
                    clean_title, extracted_links = _extract_project_links_and_title(
                        raw_title=headers_left,
                        other_fields=other_fields,
                        bullet_links=e.links,
                    )

                    # Remove any links from tech stack
                    md_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
                    clean_tech = md_re.sub('', tech_stack_raw)
                    clean_tech = re.sub(r'https?://\S+', '', clean_tech)
                    clean_tech = re.sub(r'^\s*[\|—–\-:]\s*', '', clean_tech).strip()
                    clean_tech = re.sub(r'\s*[\|—–\-:]\s*$', '', clean_tech).strip()
                    tech_text = _render_text_with_links(clean_tech)

                    # Render clickable links with \href and \underline
                    formatted_links = [
                        r'\href{' + _clean_latex_url(u) + r'}{\underline{' + _latex_escape(lbl) + r'}}'
                        for lbl, u in extracted_links
                    ]

                    date_text = _latex_escape(date_field)
                    title_bold = r"\textbf{" + _latex_escape(clean_title) + r"}"

                    # Estimate single-line width
                    title_plain_len = len(clean_title)
                    links_plain_len = sum(len(lbl) + 3 for lbl, _ in extracted_links)
                    tech_plain_len = len(clean_tech)
                    date_plain_len = len(date_field)
                    total_len = title_plain_len + links_plain_len + tech_plain_len + date_plain_len
                    # Left-side-only width (Nov 16 2026, run d946bb5f): the
                    # restored same-URL labels ("Live Demo", "(Graduation
                    # Project)", "GitHub" - all one featured URL) give a
                    # link-heavy title with NO tech text, so the old
                    # `or not tech_text` clause shipped a ~124-char single
                    # line that pushed the right-aligned date off the sheet.
                    left_len = title_plain_len + links_plain_len

                    if total_len <= 85 or (not tech_text and left_len <= 85):
                        # Fits on a single line via \resumeProjectHeading
                        # (a link-only title has an empty clean_title - the
                        # label rides the link - so no empty \textbf{}
                        # segment, no stray $|$ separator: bf58e700
                        # follow-up, Nov 15 2026).
                        left_items = []
                        if clean_title:
                            left_items.append(title_bold)
                        if formatted_links:
                            left_items.extend(formatted_links)
                        if tech_text:
                            left_items.append(r"\emph{" + tech_text + r"}")
                        left_line = " $|$ ".join(left_items)
                        parts.append(r"\resumeProjectHeading{" + left_line + r"}{" + date_text + r"}")
                        last_header_kind = "ph"
                    else:
                        # Multi-line: Title + links on Row 1, tech stack on Row 2.
                        # With no tech text but a too-long link list, the
                        # surplus links move to Row 2 (Nov 16 2026: the
                        # user's original resume shows "GitHub" under the
                        # project, not on a 124-char row 1).
                        row1_links = formatted_links
                        row2_links: list = []
                        if not tech_text and len(formatted_links) > 1:
                            budget = 85 - title_plain_len
                            used = 0
                            cut = 0
                            for idx2, (lbl, _u) in enumerate(extracted_links):
                                w = len(lbl) + 3
                                if used + w <= budget:
                                    used += w
                                    cut = idx2 + 1
                                else:
                                    break
                            cut = max(cut, 1)
                            row1_links = formatted_links[:cut]
                            row2_links = formatted_links[cut:]
                        row1_items = []
                        if clean_title:
                            row1_items.append(title_bold)
                        if row1_links:
                            row1_items.append(r"\textnormal{" + " $|$ ".join(row1_links) + r"}")
                        row1_left = " $|$ ".join(row1_items)
                        tech_row = tech_text
                        if row2_links:
                            links_line = " $|$ ".join(row2_links)
                            tech_row = (f"{tech_row} $|$ {links_line}"
                                        if tech_row else links_line)
                        parts.append(r"\resumeSubheading{" + row1_left + r"}{" + date_text + r"}{" + tech_row + r"}{}")
                        last_header_kind = "sub"
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
                    last_header_kind = "ph"
                elif is_edu_sec:
                    # Education: University & Location on row 1, Degree & Dates on row 2
                    inst_text = _render_text_with_links(headers_left)
                    loc_text = _render_text_with_links(headers_right)
                    deg_text = _render_text_with_links(sub_left)
                    date_text = _render_text_with_links(sub_right)
                    parts.append(r"\resumeSubheading{" + inst_text + "}{" +
                                 loc_text + "}{" + deg_text + "}{" + date_text + "}")
                    last_header_kind = "sub"
                else:
                    # Generic entries (Awards, Leadership, Publications, Certifications, etc.)
                    h_text = _render_text_with_links(headers_left)
                    r_text = _latex_escape(headers_right)
                    s_text = _render_text_with_links(sub_left)
                    sr_text = _latex_escape(sub_right)
                    if not s_text and not sr_text:
                        parts.append(r"\resumeProjectHeading{\textbf{" + h_text + r"}}{" + r_text + r"}")
                        last_header_kind = "ph"
                    else:
                        parts.append(r"\resumeSubheading{" + h_text + "}{" +
                                     r_text + "}{" + s_text + "}{" + sr_text + "}")
                        last_header_kind = "sub"

                if e.items:
                    parts.append(r"\resumeItemListStart")
                    for item in e.items:
                        parts.append(r"\resumeItem{" + _render_text_with_links(item) + "}")
                    parts.append(r"\resumeItemListEnd")
                    last_is_bulleted = True
                    last_header_kind = None  # the list tail takes over the spacing

            # AFTER the entry loop (single per-section tail, layout spec B):
            if last_header_kind == "sub":
                parts.append(r"\resumeSubHeadingListEnd<!--SECTION_TAIL_NUDGE:sub-->")
            elif last_header_kind == "ph":
                parts.append(r"\resumeSubHeadingListEnd<!--SECTION_TAIL_NUDGE:ph-->")
            elif last_is_bulleted:
                # The section ends on a bullet list whose tail grew when
                # the entry-title gaps went to the >=5pt tier (layout
                # spec B): pull the NEXT section heading back to the
                # shared reference distance.
                parts.append(r"\resumeSubHeadingListEnd<!--SECTION_TAIL_NUDGE:list-->")
            else:
                parts.append(r"\resumeSubHeadingListEnd")
            parts.append("")

    if section.simple_lines:
        items = []
        hang_flags = []
        for ln in section.simple_lines:
            m = _BOLD_LABEL_RE.match(ln)
            if m:
                # **Label**: value → bold label, formatted value
                label = _strip_inline(m.group(1))
                value = m.group(2)
                item_text = (r"\textbf{" + _latex_escape(label) + r"}{: " +
                             _render_text_with_links(value) + r"}")
                is_label_row = True
            else:
                # Generic label:value → bold label (never split inside a markdown
                # link or URL scheme like 'https://...' / 'mailto:...')
                cm = re.match(r'^(.*?):\s*(.*)$', ln)
                if (cm and len(cm.group(1)) <= 35
                        and '[' not in cm.group(1)
                        and not cm.group(2).startswith('//')
                        and not cm.group(1).lower().endswith('mailto')):
                    item_text = (r"\textbf{" + _latex_escape(cm.group(1)) + r"}{: " +
                                 _render_text_with_links(cm.group(2)) + r"}")
                    is_label_row = True
                else:
                    # A simple line can still carry a raw list marker when a
                    # verbatim-shipped section keeps its bullet lines (the
                    # cert links ride "- [label](url)" lines). Strip the
                    # marker so it does not print inside the \underline
                    # label (bf58e700 follow-up, Nov 15 2026).
                    item_text = _render_text_with_links(
                        re.sub(r'^\s*[-*•]\s+', '', ln))
                    is_label_row = False
            items.append(item_text)
            hang_flags.append(is_label_row)

        # One \item PER ROW with itemsep=0pt (layout spec A, Nov 11 2026):
        # the old \\-joined single item made every row a forced break inside
        # one paragraph, so TeX's \looseness could never shave a stranded
        # row tail ("Neural Networks" printed alone). As separate paragraphs
        # each row gets its own \looseness pass; itemsep=0pt keeps the
        # 12pt row rhythm identical (rows sit on baselineskip like the
        # forced breaks did). Text and row order unchanged.
        # Compact mode also tightens the list-start glue (layout spec B): the
        # skip between a section rule and the first row/entry is unclaimed
        # vertical budget the 763.2pt bottom-margin stop needs. enumitem's
        # `before` key is used (NOT topsep - probe-calibrated Nov 11 2026:
        # topsep glue is swallowed at list start, while before=\vspace
        # lands 1:1). -1.5pt keeps ~3.5pt of visual air under each rule.
        # Label rows additionally carry a hanging indent (spacing fix, live
        # runs ca102fa5 / 71f879ae, Nov 11 2026): wrapped row tails
        # ("Fine-Tuning, Neural Networks") broke flush-left and measured
        # IDENTICAL to the row-to-row gap (uniform 10.96pt dy), so a wrapped
        # tail was indistinguishable from the next skill row - the "wall of
        # text" look. Continuation lines indent 1em; first line unchanged.
        # Horizontal-only: zero vertical budget cost, safe for the compact
        # pass's ~4pt bottom slack. Probe-proven mechanics (Nov 11 2026):
        # - LaTeX breaks each item's paragraph at the NEXT \item (latex.ltx
        #   \@item: \ifhmode \unskip\unskip \par), OUTSIDE the \item{...}
        #   group - parameters assigned inside the group revert first
        #   (\hangindent measured dead there). The row therefore closes its
        #   paragraph INSIDE the group with an explicit \par.
        # - The list sets \parshape 1 \@totalleftmargin \linewidth
        #   (latex.ltx \@trivlist) and enumitem does its indenting THROUGH
        #   that shape, so \hangindent is ignored (TeX rule: nonempty
        #   \parshape wins) and emptying the shape (\parshape 0) strips the
        #   list margin itself (line 1 printed at the outer edge). The shape
        #   is REPLACED with two entries instead: line 1 keeps the shipped
        #   geometry (indent \leftmargin - equal to \@totalleftmargin at
        #   depth 1, where \list does \advance\@totalleftmargin\leftmargin -
        #   and width \linewidth); every later line is +1em indented and
        #   \linewidth-1em wide (last pair repeats for all remaining lines).
        #   \@-internal registers cannot appear in the emitted TeX: at
        #   document-body catcodes \@ is \spacefactor (compile-fatal).
        # - \par also makes the pre-existing \looseness=-1 live, which
        #   probe-measured re-breaking rows (short first line, crammed
        #   second); \looseness=0 after the pinned literal keeps the shave
        #   inert exactly as shipped. Prose rows (Summary/Certifications)
        #   keep the shipped emission byte-for-byte - no geometry change.
        _HANG = (r"\parshape 2 \leftmargin \linewidth "
                 r"\dimexpr\leftmargin+1em\relax "
                 r"\dimexpr\linewidth-1em\relax \looseness=0 ")
        row_opts = ("leftmargin=0.15in, label={}, itemsep=0pt, parsep=0pt"
                    + (r", before=\vspace{-1.5pt}" if compact else ""))
        if len(items) > 1:
            parts.append(r"\begin{itemize}[" + row_opts + "]")
            parts.append(r"    \small{")
            for it, hang in zip(items, hang_flags):
                if hang:
                    parts.append(r"    \item{\looseness=-1 " + _HANG
                                 + it + r"\par}")
                else:
                    parts.append(r"    \item{\looseness=-1 " + it + "}")
            parts.append(r"    }")
            parts.append(r"\end{itemize}<!--SECTION_TAIL_NUDGE:rows-->")
            parts.append("")
        elif items:
            parts.append(r"\begin{itemize}[" + row_opts + "]")
            if hang_flags[0]:
                parts.append(r"    \small{\item{\looseness=-1 " + _HANG
                             + items[0] + r"\par}}")
            else:
                parts.append(r"    \small{\item{\looseness=-1 "
                             + items[0] + r"}}")
            parts.append(r"\end{itemize}<!--SECTION_TAIL_NUDGE:rows-->")
            parts.append("")

    # A section that ended up with no usable content (no non-empty entries and
    # no simple lines) would render an empty \section that pdflatex rejects —
    # drop the whole section.
    if not section.simple_lines and not any(
            e.header or e.right or e.sub or e.sub_right for e in section.entries):
        return ""

    return "\n".join(parts)



# ============================================================
# Enforced single-page containment
# ============================================================

_COMPACT_BLOCK_TOKEN = "<!--COMPACT_BLOCK-->"

# Spacing-only overrides injected when the first render overflowed one
# page. They must not change any content or macro shapes - only spacing
# and margins - so the compact PDF prints the same information denser.
# \renewcommand is safe here because the macros were defined
# unconditionally earlier in this same preamble.
#
# Two severities ship (layout spec B, Nov 11/12 2026), selected by
# render_compact_tiers:
#   tier 1 "airy" (run 28e2e56f, Nov 12 2026): the tight values crushed
#       the rule-to-content air to ~2.2pt of visible ink under EVERY
#       section rule (user-reported crowding). Tier 1 restores ~3pt per
#       section (~+18pt/page) - list-start glue -1.0pt and after-rule
#       -2.0pt (now -4.0pt paired with the -2pt rule pull-up, Nov 16)
#       put the air back at base mode's ~4.7pt - and is adopted
#       only when the render stays one page AND the last baseline stops
#       above _COMPACT_BOTTOM_STOP.
#   tier 2 "balanced" (run 03a3ffc1, Nov 12 2026): with the Aegis context
#       the resume grew long enough that tier-1 no longer fit one page
#       and the old floor shipped with ~0.2pt of ink under every rule -
#       the user's "no minimal space" defect. The floor's after-rule
#       -4.5pt was 1.0pt past the base template's -3.5pt for zero page
#       gain on most resumes: the balanced tier spends that point back
#       (+6pt/page over the floor) and still fits one page on the
#       page-full with-context resume (last baseline 775.8 of a 776pt
#       block).
#   tier 3 "floor": the former tight tier - the only unclaimed vertical
#       levers after the entry-gap floors consumed the page slack.
#       Adopted whenever it yields one page (today's acceptance rule).
#       (Nov 16 2026: each tier carries BOTH content-position levers
#       paired inside \titleformat - the rule pull-up (how far the rule
#       rises into the heading's air) and the after-rule value - and the
#       pair must sum to the historic content offset (base -7.5; tiers
#       -6.0/-7.5/-8.5) so content keeps its exact position. Second
#       Nov 16 pass, live run af7f9480 (tier 2): the first pass pulled
#       the rule 4pt up on every tier and left only ~3.5-4.1pt of ink
#       between the heading and its rule - the user's "the space between
#       the section header and the ruler need to be separated a bit".
#       The pull-up is now tier-dependent: -2.0pt on the airy tiers
#       (heading air 5.5-6.1pt, rule-to-content 3.5-5.1pt on the live
#       render) and -3.0pt on the overflow floor, whose rule-to-content
#       air may never re-enter the glued zone.)
_COMPACT_TIER_VALUES = {
    1: {"list_start_glue": r"\vspace{-1.0pt}", "rule_pullup": r"\vspace{-2.0pt}",
        "after_rule": r"\vspace{-4.0pt}"},
    2: {"list_start_glue": r"\vspace{-1.5pt}", "rule_pullup": r"\vspace{-2.0pt}",
        "after_rule": r"\vspace{-5.5pt}"},
    3: {"list_start_glue": r"\vspace{-1.5pt}", "rule_pullup": r"\vspace{-3.0pt}",
        "after_rule": r"\vspace{-5.5pt}"},
}


def _compact_overrides(tier: int) -> str:
    """The auto-compact preamble block for one tier. Tokens are filled with
    str.replace - str.format is unusable on LaTeX (brace soup)."""
    values = _COMPACT_TIER_VALUES[tier]
    return (_COMPACT_OVERRIDES_TEMPLATE
            .replace("@@LIST_START_GLUE@@", values["list_start_glue"])
            .replace("@@RULE_PULLUP@@", values["rule_pullup"])
            .replace("@@AFTER_RULE@@", values["after_rule"]))


_COMPACT_OVERRIDES_TEMPLATE = r"""
% ---- AUTO-COMPACT PASS (activated after a >1-page first render) ----
% Tighter bullet spacing (enumitem is loaded in this preamble)
\setlist[itemize]{itemsep=0pt, parsep=0pt, topsep=2pt, leftmargin=*}
% Bullet breathing room (Nov 16 2026, user ask: with the global
% itemsep=0pt above, consecutive \resumeItem bullets sat at the bare
% interline gap). Scoped to the BULLET lists only - the entry-title
% lists keep their own calibrated rows (the 5pt floor / 6pt target),
% and 1.5pt x ~14 bullets is absorbed by the compact tier ladder.
\renewcommand{\resumeItemListStart}{\begin{itemize}[itemsep=1.5pt]}
% Entry-list start glue (layout spec B, Nov 11 2026): the compact pass
% pulls each section rule closer to its first entry - probe-calibrated
% (topsep is swallowed at list start; enumitem's before=\vspace lands 1:1).
% Tier-dependent value (see _COMPACT_TIER_VALUES in latex_render.py).
\renewcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}, topsep=0pt, before=@@LIST_START_GLUE@@]}
% Tighter section heading + rule spacing (tier-dependent after-rule value,
% paired with the before= start glue above; the tier-3 floor pulls the
% rule up only 3.0pt).
% Rule pull-up @@RULE_PULLUP@@ (Nov 16 2026, two live passes): content sat
% 0.0-1.7pt under every section rule (run e9f5c016 - the "text glued to
% the header" report), so the rule was lifted into the heading's air with
% every tier's after-rule value compensating (pull-up + after-rule = the
% historic content offset, content position untouched). The first pass
% pulled 4pt on EVERY tier and left only ~3.5-4.1pt of ink between the
% heading and its rule (live run af7f9480, tier 2 - the user's "the space
% between the section header and the ruler need to be separated a bit").
% The pull-up is now tier-dependent - -2.0pt on the airy tiers (heading
% air 5.5-6.1pt, rule-to-content 3.5-5.1pt on the live render, repro-
% verified at zero page cost: last baseline 774.7 unchanged) and -3.0pt
% on the overflow floor, whose rule-to-content air stays >= 2pt.)
\titleformat{\section}{
  \vspace{-7pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}@@RULE_PULLUP@@\titlerule @@AFTER_RULE@@]
% Recover a little vertical room
% (Sheet-edge cap, run 124f40cb, Nov 12 2026: with the template's text
% block bottom capped at 776pt, this lever may re-open the page only UP
% TO the same line - top margin 9.4pt + textheight 766.6pt = bottom
% 776pt. The historic +.24in would have put the compact block bottom at
% 784.6pt, close enough to the 792pt sheet edge to repeat the
% half-printed-last-line defect in the tighter mode.)
\addtolength{\topmargin}{-.12in}
\addtolength{\textheight}{.12in}
% Recover a little horizontal room (long linked project headings wrap less)
\addtolength{\oddsidemargin}{-.05in}
\addtolength{\evensidemargin}{-.05in}
\addtolength{\textwidth}{.1in}
% Trim the inter-entry breathing room
% (Run 868e406d, Nov 6 2026: the earlier -9pt spacer after the header
% tabulars pulled the following bullet lists and the next heading's own
% rows ONTO the header/tech lines - measured header->bullet baseline gap
% collapsed to 2.8pt. -4pt keeps the density win without glyph collisions.)
% VERTICAL-SPACING TIERS (layout spec B, Nov 11 2026) - the compact pass
% shrinks in this exact order and never touches resume text:
%   1) newly widened entry-title gaps are rebuilt tighter than base but the
%      visible entry gap NEVER drops below 5pt (>=2.5x the wrapped-line gap);
%   2) only then the historic margin levers (topmargin/textheight/width).
% pdfplumber-calibrated on the run-B render (Nov 11 2026):
%  - \resumeItemListEnd +3.0pt -> entry-title baseline distance 15.0pt
%    (gap 5.0pt = exactly 2.5x the 11.0pt intra-bullet wrap distance, the
%    spec floor; the 6pt target cannot fit this resume inside the 763.2pt
%    bottom-margin stop).
%  - tabular tails -2.8pt -> stack->first-bullet distance 10.0pt (gap 1.0pt,
%    above the 0.3pt collision bug this tail class fixes; the 1.5pt target
%    was scaled down per the spec's "scale items 1-2" fallback).
%  - after-rule -5.5pt (historic -7.5pt paired with the -2pt rule
%    pull-up; see _COMPACT_TIER_VALUES) - the only unclaimed lever left.
% The 539f7c1d entry-boundary minimum stays the floor for every list end.
\renewcommand{\resumeItem}[1]{\item\small\looseness=-1{{#1 \vspace{-2pt}}}}
\renewcommand{\resumeSubheading}[4]{
  \vspace{-1pt}\item
    \begin{tabular*}{\dimexpr\textwidth-0.15in\relax}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-3.1pt}
}
\renewcommand{\resumeSubSubheading}[2]{
    \item
    \begin{tabular*}{\dimexpr\textwidth-0.15in\relax}{l@{\extracolsep{\fill}}r}
      \textit{\small#1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-3.1pt}
}
\renewcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{\dimexpr\textwidth-0.15in\relax}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-3.1pt}
}
% Entry-boundary repair (live request 539f7c1d, Nov 8 2026): after the
% itemize's own -2pt item tails, a negative end-of-list spacer collapsed
% the bullet-to-(next-entry)-title baseline distance to 7.0pt - normal
% wrapped lines sit 12.0-12.2pt apart, so 10pt-font descenders ("pipeline.",
% "service.") printed straight through the next entry's bold titles
% ("Smart Lecture Assistant", "AI Resume Optimizer Agent").
% Layout spec B tier: +3.0pt keeps the compact entry gap at exactly the
% 5pt floor (>=2.5x the intra-bullet wrap gap) after the tabular tails
% and margin levers were spent; see the tier comment above.
\renewcommand{\resumeItemListEnd}{\end{itemize}\vspace{3.0pt}}
% ---------------- end auto-compact pass ----------------
"""


def pdf_page_count(pdf_path) -> Optional[int]:
    """Number of pages in a compiled PDF, or None when unreadable/missing."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return None


# Bottom-margin stop for the airy compact tier (layout spec B, Nov 11 2026):
# the compact pass must keep the last text baseline AT or above this line;
# the airy tier is only adopted when it stays inside it. The balanced tier-2
# and the floor tier-3 may still cross it on long resumes (one page beats
# two); their renders stay sheet-safe by construction (one page = inside
# the 776pt block bottom).
_COMPACT_BOTTOM_STOP = 763.2


def pdf_last_baseline(pdf_path) -> Optional[float]:
    """Baseline (top + font size) of the lowest text line, or None when
    pdfplumber is unavailable or the PDF is unreadable.

    Same measuring convention as the layout-spec B tests (_pdf_lines:
    line baseline = glyph top + font size), so the constant above and the
    test pins describe the same geometry.
    """
    try:
        import pdfplumber
    except ImportError:
        return None
    try:
        worst = None
        with pdfplumber.open(str(pdf_path)) as doc:
            for page in doc.pages:
                for w in page.extract_words(extra_attrs=["size"]):
                    v = w["top"] + w["size"]
                    if worst is None or v > worst:
                        worst = v
        return worst
    except Exception:
        return None



def render_compact_tiers(data: ResumeData, output_dir,
                         filename: str = "Optimized_Resume.pdf") -> Optional[dict]:
    """Auto-compact tier pass for the >1-page containment (pdf_exporter).

    Compiles the airy tier-1 first and adopts it only when the render stays
    one page AND the last baseline stops above _COMPACT_BOTTOM_STOP; only
    then does it pay the +18pt/page air cost. Otherwise the balanced tier-2
    renders and is adopted whenever it yields one page; the floor tier-3
    (the former tight values) is the last resort for resumes even the
    balanced tier cannot hold on one page. A one-page airy render that
    crosses the stop is kept as a last-resort fallback - one deep-margin
    page beats a two-page resume.

    Tiers compile to probe filenames first and the winner is copied onto
    ``filename``: a losing tier must never leave its (overflowing) PDF
    behind under the shipped name.

    Returns {"pdf_path", "tex", "tier", "pages"} for the adopted render, or
    None when no tier produced a one-page PDF (the caller keeps the base
    render and surfaces the overflow warning).
    """
    output_dir = Path(output_dir)
    probes: dict[int, tuple[str, Optional[str]]] = {}
    fallback_tier = None
    chosen_tier = None
    for tier in (1, 2, 3):
        tex = render_resume_tex(data, compact=True, compact_tier=tier)
        probe = latex_to_pdf(tex, output_dir,
                             _mkstemp_pdf_name(output_dir, f"__compact_tier{tier}_").name)
        probes[tier] = (tex, probe)
        if not probe or (pdf_page_count(probe) or 99) != 1:
            continue
        if tier == 1:
            last = pdf_last_baseline(probe)
            if last is not None and last > _COMPACT_BOTTOM_STOP:
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
    tex, probe = probes[winner]
    dest = output_dir / filename
    shutil.copyfile(probe, dest)
    for tier_probe in (p for _, p in probes.values() if p):
        probe_path = Path(tier_probe)
        if probe_path != dest:
            probe_path.unlink(missing_ok=True)
    return {"pdf_path": str(dest), "tex": tex, "tier": winner, "pages": 1}


def render_resume_tex(data: ResumeData, compact: bool = False,
                      compact_tier: int = 2) -> str:
    """Fill the bundled LaTeX template with the structured resume data.

    compact=True activates the auto-compact spacing block (tighter itemize
    spacing, trimmed section/entry vspaces, slightly larger text area);
    compact_tier selects the severity (1 = airy, restores rule-to-content
    air while the page budget allows; 2 = the shipped tight values, the
    default). Content itself is never altered - compaction is spacing-only.
    """
    template_path = Path(__file__).resolve().parent / "templates" / "resume_template.tex"
    template = template_path.read_text(encoding="utf-8")

    body_parts = []
    contact_tex = _render_contact(data.contact)
    if contact_tex:
        body_parts.append(contact_tex)
    for section in data.sections:
        body_parts.append(_render_section(section, compact=compact))

    body = "\n".join(body_parts) if body_parts else _latex_escape(
        "No structured content could be extracted — please provide a resume.")

    # Section-tail normalization (layout spec B, Nov 11 2026): every section
    # heading must sit 21.4pt (gap 9.4pt) below the previous section's last
    # line, regardless of how that section ended. The tokens are
    # pdfplumber-derived from zero-nudge probe renders of the run-B data
    # (Nov 11 2026) and re-checked against the shipped render:
    #   list - a section ending on a bullet list (inner itemize end + iLE
    #          + outer itemize end all land here);
    #   sub  - a section ending on a two-row header tabular;
    #   ph   - a section ending on a bare one-row \resumeProjectHeading
    #          (header-only entry); probe-calibrated, see test;
    #   rows - a section ending on the simple-lines rows block. The base
    #          rows end carries ~9pt of hidden glue (the block's own
    #          topsep-like end skip survives the heading merge), hence the
    #          large negative token.
    # Base mode has no page-budget pressure, so its tokens exist purely for
    # gap uniformity; compact mode's values are also tuned so run-B's last
    # baseline stops above the 763.2pt bottom-margin line.
    _NUDGE = {
        "list": r"\vspace{-3.9pt}" if compact else r"\vspace{-10.4pt}",
        "sub": r"\vspace{1.8pt}" if compact else r"\vspace{-3.9pt}",
        "ph": r"\vspace{2.3pt}" if compact else r"\vspace{-3.3pt}",
        "rows": r"\vspace{-2.8pt}" if compact else r"\vspace{-9.0pt}",
    }
    for kind, value in _NUDGE.items():
        body = body.replace(f"<!--SECTION_TAIL_NUDGE:{kind}-->", value)

    # Replace the placeholder comment with the rendered body
    template = template.replace("<!--RESUME_BODY-->", body)
    # Compact pass: injected only when the first render overflowed one page
    template = template.replace(
        _COMPACT_BLOCK_TOKEN,
        _compact_overrides(compact_tier) if compact else "")
    return template


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _tex_env() -> dict:
    """The TeX-Live paranoid-file-access settings, applied to the compile
    subprocess: ``openin_any=p`` (never read outside the compile dir) and
    ``openout_any=p`` (never write outside it). kpathsea honors these env
    vars on TeX Live; MiKTeX ignores them, so the app-level guard in
    ``_assert_tex_file_access`` is the engine-independent backstop.
    """
    env = os.environ.copy()
    env["openin_any"] = "p"
    env["openout_any"] = "p"
    return env


def _assert_path_ok(target: str) -> None:
    """Shared shape gate for file-reading TeX macros.

    Bare, non-path names are allowed (they name files inside or around the
    compile dir, which the engine accepts); anything path-ish that could
    escape it is refused.
    """
    target = (target or "").strip()
    if not target:
        return
    if (Path(target).is_absolute()
            or ".." in target
            or any(sep in target for sep in ("/", "\\"))
            or target.startswith((".", ":"))):
        raise ValueError(
            f"latex file access outside the compile dir: {target!r}")


def _openin_targets(tex_content: str) -> list[str]:
    """Extracts the file-name token after each ``\\openin<n>=<path>``.

    Requires a digit (the channel number) or ``=`` right after the macro
    name, so ``\\openin`` followed by e.g. ``\\end{document}`` is not
    mistaken for a file access."""
    targets: list[str] = []
    for m in re.finditer(r"\\openin", tex_content):
        tail = tex_content[m.end():].lstrip()
        if not tail or not (tail[0].isdigit() or tail[0] == "="):
            continue
        m2 = re.match(r"[\s0-9=]*=?\s*\"?([^\"\s{}]+)", tail)
        if m2:
            targets.append(m2.group(1))
    return targets


def _assert_tex_file_access(tex_content: str, allowed_dir: Path) -> None:
    """Rejects any explicit file access from untrusted TeX.

    Trusted TeX engines also expose ``openin_any`` / ``openout_any`` (TeX Live;
    MiKTeX ignores the equivalent ini/env settings). This app-level scan is
    the enforcement that works on every engine. Covered macros: ``\\input``,
    ``\\include``, ``\\openin`` (and the ``\\read`` channel it gates),
    ``\\InputIfFileExists``, ``\\usepackage`` with a path, and
    ``\\verbatiminput`` (which reads a file verbatim). All must be bare
    filenames, never paths that would read from outside the compile dir.
    """
    # \input: braced, quoted-unbraced and bare unbraced forms.
    for target in re.findall(r"\\input\s*\{([^{}]*)\}", tex_content):
        _assert_path_ok(target)
    for target in re.findall(r"\\input\s*\"([^\"]*)\"", tex_content):
        _assert_path_ok(target)
    for target in re.findall(r"\\input\s+([^\s{}]+)", tex_content):
        _assert_path_ok(target)
    # \include: same forms (\includegraphics is a different macro - untouched).
    for target in re.findall(r"\\include\s*\{([^{}]*)\}", tex_content):
        _assert_path_ok(target)
    for target in re.findall(r"\\include\s+([^\s{}]+)", tex_content):
        _assert_path_ok(target)
    # \openin<number>=<path> - the channel that \read reads through.
    for target in _openin_targets(tex_content):
        _assert_path_ok(target)
    # \read - refused outright (reads go through open channels). The
    # negative lookahead keeps \read1 (the real form, number follows the
    # control word) matched while \readFile-style words stay separate.
    if re.search(r"\\read(?![a-zA-Z])", tex_content):
        raise ValueError("latex \\read is not allowed in generated documents")
    # \InputIfFileExists<path> (braced + unbraced forms).
    for target in re.findall(r"\\InputIfFileExists\s*\{([^{}]*)\}", tex_content):
        _assert_path_ok(target)
    for target in re.findall(r"\\InputIfFileExists\s+([^\s{}]+)", tex_content):
        _assert_path_ok(target)
    # \usepackage with a path (braced + unbraced/no-space forms).
    for target in re.findall(r"\\usepackage\s*\{([^{}]*)\}", tex_content):
        _assert_path_ok(target)
    for target in re.findall(r"\\usepackage\s*([^\s{}]+)", tex_content):
        _assert_path_ok(target)
    # \verbatiminput reads a file verbatim - refused outright.
    if re.search(r"\\verbatiminput", tex_content):
        raise ValueError("latex \\verbatiminput is not allowed in generated documents")


def _run(cmd: list[str], cwd: Path, timeout: int = 120,
         env: dict | None = None) -> tuple[int, str]:
    """Run a subprocess, returning (returncode, combined stdout+stderr)."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
            env=env,
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
        try:
            _assert_tex_file_access(tex_content, tmp_dir)
        except ValueError as exc:
            logger.error("LaTeX compile refused unsafe \\input: %s", exc)
            return None
        rc, out = _run([PDFLATEX_PATH, "-interaction=nonstopmode", "-halt-on-error",
                        "-no-shell-escape", teX_file.name],
                       cwd=tmp_dir, timeout=timeout, env=_tex_env())
        if rc != 0:
            # pdflatex writes the log even on error — log the tail for help
            # Item 10: pdflatex's error output can echo source lines from the
            # resume document - keep it out of logs.
            logger.error("pdflatex failed (rc=%s); compile output not logged.", rc)
            return None

        produced = tmp_dir / "resume.pdf"
        if not produced.exists():
            logger.error("pdflatex reported success but no resume.pdf produced (rc=%s); output not logged.", rc)
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
