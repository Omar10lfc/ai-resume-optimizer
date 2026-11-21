"""Layout measurement for generated PDFs - the measurement half of a
compile-and-inspect loop, adapted from ``tools/verify_layout.py`` in the
MIT-licensed ai-job-search repository (Copyright (c) 2026 Mads Lorentzen);
see NOTICE.

The exporter's last resort is the auto-compact pass: it re-renders to buy one
page. Measurement tells the compacting what broke so the fixes can be aimed:

* orphaned entry    A job entry (a LaTeX/moderncv ``\\cventry`` in their
                    calibration) moves to the next page, or its header lands
                    at the bottom of one page with the body resuming on the
                    next ("the most common failure" per their docstring).
* internal hole     The space an ejected entry leaves behind mid-page.
* page ends early   A non-final page stopping well short of the bottom.
* final page thin   A last page mostly empty, reading as unfinished.
* footer collision  Body text pushed into the page-number band (the usual
                    result of ``\\enlargethispage`` or a negative ``\\vspace``).

Output is ADVISORY (pipeline rows for the Integrity tab), never blocking.
The thresholds below are calibrated for the stock moderncv/cover.cls
geometry; the LaTeX classes this product renders differ, so read rows, not
exit codes - a hole of 101pt under another template is noise.

Geometry comes from Poppler ``pdftotext -bbox`` (see docstring there for the
Git-for-Windows xpdf/-bbox quirk); if no ``pdftotext`` on PATH has the ``-bbox``
flag, ``measure_pdf_layout()`` returns ``[]`` (skipped). Works on the raw
word bounding boxes, never on extraction.

Unit tests build synthetic ``Page``/``Line`` structures (tests/test_verify_layout.py);
none need Poppler or a compiled PDF.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# A gap larger than this between consecutive lines is a hole, not spacing.
GAP_LIMIT_PT = 100.0
# Bottom whitespace on a page that is not the last one. The stock geometry leaves ~85pt.
BOTTOM_LIMIT_FRACTION = 0.25
# A final page emptier than this reads as an unfinished document.
LAST_PAGE_THIN_FRACTION = 0.35
# The page-number footer lives in the bottom margin: ignore this band when
# measuring the body, or every page looks like it has a hole above its footer.
FOOTER_BAND_PT = 90.0
# A line indented at least this far past the page's left edge is a bullet or
# a continuation, not an entry header or section heading.
INDENT_PT = 8.0
# A template may merge a list marker into the line it labels; Set-in limit
# (reference SET_IN_LIMIT_PT): lines set right/centred are headers or dates,
# not body where a page resumes.
SET_IN_LIMIT_PT = 60.0
# A line this much taller than the body median is a section heading.
HEADING_HEIGHT_RATIO = 1.25
# Lines whose tops are this close are parts of one printed line.
SAME_PRINTED_LINE_PT = 3.0

LIST_MARKERS = frozenset("\u2022\u25aa\u25cf\u00b7\u30fb\u00b7")

PAGE_RE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)
WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)">([^<]*)</word>'
)
TEXT_RE = re.compile(r"\w")


@dataclass
class Line:
    top: float
    bottom: float
    left: float
    height: float
    text: str
    # Where the text starts once leading LIST_MARKERS words are set aside;
    # None when the line has no such marker.
    text_left: float | None = None

    @property
    def text_x(self) -> float:
        return self.left if self.text_left is None else self.text_left


@dataclass
class Page:
    height: float
    lines: list[Line]

    def __post_init__(self):
        self.lines = sorted(self.lines, key=lambda l: l.top)

    @property
    def body(self) -> list[Line]:
        cutoff = self.height - FOOTER_BAND_PT
        return [l for l in self.lines if l.top < cutoff]

    @property
    def empty(self) -> bool:
        return not self.body

    @property
    def bottom_space(self) -> float:
        return self.height - max(l.bottom for l in self.body) if self.body else self.height

    @property
    def left_edge(self) -> float:
        return min(l.left for l in self.body) if self.body else 0.0

    @property
    def body_median_height(self) -> float:
        heights = sorted(l.height for l in self.body)
        return heights[len(heights) // 2] if heights else 0.0

    @property
    def footer_crowded(self) -> bool:
        """One line in the bottom band is a page number; two means body text spilled in."""
        band = self.height - FOOTER_BAND_PT
        return len({round(l.top, 1) for l in self.lines if l.top >= band}) > 1

    def is_indented(self, line: Line) -> bool:
        return line.left > self.left_edge + INDENT_PT

    def is_heading(self, line: Line) -> bool:
        median = self.body_median_height
        return bool(median) and line.height > median * HEADING_HEIGHT_RATIO

    def largest_gap(self) -> tuple[float, float]:
        """Largest top-to-top distance between body lines, and where it starts.

        Measured top-to-top rather than bottom-to-top, so a tall line inflates the gap
        by its own height (a 160pt void under a heading reads as ~174pt). That errs
        toward over-detection, which is the right direction for a check whose job is
        to stop a hole from shipping.
        """
        tops = sorted({round(l.top, 1) for l in self.body})
        if len(tops) < 2:
            return (0.0, 0.0)
        return max((tops[i + 1] - tops[i], tops[i]) for i in range(len(tops) - 1))


@cache
def find_pdftotext() -> str | None:
    """Find a pdftotext binary that supports Poppler's -bbox flag.

    Git for Windows prepends Git's mingw64\\bin to PATH under Git Bash
    (Claude Code's terminal), shipping an xpdf pdftotext without -bbox. If a
    working Poppler is installed further down PATH, probe each candidate
    with -h and pick the first that advertises -bbox. Falls back to the
    first candidate when none advertises -bbox, so callers keep today's
    skip message on failure. Returns None when no pdftotext is found.

    Cached so the parse probes don't re-run with subprocesses per call.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for entry in os.get_exec_path():
        candidate = shutil.which("pdftotext", path=entry)
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    if not candidates:
        return None
    for candidate in candidates:
        try:
            res = subprocess.run(
                [candidate, "-h"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = (res.stdout or "") + (res.stderr or "")
            if "-bbox" in output:
                return candidate
        except (subprocess.SubprocessError, OSError):
            continue
    return candidates[0]


def parse_pdf(path: Path) -> list[Page]:
    binary = find_pdftotext()
    if not binary:
        raise RuntimeError("pdftotext (Poppler) not found; install poppler-utils")
    try:
        out = subprocess.run(
            [binary, "-bbox", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            text=True,
            # pdftotext emits UTF-8; without this Windows decodes it as cp1252
            # and a non-ASCII glyph in the resume crashes the run.
            encoding="utf-8",
            errors="replace",
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        stderr_lines = (exc.stderr or "").strip().splitlines()
        detail = stderr_lines[0] if stderr_lines else f"exit {exc.returncode}"
        raise RuntimeError(
            f"pdftotext could not produce bounding boxes for {path} ({detail})"
        ) from exc
    pages = []
    for _w, h, body in PAGE_RE.findall(out):
        buckets: dict[float, list[tuple[float, float, float, str]]] = {}
        for x_min, y_min, y_max, text in WORD_RE.findall(body):
            key = round(float(y_min), 0)  # words on one line share a rounded yMin
            buckets.setdefault(key, []).append((float(x_min), float(y_min), float(y_max), text))
        lines = [
            Line(
                top=min(w[1] for w in words),
                bottom=max(w[2] for w in words),
                left=min(w[0] for w in words),
                height=max(w[2] - w[1] for w in words),
                text=" ".join(w[3] for w in sorted(words)),
                text_left=text_start(sorted(words)),
            )
            for words in buckets.values()
        ]
        pages.append(Page(float(h), lines))
    return pages


def text_start(words: list[tuple[float, float, float, str]]) -> float | None:
    """xMin of the first word after leading LIST_MARKERS words, if there were any."""
    lead = 0
    while lead < len(words) and words[lead][3] and set(words[lead][3]) <= LIST_MARKERS:
        lead += 1
    return words[lead][0] if 0 < lead < len(words) else None


def printed_text_x(lines: list[Line]) -> float | None:
    """Where the text of one printed line starts, from the bbox lines that make it up."""
    xs = [l.text_x for l in lines if TEXT_RE.search(l.text)]
    return min(xs) if xs else None


def find_orphans(pages: list[Page]) -> list[tuple[str, str]]:
    """Entry/heading text that does not travel with the rest of its content
    across a page break. Returns (kind, message) rows.

    Two shapes, both documented failures:
      * the last body line of a page is a section heading (stranded heading)
      * the next page's text starts further in than the text of the page's
        last printed line (an entry header), i.e. the entry was split across
        the break

    The second compares where the TEXT starts, not where the line starts -
    moderncv may merge a list marker into the line it labels, which moves the
    line's left edge but not its text.
    """
    problems = []
    body_lines = [l for p in pages for l in p.body]
    if not body_lines:
        return problems
    doc_left = min(l.left for l in body_lines)

    def indented(line: Line) -> bool:
        return line.left > doc_left + INDENT_PT

    for i in range(len(pages) - 1):
        here, nxt = pages[i], pages[i + 1]
        if here.empty or nxt.empty:
            continue
        last, first = here.body[-1], nxt.body[0]

        if here.is_heading(last):
            problems.append((
                "stranded_heading",
                f"p{i + 1} ends on the section heading {last.text.strip()!r} with its "
                f"content on p{i + 2}. Shorten the entry that follows it, or let the "
                "heading and its first entry move to the next page together",
            ))
            continue
        # The last printed line: the bbox lines within SAME_PRINTED_LINE_PT of the last one.
        tail = [l for l in here.body if l.top >= last.top - SAME_PRINTED_LINE_PT]
        last_x = printed_text_x(tail)
        if last_x is None:
            # A lone marker line: moderncv puts an itemize marker in its own bbox
            # line at the list's left edge, so a list item split across the
            # break looks like an un-indented header followed by indented text.
            if not indented(last) and indented(first):
                problems.append((
                    "split_list_marker",
                    f"p{i + 1} ends on a lone list marker whose text continues on "
                    f"p{i + 2} ({first.text.strip()[:60]!r}): a bullet is split "
                    "across the page break. Shorten the preceding content so the "
                    "whole item fits on one page",
                ))
            continue
        # The next page's first printed line with text, skipping a marker in
        # its own line and a line set right or centred.
        opener = next(
            (l for l in nxt.body
             if TEXT_RE.search(l.text) and l.text_x <= doc_left + SET_IN_LIMIT_PT),
            None,
        )
        if opener is None:
            continue
        head = [l for l in nxt.body if abs(l.top - opener.top) <= SAME_PRINTED_LINE_PT]
        next_x = printed_text_x(head)
        if next_x is not None and next_x > last_x + INDENT_PT:
            shown = max((l for l in tail if TEXT_RE.search(l.text)),
                        key=lambda l: len(l.text))
            problems.append((
                "orphaned_entry_header",
                f"p{i + 1} ends on {shown.text.strip()[:60]!r} while the text on "
                f"p{i + 2} starts further in ({opener.text.strip()[:60]!r}): an entry "
                "header is orphaned from its body. Add \\needspace before that entry, "
                "or shorten it",
            ))
    return problems


def report(path: Path, pages: list[Page]) -> list[dict]:
    """Advisory layout rows for ``pages`` of ``path``: {kind, detail}."""
    rows: list[dict] = []
    for i, page in enumerate(pages, 1):
        if page.empty:
            rows.append({"kind": "empty_page",
                         "detail": f"p{i} contains no text"})
            continue

        gap, gap_y = page.largest_gap()
        share = page.bottom_space / page.height
        if gap > GAP_LIMIT_PT:
            rows.append({"kind": "hole",
                         "detail": f"p{i} has a {gap:.0f}pt hole at y{gap_y:.0f} "
                                   f"(~{gap / 14:.0f} blank lines). A LaTeX entry is an "
                                   "unbreakable tabular: shorten the entry that follows "
                                   "the hole so it fits, or move a shorter section above it"})
        if i < len(pages) and share > BOTTOM_LIMIT_FRACTION:
            rows.append({"kind": "early_page_end",
                         "detail": f"p{i} ends {page.bottom_space:.0f}pt ({share * 100:.0f}%) "
                                   "early although more pages follow - reads as a broken page break"})
        if page.footer_crowded:
            rows.append({"kind": "footer_collision",
                         "detail": f"p{i} has body text inside the bottom margin band, "
                                   "colliding with the footer; stop stretching the page "
                                   "with \\enlargethispage and cut content"})
        if i == len(pages) > 1 and share > LAST_PAGE_THIN_FRACTION:
            rows.append({"kind": "thin_final_page",
                         "detail": f"p{i} is the last page and {share * 100:.0f}% empty - "
                                   "unfinished-looking; restore the highest-relevance "
                                   "content previously cut"})
    rows.extend({"kind": kind, "detail": msg} for kind, msg in find_orphans(pages))
    return rows


def measure_pdf_layout(path) -> list[dict]:
    """Advisory layout rows for a compiled PDF, or [] when skipped.

    Returns [] when pdftotext is unavailable or the PDF has no pages - a
    measurement problem is itself an advisory row only when we can measure;
    on skip we stay quiet like the other advisory channels do.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        pages = parse_pdf(path)
    except (RuntimeError, subprocess.SubprocessError, OSError):
        return []
    if not pages:
        return []
    return report(path, pages)


if __name__ == "__main__":
    import sys
    for row in measure_pdf_layout(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(row["kind"], "-", row["detail"])
