"""Text-layer verification for the shipped PDFs - deterministic.

The exporter has always trusted the renderer: it counts pages, compacts
overflow, and the layout tests measure a fixture at build time. The PDF an ATS
actually parses is never read back. That blind spot has two failure classes:

* a page whose fonts carry no text layer (or a broken CID mapping) extracts as
  ``(cid:123)`` noise or nothing, so a human-readable resume scores zero;
* a date printed with a non-ASCII dash. LaTeX turns ``--`` into an en dash, and
  Workday's import drops such a date along with its column (MIT ai-job-search,
  ``05-cv-templates.md``: "Date fields must be ASCII ranges"). ``_force_ascii``
  maps ``\u2014`` to ``--`` rather than to ``-``, so the parser/date fix in
  ``latex_render`` is the root-cause repair; this module is the shipped-output
  check that proves it.

``verify_pdf`` returns a flat list of checks for the result payload and the
Integrity tab. It is advisory by contract: a failed check never blocks a run
or deletes an artifact.

Adapted from ``tools/verify_pdf.py`` in the MIT-licensed ai-job-search
repository (Copyright (c) 2026 Mads Lorentzen); see NOTICE.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader

# --- ASCII-date rule: ported from tools/verify_pdf.py in the MIT
# ai-job-search repository (see NOTICE), including its quantity carve-outs.
# A year joined to a non-ASCII dash, or a dash joined to a year/Present.
# Scanned on the RAW layer: folding would hide exactly what this check exists
# to catch.
_NON_ASCII_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
# Dash-like code points a year must never be joined to in a date range. U+2013 is
# what LaTeX makes from `--` and what the documented Workday import dropped; the
# rest are the other Unicode dashes and the minus sign, which a parser that splits
# a range only on U+002D treats the same way.
_YEAR = r"(?<!\d)(?:19|20)\d{2}(?!\d)"
# Horizontal whitespace only between the year and the dash: `\s*` also matched
# a newline, so a heading ending in an ASCII date joined a dash that merely
# opened the next line.
NON_ASCII_DATE_RANGE = re.compile(
    rf"{_YEAR}[^\S\n]*[{_NON_ASCII_DASHES}]|[{_NON_ASCII_DASHES}][^\S\n]*{_YEAR}"
)
# The whitespace-delimited tokens touching the dash on either side. A range whose
# two ends are both BARE numbers is a date only when both are years: `EUR 1500-2000`
# and `2000-5000 requests/s` are quantities that happen to contain a year-shaped
# number, and the guide tells users to keep `--` in exactly those. A composite
# token (`03/2016`, `01.2016`, `2016-03`, `2016/17`) is not a bare number, so a
# numerically formatted date range still falls through to the year rule.
_TOKEN_BEFORE = re.compile(r"(\S+)[^\S\n]*$")
_TOKEN_AFTER = re.compile(r"^[^\S\n]*(\S+)")
_BARE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*[kKmM%]?")
_BARE_YEAR = re.compile(r"(?:19|20)\d{2}")
_OUTER_PUNCTUATION = ".,;:!?()[]{}\"'"
_TWO_DIGIT_YEAR = re.compile(r"\d{2}")


def _bare_number(token: str):
    """The token with outer punctuation stripped, if it is a bare number, else None.

    A dotted or comma-separated token whose groups include a year (`01.2016`,
    `31.07.2024`) is a date written numerically, not a quantity, and is left to
    the year rule; `1,234.56` and `2.0M` have no year group and stay numbers.
    """
    core = token.strip(_OUTER_PUNCTUATION)
    if not _BARE_NUMBER.fullmatch(core):
        return None
    groups = re.split(r"[.,]", core.rstrip("kKmM%"))
    if len(groups) > 1 and any(_BARE_YEAR.fullmatch(g) for g in groups):
        return None
    return core


def _currency_amount(token: str):
    """A numeric amount with an attached Unicode currency symbol, or None."""
    core = token.strip(_OUTER_PUNCTUATION)
    if core and unicodedata.category(core[0]) == "Sc":
        core = core[1:]
    elif core and unicodedata.category(core[-1]) == "Sc":
        core = core[:-1]
    else:
        return None
    return core if _BARE_NUMBER.fullmatch(core) else None


def _quantity_range(line: str, dash_index: int) -> bool:
    """True when both sides of the dash are bare numbers and not a date pair.

    A date pair is two years (`2016-2024`) or a year and a two-digit year
    (`2016-17`, `2019-20`): the short form is how academic years and seasons
    are written, and it was caught before the quantity rule existed. Anything
    else with bare numbers on both sides is a quantity (`1500-2000`,
    `2000-5000`) and is left alone. An attached currency symbol makes a
    numeric pair an amount range, even when both amounts resemble years.
    """
    before = _TOKEN_BEFORE.search(line[:dash_index])
    after = _TOKEN_AFTER.search(line[dash_index + 1:])
    if not before or not after:
        return False
    left_money, right_money = (_currency_amount(before.group(1)),
                               _currency_amount(after.group(1)))
    left = left_money or _bare_number(before.group(1))
    right = right_money or _bare_number(after.group(1))
    if left_money is not None or right_money is not None:
        return left is not None and right is not None
    if left is None or right is None:
        return False
    if (_BARE_YEAR.fullmatch(left)
            and (_BARE_YEAR.fullmatch(right)
                 or _TWO_DIGIT_YEAR.fullmatch(right))):
        return False
    return True


def find_non_ascii_date_ranges(text: str) -> list[tuple[str, str]]:
    """Return (line, dash) for every year joined to a non-ASCII dash in raw text.

    Works on the raw text layer, never on the folded output: the fold maps
    U+2013 back to `-` so `--`-based date ranges read as ASCII. A year on
    either side of the dash is enough ("Mar 2016 - Jul 2016", "2016 - Present"),
    so the check is locale-agnostic. A range whose two ends are both bare
    numbers is a date only when both are years: "EUR 600k-1M",
    "EUR 1500-2000" and "2000-5000 requests/s" are quantities and are left
    alone, while a numerically formatted date ("03/2016-07/2024", "2016-03-2024-07")
    is not a bare number on either side and is still caught.
    """
    hits = []
    for match in NON_ASCII_DATE_RANGE.finditer(text or ""):
        dash = next(c for c in match.group(0) if c in _NON_ASCII_DASHES)
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line_end = line_end if line_end != -1 else len(text)
        line = text[line_start:line_end]
        dash_index = match.start() + match.group(0).index(dash) - line_start
        if _quantity_range(line, dash_index):
            continue
        hits.append((" ".join(line.split()), dash))
    return hits


# Comparison-time fold: NFC, the ligatures LaTeX emits, typographic quotes,
# and every dash to ASCII. The layer stays raw for the ASCII-date check.
_FOLD = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
})
_WS_RE = re.compile(r"\s+")


_ACCENT_TRANSLIT = str.maketrans({
    "\u00df": "ss", "\u00e6": "ae", "\u00c6": "ae",
    "\u00f8": "o", "\u00d8": "o", "\u0142": "l", "\u0141": "l",
    "\u0111": "d", "\u0110": "d", "\u0153": "oe", "\u0152": "oe",
})


def _fold_ascii_accents(text: str) -> str:
    """Strip combining diacritics for accent-tolerant comparison against ASCII PDFs."""
    t = (text or "").translate(_ACCENT_TRANSLIT)
    decomposed = unicodedata.normalize("NFD", t)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def normalize_text(text: str, *, fold_accents: bool = False) -> str:
    """Fold a string for comparison only - never for the ASCII-date scan."""
    text = unicodedata.normalize("NFC", text or "").translate(_FOLD)
    if fold_accents:
        text = _fold_ascii_accents(text)
    return _WS_RE.sub(" ", text).strip().casefold()


def check_pdf_geometry(path: str, *, margin_slack_pt: float = 2.0,
                       sheet_edge_slack_pt: float = 12.0,
                       bottom_clip_pt: float = 6.0) -> dict:
    """Verify word bounding boxes stay within page margins, physical edges,
    and do not horizontally overlap on the same printed row."""
    file_path = Path(str(path))
    if not file_path.is_file():
        return {"ok": True, "detail": "geometry skipped (no file)", "violations": []}
    try:
        import pdfplumber
    except ImportError:
        return {"ok": True, "detail": "geometry skipped (pdfplumber unavailable)", "violations": []}

    violations: list[str] = []
    try:
        with pdfplumber.open(str(file_path)) as doc:
            for pidx, page in enumerate(doc.pages, 1):
                w_pt = float(page.width)
                h_pt = float(page.height)
                words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
                if not words:
                    continue

                rules = [
                    r for r in ((page.lines or []) + (page.rects or []))
                    if abs(float(r["top"]) - float(r["bottom"])) < 1.5
                    and (float(r["x1"]) - float(r["x0"])) > w_pt * 0.45
                ]
                rule_limit = (max(max(float(r["x1"]) for r in rules), w_pt - 28.8) + margin_slack_pt
                              if rules else w_pt - 18.0)
                right_limit = min(w_pt - sheet_edge_slack_pt, rule_limit)
                bottom_limit = h_pt - bottom_clip_pt

                for w in words:
                    x1 = float(w["x1"])
                    bot = float(w["bottom"])
                    txt = str(w.get("text") or "")[:30]
                    if x1 > w_pt + 0.5:
                        violations.append(
                            f"p{pidx} off right sheet edge ({x1:.1f}>{w_pt:.0f}pt): {txt!r}")
                    elif x1 > right_limit:
                        violations.append(
                            f"p{pidx} past right margin ({x1:.1f}>{right_limit:.1f}pt): {txt!r}")
                    if bot > bottom_limit:
                        violations.append(
                            f"p{pidx} bottom clip ({bot:.1f}>{bottom_limit:.1f}pt): {txt!r}")

                # Detect symmetric bounding-box collisions on vertical-overlapping words
                sorted_words = sorted(words, key=lambda w: float(w["top"]))
                for i in range(len(sorted_words)):
                    a = sorted_words[i]
                    a_top, a_bot = float(a["top"]), float(a["bottom"])
                    a_x0, a_x1 = float(a["x0"]), float(a["x1"])
                    for j in range(i + 1, len(sorted_words)):
                        b = sorted_words[j]
                        b_top, b_bot = float(b["top"]), float(b["bottom"])
                        if b_top >= a_bot - 2.0:
                            break
                        y_ov = min(a_bot, b_bot) - max(a_top, b_top)
                        x_ov = min(a_x1, float(b["x1"])) - max(a_x0, float(b["x0"]))
                        if y_ov > 2.5 and x_ov > 2.0:
                            violations.append(
                                f"p{pidx} text collision: {a['text']!r} ({a_x0:.1f}-{a_x1:.1f}) overlaps {b['text']!r} ({float(b['x0']):.1f}-{float(b['x1']):.1f})")
    except Exception as exc:
        return {"ok": False, "detail": f"geometry measurement failed: {exc}", "violations": [str(exc)]}

    if violations:
        return {
            "ok": False,
            "detail": "; ".join(violations[:4]) + (f" (+{len(violations) - 4} more)" if len(violations) > 4 else ""),
            "violations": violations,
        }
    return {"ok": True, "detail": "all glyphs within page margins and non-overlapping", "violations": []}


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def verify_pdf(path: str, expected_pages: int | None = 1,
               required_literals: tuple[str, ...] | list[str] = ()) -> dict:
    """Read a PDF back and verify what an ATS parser would see.

    Returns ``{"pages": int, "ok": bool, "checks": [{"name", "ok", "detail"}]}``.
    Checks: readable, text_layer, page_count, text_integrity, ascii_dates, and
    required_text (only when literals are supplied). ``expected_pages=None``
    accepts any non-zero count. Never raises - an unreadable file is a failed
    check, not a pipeline error.
    """
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        texts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        return {"pages": 0, "ok": False, "checks": [
            _check("readable", False, f"PDF could not be read: {exc}"),
        ]}

    checks = [_check("readable", True, f"{pages} page(s)")]
    if pages == 0:
        checks.append(_check("text_layer", False, "PDF has no pages"))
    else:
        empty = [str(i + 1) for i, text in enumerate(texts) if not text.strip()]
        detail = ("every page has extractable text" if not empty
                  else "no extractable text on page(s) " + ", ".join(empty))
        checks.append(_check("text_layer", not empty, detail))

    if expected_pages is None:
        checks.append(_check("page_count", pages > 0, f"{pages} page(s)"))
    else:
        checks.append(_check(
            "page_count", pages == expected_pages,
            f"{pages} page(s), expected {expected_pages}"))

    raw = "\n".join(texts)
    damage = []
    if "(cid:" in raw:
        damage.append("CID font runs (\"(cid:\")")
    if "\ufffd" in raw:
        damage.append("U+FFFD replacement characters")
    checks.append(_check(
        "text_integrity", not damage,
        "no CID/replacement damage" if not damage
        else "damaged text: " + ", ".join(damage)))

    hits = find_non_ascii_date_ranges(raw)
    checks.append(_check(
        "ascii_dates", not hits,
        "dates use ASCII separators" if not hits
        else "non-ASCII date separator(s): " + "; ".join(
            f"U+{ord(dash):04X} in {line[:80]!r}" for line, dash in hits[:5])))

    literals = [str(lit).strip() for lit in required_literals or () if str(lit).strip()]
    if literals:
        haystack = normalize_text(raw)
        folded_haystack = normalize_text(raw, fold_accents=True)
        missing = [
            lit for lit in literals
            if normalize_text(lit) not in haystack
            and normalize_text(lit, fold_accents=True) not in folded_haystack
        ]
        checks.append(_check(
            "required_text", not missing,
            "required text present" if not missing
            else "missing from the text layer: " + ", ".join(missing[:6])))

    if raw.strip():
        geom = check_pdf_geometry(path)
        checks.append(_check("geometry", geom["ok"], geom["detail"]))

    return {"pages": pages,
            "ok": all(check["ok"] for check in checks),
            "checks": checks}
