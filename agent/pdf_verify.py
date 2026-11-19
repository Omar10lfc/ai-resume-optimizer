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

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

# The typographic dashes LaTeX produces from `--`, `---`, and `$-$`.
_NON_ASCII_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
# A year joined to a non-ASCII dash, or a dash joined to a year/Present.
# Scanned on the RAW layer: folding would hide exactly what this check exists
# to catch.
_DATE_DASH_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*[" + _NON_ASCII_DASHES + r"]"
    r"|[" + _NON_ASCII_DASHES + r"]\s*"
    r"(?:(?:[A-Za-z]{3,9}\.?\s+)?(?:19|20)\d{2}\b|Present\b)",
    re.IGNORECASE,
)

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


def normalize_text(text: str) -> str:
    """Fold a string for comparison only - never for the ASCII-date scan."""
    text = unicodedata.normalize("NFC", text or "").translate(_FOLD)
    return _WS_RE.sub(" ", text).strip().casefold()


def _date_dash_hits(text: str) -> list[str]:
    """Every year/dash or dash/year hit in the raw layer, in document order."""
    return [m.group(0) for m in _DATE_DASH_RE.finditer(text or "")]


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

    hits = _date_dash_hits(raw)
    checks.append(_check(
        "ascii_dates", not hits,
        "dates use ASCII separators" if not hits
        else "non-ASCII date separator(s): " + "; ".join(hits[:5])))

    literals = [str(lit).strip() for lit in required_literals or () if str(lit).strip()]
    if literals:
        haystack = normalize_text(raw)
        missing = [lit for lit in literals if normalize_text(lit) not in haystack]
        checks.append(_check(
            "required_text", not missing,
            "required text present" if not missing
            else "missing from the text layer: " + ", ".join(missing[:6])))

    return {"pages": pages,
            "ok": all(check["ok"] for check in checks),
            "checks": checks}
