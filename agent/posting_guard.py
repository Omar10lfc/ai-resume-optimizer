"""Deterministic integrity gate for third-party job postings.

A posting is untrusted text this pipeline reads beside the candidate's personal
data. ``_untrusted`` delimits it in prompts, but delimiters do not sanitize
what a human never sees. This module removes the payload classes and reports
what was found, before the scanner, the ATS engine, or any prompt reads the
text:

* hidden HTML - elements collapsed or clipped by CSS, or marked ``hidden`` /
  ``aria-hidden`` - is stripped and its text kept as an excerpt;
* comments and ``script`` / ``style`` / ``noscript`` / ``svg`` / ``template``
  blocks are stripped (every real page carries them, so they are not
  reported);
* invisible characters - zero-width, bidi controls, soft hyphens, BOM, the
  Unicode tag block (an invisible ASCII alphabet), variation-selector runs -
  are removed; tag/variation runs are decoded so what they spell is quoted;
* instruction-shaped lines, URLs that only existed in hidden content, and
  words mixing Latin with Cyrillic/Greek are reported.

Removal is the defense. Findings are advisory: the gate never blocks a run,
and it never raises - an internal failure returns the input unchanged.

Adapted from ``tools/posting_guard.py`` in the MIT-licensed ai-job-search
repository (Copyright (c) 2026 Mads Lorentzen); see NOTICE.
"""
from __future__ import annotations

import html
import re
from collections import Counter

from bs4 import BeautifulSoup, Comment

# Excerpts are terminal/UI quotes, not documents: keep them short enough to
# scan in a list. Per-kind caps keep a hostile posting from bloating the
# result payload with thousands of one-line findings.
_MAX_EXCERPT = 200
_MAX_PER_KIND = 8

# --- Invisible character classes -------------------------------------------
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064"
_BIDI = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
# Overrides and isolates are the tampering signals; the embedding marks are
# counted but not flagged (legitimate RTL text uses them).
_BIDI_TAMPER = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
_FORMAT = "\u00ad\u2028\u2029"  # soft hyphen, line/paragraph separators
_BOM = "\ufeff"
_TAG_RUN = re.compile("[\U000e0000-\U000e007f]+")
_VARIATION = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]")
_VARIATION_RUN = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]{2,}")
# The one legitimate use of tag characters: the England, Scotland and Wales
# flag emoji (black flag + tag letters + cancel tag). Matched in full and left
# intact; any other tag run is an invisible-ASCII payload.
_RGI_TAG_FLAG = re.compile(
    "\U0001f3f4\U000e0067\U000e0062(?:"
    "\U000e0065\U000e006e\U000e0067"
    "|\U000e0073\U000e0063\U000e0074"
    "|\U000e0077\U000e006c\U000e0073)\U000e007f"
)

# --- Hidden HTML ------------------------------------------------------------
# A browser decodes entities and ignores CSS comments inside a declaration, so
# ``display&#58;none`` and ``dis/**/play:none`` hide content while a naive
# regex sees neither. Both are normalised before matching.
_HIDDEN_STYLE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|opacity\s*:\s*0(?![.\d]*[1-9])"
    r"|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?(?![\d.])"
    r"|color\s*:\s*transparent"
    r"|clip\s*:\s*rect\(\s*0"
    r"|clip-path\s*:\s*inset\(\s*100%"
    r"|(?:(?:margin-)?(?:left|top)|text-indent)\s*:\s*-\d{4,}px)",
    re.IGNORECASE,
)
# A zero-height/width box with overflow clipped hides its content as surely as
# display:none. Either half alone is everyday layout CSS, so both are required.
_COLLAPSED_BOX = re.compile(
    r"\b(?:max-)?(?:height|width)\s*:\s*0(?:\.0+)?(?:px|em|rem|%|pt)?(?![\d.])",
    re.IGNORECASE,
)
_CLIPPED = re.compile(r"\boverflow(?:-[xy])?\s*:\s*hidden\b", re.IGNORECASE)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DROP_TAGS = ("script", "style", "noscript", "svg", "template")

# --- Detection --------------------------------------------------------------
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
# Narrow on purpose: every pattern targets an agent, not a reader. The few
# direct shell/config names (`curl|wget|ssh|scp`, `.env`, `settings.json`,
# `allowed-tools`) are rarer in honest postings than the patterns above; when
# they do appear in ordinary prose (an "SSH administrator" role, a team page
# that mentions `.env`), the quote is what lets the user dismiss the flag in a
# second - same as `ignore`-shaped lines.
_INSTRUCTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bignore (?:all |any |the |your )?(?:previous|prior|above|earlier)"
    r" (?:instructions?|prompts?|directions?|rules?)\b",
    r"\bignore (?:all|any|the|previous|prior|above|earlier)\b",
    r"\b(?:disregard|forget) (?:all |any |the |your |my )?"
    r"(?:previous|prior|above|earlier|preceding|initial|original|system)"
    r" (?:instructions?|prompts?)\b",
    r"\byou are (?:an?|the) (?:ai|assistant|llm|language model"
    r"|ai (?:assistant|agent|model|language model))\b",
    r"\bsystem prompt\b",
    r"\b(?:do not|don['\u2019]?t|never) (?:tell|inform) the user\b(?!['\u2019]s)",
    r"\b(?:do not|don['\u2019]?t|never) (?:mention|reveal|disclose)"
    r" (?:this|these|that|it|the (?:above|following|previous))\b"
    r"[^.\n]{0,30}\bto the user\b",
    r"\b(?:run|execute) the following"
    r" (?:command|commands|script|code|shell|instruction|instructions|steps)\b",
    r"\b(?:curl|wget|ssh|scp)\b",
    r"\.env\b",
    r"settings\.json",
    r"allowed-tools",
    r"^\s*(?:assistant|system|user)\s*:",
    r"<\|",
    r"\[INST\]",
    r"###\s*(?:system|instruction)",
    r"\btool_use\b|\bfunction_call\b",
))
_WORD = re.compile(r"\w+", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")
_CONFUSABLE = re.compile(r"[\u0400-\u04ff\u0370-\u03ff]")

_KIND_LABELS = {
    "hidden_html": ("hidden block", "hidden blocks"),
    "invisible_unicode": ("invisible-character signal",
                          "invisible-character signals"),
    "instruction_line": ("instruction-shaped line", "instruction-shaped lines"),
    "hidden_url": ("hidden URL", "hidden URLs"),
    "homoglyph": ("mixed-script word", "mixed-script words"),
}


def _fold(text: str) -> str:
    """Collapse whitespace the way the rest of the pipeline consumes it."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\xa0\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _excerpt(line: str, at: int = 0) -> str:
    """Up to _MAX_EXCERPT characters of ``line``, always including ``at``.

    A fixed prefix hides the trigger when it sits later in a long paragraph -
    and the quote is what lets the user dismiss a flag in a second.
    """
    if len(line) <= _MAX_EXCERPT:
        return line
    lead = _MAX_EXCERPT // 3
    start = max(0, at - lead)
    end = min(len(line), start + _MAX_EXCERPT)
    start = max(0, end - _MAX_EXCERPT)
    head = "..." if start else ""
    tail = "..." if end < len(line) else ""
    return f"{head}{line[start:end]}{tail}"


def _clean_invisible(text: str) -> tuple[str, dict]:
    """Remove invisible character classes; decode what the payloads spell.

    Returns ``(cleaned, info)`` where info holds the per-category counts, the
    count of tampering signals (a zero-width/BOM character between two
    letters, or a bidi override/isolate), and the decoded strings. The
    RGI flag emoji are preserved; every other tag character is removed.
    """
    counts: dict[str, int] = {}
    for key, chars in (("zero_width", _ZERO_WIDTH), ("bidi", _BIDI),
                       ("format", _FORMAT), ("bom", _BOM)):
        n = sum(text.count(ch) for ch in chars)
        if n:
            counts[key] = n

    flags = _RGI_TAG_FLAG.findall(text)
    unflagged = _RGI_TAG_FLAG.sub("", text)
    tags = sum(1 for ch in unflagged if "\U000e0000" <= ch <= "\U000e007f")
    if tags:
        counts["tag"] = tags
    variation = len(_VARIATION.findall(text))
    if variation:
        counts["variation"] = variation

    flagged = 0
    for i, ch in enumerate(text):
        if ch in _BIDI_TAMPER:
            flagged += 1
        elif ch in _ZERO_WIDTH or ch == _BOM:
            before = text[i - 1] if i else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if before.isalpha() and after.isalpha():
                flagged += 1

    decoded: list[str] = []
    for run in _TAG_RUN.findall(unflagged):
        spelled = "".join(
            chr(ord(ch) - 0xE0000) for ch in run
            if 0x20 <= ord(ch) - 0xE0000 < 0x7F).strip()
        if spelled:
            decoded.append(spelled[:_MAX_EXCERPT])
    for run in _VARIATION_RUN.findall(text):
        data = bytes(
            ord(ch) - 0xFE00 if ord(ch) <= 0xFE0F
            else ord(ch) - 0xE0100 + 16
            for ch in run)
        spelled = "".join(
            ch for ch in data.decode("utf-8", errors="replace")
            if ch.isprintable()).strip()
        if spelled:
            decoded.append(spelled[:_MAX_EXCERPT])

    text = _VARIATION.sub("", text)
    if flags:
        # Strip around the flags so the visible emoji survive intact.
        strip = {ord(ch): None for ch in (
            _ZERO_WIDTH + _BIDI + _FORMAT + _BOM + "".join(
                chr(c) for c in range(0xE0000, 0xE0080)))}
        pieces: list[str] = []
        last = 0
        for match in _RGI_TAG_FLAG.finditer(text):
            pieces.append(text[last:match.start()].translate(strip))
            pieces.append(match.group(0))
            last = match.end()
        pieces.append(text[last:].translate(strip))
        text = "".join(pieces)
    else:
        text = text.translate({ord(ch): None for ch in (
            _ZERO_WIDTH + _BIDI + _FORMAT + _BOM + "".join(
                chr(c) for c in range(0xE0000, 0xE0080)))})
    return _fold(text), {"counts": counts, "flagged": flagged,
                         "decoded": decoded}


def _hidden_reason(tag) -> str | None:
    """Why a tag's content is invisible to a reader, or None when it is not."""
    attrs = tag.attrs or {}
    if "hidden" in attrs:
        return "hidden attribute"
    aria = attrs.get("aria-hidden")
    if isinstance(aria, list):
        aria = " ".join(aria)
    if str(aria or "").strip().lower() == "true":
        return "aria-hidden"
    style = _CSS_COMMENT.sub("", html.unescape(str(attrs.get("style") or "")))
    if _HIDDEN_STYLE.search(style):
        return "hidden by CSS"
    if _COLLAPSED_BOX.search(style) and _CLIPPED.search(style):
        return "collapsed box"
    return None


def _clean_html(markup: str) -> tuple[str, str, list[tuple[str, str]], list[str]]:
    """Visible text, hidden text, hidden blocks, and URLs found in hidden tags.

    BeautifulSoup (already a dependency) owns the parse, so malformed markup,
    implied end tags, and entity decoding follow the browser rules the posting
    was rendered with. Hidden elements are removed from the tree; nested
    hidden tags disappear with their parent.
    """
    soup = BeautifulSoup(markup, "html.parser")
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    hidden_blocks: list[tuple[str, str]] = []
    hidden_urls: list[str] = []
    for tag in soup.find_all(True):
        # A tag inside an already-decomposed parent is still in this snapshot;
        # its attrs are gone, so it contributes nothing.
        if getattr(tag, "decomposed", False):
            continue
        reason = _hidden_reason(tag)
        if not reason:
            continue
        block = _fold(tag.get_text(" ", strip=True))
        if block:
            hidden_blocks.append((reason, block))
        for attr in ("href", "src", "action"):
            value = tag.get(attr)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                hidden_urls.append(value)
        for url in _URL.findall(str(tag)):
            hidden_urls.append(url)
        tag.decompose()

    visible = _fold(soup.get_text("\n"))
    hidden_text = "\n".join(block for _, block in hidden_blocks)
    return visible, hidden_text, hidden_blocks, hidden_urls


def _scan(visible: str, hidden: str,
          hidden_urls: list[str]) -> list[dict]:
    """Instruction-shaped lines, hidden URLs, and mixed-script words."""
    findings: list[dict] = []
    seen: set[str] = set()
    for line in visible.splitlines() + hidden.splitlines():
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        match = next((m for m in (p.search(stripped)
                                  for p in _INSTRUCTION_PATTERNS) if m), None)
        if match:
            seen.add(stripped)
            findings.append({
                "kind": "instruction_line",
                "detail": "instruction-shaped line aimed at an agent",
                "excerpt": _excerpt(stripped, match.start()),
            })

    urls = [url for url in hidden_urls if url not in visible]
    urls += [url for url in _URL.findall(hidden) if url not in visible]
    for url in dict.fromkeys(urls):
        findings.append({
            "kind": "hidden_url",
            "detail": "URL present only in hidden content",
            "excerpt": _excerpt(url),
        })

    words = []
    for word in _WORD.findall(visible + "\n" + hidden):
        if _LATIN.search(word) and _CONFUSABLE.search(word):
            if word not in words:
                words.append(word)
    for word in words:
        findings.append({
            "kind": "homoglyph",
            "detail": "word mixes Latin with Cyrillic/Greek letters",
            "excerpt": _excerpt(word),
        })
    return findings


def scan_posting(text: str, is_html: bool = False) -> dict:
    """Clean a posting and report what was removed.

    Returns ``{"clean_text": str, "findings": list[dict]}``. ``findings`` rows
    carry ``kind``, a human ``detail``, and an ``excerpt``. Advisory only: the
    caller decides how to surface them, and no finding blocks a run. On any
    internal failure the input is returned unchanged with no findings - the
    gate must never turn a posting into a failed optimization.
    """
    if not text or not text.strip():
        return {"clean_text": text or "", "findings": []}
    try:
        hidden_urls: list[str] = []
        if is_html:
            visible, hidden_text, hidden_blocks, hidden_urls = _clean_html(text)
        else:
            visible, hidden_text, hidden_blocks = text, "", []

        visible, visible_info = _clean_invisible(visible)
        hidden_text, hidden_info = _clean_invisible(hidden_text)
        counts = dict(visible_info["counts"])
        for key, n in hidden_info["counts"].items():
            counts[key] = counts.get(key, 0) + n
        flagged = visible_info["flagged"] + hidden_info["flagged"]
        decoded = visible_info["decoded"] + hidden_info["decoded"]

        findings: list[dict] = []
        for reason, block in hidden_blocks[:_MAX_PER_KIND]:
            findings.append({
                "kind": "hidden_html",
                "detail": f"hidden content removed ({reason})",
                "excerpt": _excerpt(block),
            })
        if counts:
            detail = ", ".join(f"{key} {value}"
                               for key, value in sorted(counts.items()))
            findings.append({
                "kind": "invisible_unicode",
                "detail": f"invisible characters removed ({detail})",
                "excerpt": "",
            })
        if flagged:
            findings.append({
                "kind": "invisible_unicode",
                "detail": (f"{flagged} invisible character(s) inside words "
                           "- tampering signal"),
                "excerpt": "",
            })
        for spelled in decoded[:_MAX_PER_KIND]:
            findings.append({
                "kind": "invisible_unicode",
                "detail": "text spelled in invisible characters",
                "excerpt": _excerpt(spelled),
            })

        # What hidden characters spell is hidden content: scan it like the
        # hidden blocks so a tag-spelled instruction is quoted, not just
        # counted.
        scan_hidden = "\n".join([hidden_text, *decoded])
        scanned = _scan(visible, scan_hidden, hidden_urls)
        per_kind: Counter[str] = Counter()
        for finding in scanned:
            kind = finding["kind"]
            if per_kind[kind] >= _MAX_PER_KIND:
                continue
            per_kind[kind] += 1
            findings.append(finding)
        return {"clean_text": _fold(visible), "findings": findings}
    except Exception:
        return {"clean_text": text, "findings": []}


def summarize_findings(findings: list) -> str:
    """One human line for a warning banner: "2 hidden blocks, 1 hidden URL"."""
    counts = Counter(str(item.get("kind") or "") for item in findings or [])
    parts = []
    for kind, (one, many) in _KIND_LABELS.items():
        n = counts.get(kind, 0)
        if n:
            parts.append(f"{n} {one if n == 1 else many}")
    return ", ".join(parts)
