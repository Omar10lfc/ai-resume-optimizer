"""Shared utility functions: printing, truncation, text cleaning, PDF export."""
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Optional

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
    s = (text or "").strip()
    while s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline == -1:
            return ""
        s = s[first_newline + 1:]
        if s.endswith("```"):
            last_fence = s.rfind("```")
            s = s[:last_fence].rstrip()
        s = s.strip()
    return s


_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_MARKDOWN_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
_MARKDOWN_HEADING_RE = re.compile(r'^#{1,6}\s*')
_MARKDOWN_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_MARKDOWN_ITALIC_RE = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
_MARKDOWN_BULLET_RE = re.compile(r'^[-*+]\s+')


def _strip_markdown_for_review(text: str) -> str:
    """Flatten markdown to the plain text the RENDERED document shows.

    Reviewers grade the presentation of the document a recruiter sees; raw
    markdown syntax ('##', '**', '- [label](url)') never appears in the
    compiled PDF but consistently dragged the grade down (audit: implied
    ~27/100 LLM component on a clean 1-page resume). Heading markers are
    removed, emphasis unwrapped, list bullets dropped, and links collapse
    to their LABEL text - the URL is dropped because external ATS parsers
    can only see labels in the rendered PDF too (scoring alignment).
    """
    if not text:
        return ""
    lines_out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines_out.append("")
            continue
        line = _MARKDOWN_IMAGE_RE.sub(r'\1', line)
        line = _MARKDOWN_LINK_RE.sub(r'\1', line)
        line = _MARKDOWN_HEADING_RE.sub('', line)
        line = line.replace('`', '')
        line = _MARKDOWN_BOLD_RE.sub(r'\1', line)
        line = _MARKDOWN_ITALIC_RE.sub(r'\1', line)
        line = _MARKDOWN_BULLET_RE.sub('', line)
        lines_out.append(line)
    flat = "\n".join(lines_out)
    flat = re.sub(r'\n{3,}', '\n\n', flat)
    return flat.strip()


# Explicit Unicode-to-ASCII mapping for clean PDF and LaTeX output
_UNICODE_REPLACEMENTS = {
    # Status / indicator emojis
    '\u2705': '[Y]', '\u274c': '[N]', '\U0001f7e2': '[+]', '\U0001f7e1': '[~]', '\U0001f534': '[-]',
    '\U0001f4c4': '', '\U0001f4ca': '', '\U0001f4cb': '', '\U0001f4ce': '', '\U0001f4dd': '',
    '\U0001f4e5': '', '\U0001f3a4': '', '\u2709\ufe0f': '', '\U0001f50d': '', '\U0001f4a1': '',
    '\u23f3': '', '\u2764\ufe0f': '', '\u2764': '',
    # Hyphens, dashes, and minus signs
    '\u2010': '-', '\u2011': '-', '\u2012': '-',
    '\u2013': '-', '\u2014': '--',
    '\u2015': '--', '\u2212': '-',
    '\u2043': '-', '\uFE58': '-', '\uFE63': '-',
    '\uFF0D': '-',
    # Arrows and bullets
    '\u2192': '->', '\u2190': '<-', '\u2022': '-',
    '\u2023': '>', '\u25B6': '>',
    '\u2794': '->', '\u279C': '->', '\u21D2': '=>', '\u2194': '<->',
    '\u2502': '|', '\u2500': '-',
    '\u25CF': '*', '\u25CB': 'o', '\u25AA': '*',
    '\u2026': '...',
    # Quotes
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u00AB': '"', '\u00BB': '"',
    # Math & symbols
    '\u00bf': '->',   # '¿' — mojibake artifact of a mangled '→' arrow
    '\u00a1': '!',    # '¡' inverted bang (mojibake-ish)
    '\u00b1': '+/-',  # '±' plus-minus
    '\u00d7': 'x',    # '×' multiplication sign
    '\u00f7': '/',    # '÷' division sign
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u00fc': 'u',
    # Spaces and zero-width characters
    '\u202f': ' ', '\xa0': ' ', '\u2009': ' ', '\u200a': ' ', '\u2003': ' ',
    '\u200b': '', '\u200c': '', '\u200d': '',
    '\ufeff': '',
    '\ufe0f': '', '\ufe0e': '',
}


def _force_ascii(text: str) -> str:
    """Force text to pure ASCII by applying replacements and NFD decomposition."""
    if not text:
        return ""
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
    """Alias for _force_ascii for backward compatibility."""
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


# ==========================================
# Untrusting-text helpers: bullet hygiene + evidence checks
# ==========================================

# Whitespace/hyphen variants LLM bullets pick up from web-sourced JD text:
# U+00A0 (no-break), U+2007+U+202F (figures/narrow no-break) as spaces;
# U+2010-2012/FE63 hyphen variants. Live-run evidence (Nov 6 2026): scanner
# bullets carried "Claude\u202fCode" and "Hands\u2011on", which then leaked
# into comprehension-sensitive surfaces (frontend gap chips, ATS text spans).
_NBSP_CLASS = "\u00a0\u2007\u202f"
_HYPHEN_CLASS = "\u2010\u2011\u2012\u2012\ufe63"


def _tame_unicode_bullets(text: str) -> str:
    """Normalizes exotic whitespace/hyphens and per-line trailing spaces in
    LLM-produced bullet text so every downstream parser sees plain bytes."""
    if not text:
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    for ch in _NBSP_CLASS:
        t = t.replace(ch, " ")
    for ch in _HYPHEN_CLASS:
        t = t.replace(ch, "-")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    # Canon percent spacing (run 28e2e56f, Nov 12 2026): the same resume
    # printed "62 %", "70 % to 97.2 %" AND "47%", "98.6%" - LLM source-text
    # variance. Digit-adjacent "62%" is the dominant form, so collapse the
    # space (whitespace before % is never meaningful in resume prose).
    t = re.sub(r"(?<=\d)\s+(?=%)", "", t)
    return t


# Metric-shaped numerals: percentages, X-factors ("3x"), and millisecond
# rates ("1.7 ms"). Plain years/version numbers (GPT-4, Python 3.12) have no
# unit and are deliberately NOT matched - they are not metric claims.
# Fraction pass-rates ("150/150 benchmark passes" - Nov 7 run pair 1d3e1550)
# only count when a benchmark/test/pass/case/check word follows, so bare
# scores ("top 5/47") stay exempt.
_METRIC_TOKEN_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?(?:\s?(?:%|x(?![a-zA-Z])|ms\b|mm\b))"
    r"|\d+(?:[.,]\d+)?\s*/\s*\d+(?:[.,]\d+)?(?=\s*(?:benchmark|test|pass|"
    r"case|check)s?\b))"
)

# Typed-number parsing for the rounding tolerance (Nov 14 2026, the "96 %"
# survivor): a metric token decomposed into (value, unit, decimals). Ratios
# ("150/150") deliberately do NOT parse - they are exact-set claims, never
# rounding targets.
_METRIC_VALUE_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(%|x|ms|mm)$")


def _metric_value(token: str):
    """(value, unit, decimals) of a metric token, or None. '96 %' ->
    (96.0, '%', 0); '1.7 ms' -> (1.7, 'ms', 1); '150/150' -> None."""
    m = _METRIC_VALUE_RE.match((token or "").strip().lower())
    if not m:
        return None
    raw = m.group(1)
    decimals = len(raw.split(".")[1]) if "." in raw else (
        len(raw.split(",")[1]) if "," in raw else 0)
    return float(raw.replace(",", ".")), m.group(2), decimals


def _round_half_up(x: float, decimals: int) -> float:
    """Human-convention rounding (96.5 -> 97, not the float-lattice's
    banker's 96) at the claim's precision."""
    scale = 10 ** decimals
    return math.floor(x * scale + 0.5) / scale


def _claim_rounds_to(token: str, evidence_values: list) -> bool:
    """True when the claim token is a legitimate ROUNDED form of an
    evidence metric: same unit, and rounding OR truncating the evidence
    value to the claim's decimal precision reproduces the claim (96.2 ->
    "96 %" and "96.9 %"; 44.6 -> "45%"; 2.6 -> "3x"). A drifted claim
    ("400 ms" vs "380 ms") still fails."""
    parsed = _metric_value(token)
    if not parsed:
        return False
    claim_val, claim_unit, claim_dec = parsed
    for ev_val, ev_unit, _ev_dec in evidence_values:
        if ev_unit != claim_unit:
            continue
        rounded = _round_half_up(ev_val, claim_dec)
        truncated = math.floor(ev_val * (10 ** claim_dec)) / (10 ** claim_dec)
        if rounded == claim_val or truncated == claim_val:
            return True
    return False


def _evidence_metric_values(evidence_text: str) -> list:
    """(value, unit, decimals) triples for every metric-shaped token in the
    evidence text - the numeric side of the rounding tolerance."""
    out = []
    for m in _METRIC_TOKEN_RE.finditer(evidence_text or ""):
        parsed = _metric_value(m.group(1))
        if parsed:
            out.append(parsed)
    return out


def find_unevidenced_metric_claims(optimized_text: str, evidence_text: str,
                                   max_claims: int = 6) -> list[str]:
    """Returns metric-shaped tokens in the optimized resume that the evidence
    (source resume + user notes) does not contain.

    The improver is prompted never to coin numbers, but prompt guards alone
    demonstrably fold (live-run evidence Nov 6 2026, request 14751ff9: an
    empty user context still produced "cut manual data-ingestion effort by
    40%"). This check turns the violation into an explicit, user-visible
    warning instead of silently shipping invented metrics.

    Matching is whitespace-insensitive in BOTH directions: the token's
    whitespace is stripped entirely (LLM drafts carry U+202F/NBSP spaces the
    source PDF lacks - live run 868e406d: optimized "42.69 %" vs source
    "42.69%" flagged the user's REAL number), and the evidence is compacted
    per line so an spaced variant of a source metric never reads as a
    cross-line coincidence.

    Since Nov 14 2026 the scan is also ROUNDING-TOLERANT at the claim's
    decimal precision (the "96 %" survivor, live: the cover letter cited
    "96 %" where the resume says "96.2%" and was flagged as fabrication -
    typed numbers were compared as text only). Same unit + a claim that is
    the rounding or truncation of an evidence value ("96.2%" -> "96 %",
    "44.6%" -> "45%", "2.6x" -> "3x") is evidenced; a DRIFTED claim
    ("400 ms" vs "380 ms") still flags. Ratios never tolerate - exact-set
    claims.
    """
    if not optimized_text:
        return []
    src = re.sub(r"\s+", " ", (evidence_text or "").lower())
    src_lines_compact = [
        re.sub(r"\s+", "", (line or "").lower())
        for line in (evidence_text or "").splitlines()
    ]
    evidence_values = _evidence_metric_values(evidence_text)
    claims: list[str] = []
    seen: set[str] = set()
    for match in _METRIC_TOKEN_RE.finditer(optimized_text):
        token = match.group(1)
        key = token.lower()
        compact = re.sub(r"\s+", "", key)
        if key in seen:
            continue
        if key in src or compact in src \
                or any(compact in lc for lc in src_lines_compact):
            seen.add(key)
            continue
        seen.add(key)
        if _claim_rounds_to(token, evidence_values):
            continue
        claims.append(token)
        if len(claims) >= max_claims:
            break
    return claims


def find_dropped_source_metrics(final_text: str, source_resume: str,
                                max_findings: int = 6) -> list[str]:
    """The mirror of find_unevidenced_metric_claims: metric-shaped evidence
    (percentages / X-factors / ms rates - same token class) that a SOURCE
    accomplishment bullet carries but the final resume no longer mentions.

    Run bf0f7941 (Nov 12 2026): the writer's rewrite of the Smart Lecture
    entry shipped only the semantic-search half; the WER/transcription
    accomplishment (51.7%, WER 42.69 -> 20.61) vanished without a trace -
    removals were [], so no deterministic step recorded it, and the PDF
    could not be told apart from a resume that never had the claim. Tailoring
    legitimately condenses, so this only SURFACES the loss in the integrity
    channel (a deterministic re-insertion could not reconstruct the sentence
    around the number); the user decides whether the condensed form is
    acceptable.

    Only source BULLET lines count (accomplishments, not headers or contact
    lines). Same whitespace-insensitive matching as the added-metric
    tripwire so spaced variants never read as present/absent by accident.
    """
    if not final_text or not source_resume:
        return []
    final_compact = re.sub(r"\s+", "", (final_text or "").lower())
    findings: list[str] = []
    for raw in source_resume.splitlines():
        line = raw.strip()
        if not _BULLET_MARKER_RE.match(line):
            continue
        metrics = [m.group(1) for m in _METRIC_TOKEN_RE.finditer(line)]
        if not metrics:
            continue
        missing = sorted({m for m in metrics
                          if re.sub(r"\s+", "", m.lower()) not in
                          final_compact})
        if not missing:
            continue
        quote = line
        if len(quote) > 60:
            quote = quote[:57].rstrip() + "..."
        findings.append(
            f"source evidence metric(s) {', '.join(missing)} no longer in "
            f"the resume (from: '{quote}')")
        if len(findings) >= max_findings:
            break
    return findings


def find_unused_context_terms(human_notes: str, final_text: str,
                              source_resume: str, max_findings: int = 6
                              ) -> list[str]:
    """Context-incorporation tripwire, the notes-side mirror of
    find_dropped_source_metrics (round 7, Nov 12 2026): both fresh runs
    (25ff4ae5 / 1a8125f0) lost the user's pasted Aegis project and NOTHING
    recorded it - the improver's "USE THIS USER CONTEXT" instruction is
    advisory, the scrubber treats the notes as evidence (it can only
    remove content, never restore it), and a polisher section rewrite can
    drop context bullets without leaving a trace.

    Detection-only: distinctive terms the typed notes contribute that
    appear in NEITHER the source resume NOR the final resume are surfaced
    for the integrity warning. Advisory by design - tailoring may
    legitimately paraphrase or omit - but a silently dropped context
    becomes a visible warning instead of a mystery. Reuses the chip
    vocabulary filters: function-word noise (_CHIP_STOPWORDS), acronym
    noise (_GENERIC_SHORT), and tool-name-preserving tokenization
    (_WORD_SPLIT_RE keeps c++ / gpt-4 / node.js whole).
    """
    if not (human_notes or "").strip():
        return []
    final_l = (final_text or "").lower()
    source_l = (source_resume or "").lower()
    # Verb/preposition noise beyond the chip stoplist ("Ran the deploy
    # pipeline on a home lab") - kept local so _CHIP_STOPWORDS' contract
    # with the scrubber's suppression matching stays untouched. The deploy
    # family joins after the live round-11 smoke (session 17f5cfff, Nov 14
    # 2026): "deployed on a 4-node home lab" was flagged while the
    # substantive context (Aegis/Kubernetes) was already woven, so the
    # weave's one-shot attempt was spent on a generic verb.
    extra_stop = frozenset(
        "ran run runs running home lab "
        "deploy deployed deploys deploying deployment deployments".split())
    seen: set[str] = set()
    findings: list[str] = []
    for raw in _WORD_SPLIT_RE.split(human_notes):
        # trailing '.' is sentence punctuation, never part of a term
        # ("LangGraph." -> "langgraph") - internal dots stay (node.js)
        tok = raw.strip("-").rstrip(".").lower()
        if len(tok) < 3 or tok in seen:
            continue
        if (tok in _CHIP_STOPWORDS or tok in _GENERIC_SHORT
                or tok in extra_stop):
            continue
        if tok[0].isdigit() or re.fullmatch(r"[0-9.]+", tok):
            continue  # metrics have their own tripwires
        seen.add(tok)
        if tok in final_l or tok in source_l:
            continue
        # de-hyphenated variant counts as present ("heat-risk" ~ "Heat Risk")
        spaced = tok.replace("-", " ")
        if spaced in final_l or spaced in source_l:
            continue
        findings.append(raw.strip("-").rstrip("."))
    # Proper-noun tokens first (project/tool names are the actionable
    # findings), longest first within each class.
    findings.sort(key=lambda f: (f[:1].isupper(), len(f)), reverse=True)
    return findings[:max_findings]


# ============================================================
# Deterministic fabrication scrubber
# ============================================================
# Live-run evidence (request 14751ff9 and runs a32f30fc / 868e406d, Nov 6
# 2026): the reviewer's "remove the unverified claims" feedback was ignored
# twice by the model - detection-only tripwires cannot clean the artifact.
# These scrubbers deterministically remove unsupported content the gap
# channel can produce; the metric tripwire then reports whatever still
# survived.

# Words carrying no distinguishing signal inside a gap chip ("Experience
# with n8n (or similar workflow automation platform)" - the CLAIM is n8n).
_CHIP_STOPWORDS = frozenset("""
a an and any are as at based build building by demo demos direct experience
for from gained hands hands-on having in into knowledge of on or practical
similar server servers such the this that to tool tools work working with
skill skills technology technologies concepts
workflow automation platform workflows
including include includes involved involving
""".split())

# Word chars for term extraction: '+', '#', '.', '-' stay INSIDE tokens
# (c++, .net, gpt-4, node.js) - everything else splits.
_WORD_SPLIT_RE = re.compile(r"[^0-9a-zA-Z+#._-]+")

# Chip text splits into segments on punctuation that separates tool names
# inside a chip ("Claude Code, Cursor, Codex ...", "Hands-on work with
# n8n: used in production"): within a segment, connected capital/digit
# tokens belong to ONE tool name.
_SEGMENT_SPLIT_RE = re.compile(r"[,;:.()|]+")

# Single-capitalized acronyms that appear in EVERY AI-era gap chip but
# carry no distinctive signal ("AI agents and demos") - never a claim.
_GENERIC_SHORT = frozenset("ai ml api apis sdk".split())

_BULLET_MARKER_RE = re.compile(r"^[-*+•]\s")
_OR_SIMILAR_PAREN_RE = re.compile(
    r"\s*\([^()]*\bor\s+similar\b[^()]*\)", re.IGNORECASE)
# Draft skills row: "**Category**: item, item" / "Category: item, item"
_SKILL_ROW_RE = re.compile(
    r"^\s*\*{0,2}[^:*#@<>]{1,42}\*{0,2}:\s*\S")


def _significant_terms(text: str) -> list[str]:
    """TECHNOLOGY NAMES of a chip/label, precision-first.

    Live-run evidence (requests 09cf51fc / 1d3e1550, Nov 7 2026): plain-word
    terms ('using', 'agent', 'development', 'workflow-automation') culled
    innocent draft bullets that merely said 'using X' or 'development'.
    The failing heuristic - every non-stopword word is a potential claim -
    turned chip DESCRIPTIONS into removal triggers.

    A token becomes a term only when it looks like a NAME: capitalized
    ('Claude', 'MCP', 'FortyGuard') or digit-bearing ('n8n'). Generic
    filler is lowercased in scanner chips, so it drops automatically.
    Consecutive name tokens inside one comma/colon segment form ONE
    multi-word term ('Claude Code', 'FortyGuard Temperature MCP') so
    pieces like 'code' cannot hit innocent bullets ("code coverage").

    False negatives are preferred over false positives: a lowercase-only
    chip yields no terms at all, and unremoved bullying is safer than a
    removed truthful bullet.
    """
    terms: list[str] = []

    def flush(run: list[str]) -> None:
        if not run:
            return
        term = " ".join(run) if len(run) >= 2 else run[0]
        if term not in terms:
            terms.append(term)

    for segment in _SEGMENT_SPLIT_RE.split(text or ""):
        run: list[str] = []
        for token in _WORD_SPLIT_RE.split(segment):
            lw = token.lower()
            if lw in _CHIP_STOPWORDS or lw in _GENERIC_SHORT or len(lw) < 2:
                flush(run)
                run = []
                continue
            if any(ch.isupper() for ch in token) \
                    or any(ch.isdigit() for ch in token):
                run.append(lw)
            else:
                # lowercase filler breaks any capital run it interrupts
                flush(run)
                run = []
        flush(run)
    return terms


def _compact_all(s: str) -> str:
    """Lowercase + strip EVERY whitespace class (incl. U+202F/NBSP)."""
    return re.sub(r"\s+", "", (s or "").lower())


def _contains_term(term: str, haystack_compact: str) -> bool:
    """Containment of `term` inside whitespace-stripped text."""
    if not term or not haystack_compact:
        return False
    return _compact_all(term) in haystack_compact


def _term_in_text(term: str, text: str) -> bool:
    """Boundary-aware containment of a chip term in raw text.

    Compact substring matching would create false friends ("code" matches
    "codebase"); whole-word matching on the raw line is what a fabricated
    chip echo actually looks like ("Claude Code" hits "claude"). '.' stays
    in the leading class ("code" won't fire inside "node.js") but not in
    the trailing one - a sentence-final period is a word boundary.
    Multi-word terms match across any whitespace run ("Claude  Code").
    """
    if not term or not text:
        return False
    t = " ".join((term or "").strip().lower().split())
    parts = [re.escape(p) for p in t.split(" ")]
    pattern = re.compile(
        r"(?<![a-z0-9+#._])" + r"\s+".join(parts) + r"(?![a-z0-9+#])",
        re.IGNORECASE)
    return bool(pattern.search(text))


def parse_gap_chips(missing_skills_text: str) -> tuple[list[str], list[str]]:
    """Splits the gap channel (stamped `missing_skills`) into kept chips and
    'Do NOT feature' removal labels, both as raw text."""
    kept: list[str] = []
    removed_labels: list[str] = []
    for line in (missing_skills_text or "").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        s = re.sub(r"^[-*+•]\s+", "", s)
        lc = s.lower()
        for marker in ("do not feature", "don't feature", "dont feature"):
            if lc.startswith(marker):
                body = s[len(marker):].lstrip(": ").rstrip(". ")
                removed_labels.extend(
                    p.strip() for p in body.split(",") if p.strip())
                break
        else:
            kept.append(s)
    return kept, removed_labels


def _tidy_removal_residue(line: str) -> str:
    """Cleans the punctuation/space residue a removal can leave in a line:
    empty parens, doubled spaces, ' , ' chains, dangling separators."""
    t = re.sub(r"\(\s*\)", " ", line)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\(\s*,\s*", "(", t)
    t = re.sub(r",\s*\)", ")", t)
    t = re.sub(r"\s*,\s*(?=[;.)])", "", t)
    t = re.sub(r"\s*,\s*$", "", t)
    t = t.replace(" .", ".").replace(" ;", ";")
    t = re.sub(r"^[,\s;]+", "", t)
    return t.strip()


def _unevidenced_metric_tokens(line: str, ev_lines: list[str]) -> list[str]:
    """Compact forms of the metric tokens in `line` that the evidence
    (whitespace-insensitively, per line) does not contain. Since Nov 14
    2026 the same rounding tolerance as the full-text scan applies: a
    claim that is the rounding/truncation of an evidence metric at the
    claim's precision ("96 %" vs "96.2%") is evidenced; ratios never
    tolerate."""
    evidence_values: list = []
    for ln in ev_lines:
        for m in _METRIC_TOKEN_RE.finditer(ln):
            parsed = _metric_value(m.group(1))
            if parsed:
                evidence_values.append(parsed)
    out: list[str] = []
    for match in _METRIC_TOKEN_RE.finditer(line):
        compact = _compact_all(match.group(1))
        if not compact or compact in out:
            continue
        if any(compact in cl for cl in ev_lines):
            continue
        if _claim_rounds_to(match.group(1), evidence_values):
            continue
        out.append(compact)
    return out


def _drop_smallest_span(line: str, token_compact: str) -> Optional[str]:
    """Removes the smallest region of `line` holding the token:
    1. a parenthetical group,
    2. else a comma/semicolon segment (unless it is the only one).
    Returns None when the token cannot be excised cleanly."""
    for pm in re.finditer(r"\([^()]*\)", line):
        g = _compact_all(pm.group(0))
        if g and token_compact in g:
            return line[:pm.start()] + " " + line[pm.end():]
    parts = re.split(r"(?<=[,;])\s*", line)
    if len(parts) > 1:
        kept = [p for p in parts if token_compact not in _compact_all(p)]
        if kept and len(kept) < len(parts):
            return " ".join(kept)
    return None


def _scrub_metric_claims(line: str, ev_lines: list[str]) -> Optional[str]:
    """Removes unevidenced metric claims ('run 14751ff9: cut ... by 40%'):
    parenthetical, else comma segment, else None (caller drops the region).
    """
    tokens = _unevidenced_metric_tokens(line, ev_lines)
    if not tokens:
        return line
    work = line
    for tok in tokens:
        if tok not in _compact_all(work):
            continue
        work = _drop_smallest_span(work, tok)
        if work is None or tok in _compact_all(work):
            return None
    return _tidy_removal_residue(work)


def _suppressed_terms(kept_chips: list[str], removed_labels: list[str],
                      ev_lines: list[str]) -> list[str]:
    """Chip terms whose presence in the draft is unsupported:
    - every term of a REMOVED chip (the user said do not feature it),
    - terms of a KEPT chip absent from the evidence (resume + user notes).
    """
    suppressed: list[str] = []

    def add(ts: list[str]) -> None:
        for t in ts:
            if t not in suppressed:
                suppressed.append(t)

    for label in removed_labels:
        add(_significant_terms(label))
    for chip in kept_chips:
        add([t for t in _significant_terms(chip)
             if not any(_term_in_text(t, cl) for cl in ev_lines)])
    return suppressed


def _scrub_skill_row_line(stripped: str, suppressed: list[str],
                          removals: list[str]) -> Optional[str]:
    """Item-level scrub of a skills row: drops '(or similar)' hedges and
    items echoing suppressed chip terms. Returns the rebuilt row, or None
    when every item is gone. A row with nothing to remove is returned
    byte-identical: rebuilding would reformat it ("https://" -> "https: //"
    on a bare-URL contact line the row regex happens to match), and the
    scrubber's contract is removal, not reformatting."""
    m = re.match(r"^\s*([^:]{1,45}):(\s*.*)$", stripped)
    label, value = m.group(1), m.group(2)
    value, hedges = _OR_SIMILAR_PAREN_RE.subn("", value)

    items = [item.strip() for item in value.split(",") if item.strip()]
    kept_items: list[str] = []
    for item in items:
        if any(_term_in_text(t, item) for t in suppressed):
            removals.append("skills item removed: " + item.strip()[:44])
            continue
        kept_items.append(item)
    if not kept_items:
        return None
    if len(kept_items) == len(items) and not hedges:
        return stripped
    return _tidy_removal_residue(label + ": " + ", ".join(kept_items))


def scrub_unsupported_content(optimized_text: str, resume_text: str,
                              human_notes: str, missing_skills_text: str,
                              ) -> tuple[str, list[str]]:
    """Deterministically removes fabricated content the gap channel can
    produce, and returns (scrubbed_text, removal_notes).

    Removals, each grounded in agent state instead of model judgment:
      1. Bullets echoing a gap chip whose claim the evidence (source resume
         + user notes) does not demonstrate (run 868e406d: kept chips
         composed under pasted project text became fake Aegis bullets
         "Gained experience with n8n (or similar …)").
      2. Bullets echoing REMOVED chips ("Do NOT feature: X") even when the
         evidence shows them - the user explicitly suppressed that claim.
      3. Unevidenced metric claims (run 14751ff9: "cut ... effort by 40%"):
         parenthetical, else comma segment, else the whole bullet/sentence.
    Skills rows lose the offending ITEMS (and "(or similar)" hedges) rather
    than being dropped wholesale, so evidenced siblings stay.
    """
    if not optimized_text:
        return optimized_text, []

    # Chip-shaped hardening (live request 539f7c1d, Nov 8 2026): stale gap
    # chips can leak into the human_notes channel when a session reuse
    # carries them under typed evidence. The notes are EVIDENCE to this
    # scrubber by contract, so an unresolved chip line becomes "verified"
    # by definition and the fabricated echo survives. ASCII-bullet lines
    # are the extension's chip serialization (typed evidence in this app
    # uses "•" bullets or prose), so they must never count as evidence.
    notes_evidence = "\n".join(
        ln for ln in (human_notes or "").splitlines()
        if not ln.strip().startswith(("- ", "* ")))
    ev_text = f"{resume_text or ''}\n{notes_evidence}"
    ev_lines = [_compact_all(ln) for ln in ev_text.splitlines() if ln.strip()]
    kept_chips, removed_labels = parse_gap_chips(missing_skills_text)
    suppressed = _suppressed_terms(kept_chips, removed_labels,
                                   [ln for ln in ev_text.splitlines()
                                    if ln.strip()])
    removals: list[str] = []

    out_lines: list[str] = []
    for line in optimized_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        hits = [t for t in suppressed if _term_in_text(t, stripped)]

        if _BULLET_MARKER_RE.match(re.sub(r"\*+", "", stripped)):
            # A gap chip is ONE claim: a bullet built on it is fabricated as
            # a unit, so the whole bullet goes.
            if hits:
                removals.append("gap-echo bullet removed ("
                                + ", ".join(sorted(set(hits))) + ")")
                out_lines.append("")
                continue
            scrubbed = _scrub_metric_claims(stripped, ev_lines)
            if scrubbed is None:
                tokens = _unevidenced_metric_tokens(stripped, ev_lines)
                removals.append("bullet removed (unevidenced number: "
                                + ", ".join(tokens) + ")")
                out_lines.append("")
            else:
                out_lines.append(scrubbed if scrubbed != stripped else line)
            continue

        if _SKILL_ROW_RE.match(stripped):
            rebuilt = _scrub_skill_row_line(stripped, suppressed, removals)
            if rebuilt is None:
                removals.append("skills row removed: " + stripped[:40])
                out_lines.append("")
            else:
                out_lines.append(rebuilt)
            continue

        # Prose (summary etc.; contact/entry header lines carry ' | ').
        if " | " not in line:
            sentences = re.split(r"(?<=[.!?;])\s+", stripped)
            kept_sentences: list[str] = []
            for s in sentences:
                if any(_term_in_text(t, s) for t in suppressed):
                    hit_terms = sorted({t for t in suppressed
                                        if _term_in_text(t, s)})
                    removals.append(
                        "unevidenced chip claim removed from prose ("
                        + ", ".join(hit_terms) + ")")
                    continue
                s2 = _scrub_metric_claims(s, ev_lines)
                if s2 is None:
                    tokens = _unevidenced_metric_tokens(s, ev_lines)
                    removals.append(
                        "sentence removed (unevidenced number: "
                        + ", ".join(tokens) + ")")
                    continue
                kept_sentences.append(s2)
            rebuilt_prose = " ".join(kept_sentences)
            if rebuilt_prose.strip():
                out_lines.append(_tidy_removal_residue(rebuilt_prose))
            else:
                out_lines.append("")
            continue

        out_lines.append(line)

    text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", "\n".join(out_lines))
    if len(removals) > 10:
        removals = removals[:10] + [f"...and {len(removals) - 10} more removals"]
    return _tame_unicode_bullets(text), removals


# ==========================================
# Section utilities (MR1 section-targeted polish) + QW2 decline protocol
# ==========================================

# Reviewer feedback section vocabulary -> canonical draft headers of the
# improver template. Anything outside this map ("formatting", "structure",
# "overall tone") cannot be merged surgically, so the polisher falls back
# to its full-rewrite mode instead.
SECTION_ALIASES: dict[str, str] = {
    "summary": "Professional Summary",
    "professional summary": "Professional Summary",
    "profile": "Professional Summary",
    "skills": "Technical Skills",
    "technical skills": "Technical Skills",
    "experience": "Experience",
    "work experience": "Experience",
    "employment": "Experience",
    "projects": "Projects",
    "project": "Projects",
    "education": "Education",
    "certifications": "Certifications",
    "certificates": "Certifications",
    "licenses": "Certifications",
}


def canonical_feedback_section(name: str) -> Optional[str]:
    """Maps a reviewer's feedback-section name to the canonical draft
    header ('Professional Summary' ... 'Certifications'), or None when the
    target is cross-cutting/unknown and cannot be addressed surgically."""
    key = " ".join(str(name or "").strip().lower().split())
    if not key:
        return None
    return SECTION_ALIASES.get(key)


def split_markdown_sections(md: str) -> list[tuple[Optional[str], str, str]]:
    """Splits a resume draft into (canonical_name, header_line, body) blocks
    in document order. canonical_name is None for headers outside the known
    vocabulary. Content BEFORE the first '## ' header (the candidate identity
    block) is never returned - the section polish leaves it untouched."""
    blocks: list[tuple[Optional[str], str, str]] = []
    header_line: Optional[str] = None
    canonical: Optional[str] = None
    body: list[str] = []
    for line in md.splitlines():
        if line.startswith("## "):
            if header_line is not None:
                blocks.append((canonical, header_line, "\n".join(body).strip("\n")))
            header_line = line.rstrip()
            canonical = canonical_feedback_section(line[3:])
            body = []
        elif header_line is not None:
            body.append(line)
    if header_line is not None:
        blocks.append((canonical, header_line, "\n".join(body).strip("\n")))
    return blocks


def merge_rewritten_sections(md: str, rewrites: dict[str, str]) -> tuple[str, list[str]]:
    """Deterministically merges section rewrites (canonical name -> new body,
    header line EXCLUDED) into the draft: each targeted section keeps its
    ORIGINAL header line (project title/link lines must survive byte-identical)
    and every other byte of the document is preserved exactly.

    Returns (merged_text, replaced_names). When a rewrite does not find
    exactly one matching block (unknown name, duplicate section, ...), the
    merge is refused outright -> (md, []) - the caller falls back to the
    full-rewrite pass rather than shipping a partial merge.
    """
    if not rewrites:
        return md, []
    hits = list(re.finditer(r"(?m)^##[ \t].*$", md))
    if not hits:
        return md, []
    pending = dict(rewrites)
    replaced: list[str] = []
    pieces: list[str] = [md[: hits[0].start()]]
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(md)
        raw = md[m.start():end]
        parts = raw.split("\n", 1)
        header_line = parts[0].rstrip()
        canonical = canonical_feedback_section(
            re.sub(r"^##[ \t]+", "", header_line))
        if canonical in pending:
            separator = raw[len(parts[0]) + 1:]
            body_sep = separator[len(separator.rstrip()):]  # blank lines before the next header
            pieces.append(header_line + "\n" + pending.pop(canonical).strip() + body_sep)
            replaced.append(canonical)
        else:
            pieces.append(raw)
    if len(replaced) != len(rewrites) or pending:
        # Partial match would ship an inconsistent mix - refuse wholesale.
        return md, []
    return "".join(pieces), replaced


# QW2 decline protocol: the writer's own admission that a requested item is
# unsupported. Lines are stripped from the draft before any consumer (ATS,
# LaTeX, PDF) can see them; the list surfaces to the user via the results UI.
_DECLINED_LINE_RE = re.compile(r"[ \t]*DECLINED:[ \t]*(.+?)[ \t]*$", re.IGNORECASE)


def extract_declined_requests(text: str) -> tuple[str, list[str]]:
    """Removes the writer's DECLINED: lines from a draft and returns them as
    (clean_text, declined_requests) - order-preserving, deduplicated.
    A draft with no DECLINED: lines is returned untouched (byte-identical)."""
    declined: list[str] = []
    if not text or "DECLINED:" not in text:
        return text, []
    kept_lines: list[str] = []
    for line in text.splitlines():
        m = _DECLINED_LINE_RE.match(line)
        if not m:
            kept_lines.append(line)
            continue
        item = m.group(1).strip().rstrip("*_")
        if item and item not in declined:
            declined.append(item)
    if not declined:
        return text, []
    return "\n".join(kept_lines), declined


# Contact-link preservation (run 124f40cb, Nov 12 2026): the improver's
# rewrite dropped the candidate's huggingface.co profile link from the
# contact header. The source header carried it only as a bare LABEL inside
# a markdown link whose href pointed at the LinkedIn URL - 3 of 4 stored
# runs' rewrites reconstructed it anyway, this one stopped at GitHub and
# left a dangling "|". Contact links are identity, not tailored content:
# a deterministic tail re-injects any source-header link the draft's
# header lost, on every writer pass (the polisher reassembles sections
# around the draft's header line, so guarding the shared tail suffices).
# A bare host only qualifies when it carries a path (github.com/x) or an
# explicit scheme, so email domains (gmail.com inside an address) can
# never be re-injected as fake links.
_LINK_TOKEN_RE = re.compile(
    r"(?:https?://|www\.)[^\s|\)\]}>,]+"
    r"|(?:[\w-]+\.)+(?:com|net|org|io|ai|co|dev|me|app|tech|site|space)"
    r"(?:/[^\s|\)\]}>,]+)?"
)
_SECTION_MARK_RE = re.compile(r"^#{1,6}\s+\S|^[A-Z][A-Z0-9 &/'\-]{3,}$")


def _norm_link(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"^(?:https?://|www\.)", "", t, flags=re.IGNORECASE)
    return t.rstrip("/.,;:!?)\"'")


def _header_region_lines(text: str, limit: int = 8) -> list[str]:
    """The header block: the first nonempty lines before a section marker
    (a '#' heading or an ALL-CAPS heading such as 'SUMMARY')."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if out:
                break
            continue
        if _SECTION_MARK_RE.match(stripped):
            break
        out.append(stripped)
        if len(out) >= limit:
            break
    return out


def _contact_placeholder_tokens() -> list[str]:
    """Placeholder tokens a writer pass can emit in a contact line when a
    field has no value (run bf0f7941, Nov 12 2026: the draft shipped
    "...github.com/Omar10lfc | N/A | ..." and the link restoration then
    appended right after the fake token). Pipe-delimited tokens only."""
    return ["n/a", "na", "n.a.", "tbd", "none", "null"]


# A full-part markdown link in the contact line ("[Portfolio](https://...)").
_CONTACT_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _unwrap_contact_link(part: str, normalized: list[str]) -> str:
    """Contact-line part transform: a full-part markdown link becomes the
    bare URL, because the renderer prints the LABEL of a markdown link
    (_render_contact's \\href{url}{\\underline{label}}) while the accepted
    header form displays the URL text itself. Round 7 (Nov 12 2026, runs
    25ff4ae5 / 1a8125f0): both drafts shipped
    "[LinkedIn](https://...) | ... | [Portfolio](https://huggingface.co/...)"
    and the PDF showed the word "Portfolio" where every accepted run
    displays huggingface.co/Omar10lfc. Mailto links unwrap to the address;
    anything else passes through unchanged. Display form is recorded for
    the guardrail channel."""
    m = _CONTACT_MD_LINK_RE.fullmatch(part)
    if not m:
        return part
    url = m.group(2).strip().split()[0].strip().lstrip("<").rstrip(")")
    if url.lower().startswith("mailto:"):
        url = url[len("mailto:"):]
    if not url:
        return part
    display = re.sub(r"^(?:https?://|www\.)", "", url,
                     flags=re.IGNORECASE).rstrip("/")
    normalized.append(display or url)
    return url


def preserve_contact_links(
        draft: str,
        source_resume: str) -> tuple[str, list[str], list[str], list[str]]:
    """Re-injects source-header contact links that the rewritten draft's
    header lost; also strips a dangling trailing "|" from the draft's
    contact line (run 124f40cb shipped "...github.com/Omar10lfc |"),
    placeholder tokens (N/A / TBD / ...) standing in for an absent field
    (run bf0f7941 shipped "...github.com/Omar10lfc | N/A | ..."), and
    normalizes label-form markdown links to their bare URLs (round 7, Nov
    12 2026: "[Portfolio](https://huggingface.co/...)" printed the label
    instead of the URL).

    Returns (draft, restored_tokens, removed_placeholders, normalized_links)
    with tokens in display form (scheme stripped). Idempotent: an
    already-complete header restores nothing and the text passes through
    unchanged. Only the header REGION is compared - links in the body are
    content and have their own integrity paths.
    """
    if not draft or not source_resume:
        return draft, [], [], []

    lines = draft.split("\n")
    region_idx: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if region_idx:
                break
            continue
        if stripped.startswith("#"):
            if region_idx:
                break  # a section heading after the header block
            continue  # the draft's "# Name" heading
        if _SECTION_MARK_RE.match(stripped):
            break
        region_idx.append(i)
        if len(region_idx) >= 6:
            break
    if not region_idx:
        return draft, [], [], []

    # Sanitation pass on the contact line: placeholder tokens out (a
    # restored link must never land after a fake "N/A" token), label-form
    # markdown links unwrapped to bare URLs. A contact line keeps at least
    # two tokens.
    removed: list[str] = []
    normalized: list[str] = []
    changed = False
    contact_idx = None
    for i in region_idx:
        if ("|" in lines[i] or "http" in lines[i].lower() or "@" in lines[i]):
            contact_idx = i
    if contact_idx is None:
        contact_idx = region_idx[-1]
    placeholders = set(_contact_placeholder_tokens())
    parts = [p.strip() for p in lines[contact_idx].split("|")]
    unwrapped = [_unwrap_contact_link(p, normalized) for p in parts]
    kept = [p for p in unwrapped if p and p.lower() not in placeholders]
    if kept != parts and len(kept) >= 2:
        removed = [p for p in unwrapped if p and p.lower() in placeholders]
        lines[contact_idx] = " | ".join(kept)
        changed = True
        if removed:
            _safe_print(f"   [Contact] Removed placeholder token(s) from "
                        f"the contact line: {', '.join(removed)}")
        if normalized:
            _safe_print(f"   [Contact] Normalized contact link label(s) to "
                        f"the bare URL: {', '.join(normalized)}")

    src_tokens: dict[str, str] = {}
    for line in _header_region_lines(source_resume):
        for m in _LINK_TOKEN_RE.finditer(line):
            tok = m.group(0)
            norm = _norm_link(tok)
            schemeless_bare_host = (
                "/" not in norm
                and not tok.lower().startswith(("http://", "https://", "www.")))
            if schemeless_bare_host:
                continue  # an email domain or bare TLD, never a profile link
            src_tokens.setdefault(norm, tok)

    present = {_norm_link(m.group(0))
               for i in region_idx
               for m in _LINK_TOKEN_RE.finditer(lines[i])}
    missing = [tok for norm, tok in src_tokens.items() if norm not in present]

    restored: list[str] = []
    if missing:
        add: list[str] = []
        for tok in missing:
            add.append(tok if tok.lower().startswith(("http://", "https://"))
                       else "https://" + tok)
            restored.append(re.sub(r"^(?:https?://|www\.)", "", tok,
                                   flags=re.IGNORECASE).rstrip("/"))
        contact = re.sub(r"\s*\|\s*$", "", lines[contact_idx].rstrip())
        lines[contact_idx] = contact + " | " + " | ".join(add)
        changed = True
        if restored:
            _safe_print(f"   [Contact] Restored dropped contact link(s): "
                        f"{', '.join(restored)}")
    if not changed:
        return draft, [], [], []
    return "\n".join(lines), restored, removed, normalized


# BB1 Phase B: the markdown-link counter shared by the polisher's targeted
# merge gate and the improver's per-unit rewrite gate (a rewrite that drops
# a project/demo/GitHub URL the source carried is refused deterministically).
_MD_LINK_RE = re.compile(r"\]\([^)]+\)")


def markdown_link_count(text: str) -> int:
    """Number of markdown links ``[label](url)`` in ``text``."""
    return len(_MD_LINK_RE.findall(text or ""))


def accumulate_unique(existing: Optional[list], added: Optional[list]) -> list:
    """Order-preserving accumulation of state list values across passes."""
    out = list(existing or [])
    for item in (added or []):
        if item and item not in out:
            out.append(item)
    return out


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
