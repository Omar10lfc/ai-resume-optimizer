"""Shared utility functions: logging, truncation, integrity checks, and markdown cleanup."""
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Optional

from .db_utils import db_schema_version, ensure_db_migrations, remove_wal_residue

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
        heading_m = _MARKDOWN_HEADING_RE.match(line)
        if heading_m:
            is_section_h2 = heading_m.group(0).lstrip().startswith("##")
            line = _MARKDOWN_HEADING_RE.sub('', line).strip()
            line = line.replace('`', '')
            line = _MARKDOWN_BOLD_RE.sub(r'\1', line)
            line = _MARKDOWN_ITALIC_RE.sub(r'\1', line)
            # Preserve section boundary cues (H2+ -> uppercase matches PDF small-caps)
            # while keeping H1 candidate name natural case.
            if line:
                lines_out.append(line.upper() if is_section_h2 else line)
            continue
        line = line.replace('`', '')
        line = _MARKDOWN_BOLD_RE.sub(r'\1', line)
        line = _MARKDOWN_ITALIC_RE.sub(r'\1', line)
        line = _MARKDOWN_BULLET_RE.sub('\u2022 ', line)
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


def _is_bare_zero_metric(token: str) -> bool:
    """True for '0%' / '0.0%' etc. — phrases like 'reducing dropped tasks to 0%'
    reword naturally to 'eliminating dropped tasks' and should not trigger false
    metric warnings."""
    parsed = _metric_value(token)
    return parsed is not None and parsed[0] == 0.0 and parsed[1] == "%"


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
        if _is_bare_zero_metric(token):
            continue
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
        metrics = [m.group(1) for m in _METRIC_TOKEN_RE.finditer(line)
                   if not _is_bare_zero_metric(m.group(1))]
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
    """Placeholder tokens a writer pass can emit in a contact or metadata line
    when a field has no value (e.g. "... | N/A | ..." or "Location not specified").
    Pipe-delimited tokens only."""
    return [
        "n/a", "na", "n.a.", "tbd", "tba", "none", "null",
        "not specified", "unspecified", "location not specified",
        "city not specified", "not available", "not applicable",
        "not provided", "none provided", "to be determined", "to be decided",
        "unknown", "unknown location", "location unknown", "[...]", "...",
    ]


_URL_PROTECT_RE = re.compile(r"(\[[^\]]*\]\([^)]*\)|https?://\S+)", re.IGNORECASE)

_PLACEHOLDER_CORE_PATTERN = (
    r"(?:(?:location|city|address|state|country|date|dates|year|company|"
    r"employer|organization|institution|university|degree|role|title|url|link)"
    r"\s+(?:not\s+specified|unknown)"
    r"|location\s+not\s+specified"
    r"|not\s+specified"
    r"|unspecified"
    r"|not\s+available"
    r"|not\s+applicable"
    r"|not\s+provided"
    r"|none\s+provided"
    r"|to\s+be\s+determined"
    r"|to\s+be\s+decided"
    r"|(?<![a-z0-9/])n\s*/\s*a(?![a-z0-9/])"
    r"|n\.a\."
    r"|tbd|tba"
    r"|unknown(?:\s+location)?)"
)

_PIPE_SEGMENT_PLACEHOLDER_RE = re.compile(
    rf"^(?:{_PLACEHOLDER_CORE_PATTERN}|na|none|null|\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)$",
    re.IGNORECASE,
)
_BRACKET_PLACEHOLDER_RE = re.compile(
    rf"\[\s*(?:{_PLACEHOLDER_CORE_PATTERN}|na|none|null|\.\.\.|…)\s*\](?!\()",
    re.IGNORECASE,
)
_PAREN_PLACEHOLDER_RE = re.compile(
    rf"\(\s*(?:{_PLACEHOLDER_CORE_PATTERN}|na|none|null|\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)\s*\)",
    re.IGNORECASE,
)
_INLINE_PLACEHOLDER_RE = re.compile(
    rf"(?<!\w)(?:{_PLACEHOLDER_CORE_PATTERN})(?!\w)",
    re.IGNORECASE,
)


def _is_placeholder_segment(seg: str) -> bool:
    """True when a pipe-separated field is solely a placeholder marker."""
    plain = re.sub(r"[*_`~]", "", (seg or "")).strip()
    return bool(plain) and bool(_PIPE_SEGMENT_PLACEHOLDER_RE.fullmatch(plain))


def strip_placeholder_text(text: str) -> tuple[str, list[str]]:
    """Deterministically strip placeholder phrases ('Location not specified',
    'not specified', 'N/A', 'TBD', 'unknown', '[...]') across resume lines
    while preserving markdown link targets and legitimate content.

    Returns ``(cleaned_text, removed_phrases)``. Idempotent and no-op when
    no placeholder is present.
    """
    if not text:
        return text, []

    removed: list[str] = []
    out_lines: list[str] = []

    def _clean_segment_preserving_links(raw_text: str, *, pipe_mode: bool = False) -> str:
        # Temporarily stash URLs and [label](url) links; inside [label](url),
        # never alter the URL target and keep normal labels intact.
        protected: list[str] = []

        def _stash(m: re.Match) -> str:
            protected.append(m.group(0))
            return f"\x00URL{len(protected) - 1}\x00"

        work = _URL_PROTECT_RE.sub(_stash, raw_text)

        def _record(m: re.Match) -> str:
            tok = m.group(0).strip()
            if tok:
                removed.append(tok)
            return ""

        work = _BRACKET_PLACEHOLDER_RE.sub(_record, work)
        work = _PAREN_PLACEHOLDER_RE.sub(_record, work)
        work = _INLINE_PLACEHOLDER_RE.sub(_record, work)

        # Clean empty bold/italic wrappers left around a stripped placeholder.
        work = re.sub(r"\*{2}\s*\*{2}|(?<!\w)_\s*_(?!\w)", "", work)
        # Clean stray leading/trailing separator punctuation left by removal.
        work = re.sub(r"\s{2,}", " ", work)
        if not pipe_mode:
            work = re.sub(r"\s+([,.;:])", r"\1", work)
            work = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", work)
            work = re.sub(r"(?:\s*[-–—:,]\s*)+$", "", work.strip()) if work != raw_text else work
        else:
            work = work.strip()

        def _unstash(m: re.Match) -> str:
            return protected[int(m.group(1))]

        return re.sub(r"\x00URL(\d+)\x00", _unstash, work)

    for line in text.split("\n"):
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            kept_parts: list[str] = []
            line_changed = False
            for part in parts:
                if not part:
                    line_changed = True
                    continue
                if _is_placeholder_segment(part):
                    removed.append(re.sub(r"[*_`~]", "", part).strip())
                    line_changed = True
                    continue
                cleaned_part = _clean_segment_preserving_links(part, pipe_mode=True)
                if cleaned_part != part:
                    line_changed = True
                if cleaned_part and not _is_placeholder_segment(cleaned_part):
                    kept_parts.append(cleaned_part)
                elif part:
                    line_changed = True
            if line_changed:
                if kept_parts:
                    out_lines.append(" | ".join(kept_parts))
                continue
            out_lines.append(line)
            continue

        # Bullet / plain line: protect leading bullet or heading prefix.
        m_prefix = re.match(r"^(\s*(?:[-*•]\s+|#{1,6}\s+))(.*)$", line)
        prefix, body = (m_prefix.group(1), m_prefix.group(2)) if m_prefix else ("", line)
        before_count = len(removed)
        cleaned_body = _clean_segment_preserving_links(body, pipe_mode=False).strip()
        if len(removed) > before_count:
            if not cleaned_body:
                continue  # Entire line was a placeholder item; drop line.
            out_lines.append(f"{prefix}{cleaned_body}" if prefix else cleaned_body)
        else:
            out_lines.append(line)

    cleaned_text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", "\n".join(out_lines))
    return cleaned_text, removed


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


# A link construct whose LABEL spans pipe-separated fields: PDF ingestion
# wraps link-ANNOTATED text as [text](url), and a source PDF's link
# rectangle can cover a whole contact line - so the line's full text
# becomes ONE label ("[City | phone | email |](linkedin-url)" or
# "[url1 | url2 | url3](linkedin-url)"). Live run bf58e700 (Nov 15 2026):
# BB1-B pins the identity header verbatim, these constructs reached the
# renderers, and the contact classifier emitted \href garbage per
# fragment ("\href{https://](https://...)}", a "](" location field).
_CONTACT_SPAN_LINK_RE = re.compile(r"\[([^\[\]]*\|[^\[\]]*)\]\(([^)]*)\)")


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
    instead of the URL), and flattens span-label link constructs to their
    visible text (bf58e700, Nov 15 2026: a PDF link annotation covering a
    whole contact line wrapped its full text as ONE "[a | b | c](url)"
    link, which the renderers' per-segment classifier turned into broken
    \\href output).

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

    removed: list[str] = []
    normalized: list[str] = []
    changed = False

    # Flatten span-label link constructs to their visible text (bf58e700,
    # Nov 15 2026): a PDF link annotation can cover a whole contact line,
    # so ingestion wrapped the line's text as ONE link whose label spans
    # the pipe-separated fields. The renderers' contact classifier works
    # per pipe-segment and turned the fragments into broken \href output;
    # the visible text (city, phone, email, and the bare profile URLs)
    # classifies cleanly on its own. The target survives whenever it is
    # also visible text (the "[linkedin.com/... | ...](linkedin)" shape);
    # the re-inject below restores any target that is not.
    for i in region_idx:
        if not _CONTACT_SPAN_LINK_RE.search(lines[i]):
            continue
        flat = _CONTACT_SPAN_LINK_RE.sub(lambda m: m.group(1), lines[i])
        flat = re.sub(r"\s*\|\s*$", "", flat.rstrip())
        if flat != lines[i]:
            lines[i] = flat
            changed = True
            head = flat.split("|")[0].strip()[:48]
            normalized.append(f"flattened header link construct to its "
                              f"visible text ({head})")
            # Item 10: the flattened fragment is resume-derived - never logged.
            _safe_print("   [Contact] Flattened a span-label header link "
                        "to its visible text")

    # Sanitation pass on the contact line: placeholder tokens out (a
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
            # Item 10: removed/normalized tokens are resume-derived - log counts.
            _safe_print(f"   [Contact] Removed {len(removed)} placeholder "
                        "token(s) from the contact line")
        if normalized:
            _safe_print(f"   [Contact] Normalized {len(normalized)} contact "
                        "link label(s) to the bare URL")

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
            # Item 10: restored link(s) are resume-derived - log the count.
            _safe_print(f"   [Contact] Restored {len(restored)} dropped "
                        "contact link(s)")
    if not changed:
        return draft, [], [], []
    return "\n".join(lines), restored, removed, normalized


# BB1 Phase B: the markdown-link target counter shared by the polisher's targeted
# merge gate and the improver's per-unit rewrite gate (a rewrite that drops
# a project/demo/GitHub URL the source carried is refused deterministically).
_MD_LINK_RE = re.compile(r"\]\([^)]+\)")


def markdown_link_targets(text: str) -> frozenset:
    """The DISTINCT normalized targets of the markdown links in ``text``.

    The unit/merge gates compare TARGET SETS, not instance counts: the
    broken-source shapes a PDF upload can carry link the SAME url several
    times (live run bf58e700: one project title line carried three links
    to the same lnkd.in target), so a faithful canonical rewrite that
    merges duplicates must not be refused for "dropping" an instance -
    only losing a DISTINCT target is a real loss.
    """
    return frozenset(_norm_link(m.group(0)[2:-1])
                     for m in _MD_LINK_RE.finditer(text or ""))


# A complete [label](url) construct (the span-label regex above is the
# pipe-bearing subset; this is the general form for the cleanup pass).
_MD_LINK_FULL_RE = re.compile(r"\[([^\[\]]*)\]\(([^)]*)\)")
_BULLET_MARKER_RE = re.compile(r"^[-*•]\s+")

# ---------------------------------------------------------------------------
# Deterministic shape repairs for raw PDF extraction (run d946bb5f, Nov 16
# 2026). pypdf line-wraps ship damage the writer must never see: hyphenated
# words split across a wrap ("halluci-" / "nated skills"), the sentence's
# plus sign landing at a wrap start ("CAMeL-BERT" / "+ FAISS retrieval
# stack..."), education split across two lines, and entry titles that lost
# their bold/date-column shape. latex_render imports from this module, so
# the section word set lives here (no reverse import).
_PREPASS_SECTION_WORDS = frozenset({
    "summary", "objective", "profile", "about", "education", "experience",
    "employment", "history", "projects", "skills", "competencies",
    "technologies", "certifications", "certificates", "awards", "honors",
    "publications", "volunteer", "languages", "coursework", "activities",
})
# Sections whose entry-title lines the reshapes may touch (certifications
# and skills carry their own canonical shapes - never reshaped).
_PREPASS_RESHAPE_SECTIONS = frozenset(
    {"projects", "experience", "employment", "history", "education"})

_PREPASS_HR_RE = re.compile(r"^\s*[-–—_=*]{3,}\s*$")
_PREPASS_BULLET_RE = re.compile(r"^\s*[-*+•]\s+")
_PREPASS_PLUS_LINE_RE = re.compile(r"^\s*\+\s+(.*)$")
_PREPASS_DEGREE_HINT_RE = re.compile(
    r"(?i)\b(?:b\.?\s?sc|m\.?\s?sc|b\.?eng|b\.?s\b|b\.?a\b|m\.?s\b|m\.?eng|"
    r"ph\.?d|bachelor|master|doctorate|diploma|degree)\b")

# Soft-hyphenation vs compound hyphens: a wrapped "halluci-"/"nated" or
# "pro-"/"gram" must fuse WITHOUT the hyphen, but "multi-"/"class" is a
# real compound that keeps it ("multiclass" is wrong). Heuristic: when
# the fragment before the hyphen is a known compound-forming stem, the
# hyphen stays - unless the remainder after the wrap is a bare
# inflectional tail ("test-" + "ing" -> "testing").
_COMPOUND_HYPHEN_PREFIXES = frozenset({
    "multi", "cross", "co", "self", "end", "real", "open", "high", "low",
    "big", "deep", "full", "half", "well", "world", "web", "ai", "ml", "cv",
    "llm", "nlp", "ocr", "ui", "ux", "api", "data", "cloud", "edge", "state",
    "time", "user", "client", "server", "test", "type", "domain", "level",
    "side", "chain", "step", "zero", "one", "two", "three",
})
_SOFT_SUFFIX_TAILS = frozenset({
    "s", "es", "ed", "ing", "er", "est", "ly", "ion", "ions", "ment",
    "ness", "able", "ible", "ance", "ence", "ize", "ise", "ify", "ate",
    "ive", "ous", "ful", "less", "ish", "al", "ic", "ity",
})

# End-anchored date for the education merge: dash ranges ("Sep 2025 - Jun
# 2026", "2020 - Present", "2020-2021"), worded ranges ("Oct 2022 to Jun
# 2026" - pypdf-7 keeps the words, and latex_render's date regexes needed
# a dash), month-month ranges ("Aug - Sep 2025") and a single month+year.
_PREPASS_DATE_TAIL_RE = re.compile(
    r'(?:'
    r'(?:[A-Za-z]{3,9}\.?\s+)?\d{4}\s*'
    r'(?:[-–—]|\s+(?:to|through|till|until)\s+)\s*'
    r'(?:[A-Za-z]{3,9}\.?\s*,?\s*)?(?:\d{4}|Present|Current|Ongoing)\s*'
    r'|[A-Za-z]{3,9}\.?\s*[-–—]\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4}\s*'
    r'|[A-Za-z]{3,9}\.?\s+\d{4}\s*'
    r')$', re.IGNORECASE)

# A date-range START anywhere in a line (the case-C org/date split).
_PREPASS_DATE_ANY_RE = re.compile(
    r'(?:[A-Za-z]{3,9}\.?\s+)?\d{4}\s*'
    r'(?:[-–—]|\s+(?:to|through|till|until)\s+)\s*'
    r'(?:[A-Za-z]{3,9}\.?\s*,?\s*)?(?:\d{4}|Present|Current|Ongoing)'
    r'|[A-Za-z]{3,9}\.?\s*[-–—]\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{4}',
    re.IGNORECASE)


def _prepass_section_of(line: str) -> Optional[str]:
    """The section word a heading line carries ('## Projects' or bare
    'PROJECTS'), else None - the pre-pass's section context tracker."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("#"):
        cand = s.lstrip("#").strip().rstrip(":")
    elif s.isupper() and 2 <= len(s) <= 40:
        cand = s.title()
    else:
        return None
    if not cand:
        return None
    words = re.sub(r"[^a-zA-Z\s]", " ", cand.lower()).split()
    for w in words:
        if w in _PREPASS_SECTION_WORDS:
            return w
    return None


def _is_degree_line(line: str) -> bool:
    """True for a short, unadorned education line naming a degree - the
    merge partner of a university line that ends in a date range."""
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if _PREPASS_BULLET_RE.match(s) or s.startswith(("*", "#")):
        return False
    if "|" in s or "[" in s or "**" in s:
        return False
    if _PREPASS_DATE_TAIL_RE.search(s):
        return False
    return bool(_PREPASS_DEGREE_HINT_RE.search(s))


def _split_university_date(line: str) -> Optional[tuple]:
    """'Egypt-Japan University of Science and Technology (E-JUST), '
    'Alexandria - Oct 2022 to Jun 2026' -> (university, location, date)
    when the line ends in a date range and no link/markup is present."""
    s = line.strip()
    if not s or _PREPASS_BULLET_RE.match(s) or s.startswith(("*", "#")):
        return None
    if "[" in s or "**" in s or "|" in s:
        return None
    m = _PREPASS_DATE_TAIL_RE.search(s)
    if not m or m.start() == 0:
        return None
    date = s[m.start():].strip()
    head = re.sub(r"[|,\u2013\u2014\-\s]+$", "", s[:m.start()]).strip()
    if not head or len(head) > 90:
        return None
    if "," in head:
        uni, loc = head.rsplit(",", 1)
        return uni.strip(), loc.strip(), date
    return head, "", date


def _education_merged(parts: tuple, degree: str) -> str:
    """Canonical 4-field education line: parses as ONE subheading
    (university | location | degree | date) instead of a two-line split
    that rendered an empty sub-row plus a detached itemize (the blank gap
    in run d946bb5f's Education section)."""
    uni, loc, date = parts
    fields = [f"**{uni.strip()}**"]
    if loc.strip():
        fields.append(loc.strip())
    fields.append(degree.strip())
    fields.append(date)
    return " | ".join(fields)


def _merge_education_lines(lines: list) -> list:
    """(d) Education two-line merge - runs FIRST so no later rule can grab
    either half. Accepts both orders (university-with-date + degree line,
    or degree line + university-with-date)."""
    out = []
    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        sec = _prepass_section_of(line)
        if sec:
            section = sec
            out.append(line)
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        merged = None
        if section == "education" and nxt is not None:
            uni_a = _split_university_date(line)
            if uni_a and _is_degree_line(nxt):
                merged = _education_merged(uni_a, nxt)
            else:
                uni_b = _split_university_date(nxt)
                if uni_b and _is_degree_line(line):
                    merged = _education_merged(uni_b, line)
        if merged is not None:
            out.append(merged)
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def _join_dehyphenated(lines: list) -> list:
    """(b) PDF soft-hyphenation joins: a line ending in a single hyphen
    whose next line starts lowercase is one wrapped word. The hyphen is
    dropped when the fragment is a soft-split syllable ("halluci-" +
    "nated" -> "hallucinated", "pro-" + "gram" -> "program") and kept
    when it is a compound stem ("multi-" + "class" -> "multi-class").
    Inflectional tails after a compound stem still fuse ("test-" + "ing"
    -> "testing")."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if (stripped.endswith("-") and not stripped.endswith("--")
                and not _PREPASS_HR_RE.match(stripped)
                and i + 1 < len(lines)):
            nxt = lines[i + 1].strip()
            if nxt[:1].islower():
                stem = stripped[:-1]
                part1 = stem.rsplit(" ", 1)[-1].lower()
                tokens = nxt.split()
                part2 = tokens[0].strip(":;,()[]\"'").lower() if tokens else ""
                if (part1 in _COMPOUND_HYPHEN_PREFIXES
                        and part2 not in _SOFT_SUFFIX_TAILS):
                    out.append(stripped + nxt)      # multi- + class
                else:
                    out.append(stem + nxt)          # halluci- + nated
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def _join_plus_fragments(lines: list) -> list:
    """(a) Fragments that parsed as their own bullets (run d946bb5f):

    - the sentence's plus sign landing at a PDF wrap start ('...to a
      CAMeL-BERT' / '+ FAISS retrieval stack, deployed...') joined back
      into the bullet above when that bullet is a '-'/'*'/'•' item
      without trailing sentence punctuation - a real '+' list is
      protected because its previous item is itself a plus line;
    - an uppercase-starting tail after an unterminated bullet/continuation
      ('...F1 0.961, and 0.995' / 'AUC-PR - a production-ready result
      deployed live via Streamlit.') - a hard wrap the orphan-continuation
      rule cannot see because it starts with a capital. Gates: only under
      Projects/Experience/Education, the tail carries no date/pipe/link/
      bold and stays short (a following entry title always carries one of
      those shapes), and section headings are excluded on both sides.
    """
    out = []
    section = None
    for line in lines:
        sec = _prepass_section_of(line)
        if sec:
            section = sec
            out.append(line)
            continue
        m = _PREPASS_PLUS_LINE_RE.match(line)
        if m:
            joined = False
            if out:
                prev = out[-1].rstrip()
                pm = re.match(r"^\s*[-*•]\s+(.*)$", prev)
                if pm:
                    ptext = pm.group(1).rstrip()
                    if ptext and ptext[-1] not in ".!?:":
                        out[-1] = f"{prev} + {m.group(1).strip()}"
                        joined = True
            if not joined:
                out.append(line)
            continue
        s = line.strip()
        if (out and section in _PREPASS_RESHAPE_SECTIONS and s
                and len(s) <= 100
                and not _PREPASS_BULLET_RE.match(s)
                and not s.startswith(("*", "#", "["))
                and "|" not in s and "**" not in s and "[" not in s
                and _first_date_match(s) is None):
            prev = out[-1].rstrip()
            # the glue partner must be an unterminated bullet OR a
            # lowercase-starting continuation chain - an uppercase TITLE
            # line (a dateless entry heading) is never a wrap of a later
            # line, so the tail below it stays separate
            pm = re.match(r"^\s*[-*•]\s+(\S.*)$", prev)
            if (prev and _prepass_section_of(prev) is None
                    and prev[-1] not in ".!?:"):
                if ((pm is not None and pm.group(1).rstrip()[-1]
                        not in ".!?:")
                        or (pm is None and prev[:1].islower())):
                    out[-1] = f"{prev} {s}"
                    continue
        out.append(line)
    return out


def _join_wrapped_skill_rows(lines: list) -> list:
    """Merge PDF line-wrapped continuations inside Skills sections back onto
    their preceding 'Label: item1, item2,' row so orphan items like
    'Neural Networks' do not become standalone rows."""
    out = []
    section = None
    for line in lines:
        sec = _prepass_section_of(line)
        if sec:
            section = sec
            out.append(line)
            continue
        s = line.strip()
        if (out and section in {"skills", "competencies", "technologies"}
                and s and len(s) <= 80
                and not _PREPASS_BULLET_RE.match(s)
                and not s.startswith(("*", "#", "["))
                and ":" not in s and "|" not in s and "[" not in s):
            prev = out[-1].rstrip()
            if (prev and _prepass_section_of(prev) is None
                    and ":" in prev and prev.endswith(",")):
                out[-1] = f"{prev} {s}"
                continue
        out.append(line)
    return out


_MD_LINK_LEAD_RE = re.compile(r"^\s*\[([^\[\]]+)\]\(([^)]*)\)\s*(.*)$")
# A line holding exactly ONE link construct and nothing else (the "GitHub"
# row under a project title) - absorbed into the reshaped title above it.
_PURE_LINK_LINE_RE = re.compile(r"^\s*\[([^\[\]]+)\]\(([^)]*)\)\s*$")
# The reshaped title's head: bold title followed by zero or more link
# constructs (labels never contain brackets or pipes, so the pipe tail
# starts right after this match).
_RESHAPED_HEAD_RE = re.compile(
    r"^\*\*[^*]+\*\*(?: \[[^\[\]]+\]\([^)]*\))*")


def _reshape_entry_title(line: str) -> str:
    """Canonical entry-title shapes under Projects/Experience/Education
    (the model skipped these units three runs in a row, so the shipped
    shape must parse canonically on its own):

    - A  '[Title](url) [(Badge)](url) [Sep 2025 - Jun 2026 | Live Demo |](url)'
         -> '**Title** [Live Demo](url) [(Badge)](url) | Sep 2025 - Jun 2026'
       (extra link constructs survive as distinct labels - "(Badge)" is
         content, Nov 16 2026; but a span-label construct whose label
         carries the DATE range contributes its date as a plain tail
         field, and a remnant "Live Demo" chip is redundant with the
         title's own label - no date is ever shipped as a link label).
    - C  'Computer Vision & Deep Learning Trainee, NTI Aug - Sep 2025'
         -> '**Computer Vision & Deep Learning Trainee** | NTI | Aug - Sep 2025'
       (the org rode inside the right-aligned date column).
    - B  'Fil-RAG-Goal: Arabic Football Q&A (Hybrid RAG) Feb 2026 - Jun
         2026 | Live Demo'
         -> '**Fil-RAG-Goal: ...** | Live Demo | Feb 2026 - Jun 2026'.

    Gates: no leading bullet marker, no bold/heading marker, no link
    construct except at position 0 (case A), and a real date range.
    """
    s = line.strip()
    if not s or _PREPASS_BULLET_RE.match(s) or s.startswith(("*", "#")):
        return line

    # A: link-led title
    m = _MD_LINK_LEAD_RE.match(s)
    if m:
        title, url, rest = (m.group(1).strip(), m.group(2).strip(),
                            m.group(3).strip())
        if (title and url and len(title) <= 90
                and _first_date_match(rest) is not None):
            # Extra link constructs survive as distinct labels, EXCEPT a
            # span-label construct whose label carries the date range
            # ("[Sep 2025 - Jun 2026 | Live Demo |](url)" from a PDF row
            # span): its date becomes a plain tail field and a remnant
            # "Live Demo" chip is redundant with the title's own label.
            extra_links: list[str] = []
            date_fields: list[str] = []
            for lbl, tgt in _MD_LINK_FULL_RE.findall(rest):
                lbl = lbl.strip()
                tgt = tgt.strip()
                if not tgt:
                    continue
                link_pieces: list[str] = []
                for p in (pp.strip() for pp in
                          re.split(r"\s*[|·]\s*", lbl) if pp.strip()):
                    if _PREPASS_DATE_ANY_RE.search(p):
                        date_fields.append(p)
                    elif p.lower().strip(" -–—:,.") in ("live demo", "demo"):
                        continue
                    else:
                        link_pieces.append(p)
                if link_pieces:
                    extra_links.append(
                        f" [{' | '.join(link_pieces)}]({tgt})")
            # plain-text leftovers of rest are tail fields too
            rest_clean = _MD_LINK_FULL_RE.sub("", rest)
            fields = [p.strip() for p in rest_clean.split("|") if p.strip()]
            fields += date_fields
            fields = [p for p in fields
                      if p and p.lower().strip(" -–—") != "live demo"]
            tail = f" | {' | '.join(fields)}" if fields else ""
            return (f"**{title}** [Live Demo]({url})"
                    f"{''.join(extra_links)}{tail}")
        return line

    if s.startswith("[") or "**" in s:
        return line

    # C: comma org/date ('Title, ORG <date>')
    cm = _first_date_match(s)
    if cm is not None and "[" not in s:
        before = s[:cm[0]].rstrip()
        if "," in before:
            title, org = before.rsplit(",", 1)
            title, org = title.strip(), org.strip()
            date = s[cm[0]:].strip()
            if title and org and date and 3 <= len(title) <= 90 \
                    and title[-1] not in ".!?:":
                return f"**{title}** | {org} | {date}"

    # B: plain title + mid-line date + pipe fields after it (including
    # trailing [Live Demo](url) links preserved by PDF extraction)
    if "|" in s:
        link_spans = [(m.start(), m.end()) for m in _MD_LINK_FULL_RE.finditer(s)]
        for cm in _PREPASS_DATE_ANY_RE.finditer(s):
            if any(a <= cm.start() < b for a, b in link_spans):
                continue
            if cm.start() and not s[cm.start() - 1].isspace():
                continue
            title = s[:cm.start()].strip()
            fields_after = s[cm.start():].strip()
            if not title or not fields_after or "[" in title:
                continue
            if title[-1] in ".!?:" or not 8 <= len(title) <= 90:
                continue
            date_and_rest = fields_after.split("|")
            date = date_and_rest[0].strip()
            rest = [p.strip() for p in date_and_rest[1:] if p.strip()]
            if date and rest:
                links = []
                plain_fields = []
                for field in rest:
                    found_links = _MD_LINK_FULL_RE.findall(field)
                    remainder = _MD_LINK_FULL_RE.sub("", field).strip(" |-–—")
                    for lbl, tgt in found_links:
                        if lbl.strip() and tgt.strip():
                            links.append(f"[{lbl.strip()}]({tgt.strip()})")
                    if remainder:
                        plain_fields.append(remainder)
                link_part = (" " + " ".join(links)) if links else ""
                tail_part = (" | " + " | ".join(plain_fields)) if plain_fields else ""
                return f"**{title}**{link_part}{tail_part} | {date}"
    return line


def _first_date_match(s: str):
    """First date-range match whose start sits on a token boundary (the
    preceding char is whitespace or line start) - guards against matching
    the dash inside a hyphenated word ('e-commerce - retail 2024')."""
    for m in _PREPASS_DATE_ANY_RE.finditer(s):
        if m.start() == 0 or s[m.start() - 1].isspace():
            return (m.start(), m.group(0))
    return None


def normalize_markdown_links(text: str) -> str:
    """Deterministic cleanup of PDF-ingestion damage in the improver's
    INPUT (bf58e700, Nov 15 2026) - run once before the writer sees the
    resume so the broken shapes never reach the prompt, AND a unit the
    writer skips ships the cleaned text instead of the damage:

    - a link label spanning pipe-separated fields ("[City | phone |](url)")
      is flattened to its visible text;
    - a raw list marker riding inside a link label ("[- AWS ...](url)") or
      in front of a link ("- [GitHub](url)") is stripped;
    - links repeated on one line with the SAME target AND label keep the
      first instance only (no target is lost, so the target-set gates
      agree; a same-target link with a new label is content, not a dup);
    - a line holding ONLY link constructs whose every target AND label
      already appeared on an earlier line is dropped; a same-target line
      with a NEW label survives ("GitHub" linking the same featured URL
      is the user's visible label, Nov 16 2026).

    Shape repairs for wrapped PDF extraction (run d946bb5f, Nov 16 2026):
    the education two-line merge (degree line + university line with date
    -> one canonical 4-field subheading line), soft-hyphenation joins
    ("halluci-" + "nated"), the sentence's plus sign at a wrap start
    joined back into the bullet above, and entry-title reshapes to the
    canonical bold/date-column shapes under Projects/Experience/
    Education.

    Clean markdown passes through byte-identical.
    """
    text = (text or "").replace("\r\n", "\n")
    lines = text.split("\n")
    # shape repairs first: education merge (consumes both halves so no
    # later rule can grab them), then the wrapped-fragment joins.
    lines = _merge_education_lines(lines)
    lines = _join_dehyphenated(lines)
    lines = _join_plus_fragments(lines)
    lines = _join_wrapped_skill_rows(lines)
    out: list[str] = []
    seen: set = set()
    seen_labels: set = set()
    section: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        sec = _prepass_section_of(line)
        if sec:
            section = sec
        # 1. flatten span-label constructs to their visible text
        line = _CONTACT_SPAN_LINK_RE.sub(lambda m: m.group(1), line)
        # 2. strip a raw list marker from inside a link label ...
        line = _MD_LINK_FULL_RE.sub(
            lambda m: f"[{_BULLET_MARKER_RE.sub('', m.group(1).strip())}]"
                      f"({m.group(2)})",
            line)
        #    ... and from in front of the line's first link
        line = re.sub(r"^(\s*)[-*•]\s+(?=\[)", r"\1", line)
        # 3. duplicate (target AND label) links on one line: keep the first
        #    (absorb one adjacent space at the cut so no double space
        #    remains). A repeated URL with a NEW label is real content -
        #    the source links the same featured-project URL from the
        #    title, a "(Graduation Project)" badge, and a "GitHub" row,
        #    and those labels are the user's own text (run d946bb5f).
        parts: list[str] = []
        last = 0
        line_targets: set = set()
        line_labels: set = set()
        for m in _MD_LINK_FULL_RE.finditer(line):
            target = _norm_link(m.group(2))
            label = m.group(1).strip().lower()
            if target in line_targets and label in line_labels:
                end = m.end()
                if end < len(line) and line[end] == " ":
                    end += 1
                parts.append(line[last:m.start()])
                last = end
                continue
            line_targets.add(target)
            line_labels.add(label)
        parts.append(line[last:])
        line = "".join(parts)
        # 4. a pure-link line whose every target AND label was already
        #    seen on an earlier line is dropped - a NEW label survives
        #    ("GitHub" linking the same URL is the user's visible label)
        if line_targets:
            if (not _MD_LINK_FULL_RE.sub("", line).strip()
                    and line_targets <= seen
                    and line_labels <= seen_labels):
                continue
            seen |= line_targets
            seen_labels |= line_labels
        # 5. entry-title reshapes under Projects/Experience/Education,
        #    absorbing a following pure-link line (the "GitHub" row under
        #    a project title) into the title's link row so it parses
        #    as one entry header instead of a stray header field. Only a
        #    line the reshape CHANGED absorbs, so clean canonical input
        #    stays byte-identical.
        if section in _PREPASS_RESHAPE_SECTIONS:
            reshaped = _reshape_entry_title(line)
            if reshaped != line:
                m_next = (_PURE_LINK_LINE_RE.match(lines[i])
                          if i < len(lines) else None)
                if m_next:
                    link_str = (f" [{m_next.group(1).strip()}]"
                                f"({m_next.group(2).strip()})")
                    hm = _RESHAPED_HEAD_RE.match(reshaped)
                    if hm:
                        reshaped = (f"{reshaped[:hm.end()]}{link_str}"
                                    f"{reshaped[hm.end():]}")
                    else:
                        reshaped = f"{reshaped}{link_str}"
                    i += 1
            line = reshaped
        out.append(line)
    return "\n".join(out)


def accumulate_unique(existing: Optional[list], added: Optional[list]) -> list:
    """Order-preserving accumulation of state list values across passes."""
    out = list(existing or [])
    for item in (added or []):
        if item and item not in out:
            out.append(item)
    return out


def evaluate_polish_gate(
    state: dict,
    *,
    score_threshold: Optional[int] = None,
    max_iterations: Optional[int] = None,
    polish_skip_ats_threshold: Optional[int] = None,
    polish_skip_quality_threshold: Optional[int] = None,
) -> tuple[bool, str]:
    """Determines whether another polish pass should run and returns an auditable reason.

    Skips polishing ONLY when review failed, MAX_ITERATIONS is reached, sanitized
    feedback is empty, OR all three quality/coverage conditions hold together:
    ``ats_percentage >= threshold`` AND ``0`` high-severity gaps AND
    ``llm_quality_score >= quality_threshold``.
    """
    from . import config as _cfg

    s_thresh = _cfg.SCORE_THRESHOLD if score_threshold is None else score_threshold
    max_it = _cfg.MAX_ITERATIONS if max_iterations is None else max_iterations
    ats_thresh = (_cfg.POLISH_SKIP_ATS_THRESHOLD
                  if polish_skip_ats_threshold is None else polish_skip_ats_threshold)
    q_thresh = (_cfg.POLISH_SKIP_QUALITY_THRESHOLD
                if polish_skip_quality_threshold is None else polish_skip_quality_threshold)

    if state.get("review_failed", False):
        return False, "skipped: reviewer failed - proceeding with current draft"
    if int(state.get("iteration") or 0) >= max_it:
        return False, f"skipped: max iterations ({max_it}) reached"
    feedback = (state.get("feedback") or "").strip()
    if not feedback:
        return False, "skipped: no actionable reviewer feedback remaining"

    ats_pct = float(state.get("ats_percentage") or 0.0)
    llm_q = int(state.get("llm_quality_score") or 0)
    req_matches = state.get("requirement_matches")
    if isinstance(req_matches, list):
        high_gaps = sum(1 for r in req_matches if isinstance(r, dict) and r.get("status") == "gap")
    else:
        high_gaps = 1 if (state.get("missing_skills") or "").strip() else 0

    if ats_pct >= ats_thresh and high_gaps == 0 and llm_q >= q_thresh:
        return (
            False,
            f"skipped: ATS {ats_pct:.0f}% >= {ats_thresh}%, "
            f"0 high-severity gaps, quality {llm_q} >= {q_thresh}",
        )

    score = int(state.get("score") or 0)
    if score >= s_thresh:
        return (
            True,
            f"polisher: feedback/gaps remain (ATS {ats_pct:.0f}%, gaps={high_gaps}, quality={llm_q})",
        )
    return (
        True,
        f"polisher: composite {score} < {s_thresh} (ATS {ats_pct:.0f}%, gaps={high_gaps}, quality={llm_q})",
    )


# Re-exported for backwards compatibility (implementation lives in agent/fallback_pdf.py).
from .fallback_pdf import _html_to_pdf, _sanitize_html_for_pdf  # noqa: E402, F401
