"""Round 13 (roadmap 12): Google XYZ bullet-impact audit - deterministic.

Audits every achievement bullet against Google's XYZ formula ("Accomplished
[X] as measured by [Y] by doing [Z]"): does the bullet lead with a strong
action verb (Z) and state a measured outcome (X/Y)? Bullets that do neither
are duty statements - the shape recruiters and ATS reviewers skim past.

DESIGNED deterministic-only, zero LLM calls (the MR2 Fit-table precedent:
classification runs in code, so the audit costs nothing, adds no ladder
pressure, and can never disagree between iterations). The parked LLM pass is
documented at the bottom of this file as a design seam: when trace data shows
the deterministic audit's blind spots matter, one structured call (the
``SemanticReview`` pattern, the same schema-repair ladder, the same
"degrade to no-op on failure" direction) could reclassify bullets the
verb/metric heuristics cannot see - wired as ``bullet_impact`` state, it
would follow the SAME best-of guard as the requirement table.

Quantification reuses the integrity scanner's metric vocabulary
(:data:`agent.helpers._METRIC_TOKEN_RE` - percentages, X-factors, ms/mm
rates, fraction pass-rates) PLUS the broader quantification class an impact
statement can carry ($ amounts, K/M/B scales, curated count units). Plain
numbers stay exempt on purpose - years ("in 2024"), versions ("Python
3.12"), and model names ("GPT-4") are not measured outcomes.

Rows follow the MR2 row contract: everything a UI needs is computed here and
the reason strings are human-readable. The table travels with the winning
draft via the best-of guard (ats_check), so a regressed polish pass can never
ship an audit describing a resume that is not the one in the download.
"""
import re

from .helpers import _METRIC_TOKEN_RE, _safe_truncate
from .resume_parser import iter_achievement_bullets

# Payload/readability cap: a 1-page resume carries ~12-20 bullets; 40 is a
# generous ceiling (the same "first N in document order" contract as the
# requirement table).
_MAX_ROWS = 40

# Row text cap: the audit reasons about shape, not full text; the resume tab
# holds the complete bullet.
_MAX_BULLET_CHARS = 160

# Education sections are skipped by design: their bullets are coursework
# lists ("Coursework: ML, DL, CV"), not impact statements - grading them
# would be noise (live parser evidence).
_EDU_SECTION_RE = re.compile(
    r"education|academic|university|college|school", re.IGNORECASE)

# --- Quantification signals -------------------------------------------------
# 1. The integrity vocabulary (percentages, X-factors, ms/mm, pass-rates).
# 2. Dollar amounts ("saved $1.2M", "$50,000").
# 3. Scaled numbers ("10K events", "3M requests", "1.5 million users").
# 4. Curated count units ("2,195 instances", "12 services", "40 users").
# 5. Precision-point deltas ("34 points", "3.5-point gain") - the
#    retrieval/NLP vocabulary lives entirely outside 1-4 (live-trace
#    evidence cab7f3f7, Nov 15: a 34-point P@1 improvement audited LOW
#    with the false reason "no outcome is measured").
# 6. Explicit before->after decimal deltas ("WER 42.69 -> 20.61",
#    "P@1 0.64 -> 0.86") - the arrow IS the measured-outcome signature.
#    Deliberately NOT the word "to" ("2019 to 2020" is a date range).
_DOLLAR_RE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[KMB]\b|thousand|million|billion))?",
    re.IGNORECASE)
_SCALED_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:[KMB]\b|thousand|million|billion)",
    re.IGNORECASE)
_COUNT_STEMS = (
    "user", "customer", "client", "tenant", "request", "transaction",
    "query", "record", "row", "entry", "instance", "service", "system",
    "project", "team", "engineer", "people", "student", "participant",
    "applicant", "candidate", "article", "paper", "report", "document",
    "dataset", "model", "test", "case", "repo", "repository", "commit",
    "deployment", "incident", "ticket", "bug", "hour", "day", "week",
    "month", "office", "location", "country", "store", "product",
    "feature", "endpoint", "microservice", "pipeline", "workflow",
    "vendor", "partner", "account", "review", "screening", "interview",
    "facility", "vehicle", "device", "sensor", "ward", "clinic", "lab",
    "hospital", "campus", "event", "session", "shipment", "order",
)
_COUNT_UNITS_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s+|-)(?:" + "|".join(_COUNT_STEMS) + r")s?\b",
    re.IGNORECASE)
_DELTA_SCORE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:\u2192|->)\s*\d+(?:\.\d+)?\b")
_POINTS_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s+|-)points?\b", re.IGNORECASE)

_QUANT_RES = (_METRIC_TOKEN_RE, _DOLLAR_RE, _SCALED_RE, _COUNT_UNITS_RE,
              _DELTA_SCORE_RE, _POINTS_RE)


def _quant_tokens(text: str, cap: int = 4) -> list[str]:
    """The measured-outcome tokens in a bullet, integrity vocabulary first."""
    found: list[str] = []
    seen: set[str] = set()
    for regex in _QUANT_RES:
        for match in regex.finditer(text):
            token = match.group(0).strip()
            key = re.sub(r"[\s$]+", "", token.lower())
            if key and key not in seen:
                seen.add(key)
                found.append(token)
            if len(found) >= cap:
                return found
    return found


# --- Verb signals ------------------------------------------------------------
# Google-coached accomplishment verbs (Z = "by doing" with a strong verb).
# Deliberately NOT exhaustive: unknown openers are neutral, they only lose
# the "strong" credit - they are never called weak.
_STRONG_VERBS = frozenset({
    "accomplished", "achieved", "accelerated", "architected", "automated",
    "built", "consolidated", "created", "cut", "delivered", "deployed",
    "designed", "developed", "devised", "directed", "doubled", "drove",
    "eliminated", "engineered", "enhanced", "established", "exceeded",
    "expanded", "formulated", "founded", "generated", "grew", "halved",
    "implemented", "improved", "increased", "initiated", "integrated",
    "introduced", "launched", "led", "lowered", "managed", "mentored",
    "migrated", "modernized", "negotiated", "onboarded", "optimized",
    "orchestrated", "overhauled", "owned", "pioneered", "prototyped",
    "rebuilt", "reduced", "refactored", "redesigned", "resolved", "saved",
    "scaled", "secured", "set", "shipped", "shortened", "simplified",
    "spearheaded", "standardized", "streamlined", "strengthened",
    "supervised", "trained", "transformed", "trimmed", "tripled",
    "unified", "upgraded", "validated", "won",
})

# Duty/standby phrasings - the opener itself proves no accomplishment verb
# was attempted ("Responsible for maintaining...", "Worked on...").
_WEAK_OPENERS = (
    re.compile(r"^responsible\b"), re.compile(r"^responsibilities\b"),
    re.compile(r"^dut(y|ies)\b"), re.compile(r"^tasked\b"),
    re.compile(r"^worked\b"), re.compile(r"^helped\b"),
    re.compile(r"^assisted\b"), re.compile(r"^participated\b"),
    re.compile(r"^involved\b"), re.compile(r"^in charge of\b"),
    re.compile(r"^part of\b"), re.compile(r"^member of\b"),
    re.compile(r"^was\b"), re.compile(r"^were\b"),
    re.compile(r"^familiar\b"), re.compile(r"^handled\b"),
    re.compile(r"^used\b"), re.compile(r"^did\b"), re.compile(r"^made\b"),
    re.compile(r"^attended\b"), re.compile(r"^learned\b"),
    re.compile(r"^studied\b"),
)

_FIRST_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*")


def _first_word(text: str) -> str:
    """The bullet's opening word, 'Co-' prefixed forms folded to the verb
    ('Co-led a team' audits as 'led')."""
    match = _FIRST_WORD_RE.match(text)
    if not match:
        return ""
    word = match.group(0).lower()
    return word[3:] if word.startswith("co-") else word


def _weak_opener(text: str) -> str:
    """The matched duty/standby phrase at the bullet start, or ''."""
    lower = text.lower()
    for regex in _WEAK_OPENERS:
        match = regex.match(lower)
        if match:
            return match.group(0)
    return ""


def grade_bullet(text: str) -> dict:
    """One bullet's XYZ verdict: ``{"grade", "verb", "metrics", "reason"}``.

    Grades: "high" (strong verb + measured outcome), "medium" (exactly one
    of the two), "low" (neither - a duty statement). The reason names the
    signal that is present and the one that is missing, so the fix is
    actionable without reading the formula first.
    """
    metrics = _quant_tokens(text)
    word = _first_word(text)
    strong = word in _STRONG_VERBS
    weak = _weak_opener(text)
    metric_list = f"({', '.join(metrics)})" if metrics else ""

    if metrics and strong:
        return {"grade": "high", "verb": word, "metrics": metrics,
                "reason": (f"Opens with '{word.capitalize()}' and measures "
                           f"the outcome {metric_list}.")}
    if metrics:
        if weak:
            reason = (f"Measures the outcome {metric_list} but opens with "
                      f"the weak phrase '{weak}' - lead with an action "
                      "verb.")
        elif word:
            reason = (f"Measures the outcome {metric_list} but the opener "
                      f"'{word}' is not a strong action verb - lead with "
                      "what you did.")
        else:
            reason = (f"Measures the outcome {metric_list} but does not "
                      "open with an action verb - lead with what you did.")
        return {"grade": "medium", "verb": word or None,
                "metrics": metrics, "reason": reason}
    if strong:
        return {"grade": "medium", "verb": word, "metrics": [],
                "reason": (f"Opens with '{word.capitalize()}' but states no "
                           "measured outcome - add the result "
                           "(%, count, $, time).")}
    if weak:
        reason = (f"Neither a strong opener ('{weak}') nor a measured "
                  "outcome - rewrite as [action verb] + [result].")
    elif word:
        reason = (f"The opener '{word}' is not a strong action verb and no "
                  "outcome is measured - rewrite as [action verb] + "
                  "[result].")
    else:
        reason = ("No action-verb opener and no measured outcome - rewrite "
                  "as [action verb] + [result].")
    return {"grade": "low", "verb": word or None, "metrics": [],
            "reason": reason}


def compute_bullet_impact(resume_text: str, max_rows: int = _MAX_ROWS
                          ) -> list[dict]:
    """The deterministic XYZ audit of one resume's achievement bullets.

    Returns ``{"bullet", "section", "grade", "verb", "metrics", "reason"}``
    rows in resume order (capped at ``max_rows``). Achievement bullets come
    from :func:`agent.resume_parser.iter_achievement_bullets`, which skips link-only
    bullets and education coursework sections.
    """
    rows: list[dict] = []
    for title, text in iter_achievement_bullets(
            resume_text, skip_section_re=_EDU_SECTION_RE):
        if len(rows) >= max_rows:
            return rows
        verdict = grade_bullet(text)
        rows.append({
            "bullet": _safe_truncate(text, _MAX_BULLET_CHARS, "bullet"),
            "section": title,
            **verdict,
        })
    return rows
