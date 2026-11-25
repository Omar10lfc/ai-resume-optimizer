"""Relevance-weighted trim suggestions when one-page containment fails.

The exporter's last resort is a generic "trim content" warning: true, but it
does not answer the only question the user then has - *what do I cut?* Static
orders ("drop the oldest role first") are wrong when an older bullet speaks
to the posting and a recent one does not. This module scores every line that
can safely leave the resume and returns the five cheapest cuts with a reason,
in code and with no model call.

Signals (adapted from the MIT ai-job-search cutting rule, see NOTICE):

* **Relevance** - does the line hit a keyword or covered requirement from THIS
  posting? Requirement evidence outranks a bare keyword match.
* **Impact** - the Round-13 XYZ grade already computed for bullets; a
  duty-shaped, unmeasured line is cheaper to cut than a quantified one.
* **Uniqueness** - how much of the line already appears in another bullet or
  skills row. A duplicated skills row is the cleanest cut: the concrete bullet
  that repeats it stays.
* **Narrative load** - does the cover letter lean on this line? If cutting it
  would force a paragraph rewrite, it is expensive.

The output is advisory only. Nothing here edits, reorders, or deletes a
resume line; the exporter never changes what ships.
"""
from __future__ import annotations

import re

from .ats import _boundary_contains, _extract_keywords, _simple_stem
from .bullet_impact import grade_bullet
from .helpers import _safe_truncate
from .resume_parser import _is_link_item, _strip_inline, parse_markdown_resume

_MAX_SUGGESTIONS = 5
_MAX_EXCERPT_CHARS = 160

_EDU_SECTION_RE = re.compile(
    r"education|academic|university|college|school", re.IGNORECASE)
_SKILLS_SECTION_RE = re.compile(
    r"skill|competenc|technolog|expertise|stack", re.IGNORECASE)
_SUMMARY_SECTION_RE = re.compile(
    r"summary|profile|objective|about", re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./_-]{2,}")

# Words that carry no distinguishing signal in resume prose. Pronouns and
# glue are removed before overlap math; verbs are kept where they describe
# the work ("built", "shipped") but the most generic ones are dropped.
_STOPWORDS = frozenset({
    "and", "the", "for", "with", "from", "that", "this", "into", "over",
    "through", "using", "used", "use", "via", "per", "its", "their", "our",
    "his", "her", "your", "you", "they", "them", "were", "was", "are", "has",
    "have", "had", "been", "being", "not", "but", "also", "more", "most",
    "than", "then", "when", "where", "which", "who", "whom", "what", "how",
    "all", "any", "each", "every", "both", "some", "such", "other", "same",
    "new", "including", "across", "within", "while", "about", "after",
    "before", "between", "under", "during", "without", "team", "teams",
    "work", "worked", "working", "role", "roles", "project", "projects",
    "business", "company", "companies", "client", "clients", "various",
    "multiple", "several", "well", "strong", "solid", "good", "great",
    "excellent", "experienced", "experience", "skills", "skill",
})


def _tokens(text: str) -> list[str]:
    """Distinct normalized content tokens for overlap comparisons.

    Stems are used so "building" / "built" and "deploy" / "deployed" compare
    equal; the excerpt and reason always keep the author's wording.
    """
    seen: set[str] = set()
    out: list[str] = []
    for word in _WORD_RE.findall(text or ""):
        word = word.lower().strip("._-/")
        if len(word) < 3 or word in _STOPWORDS:
            continue
        stem = _simple_stem(word)
        if stem and stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def _has_metric(text: str) -> bool:
    """Cheap measured-outcome probe for summary lines (no XYZ grade there)."""
    return bool(re.search(
        r"\d[\d,.]*\s*(?:%|percent|x\b|k\b|m\b|b\b|points?\b)"
        r"|\$\s?\d|\b\d+(?:\.\d+)?\s*(?:\u2192|->)\s*\d+",
        text or "", re.IGNORECASE))


def _impact_grade(bullet: str, audit: list[dict]) -> str:
    """The precomputed XYZ grade for a bullet, or recompute it cheaply.

    ``bullet_impact`` rows are truncated at 160 chars; startswith on the
    normalized forms matches the row to its full bullet without a model.
    """
    target = _norm(bullet)
    for row in audit or []:
        row_text = _norm(str(row.get("bullet") or ""))
        if not row_text:
            continue
        if target.startswith(row_text) or row_text.startswith(target[:60]):
            grade = str(row.get("grade") or "")
            if grade in {"high", "medium", "low"}:
                return grade
    return grade_bullet(bullet)["grade"]


def _candidates(resume_text: str) -> list[dict]:
    """Every line eligible for cutting, in resume order.

    Education bullets are excluded by design: their bullets are coursework
    lists, not evidence, and the reference's cut order keeps education until
    the relevance-weighted cuts are exhausted. A skills/summary simple line is
    a candidate as a whole row; the suggestion asks the user to trim it, never
    claims the line is worthless.
    """
    try:
        data = parse_markdown_resume(resume_text or "")
    except Exception:
        return []

    candidates: list[dict] = []
    order = 0

    def _add(section: str, text: str, kind: str) -> None:
        nonlocal order
        clean = _strip_inline(text or "").strip()
        if not clean:
            return
        candidates.append({"section": section, "text": clean,
                           "kind": kind, "order": order})
        order += 1

    for section in data.sections:
        title = (section.title or "").strip()
        if not title:
            continue
        is_education = bool(_EDU_SECTION_RE.search(title))
        is_skills = bool(_SKILLS_SECTION_RE.search(title))
        is_summary = bool(_SUMMARY_SECTION_RE.search(title))

        if not is_education:
            if is_skills:
                for line in section.simple_lines:
                    _add(title, line, "skills")
            elif is_summary:
                for line in section.simple_lines:
                    _add(title, line, "summary")

        if is_education:
            continue
        for entry in section.entries:
            for item in entry.items:
                if _is_link_item(item or ""):
                    continue
                _add(title, item, "bullet")

    return candidates


def _score(candidate: dict, *, posting_terms: list[str],
           evidence_terms: list[str], corpus_tokens: list[str],
           cover_tokens: set[str], audit: list[dict]) -> dict | None:
    """One candidate's cut cost - lower means safer to remove, None if unsafe.

    The score is intentionally additive and small-integer so it can be
    explained: relevance (0-3) + requirement support (0-1) + impact (0-2)
    + uniqueness (0-2) + narrative load (0-2), minus a redundancy discount
    for a skills row that repeats bullets.

    Whole-row candidates are filtered: a skills row is only a candidate when
    at least half of it already appears in a bullet (otherwise "cut this row"
    would delete skills, not redundancy), and a profile line only when it
    carries no posting keyword and no measured result (otherwise it is not
    fluff).
    """
    text = candidate["text"]
    lower = text.lower()
    kind = candidate["kind"]

    hits = sum(1 for term in posting_terms if _boundary_contains(term, lower))
    relevance = min(3, hits)
    support = 1 if any(_boundary_contains(term, lower)
                       for term in evidence_terms) else 0

    if kind == "bullet":
        grade = _impact_grade(text, audit)
        impact = {"high": 2, "medium": 1, "low": 0}[grade]
    else:
        grade = "medium"
        impact = 1  # rows are not XYZ-audited; stay neutral

    own = _tokens(text)
    own_set = set(own)
    shared_resume = own_set & set(corpus_tokens)
    dup_ratio = (len(shared_resume) / len(own_set)) if own_set else 0.0
    uniqueness = 0 if dup_ratio >= 0.6 else 1 if dup_ratio >= 0.3 else 2

    if kind == "skills" and dup_ratio < 0.5:
        return None
    if kind == "summary" and (relevance > 0 or _has_metric(text)):
        return None

    if cover_tokens:
        cover_ratio = (len(own_set & cover_tokens) / len(own_set)
                       if own_set else 0.0)
        narrative = 2 if cover_ratio >= 0.35 else 1 if cover_ratio >= 0.15 else 0
    else:
        narrative = 1  # no letter yet: neutral, never a reason to keep

    cost = relevance + support + impact + uniqueness + narrative
    if kind == "skills" and dup_ratio >= 0.5:
        cost -= 2
    if kind == "summary" and relevance == 0 and not _has_metric(text):
        cost -= 1

    if kind == "skills" and dup_ratio >= 0.5:
        repeated = ", ".join(sorted(shared_resume)[:4])
        reason = ("Skills row repeats terms already shown in a role/project "
                  f"bullet ({repeated}); trim the duplicates and keep the "
                  "concrete evidence.")
    elif kind == "summary" and relevance == 0 and not _has_metric(text):
        reason = ("Profile line restates skills without a posting keyword or "
                  "a measured result.")
    elif relevance == 0 and support == 0:
        if kind == "bullet" and impact == 0:
            reason = ("Duty-shaped and unmeasured, with no posting keyword - "
                      "the cheapest bullet to cut.")
        elif kind == "bullet" and impact >= 2:
            reason = ("No posting keyword appears here, but the result is "
                      "quantified - a later cut, not a free one.")
        else:
            reason = ("No keyword or covered requirement from this posting "
                      "appears in this line.")
    elif impact == 0:
        reason = ("Duty-shaped and unmeasured (XYZ audit: low), so cutting "
                  "it removes no evidence.")
    elif uniqueness == 0:
        reason = "Repeats content already stated elsewhere in the resume."
    elif narrative == 0:
        reason = "The cover letter does not lean on this line."
    else:
        reason = ("Lowest combined relevance and impact among the lines that "
                  "are safe to cut.")

    return {"section": candidate["section"],
            "excerpt": _safe_truncate(text, _MAX_EXCERPT_CHARS, "line"),
            "reason": reason,
            "_cost": cost,
            "_order": candidate["order"]}


def suggest_trims(resume_text: str, *, job_text: str = "",
                  cover_text: str = "",
                  requirement_matches: list | None = None,
                  bullet_impact: list | None = None,
                  overflowed: bool = False,
                  max_suggestions: int = _MAX_SUGGESTIONS) -> list[dict]:
    """The cheapest resume lines to cut, or ``[]`` without an overflow.

    ``{section, excerpt, reason}`` rows ordered cheapest-first. Ties keep
    resume order, so identical inputs always produce identical suggestions.
    Never raises and never mutates its inputs - a malformed resume or an
    unexpected row shape degrades to "no suggestions", never to a blocked
    export.
    """
    if not overflowed or not resume_text:
        return []

    try:
        candidates = _candidates(resume_text)
        if not candidates:
            return []

        posting_terms = [str(term).lower()
                         for term in _extract_keywords(job_text or "", top_n=20)
                         if str(term).strip()]
        evidence_terms: list[str] = []
        for row in requirement_matches or []:
            if row.get("status") not in {"covered", "partial"}:
                continue
            for term in row.get("evidence") or []:
                term = str(term).lower().strip()
                if term and term not in evidence_terms:
                    evidence_terms.append(term)

        corpus: list[str] = []
        for cand in candidates:
            corpus.extend(_tokens(cand["text"]))
        cover_tokens = set(_tokens(cover_text))

        scored = []
        for cand in candidates:
            others = list(corpus)
            for token in _tokens(cand["text"]):
                if token in others:
                    others.remove(token)
            row = _score(
                cand, posting_terms=posting_terms,
                evidence_terms=evidence_terms, corpus_tokens=others,
                cover_tokens=cover_tokens, audit=bullet_impact or [])
            if row is not None:
                scored.append(row)

        scored.sort(key=lambda row: (row["_cost"], row["_order"]))
        return [{"section": row["section"], "excerpt": row["excerpt"],
                 "reason": row["reason"]}
                for row in scored[:max(0, max_suggestions)]]
    except Exception:
        return []
