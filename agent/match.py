"""MR2: deterministic requirement-level match record (explainable fit table).

Dossier-study pattern (AienTech/dossier `match-requirements.ts`), re-implemented
from scratch in this repo's own matcher machinery (AGPL-safe: patterns only).
The philosophy carries over: segmentation, classification, and reasons ALL run
in code - no model decides what "covered" means - while reusing the ATS
semantic verifier's reclassification set (passed in by ats_check) so "the
resume expresses it in different words" counts as coverage consistently in
both views. No extra LLM call is made here.

The ATS percentage is untouched (it is calibrated and honest); the table
answers a different question: WHICH requirements are covered. A front-page
"ATS 100%" produced by echoing JD vocabulary with zero requirement coverage is
exposed instantly as gap rows (the analytical failure class keyword matching
alone cannot see).

Rows are recomputed per ATS pass and follow the best-of guard (ats_check keeps
the winner's table), so the payload always describes the resume that ships -
the polisher can never leave the table describing a regressed draft.
"""

import re

from .ats import (
    _classify_keyword_tier,
    _extract_keywords,
    _flatten_link_targets,
    _keyword_found_in_text,
    _strip_boilerplate_sections,
)

# Cap keeps the payload light and the table readable; requirement-dense JDs
# truncate by first appearance (top-of-posting requirements lead any JD).
_MAX_REQUIREMENTS = 12

# Bare fragments ("Requirements", "You have:") never make rows; real
# requirement lines carry at least an amount of verbage.
_MIN_CANDIDATE_WORDS = 4

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25cf]|\d{1,2}[.)])\s+")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Terms that describe the ROLE or the generic context of a posting, not
# requirements a resume can carry evidence for. Probe evidence (Nov 10,
# golden JD + run JDs): they dominated match/gap reasons with noise and would
# tell users to "add" engineer/developer/infrastructure - role filler filtered
# from the table's evidence/missing lists (statuses and ATS score unaffected).
# "apache/event/streaming" are prose around real skills ("Apache Kafka for
# event streaming"); the skill itself is the evidence.
_NON_EVIDENCE_TERMS = {
    "engineer", "engineers", "developer", "development",
    "infrastructure", "provisioning", "version control",
    "workflows", "automation", "data", "agent", "agents",
    "apache", "event", "streaming", "integrations",
}


def _normalize_jd_text(job_text: str) -> str:
    """Normalizes literal-escape payloads into real line structure.

    The golden-fixture bug class (runs before Sep 2026: literal "\n" escape
    sequences reached the engine and glued words together) never reaches the
    segmenter: "\\n" becomes a genuine line break here, so bullet parsing sees
    the payload a human read.
    """
    text = (job_text or "").replace("\\r", "\n").replace("\\n", "\n")
    text = text.replace("\\t", " ").replace("\t", " ").replace("\r", "\n")
    return text


def _clean_candidate(candidate: str) -> str:
    text = _URL_RE.sub("", candidate)
    return re.sub(r"\s+", " ", text).strip().rstrip(" -:;")


def _split_requirement_candidates(job_text: str) -> list[tuple[str, bool]]:
    """Segments the JD into (candidate, is_structured) requirement lines.

    One candidate per non-empty source line: bullets/numbered items keep their
    content (is_structured=True), a heading is a marker only (a bare
    "Requirements" line is not a requirement), and a prose line carrying
    several sentences splits at the sentence enders (is_structured=False).
    Line-based parsing keeps skill lists that use plain (non-bulleted) lines
    from bleeding into each other (live golden-JD evidence). Runs on the SAME
    boilerplate-stripped text as ATS keyword extraction, so perk / legal /
    application-logistics sections can never become requirement rows.
    """
    text = _strip_boilerplate_sections(_normalize_jd_text(job_text))
    candidates: list[tuple[str, bool]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _MD_HEADING_RE.match(line):
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            content = line[bullet.end():].strip()
            if content:
                candidates.append((content, True))
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if sentence:
                candidates.append((sentence, False))
    return candidates


_DECLINED_FRAMING_WORDS = frozenset({
    "do", "not", "feature", "never", "use", "using", "mention", "include",
    "avoid", "exclude", "declined", "decline", "add", "or", "and", "the",
    "with", "for", "from", "to", "in", "on", "of", "no", "tool", "tools",
    "skill", "skills", "experience", "request", "requested",
})


def _extract_declined_tools(declined_requests) -> set[str]:
    """Normalize declined tool/skill names from declined_requests entries."""
    from .ats import _STOPWORDS
    from .tool_claims import matching_tools

    declined: set[str] = set()
    for item in (declined_requests or ()):
        text = str(item or "").strip()
        if not text:
            continue
        declined |= {t.lower() for t in matching_tools(text)}
        for tok in re.findall(r"[a-zA-Z0-9#+/.:-]{2,}", text.lower()):
            cleaned = tok.strip(".-/:")
            if (
                cleaned
                and cleaned not in _DECLINED_FRAMING_WORDS
                and cleaned not in _STOPWORDS
                and any(c.isalpha() for c in cleaned)
            ):
                declined.add(cleaned)
    return declined


def _extract_req_named_tools(requirement: str) -> list[str]:
    """Extract specific named tools (canonical, alphanumeric, Tier-1, short-tech)
    mentioned directly inside a single requirement line.
    """
    from .ats import _SHORT_TECH_TERMS, _is_alnum_tool_token
    from .tool_claims import matching_tools

    found: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        t = (term or "").strip(".-/:").lower()
        if t and t not in seen:
            seen.add(t)
            found.append(t)

    # Maintain stable appearance order from requirement tokens first, then canonical.
    for raw in re.findall(r"[a-zA-Z0-9#+/.:-]{2,}", requirement.lower()):
        tok = raw.strip(".-/:")
        if not tok or not any(c.isalpha() for c in tok):
            continue
        if (
            _is_alnum_tool_token(tok)
            or tok in _SHORT_TECH_TERMS
            or _classify_keyword_tier(tok) == 3
            or "#" in tok
            or "+" in tok
        ):
            _add(tok)

    for canon in sorted(matching_tools(requirement)):
        _add(canon)

    return found


def _classify_row(requirement: str, evidence: list[str], resume_lower: str,
                  semantic_keys: set[str], *,
                  declined_tools: set[str] = frozenset(),
                  named_tools: set[str] = frozenset(),
                  canonical_evidence: set[str] = frozenset()) -> dict:
    """Classifies one requirement against the resume with human reasons.

    found/missing are the requirement's OWN key terms (the JD-vocabulary
    members the line names plus any specific named tools in the line), not a
    model's opinion: named tools require exact or alias presence in the
    resume/notes (never generic neighbor words or stem guessing) and declined
    tools can never be marked covered.
    """
    from .ats import _SYNONYMS, _boundary_contains, _is_alnum_tool_token
    from .tool_claims import matching_tools

    found: list[str] = []
    missing: list[str] = []
    for keyword in evidence:
        kw_l = keyword.lower()
        kw_canon = {t.lower() for t in matching_tools(keyword)}
        if kw_l in declined_tools or (kw_canon & declined_tools):
            missing.append(keyword)
            continue

        is_tool = (
            kw_l in named_tools
            or bool(kw_canon)
            or _classify_keyword_tier(keyword) == 3
            or _is_alnum_tool_token(kw_l)
        )
        if is_tool:
            exact_hit = _boundary_contains(kw_l, resume_lower)
            alias_hit = any(
                _boundary_contains(alias.lower(), resume_lower)
                for alias in _SYNONYMS.get(kw_l, [])
            )
            canon_hit = bool(kw_canon & canonical_evidence)
            hit = exact_hit or alias_hit or canon_hit
        else:
            hit, _method = _keyword_found_in_text(keyword, resume_lower)

        if hit or kw_l in semantic_keys:
            found.append(keyword)
        else:
            missing.append(keyword)

    if not missing:
        return {"requirement": requirement, "status": "covered",
                "evidence": found, "missing": [],
                "reason": f"Covers all key terms ({', '.join(found)})."}
    if not found:
        return {"requirement": requirement, "status": "gap",
                "evidence": [], "missing": missing,
                "reason": f"Missing from the resume: {', '.join(missing)}."}
    return {"requirement": requirement, "status": "partial",
            "evidence": found, "missing": missing,
            "reason": f"Covers {', '.join(found)}; missing {', '.join(missing)}."}


def compute_requirement_match_table(job_text: str, resume_text: str,
                                    semantic_matches=None,
                                    human_notes: str = "",
                                    declined_requests=None) -> list[dict]:
    """Deterministic per-requirement fit rows for one JD against one resume.

    Returns a list of ``{"requirement", "status", "evidence", "missing",
    "reason"}`` dicts in JD order (capped at the first _MAX_REQUIREMENTS
    qualifying lines), where ``status`` is "covered" / "partial" / "gap".
    """
    from .ats import _boundary_contains
    from .tool_claims import matching_tools

    vocabulary = _extract_keywords(job_text or "", top_n=20)
    semantic_keys = {(key or "").lower() for key in (semantic_matches or []) if key}
    declined_tools = _extract_declined_tools(declined_requests)

    combined_parts = [resume_text or ""]
    if human_notes and str(human_notes).strip():
        combined_parts.append(str(human_notes))
    combined_raw = "\n".join(combined_parts)
    evidence_clean = _flatten_link_targets(
        combined_raw.replace('\r', ' ').replace('\n', ' ')
        .replace('\t', ' ').replace('\\r', ' ').replace('\\n', ' ')
        .replace('\\t', ' '))
    resume_lower = evidence_clean.lower()
    canonical_evidence = {t.lower() for t in matching_tools(evidence_clean)}

    rows: list[dict] = []
    seen: set[str] = set()
    for candidate, is_structured in _split_requirement_candidates(job_text):
        if len(rows) >= _MAX_REQUIREMENTS:
            break
        requirement = _clean_candidate(candidate)
        if len(requirement.split()) < _MIN_CANDIDATE_WORDS:
            continue
        key = re.sub(r"[^a-z0-9]+", "", requirement.lower())
        if not key or key in seen:
            continue

        req_lower = requirement.lower()
        hits = [
            keyword
            for keyword in vocabulary
            if _keyword_found_in_text(keyword, req_lower)[0]
        ]
        # Augment hits with specific named tools in this requirement so top_n
        # truncation or short/alphanumeric names never let generic neighbor
        # words mark an absent or declined tool as 'covered'.
        req_named = _extract_req_named_tools(requirement)
        for decl in sorted(declined_tools):
            if _boundary_contains(decl, req_lower) and decl not in req_named:
                req_named.append(decl)

        named_set = set(req_named)
        existing_canon = set()
        for kw in hits:
            existing_canon.add(kw.lower())
            existing_canon |= {t.lower() for t in matching_tools(kw)}

        for tool in req_named:
            tool_canon = {t.lower() for t in matching_tools(tool)}
            already_covered = (
                tool in existing_canon
                or bool(tool_canon & existing_canon)
                or any(tool in kw.lower() or kw.lower() in tool for kw in hits)
            )
            if not already_covered:
                hits.append(tool)
                existing_canon.add(tool)
                existing_canon |= tool_canon

        # Role-filler terms never carry evidence: they describe the ROLE a
        # posting hires for, not something a resume carries proof of (probe
        # evidence, Nov 10: they otherwise dominate reasons as noise).
        evidence = [kw for kw in hits if kw.lower() not in _NON_EVIDENCE_TERMS]
        # Qualification: structured lines (bullets/numbered items) qualify
        # with any real evidence; prose sentences must keep at least one
        # Tool/Concept term (tier >= 2) or specific named tool after filtering.
        if is_structured:
            qualifies = bool(evidence)
        else:
            qualifies = any(
                _classify_keyword_tier(kw) >= 2 or kw.lower() in named_set
                for kw in evidence
            )
        if not qualifies:
            continue
        seen.add(key)
        rows.append(_classify_row(
            requirement,
            evidence,
            resume_lower,
            semantic_keys,
            declined_tools=declined_tools,
            named_tools=named_set,
            canonical_evidence=canonical_evidence,
        ))
    return rows
