"""Round 16 (per-bullet typed source rows): deterministic provenance.

For every achievement bullet in the OPTIMIZED resume, which bullet in the
user's ORIGINAL did it come from, and what happened to the measured
outcomes on the way: kept, reworded, added, or silently dropped? The
headline case (live run bf0f7941, Nov 12 2026: the writer's rewrite
shipped only the semantic-search half and the 51.7% / WER accomplishment
vanished) is exactly what the ``dropped`` rows make visible per bullet -
"your original said 96.2% in this bullet; the optimized resume dropped
it" - instead of one aggregate integrity string.

DETERMINISTIC, zero LLM calls (the MR2 Fit-table / round-13 impact-audit
precedent). Matching is a global greedy one-to-one binding on
difflib.SequenceMatcher word-overlap similarity; metric lineage reuses the
integrity scanner's exact tools (``_METRIC_TOKEN_RE`` vocabulary,
whitespace-insensitive compact matching, and the Nov 14 rounding
tolerance), so this table can never contradict the integrity scan on the
same pair of texts: a source ``96.2%`` whose optimized bullet says
``96%`` is a ROUNDING, not a drop; a source ``380 ms`` whose optimized
bullet says ``400 ms`` is a drop.

The table follows the SAME best-of guard as requirement_matches /
bullet_impact (ats_check), so a regressed polish pass can never ship
provenance describing a resume that is not the one in the download.
"""
import difflib
import re

from .bullet_impact import _EDU_SECTION_RE, _quant_tokens
from .helpers import (_claim_rounds_to, _evidence_metric_values,
                      _safe_truncate)
from .latex_render import _is_link_item, _strip_inline, parse_markdown_resume

# Payload/readability cap: matched/new rows (they describe the shipped
# resume) fill first in optimized order, then dropped-source rows in
# original order - the same "first N in document order" contract as the
# requirement table. A 1-page resume carries ~12-20 bullets per side.
_MAX_ROWS = 40

# Row text cap: the same contract as the impact audit (the resume tab holds
# the complete bullet).
_MAX_BULLET_CHARS = 160

# Below this word-overlap the optimized bullet is so loose a rewrite that
# calling it a rewrite of the source would be a guess - it is NEW content.
_MATCH_THRESHOLD = 0.62


def _norm(text: str) -> str:
    """Whitespace-collapsed lowercase - the comparison space."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    """Whitespace-stripped lowercase - the metric-token matching space
    (the integrity scan's own idiom: spaced variants never read as
    present/absent by accident)."""
    return re.sub(r"\s+", "", (text or "").lower())


def _metric_kept(source_token: str, optimized_text: str,
                 optimized_metrics: list[str]) -> bool:
    """True when a source bullet's metric survives in the optimized bullet:
    a whitespace-insensitive text match, OR the optimized bullet carries a
    token that is a legitimate rounding/truncation of it (the Nov 14
    tolerance: 96.2% -> "96%" and 2.6x -> "3x" are KEPT; 380 ms -> 400 ms
    is a drop)."""
    if _compact(source_token) in _compact(optimized_text):
        return True
    evidence = _evidence_metric_values(source_token)
    return any(_claim_rounds_to(token, evidence) for token in optimized_metrics)


def _achievements(resume_text: str) -> list[tuple[str, str]]:
    """(section title, bullet text) for every achievement bullet - the
    exact extraction contract of the round-13 impact audit (the
    battle-tested markdown parser separates link-only bullets from
    achievement items; education sections are coursework, not claims)."""
    try:
        data = parse_markdown_resume(resume_text or "")
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for section in data.sections:
        title = (section.title or "").strip()
        if not title or _EDU_SECTION_RE.search(title):
            continue
        for entry in section.entries:
            for item in entry.items:
                text = _strip_inline(item or "")
                if not text or _is_link_item(text):
                    continue
                out.append((title, text))
    return out


def compute_bullet_sources(original_resume: str, optimized_resume: str,
                           max_rows: int = _MAX_ROWS) -> list[dict]:
    """Per-bullet provenance: what the optimization did to every bullet.

    Returns rows in this order: bound + new bullets in OPTIMIZED document
    order (they describe the shipped resume), then fully-dropped source
    bullets in ORIGINAL document order (capped together at ``max_rows``).
    Row shape::

        {"bullet": str,        # the optimized bullet ("" for dropped rows)
         "section": str,       # optimized side (original side for dropped)
         "status": "unchanged" | "reworded" | "new" | "dropped",
         "source": str | None, # the original bullet it came from (None=new)
         "similarity": float,  # word-overlap 0..1, 2dp (0.0 when unbound)
         "metrics": [str],     # measured outcomes in the optimized bullet
         "dropped": [str],     # source metrics the optimized bullet lost
         "reason": str}        # human-readable, actionable

    Matching is one-to-one (an original bullet never explains two optimized
    bullets): all pairs at/above the threshold are bound greedily strongest
    first, so the closest explanation always wins regardless of document
    order. Dropped-metric detection applies the integrity scan's rounding
    tolerance, so "unchanged/reworded" rows never cry wolf about a metric
    the resume still states at coarser precision.
    """
    sources = _achievements(original_resume)
    targets = _achievements(optimized_resume)
    if not targets or not sources:
        # No bullets to compare (prose resumes, empty inputs): provenance is
        # empty by definition - the aggregate integrity scan still covers
        # metric drops at the resume level.
        return []

    # All candidate pairs at/above the threshold, strongest first. Ties keep
    # the earlier (i, j) so binding is deterministic run over run.
    pairs: list[tuple[float, int, int]] = []
    for i, (_, src) in enumerate(sources):
        norm_src = _norm(src)
        for j, (_, tgt) in enumerate(targets):
            norm_tgt = _norm(tgt)
            if norm_src == norm_tgt:
                sim = 1.0
            else:
                sim = difflib.SequenceMatcher(None, norm_src,
                                              norm_tgt).ratio()
            if sim >= _MATCH_THRESHOLD:
                pairs.append((sim, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    src_used: set[int] = set()
    tgt_binding: dict[int, tuple[int, float]] = {}
    for sim, i, j in pairs:
        if i in src_used or j in tgt_binding:
            continue
        src_used.add(i)
        tgt_binding[j] = (i, sim)

    rows: list[dict] = []

    # Bound + new bullets, in optimized document order.
    for j, (tgt_section, tgt_text) in enumerate(targets):
        if len(rows) >= max_rows:
            return rows
        binding = tgt_binding.get(j)
        tgt_metrics = _quant_tokens(tgt_text)
        tgt_norm = _norm(tgt_text)
        if binding is None:
            rows.append({
                "bullet": _safe_truncate(tgt_text, _MAX_BULLET_CHARS,
                                         "bullet"),
                "section": tgt_section,
                "status": "new",
                "source": None,
                "similarity": 0.0,
                "metrics": tgt_metrics,
                "dropped": [],
                "reason": ("New content - no bullet in your original "
                           "traces to this text (the integrity scan "
                           "separately flags any unevidenced metric)."),
            })
            continue
        i, sim = binding
        src_section, src_text = sources[i]
        src_metrics = _quant_tokens(src_text)
        dropped = [m for m in src_metrics
                   if not _metric_kept(m, tgt_text, tgt_metrics)]
        if tgt_norm == _norm(src_text):
            status = "unchanged"
            reason = "Kept verbatim from your original."
        else:
            pct = int(round(sim * 100))
            status = "reworded"
            reason = (f"Rewritten from your original ({pct}% text "
                      "overlap) - verify the meaning survived.")
            if dropped:
                reason = (f"Rewritten from your original ({pct}% text "
                          f"overlap) - the measured outcome(s) "
                          f"{', '.join(dropped)} no longer appear in this "
                          "bullet.")
        rows.append({
            "bullet": _safe_truncate(tgt_text, _MAX_BULLET_CHARS, "bullet"),
            "section": tgt_section,
            "status": status,
            "source": _safe_truncate(src_text, _MAX_BULLET_CHARS, "source"),
            "similarity": round(sim, 2),
            "metrics": tgt_metrics,
            "dropped": dropped,
            "reason": reason,
        })

    # Fully-dropped source bullets, in original document order: the
    # per-bullet, typed version of the aggregate "source evidence metric(s)
    # no longer in the resume" integrity finding.
    for i, (src_section, src_text) in enumerate(sources):
        if i in src_used or len(rows) >= max_rows:
            continue
        src_metrics = _quant_tokens(src_text)
        rows.append({
            "bullet": "",
            "section": src_section,
            "status": "dropped",
            "source": _safe_truncate(src_text, _MAX_BULLET_CHARS, "source"),
            "similarity": 0.0,
            "metrics": src_metrics,
            "dropped": list(src_metrics),
            "reason": ("No bullet in the optimized resume traces to this "
                       "source bullet - tailoring may legitimately "
                       "condense, but verify the loss is intentional."),
        })
    return rows
