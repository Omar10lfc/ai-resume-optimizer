"""NODE 4: ATS Check - deterministic keyword match + semantic verification + composite score."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted, SemanticReview
from .. import llms as _llms
from ..config import ATS_SCORE_WEIGHT, LLM_SCORE_WEIGHT
from ..ats import compute_ats_match, apply_semantic_matches
from ..helpers import _safe_print, _safe_truncate
from ..match import compute_requirement_match_table
from ..bullet_impact import compute_bullet_impact
from ..bullet_sources import compute_bullet_sources
from ..provider import invoke_structured_with_repair
from langchain_core.messages import HumanMessage, SystemMessage

# Bump when the semantic-verification prompt or policy changes materially.
ATS_SEMANTIC_PROMPT_VERSION = "ats-semantic@1"


def _semantic_verify_missing(resume_text: str, missing_keywords: list[str],
                             max_keywords: int = 8) -> set[str]:
    """Asks the fast LLM which 'missing' keywords the resume actually expresses
    in different words. Returns keywords to reclassify as semantic matches."""
    keywords = [k for k in missing_keywords if k][:max_keywords]
    if not keywords:
        return set()

    prompt = f"""
    For each keyword below, decide whether the resume ALREADY expresses that
    concept using different wording. Be strict: only mark genuinely_missing=false
    if the resume clearly demonstrates the concept, even under a different name
    (e.g. "built Airflow pipelines" expresses "ETL orchestration").

    Keywords: {keywords}

    {_untrusted('RESUME', _safe_truncate(resume_text, 6000, 'Resume (semantic check)'))}

    Return one verdict per keyword.
    """
    try:
        # MR5: the same schema-repair ladder the reviewer uses (the
        # "Tool choice is required" flake class hit the reviewer THEN the
        # ATS semantic check). One repair attempt showing the model its own
        # answer + the schema before degrading to "keep keywords as
        # missing" - the safe failure direction.
        review, _ = invoke_structured_with_repair(
            _llms.llm_fast, SemanticReview, [
                SystemMessage(content=SYSTEM_GUARDRAIL),
                HumanMessage(content=prompt),
            ], label="ats-semantic")
        expressed = {v.keyword.strip().lower() for v in review.verdicts
                     if not v.genuinely_missing}
        _safe_print(f"-> Semantic matches: {sorted(expressed) if expressed else 'none'}")
        return expressed
    except Exception as e:
        _safe_print(f"   [Warning] Semantic verification failed: {e}. Keeping keywords as missing.")
        return set()


def ats_check_node(state: AgentState):
    _safe_print(f"\n--- NODE 4: ATS KEYWORD CHECK ---")

    result = compute_ats_match(state['job_text'], state['optimized_resume'])

    missing_kws = [r["keyword"] for r in result["keywords"] if not r["found"]]
    semantic = _semantic_verify_missing(state['optimized_resume'], missing_kws)
    if semantic:
        result = apply_semantic_matches(result, semantic)

    # MR2: deterministic requirement table for THIS draft (reuses this pass's
    # semantic reclassification set; no extra LLM call). The best-of guard
    # snapshots the winner's table, so a regresssed polish pass keeps the
    # table that describes the resume that actually ships.
    requirement_table = compute_requirement_match_table(
        state['job_text'], state['optimized_resume'], semantic)

    # Round 13 (roadmap 12): the Google XYZ bullet-impact audit for THIS
    # draft - deterministic, zero LLM calls (the Fit-table precedent).
    # Follows the SAME best-of guard: a regressed polish pass keeps the
    # audit that describes the resume that actually ships.
    bullet_table = compute_bullet_impact(state['optimized_resume'])

    # Round 16: the per-bullet typed source rows for THIS draft - which
    # original bullet each optimized bullet traces to and which measured
    # outcomes a rewrite dropped (the integrity scan's rounding tolerance
    # applies, so "96.2% -> 96%" is kept). Deterministic, zero LLM calls;
    # rides the SAME best-of guard. For file/URL imports original_resume
    # carries the SOURCE (a path/URL), not text - live trace cab7f3f7
    # (Nov 15): a PDF upload left the provenance empty on a fully
    # bulleted resume. The loader's resume_text IS the original content
    # then, so bind against whichever side has bullets.
    original_text = state.get('original_resume') or ''
    source_table = compute_bullet_sources(original_text,
                                          state['optimized_resume'])
    if not source_table:
        loaded = (state.get('resume_text') or '').strip()
        if loaded and loaded != original_text.strip():
            source_table = compute_bullet_sources(
                loaded, state['optimized_resume'])

    llm_quality = state.get('llm_quality_score', 0)
    composite = round(
        ATS_SCORE_WEIGHT * result['percentage'] + LLM_SCORE_WEIGHT * llm_quality
    )

    _safe_print(f"-> ATS Match: {result['percentage']}% ({result['match_count']}/{result['total']})")
    _safe_print(f"-> Composite score: {composite}/100 (ATS {result['percentage']:.0f}% x 0.6 + LLM quality {llm_quality} x 0.4)")

    # Best-of guard: a polish iteration (MAX_ITERATIONS > 1) must never leave
    # the user with a worse resume than an earlier draft. Composite ties keep
    # the newest draft — it carries one more feedback-driven revision.
    best = dict(state.get("best_result") or {})
    best_score = int(best.get("score", -1))
    if composite >= best_score:
        best = {
            "score": composite,
            "optimized_resume": state["optimized_resume"],
            "ats_result": result["formatted"],
            "ats_percentage": result["percentage"],
            "llm_quality_score": llm_quality,
            # MR2: the table travels with the winning draft, so later,
            # worse passes keep a table that matches the shipped resume.
            "requirement_matches": requirement_table,
            # Round 13: the bullet audit travels with it for the same reason.
            "bullet_impact": bullet_table,
            # Round 16: the per-bullet provenance travels with it too.
            "bullet_sources": source_table,
        }
    if composite < best_score:
        _safe_print(f"-> Draft {state.get('iteration', 0)} regressed ({composite} < {best_score}); keeping the best draft.")
        return {
            "ats_result": best["ats_result"],
            "ats_percentage": best["ats_percentage"],
            "score": best["score"],
            "optimized_resume": best["optimized_resume"],
            "llm_quality_score": best["llm_quality_score"],
            "best_result": best,
            "requirement_matches": list(best.get("requirement_matches") or []),
            "bullet_impact": list(best.get("bullet_impact") or []),
            "bullet_sources": list(best.get("bullet_sources") or []),
            "prompt_versions": {"ats-semantic": ATS_SEMANTIC_PROMPT_VERSION},
        }

    return {
        "ats_result": result["formatted"],
        "ats_percentage": result["percentage"],
        "score": composite,
        "best_result": best,
        "requirement_matches": requirement_table,
        "bullet_impact": bullet_table,
        "bullet_sources": source_table,
        "prompt_versions": {"ats-semantic": ATS_SEMANTIC_PROMPT_VERSION},
    }
