"""NODE 4: ATS Check - deterministic keyword match + semantic verification + composite score."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted, SemanticReview
from .. import llms as _llms
from ..config import ATS_SCORE_WEIGHT, LLM_SCORE_WEIGHT
from ..ats import compute_ats_match, apply_semantic_matches
from ..helpers import _safe_print, _safe_truncate
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
        review = _llms.llm_fast.with_structured_output(SemanticReview).invoke([
            SystemMessage(content=SYSTEM_GUARDRAIL),
            HumanMessage(content=prompt),
        ])
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
            "prompt_versions": {"ats-semantic": ATS_SEMANTIC_PROMPT_VERSION},
        }

    return {
        "ats_result": result["formatted"],
        "ats_percentage": result["percentage"],
        "score": composite,
        "best_result": best,
        "prompt_versions": {"ats-semantic": ATS_SEMANTIC_PROMPT_VERSION},
    }
