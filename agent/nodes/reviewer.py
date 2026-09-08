"""NODE 3: Reviewer - LLM quality judgment."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted, ReviewOutput
from .. import llms as _llms
from ..helpers import _safe_print, _safe_truncate
from langchain_core.messages import HumanMessage, SystemMessage


def reviewer_node(state: AgentState):
    _safe_print(f"\n--- NODE 3: REVIEWING DRAFT ---")

    structured_llm = _llms.llm_strict.with_structured_output(ReviewOutput)

    prompt = f"""
    You are a senior hiring manager judging the QUALITY of a resume's presentation.
    Do NOT judge keyword coverage - that is measured separately.

    Rate each dimension 0-100 and combine them:
    - Relevance & Tailoring (40%): Is the experience presented to directly address the job requirements?
    - Quantified Achievements (30%): Does the resume use metrics, numbers, or concrete outcomes?
    - Formatting & Clarity (30%): Well-structured, scannable, free of filler?

    {_untrusted('JOB DESCRIPTION', _safe_truncate(state['job_text'], 2500, 'Job text (reviewer)'))}

    {_untrusted('RESUME', state['optimized_resume'])}

    Provide the combined weighted quality score and one specific sentence of feedback to improve it.
    """

    try:
        result = structured_llm.invoke([
            SystemMessage(content=SYSTEM_GUARDRAIL),
            HumanMessage(content=prompt),
        ])
        _safe_print(f"-> LLM quality score: {result.score}/100")
        _safe_print(f"-> Feedback: {result.feedback}")
        return {"llm_quality_score": result.score, "feedback": result.feedback,
                "review_failed": False}
    except Exception as e:
        _safe_print(f"   [Warning] Structured output failed: {e}. Flagging review as failed.")
        return {"review_failed": True,
                "feedback": "Review could not be completed. Proceeding with current draft."}
