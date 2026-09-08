"""NODE 5: Cover Letter - generates tailored cover letter."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..ats import _job_brief
from ..helpers import _safe_print, _strip_code_fences
from langchain_core.messages import HumanMessage, SystemMessage


def cover_letter_node(state: AgentState):
    _safe_print(f"\n--- NODE 5: GENERATING COVER LETTER ---")
    prompt = f"""
    Write a professional Cover Letter based on the Candidate's Resume for this role.

    {_untrusted('ROLE BRIEF (from job description)', _job_brief(state['job_text']))}

    {_untrusted('CANDIDATE RESUME', state['optimized_resume'])}

    STRICT FORMAT RULES:
    - Write in standard PROSE paragraphs only (Dear..., body paragraphs, Sincerely)
    - Do NOT use tables, bullet lists, or any tabular formatting
    - Do NOT use markdown tables (no | pipes)
    - Do NOT wrap output in code fences (no ``` blocks)
    - Keep it to 3-4 paragraphs maximum
    - Use a warm, confident, professional tone

    STRUCTURE:
    1. Opening: State the position and express genuine interest
    2. Body (1-2 paragraphs): Connect your strongest relevant experiences to the key requirements. Be specific - mention projects, metrics, and tools
    3. Closing: Express enthusiasm and call to action

    Return ONLY the cover letter text as plain prose.
    """
    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    return {"cover_letter": _strip_code_fences(response.content)}
