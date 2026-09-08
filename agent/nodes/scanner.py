"""NODE 1: Scanner - analyzes gaps between resume and job."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import _safe_print, _safe_truncate
from langchain_core.messages import HumanMessage, SystemMessage


def scanner_node(state: AgentState):
    _safe_print(f"\n--- NODE 1: SCANNING FOR GAPS ---")

    prompt = f"""
    Compare the Resume to the Job Description.
    Identify the 3 biggest MISSING SKILLS or Keywords.

    {_untrusted('JOB DESCRIPTION', _safe_truncate(state['job_text'], 3000, 'Job text (scanner)'))}

    {_untrusted('RESUME', _safe_truncate(state['resume_text'], 3000, 'Resume text (scanner)'))}

    Return ONLY a bulleted list of the missing skills.
    """
    response = _llms.llm_strict.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    _safe_print(f"   Identified Gaps:\n{response.content}")

    return {"missing_skills": response.content, "iteration": 0}
