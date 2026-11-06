"""NODE 1: Scanner - analyzes gaps between resume and job."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import _safe_print, _safe_truncate, _tame_unicode_bullets
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
    # Tame web-sourced Unicode before the text reaches any consumer: gap
    # chips, ATS text spans, and notes all assume plain ASCII separators.
    # Live-run evidence (Nov 6 2026): "Claude\u202fCode" / "Hands\u2011on".
    missing = _tame_unicode_bullets(str(response.content))
    _safe_print(f"   Identified Gaps:\n{missing}")
    return {"missing_skills": missing, "iteration": 0}
