"""NODE 1: Scanner - analyzes gaps between resume and job."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import _safe_print, _safe_truncate
from langchain_core.messages import HumanMessage, SystemMessage


def scanner_node(state: AgentState):
    _safe_print(f"\n--- NODE 1: SCANNING FOR GAPS ---")

    prompt = f"""
    Compare the Resume to the Job Description and identify the 3-5 biggest
    MISSING requirements, ranked by importance to this specific role.

    Classification rules:
    - MUST-HAVE: JD language like "must have", "required", "essential", listed
      under "Requirements"/"You have", or mentioned 3+ times.
      NICE-TO-HAVE: "preferred", "nice to have", "bonus", "a plus", or
      mentioned only 1-2 times.
    - Prioritize HARD SKILLS: tools, programming languages, frameworks,
      platforms, methodologies, certifications, and concrete industry/domain
      terms. These are what keyword matching actually rewards.
    - Do NOT report generic soft skills (leadership, communication) or action
      verbs/duties (support, collaborate, design) as missing skills - only if
      the JD names one explicitly AND the resume lacks it.
    - Never report recruiter boilerplate ("fast-paced environment",
      "wear many hats") or company names.

    {_untrusted('JOB DESCRIPTION', _safe_truncate(state['job_text'], 3000, 'Job text (scanner)'))}

    {_untrusted('RESUME', _safe_truncate(state['resume_text'], 3000, 'Resume text (scanner)'))}

    Return ONLY a bulleted list of the missing skills, most important first,
    each tagged [must-have] or [nice-to-have].
    """
    response = _llms.llm_strict.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    _safe_print(f"   Identified Gaps:\n{response.content}")

    return {"missing_skills": response.content, "iteration": 0}
