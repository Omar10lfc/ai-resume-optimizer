"""NODE 6: Interview Prep - generates targeted interview questions."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..ats import _job_brief
from ..helpers import _safe_print, _strip_code_fences
from langchain_core.messages import HumanMessage, SystemMessage


def interview_prep_node(state: AgentState):
    _safe_print(f"\n--- NODE 6: GENERATING INTERVIEW PREP QUESTIONS ---")

    prompt = f"""
    You are a senior hiring manager preparing for an interview.

    {_untrusted('ROLE BRIEF (from job description)', _job_brief(state['job_text']))}

    {_untrusted('CANDIDATE RESUME', state['optimized_resume'])}
    Skills that were identified as gaps: {state.get('missing_skills', 'None identified')}

    Generate exactly 7 likely interview questions. The questions should:

    1. Target the specific GAPS between the resume and job requirements
    2. Test whether the candidate truly has the experience they claim
    3. Include a mix of:
       - Technical questions (about tools/skills listed)
       - Behavioral questions ("Tell me about a time when...")
       - Situational questions ("How would you handle...")
    4. For each question, add a brief hint in parentheses about what the interviewer
       is really looking for

    Format as a numbered list in MARKDOWN. Each question on its own line.
    After the list, add a brief "Preparation Tips" section with 3 bullet points.
    Do NOT wrap your output in code fences (no ``` blocks).
    """

    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    _safe_print(f"   Generated interview prep content")

    return {"interview_questions": _strip_code_fences(response.content)}
