"""NODE 6: Interview Prep - generates targeted interview questions."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..ats import _job_brief
from ..helpers import _safe_print, _strip_code_fences
from langchain_core.messages import HumanMessage, SystemMessage


def interview_prep_node(state: AgentState):
    _safe_print(f"\n--- NODE 6: GENERATING INTERVIEW PREP QUESTIONS ---")

    prompt = f"""
    You are a senior hiring manager preparing interview prep for this candidate.

    {_untrusted('ROLE BRIEF (from job description)', _job_brief(state['job_text']))}

    {_untrusted('CANDIDATE RESUME', state['optimized_resume'])}
    Skills that were identified as gaps: {state.get('missing_skills', 'None identified')}

    SECTION 1 - ## Top 7 Likely Interview Questions (numbered list, exactly 7).
    The questions should:
    1. Target the specific GAPS between the resume and job requirements
    2. Test whether the candidate truly has the experience they claim
    3. Include a mix of:
       - Technical questions (about tools/skills listed)
       - Behavioral questions ("Tell me about a time when...")
       - Situational questions ("How would you handle...")
    4. For each question, add a brief hint in parentheses about what the interviewer
       is really looking for

    SECTION 2 - ## Preparation Tips with exactly 3 bullet points.

    SECTION 3 - ## STAR Story Bank: convert the 3 strongest resume bullets into
    3 STAR stories, each formatted as:
    **Story: <short name>** - Use for: <competencies it proves>
    - S: [one-sentence situation flowing from that bullet's real context]
    - T: [the candidate's specific responsibility]
    - A: [2-3 concrete actions the resume actually shows]
    - R: [the real quantified result from the resume - NEVER invent metrics]
    Keep each story to ~90 seconds of speaking time. Every fact must come from
    the resume or the user's own notes - never fabricate.

    SECTION 4 - ## "Tell Me About Yourself" Pitch: a 60-90 second first-person
    pitch tailored to THIS role: current professional identity + the 2
    highest-relevance achievements (with their real metrics) + why this role
    is the right next step. Reuse only resume facts.

    SECTION 5 - ## Questions to Ask the Interviewer: 3 thoughtful questions
    (e.g. success expectations at 30/60/90 days, the team's biggest current
    challenge, how the role's work is measured). NEVER salary/benefits questions
    or anything trivially searchable.

    Format all sections in MARKDOWN. Do NOT wrap your output in code fences
    (no ``` blocks).
    """

    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    _safe_print(f"   Generated interview prep content")

    return {"interview_questions": _strip_code_fences(response.content)}
