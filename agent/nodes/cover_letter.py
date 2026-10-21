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
    - Write in standard PROSE paragraphs only (Dear ..., body paragraphs, Sincerely)
    - Do NOT use tables, bullet lists, or any tabular formatting
    - Do NOT use markdown tables (no | pipes)
    - Do NOT wrap output in code fences (no ``` blocks)
    - 250-400 words in total, 3-4 paragraphs maximum
    - Warm, confident, professional tone - never arrogant, never apologetic

    QUALITY RULES (professional cover-letter standards):
    1. OPENING HOOK - never a formality. Open with something specific: a real
       achievement from the resume, a problem in the posting the candidate can
       solve, or genuine enthusiasm for something concrete in this role.
       NEVER open with "I am writing to apply for", "I saw your job posting",
       or "To Whom It May Concern", and avoid opening sentences that start
       with "I".
    2. BODY - DIRECT MATCH: connect the candidate's strongest TRUE experience
       to the role's top requirement: [their need] + [candidate's specific
       experience] + [quantified result]. Reference real projects, tools, and
       metrics from the resume - every claim must exist in the resume.
    3. BODY - VALUE + GAPS: add one more relevant strength. If the resume has
       an obvious gap for this role, address it briefly and positively with a
       related strength or transferable skill - never apologize, never fabricate.
    4. CLOSING - CALL TO ACTION: end with genuine enthusiasm for something
       specific and a confident invitation to discuss next steps. NEVER close
       with "I look forward to hearing from you" or "Please find my resume
       attached".

    Return ONLY the cover letter text as plain prose, starting with the
    greeting line ("Dear Hiring Manager," unless a real contact name is known).
    """
    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    return {"cover_letter": _strip_code_fences(response.content)}
