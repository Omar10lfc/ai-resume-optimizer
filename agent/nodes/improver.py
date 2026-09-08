"""NODE 2: Improver - rewrites resume using human notes."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import _safe_print, _strip_code_fences, _filter_hallucinated_sections
from langchain_core.messages import HumanMessage, SystemMessage


def improver_node(state: AgentState):
    current_iter = state.get("iteration", 0) + 1
    _safe_print(f"\n--- NODE 2: IMPROVING RESUME (Iteration {current_iter}) ---")

    base_content = state['optimized_resume'] if state['optimized_resume'] else state['resume_text']

    prompt = f"""
    You are an expert Resume Writer.

    TASK: Rewrite the resume to match the Job Description.
    Output the result in clean MARKDOWN format. Do NOT wrap your output in code fences.

    STRICT MARKDOWN STRUCTURE TO FOLLOW:
    # Full Candidate Name
    Location | Phone Number | Email Address | LinkedIn URL | GitHub URL | Portfolio URL

    ## Professional Summary
    2-3 impactful sentences summarizing qualifications, target domain, and core strengths.

    ## Technical Skills
    **Category 1**: Skill A, Skill B, Skill C
    **Category 2**: Skill D, Skill E, Skill F
    (Always format skills as **Category**: comma-separated list)

    ## Experience
    **Job Title** | Company Name | Location | Date Range
    - Bullet starting with strong action verb and quantified outcome
    - Another bullet with metrics and technical tools

    ## Projects
    **Project Name** [Link](URL) | Technologies Used | Date Range
    - Bullet describing what you built, technical implementation, and measurable impact
    - Keep Project Name concise (2-4 words, e.g. **Alexandria Port Digital Twin**; put descriptions in bullets)
    - Keep dates on the same line (e.g. **Project Name** [GitHub](URL) | Tech Stack | Jan 2024 - Present)
    - NEVER wrap dates in parentheses across line breaks (do NOT write **Title** (\n*Date*))
    - CRITICAL: PRESERVE all links from the original resume (GitHub repos, live demos, project URLs, e.g. [GitHub](https://...), [Live Demo](https://...)). Never drop project links!

    ## Education
    **University Name** | Location | Degree Name | Graduation Date Range

    ## Certifications
    Certification Name - Issuing Organization (Year)

    (Note: Also preserve any additional legitimate sections present in the original resume, such as Publications, Honors & Awards, Leadership, Volunteering, or Coursework.)

    INSTRUCTIONS:
    1. Address these missing skills where the candidate plausibly has relevant experience:
       {state['missing_skills']}
    2. USE THIS USER CONTEXT: "{state['human_notes']}" (Incorporate this experience if valid).
    3. If the user provided no evidence for a specific missing TOOL, OMIT IT entirely.
       For general concepts, phrasing like "Conceptual Knowledge of..." is acceptable.
    4. Do NOT invent projects, jobs, metrics, or credentials not supported by the
       original resume or the user context.
    5. Feedback from previous review (if any): {state.get('feedback', 'None')}
    6. PRESERVE ALL REAL LINKS: Never drop GitHub links, live demos, portfolio URLs, LinkedIn profile links, or email addresses from the original resume.
    7. Retain legitimate existing sections from the candidate's background.

    {_untrusted('JOB DESCRIPTION', state['job_text'])}

    {_untrusted('ORIGINAL RESUME', base_content)}

    Return ONLY the rewritten resume text in MARKDOWN. Do NOT use code fences.
    """

    response = _llms.llm_creative.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    optimized = _strip_code_fences(response.content)
    optimized = _filter_hallucinated_sections(optimized, state['resume_text'])

    return {"optimized_resume": optimized, "iteration": current_iter}
