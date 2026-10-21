"""NODE 2: Improver - rewrites resume using human notes."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import _safe_print, _strip_code_fences, _filter_hallucinated_sections, _weak_bullet_samples
from langchain_core.messages import HumanMessage, SystemMessage


def improver_node(state: AgentState):
    current_iter = state.get("iteration", 0) + 1
    _safe_print(f"\n--- NODE 2: IMPROVING RESUME (Iteration {current_iter}) ---")

    base_content = state['optimized_resume'] if state['optimized_resume'] else state['resume_text']
    weak_bullets = _weak_bullet_samples(base_content)

    prompt = f"""
    You are an expert Resume Writer applying professional tailoring standards.

    TASK: Rewrite the resume to match the Job Description.
    Output the result in clean MARKDOWN format. Do NOT wrap your output in code fences.

    TAILORING PHILOSOPHY (non-negotiable):
    - You are HIGHLIGHTING the most relevant parts of the candidate's TRUE
      experience. Reordering true information, re-emphasizing relevant jobs and
      bullets, and matching the JD's terminology is tailoring.
    - Adding skills, employers, titles, credentials, numbers, or projects that
      the source resume/user context does not support is fabrication - forbidden.

    STRICT MARKDOWN STRUCTURE TO FOLLOW:
    # Full Candidate Name
    Location | Phone Number | Email Address | LinkedIn URL | GitHub URL | Portfolio URL

    ## Professional Summary
    2-3 impactful sentences that MIRROR THE EXACT TARGET ROLE from the JD, name
    the top 2-3 required skills, and include one true metric if the original
    resume has one. Never a generic summary ("results-driven professional ...").

    ## Technical Skills
    **Category 1**: Skill A, Skill B, Skill C
    **Category 2**: Skill D, Skill E, Skill F
    (Always format skills as **Category**: comma-separated list.
    ORDER CATEGORIES AND SKILLS BY RELEVANCE TO THIS JD - most relevant first.
    Use the JD's exact terminology when it truthfully describes the candidate.
    Include only technologies the resume supports seriously: never Microsoft
    Office, never operating systems, never one-time-touched tools, never
    proficiency bars/ratings.)

    ## Experience
    **Job Title** | Company Name | Location | Date Range
    - Bullet starting with strong action verb and quantified outcome
    - Another bullet with metrics and technical tools

    BULLET RULES - wherever the source supports it, follow the X-Y-Z formula
    ("Accomplished [X] as measured by [Y] by doing [Z]"):
    - FIRST WORD is a strong action verb (Led, Built, Automated, Reduced,
      Designed, Migrated, Shipped...).
    - NEVER open a bullet with: "Responsible for", "Helped with", "Assisted
      with/in", "Worked on", "Worked with", "Participated in", "Duties
      included", "Tasked with" - rewrite those into achievement statements.
    - Every bullet shows IMPACT, not duties. Where the source provides any
      basis, include a number: money saved/earned, percentage, time reduced,
      scale (users, requests/sec, team size), or before -> after.
    - NEVER invent a metric that is not in the original resume or user context.
    - Keep bullets to 1-2 lines. Do not keyword-stuff: each critical JD keyword
      used naturally in summary + skills + one bullet is enough, vary phrasing.

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
    8. Keep the original factual content (employers, titles, dates, numbers,
       links) intact while re-emphasizing and rephrasing toward this JD.

    {_untrusted('BULLETS CURRENTLY OPENING WITH PASSIVE/WEAK PHRASES (rewrite these first)',
                chr(10).join(weak_bullets) if weak_bullets else 'None detected')}

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
