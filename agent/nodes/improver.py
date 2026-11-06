"""NODE 2: Improver - rewrites resume using human notes."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import (_safe_print, _strip_code_fences, _filter_hallucinated_sections,
                       _tame_unicode_bullets, find_unevidenced_metric_claims)
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
    **Project Name** [Live Demo](URL) [GitHub](URL) | Technologies Used | Date Range
    - Bullet describing what you built, technical implementation, and measurable impact
    - Keep Project Name concise (2-4 words, e.g. **Distributed Event Streamer**; put descriptions in bullets)
    - Keep dates and all links on the same header line (e.g. **Real-Time Data Pipeline** [Live Demo](URL) [GitHub](URL) | Tech Stack | Jan 2026 - May 2026)
    - NEVER put links as standalone bullet points! (Do NOT write "- Live Demo" or "- [GitHub](URL)" as a bullet item). All project links belong on the project title line.
    - CRITICAL: PRESERVE all links from the original resume (GitHub repos, live demos, project URLs, e.g. [GitHub](https://...), [Live Demo](https://...)). Never drop project links!

    ## Education
    **University Name** | Location | Degree Name | Graduation Date Range

    ## Certifications
    Certification Name - Issuing Organization (Year)

    (Note: Also preserve any additional legitimate sections present in the original resume, such as Publications, Honors & Awards, Leadership, Volunteering, or Coursework.)

    INSTRUCTIONS:
    1. RULES FOR THE MISSING-SKILLS LIST (treat it as RESEARCH TOPICS, never
       as experience to claim). A gap item may only appear in the rewrite if
       the original resume or the USER CONTEXT already demonstrates it; in
       that case weave the demonstrated evidence in. Otherwise OMIT the item
       entirely. NEVER describe building, deploying, or integrating anything
       with a named tool, platform, vendor server, or product that neither
       the resume nor the user context shows (that includes phrasing gaps as
       existing achievements). For genuinely conceptual items only, phrasing
       like "Conceptual Knowledge of..." is acceptable.
       Missing skills under review: {state['missing_skills']}
    2. USE THIS USER CONTEXT: "{state['human_notes']}" (Incorporate this experience if valid).
    3. Do NOT invent projects, jobs, tools, integrations, credentials, or
       numbers not supported by the original resume or the user context.
    4. METRIC INTEGRITY (zero tolerance): every number in your output must
       be copied from the ORIGINAL RESUME or the USER CONTEXT. Never coin,
       copy, or round numbers from the job description or the feedback text.
    5. Feedback from previous review (if any): {state.get('feedback', 'None')}
    6. PRESERVE ALL REAL LINKS: Never drop GitHub links, live demos, portfolio URLs, LinkedIn profile links, or email addresses from the original resume. Attach project links directly to the project header line beside the project title, NEVER as a standalone bullet item.
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

    # Evidence tripwire: metric-shaped numbers the source resume and user
    # notes do not contain are flagged for the user (server warning event)
    # instead of shipping silently. Detection only - the draft stays intact.
    evidence = f"{state.get('resume_text') or ''}\n{state.get('human_notes') or ''}"
    warnings = find_unevidenced_metric_claims(optimized, evidence)
    if warnings:
        _safe_print(f"   [Integrity] Unevidenced metric claims in draft: {', '.join(warnings)}")

    return {"optimized_resume": optimized, "iteration": current_iter,
            "guardrail_warnings": warnings}
