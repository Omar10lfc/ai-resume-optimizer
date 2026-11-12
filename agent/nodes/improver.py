"""NODE 2: Improver - rewrites resume using human notes.

Also hosts the pieces the POLISHER node shares with it: the full-rewrite
prompt/enforcement chain and the deterministic post-processing tail. The
polisher (agent/nodes/polisher.py) reuses this module's full-rewrite path
as its fallback mode and runs its own section-targeted mode on top.
"""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import (_safe_print, _strip_code_fences, _filter_hallucinated_sections,
                       _tame_unicode_bullets, find_unevidenced_metric_claims,
                       scrub_unsupported_content, extract_declined_requests,
                       accumulate_unique, preserve_contact_links)
from langchain_core.messages import HumanMessage, SystemMessage

# Bump when the rewrite prompt changes materially. Runners stamp this into
# every artifact so "which prompt produced this draft" is a one-lookup
# answer instead of a trace-scavenging exercise.
IMPROVER_PROMPT_VERSION = "improver@2"


def _build_improver_prompt(state: AgentState) -> str:
    """The full-rewrite prompt (iteration 1 and the polisher fallback)."""
    base_content = state['optimized_resume'] if state['optimized_resume'] else state['resume_text']
    feedback = state.get('feedback') or ""

    return f"""
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
    5. Feedback from previous review (if any): {feedback}
    6. PRESERVE ALL REAL LINKS: Never drop GitHub links, live demos, portfolio URLs, LinkedIn profile links, or email addresses from the original resume. Attach project links directly to the project header line beside the project title, NEVER as a standalone bullet item.
    7. Retain legitimate existing sections from the candidate's background.
    8. DECLINE PROTOCOL: when the missing-skills list, the user context, or
       the feedback asks you to feature something the evidence (original
       resume + user notes) does not support - tools, employers, projects,
       certifications, metrics - do NOT write it. Instead end the resume
       with one line per refused request, in this exact shape:
           DECLINED: <short restatement of the requested item>
       Example: "DECLINED: Experience with n8n". These lines are removed
       automatically and shown to the user as "requests the writer
       declined"; they exist so silence can never hide an omission.

    {_untrusted('JOB DESCRIPTION', state['job_text'])}

    {_untrusted('ORIGINAL RESUME', base_content)}

    Return ONLY the rewritten resume text in MARKDOWN. Do NOT use code fences.
    """


def _postprocess_draft(state: AgentState, draft: str, current_iter: int) -> dict:
    """Deterministic enforcement tail shared by the improver and the polisher.

    Chain (in order): decline-protocol parsing -> hallucinated-section
    filter -> fabrication scrubber -> unevidenced-metric tripwire. Accumulates
    the QW2/QW3 channels (declined_requests, scrub_removals,
    unevidenced_metrics) across passes; guardrail_warnings keeps its existing
    last-pass contract (server relays the final list as one warning event).
    """
    # The writer's DECLINED: lines are protocol, not resume content: strip
    # them before any downstream consumer (ATS, LaTeX, PDF) can see them.
    draft, declined = extract_declined_requests(draft)

    draft = _tame_unicode_bullets(draft)
    draft = _filter_hallucinated_sections(draft, state['resume_text'])

    # Deterministic fabrication scrubber (runs 14751ff9 / a32f30fc /
    # 868e406d): the model ignored the reviewer's "remove the unverified
    # claims" feedback twice, so unsupported gap-echo bullets, removed-chip
    # echoes and unevidenced numbers are removed from the artifact here -
    # detection alone demonstrably cannot clean the draft. The polisher node
    # shares this code path, so polish passes are scrubbed identically.
    scrubbed, removals = scrub_unsupported_content(
        draft, state.get('resume_text') or '',
        state.get('human_notes') or '', state.get('missing_skills') or '')
    guardrail: list[str] = []
    final_text = draft
    if scrubbed.strip() and scrubbed != draft:
        _safe_print(f"   [Scrub] Removed {len(removals)} unsupported "
                    f"region(s) deterministically")
        final_text = scrubbed
        guardrail.extend(removals)
    elif removals:
        # Pathological draft: every removal would have emptied it, so the
        # unscrubbed draft is kept and the flagged content is surfaced.
        guardrail.append("integrity scrub kept draft (all removals would "
                         "empty it); review flagged items below")

    # Evidence tripwire: metric-shaped numbers the source resume and user
    # notes do not contain are flagged for the user (server warning event)
    # instead of shipping silently. Remaining findings are post-scrub, i.e.
    # claims the scrubber could not excise cleanly.
    evidence = f"{state.get('resume_text') or ''}\n{state.get('human_notes') or ''}"
    warnings = find_unevidenced_metric_claims(final_text, evidence)
    if warnings:
        _safe_print(f"   [Integrity] Unevidenced metric claims in draft: {', '.join(warnings)}")

    # Removal notes surface alongside metric findings so the user sees what
    # was deterministically removed before the result landed.
    guardrail += [f"unevidenced number left in draft: {w}" for w in warnings]

    # Contact-link preservation (run 124f40cb, Nov 12 2026): a rewrite that
    # drops one of the candidate's own header links loses identity data no
    # reviewer can see. Deterministic restore on every writer pass; the
    # polisher reassembles around the draft's header line, so this tail
    # covers both paths.
    final_text, restored_links = preserve_contact_links(
        final_text, state.get('resume_text') or '')
    guardrail += [f"restored dropped contact link: {t}"
                  for t in restored_links]

    return {
        "optimized_resume": final_text,
        "iteration": current_iter,
        "guardrail_warnings": guardrail,
        "scrub_removals": accumulate_unique(state.get("scrub_removals"), removals),
        "unevidenced_metrics": accumulate_unique(state.get("unevidenced_metrics"), warnings),
        "declined_requests": accumulate_unique(state.get("declined_requests"), declined),
    }


def _full_rewrite(state: AgentState, *, node_label: str, version_key: str,
                  version: str) -> dict:
    """Full-rewrite invocation (improver pass and the polisher fallback)."""
    return {**_postprocess_draft(state, _strip_code_fences(
        _llms.llm_creative.invoke([
            SystemMessage(content=SYSTEM_GUARDRAIL),
            HumanMessage(content=_build_improver_prompt(state)),
        ]).content), state.get("iteration", 0) + 1),
        "prompt_versions": {version_key: version}}


def improver_node(state: AgentState):
    current_iter = state.get("iteration", 0) + 1
    _safe_print(f"\n--- NODE 2: IMPROVING RESUME (Iteration {current_iter}) ---")
    return _full_rewrite(state, node_label="improver",
                         version_key="improver", version=IMPROVER_PROMPT_VERSION)
