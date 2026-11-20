"""NODE 5: Cover Letter - generates tailored cover letter."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..ats import _job_brief
from ..helpers import (_safe_print, _strip_code_fences,
                       find_unevidenced_metric_claims)
from ..tool_claims import build_denied_tools, remove_sentences_naming_tools
from langchain_core.messages import HumanMessage, SystemMessage

# Bump when the cover-letter prompt changes materially.
# @2 (Nov 11 2026, live run 22e943c6): the letter fabricated "70%"/"45%"
# metrics absent from every input and claimed production use of tools it
# only knew from the JD chips. Adds the truthfulness gate; the generated
# letter is additionally scanned by the resume path's unevidenced-metric
# tripwire below (findings -> cover_letter_metrics, user-visible).
COVER_LETTER_PROMPT_VERSION = "cover-letter@2"


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
    5. TRUTHFULNESS GATE (live run 22e943c6: the letter coined "reduced
       manual processing time by 70%" and "increased user engagement by 45%"
       - numbers that appear in NO input - and claimed production use of
       tools the JD merely lists as requirements): the candidate's evidence
       is the CANDIDATE RESUME above only. Never state a number, percentage,
       multiplier, or duration the resume does not contain; JD requirements
       are NOT experience; if a point is stronger quantified but the evidence
       does not show a number, make the point qualitatively instead of
       inventing one.

    Return ONLY the cover letter text as plain prose, starting with the
    greeting line ("Dear Hiring Manager," unless a real contact name is known).
    """
    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    letter = _strip_code_fences(response.content)

    # Integrity tripwires (same rules as the resume path): metric-shaped
    # tokens and, since run 4afa44cb, unevidenced TOOL names. Prompt guards
    # alone demonstrably fold - the resume path learned the same lesson
    # (request 14751ff9) - so both scans run on the generated letter.
    # The shipped resume draft is deliberately NOT evidence here: a claim
    # the user is asked to verify on the resume must not silently become
    # legitimate because the letter repeated it.
    evidence = "\n".join(str(state.get(k) or "") for k in
                         ("resume_text", "human_notes", "profile",
                          "profile_text", "candidate_profile"))
    denied = build_denied_tools(
        evidence,
        declined_requests=state.get("declined_requests") or [],
        job_text=state.get("job_text") or "",
    )
    # Every mention of a denied tool is a violation - including
    # requirement-framed name-drops ("core requirements for Cursor demos")
    # and analogies ("n8n-style automation"): only the sentence goes, never
    # a rewrite that could reintroduce the term. Same sentence split and
    # residue tidy as scrub_unsupported_content uses for the resume.
    letter, removals = remove_sentences_naming_tools(
        letter, denied, stage="cover letter")
    removed_tools = sorted({t for r in removals for t in r.terms})
    if removals:
        _safe_print(f"   [Integrity] Removed {len(removals)} sentence(s) "
                    "naming unevidenced tool(s) from the cover letter")
    letter_metrics = find_unevidenced_metric_claims(letter, evidence)
    if letter_metrics:
        _safe_print(f"   [Integrity] Unevidenced metric claims in cover "
                    f"letter: {', '.join(letter_metrics)}")

    warnings = list(state.get("guardrail_warnings") or [])
    warnings.extend(r.warning for r in removals)
    return {"cover_letter": letter,
            "cover_letter_metrics": letter_metrics,
            "cover_letter_unevidenced_tools": removed_tools,
            "guardrail_warnings": warnings,
            "prompt_versions": {"cover-letter": COVER_LETTER_PROMPT_VERSION}}
