"""NODE 5: Cover Letter - generates tailored cover letter."""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..ats import _job_brief
from ..helpers import (_safe_print, _safe_truncate, _strip_code_fences,
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


def _repair_cover_paragraphs(
    original_letter: str,
    cleaned_letter: str,
    denied: set[str],
    evidence: str,
) -> str:
    """Smooths body paragraphs shrunk or left single-sentence by sentence removal,
    ensuring repaired prose never reintroduces denied tools or whole-letter echoes."""
    orig_paras = [p.strip() for p in (original_letter or "").split("\n\n") if p.strip()]
    clean_paras = [p.strip() for p in (cleaned_letter or "").split("\n\n") if p.strip()]
    if not clean_paras:
        return cleaned_letter
    denied_str = ", ".join(sorted(denied)[:30]) if denied else "(none)"
    repaired_out: list[str] = []
    for idx, para in enumerate(clean_paras):
        low = para.lower()
        # Leave greeting and sign-off untouched
        if low.startswith(("dear ", "sincerely", "best regards", "regards,")):
            repaired_out.append(para)
            continue
        orig_len = len(orig_paras[idx]) if idx < len(orig_paras) else len(para)
        was_trimmed = len(para) + 25 < orig_len or idx >= len(orig_paras)
        # Only repair body paragraphs significantly truncated into a short choppy fragment
        if was_trimmed and len(para) < 120:
            repair_prompt = f"""
            Rewrite ONLY this single cover-letter body paragraph so it flows naturally
            and persuasively without choppy gaps, using ONLY the factual content already
            in the paragraph below and the factual evidence.

            FORBIDDEN / UNEVIDENCED TOOLS (NEVER mention or imply these): {denied_str}

            CURRENT PARAGRAPH:
            {para}

            CANDIDATE EVIDENCE (facts/numbers allowed):
            {_safe_truncate(evidence, 3000, 'Cover repair evidence')}

            Return ONLY 2-3 cohesive prose sentences for this single paragraph (no greeting,
            no closing sign-off, no bullet points, no markdown).
            """
            try:
                ans = _llms.llm_fast.invoke([
                    SystemMessage(content=SYSTEM_GUARDRAIL),
                    HumanMessage(content=repair_prompt),
                ])
                cand = _strip_code_fences(str(ans.content)).strip()
                cand_low = cand.lower()
                if (
                    cand
                    and "\n\n" not in cand
                    and not cand_low.startswith("dear ")
                    and "sincerely," not in cand_low
                ):
                    cand_clean, cand_rem = remove_sentences_naming_tools(
                        cand, denied, stage="cover letter repair")
                    if cand_clean.strip() and not cand_rem and not find_unevidenced_metric_claims(cand_clean, evidence):
                        repaired_out.append(cand_clean.strip())
                        continue
            except Exception:
                pass
        repaired_out.append(para)
    return "\n\n".join(repaired_out)


def cover_letter_node(state: AgentState):
    _safe_print(f"\n--- NODE 5: GENERATING COVER LETTER ---")
    evidence = "\n".join(str(state.get(k) or "") for k in
                         ("resume_text", "human_notes", "profile",
                          "profile_text", "candidate_profile"))
    denied = build_denied_tools(
        evidence,
        declined_requests=state.get("declined_requests") or [],
        job_text=state.get("job_text") or "",
    )
    denied_block = (
        f"\n    FORBIDDEN / UNEVIDENCED TOOLS (do NOT name, claim, or allude to any "
        f"of these tools anywhere in the cover letter): {', '.join(sorted(denied)[:30])}\n"
        if denied else ""
    )

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
       experience] + [quantified result]. Reference real projects, tools,
       employers, and metrics from the resume - every claim must exist in
       the resume, and every role/metric must be attributed to the exact
       employer or project shown (never turn a track/course label like
       "(IBM Data Science Track)" at DEPI into employment at IBM).
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
    greeting line ("Dear Hiring Manager," unless a real contact name is known).{denied_block}
    """
    response = _llms.llm_fast.invoke([
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=prompt),
    ])
    raw_letter = _strip_code_fences(response.content)
    letter, removals = remove_sentences_naming_tools(
        raw_letter, denied, stage="cover letter")
    removed_tools = sorted({t for r in removals for t in r.terms})
    prevention_failures = []
    if removals:
        prevention_failures.append({"stage": "cover_letter", "terms": removed_tools})
        _safe_print(f"   [Integrity] prompt_prevention_failure (cover_letter): "
                    f"removed {len(removals)} sentence(s)")
        letter = _repair_cover_paragraphs(raw_letter, letter, denied, evidence)
    letter_metrics = find_unevidenced_metric_claims(letter, evidence)
    if letter_metrics:
        _safe_print(f"   [Integrity] Unevidenced metric claims in cover "
                    f"letter: {len(letter_metrics)}")  # Item 10

    warnings = list(state.get("guardrail_warnings") or [])
    warnings.extend(r.warning for r in removals)
    out = {"cover_letter": letter,
           "cover_letter_metrics": letter_metrics,
           "cover_letter_unevidenced_tools": removed_tools,
           "guardrail_warnings": warnings,
           "prompt_versions": {"cover-letter": COVER_LETTER_PROMPT_VERSION}}
    if prevention_failures:
        out["prompt_prevention_failures"] = prevention_failures
    return out
