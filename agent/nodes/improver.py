"""NODE 2: Improver - rewrites resume using human notes.

Also hosts the pieces the POLISHER node shares with it: the structured
full-rewrite path (BB1 Phase B: typed section units keyed by source ids,
code-assembled delivery - agent/resume_units.py) and the deterministic
post-processing tail. The polisher (agent/nodes/polisher.py) reuses this
module's full-rewrite path as its fallback mode and runs its own
section-targeted mode on top.
"""
from typing import Optional

from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import (_safe_print, _filter_hallucinated_sections,
                       _tame_unicode_bullets, find_unevidenced_metric_claims,
                       find_dropped_source_metrics, find_unused_context_terms,
                       scrub_unsupported_content, extract_declined_requests,
                       accumulate_unique, preserve_contact_links)
from ..provider import invoke_structured_with_repair, extract_failed_generation
from ..resume_units import (ImproverOutput, RESUME_UNIT_ID,
                            assemble_resume_units,
                            improver_output_from_markdown,
                            parse_resume_units, STANDARD_SECTION_NAMES)
from langchain_core.messages import HumanMessage, SystemMessage

# Bump when the rewrite prompt changes materially. Runners stamp this into
# every artifact so "which prompt produced this draft" is a one-lookup
# answer instead of a trace-scavenging exercise. improver@4 (BB1 Phase B,
# Nov 13 2026): the writer no longer delivers a whole markdown document -
# it returns typed section replacements keyed by source-unit ids (the
# Dossier study's resumeUnits() "trusting nothing" pattern) and the delivery
# code assembles the resume: the identity header is pinned from the source,
# skipped units ship verbatim, unknown ids are dropped with a visible
# integrity note, and a total writer failure falls back to the source
# verbatim instead of killing the run.
IMPROVER_PROMPT_VERSION = "improver@4"

# Round 11 (Nov 13 2026): the deterministic context weave-in. When the
# post-processing tail finds notes-sourced terms the final draft dropped,
# ONE extra structured pass offers to re-weave them; adoption is decided in
# code (the woven draft re-enters the same patch stack and is kept only if
# the unused-term count strictly decreases). Stamped on attempt, whether or
# not the weave is adopted.
CONTEXT_WEAVE_PROMPT_VERSION = "weave@1"


def _build_improver_prompt(state: AgentState, units) -> str:
    """The structured full-rewrite prompt (iteration 1 and the polisher
    fallback). The source is presented as typed units; the writer returns
    replacements by id and never assembles the document itself."""
    feedback = state.get('feedback') or ""

    if units.whole_document:
        units_block = (
            f'[id="{RESUME_UNIT_ID}"] (the ENTIRE source resume, shown in full):\n'
            f"{units.sections[0].body}")
        delivery = (
            "HOW DELIVERY WORKS - whole-document mode: the entire source is one"
            "\nunit. Return a replacement for it whose body is your complete"
            "\nrewritten resume, starting with \"# <Full Name>\" then the contact"
            "\nline (bare URLs - no [label](url) markdown in the header line),"
            "\nthen the sections in the standard template order. Returning an"
            "\nEMPTY sections list keeps the source verbatim. Do not return any"
            "\nother section id in this mode.")
    else:
        listed = "\n\n".join(
            f'[id="{u.id}"] (source heading: "{u.heading_line}")\n{u.body}'
            for u in units.sections)
        units_block = (
            "IDENTITY HEADER (pinned verbatim by the delivery code - it is NEVER"
            "\nyours to write, and no section body may contain it):\n"
            f"{units.header_block}\n\n"
            "SOURCE RESUME UNITS (return replacements by id; any unit you do not"
            "\nreturn ships VERBATIM from the source):\n"
            f"{listed}")
        delivery = (
            "HOW DELIVERY WORKS - by-section mode: every returned section"
            "\nreplaces ONLY its unit's body (the source heading is kept by the"
            "\ndelivery code); units you skip ship byte-identical from the"
            "\nsource. Returned ids the source does not offer are DROPPED with"
            "\na visible integrity note - only use the ids listed above. You may"
            "\nalso ADD one of the standard sections the source lacks: "
            + ", ".join(STANDARD_SECTION_NAMES)
            + " - use exactly that name as the id.")

    return f"""
    You are an expert Resume Writer.

    TASK: Rewrite the resume to match the Job Description. Return a
    STRUCTURED answer through the ImproverOutput tool - the delivery code
    assembles the final resume document deterministically from what you
    return, trusting nothing.

    {delivery}

    BODY FORMAT (the delivery code wraps every returned body in its '## '
    heading - the body itself must NOT repeat the heading line):
    - Professional Summary: 2-3 impactful sentences summarizing qualifications, target domain, and core strengths.
    - Technical Skills: **Category 1**: Skill A, Skill B, Skill C (always this format).
    - Experience: **Job Title** | Company Name | Location | Date Range on the first line, then bullets starting with strong action verbs and quantified outcomes.
    - Projects: **Project Name** [Live Demo](URL) [GitHub](URL) | Technologies Used | Date Range on the title line (keep names concise, 2-4 words; descriptions go in bullets). NEVER put links as standalone bullet points (do NOT write "- [GitHub](URL)" as a bullet item).
    - Education: **University Name** | Location | Degree Name | Graduation Date Range.
    - Certifications: Certification Name - Issuing Organization (Year).

    INSTRUCTIONS:
    1. RULES FOR THE MISSING-SKILLS LIST (treat it as RESEARCH TOPICS, never
       as experience to claim). A gap item may only appear in a section body
       if the original resume or the USER CONTEXT already demonstrates it; in
       that case weave the demonstrated evidence in. Otherwise OMIT the item
       entirely. NEVER describe building, deploying, or integrating anything
       with a named tool, platform, vendor server, or product that neither
       the resume nor the user context shows (that includes phrasing gaps as
       existing achievements). For genuinely conceptual items only, phrasing
       like "Conceptual Knowledge of..." is acceptable.
       Missing skills under review: {state['missing_skills']}
    2. USER CONTEXT (binding, not optional): "{state['human_notes']}".
       Every project, role, or experience it describes MUST be woven into
       the section bodies you return (Projects / Experience / Summary as
       fits) - silently omitting one is a protocol violation. Only when you
       judge an item unsupported or irrelevant to THIS job, refuse it with
       an entry in `declined` instead (see the Decline Protocol).
    3. Do NOT invent projects, jobs, tools, integrations, credentials, or
       numbers not supported by the original resume or the user context.
    4. METRIC INTEGRITY (zero tolerance): every number in your output must
       be copied from the ORIGINAL RESUME or the USER CONTEXT. Never coin,
       copy, or round numbers from the job description or the feedback text.
    5. Feedback from previous review (if any): {feedback}
    6. PRESERVE ALL REAL LINKS inside the bodies you return: never drop
       GitHub links, live demos, portfolio URLs, LinkedIn profile links, or
       email addresses the source shows. Attach project links to the project
       title line, NEVER as a standalone bullet item.
    7. Retain legitimate existing sections from the candidate's background:
       any unit you skip ships verbatim, so a section stays in the resume
       unless you explicitly replace its body.
    8. DECLINE PROTOCOL: when the missing-skills list, the user context, or
       the feedback asks you to feature something the evidence (original
       resume + user notes) does not support - tools, employers, projects,
       certifications, metrics - do NOT write it. Instead list every refused
       request in `declined`, one short restatement per item:
           "Experience with n8n"
       The same entry is required for any USER CONTEXT item you chose not
       to feature (never omit context silently). These are shown to the
       user as "requests the writer declined"; they exist so silence can
       never hide an omission. (If you answer in prose instead of the tool,
       "DECLINED: <item>" lines are parsed the same way.)

    {_untrusted('SOURCE RESUME UNITS', units_block)}

    {_untrusted('JOB DESCRIPTION', state['job_text'])}

    ANSWER SHAPE: return ONLY the ImproverOutput tool call:
        sections: [{{"id": <unit id or standard name>, "body": <markdown body, no heading line>}}, ...]
        declined: ["<refused request>", ...]
    Do NOT wrap the answer in code fences.
    """


def _make_improver_salvage(units):
    """The ladder's salvage rung for the improver: a writer that answered in
    prose/markdown instead of the tool call gets its answer parsed into the
    structured shape (unknown ids are dropped by the assembly, declined
    lines ride the protocol)."""
    def _salvage(exc):
        text = extract_failed_generation(exc)
        if not text:
            return None
        return improver_output_from_markdown(
            text, whole_document=units.whole_document)
    return _salvage


def _patch_stack(state: AgentState, draft: str) -> tuple[
        str, list[str], list[str], list[str], list[str], list[str]]:
    """The deterministic patch stack shared by the improver and the polisher
    (round 11 extraction so the context weave can re-run it on a woven
    draft). Chain (in order): decline-protocol parsing -> unicode bullet
    taming -> hallucinated-section filter -> fabrication scrubber ->
    unevidenced-metric tripwire -> contact-link guard -> dropped-evidence
    tripwire -> unused-context tripwire. Prints and guardrail order are
    preserved verbatim. Returns ``(final_text, guardrail, scrub_removals,
    unevidenced, declined, unused_context)``."""
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
    # covers both paths. Placeholder sanitation (run bf0f7941) rides the
    # same pass: "N/A" / "TBD" tokens in the contact line are fake fields,
    # and a restored link must never land after one.
    final_text, restored_links, removed_placeholders, normalized_links = \
        preserve_contact_links(final_text, state.get('resume_text') or '')
    guardrail += [f"restored dropped contact link: {t}"
                  for t in restored_links]
    guardrail += [f"removed placeholder token from the contact line: {t}"
                  for t in removed_placeholders]
    guardrail += [f"normalized contact-line link label to the URL: {t}"
                  for t in normalized_links]

    # Dropped-evidence tripwire (run bf0f7941, Nov 12 2026): a rewrite that
    # loses a source accomplishment's quantified evidence (the Smart
    # Lecture WER claim) weakened the resume silently - removals were []
    # and nothing recorded it. Tailoring may condense, so this surfaces
    # the loss for the user instead of deciding for them.
    dropped_evidence = find_dropped_source_metrics(
        final_text, state.get('resume_text') or '')
    if dropped_evidence:
        _safe_print(f"   [Integrity] Source evidence metric(s) no longer "
                    f"in the resume: {len(dropped_evidence)} finding(s)")
    guardrail += dropped_evidence

    # Context-incorporation tripwire (round 7, runs 25ff4ae5 / 1a8125f0,
    # Nov 12 2026): both runs lost the user's pasted Aegis project and no
    # channel recorded it. Detection-only, like the dropped-evidence
    # tripwire above: notes-sourced terms absent from the final resume
    # surface in the integrity warning so the user can verify (or rephrase)
    # instead of discovering the loss in the PDF.
    unused_context = find_unused_context_terms(
        state.get('human_notes') or '', final_text,
        state.get('resume_text') or '')
    if unused_context:
        _safe_print(f"   [Integrity] User-context term(s) not in the "
                    f"resume: {', '.join(unused_context)}")
    guardrail += [f"user-context term not in the final resume: {t}"
                  for t in unused_context]

    return (final_text, guardrail, removals, warnings, declined,
            unused_context)


def _build_context_weave_prompt(state: AgentState, units, terms: list[str]) -> str:
    """The one-shot weave prompt (round 11): the writer sees the CURRENT
    draft's units - NOT the source, so first-pass rewrites are never lost -
    and the exact "DROPPED CONTEXT TERMS" marker the tests and the fake
    server key on. The job description is omitted on purpose: the draft is
    already role-shaped, and this is a repair pass, not a re-tailor."""
    notes = str(state.get('human_notes') or '')
    if len(notes) > 2000:
        notes = notes[:2000] + " ... [truncated]"

    if units.whole_document:
        units_block = (
            f'[id="{RESUME_UNIT_ID}"] (the ENTIRE current draft, shown in '
            "full):\n"
            f"{units.sections[0].body}")
        delivery = (
            "HOW DELIVERY WORKS - whole-document mode: the entire draft is"
            "\none unit. Return a replacement for it whose body is the"
            "\ncomplete rewoven resume (same shape and order as the draft)."
            "\nReturning an EMPTY sections list keeps the draft verbatim. Do"
            "\nnot return any other section id in this mode.")
    else:
        listed = "\n\n".join(
            f'[id="{u.id}"] (draft heading: "{u.heading_line}")\n{u.body}'
            for u in units.sections)
        units_block = (
            "DRAFT UNITS (the delivery code reassembles the document from"
            "\nwhat you return; any unit you do not return ships VERBATIM):\n"
            f"{listed}")
        delivery = (
            "HOW DELIVERY WORKS - by-section mode: every returned section"
            "\nreplaces ONLY its unit's body (the heading line is kept by"
            "\nthe delivery code); units you skip ship byte-identical. Return"
            "\nONLY the sections you actually change - a minimal answer is"
            "\nthe expected one. Returned ids the draft does not offer are"
            "\nDROPPED with an integrity note - only use the ids listed"
            "\nabove.")

    return f"""
    You are an expert Resume Writer performing a TARGETED REPAIR pass.

    TASK: a deterministic check found terms from the user's typed notes that
    the current draft never mentions. Re-weave that context into the draft
    faithfully. Return a STRUCTURED answer through the ImproverOutput tool -
    the delivery code assembles the document deterministically from what you
    return, trusting nothing.

    {delivery}

    DROPPED CONTEXT TERMS: {', '.join(terms)}

    RULES:
    1. TRUTHFULNESS GATE: weave only what the USER CONTEXT actually
       supports. Never invent projects, jobs, tools, integrations,
       credentials, or numbers in order to place a term. If a term cannot
       be woven truthfully, do NOT force it - decline it instead.
    2. METRIC INTEGRITY (zero tolerance): every number in your output must
       be copied from the USER CONTEXT or already present in the draft.
       Never coin, copy, or round numbers from anywhere else.
    3. PRESERVE ALL REAL LINKS and the identity header exactly as the draft
       shows them; never drop GitHub links, live demos, portfolio URLs,
       LinkedIn profile links, or email addresses.
    4. This is a weave, not a rewrite: keep the draft's wording, structure,
       and section order otherwise intact, and weave each term into the
       most fitting existing section body (Projects / Experience / Summary
       / Skills as fits).
    5. DECLINE PROTOCOL: any dropped term you cannot truthfully weave goes
       in `declined`, one short restatement per item, e.g.:
           "Experience with k8s"
       Never omit a dropped term silently.

    {_untrusted('USER NOTES', notes)}

    {_untrusted('CURRENT DRAFT UNITS', units_block)}

    ANSWER SHAPE: return ONLY the ImproverOutput tool call:
        sections: [{{"id": <unit id>, "body": <markdown body, no heading line>}}, ...]
        declined: ["<refused term>", ...]
    Do NOT wrap the answer in code fences.
    """


def _context_weave_pass(state: AgentState, draft: str, terms: list[str]
                        ) -> tuple[Optional[str], list[str], list[str]]:
    """The one-shot targeted weave pass (round 11): parses the CURRENT draft
    into units and asks the writer to re-weave the dropped context terms.
    Returns ``(assembled, structured_declined, assembly_warnings)``, or
    ``(None, [], [])`` when the structured call fails entirely - the caller
    then keeps the detection-only warning with no content change."""
    units = parse_resume_units(draft)
    messages = [
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=_build_context_weave_prompt(state, units, terms)),
    ]
    try:
        answer, _method = invoke_structured_with_repair(
            _llms.llm_creative, ImproverOutput, messages,
            salvage=_make_improver_salvage(units), label="context-weave")
    except Exception as exc:
        _safe_print(f"   [context-weave] structured pass unavailable "
                    f"({exc}) - keeping the detection-only warning")
        return None, [], []
    assembled, assembly_warnings, structured_declined = \
        assemble_resume_units(units, answer)
    return assembled, structured_declined, assembly_warnings


def _postprocess_draft(state: AgentState, draft: str, current_iter: int, *,
                       allow_context_weave: bool = True) -> dict:
    """Deterministic enforcement tail shared by the improver and the polisher.

    Chain (in order): decline-protocol parsing -> hallucinated-section
    filter -> fabrication scrubber -> unevidenced-metric tripwire ->
    contact-link guard -> dropped-evidence tripwire -> unused-context
    tripwire (all in ``_patch_stack``). Accumulates the QW2/QW3 channels
    (declined_requests, scrub_removals, unevidenced_metrics) across passes;
    guardrail_warnings keeps its existing last-pass contract (server relays
    the final list as one warning event).

    Round 11 context weave-in: when the patch stack finds notes-sourced
    terms the final draft dropped and the attempt cap allows it, ONE extra
    structured pass re-weaves them. The woven draft re-enters the SAME
    patch stack and is adopted only if the unused-term count strictly
    decreases - the tripwire, not the model, decides.
    ``allow_context_weave=False`` marks a draft the structured writer did
    not produce (total writer failure): the one-shot attempt is not spent
    on a verbatim-source fallback a weave pass cannot improve.
    """
    final_text, guardrail, removals, warnings, declined, unused_context = \
        _patch_stack(state, draft)

    weave_attempted = False
    if (allow_context_weave and unused_context
            and not state.get("context_weave_attempted")):
        weave_attempted = True
        _safe_print(f"   [Integrity] Context weave-in pass triggered: "
                    f"{', '.join(unused_context)}")
        woven, weave_declined, weave_warnings = _context_weave_pass(
            state, final_text, unused_context)
        if woven is not None:
            # Deterministic acceptance: the woven draft re-enters the SAME
            # patch stack; adopt only if the unused-term count strictly
            # decreases. The tripwire is the acceptance test, never the
            # model's word - an unfaithful weave (or a plain echo of the
            # draft) fails this check and changes nothing.
            w_final, w_guardrail, w_removals, w_warnings, w_declined, \
                w_unused = _patch_stack(state, woven)
            if len(w_unused) < len(unused_context):
                _safe_print(f"   [Integrity] Context weave adopted: unused "
                            f"context terms {len(unused_context)} -> "
                            f"{len(w_unused)}")
                final_text, unused_context = w_final, w_unused
                guardrail = [*weave_warnings,
                             "user context woven via targeted pass",
                             *w_guardrail]
                removals = accumulate_unique(removals, w_removals)
                warnings = accumulate_unique(warnings, w_warnings)
                declined = accumulate_unique(
                    declined, [*w_declined, *weave_declined])

    result = {
        "optimized_resume": final_text,
        "iteration": current_iter,
        "guardrail_warnings": guardrail,
        "scrub_removals": accumulate_unique(state.get("scrub_removals"), removals),
        "unevidenced_metrics": accumulate_unique(state.get("unevidenced_metrics"), warnings),
        "declined_requests": accumulate_unique(state.get("declined_requests"), declined),
    }
    if weave_attempted:
        # Stamped on attempt (QW1), adoption or not. The CALLER merges this
        # with its own stamp - both _full_rewrite and the polisher's
        # targeted tail merge, never clobber.
        result["context_weave_attempted"] = True
        result["prompt_versions"] = {"context-weave": CONTEXT_WEAVE_PROMPT_VERSION}
    return result


def _full_rewrite(state: AgentState, *, node_label: str, version_key: str,
                  version: str) -> dict:
    """Structured full-rewrite invocation (improver pass and the polisher
    fallback), BB1 Phase B: the writer returns typed section units through
    the MR5 repair ladder and the delivery code assembles the document.
    Total failure falls back to the source resume verbatim with a loud
    integrity warning - an outage degrades to "untailored but honest"
    instead of killing the run."""
    base_content = state['optimized_resume'] if state['optimized_resume'] else state['resume_text']
    units = parse_resume_units(base_content)
    messages = [
        SystemMessage(content=SYSTEM_GUARDRAIL),
        HumanMessage(content=_build_improver_prompt(state, units)),
    ]
    assembly_warnings: list[str] = []
    structured_declined: list[str] = []
    allow_weave = True
    try:
        answer, _method = invoke_structured_with_repair(
            _llms.llm_creative, ImproverOutput, messages,
            salvage=_make_improver_salvage(units), label=node_label)
        _assembled, assembly_warnings, structured_declined = \
            assemble_resume_units(units, answer)
        assembled = _assembled
    except Exception as exc:
        _safe_print(f"   [{node_label or 'improver'}] structured writer "
                    f"unavailable ({exc}) - shipping the source resume "
                    f"verbatim")
        assembled = base_content
        assembly_warnings = [
            "resume writer unavailable - shipped the source resume "
            f"verbatim ({exc})"]
        # Round 11: skip the context weave on this path. The one-shot
        # attempt is reserved for drafts the writer actually produced -
        # burning it on the verbatim source cannot help the run.
        allow_weave = False

    result = _postprocess_draft(state, assembled,
                                state.get("iteration", 0) + 1,
                                allow_context_weave=allow_weave)
    # The assembly's structural findings lead the integrity list; the
    # shared deterministic tail's findings follow.
    result["guardrail_warnings"] = [
        *assembly_warnings, *(result.get("guardrail_warnings") or [])]
    result["declined_requests"] = accumulate_unique(
        result.get("declined_requests"), structured_declined)
    # Merge, never clobber: _postprocess_draft may have stamped the
    # context-weave attempt (QW1), and both stamps must survive.
    result["prompt_versions"] = {**(result.get("prompt_versions") or {}),
                                 version_key: version}
    return result


def improver_node(state: AgentState):
    current_iter = state.get("iteration", 0) + 1
    _safe_print(f"\n--- NODE 2: IMPROVING RESUME (Iteration {current_iter}) ---")
    return _full_rewrite(state, node_label="improver",
                         version_key="improver", version=IMPROVER_PROMPT_VERSION)
