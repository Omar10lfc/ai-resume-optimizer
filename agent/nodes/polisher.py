"""NODE 2b: Polisher - the second (and final) rewriter pass.

Two modes:

- Section-targeted (default): the reviewer's structured ``sections`` answer
  names the drafts' own sections the feedback addresses; each of those gets
  ONE small rewrite call, and the results are merged back into the draft
  deterministically (agent/helpers.py ``merge_rewritten_sections``) - code
  builds the document, the model only rewrites a section. Typical feedback
  touches 1-2 sections, so a pass drops from a ~40 s whole-document rewrite
  to a few seconds, and whole-document drift (Nov 8 run 539f7c1d: the
  polisher edition dropped the Aegis header while keeping fabricated lines)
  becomes structurally impossible: untouched sections are byte-identical.

- Full-rewrite (fallback): the improver's structured units path (BB1
  Phase B: typed section replacements, code-assembled delivery) is reused
  verbatim whenever the feedback has no surgically addressable section
  (empty ``sections``, cross-cutting targets like "formatting", or a draft
  whose headers do not match the vocabulary). Best-of guard in ats_check
  still caps regressions.
"""
from ..state import AgentState, SYSTEM_GUARDRAIL, _untrusted
from .. import llms as _llms
from ..helpers import (_safe_print, _strip_code_fences, _safe_truncate,
                       split_markdown_sections, canonical_feedback_section,
                       merge_rewritten_sections, markdown_link_count)
from .improver import _full_rewrite, _postprocess_draft
from langchain_core.messages import HumanMessage, SystemMessage

# Full mode rides the improver's structured units path (improver@4, BB1
# Phase B); the version below marks the polisher's own decision (which mode
# shipped) for the prompt_versions stamp.
# polisher-target@2 (round 7, Nov 12 2026): section rewrites must keep
# claims the USER CONTEXT supports - a targeted rewrite of the section
# holding the pasted context's bullets could otherwise drop them silently.
POLISHER_FULL_VERSION = "polisher-full@1"
POLISHER_TARGET_VERSION = "polisher-target@2"

# Hard cap keeps the worst-case pass bound: a reviewer that lists five
# sections on one pass costs five small calls, capped here.
_MAX_TARGETS_PER_PASS = 3

def _rewrite_kept_links(old_body: str, new_body: str) -> bool:
    """Deterministic gate: a section rewrite must not drop markdown links
    (project/demo/GitHub URLs) the original body carried - the improver's
    "preserve all real links" contract is enforced in code, not trusted to
    the model."""
    return (markdown_link_count(new_body or "")
            >= markdown_link_count(old_body or ""))


def _build_section_prompt(state: AgentState, header_line: str, section_body: str,
                          rest_of_draft: str, feedback: str) -> str:
    return f"""
    You are an expert Resume Writer making a SURGICAL REVISION.

    TASK: Rewrite ONLY ONE section of an already-strong resume so it
    addresses the reviewer feedback. Everything else in the resume stays
    exactly as it is.

    REVISION TARGET (return the full replacement of this section only):
    {header_line}

    {_untrusted('CURRENT SECTION TEXT', section_body or '(empty)')}

    REVIEWER FEEDBACK TO ADDRESS:
    "{feedback}"

    RULES:
    1. Change as little as possible: keep every TRUE claim, tool, employer,
       project link, and metric the current section shows - including any
       claim the USER CONTEXT below supports. Rephrase,
       re-order, or sharpen only what the feedback asks about.
    2. CONTEXT-AWARE BREVITY: the rest of the resume is below; do not
       repeat what other sections already state.
    3. TRUTHFULNESS GATE: the candidate's evidence is the ORIGINAL RESUME
       plus the USER CONTEXT below only. Never add a tool, employer,
       project, credential, or number that evidence does not show - if an
       improvement would need invented content, skip it and keep the
       original wording for that part.
    4. Keep the draft's markdown conventions (bullets start with "- ",
       links live inline in the text, no code fences, no new sections).

    REST OF THE RESUME (context only - never rewrite or echo it):
    {rest_of_draft}

    {_untrusted('ORIGINAL RESUME', _safe_truncate(state.get('resume_text') or '', 6000, 'Original resume (polisher)'))}

    USER CONTEXT: {state.get('human_notes') or '(none)'}

    ANSWER SHAPE: return ONLY the replacement for the target section,
    starting with its original header line verbatim, then the rewritten
    body. No code fences, no commentary, no other sections.
    """


def _extract_section_body(response_text: str, header_line: str) -> str | None:
    """Parses a targeted-section rewrite out of the model answer: the body
    after its '## ' header. Returns None when nothing usable came back
    (the caller then refuses the whole merge and falls back)."""
    text = _strip_code_fences(str(response_text)).strip()
    if not text:
        return None
    blocks = split_markdown_sections(text)
    target = canonical_feedback_section(
        header_line[3:] if header_line.startswith("## ") else header_line)
    for _canon, _line, body in blocks:
        if _canon is not None and _canon == target:
            return body
    # A single-body answer without a header still counts: the model was
    # told to start with the header, but code decides from what arrived.
    if "##" not in text and len(blocks) == 0:
        return text
    return None


def polisher_node(state: AgentState):
    current_iter = state.get("iteration", 0) + 1
    feedback = (state.get("feedback") or "").strip()

    draft = state.get("optimized_resume") or ""
    wanted: list[str] = []
    for raw in (state.get("feedback_sections") or []):
        canon = canonical_feedback_section(raw)
        if canon and canon not in wanted:
            wanted.append(canon)
    if len(wanted) > _MAX_TARGETS_PER_PASS:
        _safe_print(f"   [Polish] {len(wanted)} sections requested - keeping first "
                    f"{_MAX_TARGETS_PER_PASS}.")
        wanted = wanted[:_MAX_TARGETS_PER_PASS]

    # Intersect with the draft's actual headers; unknown/cross-cutting names
    # cannot be merged surgically and simply do not count as targets.
    blocks = split_markdown_sections(draft)
    by_canon: dict[str, tuple[str, str]] = {}
    for canon, header_line, body in blocks:
        if canon is not None and canon not in by_canon:
            by_canon[canon] = (header_line, body)
    targets = [(c, *by_canon[c]) for c in wanted if c in by_canon]

    if not feedback or not targets:
        reason = ("reviewer feedback is empty" if not feedback
                  else "no feedback section matches the draft headers")
        _safe_print(f"   [Polish] Full-rewrite pass ({reason}).")
        return _full_rewrite(state, node_label="polisher",
                             version_key="polisher",
                             version=POLISHER_FULL_VERSION)

    _safe_print(f"--- NODE 2b: POLISHING (targeted: {', '.join(c for c, *_ in targets)}) "
                f"(Iteration {current_iter}) ---")

    # Context of everything that must NOT be rewritten (given to the model so
    # it avoids repeating what other sections already state - the sections
    # themselves are off-limits by construction: only targeted bodies merge).
    rest = "\n\n".join(f"{h}\n{b}" for c, h, b in blocks
                       if c not in {t[0] for t in targets})

    rewrites: dict[str, str] = {}
    for canon, header_line, section_body in targets:
        answer = _llms.llm_creative.invoke([
            SystemMessage(content=SYSTEM_GUARDRAIL),
            HumanMessage(content=_build_section_prompt(
                state, header_line, section_body, rest, feedback)),
        ])
        body = _extract_section_body(str(answer.content), header_line)
        if body is None:
            _safe_print(f"   [Polish] No usable rewrite returned for '{canon}'.")
        elif not _rewrite_kept_links(section_body, body):
            # Link-dropping rewrite is refused deterministically: keep the
            # original section rather than lose a project/demo URL.
            _safe_print(f"   [Polish] Rewrite for '{canon}' dropped links - keeping original.")
        else:
            rewrites[canon] = body

    merged, replaced = merge_rewritten_sections(draft, rewrites)
    if not replaced:
        # Refused merge (unknown shape / duplicate headers / empty answers):
        # fall through to the full-rewrite pass so the iteration is never lost.
        _safe_print("   [Polish] Targeted merge refused - falling back to full rewrite.")
        return _full_rewrite(state, node_label="polisher",
                             version_key="polisher",
                             version=POLISHER_FULL_VERSION)

    _safe_print(f"   [Polish] Merged {len(replaced)} rewritten section(s): "
                f"{', '.join(replaced)}.")
    return {**_postprocess_draft(state, merged, current_iter),
            "polish_sections_targeted": replaced,
            "prompt_versions": {"polisher": POLISHER_TARGET_VERSION}}
