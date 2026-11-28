"""Deterministic guard against unevidenced TOOL claims in model prose.

Live run 4afa44cb (Nov 19 2026): the reviewer told the polisher to "link your
existing automation work to workflow orchestration and n8n-like platforms"
even though the reviewer prompt forbids recommending tools the evidence does
not show. The polisher followed it; the resume scrubber removed the two n8n
claims, but the cover letter still shipped "n8n-style automation" and
requirement-framed name-drops of Claude Code / Cursor / Codex / MCP. The
letter's existing tripwire watches NUMBERS only, so this class of claim
survived every deterministic gate.

This module builds an evidence-derived deny set and removes whole sentences
that name a denied tool. It is shared by the reviewer (before its feedback
reaches the polisher) and the cover-letter node (before the letter ships).
No LLM calls, no input mutation, case-insensitive word-boundary matching with
variant forms, and a context rule for ambiguous common nouns (Make / Cursor).
"""
import re
from dataclasses import dataclass

from .helpers import _tidy_removal_residue


@dataclass(frozen=True)
class _ToolTerm:
    canonical: str
    pattern: re.Pattern
    ambiguous: bool = False


def _p(expr: str, *, ignore_case: bool = True) -> re.Pattern:
    return re.compile(expr, re.IGNORECASE if ignore_case else 0)


_WORD = r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"

# Maintained vocabulary. A term is only DENIED when it is absent from the
# candidate's evidence, so evidenced tools are never flagged. Ambiguous words
# require a capitalised form that is not sentence-initial (or an explicit
# multiword/domain form), so "make a plan" and "database cursor" pass.
TOOL_TERMS: tuple[_ToolTerm, ...] = (
    _ToolTerm("n8n", _p(_WORD.format(r"n8n"))),
    _ToolTerm("Zapier", _p(_WORD.format(r"zapier"))),
    _ToolTerm("Make", _p(_WORD.format(r"make\.com"))),
    _ToolTerm("Make", _p(_WORD.format(r"Make"), ignore_case=False),
              ambiguous=True),
    _ToolTerm("Claude Code", _p(_WORD.format(r"claude[\s\u00a0\-–—]+code"))),
    _ToolTerm("Cursor", _p(_WORD.format(r"Cursor"), ignore_case=False),
              ambiguous=True),
    _ToolTerm("Codex", _p(_WORD.format(r"Codex"), ignore_case=False)),
    _ToolTerm("MCP", _p(_WORD.format(r"MCP"))),
    _ToolTerm("MCP", _p(_WORD.format(r"model context protocol"))),
    _ToolTerm("LangChain", _p(_WORD.format(r"langchain"))),
    _ToolTerm("LangGraph", _p(_WORD.format(r"langgraph"))),
    _ToolTerm("LlamaIndex", _p(_WORD.format(r"llamaindex"))),
    _ToolTerm("Airflow", _p(_WORD.format(r"airflow"))),
    _ToolTerm("Kubernetes", _p(_WORD.format(r"kubernetes"))),
    _ToolTerm("Docker", _p(_WORD.format(r"docker"))),
    _ToolTerm("Terraform", _p(_WORD.format(r"terraform"))),
    _ToolTerm("Jenkins", _p(_WORD.format(r"jenkins"))),
    _ToolTerm("Salesforce", _p(_WORD.format(r"salesforce"))),
    _ToolTerm("HubSpot", _p(_WORD.format(r"hubspot"))),
    _ToolTerm("Figma", _p(_WORD.format(r"figma"))),
    _ToolTerm("Tableau", _p(_WORD.format(r"tableau"))),
    _ToolTerm("Notion", _p(_WORD.format(r"notion"))),
)

ALL_CANONICAL = frozenset(t.canonical for t in TOOL_TERMS)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_NEUTRAL_FEEDBACK = ("Strengthen action verbs and quantify existing "
                     "achievements using only evidence already in the resume.")


@dataclass(frozen=True)
class Removal:
    """One sentence removed, with the canonical tools that triggered it."""
    sentence: str
    terms: tuple[str, ...]
    stage: str

    @property
    def warning(self) -> str:
        return (f"{self.stage}: removed sentence naming unevidenced tool "
                + ", ".join(self.terms))


def _sentence_initial(text: str, pos: int) -> bool:
    """True when `pos` starts a sentence (or the string), used to reject the
    ambiguous capitalised forms at sentence start ("Make a plan")."""
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i < 0 or text[i] in ".!?;\n"


def matching_tools(text: str) -> set[str]:
    """Canonical tools named in `text` (case-insensitive, word-boundary)."""
    found: set[str] = set()
    if not text:
        return found
    for term in TOOL_TERMS:
        for m in term.pattern.finditer(text):
            if term.ambiguous and _sentence_initial(text, m.start()):
                continue
            found.add(term.canonical)
            break
    return found


def build_denied_tools(evidence_text: str = "", *,
                       declined_requests=None, job_text: str = "") -> set[str]:
    """Deny = maintained vocabulary + tools in the declined requests + tools
    named in the posting, MINUS anything the candidate's evidence shows.

    Evidence is the original resume plus notes plus any candidate profile
    text the caller supplies; an evidenced tool is NEVER denied.
    """
    allowed = matching_tools(evidence_text or "")
    candidates: set[str] = set(ALL_CANONICAL)
    for request in (declined_requests or []):
        candidates |= matching_tools(str(request))
    candidates |= matching_tools(job_text or "")
    return {term for term in candidates if term not in allowed}


def remove_sentences_naming_tools(text: str, denied: set[str], *,
                                  stage: str) -> tuple[str, list[Removal]]:
    """Removes whole sentences naming a denied tool, preserving paragraphs.

    Uses the resume scrubber's sentence split + residue tidy, so behaviour is
    identical to the existing deterministic removal path. Returns the cleaned
    text and one Removal per sentence (with every canonical term it named).
    """
    if not text or not denied:
        return text, []
    removals: list[Removal] = []
    paragraphs: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            paragraphs.append("")
            continue
        kept: list[str] = []
        removed_here = False
        for sentence in _SENTENCE_SPLIT.split(para):
            hits = tuple(sorted(matching_tools(sentence) & denied))
            if hits:
                removals.append(Removal(sentence.strip(), hits, stage))
                removed_here = True
                continue
            kept.append(sentence)
        if not removed_here:
            # Untouched prose - including salutations and signatures - stays
            # byte-identical (the residue tidy strips trailing commas).
            paragraphs.append(para)
            continue
        paragraphs.append(_tidy_removal_residue(" ".join(kept)) if kept else "")
    cleaned = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", "\n".join(paragraphs))
    return cleaned, removals


def sanitize_feedback(feedback: str, denied: set[str], *,
                      stage: str = "reviewer") -> tuple[str, list[Removal]]:
    """Drops feedback sentences naming a denied tool. If nothing is left,
    keeps cleaned empty so empty sanitized feedback does not trigger a wasted
    neutral polish pass."""
    cleaned, removals = remove_sentences_naming_tools(
        feedback, denied, stage=stage)
    return cleaned.strip(), removals


_GAP_FILLER_TOKENS = {
    "experience", "experienced", "with", "using", "knowledge", "skill",
    "skills", "tool", "tools", "strong", "hands", "building", "build",
    "development", "develop", "design", "systems", "system", "platform",
    "platforms", "pipelines", "pipeline", "modern", "production", "familiarity",
    "proficiency", "working", "understanding", "ability", "years", "and",
    "the", "for", "into", "from", "across", "based",
}
_GAP_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#/_-]{1,}")


def classify_gap_triage(
    missing_skills_text: str,
    resume_text: str,
    human_notes: str = "",
    declined_requests: tuple[str, ...] | list[str] = (),
) -> list[dict]:
    """Conservatively classifies each scanner gap into ``rephrasable`` vs
    ``needs_user_input``.

    Marks a gap ``rephrasable`` ONLY when unambiguous evidence already exists
    in ``resume_text`` or ``human_notes`` (and is not declined). Defaults to
    ``needs_user_input`` whenever unsure so the pipeline never fabricates
    missing tools or unverified experience.
    """
    evidence = "\n".join(p for p in (resume_text, human_notes) if p).strip()
    ev_lower = evidence.lower()
    ev_tools = matching_tools(evidence) if evidence else set()
    declined_lower = {str(d).strip().lower() for d in (declined_requests or []) if d}

    triage: list[dict] = []
    seen: set[str] = set()
    for raw in (missing_skills_text or "").splitlines():
        label = re.sub(r"^\s*(?:[-•*·]|\d+[.)])\s*", "", raw).strip()
        if not label or len(label) < 2:
            continue
        if re.match(r"^(?:missing|skills?)\s*:?$", label, re.IGNORECASE):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)

        if key in declined_lower:
            triage.append({
                "skill": label,
                "status": "needs_user_input",
                "reason": "Declined by user; requires explicit confirmation to include.",
            })
            continue

        gap_tools = matching_tools(label)
        if gap_tools:
            if gap_tools <= ev_tools:
                matched_list = ", ".join(sorted(gap_tools))
                triage.append({
                    "skill": label,
                    "status": "rephrasable",
                    "reason": f"Evidence already mentions {matched_list}; can be foregrounded by rewording.",
                })
            else:
                missing_list = ", ".join(sorted(gap_tools - ev_tools))
                triage.append({
                    "skill": label,
                    "status": "needs_user_input",
                    "reason": f"Missing tool/platform evidence ({missing_list}); needs user confirmation.",
                })
            continue

        # No canonical tool named: require ALL substantive non-filler terms to
        # appear in evidence; default to needs_user_input when unsure.
        tokens = [
            t for t in _GAP_TOKEN_RE.findall(label.lower())
            if t not in _GAP_FILLER_TOKENS and len(t) >= 3
        ]
        def _tok_present(tok: str) -> bool:
            stem = tok[:-1] if len(tok) > 4 and tok.endswith("s") else tok
            return bool(re.search(rf"(?<!\w){re.escape(stem)}s?(?!\w)", ev_lower))

        if tokens and all(_tok_present(tok) for tok in tokens):
            triage.append({
                "skill": label,
                "status": "rephrasable",
                "reason": "Related evidence terms exist in your resume/notes and can be emphasized directly.",
            })
        else:
            triage.append({
                "skill": label,
                "status": "needs_user_input",
                "reason": "No direct evidence found in your resume or notes; confirm before claiming.",
            })

    return triage
