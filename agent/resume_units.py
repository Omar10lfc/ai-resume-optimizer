"""BB1 Phase B: typed section units + the code-assembled resume delivery.

The Dossier study's ``resumeUnits()`` pattern ("trusting nothing"): the
writer never delivers a whole document. Code first sections the SOURCE
resume into typed units with ids, the writer returns replacements BY ID,
and the delivery code assembles the final document deterministically:

- the identity header (name + contact line) is pinned from the source
  verbatim - it is never model-owned (the rounds 5-7 defect class:
  rewrites that mangled or dropped contact data);
- any source unit the writer does not return ships VERBATIM (whole-document
  drift becomes structurally impossible for everything the writer skipped);
- returned ids the source does not offer are DROPPED with a visible
  integrity note - the writer cannot invent a section by naming it;
- standard sections the source lacks may be ADDED by id (the improver
  template shape), still subject to the evidence scrubbers downstream.

The reviewer/polisher stack keeps its own (already deterministic) section
merge; this module serves the improver's full-rewrite path - and the
polisher's full-rewrite fallback through it.

Degenerate input: a source with NO section marks (a heading-less paste)
becomes ONE whole-document unit (id ``Resume``) the writer may replace
wholesale; there is no reliably identifiable identity header to pin in
that shape, which is exactly today's semantics plus structure.
"""
import re
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field

from .helpers import (_strip_code_fences, canonical_feedback_section,
                      extract_declined_requests, markdown_link_count)

# The whole-document unit id (heading-less pastes). The prompt offers it
# verbatim and the test fakes key on "[id=\"Resume\"]" to mirror the offer.
RESUME_UNIT_ID = "Resume"

# Standard sections a writer may ADD when the source lacks them (the
# improver template shape); anything else outside the source ids is dropped.
STANDARD_SECTION_NAMES = (
    "Professional Summary", "Technical Skills", "Experience",
    "Projects", "Education", "Certifications",
)
_STANDARD_SECTION_KEYS = frozenset(name.casefold()
                                   for name in STANDARD_SECTION_NAMES)

# A source line is a section heading when it is a markdown ##+ heading or a
# short ALL-CAPS line (the same class _SECTION_MARK_RE uses for the contact
# region; single-# lines are NAME headings and belong to the header block).
_MARKDOWN_HEADING_RE = re.compile(r"^#{2,6}\s+\S")
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 &/'\-.:]{2,}$")

# Pre-heading lines are pinned as the identity header only when the source
# actually has section marks; a blob with no marks is whole-document instead.
_HEADER_PIN_MAX_LINES = 6


class SourceUnit(NamedTuple):
    """One source section: ``id`` (canonical name or cleaned raw heading),
    ``heading_line`` (the raw source heading; None for the whole-document
    unit), ``body`` (lines between headings, without the heading), and
    ``key`` (the matching key writer ids normalize to)."""
    id: str
    heading_line: Optional[str]
    body: str
    key: str


class ResumeUnits(NamedTuple):
    header_block: str
    sections: list
    whole_document: bool


def _clean_heading_text(heading_line: str) -> str:
    """'## Skills:' / 'SKILLS:' -> 'Skills'/'SKILLS' for id/key derivation."""
    text = re.sub(r"^#{1,6}\s*", "", str(heading_line or "").strip())
    return re.sub(r"\s+", " ", text).strip().rstrip(":").strip()


def _raw_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _unit_key(name: str) -> str:
    """The matching key for a writer id: canonical vocabulary name when the
    id maps to one (aliases like 'Skills' -> 'Technical Skills'), else the
    cleaned raw heading casefolded."""
    canon = canonical_feedback_section(_clean_heading_text(name))
    return canon.casefold() if canon else _raw_name(_clean_heading_text(name))


def parse_resume_units(source: str) -> ResumeUnits:
    """Sections an arbitrary source resume (raw PDF paste or markdown draft)
    into typed units. Deterministic: same source in, same units out."""
    lines = str(source or "").replace("\r\n", "\n").split("\n")
    marks = [i for i, line in enumerate(lines)
             if _MARKDOWN_HEADING_RE.match(line.rstrip())
             or _ALLCAPS_HEADING_RE.match(line.strip())]
    whole_document = not marks or len(lines[:marks[0]]) > _HEADER_PIN_MAX_LINES
    if whole_document:
        body = "\n".join(lines).strip("\n")
        return ResumeUnits(
            header_block="",
            sections=[SourceUnit(id=RESUME_UNIT_ID, heading_line=None,
                                 body=body, key=RESUME_UNIT_ID.casefold())],
            whole_document=True)

    header_block = "\n".join(lines[:marks[0]]).strip("\n")
    sections: list = []
    for pos, start in enumerate(marks):
        end = marks[pos + 1] if pos + 1 < len(marks) else len(lines)
        heading_line = lines[start].rstrip()
        name = _clean_heading_text(heading_line)
        canon = canonical_feedback_section(name)
        body = "\n".join(lines[start + 1:end]).strip("\n")
        sections.append(SourceUnit(
            id=canon or name,
            heading_line=heading_line,
            body=body,
            key=canon.casefold() if canon else _raw_name(name)))
    return ResumeUnits(header_block=header_block, sections=sections,
                       whole_document=False)


class ImproverSection(BaseModel):
    """One writer-returned section replacement (or addition)."""
    id: str = Field(
        description=(
            "The source unit id this body replaces, or a standard section "
            "name (Professional Summary, Technical Skills, Experience, "
            "Projects, Education, Certifications) to add. Ids the source "
            "does not offer are dropped."))
    body: str = Field(
        default="",
        description=(
            "The markdown body WITHOUT the '## ' heading line - the "
            "delivery code emits the heading itself."))


class ImproverOutput(BaseModel):
    """The improver's structured answer (BB1 Phase B): section replacements
    keyed by source-unit ids plus the decline-protocol list. The delivery
    code - not the writer - assembles the final resume document."""
    sections: list[ImproverSection] = Field(default_factory=list)
    declined: list[str] = Field(
        default_factory=list,
        description=("Requests (from the missing-skills list, the user "
                     "context, or the feedback) the writer refused because "
                     "the evidence does not support them."))


def improver_output_from_markdown(markdown: str, *,
                                  whole_document: bool = False) -> ImproverOutput:
    """Converts a markdown draft into ImproverOutput - the salvage rung for
    a writer that answered in prose instead of the tool call, and the seam
    the test fakes use to serve the structured contract from their
    markdown. DECLINED: lines become structured entries; content before the
    first '## ' heading is dropped (the identity header is pinned from the
    source, never taken from the writer)."""
    text = _strip_code_fences(str(markdown or "")).replace("\r\n", "\n")
    text, declined = extract_declined_requests(text)
    if whole_document or "## " not in text:
        return ImproverOutput(
            sections=[ImproverSection(id=RESUME_UNIT_ID, body=text.strip())],
            declined=declined)
    lines = text.split("\n")
    marks = [i for i, line in enumerate(lines)
             if _MARKDOWN_HEADING_RE.match(line.rstrip())]
    if not marks:
        return ImproverOutput(
            sections=[ImproverSection(id=RESUME_UNIT_ID, body=text.strip())],
            declined=declined)
    sections = []
    for pos, start in enumerate(marks):
        end = marks[pos + 1] if pos + 1 < len(marks) else len(lines)
        name = _clean_heading_text(lines[start])
        body = "\n".join(lines[start + 1:end]).strip("\n")
        sections.append(ImproverSection(id=name, body=body))
    return ImproverOutput(sections=sections, declined=declined)


def _emit_source_unit(unit: SourceUnit) -> str:
    """The verbatim source unit, heading normalized to the draft shape:
    markdown headings pass through byte-identical (the polisher's
    byte-stability principle); ALL-CAPS raw headings become '## <id>'.
    The whole-document unit (no heading line) ships its body verbatim."""
    if unit.heading_line is None:
        return unit.body
    if unit.heading_line.lstrip().startswith("#"):
        heading = unit.heading_line
    else:
        heading = f"## {unit.id}"
    if unit.body:
        return f"{heading}\n{unit.body}"
    return heading


def _emit_rewrite(unit: SourceUnit, body: str) -> str:
    """Heading (kept from the source, normalized) + the WRITER's body - the
    heading is code-owned, the body is the only part the writer replaces."""
    if unit.heading_line is not None and unit.heading_line.lstrip().startswith("#"):
        heading = unit.heading_line
    else:
        heading = f"## {unit.id}"
    if body.strip():
        return f"{heading}\n{body.strip()}"
    return heading


def _gated_rewrite(unit: SourceUnit,
                   body: str) -> tuple[Optional[str], Optional[str]]:
    """Deterministic per-unit gates for a writer body. Returns
    (emitted_text, warning) - emitted None means keep the source unit."""
    if not body.strip():
        if not unit.body.strip():
            return _emit_source_unit(unit), None   # empty -> empty: no-op
        return None, (f"writer returned an empty rewrite for '{unit.id}' "
                      f"- kept the source section")
    if markdown_link_count(body) < markdown_link_count(unit.body):
        return None, (f"rewrite for '{unit.id}' dropped a source link - "
                      f"kept the source section")
    return _emit_rewrite(unit, body), None


def assemble_resume_units(units: ResumeUnits,
                          answer: ImproverOutput) -> tuple[str, list[str], list[str]]:
    """The trusting-nothing assembly. Returns
    ``(assembled_text, assembly_warnings, declined_requests)``.

    By-section mode: rewrites replace their unit's body in source order
    (heading kept), skipped units ship verbatim, unknown ids are dropped
    with an integrity note, standard additions are appended. Whole-document
    mode: one 'Resume' replacement ships verbatim; anything else is dropped
    with a note, and no answer at all keeps the source.
    """
    warnings: list[str] = []
    declined = [d for d in (answer.declined or []) if d]

    if units.whole_document:
        source = _emit_source_unit(units.sections[0])
        replacement: Optional[ImproverSection] = None
        outsiders: list[str] = []
        for section in answer.sections:
            if _unit_key(section.id) == units.sections[0].key:
                if replacement is None:
                    replacement = section
                else:
                    warnings.append(
                        f"writer returned the '{units.sections[0].id}' unit "
                        f"twice - kept the first")
            else:
                outsiders.append(section.id)
        if replacement is None:
            if outsiders:
                warnings.append(
                    "writer returned section id(s) outside the "
                    f"'{units.sections[0].id}' whole-document unit - "
                    f"dropped: {', '.join(outsiders)}")
            return source, warnings, declined
        body = replacement.body.strip()
        if markdown_link_count(body) < markdown_link_count(units.sections[0].body):
            warnings.append(
                "whole-document rewrite dropped a source link - kept the "
                "source resume verbatim")
            return source, warnings, declined
        return (body if body else source), warnings, declined

    by_key: dict[str, ImproverSection] = {}
    adds: list[ImproverSection] = []
    for section in answer.sections:
        key = _unit_key(section.id)
        if key in {u.key for u in units.sections}:
            if key in by_key:
                warnings.append(f"writer returned section id "
                                f"'{section.id}' twice - kept the first")
                continue
            by_key[key] = section
        elif key in _STANDARD_SECTION_KEYS:
            if not section.body.strip():
                warnings.append(f"writer returned an empty addition for "
                                f"'{section.id}' - dropped")
                continue
            adds.append(section)
        else:
            warnings.append(f"writer returned unknown section id "
                            f"'{section.id}' - dropped")

    pieces: list[str] = []
    if units.header_block:
        pieces.append(units.header_block)
    for unit in units.sections:
        section = by_key.get(unit.key)
        if section is None:
            pieces.append(_emit_source_unit(unit))
            continue
        text, gate_warning = _gated_rewrite(unit, section.body)
        if gate_warning:
            warnings.append(gate_warning)
        pieces.append(text if text is not None else _emit_source_unit(unit))
    for add in adds:
        pieces.append(f"## {add.id}\n{add.body.strip()}")
    if not answer.sections:
        warnings.append("writer returned no sections - the source resume "
                        "shipped verbatim")
    return "\n".join(pieces), warnings, declined
