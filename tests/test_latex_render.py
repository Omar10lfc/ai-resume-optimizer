"""Tests for the LaTeX resume renderer.

The markdown→structured and tex-template tests are pure (no external tools).
PDF/DOCX compile tests run only when pdflatex/pandoc are present on PATH, so
they pass on CI machines without a TeX distribution.
"""

import os
import re
import sys
from pathlib import Path

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
from agent.latex_render import (  # noqa: E402
    parse_markdown_resume,
    render_resume_tex,
    latex_to_pdf,
    latex_to_docx,
    HAVE_PDFLATEX,
    HAVE_PANDOC,
    _latex_escape,
)


SAMPLE = """# Jake Ryan
123-456-7890 | jake@su.edu | linkedin.com/in/jake | github.com/jake
## Education
Southwestern University, Georgetown, TX
B.S. in Computer Science, May 2021
- Relevant coursework: Data Structures, Algorithms
## Experience
Software Engineer | Acme Corp | June 2020 -- Present
- Developed a REST API using FastAPI and PostgreSQL
## Technical Skills
**Languages**: Python, Java, C
**Frameworks**: React, Node.js, FastAPI"""


# ==========================================
# parse_markdown_resume
# ==========================================

def test_parse_extracts_contact():
    data = parse_markdown_resume(SAMPLE)
    assert data.contact.name == "Jake Ryan"
    assert "@" in data.contact.details
    assert "123-456-7890" in data.contact.details


def test_parse_sections_in_order():
    data = parse_markdown_resume(SAMPLE)
    titles = [s.title for s in data.sections]
    assert titles == ["Education", "Experience", "Technical Skills"]


def test_parse_education_maps_four_columns():
    data = parse_markdown_resume(SAMPLE)
    edu = data.sections[0].entries[0]
    assert edu.header == "Southwestern University"
    assert edu.right == "Georgetown, TX"
    assert edu.sub == "B.S. in Computer Science"
    assert "May 2021" in edu.sub_right


def test_parse_experience_keeps_dates_and_bullets():
    data = parse_markdown_resume(SAMPLE)
    exp = data.sections[1].entries[0]
    assert exp.header == "Software Engineer"
    assert exp.right == "Acme Corp"
    assert "2020" in exp.sub_right
    assert exp.items == ["Developed a REST API using FastAPI and PostgreSQL"]


def test_parse_skills_as_simple_lines():
    data = parse_markdown_resume(SAMPLE)
    skills = data.sections[2]
    assert skills.entries == []
    assert any("Languages" in ln for ln in skills.simple_lines)


def test_parse_empty_input_returns_empty():
    data = parse_markdown_resume("")
    assert data.contact.name == ""
    assert data.sections == []


def test_parse_handles_no_leading_heading():
    data = parse_markdown_resume(
        "## Experience\nEngineer | Co | 2020 -- 2021\n- Did stuff")
    assert data.sections[0].title == "Experience"
    assert data.sections[0].entries[0].header == "Engineer"


def test_parse_bullets_following_entry_stay_with_entry():
    data = parse_markdown_resume(
        "## Experience\nRole | Org\n2020 -- Present\n- One\n- Two\n## Skills\nX")
    exp = data.sections[0].entries[0]
    assert exp.items == ["One", "Two"]


def test_parse_strips_bold_and_italic_spanning_pipes():
    """Markdown emphasis can span across '|' separators (e.g.
    '**Role** | Org | *Date*'). Markers on ALL columns must be stripped."""
    data = parse_markdown_resume(
        "## Experience\n"
        "**Computer Vision Trainee** | NTI | *Aug 2025 - Sep 2025*\n"
        "- Did work")
    e = data.sections[0].entries[0]
    assert e.header == "Computer Vision Trainee"
    assert e.right == "NTI"
    assert e.sub_right == "Aug 2025 - Sep 2025"
    assert "**" not in e.header and "*" not in e.sub_right


def test_parse_skips_horizontal_rules():
    """Markdown --- dividers must be ignored, not turned into empty entries."""
    data = parse_markdown_resume(
        "## Experience\n"
        "Role A | Co | 2020 -- 2021\n- One\n\n---\n\n"
        "Role B | Co2 | 2022 -- 2023\n- Two")
    entries = data.sections[0].entries
    assert [e.header for e in entries] == ["Role A", "Role B"]


def test_render_omits_empty_entries():
    """An entry with no header/right/sub/sub_right must not render as filler."""
    from agent.latex_render import Entry, Section, ResumeData, Contact
    data = ResumeData(contact=Contact())
    data.sections.append(Section(title="Experience", entries=[Entry()]))
    tex = render_resume_tex(data)
    body = tex.split("\\begin{document}")[1]
    # no subheading rows or empty entry lists for the all-empty section
    assert "\\resumeSubheading" not in body
    assert "\\resumeSubHeadingListStart" not in body
    assert "\\resumeSubHeadingListEnd" not in body


def test_parse_strips_parens_wrapping_dates():
    """LLM often wraps a date in parens split across lines, e.g.
    '**Title** (\n*Sep 2025 - Jun 2026)*'. The '(' and ')' must not leak into
    the neighbouring header / date fields."""
    data = parse_markdown_resume(
        "## Projects\n"
        "**Alexandria Port Digital Twin - Live AIS-Fed Optimization System** (\n"
        "*Sep 2025 - Jun 2026)*\n"
        "- Built a solver")
    e = data.sections[0].entries[0]
    assert e.header == "Alexandria Port Digital Twin - Live AIS-Fed Optimization System"
    assert not e.header.endswith("(")
    assert e.sub_right == "Sep 2025 - Jun 2026"
    assert not e.sub_right.endswith(")") and not e.sub_right.startswith("(")


def test_parse_keeps_long_summary_as_simple_line():
    """A long Summary paragraph must not be split into a subheading entry."""
    data = parse_markdown_resume(
        "## Summary\n"
        "AI Engineer with a strong foundation in Python, LLM/RAG systems, and "
        "multi-agent orchestration. Proven ability to design, implement, and "
        "deploy end-to-end AI solutions for Arabic NLP and generative applications.")
    sec = data.sections[0]
    assert sec.entries == []
    assert len(sec.simple_lines) == 1
    assert "AI Engineer with a strong foundation" in sec.simple_lines[0]


def test_parse_plain_label_line_stays_simple():
    """A non-bold 'label: value' skill row must not be split into an entry."""
    data = parse_markdown_resume(
        "## Skills\n"
        "Machine Learning & Deep Learning: PyTorch, Hugging Face Transformers, "
        "Scikit-learn, model fine-tuning\n"
        "**Languages**: Python, SQL")
    sec = data.sections[0]
    assert sec.entries == []
    assert any("Machine Learning & Deep Learning" in ln for ln in sec.simple_lines)


def test_parse_summary_section_forces_simple_lines():
    """Under a prose section (Summary), even short/comma-heavy lines must stay
    simple lines and never become structured entries."""
    data = parse_markdown_resume(
        "## Summary\n"
        "AI Engineer experienced in Python, LLM/RAG, and multi-agent systems.\n"
        "- Focused on Arabic NLP and generative applications.")
    sec = data.sections[0]
    assert sec.entries == []
    assert len(sec.simple_lines) == 2
    assert any("AI Engineer experienced" in ln for ln in sec.simple_lines)
    assert any("Focused on Arabic NLP" in ln for ln in sec.simple_lines)


def test_parse_objective_profile_about_also_prose():
    for title in ("Objective", "Profile", "About Me"):
        data = parse_markdown_resume(
            f"## {title}\nResults-driven engineer, Python, scalable systems.")
        sec = data.sections[0]
        assert sec.entries == []
        assert len(sec.simple_lines) == 1


def test_contact_splits_on_bullet_separator():
    """Contact details joined by '•' (or ' - ') must be split so URLs/emails
    get \\href wrapping, matching the '|' behavior."""
    from agent.latex_render import Contact
    tex_body = __render_contact_fixture(Contact(name="Jane",
                                                details="jane@x.com \u2022 github.com/jane \u2022 123-456-7890"))
    assert "\\href{mailto:jane@x.com}" in tex_body
    assert "github" in tex_body
    assert "\\href" in tex_body


def __render_contact_fixture(contact):
    from agent.latex_render import _render_contact
    return _render_contact(contact)



# ==========================================
# _latex_escape
# ==========================================

def test_latex_escape_specials():
    assert _latex_escape("a&b%c#_d") == r"a\&b\%c\#\_d"
    assert _latex_escape("$5{0}") == r"\$5\{0\}"


def test_latex_escape_plain_text_unchanged():
    assert _latex_escape("Hello World 123") == "Hello World 123"


# ==========================================
# render_resume_tex
# ==========================================

def test_render_produces_document():
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    assert "\\documentclass" in tex
    assert "\\begin{document}" in tex
    assert "\\end{document}" in tex


def test_render_contains_contact_and_sections():
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    assert "Jake Ryan" in tex
    assert "Southwestern University" in tex
    assert "Software Engineer" in tex
    assert "Languages" in tex


def test_render_uses_template_macros():
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    assert "\\resumeSubheading" in tex
    assert "\\resumeItem" in tex
    assert "\\section{Education}" in tex


def test_render_escapes_special_chars_in_bullets():
    data = parse_markdown_resume(
        "## Skills\n- Python & Shell (100%)\n- C# and C++")
    tex = render_resume_tex(data)
    assert r"\&" in tex
    assert r"\%" in tex
    assert r"\#" in tex


def test_render_empty_has_placeholder_note():
    tex = render_resume_tex(parse_markdown_resume(""))
    assert "No structured content" in tex


def test_render_skips_empty_sections():
    """A bare section heading (no entries, no bullets) must not emit an
    executed itemize with zero \\item (pdflatex: 'missing \\item').
    """
    tex = render_resume_tex(
        parse_markdown_resume("## Skills\n**Languages**: Python\n## References"))
    body = tex.split("\\end{document}")[0]
    # the one populated section renders an itemize with an \\item
    assert "\\begin{itemize}" in body
    assert "\\item" in body
    # the empty "References" section is dropped entirely
    assert "References" not in body


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_latex_to_pdf_compiles_with_empty_section(tmp_path):
    """Regression: a section with no entries and no bullets used to emit an
    empty \\begin{itemize}, making pdflatex fail with 'missing \\item'. Now the
    empty section is skipped and the PDF still compiles."""
    tex = render_resume_tex(
        parse_markdown_resume(
            "# Jane\njane@x.com\n## Skills\n**Languages**: Python\n"
            "## Certifications\n## Awards"))
    pdf = latex_to_pdf(tex, tmp_path, "out.pdf")
    assert pdf
    assert Path(pdf).exists() and Path(pdf).stat().st_size > 0


def test_latex_escape_strips_non_ascii_for_pdflatex():
    """pdflatex errors on non-ASCII glyphs like U+202F narrow NBSP, •, and
    en-dashes. _latex_escape must map them to ASCII before escaping."""
    escaped = _latex_escape("a\u202fb\u2022c\u2013d\xa0e")
    assert escaped == r"a b-c-d e"
    # every char in the result must be ASCII
    assert all(ord(c) < 128 for c in escaped)


def test_latex_escape_maps_arrow_mojibake_to_ascii():
    """'→' must map to '->' and the '¿' mojibake artifact must not leak."""
    escaped = _latex_escape("ROUGE-L 13.48 \u2192 29.56")
    assert escaped == r"ROUGE-L 13.48 -> 29.56"
    assert _latex_escape("raising ROUGE-L from 13.48 \u00bf 29.56") == \
        r"raising ROUGE-L from 13.48 -> 29.56"
    assert all(ord(c) < 128 for c in escaped)


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_latex_to_pdf_compiles_with_unicode_glyphs(tmp_path):
    """Regression: a resume containing U+202F narrow no-break space and •
    separators (common in scraped/LLM contact lines) used to make pdflatex
    fail with 'Unicode character ... not set up'. It must now compile."""
    from pypdf import PdfReader
    tex = render_resume_tex(parse_markdown_resume(
        "# Omar\nOmar10lfc\u2022 huggingface.co/Omar10lfc | \u202fx@y.com | 123\n"
        "## Experience\nEngineer | Acme | 2020 -- Present\n"
        "- Built a REST API using FastAPI\u202fand PostgreSQL\n"
        "## Skills\n**Languages**: Python, C#\n"))
    # no residual non-ASCII in the rendered body
    body = tex.split("\\end{document}")[0]
    assert all(ord(c) < 128 for c in body)
    pdf = latex_to_pdf(tex, tmp_path, "out.pdf")
    assert pdf
    assert Path(pdf).exists() and Path(pdf).stat().st_size > 0
    text = "".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
    assert "Omar" in text


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_full_resume_compiles(tmp_path):
    """A realistic multi-section resume (matching how the LLM formats output)
    must compile via pdflatex with all sections present and no stray markdown
    markers or parenthetical leaks in the rendered text."""
    from pypdf import PdfReader
    realistic = """# Omar Mashaly
Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com
linkedin.com/in/omar-mashaly • github.com/Omar10lfc

## Professional Summary
AI Engineer with a strong foundation in Python, LLM/RAG systems, and multi-agent
orchestration. Proven ability to design and deploy end-to-end AI solutions.

## Skills
**Programming & Scripting**: Python, SQL, Bash, Git version control
Machine Learning & Deep Learning: PyTorch, Hugging Face Transformers, scikit-learn
**Languages**: Arabic (native), English (fluent)

## Experience
**Computer Vision & Deep Learning Trainee** | NTI | *Aug 2025 - Sep 2025*
- Built and evaluated five CV models across medical, agricultural domains
- Applied automated testing and CI pipelines for reproducibility

**Data Science Intern** | DEPI | *Apr 2024 - Oct 2024*
- Led end-to-end development of a clinical diagnosis system

## Projects
**Alexandria Port Digital Twin - Live AIS-Fed Optimization System** (
*Sep 2025 - Jun 2026)*
- Developed a CP-SAT berth-allocation solver with OR-Tools
---

**Fil-RAG-Goal: Arabic Football Q&A** | 2025
- Combined BM25 and dense retrieval (RRF), improving MRR by 7 %
- Built a two-tier Groq cascade with rate-limit fallback

## Education
**B.Sc. in Computer Science, AI & Data Science** | E-JUST
*Oct 2022 - Jun 2026*

## Certifications
AWS Future AI Engineer Nanodegree - Udacity (Nov 2025)
Natural Language Processing in Python - Maven Analytics (Oct 2025)
"""
    data = parse_markdown_resume(realistic)
    tex = render_resume_tex(data)
    body = tex.split("\\end{document}")[0]
    # no stray markdown markers in rendered text
    assert "**" not in body
    # no parenthetical leak in any rendered subheading header
    for m in re.findall(r"\\resumeSubheading\{([^}]*)\}", body):
        assert not m.endswith("("), f"trailing '(' leaked into: {m!r}"
        assert not m.endswith(")"), f"trailing ')' leaked into: {m!r}"
    pdf = latex_to_pdf(tex, tmp_path, "out.pdf")
    assert pdf
    assert Path(pdf).exists() and Path(pdf).stat().st_size > 0
    text = "".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
    for token in ("Omar", "Professional Summary", "Skills", "Experience",
                  "Projects", "Education", "Certifications", "NTI"):
        assert token in text, f"missing '{token}' in compiled PDF"

@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_latex_to_pdf_compiles(tmp_path):
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    pdf = latex_to_pdf(tex, tmp_path, "out.pdf")
    assert pdf
    assert Path(pdf).exists() and Path(pdf).stat().st_size > 0


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_latex_to_pdf_pdf_is_parsable(tmp_path):
    from pypdf import PdfReader
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    pdf = latex_to_pdf(tex, tmp_path, "out.pdf")
    reader = PdfReader(pdf)
    text = "".join(p.extract_text() or "" for p in reader.pages)
    assert "Jake Ryan" in text
    assert "Software Engineer" in text


# ==========================================
# latex_to_docx (only if pandoc present)
# ==========================================

@pytest.mark.skipif(not HAVE_PANDOC, reason="pandoc not installed")
def test_latex_to_docx_compiles(tmp_path):
    from zipfile import ZipFile
    tex = render_resume_tex(parse_markdown_resume(SAMPLE))
    docx = latex_to_docx(tex, tmp_path, "out.docx")
    assert docx
    assert Path(docx).exists() and Path(docx).stat().st_size > 0
    xml = ZipFile(docx).read("word/document.xml").decode("utf-8")
    assert "Jake Ryan" in xml
    assert "Software Engineer" in xml


def test_latex_to_docx_returns_none_without_pandoc(monkeypatch, tmp_path):
    if not HAVE_PANDOC:
        pytest.skip("pandoc not installed")
    import agent.latex_render as lr
    monkeypatch.setattr(lr, "HAVE_PANDOC", False)
    assert latex_to_docx("dummy tex", tmp_path) is None


# ==========================================
# New robustness tests
# ==========================================

def test_contact_extracted_from_h2_name():
    """Candidate name given as ## Name must be recognized as contact, not as a section."""
    data = parse_markdown_resume(
        "## Omar Mashaly\n"
        "Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com\n"
        "linkedin.com/in/omar-mashaly • github.com/Omar10lfc\n"
        "## Experience\n"
        "Engineer | Acme | 2024 -- Present\n- Built systems"
    )
    assert data.contact.name == "Omar Mashaly"
    assert "omarmashaly86@gmail.com" in data.contact.details
    assert "Omar Mashaly" not in [s.title for s in data.sections]
    assert len(data.sections) == 1
    assert data.sections[0].title == "Experience"


def test_contact_extracted_from_contact_heading():
    """Contact under ## Contact heading must populate contact without a Contact section."""
    data = parse_markdown_resume(
        "## Contact\n"
        "Omar Mashaly\n"
        "Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com\n"
        "## Skills\n"
        "**Languages**: Python, C++"
    )
    assert data.contact.name == "Omar Mashaly"
    assert "omarmashaly86@gmail.com" in data.contact.details
    assert "Contact" not in [s.title for s in data.sections]
    assert len(data.sections) == 1
    assert data.sections[0].title == "Skills"


def test_resume_project_heading_used_for_projects():
    """Projects section must use \\resumeProjectHeading instead of \\resumeSubheading."""
    data = parse_markdown_resume(
        "## Projects\n"
        "**Gitlytics** | Python, Flask, React | June 2020 -- Present\n"
        "- Built full-stack app\n"
        "**Simple Paintball** | May 2018 -- May 2020\n"
        "- Developed plugin"
    )
    tex = render_resume_tex(data)
    assert "\\resumeProjectHeading" in tex
    assert "Gitlytics" in tex
    assert "Simple Paintball" in tex


def test_comma_separated_prose_not_split():
    """Prose sentence with commas must not be split into header and right columns."""
    from agent.latex_render import _split_header_and_dates, _should_split_on_comma
    sentence = "Across all projects and roles, consistently applied infrastructure-as-code (AWS CloudFormation/Terraform) and automated testing."
    assert not _should_split_on_comma(sentence)
    h, s, r, sr = _split_header_and_dates(sentence)
    assert r == ""
    assert "consistently applied" in h


def test_skills_rendered_compact_single_item():
    """Skills section simple lines must be rendered in a single compact item."""
    data = parse_markdown_resume(
        "## Technical Skills\n"
        "**Languages**: Java, Python, C/C++\n"
        "**Frameworks**: React, Node.js, FastAPI\n"
        "**Developer Tools**: Git, Docker"
    )
    tex = render_resume_tex(data)
    body = tex.split("\\begin{document}")[1]
    # Should use the compact \\ style, not 3 separate \item blocks
    assert "\\\\" in body
    assert body.count("\\small{\\item{") == 1


def test_parse_education_four_columns_university_location_degree_dates():
    """Education in 'University | Location | Degree | Dates' format must map:
    header = University (Row 1 left), right = Location (Row 1 right),
    sub = Degree (Row 2 left), sub_right = Dates (Row 2 right) to prevent collision."""
    md = (
        "## Education\n"
        "**Egypt-Japan University of Science and Technology (E-JUST)** | Alexandria | "
        "B.Sc. in Computer Science, AI & Data Science | Oct 2022 - Jun 2026\n"
    )
    data = parse_markdown_resume(md)
    edu = data.sections[0].entries[0]
    assert edu.header == "Egypt-Japan University of Science and Technology (E-JUST)"
    assert edu.right == "Alexandria"
    assert edu.sub == "B.Sc. in Computer Science, AI & Data Science"
    assert edu.sub_right == "Oct 2022 - Jun 2026"


def test_long_project_heading_uses_two_line_subheading_no_collision():
    """Long project titles (> 70 chars with stack + date) must use \resumeSubheading
    to place tech stack on Row 2, preventing column overflow and date collision."""
    md = (
        "## Projects\n"
        "**Alexandria Port Digital Twin - Live AIS-Fed Optimization System** | "
        "Python, OR-Tools, AWS, FastAPI | Sep 2025 - Jun 2026\n"
        "- Engineered solver\n"
    )
    data = parse_markdown_resume(md)
    tex = render_resume_tex(data)
    assert "\\resumeSubheading{Alexandria Port Digital Twin - Live AIS-Fed Optimization System}{Sep 2025 - Jun 2026}{Python, OR-Tools, AWS, FastAPI}{}" in tex


def test_url_with_hash_percent_ampersand_compiles():
    """URLs containing #, %, and & must not break hyperref or tabular columns."""
    md = (
        "## Projects\n"
        "**Port Optimizer** [GitHub](https://github.com/user/repo#readme?tab=code&ref=main%20branch) | Python | 2025\n"
        "- Bullet\n"
    )
    data = parse_markdown_resume(md)
    tex = render_resume_tex(data)
    assert r"\href{https://github.com/user/repo\#readme?tab=code\&ref=main\%20branch}" in tex


def test_name_heading_size_is_reduced_not_huge():
    r"""Name heading must be rendered with \LARGE (or \Large for long names), never oversized \Huge."""
    from agent.latex_render import Contact, _render_contact
    c = Contact(name="Omar Mashaly", details="Alexandria, Egypt | omarmashaly86@gmail.com")
    tex = _render_contact(c)
    assert r"\LARGE" in tex or r"\Large" in tex
    assert r"\Huge" not in tex


def test_contact_hyperlinks_linkedin_github_clickable():
    r"""Contact header must turn LinkedIn, GitHub, and email into working \href hyperlinks."""
    from agent.latex_render import Contact, _render_contact
    c = Contact(name="Jane Doe", details="jane@example.com | linkedin.com/in/janedoe | github.com/janedoe")
    tex = _render_contact(c)
    assert r"\href{mailto:jane@example.com}{\underline{jane@example.com}}" in tex
    assert r"\href{https://linkedin.com/in/janedoe}{\underline{linkedin.com/in/janedoe}}" in tex
    assert r"\href{https://github.com/janedoe}{\underline{github.com/janedoe}}" in tex


def test_project_links_preserved_in_rendered_tex():
    r"""Project markdown links [GitHub](url) must be converted into clickable \href hyperlinks."""
    md = (
        "## Projects\n"
        "**Alexandria Port Digital Twin** [GitHub](https://github.com/user/port) | Python, FastAPI | 2025\n"
        "- Built API with [FastAPI Docs](https://fastapi.tiangolo.com)\n"
    )
    data = parse_markdown_resume(md)
    tex = render_resume_tex(data)
    assert r"\href{https://github.com/user/port}{\underline{GitHub}}" in tex
    assert r"\href{https://fastapi.tiangolo.com}{\underline{FastAPI Docs}}" in tex


def test_experience_rendering_company_beside_position():
    """Experience entries must place Role and Company side-by-side using \\resumeProjectHeading."""
    md = (
        "## Experience\n"
        "**Computer Vision & Deep Learning Trainee** | NTI | Aug 2025 - Sep 2025\n"
        "- Built 5 CV models\n"
    )
    data = parse_markdown_resume(md)
    tex = render_resume_tex(data)
    assert r"\resumeProjectHeading{\textbf{Computer Vision \& Deep Learning Trainee} $|$ \emph{NTI}}{Aug 2025 - Sep 2025}" in tex


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_realistic_dense_resume_compiles_single_page(tmp_path):
    """A realistic dense technical resume (4 projects, 2 jobs, full education, 9 skills,
    summary, and all 4 certifications) must compile to exactly 1 page with no cutoff."""
    from pypdf import PdfReader
    dense_md = """# Omar Mashaly
Alexandria, Egypt | +20 100 170 1821 | omarmashaly86@gmail.com | linkedin.com/in/omar-mashaly | github.com/Omar10lfc

## Summary
AI Engineer with a strong foundation in Python, LLM/RAG systems, and multi-agent orchestration. Proven ability to design and deploy end-to-end AI solutions.

## Skills
**Programming & Scripting**: Python, SQL, Bash, Git version control
**Machine Learning & Deep Learning**: PyTorch, Hugging Face Transformers, Scikit-learn, Model fine-tuning, Neural networks
**Optimization & OR**: OR-Tools CP-SAT, combinatorial optimization, IsolationForest, anomaly detection
**LLM & Generative AI**: Retrieval-Augmented Generation (RAG), multi-agent systems, LLM orchestration, prompt engineering
**NLP & Speech**: Arabic NLP, speech recognition, text summarization, vector search, cross-encoder reranking
**AI Coding Assistants**: Hands-on use of GitHub Copilot and Claude Code to improve code quality and development speed
**Agentic AI Frameworks**: Practical experience with LangGraph (or comparable orchestration tools) for building agentic workflows
**Cloud & MLOps**: AWS, Infrastructure as Code (IaC), REST API development, model deployment, test automation, CI/CD pipeline design, DevSecOps fundamentals
**Languages**: Arabic (native), English (fluent)

## Projects
**Alexandria Port Digital Twin - Live AIS-Fed Optimization System** [GitHub](https://github.com/Omar10lfc/port-optimization) | Python, OR-Tools, AWS, FastAPI | Sep 2025 - Jun 2026
- Engineered a CP-SAT berth-allocation solver that cut average vessel wait time by 47 % versus a FCFS baseline, delivering optimal schedules within a 5 s budget for 1,533 live instances.
- Developed an IsolationForest-based disruption-detection pipeline that auto-classifies 31 disruption events across 528 days, eliminating manual retraining overhead.

**Fil-RAG-Goal: Arabic Football Q&A (Hybrid RAG)** [GitHub](https://github.com/Omar10lfc/Fil-RAG-Goal) | Python, Hugging Face, FAISS, LangGraph | Feb 2026 - Jun 2026
- Integrated BM25 and dense retrieval via weighted RRF, boosting MRR by 7 % and raising intent-classification accuracy from 70 % to 97.2 %.
- Reduced LLM inference cost by ~70 % using a two-tier Groq cascade and hardened the system with prompt-injection defenses validated by a 66-test CI pipeline.
- Utilized LangGraph to orchestrate retrieval, ranking, and answer generation agents, enabling dynamic workflow adjustments.

**Smart Lecture Assistant - Arabic Audio Understanding and Retrieval** [GitHub](https://github.com/Omar10lfc/Smart-Lecture-Assistant) | Python, Whisper, AraBART, FAISS, Gradio | Mar 2026 - May 2026
- Fine-tuned Whisper-small, decreasing Arabic transcription WER from 42.69 % to 20.61 %; doubled summarization ROUGE-L from 13.48 to 29.56 with AraBART.
- Added a cross-encoder reranker to a CAMeL-BERT + FAISS stack, raising P@1 from 0.64 to 0.86. Deployed as a Gradio Space and FastAPI service with automated testing.

**AI Resume Optimizer Agent - Multi-Agent LLM System** [GitHub](https://github.com/Omar10lfc/ai-resume-optimizer) | Python, LangGraph, Hugging Face Spaces | Dec 2025 - Jan 2026
- Designed a self-correcting multi-agent workflow that iteratively removes hallucinated skills before human sign-off, delivering polished PDFs via automated CSS styling.
- Employed LangGraph for agent orchestration, enabling seamless task routing and state management across critique, generation, and formatting modules.

## Experience
**Computer Vision & Deep Learning Trainee** | NTI | Aug 2025 - Sep 2025
- Built and evaluated five CV models across medical, agricultural, and automotive domains, achieving up to 98.6 % pixel accuracy on lane segmentation.
- Utilized GitHub Copilot to accelerate model prototyping and debugging.

**Data Science Intern (IBM Data Science Track)** | DEPI | Apr 2024 - Oct 2024
- Led end-to-end development of a clinical diagnosis system on 100K patient records; shipped an XGBoost model (96.2 % accuracy, F1 0.961).
- Implemented CI/CD pipelines with automated testing and containerized deployment, following DevSecOps best practices.

## Education
**Egypt-Japan University of Science and Technology (E-JUST)** | Alexandria | B.Sc. in Computer Science, AI & Data Science | Oct 2022 - Jun 2026

## Certifications
AWS Future AI Engineer Nanodegree - Udacity (Nov 2025)
Natural Language Processing in Python - Maven Analytics (Oct 2025)
Generative AI Summer Code Camp - ITI & NVIDIA DLI (Sep 2025)
McKinsey Forward Program (Dec 2024)
"""
    data = parse_markdown_resume(dense_md)
    tex = render_resume_tex(data)
    pdf_path = latex_to_pdf(tex, tmp_path, "dense_resume.pdf")
    assert pdf_path is not None
    assert Path(pdf_path).exists()
    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 1
    page_text = reader.pages[0].extract_text()
    assert "AWS Future AI Engineer" in page_text
    assert "Natural Language Processing" in page_text
    assert "Generative AI Summer Code Camp" in page_text
    assert "McKinsey Forward Program" in page_text


