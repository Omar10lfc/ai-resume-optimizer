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
    pdf_page_count,
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


def test_bold_label_lines_with_nested_bold_not_parsed_as_sections():
    """Verify that skill lines with multiple bold phrases (like '**Category**: text **Note**')
    are never misidentified as section headings."""
    sample = """# Jane Doe
test@example.com | 123-456-7890

## Technical Skills
**Machine Learning & Deep Learning**: Python, PyTorch, Scikit-learn
**LLM & Generative AI**: RAG, Multi-Agent Systems, **Conceptual Knowledge of LangGraph**
**Cloud & MLOps**: AWS, Docker, **Hands-on CI/CD pipeline automation**
**Languages**: Arabic (Native), English (Fluent)

## Experience
**Software Engineer** | Tech Corp | 2023 - Present
- Built scalable data pipelines
"""
    data = parse_markdown_resume(sample)
    section_titles = [s.title for s in data.sections]
    assert section_titles == ["Technical Skills", "Experience"]
    skills_sec = data.sections[0]
    assert len(skills_sec.simple_lines) == 4
    assert any("LLM & Generative AI" in ln for ln in skills_sec.simple_lines)
    assert any("Cloud & MLOps" in ln for ln in skills_sec.simple_lines)


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_universal_project_links_for_arbitrary_candidates(tmp_path):
    r"""Verify that any candidate with project links (on title line or in bullets)
    has their links formatted as clickable LaTeX \href items beside the title,
    never as bullets, and without any hardcoded names."""
    candidate_md = """# Alice Smith
alice@example.com | github.com/alicesmith | linkedin.com/in/alicesmith

## Projects
**Distributed Stream Processor** [Live Demo](https://stream.alicesmith.dev) [GitHub](https://github.com/alicesmith/stream-processor) | Go, Kafka, Docker | 2025
- Architected high-throughput event processing engine handling 100K events/sec
- Implemented fault tolerance with raft consensus

**Neural Code Search** | Python, Transformers | 2024
- [Demo](https://huggingface.co/spaces/alicesmith/code-search)
- [GitHub](https://github.com/alicesmith/neural-search)
- Built vector search engine with bi-encoder architecture

**Static Site Generator** | Rust | 2023
- Designed fast markdown parser
"""
    data = parse_markdown_resume(candidate_md)
    tex = render_resume_tex(data)

    # 1. Links from project 1 must be rendered on the header line
    assert r"\href{https://stream.alicesmith.dev}{\underline{Live Demo}}" in tex
    assert r"\href{https://github.com/alicesmith/stream-processor}{\underline{GitHub}}" in tex

    # 2. Links from project 2 (which were bullets) must be moved to header line and NOT remain as bullets
    assert r"\href{https://huggingface.co/spaces/alicesmith/code-search}{\underline{Demo}}" in tex
    assert r"\href{https://github.com/alicesmith/neural-search}{\underline{GitHub}}" in tex
    assert r"\resumeItem{Demo}" not in tex
    assert r"\resumeItem{[Demo]" not in tex

    # 3. Project 2 should only have 1 bullet item (the achievement bullet)
    proj_sec = next(s for s in data.sections if "project" in s.title.lower())
    assert len(proj_sec.entries[1].items) == 1
    assert "Built vector search engine" in proj_sec.entries[1].items[0]

    # 4. Zero references to other candidates
    assert "Omar" not in tex


def test_no_fabricated_links_for_bare_link_labels():
    r"""ANTI-FABRICATION (audit P0): a project whose bullets or title mention
    link LABELS without URLs (e.g. bare '- GitHub', 'Live Demo', 'GitHub |
    Repo') must render zero \href commands. Guessed urls.github.com/<handle>/
    <title-slug> repos are dead 404s on a recruiter's screen - worse than an
    honest absence - and violate the truthful-output contract."""
    from agent.latex_render import _extract_project_links_and_title

    # Bare label bullets: no URL anywhere in the source
    title, links = _extract_project_links_and_title(
        raw_title="**Distributed Event Streamer** | Kafka | 2025",
        other_fields=["Kafka", "2025"],
        bullet_links=["GitHub", "Live Demo", "Repo | Code"],
    )
    assert links == [], f"fabricated links: {links}"

    # Even with a candidate contact handle available, the fallback must NOT
    # synthesize github.com/<handle>/<title-slug> (a repo that usually
    # does not exist - e.g. 'distributed-event-streamer').
    title, links = _extract_project_links_and_title(
        raw_title="**AI Resume Optimizer Agent**",
        other_fields=[],
        bullet_links=["GitHub"],
    )
    assert links == [], f"fabricated links: {links}"
    assert "github.com" not in " ".join(u for _, u in links)

    # Real links still extract (control)
    title, links = _extract_project_links_and_title(
        raw_title="**Real Project** [GitHub](https://github.com/real/real-project) | Python | 2025",
        other_fields=["Python", "2025"],
        bullet_links=["- Live Demo: https://real.dev/demo"],
    )
    assert ("GitHub", "https://github.com/real/real-project") in links
    assert ("Live Demo", "https://real.dev/demo") in links


def test_no_fabricated_links_in_full_render(tmp_path):
    r"""End-to-end anti-fabrication: a resume whose projects have NO links
    must compile with zero \href project links (contact-line links are the
    candidate's own and unrelated)."""
    if not HAVE_PDFLATEX:
        pytest.skip("pdflatex not installed")
    candidate_md = """# Bob Jones
bob@example.com | github.com/bobjones

## Projects
**Distributed Event Streamer** | Go, Kafka, Docker | 2025
- Architected high-throughput event processing engine handling 100K events/sec

**Neural Code Search** | Python, Transformers | 2024
- GitHub
- Live Demo
- Built vector search engine with bi-encoder architecture
"""
    data = parse_markdown_resume(candidate_md)
    tex = render_resume_tex(data)

    # Only the contact line may carry a href (github.com/bobjones);
    # project sections must carry none.
    body_start = tex.index("Projects")
    project_tex = tex[body_start:]
    assert r"\href" not in project_tex, project_tex
    assert "github.com/bobjones/distributed-event-streamer" not in tex


def test_compact_mode_injects_spacing_overrides_only_when_asked():
    """The COMPACT_BLOCK token must never leak into a compiled document:
    plain render removes it entirely; compact render swaps it for the
    spacing-only override block. Content-neutral by construction."""
    base_md = """# Jane Doe
jane@example.com | github.com/janedoe | linkedin.com/in/janedoe

## Skills
Python, Docker, Kubernetes
"""
    data = parse_markdown_resume(base_md)
    plain = render_resume_tex(data)
    compact = render_resume_tex(data, compact=True)

    assert "<!--COMPACT_BLOCK-->" not in plain
    assert "<!--COMPACT_BLOCK-->" not in compact
    assert "AUTO-COMPACT PASS" not in plain
    assert "AUTO-COMPACT PASS" in compact
    # Overrides touch spacing only - both documents carry the same body
    assert "Jane Doe" in compact and "Docker" in compact
    assert r"\renewcommand{\resumeSubheading}" in compact
    assert r"\renewcommand{\resumeSubheading}" not in plain


def _overflow_resume_markdown() -> str:
    """Dense fixture that reliably exceeds one page in plain spacing."""
    projects = []
    for i in range(1, 7):
        projects.append(
            f"**Project {i} - Data Processing Platform {i}** [GitHub](https://github.com/user/project-{i}) | Python, Kafka, Docker, AWS | Jan 2025 - Mar 2025\n"
            f"- Architected event-driven pipeline processing {i}0K events per second with sub-second latency\n"
            f"- Implemented fault-tolerant checkpointing reducing data loss from {i}% to 0 incidents\n"
            f"- Deployed to production via CI/CD pipeline with canary releases and monitoring dashboards\n\n")
    return (
        "# Jane Doe\n"
        "jane@example.com | github.com/janedoe | linkedin.com/in/janedoe\n\n"
        "## Experience\n"
        "**Senior Software Engineer** | Tech Corp | Jan 2024 - Present\n"
        "- Led migration of monolith services to domain-driven microservices serving 5M users\n"
        "- Cut p99 latency by 40 percent through cache redesign and connection pooling\n"
        "- Mentored 4 engineers through quarterly promotion-focused growth plans\n"
        "- Drove incident postmortem culture lowering repeat incidents from 8 to 1 per quarter\n\n"
        "**Software Engineer** | Startup Co | Jun 2021 - Dec 2023\n"
        "- Built multi-tenant billing system handling 50K daily transactions without errors\n"
        "- Automated release process shrinking deploy window from 3 hours to 12 minutes\n"
        "- Introduced contract testing preventing 15+ regressions across service boundaries\n"
        "- Owned on-call rotation reliability reducing pager noise by 60 percent\n\n"
        "## Skills\n"
        "- Python\n- Go\n- Java\n- Kafka\n- Docker\n- Kubernetes\n- AWS\n- GCP\n- Terraform\n"
        "- PostgreSQL\n- Redis\n- Elasticsearch\n- React\n- TypeScript\n- GraphQL\n- gRPC\n"
        "- CI/CD\n- GitOps\n- Helm\n- ArgoCD\n\n"
        + "".join(projects)
        + "## Education\n"
        "**M.S. Computer Science** | State University | 2019 - 2021\n"
        "- GPA 3.9/4.0; thesis on distributed consensus under partial network failure\n\n"
        "**B.S. Computer Science** | City College | 2015 - 2019\n"
        "- Graduated summa cum laude; teaching assistant for data structures\n\n"
        "## Certifications\n"
        "- AWS Certified Solutions Architect - Professional (2024)\n"
        "- HashiCorp Certified Terraform Associate (2024)\n"
        "- CKAD Certified Kubernetes Application Developer (2023)\n"
        "- Google Professional Cloud Architect (2023)\n"
    )


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_pdf_page_count_reads_compiled_pdf(tmp_path):
    base_md = ("# Jane Doe\njane@example.com | github.com/janedoe\n\n"
               "## Skills\n- Python\n- Docker\n- Kubernetes\n")
    tex = render_resume_tex(parse_markdown_resume(base_md))
    pdf = latex_to_pdf(tex, tmp_path, "single.pdf")
    assert pdf and pdf_page_count(pdf) == 1

    # Deterministic 2-pager: real content after an explicit \newpage
    two_page_tex = latex_to_pdf(
        tex.replace(
            r"\end{document}",
            "\\newpage\n\\section{Overflow Section}\n- This line lives on page two\n\\end{document}"),
        tmp_path, "double.pdf")
    assert two_page_tex and pdf_page_count(two_page_tex) == 2


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_compact_render_pulls_overflow_back_to_one_page(tmp_path):
    """Enforced 1-page containment (audit): the exporter must not ship an
    overflowing resume silently - the spacing-only compact pass reclaims
    moderate overflow. Content of both renders is identical."""
    md = _overflow_resume_markdown()
    data = parse_markdown_resume(md)

    base_pdf = latex_to_pdf(render_resume_tex(data), tmp_path, "base.pdf")
    base_pages = pdf_page_count(base_pdf) if base_pdf else None
    assert base_pages and base_pages >= 2, "overflow fixture must exceed one page in plain spacing"

    compact_pdf = latex_to_pdf(render_resume_tex(data, compact=True), tmp_path, "compact.pdf")
    assert compact_pdf and pdf_page_count(compact_pdf) == 1


def test_contact_block_does_not_swallow_custom_section_headings():
    """Regression: the contact header collection loop absorbed custom short
    markdown headings entirely ('## Tech Stack'), deleting that section and
    turning its content into contact details. Any markdown heading other
    than a contact-family heading now ends the header block."""
    md = ("# Jane Doe\n"
          "jane@example.com | github.com/janedoe\n\n"
          "## Tech Stack\n"
          "Go, Redis, Postgres\n")
    data = parse_markdown_resume(md)
    titles = [s.title.lower() for s in data.sections]
    assert any("stack" in t for t in titles), titles
    assert data.contact.name == "Jane Doe"
    assert "jane@example.com" in data.contact.details
    assert "Tech Stack" not in data.contact.details
    # Contact-family headings still feed the details line
    md2 = ("# Jane Doe\n"
           "## Contact\n"
           "jane@example.com | +1 123 456\n\n"
           "## Skills\n"
           "- Python\n")
    data2 = parse_markdown_resume(md2)
    assert data2.contact.name == "Jane Doe"
    assert "jane@example.com" in data2.contact.details and "+1 123 456" in data2.contact.details
    assert any("skill" in s.title.lower() for s in data2.sections)


def test_skills_section_comma_lines_stay_simple_lines():
    """Plain comma-list lines inside Skills-family sections must be simple
    lines, not comma-split header/right subheading columns. Regression:
    '## Skills\nPython, Docker, Kubernetes' parsed as an entry with
    header 'Python' and right 'Docker', lopping the tail of the list."""
    md = """# Jane Doe
jane@example.com | github.com/janedoe

## Skills
Python, Docker, Kubernetes, Kafka, Terraform
"""
    data = parse_markdown_resume(md)
    skills_sec = next(s for s in data.sections if "skill" in s.title.lower())
    assert skills_sec.simple_lines == ["Python, Docker, Kubernetes, Kafka, Terraform"]
    assert skills_sec.entries == []

    tex = render_resume_tex(data)
    assert "Kubernetes, Kafka, Terraform" in tex
    assert r"resumeProjectHeading{\textbf{Python}}{Docker" not in tex

    # Tech-stack style titles equally protected
    data2 = parse_markdown_resume("# Jane Doe\njane@example.com | github.com/janedoe\n\n## Tech Stack\nGo, Redis, Postgres\n")
    stack = next(s for s in data2.sections if "stack" in s.title.lower())
    assert stack.simple_lines == ["Go, Redis, Postgres"]


@pytest.mark.skipif(not HAVE_PDFLATEX, reason="pdflatex not installed")
def test_latex_arrow_encoding_never_renders_inverted_question_mark(tmp_path):
    """Arrows like '21 -> 0' or 'A --> B' must compile cleanly and never render as '¿'."""
    from pypdf import PdfReader
    md = """# Bob Developer
bob@example.com

## Experience
**Systems Engineer** | Cloud Inc | 2023 - Present
- Reduced latency from 21 -> 0 ms using in-memory cache (A -> B)
- Migrated legacy services (v1 --> v2) with zero downtime
"""
    data = parse_markdown_resume(md)
    tex = render_resume_tex(data)
    pdf = latex_to_pdf(tex, tmp_path, "arrow_test.pdf")
    assert pdf is not None
    reader = PdfReader(pdf)
    text = reader.pages[0].extract_text()
    assert "¿" not in text
    assert "21" in text



