"""BB2 (Nov 14 2026): the HTML/CSS renderer printed by headless Chromium.

Layers:
- PURE-STRING structure tests (always run, no browser): section/entry
  rendering for every section class the LaTeX path renders, XSS escaping,
  the strict network isolation (no external fetchable URL by construction
  + the CSP meta), the contact-link classification, the per-kind semantic
  corrections ported from the LaTeX renderer (project date/tech/links +
  glue defense, experience role/company swap, education rows), the
  density CSS ladder, and the anti-fabrication link rule.
- MEASURED-LOOP logic tests (fake measurer, always run): the density-tier
  ladder's contract mirrors the LaTeX tier pass - winner copied onto the
  shipped filename, losing probes unlinked, the bottom-stop gate on tier
  1, None when no tier fits one page.
- ENGINE-SELECTION tests (faked engines, always run): export_documents
  routes by config.RESUME_RENDERER (auto/html/latex) with graceful
  fallbacks; the render_engine stamp rides the return dict in every case.
- SCRIPTED ACCEPTANCE (skipif no browser/pdfplumber): the SAME fixture
  family the LaTeX acceptance test uses, pdfplumber-measured - one page,
  the name/heading/boundary floors, the bottom stop, ATS-safe text
  extraction (keywords, '->' arrows, scheme-stripped link text), and the
  density ladder reclaiming a long fixture to one page.
"""
import re

import pytest

import agent
from agent import config
from agent.html_render import (
    HTML_RENDERER_VERSION,
    _DENSITY_VARS,
    _entry_head_rows,
    _inline,
    _simple_rows,
    html_to_pdf,
    render_density_tiers,
    render_resume_html,
)
from agent.latex_render import HAVE_PDFLATEX, parse_markdown_resume
from agent.nodes import pdf_exporter as _exporter

try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except ImportError:
    HAVE_PDFPLUMBER = False

from agent.html_render import BROWSER_PATH, HAVE_BROWSER  # noqa: E402


SPEC_B_MD = (
    "# Test Candidate\n"
    "tc@example.com | +1 555 010 0100 | linkedin.com/in/testcandidate | "
    "github.com/testcandidate\n\n"
    "## Professional Summary\n"
    "ML engineer with a strong track record building end-to-end LLM/RAG "
    "pipelines, AI agents and workflow automations.\n\n"
    "## Technical Skills\n"
    "**Machine Learning & Deep Learning**: Python, SQL, PyTorch, Hugging "
    "Face Transformers, Scikit-learn, Model Fine-Tuning, Neural Networks\n"
    "**LLM & Generative AI**: RAG, Multi-Agent Systems, LLM Orchestration, "
    "Prompt Engineering\n\n"
    "## Experience\n"
    "**Computer Vision Engineer** | Tech Corp | Jan 2024 - Present\n"
    "- Built and evaluated computer-vision models across medical and "
    "automotive datasets -> shipped to production on edge devices\n\n"
    "**Data Science Intern** | Data Inc | Apr 2023 - Oct 2023\n"
    "- Led development of a clinical diagnosis system -> validated on a "
    "2,195-instance held-out set with radiologists\n\n"
    "## Projects\n"
    "**Aegis: Heat-Risk Operations Agent** "
    "[Live](https://aegis.example.com) | FastAPI, LangGraph | "
    "Aug 2025 - Sep 2026\n"
    "- Engineered a FastAPI service around a 4-node LangGraph agent with "
    "bounded LLM timeouts\n\n"
    "**Port Digital Twin: AIS-Fed Optimization** "
    "[Code](https://port.example.com) | OR-Tools CP-SAT | "
    "Sep 2024 - Jun 2025\n"
    "- Engineered a CP-SAT berth-allocation solver cutting average "
    "vessel wait time by 47% vs the FCFS baseline\n\n"
    "**Probe Header-Only Project | Jan 2026 - Feb 2026**\n\n"
    "## Education\n"
    "**Test University of Science and Technology** | Testville\n"
    "B.Sc. in Computer Science, AI & Data Science | Oct 2022 - Jun 2026\n\n"
    "## Certifications\n"
    "**AWS Future AI Engineer**: Udacity (Nov 2025) - Certificate\n"
    "**Natural Language Processing in Python**: Maven Analytics "
    "(Oct 2025) - Certificate\n"
)


# ---------------------------------------------------------------------------
# Structure (pure string work - no browser needed)
# ---------------------------------------------------------------------------


def _specb_html(density: int = 0) -> str:
    return render_resume_html(parse_markdown_resume(SPEC_B_MD),
                              density=density)


def test_render_resume_html_sections_and_entries():
    page = _specb_html()
    assert page.count("<section>") == 6  # summary, skills, exp, projects, edu, certs
    assert page.count('class="eh"') == 7  # 2 exp + 3 projects + 2 edu rows
    assert page.count("<li>") == 4  # bullets
    assert page.count('class="row"') == 5  # summary prose + 2 skills + 2 certs
    assert "Professional Summary" in page  # uppercased by CSS text-transform
    assert "Test Candidate" in page


def test_render_resume_html_carries_strict_csp_and_no_external_fetchables():
    """Network isolation by construction: system font stacks only, no
    <img>/<link>/src/@import, and the CSP meta fails any leak closed."""
    page = _specb_html()
    assert "Content-Security-Policy" in page and "default-src 'none'" in page
    assert "<script" not in page and "<img" not in page
    assert 'src=' not in page and "@import" not in page and "<link" not in page
    # the ONLY allowed fetch style is inline (<style> / style="")
    assert "<style>" in page


def test_html_escapes_untrusted_content_and_rejects_evil_hrefs():
    md = (
        "# Jane <script>alert(1)</script>Roe\n"
        "a@b.co | <a hrefx=x>[Click](javascript:alert(1))</a>\n\n"
        "## Summary\n"
        "Built <b>things</b> & \"quotes\" [Fake](javascript:alert(2)) end\n"
    )
    page = render_resume_html(parse_markdown_resume(md))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "&amp;" in page and "&quot;" in page
    # javascript: targets never become hrefs - the label renders as text
    assert 'href="javascript:' not in page


def test_inline_links_and_bare_urls():
    # markdown link: full URL in href, scheme-stripped display text
    got = _inline("see [Live](https://aegis.example.com/x) now")
    assert '<a href="https://aegis.example.com/x">Live</a>' in got
    # bare URL: href full, display scheme-stripped
    got = _inline("at https://github.com/jane/x projects")
    assert '<a href="https://github.com/jane/x">github.com/jane/x</a>' in got
    # mailto-schemeless email (the CONTACT renderer classifies bare emails;
    # inline body text keeps them plain - LaTeX parity)
    got = _inline("mail jane@doe.dev thanks")
    assert "jane@doe.dev" in got and "mailto" not in got
    # bold/italic/code markers stripped, ASCII arrows kept (ATS extraction)
    got = _inline("**Bold** *em* `code` a -> b")
    assert "<b>" not in got and "**" not in got and "`" not in got
    assert "a -&gt; b" in got


def test_contact_link_classification():
    page = _specb_html()
    # email -> mailto; linkedin/github bare domains -> https hrefs with
    # scheme-stripped display text
    assert '<a href="mailto:tc@example.com">tc@example.com</a>' in page
    assert ('<a href="https://linkedin.com/in/testcandidate">'
            "linkedin.com/in/testcandidate</a>") in page
    assert ('<a href="https://github.com/testcandidate">'
            "github.com/testcandidate</a>") in page
    assert "+1 555 010 0100" in page  # plain text (phone)


def test_project_row_semantics():
    """The LaTeX port: date/tech-stack disambiguation, links extracted from
    title/fields/bullets (anti-fabrication: only REAL links render), and
    the pipe-glue defense."""
    data = parse_markdown_resume(SPEC_B_MD)
    projects = next(s for s in data.sections
                    if "project" in s.title.lower())
    rows = _entry_head_rows(projects.entries[0], "projects")
    left, date = rows[0]
    assert "Aegis: Heat-Risk Operations Agent" in left
    assert '<a href="https://aegis.example.com">Live</a>' in left
    assert "FastAPI, LangGraph" in left
    assert date == "Aug 2025 - Sep 2026"
    # header-only project: date parsed from the glued title field
    rows3 = _entry_head_rows(projects.entries[2], "projects")
    left3, date3 = rows3[0]
    assert "Probe Header-Only Project" in left3
    assert date3 == "Jan 2026 - Feb 2026"
    # anti-fabrication: a bare 'GitHub' label with no URL renders no anchor
    md = ("# J\n\n## Projects\n**Solo Tool** Live Demo | GitHub | Jan 2026\n"
          "- Used Python\n")
    page = render_resume_html(parse_markdown_resume(md))
    assert "Solo Tool" in page and "<a" not in page.split("Projects")[1]


def test_project_glue_defense():
    """Run 28e2e56f: a date column carrying '| more fields' once printed
    right-aligned and ran off the sheet - the date splits at the pipe."""
    entry = parse_markdown_resume(
        "# J\n\n## Projects\n**Tool** Jan 2026 - Feb 2026 | Docker\n- x\n"
    ).sections[0].entries[0]
    rows = _entry_head_rows(entry, "projects")
    left, date = rows[0]
    assert date == "Jan 2026 - Feb 2026"
    assert "Docker" in left  # the remainder became the tech stack


def test_experience_role_company_swap():
    """The role keyword decides which line is the role (ported from the
    LaTeX renderer's experience branch). Date cell parity with the LaTeX
    path is parser-owned and pinned by the spec-B acceptance test."""
    entry = parse_markdown_resume(
        "# J\n\n## Experience\n**Acme Corp** | New York\n"
        "Software Engineer | Mar 2024 - Present\n- Did things\n"
    ).sections[0].entries[0]
    rows = _entry_head_rows(entry, "experience")
    left, date = rows[0]
    assert "<b>Software Engineer</b>" in left
    assert "<em>Acme Corp</em>" in left
    assert "(New York)" in left
    assert date == entry.sub_right


def test_education_and_generic_rows():
    data = parse_markdown_resume(SPEC_B_MD)
    edu = next(s for s in data.sections if "education" in s.title.lower())
    rows = _entry_head_rows(edu.entries[0], "education")
    assert rows[0] == ("<b>Test University of Science and Technology</b>",
                       "Testville")
    assert rows[1] == ("B.Sc. in Computer Science, AI &amp; Data Science", "")
    # generic section kinds: single row when no sub fields, two rows with
    # them (hand-built entries - the generic classification of the SHARED
    # parser is the LaTeX path's contract, already pinned there)
    from agent.latex_render import Entry
    rows = _entry_head_rows(Entry(header="Best Paper", right="2025"),
                            "awards")
    assert rows == [("<b>Best Paper</b>", "2025")]
    rows = _entry_head_rows(
        Entry(header="X", right="R", sub="S", sub_right="SR"), "awards")
    assert rows == [("<b>X</b>", "R"), ("S", "SR")]


def test_simple_rows_label_semantics():
    rows = _simple_rows([
        "**ML & DL**: PyTorch, SQL",
        "Plain skill line without a label",
        "**Cert**: Udacity (Nov 2025) - Certificate",
    ])
    assert rows[0][0] is True
    assert '<span class="lbl">ML &amp; DL:</span>' in rows[0][1]
    assert rows[1][0] is False
    assert rows[1][1] == "Plain skill line without a label"
    assert rows[2][0] is True


def test_density_css_ladder():
    """Tiers tighten monotonically; unknown tier falls back to base."""
    base = _DENSITY_VARS[0]
    for tier in (1, 2, 3):
        v = _DENSITY_VARS[tier]
        assert float(v["fs"].rstrip("pt")) < float(base["fs"].rstrip("pt"))
        assert float(v["secgap"].rstrip("pt")) < float(
            base["secgap"].rstrip("pt"))
    assert "--fs:10pt" in render_resume_html(
        parse_markdown_resume(SPEC_B_MD), density=99)


def test_empty_section_dropped_and_unicode_preserved():
    md = ("# José Muñoz\nj@x.co\n\n## Empty Section\n\n"
          "## Summary\nCut latency by 47% — shipped ✓\n")
    page = render_resume_html(parse_markdown_resume(md))
    assert "Empty Section" not in page
    assert "José Muñoz" in page and "— shipped ✓" in page  # no ASCII folding


# ---------------------------------------------------------------------------
# The measured loop (fake measurer - the tier ladder's contract)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_measurer(monkeypatch, tmp_path):
    """Scripts html_to_pdf + the pdfplumber readers so the ladder's
    bookkeeping runs without a browser. `script` maps density ->
    (page_count, last_baseline); base density 0 = the caller's render (the
    ladder never probes it)."""
    state = {"script": {1: (1, 700.0), 2: (2, 700.0), 3: (2, 700.0)},
             "probes": []}

    def fake_print(page_html, output_dir, filename="Resume.pdf", timeout=90):
        density = 0
        for d in _DENSITY_VARS:
            if f"--fs:{_DENSITY_VARS[d]['fs']};" in page_html:
                density = d
                break
        state["probes"].append((density, filename))
        out = tmp_path / filename
        out.write_bytes(b"%PDF-fake")  # content irrelevant (readers faked)
        return str(out)

    def fake_pages(pdf_path):
        m = re.search(r"__density_tier(\d+)\.pdf$", str(pdf_path))
        if not m:
            return 2
        pages, _ = state["script"].get(int(m.group(1)), (2, 700.0))
        return pages

    def fake_baseline(pdf_path):
        m = re.search(r"__density_tier(\d+)\.pdf$", str(pdf_path))
        if not m:
            return 700.0
        _, baseline = state["script"].get(int(m.group(1)), (2, 700.0))
        return baseline

    monkeypatch.setattr("agent.html_render.html_to_pdf", fake_print)
    monkeypatch.setattr("agent.html_render.pdf_page_count", fake_pages)
    monkeypatch.setattr("agent.html_render.pdf_last_baseline", fake_baseline)
    return state


def test_density_tiers_adopt_first_fitting_tier(fake_measurer, tmp_path):
    fake_measurer["script"] = {1: (1, 700.0), 2: (2, 700.0), 3: (2, 700.0)}
    got = render_density_tiers(
        parse_markdown_resume(SPEC_B_MD), tmp_path, "Optimized_Resume.pdf")
    assert got and got["pages"] == 1 and got["density"] == 1
    assert got["pdf_path"].endswith("Optimized_Resume.pdf")
    # winner copied onto the shipped name; probes unlinked
    assert (tmp_path / "Optimized_Resume.pdf").exists()
    for probe in ("__density_tier1.pdf", "__density_tier2.pdf",
                  "__density_tier3.pdf"):
        assert not (tmp_path / probe).exists(), probe


def test_density_tiers_gate_tier1_on_bottom_stop(fake_measurer, tmp_path):
    """Tier 1 crossing the stop is NOT adopted when tier 2 fits inside it
    (the LaTeX airy-tier gate); a crossing tier-1 is kept only as a
    last-resort fallback."""
    fake_measurer["script"] = {1: (1, 750.0), 2: (1, 700.0), 3: (2, 700.0)}
    got = render_density_tiers(
        parse_markdown_resume(SPEC_B_MD), tmp_path, "out.pdf")
    assert got and got["density"] == 2, "tier 1 past the stop must fall to tier 2"


def test_density_tiers_last_resort_keeps_crossing_tier1(fake_measurer,
                                                       tmp_path):
    """No tier after tier 1 fits: the one-page tier 1 that crossed the stop
    still beats a two-page resume (the LaTeX fallback contract)."""
    fake_measurer["script"] = {1: (1, 750.0), 2: (2, 700.0), 3: (2, 700.0)}
    got = render_density_tiers(
        parse_markdown_resume(SPEC_B_MD), tmp_path, "out.pdf")
    assert got and got["density"] == 1


def test_density_tiers_none_when_nothing_fits(fake_measurer, tmp_path):
    fake_measurer["script"] = {1: (2, 700.0), 2: (2, 700.0), 3: (2, 700.0)}
    got = render_density_tiers(
        parse_markdown_resume(SPEC_B_MD), tmp_path, "out.pdf")
    assert got is None
    for probe in ("__density_tier1.pdf", "__density_tier2.pdf",
                  "__density_tier3.pdf"):
        assert not (tmp_path / probe).exists(), probe


# ---------------------------------------------------------------------------
# Engine selection (faked engines - the routing contract)
# ---------------------------------------------------------------------------


@pytest.fixture()
def latex_engine_fakes(monkeypatch, tmp_path):
    """Fakes the LaTeX compile/docx so export_documents' routing runs fast
    and hermetic (the .tex source write stays real - it is pure string
    work)."""
    def fake_pdf(tex, output_dir, filename="Resume.pdf"):
        out = tmp_path / filename
        out.write_bytes(b"%PDF-fake")
        return str(out)

    monkeypatch.setattr("agent.latex_render.latex_to_pdf", fake_pdf)
    monkeypatch.setattr("agent.latex_render.latex_to_docx",
                        lambda *a, **k: str(tmp_path / "x.docx"))
    monkeypatch.setattr("agent.latex_render.pdf_page_count", lambda p: 1)
    return tmp_path


def test_export_documents_latex_mode_stamps_classic_engine(
        monkeypatch, tmp_path, latex_engine_fakes):
    monkeypatch.setattr(config, "RESUME_RENDERER", "latex")
    got = _exporter.export_documents("# J\n\n## Summary\nDid things.\n",
                                     "Dear hiring manager,\n", tmp_path)
    assert got["render_engine"] == "latex@1"
    assert got["resume_pdf_path"]
    assert got["resume_pages"] == 1
    assert got["cover_letter_pdf_path"]  # the cover letter path unchanged


def test_export_documents_auto_prefers_html(monkeypatch, tmp_path,
                                            latex_engine_fakes):
    monkeypatch.setattr(config, "RESUME_RENDERER", "auto")
    monkeypatch.setattr(_exporter._html, "HAVE_BROWSER", True)
    monkeypatch.setattr(_exporter._html, "html_to_pdf",
                        lambda *a, **k: str(tmp_path / "Optimized_Resume.pdf"))
    (tmp_path / "Optimized_Resume.pdf").write_bytes(b"%PDF-fake")
    monkeypatch.setattr(_exporter._latex, "pdf_page_count", lambda p: 1)
    got = _exporter.export_documents("# J\n\n## Summary\nDid things.\n",
                                     "Dear hiring manager,\n", tmp_path)
    assert got["render_engine"] == HTML_RENDERER_VERSION
    assert got["resume_tex_path"], ".tex source still written (contract)"


def test_export_documents_auto_without_browser_falls_to_latex(
        monkeypatch, tmp_path, latex_engine_fakes):
    monkeypatch.setattr(config, "RESUME_RENDERER", "auto")
    monkeypatch.setattr(_exporter._html, "HAVE_BROWSER", False)
    got = _exporter.export_documents("# J\n\n## Summary\nDid things.\n",
                                     "Dear hiring manager,\n", tmp_path)
    assert got["render_engine"] == "latex@1"


def test_export_documents_html_failure_falls_to_latex(
        monkeypatch, tmp_path, latex_engine_fakes):
    monkeypatch.setattr(config, "RESUME_RENDERER", "html")
    monkeypatch.setattr(_exporter._html, "HAVE_BROWSER", True)
    monkeypatch.setattr(_exporter._html, "html_to_pdf",
                        lambda *a, **k: None)  # print failed
    got = _exporter.export_documents("# J\n\n## Summary\nDid things.\n",
                                     "Dear hiring manager,\n", tmp_path)
    assert got["render_engine"] == "latex@1"
    assert got["resume_pdf_path"]


def test_export_documents_xhtml2pdf_last_resort(monkeypatch, tmp_path,
                                                latex_engine_fakes):
    monkeypatch.setattr(config, "RESUME_RENDERER", "latex")
    monkeypatch.setattr(_exporter._latex, "latex_to_pdf",
                        lambda *a, **k: None)  # pdflatex gone
    monkeypatch.setattr(_exporter, "_html_to_pdf",
                        lambda *a, **k: str(tmp_path / "fallback.pdf"))
    got = _exporter.export_documents("# J\n\n## Summary\nDid things.\n",
                                     "Dear hiring manager,\n", tmp_path)
    assert got["render_engine"] == "xhtml2pdf@1"
    assert got["resume_pdf_path"].endswith("fallback.pdf")
    assert got["cover_letter_pdf_path"].endswith("fallback.pdf")


# ---------------------------------------------------------------------------
# Scripted acceptance (pdfplumber-measured; skips without a browser)
# ---------------------------------------------------------------------------


def _pdf_lines(pdf):
    """The LaTeX acceptance suite's measuring convention."""
    pages = []
    with pdfplumber.open(str(pdf)) as doc:
        for page in doc.pages:
            words = page.extract_words(extra_attrs=["size"],
                                       keep_blank_chars=False)
            lines = []
            for w in sorted(words, key=lambda x: (round(x["top"], 1),
                                                  x["x0"])):
                if not w["text"].strip():
                    continue
                for ln in lines:
                    if abs(ln["top"] - w["top"]) <= 2.0:
                        ln["text"] += " " + w["text"]
                        ln["x1"] = max(ln["x1"], w["x1"])
                        break
                else:
                    lines.append({"top": w["top"], "x0": w["x0"],
                                  "x1": w["x1"], "size": round(w["size"], 1),
                                  "text": w["text"]})
            for ln in lines:
                ln["baseline"] = ln["top"] + ln["size"]
            lines.sort(key=lambda l: (l["top"], l["x0"]))
            pages.append({"lines": lines, "words": words})
    return pages


@pytest.mark.skipif(not (HAVE_BROWSER and HAVE_PDFPLUMBER),
                    reason="no Chromium-family browser / pdfplumber")
def test_html_render_acceptance_spec_b(tmp_path):
    """The spec-B fixture, pdfplumber-measured on the real Chromium print:
    one page, the name/heading/boundary floors, the bottom stop, and
    ATS-safe extraction (keywords, '->' arrows, scheme-stripped links)."""
    data = parse_markdown_resume(SPEC_B_MD)
    pdf = html_to_pdf(render_resume_html(data), tmp_path, "specb.pdf")
    assert pdf and agent.latex_render.pdf_page_count(pdf) == 1
    led = _pdf_lines(pdf)[0]
    assert led["lines"][0]["top"] >= 14.0  # name floor
    assert led["lines"][-1]["baseline"] <= 743.2 + 1.0  # bottom stop
    HEADS = ("professional summary", "technical skills", "experience",
             "projects", "education", "certifications")
    for i, ln in enumerate(led["lines"]):
        t = ln["text"].strip().lower()
        if any(t.startswith(h.split()[0]) and h in t for h in HEADS):
            if i > 0:
                assert ln["top"] - led["lines"][i - 1]["baseline"] >= 9.0, t
    # entry-title boundary floors (>= 5pt above the wrapped-line rhythm)
    TITLES = ("Computer Vision Engineer", "Data Science Intern",
              "Aegis: Heat-Risk", "Port Digital Twin",
              "Probe Header-Only Project", "Test University")
    for i, ln in enumerate(led["lines"]):
        if any(ln["text"].startswith(t) for t in TITLES):
            if i > 0:
                assert ln["top"] - led["lines"][i - 1]["baseline"] >= 5.0, \
                    ln["text"]
    text = " ".join(l["text"] for l in led["lines"])
    for kw in ("FastAPI", "LangGraph", "OR-Tools", "PyTorch"):
        assert kw in text, kw  # ATS keywords survive extraction
    assert "->" in text  # ASCII arrows survive
    assert "https://" not in text  # displayed link text keeps no scheme


@pytest.mark.skipif(not (HAVE_BROWSER and HAVE_PDFPLUMBER),
                    reason="no Chromium-family browser / pdfplumber")
def test_html_render_density_ladder_reclaims_long_fixture(tmp_path):
    """The containment loop live: a 2+ page fixture renders, the ladder
    probes tighter tiers, and the shipped PDF is ONE page with the
    invariants held."""
    lines = ["# Long Candidate", "long@example.com", "",
             "## Technical Skills"]
    lines += [f"**Skill Area {i}**: Python, SQL, PyTorch, Docker, Kubernetes"
              for i in range(8)]
    lines.append("\n## Experience")
    for j in range(6):
        lines.append(f"**Senior Engineer {j}** | Corp {j} | Jan 2024 - Present")
        for k in range(6):
            lines.append(f"- Owned subsystem {k} of platform {j} -> shipped "
                         f"to production and cut p95 latency by {5 + j + k}% "
                         f"across {10 + j} services")
        lines.append("")
    md = "\n".join(lines)
    data = parse_markdown_resume(md)
    base_pdf = html_to_pdf(render_resume_html(data), tmp_path, "long_base.pdf")
    assert base_pdf and agent.latex_render.pdf_page_count(base_pdf) > 1, \
        "fixture must overflow at base density"

    got = render_density_tiers(data, tmp_path, "long_shipped.pdf")
    assert got and got["pages"] == 1
    led = _pdf_lines(got["pdf_path"])[0]
    assert len(led["lines"]) > 40
    if got["density"] == 1:
        # the airy-tight gate: inside the bottom stop
        assert led["lines"][-1]["baseline"] <= 743.2 + 1.0
    assert led["lines"][0]["top"] >= 14.0
