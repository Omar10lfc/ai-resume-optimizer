"""Relevance-weighted trim suggestions: deterministic ranking, no edits."""
import copy

from agent import trim
from agent.nodes import pdf_exporter as _exporter
from agent.trim import suggest_trims
from server import runners as _runners


TRIM_RESUME = """# Jake Ryan
jake@su.edu

## Summary
AI engineer who builds RAG pipelines and ships production APIs.

## Skills
**Languages**: Python, Java, Go
**Frameworks**: FastAPI, React

## Experience
**Software Engineer** | Acme | 2020 - 2024
- Built a FastAPI service used by 10K users, cutting latency by 40%.
- Managed office supplies and scheduled meetings.
- Shipped a React dashboard for the operations team.

## Projects
**RAG Assistant** | Python | 2024 - 2025
- Built a retrieval pipeline in Python that improved answer accuracy by 12%.

## Education
**State University** | 2016 - 2020
- Coursework: Data Structures, Algorithms, Databases
"""

JOB = "We need Python, FastAPI, and RAG experience. Kubernetes a plus."


def _resume_with_bullets(*bullets: str) -> str:
    body = "\n".join(f"- {bullet}" for bullet in bullets)
    return (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Experience\n"
        "**Engineer** | Acme | 2020 - 2024\n"
        f"{body}\n"
    )


def _excerpts(rows):
    return [row["excerpt"] for row in rows]


# ==========================================
# Contract
# ==========================================

def test_no_overflow_returns_nothing():
    assert suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=False) == []


def test_empty_resume_returns_nothing():
    assert suggest_trims("", job_text=JOB, overflowed=True) == []
    assert suggest_trims(None, job_text=JOB, overflowed=True) == []


def test_returns_at_most_five_rows_with_the_documented_shape():
    bullets = [f"Built component number {i} for the platform." for i in range(9)]
    rows = suggest_trims(_resume_with_bullets(*bullets), overflowed=True)

    assert len(rows) == 5
    assert all(set(row) == {"section", "excerpt", "reason"} for row in rows)
    assert all(row["reason"] for row in rows)


def test_education_coursework_is_never_suggested():
    rows = suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=True)

    assert all("Coursework" not in row["excerpt"] for row in rows)


# ==========================================
# Ranking signals
# ==========================================

def test_low_relevance_bullet_ranks_before_relevant_evidence():
    rows = suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=True,
                         max_suggestions=10)
    excerpts = _excerpts(rows)

    office = next(i for i, e in enumerate(excerpts) if "office supplies" in e)
    fastapi = next(i for i, e in enumerate(excerpts) if "FastAPI service" in e)
    assert office < fastapi


def test_duplicated_skills_row_is_the_cheapest_cut():
    rows = suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=True)

    top = rows[0]
    assert top["section"] == "Skills"
    assert "repeats terms" in top["reason"].lower()


def test_unique_skills_row_is_not_suggested():
    resume = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Skills\n"
        "**Languages**: Python, Rust, Elixir\n\n"
        "## Experience\n"
        "**Engineer** | Acme | 2020 - 2024\n"
        "- Built a billing service for internal teams.\n"
        "- Managed office supplies and scheduled meetings.\n"
    )

    rows = suggest_trims(resume, overflowed=True)

    assert all(row["section"] != "Skills" for row in rows)


def test_specific_summary_is_not_suggested():
    resume = (
        "# Jake Ryan\n"
        "jake@su.edu\n\n"
        "## Summary\n"
        "AI engineer shipping Python RAG pipelines.\n\n"
        "## Experience\n"
        "**Engineer** | Acme | 2020 - 2024\n"
        "- Managed office supplies and scheduled meetings.\n"
    )

    rows = suggest_trims(resume, job_text="Python RAG experience required.",
                         overflowed=True)

    assert all(row["section"] != "Summary" for row in rows)


def test_cover_letter_echo_makes_a_bullet_more_expensive():
    resume = _resume_with_bullets(
        "Built a Python analytics pipeline for finance.",
        "Built a Python billing service for healthcare.",
    )
    cover = "I built the analytics pipeline for finance."

    rows = suggest_trims(resume, cover_text=cover, overflowed=True)
    excerpts = _excerpts(rows)

    assert excerpts.index("Built a Python billing service for healthcare.") < \
        excerpts.index("Built a Python analytics pipeline for finance.")


def test_covered_requirement_evidence_protects_a_bullet():
    resume = _resume_with_bullets(
        "Built a FastAPI service for internal teams.",
        "Built a billing service for internal teams.",
    )

    rows = suggest_trims(
        resume, overflowed=True,
        requirement_matches=[{
            "requirement": "FastAPI experience",
            "status": "covered",
            "evidence": ["FastAPI"],
            "missing": [],
            "reason": "",
        }])
    excerpts = _excerpts(rows)

    assert excerpts.index("Built a billing service for internal teams.") < \
        excerpts.index("Built a FastAPI service for internal teams.")


def test_precomputed_impact_grade_is_used():
    resume = _resume_with_bullets(
        "Built a queue worker that cut latency by 40%.",
        "Coordinated weekly planning meetings across teams.",
    )
    # Deliberately contradict the local heuristic: the precomputed channel is
    # authoritative, so the metric bullet becomes the cheapest cut.
    audit = [
        {"bullet": "Built a queue worker that cut latency by 40%.",
         "section": "Experience", "grade": "low"},
        {"bullet": "Coordinated weekly planning meetings across teams.",
         "section": "Experience", "grade": "high"},
    ]

    rows = suggest_trims(resume, overflowed=True, bullet_impact=audit)

    # Without the audit the metric bullet would be the expensive one; the
    # provided grade flips the order, proving the channel is authoritative.
    assert rows[0]["excerpt"] == \
        "Built a queue worker that cut latency by 40%."


# ==========================================
# Determinism + immutability
# ==========================================

def test_identical_inputs_produce_identical_suggestions():
    first = suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=True)
    second = suggest_trims(TRIM_RESUME, job_text=JOB, overflowed=True)

    assert first == second


def test_inputs_are_not_mutated():
    requirements = [{
        "requirement": "Python", "status": "covered",
        "evidence": ["Python"], "missing": [], "reason": "",
    }]
    audit = [{"bullet": "Built things.", "section": "Experience",
              "grade": "medium"}]
    requirements_before = copy.deepcopy(requirements)
    audit_before = copy.deepcopy(audit)

    suggest_trims(TRIM_RESUME, job_text=JOB, cover_text="Dear team,",
                  requirement_matches=requirements, bullet_impact=audit,
                  overflowed=True)

    assert requirements == requirements_before
    assert audit == audit_before


# ==========================================
# Wiring
# ==========================================

def test_pdf_exporter_node_ranks_trims_only_on_overflow(monkeypatch, tmp_path):
    captured = {}

    def fake_export(resume_text, cover_text, request_dir, theme="", job_text=""):
        return {"resume_pdf_path": "", "cover_letter_pdf_path": "",
                "resume_docx_path": "", "resume_tex_path": "",
                "resume_html_path": "", "resume_pages": 2,
                "resume_compacted": True,
                "page_overflow_warning": "Optimized resume is 2 pages.",
                "render_engine": "latex@1", "pdf_checks": []}

    def fake_suggest(resume_text, **kwargs):
        captured["resume"] = resume_text
        captured.update(kwargs)
        return [{"section": "Experience", "excerpt": "Line.",
                 "reason": "Reason."}]

    monkeypatch.setattr(_exporter, "export_documents", fake_export)
    monkeypatch.setattr(_exporter, "suggest_trims", fake_suggest)

    out = _exporter.pdf_exporter_node({
        "optimized_resume": "## Experience\n- Line.",
        "cover_letter": "Dear team,",
        "job_text": JOB,
        "output_dir": str(tmp_path),
        "requirement_matches": [{"requirement": "Python"}],
        "bullet_impact": [{"bullet": "Line.", "grade": "high"}],
    })

    assert out["trim_suggestions"] == [
        {"section": "Experience", "excerpt": "Line.", "reason": "Reason."}]
    assert captured["overflowed"] is True
    assert captured["job_text"] == JOB
    assert captured["requirement_matches"] == [{"requirement": "Python"}]
    assert captured["bullet_impact"] == [{"bullet": "Line.", "grade": "high"}]


def test_pdf_exporter_node_skips_trims_without_overflow(monkeypatch, tmp_path):
    captured = {}

    def fake_export(resume_text, cover_text, request_dir, theme="", job_text=""):
        return {"resume_pdf_path": "", "cover_letter_pdf_path": "",
                "resume_docx_path": "", "resume_tex_path": "",
                "resume_html_path": "", "resume_pages": 1,
                "resume_compacted": False, "page_overflow_warning": "",
                "render_engine": "latex@1", "pdf_checks": []}

    def fake_suggest(resume_text, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(_exporter, "export_documents", fake_export)
    monkeypatch.setattr(_exporter, "suggest_trims", fake_suggest)

    out = _exporter.pdf_exporter_node({
        "optimized_resume": "## Experience\n- Line.",
        "cover_letter": "Dear team,",
        "output_dir": str(tmp_path),
    })

    assert out["trim_suggestions"] == []
    assert captured["overflowed"] is False


def test_result_payload_carries_trim_suggestions():
    payload = _runners._extract_results({
        "trim_suggestions": [{"section": "Skills", "excerpt": "Go",
                              "reason": "Duplicated."}],
    })

    assert payload["trim_suggestions"] == [
        {"section": "Skills", "excerpt": "Go", "reason": "Duplicated."}]
