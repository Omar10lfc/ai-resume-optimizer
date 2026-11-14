"""Round 13 (roadmap 12): the deterministic Google XYZ bullet-impact audit.

Unit tests for the scorer (grades, quantification classes, verb signals,
section selection, row shape/caps) plus the ats_check wiring through the
best-of guard and the server payload (those live in test_resume_agent.py /
test_fake_llm_wire.py beside their requirement-table siblings).
"""
import agent
from agent.bullet_impact import (
    _MAX_BULLET_CHARS,
    _MAX_ROWS,
    compute_bullet_impact,
    grade_bullet,
)
from agent.latex_render import parse_markdown_resume


# ---------------------------------------------------------------------------
# grade_bullet: the 2x2 verdict matrix + reason contract
# ---------------------------------------------------------------------------

def test_high_bullet_strong_verb_and_metric():
    verdict = grade_bullet(
        "Engineered a CV pipeline cutting p95 latency by 47% across 12 services")
    assert verdict["grade"] == "high"
    assert verdict["verb"] == "engineered"
    assert verdict["metrics"] == ["47%", "12 services"]
    assert "Opens with 'Engineered'" in verdict["reason"]
    assert "47%" in verdict["reason"]


def test_medium_strong_verb_no_metric():
    verdict = grade_bullet("Led development of a clinical diagnosis system")
    assert verdict["grade"] == "medium"
    assert verdict["verb"] == "led"
    assert verdict["metrics"] == []
    assert "no measured outcome" in verdict["reason"]
    assert "(%, count, $, time)" in verdict["reason"]


def test_medium_metric_weak_opener_names_the_phrase():
    verdict = grade_bullet("Responsible for cutting deployment cost 40%")
    assert verdict["grade"] == "medium"
    assert verdict["metrics"] == ["40%"]
    assert verdict["reason"] == (
        "Measures the outcome (40%) but opens with the weak phrase "
        "'responsible' - lead with an action verb.")


def test_low_weak_opener_without_metric():
    verdict = grade_bullet("Responsible for maintaining the deployment process")
    assert verdict["grade"] == "low"
    assert verdict["metrics"] == []
    assert verdict["reason"] == (
        "Neither a strong opener ('responsible') nor a measured outcome - "
        "rewrite as [action verb] + [result].")


def test_medium_metric_neutral_opener_names_the_word():
    verdict = grade_bullet(
        "Processed 10K events per second with 150/150 benchmark passes")
    assert verdict["grade"] == "medium"
    assert "Processed" not in verdict["reason"]  # neutral verbs are not shamed
    assert "'processed' is not a strong action verb" in verdict["reason"]


def test_low_bullet_neither_signal():
    verdict = grade_bullet("Helped with the testing process")
    assert verdict["grade"] == "low"
    assert verdict["verb"] == "helped"
    assert "rewrite as [action verb] + [result]" in verdict["reason"]


def test_co_prefixed_verb_folds_to_the_root():
    verdict = grade_bullet("Co-led a team of 6 engineers")
    assert verdict["grade"] == "high"
    assert verdict["verb"] == "led"
    assert verdict["metrics"] == ["6 engineers"]


def test_bullet_without_action_opener():
    """A noun opener ('Pipeline latency...') is not shamed as weak, but it
    also earns no strong-verb credit - the reason names the word."""
    verdict = grade_bullet("Pipeline latency cut of 47% was achieved")
    assert verdict["grade"] == "medium"
    assert verdict["verb"] == "pipeline"
    assert "'pipeline' is not a strong action verb" in verdict["reason"]


# ---------------------------------------------------------------------------
# Quantification classes: what counts as a measured outcome (and what does not)
# ---------------------------------------------------------------------------

def test_quant_classes_dollar_scaled_and_counts():
    assert grade_bullet("Shipped to 2,195 users and saved $1.2M")["metrics"] \
        == ["$1.2M", "2,195 users"]
    assert grade_bullet("Processed 10K events per second")["metrics"] == ["10K"]
    assert grade_bullet("Reduced p95 latency by 380 ms")["metrics"] == ["380 ms"]
    assert grade_bullet("Cut ingestion effort by 55%")["metrics"] == ["55%"]
    assert grade_bullet("Grew throughput 3x")["metrics"] == ["3x"]
    # hyphenated count compounds count ("2,195-instance held-out set")
    assert grade_bullet("Led a 2,195-instance migration")["metrics"] \
        == ["2,195-instance"]


def test_non_metric_numbers_stay_exempt():
    """Years, versions, and model names are not measured outcomes - the same
    exemption the integrity scanner's vocabulary enforces."""
    assert grade_bullet("Migrated services from Python 3.11 to 3.12 in 2024") \
        ["metrics"] == []
    assert grade_bullet("Built a GPT-4 powered assistant")["metrics"] == []


def test_metrics_dedupe_and_cap_at_four():
    text = ("Cut cost 40% saving $50K, 30 users, 5 teams, 2 pipelines, "
            "8 services, 9 requests")
    verdict = grade_bullet(text)
    assert len(verdict["metrics"]) == 4


# ---------------------------------------------------------------------------
# compute_bullet_impact: extraction, section selection, row shape
# ---------------------------------------------------------------------------

RESUME = """# Test Candidate
tc@example.com

## Experience
**CV Engineer** | Tech Corp | Jan 2024 - Present
- Engineered a CV pipeline cutting p95 latency by 47% across 12 services
- Responsible for maintaining the deployment process
- [Live Demo](https://aegis.example.com) | [GitHub](https://github.com/x)

## Education
**Test University** | Testville
B.Sc. in Computer Science | Oct 2022 - Jun 2026
- Coursework: ML, DL, CV
"""


def test_rows_carry_section_grade_verb_metrics_reason():
    rows = compute_bullet_impact(RESUME)
    assert [r["section"] for r in rows] == ["Experience", "Experience"]
    assert [r["grade"] for r in rows] == ["high", "low"]
    assert rows[0]["verb"] == "engineered"
    assert rows[0]["metrics"] == ["47%", "12 services"]
    assert rows[1]["verb"] == "responsible"
    assert rows[1]["metrics"] == []
    assert all(set(r) == {"bullet", "section", "grade", "verb", "metrics",
                          "reason"} for r in rows)


def test_education_sections_skipped():
    rows = compute_bullet_impact(RESUME)
    assert not any("Coursework" in r["bullet"] for r in rows)


def test_link_only_bullets_never_audit():
    """The parser routes '- [Live Demo](url)' to entry.links; the audit must
    never see a content-free 'link bullet' row."""
    rows = compute_bullet_impact(RESUME)
    assert not any("Live Demo" in r["bullet"] for r in rows)


def test_projects_and_other_sections_included():
    md = ("# J\n\n## Projects\n**Tool** | FastAPI | 2026\n"
          "- Shipped a 3-service backend serving 1,000 requests\n")
    rows = compute_bullet_impact(md)
    assert rows and rows[0]["section"] == "Projects"
    assert rows[0]["grade"] == "high"


def test_row_text_truncated_at_the_cap():
    long_bullet = "- " + "Engineered a very long service description " * 10
    md = f"# J\n\n## Experience\n**E** | Corp | 2026\n{long_bullet}\n"
    rows = compute_bullet_impact(md)
    assert rows and len(rows[0]["bullet"]) <= _MAX_BULLET_CHARS


def test_row_cap_stops_extraction():
    bullets = "\n".join(f"- Built service {i} serving 5 users"
                        for i in range(_MAX_ROWS + 5))
    md = f"# J\n\n## Experience\n**E** | Corp | 2026\n{bullets}\n"
    rows = compute_bullet_impact(md)
    assert len(rows) == _MAX_ROWS


def test_empty_and_bullet_free_resumes_return_empty_list():
    assert compute_bullet_impact("") == []
    assert compute_bullet_impact(None) == []  # type: ignore[arg-type]
    # summary prose + skills rows are not bullets (simple lines)
    assert compute_bullet_impact(
        "# Jane\nj@x.co\n\n## Summary\nBuilt things.\n") == []


def test_inline_markdown_stripped_from_row_text():
    md = ("# J\n\n## Experience\n**E** | Corp | 2026\n"
          "- Built a **real-time** pipeline cutting cost by 45%\n")
    rows = compute_bullet_impact(md)
    assert "**" not in rows[0]["bullet"]
    assert "real-time" in rows[0]["bullet"]


def test_parser_is_the_single_extraction_source():
    """The audit consumes parse_markdown_resume - the same parser the LaTeX
    and HTML renderers use (zero extraction drift across the codebase)."""
    data = parse_markdown_resume(RESUME)
    experience = next(s for s in data.sections if s.title == "Experience")
    items = experience.entries[0].items
    rows = compute_bullet_impact(RESUME)
    assert len(rows) == len(items)
    assert rows[0]["bullet"].startswith(items[0][:10])


def test_scorer_exported_from_package():
    assert agent.compute_bullet_impact is compute_bullet_impact
    assert agent.grade_bullet is grade_bullet
