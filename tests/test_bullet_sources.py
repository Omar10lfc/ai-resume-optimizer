"""Round 16: per-bullet typed source rows (agent/bullet_sources.py).

Unit tests for the provenance table: the unchanged/reworded/new/dropped
statuses, the one-to-one strongest-first binding, the metric lineage with
the integrity scan's rounding tolerance (96.2% -> "96%" is KEPT, 380 ms ->
400 ms is a drop), the extraction contract (education + link-only skip),
and the row caps/truncation. The ats_check best-of-guard wiring and the
wire payload pin live in test_resume_agent.py / test_fake_llm_wire.py
beside their impact-table siblings.
"""
import agent
from agent.bullet_sources import (
    _MATCH_THRESHOLD,
    _MAX_BULLET_CHARS,
    _MAX_ROWS,
    compute_bullet_sources,
)


# ---------------------------------------------------------------------------
# Status classification: unchanged / reworded / new / dropped
# ---------------------------------------------------------------------------

def test_unchanged_verbatim_row():
    original = "## Experience\n**Engineer** | Corp | 2024\n- Built Python services on Kubernetes with Docker."
    rows = compute_bullet_sources(original, original)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "unchanged"
    assert row["bullet"] == "Built Python services on Kubernetes with Docker."
    assert row["source"] == row["bullet"]
    assert row["similarity"] == 1.0
    assert row["section"] == "Experience"
    assert row["dropped"] == []
    assert row["reason"] == "Kept verbatim from your original."


def test_reworded_row_keeps_metric():
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Built Python services on Kubernetes with Docker, cutting latency 40%.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Built Kubernetes services with Docker in Python, cutting latency 40%.")
    rows = compute_bullet_sources(original, optimized)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "reworded"
    assert row["source"] == "Built Python services on Kubernetes with Docker, cutting latency 40%."
    assert row["metrics"] == ["40%"]
    assert row["dropped"] == []
    assert "% text overlap" in row["reason"]
    assert "verify the meaning survived" in row["reason"]


def test_reworded_row_dropped_metric_names_it():
    """The headline case: your original said 96.2% in this bullet; the
    optimized resume dropped it - the row says so, naming the token."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Built the parser reaching 96.2% accuracy on the benchmark suite.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Built the parser on the benchmark suite.")
    rows = compute_bullet_sources(original, optimized)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "reworded"
    assert row["metrics"] == []
    assert row["dropped"] == ["96.2%"]
    assert "96.2% no longer appear in this bullet" in row["reason"]


def test_rounding_tolerance_keeps_metric():
    """The Nov 14 integrity closure applies verbatim: a source 96.2% whose
    optimized bullet says 96% is a ROUNDING, not a drop - the provenance
    table can never contradict the integrity scan on the same pair."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Built the pipeline with 96.2% accuracy.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Built the pipeline with 96% accuracy.")
    rows = compute_bullet_sources(original, optimized)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "reworded"
    assert row["dropped"] == []
    assert "no longer appear" not in row["reason"]


def test_drifted_metric_is_dropped():
    """The tolerance is one-directional: a source 380 ms whose optimized
    bullet says 400 ms is a DRIFT, and the source measurement is dropped."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Cut p95 latency to 380 ms.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Cut p95 latency to 400 ms.")
    rows = compute_bullet_sources(original, optimized)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "reworded"
    assert row["dropped"] == ["380 ms"]


def test_new_bullet_row():
    original = "## Experience\n**Engineer** | Corp | 2024\n- Built Python services on Kubernetes with Docker."
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Built Python services on Kubernetes with Docker.\n"
                 "- Volunteered teaching weekend coding workshops for students.")
    rows = compute_bullet_sources(original, optimized)
    assert [r["status"] for r in rows] == ["unchanged", "new"]
    new_row = rows[1]
    assert new_row["source"] is None
    assert new_row["similarity"] == 0.0
    assert new_row["dropped"] == []
    assert "New content" in new_row["reason"]
    assert "integrity scan" in new_row["reason"]


def test_dropped_source_bullet_row():
    """A source bullet the optimized resume no longer carries surfaces as a
    dropped row - metric-free bullets included (the whole bullet is the
    loss, not just its numbers)."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Built Python services on Kubernetes with Docker.\n"
                "- Organized the weekly team sync.")
    optimized = "## Experience\n**Engineer** | Corp | 2024\n- Built Python services on Kubernetes with Docker."
    rows = compute_bullet_sources(original, optimized)
    assert [r["status"] for r in rows] == ["unchanged", "dropped"]
    dropped = rows[1]
    assert dropped["bullet"] == ""
    assert dropped["source"] == "Organized the weekly team sync."
    assert dropped["section"] == "Experience"
    assert dropped["similarity"] == 0.0
    assert dropped["metrics"] == []
    assert dropped["dropped"] == []
    assert "verify the loss is intentional" in dropped["reason"]


# ---------------------------------------------------------------------------
# Matching: one-to-one, strongest-first, deterministic
# ---------------------------------------------------------------------------

def test_one_to_one_binding_source_never_explains_two_bullets():
    """An original bullet never binds twice: the verbatim copy wins the
    binding, the near-copy variant falls through to 'new'."""
    bullet = "- Led the migration of 12 services to Docker."
    original = f"## Experience\n**Engineer** | Corp | 2024\n{bullet}"
    optimized = (f"## Experience\n**Engineer** | Corp | 2024\n{bullet}\n"
                 "- Led the migration of 12 services onto Docker.")
    rows = compute_bullet_sources(original, optimized)
    assert [r["status"] for r in rows] == ["unchanged", "new"]


def test_strongest_pair_wins_regardless_of_document_order():
    """Greedy binding is strongest-first: a weakly-matching earlier bullet
    cannot steal the source from the exact match that comes later."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Built Python services on Kubernetes with Docker.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Built Python services on Kubernetes.\n"
                 "- Built Python services on Kubernetes with Docker.")
    rows = compute_bullet_sources(original, optimized)
    assert rows[0]["status"] == "new"      # the condensed variant, unbound
    assert rows[1]["status"] == "unchanged"  # the verbatim copy holds the binding


def test_order_contract_matched_new_then_dropped():
    """Rows describe the shipped resume first (optimized document order,
    bound + new interleaved), then fully-dropped source bullets in original
    document order."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Prow first source bullet.\n"
                "- Q source bullet kept as reworded copy.\n"
                "- R third source bullet.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Q source bullet kept as a reworded copy.\n"
                 "- A genuinely new bullet about systems design.")
    rows = compute_bullet_sources(original, optimized)
    assert [r["status"] for r in rows] == ["reworded", "new", "dropped", "dropped"]
    assert rows[0]["source"] == "Q source bullet kept as reworded copy."
    assert rows[1]["source"] is None
    assert [r["source"] for r in rows[2:]] == [
        "Prow first source bullet.", "R third source bullet."]
    # every dropped row quotes the ORIGINAL side's section
    assert all(r["section"] == "Experience" for r in rows)


# ---------------------------------------------------------------------------
# Extraction contract + caps
# ---------------------------------------------------------------------------

def test_education_and_link_only_bullets_skipped():
    """The impact audit's extraction contract verbatim: education
    coursework is not provenance material and a '- [Live Demo](url)' line
    never becomes a row."""
    text = ("## Education\n"
            "**State University** | 2020\n"
            "- Coursework: ML, DL, CV\n"
            "## Projects\n"
            "**Parser** | 2024\n"
            "- [Live Demo](https://example.com)\n"
            "- Built a parser reaching 96.2% accuracy.")
    rows = compute_bullet_sources(text, text)
    assert len(rows) == 1
    assert rows[0]["status"] == "unchanged"
    assert rows[0]["bullet"] == "Built a parser reaching 96.2% accuracy."


def test_row_cap_prefer_describing_the_shipped_resume():
    """max_rows fills with the optimized-side rows first; dropped rows only
    fill the remainder."""
    bullets = "\n".join(f"- Source bullet number {i} about systems." for i in range(4))
    original = f"## Experience\n**Engineer** | Corp | 2024\n{bullets}"
    opt_bullets = ("- Completely different content regarding clouds.\n"
                   "- Wholly unrelated text about gardens.")
    optimized = f"## Experience\n**Engineer** | Corp | 2024\n{opt_bullets}"
    rows = compute_bullet_sources(original, optimized, max_rows=3)
    assert len(rows) == 3
    assert [r["status"] for r in rows] == ["new", "new", "dropped"]


def test_truncation_caps():
    long_text = "- " + ("Built services " * 30) + "end."
    original = f"## Experience\n**Engineer** | Corp | 2024\n{long_text}"
    rows = compute_bullet_sources(original, original)
    assert len(rows) == 1
    assert len(rows[0]["bullet"]) <= _MAX_BULLET_CHARS
    assert rows[0]["bullet"].startswith("Built services")


def test_empty_and_prose_inputs():
    assert compute_bullet_sources("", "") == []
    assert compute_bullet_sources("Python developer.", "## X\n- A bullet.") == []
    assert compute_bullet_sources("## X\n- A bullet.", "") == []
    # prose resumes (no '- ' items) have no bullets to trace: empty by
    # definition - the aggregate integrity scan still covers metric drops
    assert compute_bullet_sources(
        "Python developer with Docker.", "Kubernetes engineer.") == []


def test_dollar_and_count_metric_classes_ride():
    """The quantification vocabulary is the impact audit's ($ amounts and
    curated count units ride beside the integrity metric tokens)."""
    original = ("## Experience\n**Engineer** | Corp | 2024\n"
                "- Cut infra spend by $1.2M across 12 services.")
    optimized = ("## Experience\n**Engineer** | Corp | 2024\n"
                 "- Cut infra spend across 12 services.")
    rows = compute_bullet_sources(original, optimized)
    assert rows[0]["metrics"] == ["12 services"]
    assert rows[0]["dropped"] == ["$1.2M"]


def test_threshold_constant_below_is_new():
    """The threshold is the module's honesty contract: below it a rewrite
    would be a guess, so the row reads 'new'."""
    assert _MATCH_THRESHOLD == 0.62
    assert _MAX_ROWS == 40


def test_reexport():
    assert agent.compute_bullet_sources is compute_bullet_sources
