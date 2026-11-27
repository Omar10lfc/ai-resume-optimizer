"""Tests for Issue 2: named tools must not be marked 'covered' by generic
neighbor words, short/alphanumeric tools must survive keyword extraction, and
declined tools must never be marked covered.
"""
from hypothesis import given, settings, strategies as st

from agent.ats import _extract_keywords, compute_ats_match
from agent.match import compute_requirement_match_table
from agent.nodes import ats_check as ats_check_mod


TOOL_CASES = [
    "n8n",
    "k3s",
    "s3",
    "ec2",
    "d3",
    "v8",
    "k6",
    "jq",
    "dbt",
    "oauth2",
    "vue3",
    "log4j",
    "web3",
    "airflow",
    "terraform",
    "snowflake",
    "kafka",
    "zapier",
    "langchain",
    "pytorch",
    "databricks",
]

NEIGHBOR_WORDS = [
    "automations",
    "workflows",
    "pipelines",
    "orchestration",
    "integrations",
    "analytics",
    "infrastructure",
    "deployments",
    "microservices",
    "messaging",
]


def test_n8n_declined_tool_never_covered_by_automations():
    """Fixed regression: 'n8n automations that help our internal team' must not
    be 'covered' by ['automations'] when n8n is absent or in declined_requests.
    """
    jd = (
        "## Requirements\n"
        "- Experience building n8n automations that help our internal team.\n"
        "- Strong Python backend skills.\n"
    )
    resume = (
        "## Skills\nPython\n"
        "## Experience\n"
        "- Built internal automations and workflows in Python.\n"
    )

    extracted = _extract_keywords(jd, top_n=20)
    assert "n8n" in extracted

    rows = compute_requirement_match_table(
        jd,
        resume,
        human_notes="",
        declined_requests=["Do NOT feature: n8n"],
    )
    n8n_row = next(r for r in rows if "n8n" in r["requirement"].lower())
    assert n8n_row["status"] != "covered"
    assert "n8n" in n8n_row["missing"]

    # Even if a semantic LLM verifier tried to include 'n8n', declined tools stay uncovered.
    rows_with_sem = compute_requirement_match_table(
        jd,
        resume,
        semantic_matches={"n8n", "automations"},
        human_notes="",
        declined_requests=["Do NOT feature: n8n"],
    )
    n8n_sem_row = next(r for r in rows_with_sem if "n8n" in r["requirement"].lower())
    assert n8n_sem_row["status"] != "covered"
    assert "n8n" in n8n_sem_row["missing"]

    # Without declined_requests, 'automations' alone still cannot cover 'n8n'.
    rows_undeclined = compute_requirement_match_table(jd, resume)
    n8n_undec = next(r for r in rows_undeclined if "n8n" in r["requirement"].lower())
    assert n8n_undec["status"] != "covered"
    assert "n8n" in n8n_undec["missing"]

    # Confirmed human_notes naming n8n DOES cover it when not declined.
    rows_with_notes = compute_requirement_match_table(
        jd,
        resume,
        human_notes="Built internal n8n webhooks for alert routing.",
        declined_requests=[],
    )
    n8n_noted = next(r for r in rows_with_notes if "n8n" in r["requirement"].lower())
    assert n8n_noted["status"] == "covered"
    assert "n8n" in n8n_noted["evidence"]


def test_short_alphanumeric_tool_survives_keyword_extraction_even_under_filler():
    """Short alphanumeric tools like n8n, s3, k3s, v8 must survive top_n extraction
    and not be dropped by digitless tokenization or eclipsed by generic chatter."""
    jd = (
        "## Responsibilities\n"
        + "\n".join(
            f"- Collaborate across teams on scalable internal systems, "
            f"microservices, analytics, pipelines, cloud platforms, "
            f"automation, monitoring, testing, governance, documentation, "
            f"security, reliability, performance, scalability, architecture ({i})."
            for i in range(6)
        )
        + "\n## Requirements\n"
        "- Build n8n automations and S3 storage pipelines with k3s.\n"
    )
    kws = _extract_keywords(jd, top_n=20)
    assert "n8n" in kws
    assert "s3" in kws
    assert "k3s" in kws

    ats = compute_ats_match(jd, "## Experience\nBuilt internal automations and storage pipelines.")
    missing = {r["keyword"] for r in ats["keywords"] if not r["found"]}
    assert {"n8n", "s3", "k3s"}.issubset(missing)


@settings(max_examples=60)
@given(
    tool=st.sampled_from(TOOL_CASES),
    neighbor=st.sampled_from(NEIGHBOR_WORDS),
    extra_filler=st.lists(
        st.sampled_from(["team", "internal", "production", "reliable", "scalable"]),
        min_size=0,
        max_size=3,
    ),
)
def test_named_tools_not_covered_by_neighbor_words_hypothesis(tool, neighbor, extra_filler):
    """Any specific named tool in a requirement must never be classified 'covered'
    merely because a generic neighbor word ('automations', 'workflows', ...) is in the resume.
    """
    filler = " ".join(extra_filler)
    jd = f"## Requirements\n- Experience building {tool} {neighbor} for {filler or 'our team'}.\n"
    resume = (
        "## Skills\nPython, SQL\n"
        "## Experience\n"
        f"- Built {neighbor} for enterprise {filler or 'systems'} using Python.\n"
    )
    rows = compute_requirement_match_table(jd, resume)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] != "covered"
    assert tool in row["missing"]


@settings(max_examples=30)
@given(
    tool=st.sampled_from(TOOL_CASES),
    neighbor=st.sampled_from(NEIGHBOR_WORDS),
)
def test_declined_tool_never_covered_hypothesis(tool, neighbor):
    """A tool listed in declined_requests can never be marked 'covered', even
    if a semantic match set or neighbor words overlap."""
    jd = f"## Requirements\n- Hands-on {tool} {neighbor} experience.\n"
    resume = f"## Experience\n- Built {neighbor} solutions.\n"
    rows = compute_requirement_match_table(
        jd,
        resume,
        semantic_matches={tool, neighbor},
        declined_requests=[f"Do NOT feature: {tool}"],
    )
    assert len(rows) == 1
    assert rows[0]["status"] != "covered"
    assert tool in rows[0]["missing"]


def test_ats_check_node_passes_human_notes_and_declined_requests(monkeypatch):
    """End-to-end node check: declined_requests prevents coverage in requirement_table."""
    monkeypatch.setattr(
        ats_check_mod,
        "_semantic_verify_missing",
        lambda *_a, **_kw: {"n8n", "automations"},
    )
    state = {
        "job_text": (
            "## Requirements\n"
            "- Build n8n automations for internal operations.\n"
            "- Write Python services.\n"
        ),
        "optimized_resume": "## Skills\nPython\n## Experience\nBuilt automations in Python.",
        "resume_text": "## Skills\nPython\n## Experience\nBuilt automations in Python.",
        "human_notes": "",
        "declined_requests": ["Do NOT feature: n8n"],
        "semantic_matches": ["n8n", "automations"],
        "input_pages": 1,
        "target_pages": 1,
    }
    out = ats_check_mod.ats_check_node(state)
    n8n_row = next(
        r for r in out["requirement_matches"] if "n8n" in r["requirement"].lower()
    )
    assert n8n_row["status"] != "covered"
    assert "n8n" in n8n_row["missing"]
