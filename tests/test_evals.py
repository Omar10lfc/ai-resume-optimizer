"""The LangSmith eval-set round (Nov 15 2026): extraction, dedupe, the three
code-only evaluators, and the deprecation-safe runs query - all against
fakes; no network, no real LangSmith client, no LLM calls."""
import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langsmith.schemas import Run

from agent.evals import (EVAL_DATASET_NAME, build_eval_set, ensure_dataset,
                         eval_integrity, eval_keyword_precision,
                         eval_score_sanity, example_key, existing_example_keys,
                         extract_example, fetch_recent_runs, push_examples,
                         replay_evals)

JD = ("Senior Backend Engineer. You will own services in Python and "
      "FastAPI, deploy REST APIs on Kubernetes and Docker, and tune SQL "
      "stores.")

RESUME = ("# John Doe\njohn@example.com | +1 555 010 2030\n\n"
          "## Experience\n- Built services in Python and FastAPI\n"
          "- Cut manual data-ingestion effort by 40 percent")

OUTPUTS = {
    "job_description": JD,
    "job_text": JD,
    "optimized_resume": "# John Doe\n## Experience\n- Built Python + FastAPI "
                        "services, cut manual data-ingestion effort by 40 percent",
    "cover_letter": "Dear team, I cut manual data-ingestion effort by 40 percent.",
    "score": 72,
    "ats_percentage": 65.0,
    "review_failed": False,
    "scrub_removals": [],
    "unevidenced_metrics": [],
    "guardrail_warnings": [],
    "declined_requests": [],
    "cover_letter_metrics": [],
    "human_notes": "",
    "original_resume": RESUME,
    "resume_text": RESUME,
    "ats_result": ("## ATS Keyword Match: 65% (weighted)\n"
                   "*6 of 6 keywords found*\n\n"
                   "### Found in Resume\n"
                   "- **Tools:** `Python`, `FastAPI`, `Docker`\n"
                   "- **Concepts:** `Kubernetes`, `SQL`\n"
                   "- **General:** `REST APIs`\n\n"
                   "### Missing from Resume\n"),
    "prompt_versions": {"improver": "improver@4"},
    "resume_pdf_path": "C:/local/layout/req-1/Resume.pdf",
    "resume_docx_path": "C:/local/layout/req-1/Resume.docx",
    "output_dir": "C:/local/layout/req-1",
    "request_id": "req-1",
}


def _run(*, outputs=None, inputs=None, parent=None, run_type="chain",
         name="resume-optimizer"):
    return Run(id=uuid.uuid4(), name=name, run_type=run_type,
               inputs=inputs if inputs is not None else {
                   # the LIVE traced shape (verified Nov 15 2026): the
                   # server stores streamed LangGraph root inputs as
                   # {"input": None}; the full final state rides outputs
                   "input": None},
               outputs=outputs if outputs is not None else OUTPUTS,
               start_time=datetime.now(timezone.utc),
               parent_run_ids=list(parent or []))


def _example(outputs=None, jd=JD, resume=RESUME):
    run = _run(outputs=outputs if outputs is not None else dict(OUTPUTS),
               inputs={"job_description": jd, "resume_text": resume})
    return extract_example(run)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_example_from_root_pipeline_run():
    ex = extract_example(_run())
    assert ex is not None
    assert set(ex["inputs"]) == {"job_description", "resume_text"}
    assert ex["inputs"]["job_description"] == JD
    for key in ("resume_pdf_path", "resume_docx_path", "output_dir",
                "request_id"):
        assert key not in ex["outputs"]
    assert ex["outputs"]["optimized_resume"].startswith("# ")
    assert ex["metadata"]["run_id"]
    assert ex["metadata"]["start_time"]


def test_extract_redacts_pii_defense_in_depth():
    ex = extract_example(_run())
    # the raw email/phone never survive into the example - the tracer
    # already redacted at trace time; the builder re-applies the mask
    assert "john@example.com" not in str(ex["inputs"])
    assert "+1 555 010 2030" not in str(ex["inputs"])
    assert "[redacted-email]" in ex["inputs"]["resume_text"]


def test_extract_skips_children_llm_and_step1_roots():
    assert extract_example(_run(parent=[uuid.uuid4()])) is None
    assert extract_example(_run(run_type="llm")) is None
    # step-1 scan root: no optimized_resume in the outputs
    step1 = dict(OUTPUTS)
    step1.pop("optimized_resume")
    assert extract_example(_run(outputs=step1)) is None
    # no JD anywhere - neither in the (unwrapped) inputs nor the outputs
    no_jd = {k: v for k, v in OUTPUTS.items()
             if k not in ("job_description", "job_text")}
    assert extract_example(_run(outputs=no_jd,
                                inputs={"input": {"resume_text": RESUME}})) is None
    # empty outputs (stripped / still-running run): nothing to extract
    assert extract_example(_run(outputs={})) is None


def test_extract_accepts_flat_inputs_and_inputs_win():
    flat = {"job_description": "OTHER JD", "resume_text": RESUME}
    ex = extract_example(_run(inputs=flat))
    assert ex is not None
    assert ex["inputs"]["job_description"] == "OTHER JD"   # inputs, not outputs


def test_extract_falls_back_to_outputs_for_streamed_roots():
    # the live step-2 shape: {"input": None} on the wire, the full final
    # state (which carries the original inputs) in the outputs
    ex = extract_example(_run(inputs={"input": None}))
    assert ex is not None
    assert ex["inputs"]["job_description"] == JD
    # the fallback text rides the whole-example PII redaction
    assert "[redacted-email]" in ex["inputs"]["resume_text"]
    assert "Built services in Python and FastAPI" in ex["inputs"]["resume_text"]
    # a fully-empty inputs dict behaves the same
    assert extract_example(_run(inputs={})) is not None


def test_example_key_is_pair_scoped_and_stable():
    a = _example()
    b = _example(outputs=dict(OUTPUTS, score=91))  # same pair, new result
    assert example_key(a) == example_key(b)
    other_resume = RESUME + "\n- Extra bullet"
    assert example_key(a) != example_key(_example(resume=other_resume))


# ---------------------------------------------------------------------------
# Evaluator: score sanity
# ---------------------------------------------------------------------------


def test_score_sanity_passes_on_in_range_scores():
    result = eval_score_sanity(_example())
    assert result["passed"] is True and result["score"] == 1.0


def test_score_sanity_fails_on_out_of_range_and_missing():
    bad = dict(OUTPUTS, score=140)
    result = eval_score_sanity(_example(outputs=bad))
    assert result["passed"] is False and "out of range: 140" in result["reason"]
    missing = {k: v for k, v in OUTPUTS.items() if k != "ats_percentage"}
    result = eval_score_sanity(_example(outputs=missing))
    assert result["passed"] is False and "missing ATS percentage" in result["reason"]
    result = eval_score_sanity(_example(outputs=dict(OUTPUTS, score=True)))
    assert result["passed"] is False and "missing composite score" in result["reason"]


# ---------------------------------------------------------------------------
# Evaluator: integrity (the pipeline's own scan re-run)
# ---------------------------------------------------------------------------


def test_integrity_passes_on_clean_and_rounding_tolerant_artifacts():
    assert eval_integrity(_example())["passed"] is True
    # the exact round-12 incident shape: "96.2%" in evidence, "96 %" shipped
    evidence = RESUME + "\nAccuracy 96.2%"
    shipped = dict(OUTPUTS, original_resume=evidence, resume_text=evidence,
                   optimized_resume="- Verified accuracy 96 %\n")
    assert eval_integrity(_example(outputs=shipped))["passed"] is True


def test_integrity_fails_on_shipped_fabrication():
    fabricated = dict(OUTPUTS, optimized_resume=RESUME +
                      "\n- Improved conversion 45%")
    result = eval_integrity(_example(outputs=fabricated))
    assert result["passed"] is False and "45%" in result["reason"]
    letter_only = dict(OUTPUTS, cover_letter="I improved conversion 77%.")
    result = eval_integrity(_example(outputs=letter_only))
    assert result["passed"] is False and "letter: 77%" in result["reason"]


# ---------------------------------------------------------------------------
# Evaluator: keyword precision
# ---------------------------------------------------------------------------


def test_keyword_precision_all_keywords_from_the_jd_passes():
    result = eval_keyword_precision(_example())
    assert result["passed"] is True and result["score"] == 1.0


def test_keyword_precision_fails_below_floor_and_names_misses():
    ats = ("### Found in Resume\n"
           "- **Tools:** `Python`, `FastAPI`, `Rust`\n\n"
           "### Missing from Resume\n"
           "- `temperature` - *try adding to: Skills or Experience*\n")
    jd = "Backend role: Python, FastAPI"  # Rust/temperature are not in the JD
    result = eval_keyword_precision(_example(outputs=dict(OUTPUTS,
                                                          ats_result=ats),
                                             jd=jd))
    assert result["score"] == 0.5            # 2 misses of 4
    assert result["passed"] is False
    assert "Rust" in result["reason"] and "temperature" in result["reason"]


def test_keyword_precision_still_parses_the_plain_tier_format():
    ats = "Tools: Python, FastAPI\n"
    result = eval_keyword_precision(_example(outputs=dict(OUTPUTS,
                                                          ats_result=ats),
                                             jd="role: Python, FastAPI"))
    assert result["passed"] is True and result["score"] == 1.0


def test_keyword_precision_unscored_without_tier_lines():
    result = eval_keyword_precision(_example(outputs=dict(OUTPUTS,
                                                          ats_result="")))
    assert result["score"] is None and result["passed"] is None
    assert "unscored" in result["reason"]


# ---------------------------------------------------------------------------
# Replay report
# ---------------------------------------------------------------------------


def test_replay_evals_aggregates_and_caps_failures():
    examples = [_example() for _ in range(3)]
    for i in range(12):
        examples.append(_example(outputs=dict(OUTPUTS, score=None)))
    report = replay_evals(examples)
    assert report["examples"] == 15
    sanity = report["evaluators"]["score_sanity"]
    assert sanity["passed"] == 3 and sanity["failed"] == 12
    integrity = report["evaluators"]["integrity"]
    assert integrity["passed"] == 15 and integrity["failed"] == 0
    assert len(report["failures"]) == 10       # the cap
    assert report["total_failed"] == 12        # only score_sanity failed


# ---------------------------------------------------------------------------
# Dataset: get-or-create + deduped push
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, runs=(), existing=()):
        self.runs = _FakeRuns(runs)
        self._existing = list(existing)
        self._datasets = {}
        self.created_datasets = []
        self.created_examples = []
        self.read_dataset_calls = 0

    async def aread_project(self, project_name=None):
        return SimpleNamespace(id=uuid.uuid4())

    def read_dataset(self, dataset_name=None):
        self.read_dataset_calls += 1
        if dataset_name in self._datasets:
            return self._datasets[dataset_name]
        raise RuntimeError("not found")

    def create_dataset(self, name, description=None):
        ds = SimpleNamespace(id=str(uuid.uuid4()), name=name)
        self._datasets[name] = ds
        self.created_datasets.append(name)
        return ds

    def list_examples(self, dataset_id=None):
        return [SimpleNamespace(inputs=ex) for ex in self._existing]

    def create_example(self, inputs=None, outputs=None, dataset_id=None):
        self.created_examples.append(inputs)


def test_ensure_dataset_creates_once_and_reuses():
    client = _FakeClient()
    first = ensure_dataset(client)
    second = ensure_dataset(client)
    assert first.id == second.id
    assert client.created_datasets == [EVAL_DATASET_NAME]
    assert client.read_dataset_calls == 2


def test_push_examples_dedupes_on_the_pair_key():
    ex = _example()
    client = _FakeClient(existing=[ex["inputs"]])
    stats = push_examples(client, [ex], dataset_id="ds-1",
                          existing=existing_example_keys(client, "ds-1"))
    assert stats == {"pushed": 0, "skipped": 1}
    # a different (jd, resume) pair is a NEW example even with the same shape
    fresh = _example(outputs=dict(OUTPUTS, score=95),
                     resume=RESUME + "\n- Later run, fresh pair")
    stats = push_examples(client, [fresh], dataset_id="ds-1",
                          existing=existing_example_keys(client, "ds-1"))
    assert stats == {"pushed": 1, "skipped": 0}
    assert len(client.created_examples) == 1


# ---------------------------------------------------------------------------
# Fetch: the deprecation-safe runs query
# ---------------------------------------------------------------------------


class _FakeRuns:
    def __init__(self, runs=(), failing_selects=()):
        self._runs = runs
        self._failing = list(failing_selects)
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("selects") in self._failing:
            raise TypeError("unsupported field")
        return _AsyncIter(self._runs)


class _AsyncIter:
    def __init__(self, items):
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def test_fetch_recent_runs_returns_roots_only_and_respects_limit():
    child = _run(parent=[uuid.uuid4()])
    root1, root2 = _run(), _run()
    fake = _FakeRuns([child, root1, root2])
    roots = asyncio.run(fetch_recent_runs(SimpleNamespace(runs=fake),
                                          "proj-1", limit=2))
    assert [r.id for r in roots] == [root1.id, root2.id]
    call = fake.calls[0]
    assert call["project_ids"] == ["proj-1"]
    assert call["min_start_time"] is not None      # the "1 day ago" trap
    assert call["selects"] and call["selects"][0] == "ID"


def test_fetch_recent_runs_falls_back_when_a_select_is_rejected():
    from agent.evals import _SELECTS_CORE, _SELECTS_EVAL

    root = _run()
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        if kwargs.get("selects") == _SELECTS_EVAL:
            raise TypeError("unsupported field")
        return _AsyncIter([root])
    fake = SimpleNamespace(runs=SimpleNamespace(query=query))
    runs = asyncio.run(fetch_recent_runs(fake, "proj-1", limit=5))
    assert [r.id for r in runs] == [root.id]
    assert len(calls) == 2                    # the two-variant ladder
    assert calls[0]["selects"] == _SELECTS_EVAL
    assert calls[1]["selects"] == _SELECTS_CORE


def test_fetch_recent_runs_raises_after_all_variants_fail():
    def query(**kwargs):
        raise TypeError("unsupported field")
    fake = SimpleNamespace(runs=SimpleNamespace(query=query))
    with pytest.raises(RuntimeError, match="all selects variants"):
        asyncio.run(fetch_recent_runs(fake, "proj-1"))


# ---------------------------------------------------------------------------
# Orchestration end-to-end (fake client)
# ---------------------------------------------------------------------------


def test_build_eval_set_end_to_end_extracts_pushes_and_reports():
    step2_root = _run()
    step1_root = _run(outputs={k: v for k, v in OUTPUTS.items()
                               if k != "optimized_resume"})
    child = _run(parent=[uuid.uuid4()])
    client = _FakeClient(runs=[step2_root, step1_root, child])
    stats = asyncio.run(build_eval_set(client, "resume-optimizer",
                                       push=True))
    assert stats["fetched"] == 2          # the child is filtered at fetch
    assert stats["extracted"] == 1        # the step-1 root at extraction
    assert stats["pushed"] == 1 and stats["skipped"] == 0
    assert stats["report"]["examples"] == 1
    assert client.created_datasets == [EVAL_DATASET_NAME]
    assert client.created_examples[0]["job_description"] == JD
