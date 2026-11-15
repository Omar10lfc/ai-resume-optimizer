"""LangSmith eval set (Phase-4 bet, Nov 15 2026): replay real pipeline runs
into a dataset and score them with deterministic evaluators.

Two halves:

1. DATASET BUILD - root runs of the ``resume-optimizer`` project whose
   outputs carry the scored result become dataset examples:
   ``{job_description, resume_text}`` -> the result payload. The jd +
   resume come from the root inputs when the server stored them; the
   streamed step-2 roots store ``{"input": None}`` (the invoke payload
   never reached storage), so they fall back to the final state in the
   outputs, which carries the original inputs verbatim. Everything is
   re-redacted through
   ``agent.tracing._redact_pii`` (defense in depth - the tracer already
   redacted at trace time, ``hide_inputs``/``hide_outputs``). File-path keys
   (``*_path``, ``output_dir``, ``request_id``) are dropped: local paths
   have no eval value and leak machine layout. Examples dedupe on a sha256
   of the (jd, resume) pair, so re-running the builder only pushes NEW
   runs.

2. REPLAY EVAL - three CODE-ONLY evaluators (zero LLM calls, zero token
   spend), run over the freshly extracted examples and aggregated into a
   report:

   - ``score_sanity``: the recorded composite ``score`` and
     ``ats_percentage`` are numbers inside 0..100 (missing = fail).
   - ``integrity``: the pipeline's OWN deterministic fabrication scan
     (``agent.helpers.find_unevidenced_metric_claims``, rounding-tolerant)
     re-runs on the SHIPPED optimized resume + cover letter against the
     evidence (source resume + user notes). Any finding = the shipped
     artifact contains an unevidenced metric the pipeline's gates should
     have caught - a real integrity regression, not a recording artifact.
   - ``keyword_precision``: the ATS tier keywords surfaced to the user
     (``ats_result``'s ``Tools:/Concepts:/General:`` lines) must all come
     from the JD text - precision is reported per example; below 0.8 is a
     fail, an unparseable ``ats_result`` is "unscored" (reported, never a
     fail - the field is display-shaped).

The runs are fetched through ``client.runs.query`` - the SmithDB-backed
API that replaces ``list_runs`` (which deprecates after Jan 31 2027).
The query traps are the ones the export script documented: the project
NAME must be resolved to a UUID first; ``min_start_time`` must be passed
explicitly (it otherwise defaults to "1 day ago"); only ``ID`` comes back
unless ``selects`` (uppercase field names) are requested - requested with
a progressive fallback ladder, mirroring
``scripts/export_langsmith_traces.py``.

The dataset ALSO makes the examples available to LLM-judged evals in the
LangSmith UI; this module deliberately stops at the deterministic layer.

CLI: ``scripts/build_eval_set.py``. No LLM calls anywhere - the only API
traffic is LangSmith reads/writes (dataset + examples), keyed by
``LANGCHAIN_API_KEY``.
"""
import asyncio
import hashlib
import re
from datetime import datetime, timedelta, timezone

from .helpers import find_unevidenced_metric_claims
from .tracing import _redact_pii

EVAL_DATASET_NAME = "resume-optimizer-evals"
EVAL_DATASET_DESCRIPTION = (
    "resume-optimizer pipeline replays: {job_description, resume_text} -> "
    "the scored result payload (optimized resume + ATS + integrity "
    "channels). Built by scripts/build_eval_set.py from real LangSmith "
    "traces; PII re-redacted locally.")

# Output keys with no eval value (local machine layout) - dropped.
_PATH_KEYS = frozenset({
    "resume_pdf_path", "cover_letter_pdf_path", "resume_docx_path",
    "resume_tex_path", "resume_html_path", "output_dir", "request_id",
})

# Uppercase SmithDB v2 field names (see scripts/export_langsmith_traces.py).
# The eval set needs inputs/outputs + identity only - no events, costs,
# feedback stats or tokens. Progressive fallback mirrors the export script.
_SELECTS_EVAL = [
    "ID", "NAME", "RUN_TYPE", "STATUS", "START_TIME", "END_TIME", "ERROR",
    "INPUTS", "OUTPUTS", "TRACE_ID", "DOTTED_ORDER", "PARENT_RUN_IDS",
]
_SELECTS_CORE = ["ID", "NAME", "RUN_TYPE", "STATUS", "START_TIME",
                 "INPUTS", "OUTPUTS", "PARENT_RUN_IDS"]


# ---------------------------------------------------------------------------
# Extraction: a root run -> one dataset example
# ---------------------------------------------------------------------------


def _run_type(run) -> str:
    value = getattr(run, "run_type", None)
    value = getattr(value, "value", value)
    return str(value or "").upper()


def extract_example(run):
    """Converts a ROOT pipeline run into a redacted dataset example, or None
    when the run is not one (children, step-1 scan roots, LLM runs, error
    runs). The example carries ONLY the eval-relevant shape: the jd + resume
    inputs, the result payload minus path keys, and a small metadata block.
    PII is re-redacted over the whole example (defense in depth).

    Real trace shapes (verified live Nov 15 2026): the root run is named
    ``LangGraph`` (CHAIN, success) and its outputs are the FLAT final state
    (``optimized_resume`` etc.). The server stores the STREAMED step-2 root
    inputs as ``{"input": None}`` - the invoke payload never reached
    storage - so the jd + resume fall back to the final state in the
    outputs, which carries the original inputs verbatim (job_description /
    resume_text / original_resume are state channels). Wrapped-dict inputs
    (older exports) are unwrapped, and stored inputs always win over the
    outputs fallback (closest to what the user actually submitted)."""
    inputs = getattr(run, "inputs", None) or {}
    if not any(k in inputs for k in ("job_description", "job_text")) \
            and isinstance(inputs.get("input"), dict):
        inputs = inputs["input"]
    outputs = getattr(run, "outputs", None) or {}
    if getattr(run, "parent_run_ids", None):
        return None
    if _run_type(run) not in ("", "CHAIN"):
        return None
    # Streamed step-2 roots store {"input": None} (verified live): the
    # invoke inputs never made it to server storage. The OUTPUTS are the
    # complete final state and carry the original inputs verbatim, so they
    # fill the gap. Stored inputs win when present.
    jd = (inputs.get("job_description") or inputs.get("job_text")
          or outputs.get("job_description") or outputs.get("job_text"))
    resume = (inputs.get("resume_text") or inputs.get("original_resume")
              or outputs.get("resume_text") or outputs.get("original_resume"))
    optimized = outputs.get("optimized_resume")
    if not (isinstance(jd, str) and jd.strip()
            and isinstance(resume, str) and resume.strip()
            and isinstance(optimized, str) and optimized.strip()):
        return None
    kept_outputs = {k: v for k, v in outputs.items()
                    if k not in _PATH_KEYS and not k.endswith("_path")}
    start = getattr(run, "start_time", None)
    example = {
        "inputs": {"job_description": jd, "resume_text": resume},
        "outputs": kept_outputs,
        "metadata": {
            "run_id": str(getattr(run, "id", "") or ""),
            "name": getattr(run, "name", None) or "",
            "start_time": start.isoformat() if isinstance(start, datetime) else None,
        },
    }
    return _redact_pii(example)


def example_key(example: dict) -> str:
    """Stable dedupe key: sha256 over the (jd, resume) pair, whitespace-
    stripped. The same JD+resume re-optimized later maps to the SAME key -
    the dataset holds one example per pair (the freshest outputs win only
    by explicit manual curation; the builder skips existing keys)."""
    pair = "\x00".join((example["inputs"]["job_description"].strip(),
                        example["inputs"]["resume_text"].strip()))
    return hashlib.sha256(pair.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fetch: the deprecation-safe runs query
# ---------------------------------------------------------------------------


async def fetch_recent_runs(client, project_id: str, *, limit: int = 50,
                            window_days: int = 400) -> list:
    """Root runs of the project through ``client.runs.query`` (the
    ``list_runs`` replacement), newest first, capped at ``limit``.

    Window: the v2 endpoint caps any query at 401 days, so ``window_days``
    is clamped to 400 and the window start is passed explicitly (omitting
    it silently defaults to "1 day ago" - the trap the export script
    documented). ``selects`` rides a two-variant fallback ladder for
    backends that reject newer fields."""
    now = datetime.now(timezone.utc)
    min_start = now - timedelta(days=min(window_days, 400))
    max_start = now + timedelta(seconds=2)
    roots: list = []
    last_error = None
    for selects in (_SELECTS_EVAL, _SELECTS_CORE):
        try:
            async for run in client.runs.query(
                    project_ids=[project_id],
                    min_start_time=min_start,
                    max_start_time=max_start,
                    page_size=100,
                    selects=selects):
                if getattr(run, "parent_run_ids", None):
                    continue  # children never carry the full pipeline shape
                roots.append(run)
                if limit and len(roots) >= limit:
                    break
            break
        except Exception as exc:  # unsupported select field -> smaller ladder
            last_error = exc
            roots = []
            continue
    else:
        raise RuntimeError(
            f"runs.query failed with all selects variants: {last_error}")
    return roots


# ---------------------------------------------------------------------------
# The dataset: get-or-create + deduped push
# ---------------------------------------------------------------------------


def ensure_dataset(client, name: str = EVAL_DATASET_NAME):
    """Returns the LangSmith dataset, creating it on first use."""
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        return client.create_dataset(name, description=EVAL_DATASET_DESCRIPTION)


def existing_example_keys(client, dataset_id: str) -> set:
    """sha256 keys of every example already in the dataset."""
    keys = set()
    for ex in client.list_examples(dataset_id=dataset_id):
        inputs = getattr(ex, "inputs", None) or {}
        jd = inputs.get("job_description")
        resume = inputs.get("resume_text")
        if isinstance(jd, str) and isinstance(resume, str):
            keys.add(example_key({"inputs": {
                "job_description": jd, "resume_text": resume}}))
    return keys


def push_examples(client, examples: list, dataset_id: str,
                  existing: set) -> dict:
    """Pushes examples whose (jd, resume) key is not already present.
    Returns {pushed, skipped}."""
    pushed = skipped = 0
    for example in examples:
        if example_key(example) in existing:
            skipped += 1
            continue
        client.create_example(inputs=example["inputs"],
                              outputs=example["outputs"],
                              dataset_id=dataset_id)
        pushed += 1
    return {"pushed": pushed, "skipped": skipped}


# ---------------------------------------------------------------------------
# The three code-only evaluators
# ---------------------------------------------------------------------------


def _evidence_text(outputs: dict) -> str:
    """Original resume (either input key - pasted lands in
    ``original_resume``, URL imports fill ``resume_text``) + user notes."""
    parts = [outputs.get("original_resume") or "",
             outputs.get("resume_text") or "",
             outputs.get("human_notes") or ""]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())


def eval_score_sanity(example: dict) -> dict:
    """The recorded composite score and ATS percentage are numbers inside
    0..100. Missing either = fail (a replayed pipeline run without a score
    is itself the regression)."""
    outputs = example["outputs"]
    findings = []
    for key, label in (("score", "composite score"),
                       ("ats_percentage", "ATS percentage")):
        value = outputs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            findings.append(f"missing {label}")
        elif not 0 <= float(value) <= 100:
            findings.append(f"{label} out of range: {value}")
    if findings:
        return {"key": "score_sanity", "score": 0.0, "passed": False,
                "reason": "; ".join(findings)}
    return {"key": "score_sanity", "score": 1.0, "passed": True,
            "reason": f"score {outputs.get('score')}, "
                      f"ats {outputs.get('ats_percentage')}"}


def eval_integrity(example: dict) -> dict:
    """The pipeline's OWN fabrication scan re-runs on the SHIPPED optimized
    resume AND cover letter against the evidence (source resume + user
    notes). A fresh finding means an unevidenced metric survived every
    in-pipeline gate into the shipped artifact."""
    outputs = example["outputs"]
    evidence = _evidence_text(outputs)
    resume = outputs.get("optimized_resume") or ""
    letter = outputs.get("cover_letter") or ""
    findings = find_unevidenced_metric_claims(resume, evidence)
    letter_findings = find_unevidenced_metric_claims(letter, evidence)
    if findings or letter_findings:
        named = findings + (["letter: " + t for t in letter_findings])
        return {"key": "integrity", "score": 0.0, "passed": False,
                "reason": "unevidenced metric(s) in the shipped artifact: "
                          + ", ".join(named)}
    return {"key": "integrity", "score": 1.0, "passed": True,
            "reason": "clean (pipeline scan re-run on the shipped artifact)"}


def eval_keyword_precision(example: dict) -> dict:
    """Every ATS keyword the report surfaces (the ``ats_result`` markdown:
    ``### Found in Resume`` tier lines like ``- **Tools:** `` `a`, `b` ``
    plus the ``### Missing from Resume`` entries) must appear in the JD
    text (case-insensitive) - the engine extracts FROM the JD, so a
    keyword it does not contain is an extraction leak. Precision >= 0.8
    passes; the misses are named. An empty/unparseable ``ats_result`` is
    UNSCORED (score None) - the field is display-shaped, never
    fabricated. Older ``Tools: a, b`` plain tier lines parse too."""
    outputs = example["outputs"]
    ats_result = outputs.get("ats_result") or ""
    jd = example["inputs"]["job_description"]
    keywords: list[str] = []
    for line in ats_result.splitlines():
        stripped = line.strip()
        # "- **Tools:** `a`, `b`" (the live format puts the colon inside the
        # bold) or "- **Tools**: `a`" - accept both orderings
        tier = re.match(
            r"-\s*\*\*(Tools|Concepts|General)(?:\*\*:|:\*\*)\s*(.+)",
            stripped)
        if tier:
            keywords += [t.strip().strip("`") for t in tier.group(2).split(",")]
            continue
        # old plain format: "Tools: a, b" / "Concepts: c"
        plain = re.match(r"(Tools|Concepts|General):\s*(.+)$", stripped)
        if plain:
            keywords += [t.strip().strip("`")
                         for t in plain.group(2).split(",")]
            continue
        missing = re.match(r"-\s*`([^`]+)`", stripped)
        if missing:
            keywords.append(missing.group(1))
    keywords = [kw for kw in keywords if kw]
    if not keywords:
        return {"key": "keyword_precision", "score": None, "passed": None,
                "reason": "unscored: no ATS keywords parsed from ats_result"}
    jd_low = jd.lower()
    misses = [kw for kw in keywords if kw.lower() not in jd_low]
    precision = 1.0 - len(misses) / len(keywords)
    passed = precision >= 0.8
    if passed:
        reason = (f"{len(keywords)}/{len(keywords)} in JD" if not misses
                  else f"precision {precision:.2f} ({len(misses)} miss: "
                       + ", ".join(misses) + ")")
    else:
        reason = f"precision {precision:.2f} - not in JD: {', '.join(misses)}"
    return {"key": "keyword_precision", "score": precision, "passed": passed,
            "reason": reason}


EVALUATORS = (eval_score_sanity, eval_integrity, eval_keyword_precision)


# ---------------------------------------------------------------------------
# The replay report
# ---------------------------------------------------------------------------


def replay_evals(examples: list) -> dict:
    """Runs the three evaluators over every example and aggregates: totals,
    per-evaluator pass/fail/unscored counts, and the failing examples
    (capped at 10) with their reasons. Zero LLM calls. Keys are the
    evaluators' public result keys (score_sanity / integrity /
    keyword_precision)."""
    per_key: dict[str, dict[str, int]] = {
        "score_sanity": {"passed": 0, "failed": 0, "unscored": 0},
        "integrity": {"passed": 0, "failed": 0, "unscored": 0},
        "keyword_precision": {"passed": 0, "failed": 0, "unscored": 0},
    }
    failures = []
    for example in examples:
        meta = example.get("metadata") or {}
        for result in (evaluator(example) for evaluator in EVALUATORS):
            stats = per_key[result["key"]]
            if result["passed"] is True:
                stats["passed"] += 1
            elif result["passed"] is False:
                stats["failed"] += 1
                if len(failures) < 10:
                    failures.append({
                        "run_id": meta.get("run_id") or "",
                        "eval": result["key"],
                        "reason": result["reason"]})
            else:
                stats["unscored"] += 1
    return {
        "examples": len(examples),
        "evaluators": per_key,
        "failures": failures,
        "total_failed": sum(s["failed"] for s in per_key.values()),
    }


# ---------------------------------------------------------------------------
# Orchestration (used by the CLI; all network I/O via the client)
# ---------------------------------------------------------------------------


async def build_eval_set(client, project_name: str, *, limit: int = 50,
                         window_days: int = 400, push: bool = True) -> dict:
    """Fetch -> extract -> (optionally) dedupe-push into the dataset, and
    return {fetched, extracted, pushed, skipped, report}."""
    project = None
    try:
        project = await client.aread_project(project_name=project_name)
    except AttributeError:
        project = client.read_project(project_name=project_name)
    if project is None:
        raise RuntimeError(f"Project '{project_name}' not found on LangSmith.")

    runs = await fetch_recent_runs(client, str(project.id), limit=limit,
                                   window_days=window_days)
    examples = [ex for ex in (extract_example(run) for run in runs) if ex]
    stats = {"fetched": len(runs), "extracted": len(examples)}
    if push and examples:
        dataset = ensure_dataset(client)
        existing = existing_example_keys(client, str(dataset.id))
        stats.update(push_examples(client, examples, str(dataset.id),
                                   existing))
    stats["report"] = replay_evals(examples)
    return stats
