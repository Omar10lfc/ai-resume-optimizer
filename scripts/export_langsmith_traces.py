"""Bulk-export ALL runs of the LangSmith project to local JSON + trace tree files.

Uses the SmithDB-backed API (client.runs.query — the list_runs replacement that
deprecates after Jan 31 2027). Traps handled per the official migration guide:
- project name must be resolved to a UUID first (project_ids, not project_name)
- min_start_time defaults to "1 day ago" when omitted — an explicit epoch is
  passed so nothing in the project history is silently skipped
- only "ID" is returned unless `selects` (uppercase field names) are requested
- token/cost fields are per-run own usage (not summed from children); trace
  totals are reconstructed during ANALYSIS by summing LLM runs per trace_id

Everything stored on the server was already client-side redacted at trace time
(agent/tracing.py hide_inputs); this script re-applies _redact_pii to the
downloaded payloads as defense in depth before writing to disk.

Usage:
    venv\\Scripts\\python.exe scripts\\export_langsmith_traces.py [project_name]

Requires LANGCHAIN_API_KEY in the environment or .env (read-only; no LLM calls,
no token spend — pure LangSmith API reads).
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.tracing import _redact_pii  # noqa: E402

OUT_DIR = ROOT / "langsmith_export"
PROJECT_NAME_DEFAULT = os.environ.get("LANGCHAIN_PROJECT", "resume-optimizer")

# Uppercase SmithDB v2 field names. Progressive fallback lists in case a field
# is not supported by the deployed backend version.
_SELECTS_FULL = [
    "ID", "NAME", "RUN_TYPE", "STATUS", "START_TIME", "END_TIME", "ERROR",
    "INPUTS", "OUTPUTS", "TAGS", "EXTRA", "EVENTS", "FEEDBACK_STATS",
    "TRACE_ID", "DOTTED_ORDER", "PARENT_RUN_IDS", "TOTAL_TOKENS",
    "PROMPT_TOKENS", "COMPLETION_TOKENS", "TOTAL_COST", "LATENCY_SECONDS",
    "FIRST_TOKEN_TIME",
]
_SELECTS_MID = [f for f in _SELECTS_FULL
                if f not in ("EVENTS", "FEEDBACK_STATS", "FIRST_TOKEN_TIME")]
_SELECTS_CORE = [f for f in _SELECTS_MID if f not in ("LATENCY_SECONDS", "TAGS")]


def _load_dotenv() -> None:
    """Sets LANGCHAIN_* / GROQ_* variables from the repo .env (never printed)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _enum_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.upper() if value else None
    return str(getattr(value, "value", value)).upper()


def run_to_record(run) -> dict:
    """Converts a v2 Run object into a plain JSON-safe dict."""
    record = {
        "id": str(run.id),
        "trace_id": str(run.trace_id) if getattr(run, "trace_id", None) else None,
        "parent_run_ids": [str(p) for p in (getattr(run, "parent_run_ids", None) or [])],
        "dotted_order": getattr(run, "dotted_order", None),
        "name": getattr(run, "name", None),
        "run_type": _enum_str(getattr(run, "run_type", None)),
        "status": getattr(run, "status", None),
        "start_time": _iso(getattr(run, "start_time", None)),
        "end_time": _iso(getattr(run, "end_time", None)),
        "error": getattr(run, "error", None),
        "latency_seconds": getattr(run, "latency_seconds", None),
        "first_token_time": _iso(getattr(run, "first_token_time", None)),
        "total_tokens": getattr(run, "total_tokens", None),
        "prompt_tokens": getattr(run, "prompt_tokens", None),
        "completion_tokens": getattr(run, "completion_tokens", None),
        "total_cost": getattr(run, "total_cost", None),
        "prompt_cost": getattr(run, "prompt_cost", None),
        "completion_cost": getattr(run, "completion_cost", None),
        "tags": list(getattr(run, "tags", None) or []),
        "extra": getattr(run, "extra", None),
        "feedback_stats": getattr(run, "feedback_stats", None),
    }
    inputs = getattr(run, "inputs", None)
    outputs = getattr(run, "outputs", None)
    record["inputs"] = _redact_pii(inputs) if inputs else None
    record["outputs"] = _redact_pii(outputs) if outputs else None
    return record


def build_trees(runs: list[dict]) -> list[dict]:
    """Nests flat runs into trace trees via dotted_order prefixes.

    A run's direct parent is the run with the LONGEST dotted_order that is a
    proper string prefix (runs returned by the API may be unordered and
    paginated across parents and children).
    """
    orders = [r for r in runs if r.get("dotted_order")]
    trees, children = [], defaultdict(list)
    for run in runs:
        parent = None
        me = run.get("dotted_order") or ""
        for other in orders:
            other_do = other["dotted_order"]
            if me.startswith(other_do) and len(other_do) < len(me):
                if parent is None or len(other_do) > len(parent["dotted_order"]):
                    parent = other
        if parent is None:
            trees.append({"run": run, "children": []})
        else:
            children[parent["id"]].append({"run": run, "children": []})

    def attach(nodes: list[dict]) -> list[dict]:
        for node in nodes:
            node["children"] = attach(children.get(node["run"]["id"], []))
        return nodes

    return attach(trees)


async def _fetch_all_runs(client, project_id: str) -> list:
    """Iterates every run in the project across pagination cursors.

    The v2 endpoint caps any query window to 401 days, so history is walked
    backwards in <=400-day windows with a 2-second overlap, deduplicated by
    run id (covers projects older than the cap; young projects return after
    the first window).
    """
    from datetime import datetime as _dt, timedelta as _td

    now = _dt.now(timezone.utc)
    window = _td(days=400)
    overlap = _td(seconds=2)
    max_start = now + _td(seconds=2)
    collected: dict[str, object] = {}
    last_error = None
    for _window_index in range(40):
        window_min = max_start - window - overlap
        now_window: list = []
        for selects in (_SELECTS_FULL, _SELECTS_MID, _SELECTS_CORE):
            try:
                async for run in client.runs.query(
                        project_ids=[project_id],
                        min_start_time=window_min,
                        max_start_time=max_start,
                        page_size=200,
                        selects=selects):
                    now_window.append(run)
                break
            except Exception as exc:  # unsupported field -> retry smaller selects
                last_error = exc
                continue
        else:
            raise RuntimeError(f"runs.query failed with all selects variants: {last_error}")

        earliest = None
        for run in now_window:
            collected[str(run.id)] = run
            run_start = getattr(run, "start_time", None)
            if run_start is not None and (earliest is None or run_start < earliest):
                earliest = run_start
        if earliest is None:
            break  # reached the beginning of the project history
        max_start = earliest + overlap
    return list(collected.values())


def main(project_name: str) -> None:
    _load_dotenv()
    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise SystemExit("LANGCHAIN_API_KEY not set (env or .env) - cannot export.")

    from langsmith import Client

    client = Client()

    async def _async_main():
        project = None
        try:
            project = await client.aread_project(project_name=project_name)
        except AttributeError:
            project = client.read_project(project_name=project_name)
        if project is None:
            raise SystemExit(f"Project '{project_name}' not found on LangSmith.")
        return await _fetch_all_runs(client, str(project.id))

    raw_runs = asyncio.run(_async_main())
    records = [run_to_record(run) for run in raw_runs]
    records.sort(key=lambda r: (r["start_time"] or "", r["dotted_order"] or ""))

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    runs_file = OUT_DIR / f"runs-all-{stamp}.jsonl"
    with runs_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    trees = build_trees(records)
    traces_dir = OUT_DIR / "traces"
    traces_dir.mkdir(exist_ok=True)
    for tree in trees:
        tid = tree["run"]["trace_id"] or tree["run"]["id"]
        short = str(tid)[:8]
        date = (tree["run"]["start_time"] or "")[:10]
        outline = (tree["run"]["name"] or "run").replace(" ", "-")
        path = traces_dir / f"trace-{short}-{date}-{outline}.json"
        path.write_text(json.dumps(tree, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")

    named_errors = [r for r in records if r["status"] == "ERROR"]
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "project_id": None,
        "runs": len(records),
        "traces": len(trees),
        "llm_runs": sum(1 for r in records if r["run_type"] == "LLM"),
        "error_runs": len(named_errors),
        "note": ("Token/cost fields are per-run own usage (SmithDB v2); trace "
                 "totals are computed by scripts/analyze_langsmith_traces.py. "
                 "Server data was redacted at trace time (hide_inputs) and "
                 "re-redacted locally before writing."),
        "runs_file": runs_file.name,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[export] {len(records)} runs / {len(trees)} traces "
          f"({meta['llm_runs']} LLM runs, {len(named_errors)} errors)")
    print(f"[export] flat log:   {runs_file.relative_to(ROOT)}")
    print(f"[export] tree files: {traces_dir.relative_to(ROOT)}\\ ({len(trees)} files)")
    print(f"[export] meta:       {(OUT_DIR / 'meta.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PROJECT_NAME_DEFAULT)
