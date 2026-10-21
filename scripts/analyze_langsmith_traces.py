"""Analyze the bulk LangSmith export produced by export_langsmith_traces.py.

Reads the flat runs JSONL (any file passed as argv[1], else the newest
langsmith_export/runs-all-*.jsonl) and emits:

- a console summary (time span, run counts, errors, token totals)
- a full Markdown report: langsmith_export/analysis_report-<file>.md with
  per-run-name stats, per-trace table, model/token breakdown, error catalog,
  and reviewer-score history

No network access: analysis is fully offline over the exported data.

Usage:
    venv\\Scripts\\python.exe scripts\\analyze_langsmith_traces.py [runs.jsonl]
"""

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "langsmith_export"


def _latest_runs_file() -> Path:
    files = sorted(OUT_DIR.glob("runs-all-*.jsonl"))
    if not files:
        raise SystemExit(f"No runs-all-*.jsonl found in {OUT_DIR} - run "
                         f"scripts/export_langsmith_traces.py first.")
    return files[-1]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def _model_of(run: dict) -> str:
    extra = run.get("extra") or {}
    metadata = (extra.get("metadata") or {}) if isinstance(extra, dict) else {}
    invocation = (extra.get("invocation_params") or {}) if isinstance(extra, dict) else {}
    for key in ("ls_model_name", "model", "model_name", "ls_groq_model"):
        if isinstance(metadata, dict) and metadata.get(key):
            return str(metadata[key])
        if isinstance(invocation, dict) and invocation.get(key):
            return str(invocation[key])
    return "unknown"


def _score_from_outputs(run: dict) -> float | None:
    """Reviewer structured-output runs carry {score, feedback} JSON."""
    outputs = run.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for value in outputs.values():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
            return float(value["score"])
    return None


def _fmt_ts(value: str | None) -> str:
    parsed = _parse_ts(value)
    if not parsed:
        return "-"
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def analyze(runs: list[dict]) -> str:
    traces: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        traces[run.get("trace_id") or run["id"]].append(run)

    starts = [t for r in runs if (t := _parse_ts(r.get("start_time")))]
    span = (max(starts) - min(starts)) if starts else None
    error_runs = [r for r in runs if (r.get("status") or "").lower() == "ERROR".lower()]
    llm_runs = [r for r in runs if r.get("run_type") == "LLM"]
    tokens_total = sum(r.get("total_tokens") or 0 for r in llm_runs)
    prompt_total = sum(r.get("prompt_tokens") or 0 for r in llm_runs)
    completion_total = sum(r.get("completion_tokens") or 0 for r in llm_runs)

    lines = [
        "# LangSmith trace analysis report",
        "",
        f"**Project:** resume-optimizer"
        f"  \n**Span:** {_fmt_ts(runs[0].get('start_time')) if runs else '-'}"
        f" -> {_fmt_ts(max(runs, key=lambda r: r.get('start_time') or '').get('start_time')) if runs else '-'}"
        f"  \n**Wall-clock covered:** {span.total_seconds() / 3600:.1f} h" if span else "",
        "",
        f"**Total runs:** {len(runs)} ({len(llm_runs)} LLM, {len(error_runs)} errored)"
        f"  \n**Traces:** {len(traces)}"
        f"  \n**LLM tokens (own-usage sum over LLM runs):** {tokens_total:,}"
        f" (prompt {prompt_total:,} / completion {completion_total:,})",
        "",
    ]

    # ---- by run name ------------------------------------------------------
    by_name: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_name[run.get("name") or "(none)"].append(run)
    lines += [
        "## Runs by name", "",
        "| Run name | Type | Count | Errors | Median latency (s) | p95 latency (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, group in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        kind = group[0].get("run_type") or "-"
        latencies = [r.get("latency_seconds") or 0.0 for r in group]
        errors = sum(1 for r in group if r.get("status") == "ERROR")
        lines.append(f"| `{name}` | {kind} | {len(group)} | {errors} | "
                     f"{statistics.median(latencies) or 0:.2f} | "
                     f"{_percentile(latencies, 0.95):.2f} |")

    # ---- by model ---------------------------------------------------------
    by_model = defaultdict(lambda: {"runs": 0, "prompt": 0, "completion": 0, "total": 0})
    for run in llm_runs:
        stats = by_model[_model_of(run)]
        stats["runs"] += 1
        stats["prompt"] += run.get("prompt_tokens") or 0
        stats["completion"] += run.get("completion_tokens") or 0
        stats["total"] += run.get("total_tokens") or 0
    if by_model:
        lines += ["", "## LLM usage by model", "",
                  "| Model | LLM runs | Prompt tokens | Completion tokens | Total tokens |",
                  "|---|---:|---:|---:|---:|"]
        for model, stats in sorted(by_model.items(), key=lambda kv: -kv[1]["total"]):
            lines.append(f"| `{model}` | {stats['runs']} | {stats['prompt']:,} | "
                         f"{stats['completion']:,} | {stats['total']:,} |")

    # ---- per trace --------------------------------------------------------
    lines += ["", "## Traces (most recent first)", "",
              "| Trace | UTC start | Runs | LLM runs | Tokens (sum over LLM) | Wall time (s) | Reviewer score | Errors |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    trace_rows = []
    for tid, group in traces.items():
        first = min(group, key=lambda r: r.get("start_time") or "")
        last_ends = max((_parse_ts(r.get("end_time")) for r in group if r.get("end_time")),
                        default=None)
        wall = (last_ends - _parse_ts(first.get("start_time"))).total_seconds() \
            if last_ends and _parse_ts(first.get("start_time")) else None
        group_llm = [r for r in group if r.get("run_type") == "LLM"]
        tokens = sum(r.get("total_tokens") or 0 for r in group_llm)
        scores = [s for r in group if (s := _score_from_outputs(r)) is not None]
        group_errors = [r for r in group if (r.get("status") or "").lower() == "error"]
        root = [r for r in group if not r.get("parent_run_ids")]
        own_name = (root[0].get("name") or "-") if root else (first.get("name") or "-")
        trace_rows.append((first.get("start_time") or "", tid, own_name, len(group),
                           len(group_llm), tokens, wall, scores, group_errors))
    for start, tid, root_name, n_runs, n_llm, tokens, wall, scores, err_group \
            in sorted(trace_rows, reverse=True):
        score_txt = "/".join(f"{s:.0f}" for s in scores) if scores else "-"
        wall_txt = f"{wall:.1f}" if wall is not None else "-"
        lines.append(f"| `{str(tid)[:12]}` {root_name} | {start[:19].replace('T', ' ')} | "
                     f"{n_runs} | {n_llm} | {tokens:,} | {wall_txt} | {score_txt} | "
                     f"{len(err_group)} |")

    # ---- reviewer score history -------------------------------------------
    all_scores = [(r.get("start_time") or "", _score_from_outputs(r))
                  for r in runs if _score_from_outputs(r) is not None]
    if all_scores:
        values = [s for _, s in all_scores]
        lines += ["", "## Reviewer score history (structured ReviewOutput runs)", "",
                  f"- Runs with a score: {len(values)}",
                  f"- Min / median / max: {min(values):.0f} / "
                  f"{statistics.median(values):.0f} / {max(values):.0f}",
                  f"- Latest 10 oldest->newest: "
                  + ", ".join(f"{s:.0f}" for _, s in sorted(all_scores)[-10:])]

    # ---- error catalog ----------------------------------------------------
    if error_runs:
        lines += ["", "## Errors", ""]
        for run in sorted(error_runs, key=lambda r: r.get("start_time") or ""):
            message = (run.get("error") or "").replace("\n", " ")[:220]
            lines.append(f"- `{run.get('name')}` @ {_fmt_ts(run.get('start_time'))}: "
                         f"{message or '(no message captured)'}")

    return "\n".join(lines) + "\n"


def main() -> None:
    runs_file = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_runs_file()
    runs = [json.loads(line) for line in runs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not runs:
        raise SystemExit(f"{runs_file} contains no runs.")
    report = analyze(runs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_file = OUT_DIR / f"analysis_report-{runs_file.stem}-{stamp}.md"
    report_file.write_text(report, encoding="utf-8")
    print(report)
    print(f"[analysis] report saved to {report_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
