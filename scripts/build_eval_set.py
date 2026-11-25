"""Build + replay the LangSmith eval set from real pipeline traces.

Fetches the project's ROOT pipeline runs through ``client.runs.query``
(the ``list_runs`` replacement that deprecates after Jan 31 2027),
extracts ``{job_description, resume_text} -> result`` examples
(PII re-redacted locally), pushes them into the
``resume-optimizer-evals`` dataset (deduped by the jd+resume pair), and
replays the three CODE-ONLY evaluators (score sanity, integrity scan
re-run, keyword precision) into a console report.

Usage:
    venv\\Scripts\\python.exe scripts\\build_eval_set.py [--project NAME]
        [--limit 50] [--window-days 400] [--no-push] [--strict]

Requires LANGCHAIN_API_KEY in the environment or .env. Zero LLM calls -
the only traffic is LangSmith reads/writes.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evals import EVAL_DATASET_NAME, build_eval_set  # noqa: E402


def _load_dotenv() -> None:
    """Sets LANGCHAIN_* variables from the repo .env (never printed)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def _print_report(report: dict) -> None:
    print(f"\n== replay report ({report['examples']} examples) ==")
    for key, stats in report["evaluators"].items():
        print(f"  {key:<20} passed {stats['passed']:>3}  "
              f"failed {stats['failed']:>3}  unscored {stats['unscored']:>3}")
    if report["failures"]:
        print("  failures:")
        for f in report["failures"]:
            print(f"    [{f['eval']}] run {f['run_id'][:8]}: {f['reason']}")
    else:
        print("  no failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project",
                        default=os.environ.get("LANGCHAIN_PROJECT",
                                               "resume-optimizer"))
    parser.add_argument("--limit", type=int, default=50,
                        help="max root runs to fetch (newest first)")
    parser.add_argument("--window-days", type=int, default=400,
                        help="trace window (clamped to the 401-day API cap)")
    parser.add_argument("--no-push", action="store_true",
                        help="extract + replay only; leave the dataset alone")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any evaluator fails "
                             "(CI-ready)")
    args = parser.parse_args()

    _load_dotenv()
    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise SystemExit("LANGCHAIN_API_KEY not set (env or .env) - cannot "
                         "read LangSmith.")

    from langsmith import Client

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    stats = asyncio.run(build_eval_set(
        client, args.project, limit=args.limit,
        window_days=args.window_days, push=not args.no_push))

    print(f"project: {args.project}  dataset: {EVAL_DATASET_NAME}")
    print(f"fetched roots: {stats['fetched']}  "
          f"pipeline examples: {stats['extracted']}")
    if not args.no_push:
        print(f"dataset: pushed {stats.get('pushed', 0)}  "
              f"skipped (already present) {stats.get('skipped', 0)}")
    _print_report(stats["report"])

    if args.strict and stats["report"]["total_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
