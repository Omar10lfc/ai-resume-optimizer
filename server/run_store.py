"""MR3 durable runs: a SQLite ledger of run status that survives restarts.

One row per session - the latest run's status, the last step reached, the
step-1 scanner chips, the error, and (for done runs) the full result
payload - so the SPA can reattach after a reload/disconnect via
``GET /api/v1/runs/{session}`` and the backend can recover honestly:

- ``running`` rows found at startup are marked failed-with-retry (the
  Dossier study's ``recoverInterruptedJobs()``): a process restart kills
  mid-run LLM work, and the leftover LangGraph checkpoint must never be
  resumed silently (that produced the 7.4s band-aid run and the stale-ATS
  snapshot on live runs 29afd91b / 62725081).
- ``awaiting_notes`` rows (step 1 finished, waiting for the user) survive
  restarts together with the LangGraph checkpoint, so the review
  checkpoint is durable, not just the artifacts.
- mid-pipeline recovery reads the step-1 chips from here: the checkpoint's
  own ``missing_skills`` copy was already overwritten by the crashed
  attempt's note injection.

The ledger lives next to the artifacts under ``OUTPUT_ROOT`` (gitignored
temp space, same 24h retention posture: rows older than the retention
window are purged on the next ``start()``). Result payloads therefore
hold resume content at rest for the retention window - the same exposure
as the artifact files already written there.

Recording is deliberately non-fatal: a ledger failure prints a warning
and never breaks the pipeline.
"""

import json
import sqlite3
import time
from pathlib import Path

from agent import config as agent_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    session       TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    step          TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    missing_skills TEXT NOT NULL DEFAULT '',
    result_json   TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
)
"""

# Status lifecycle: running (streaming) -> awaiting_notes (step 1 done) ->
# done (result delivered) | failed (error / disconnect / restart recovery).


def _connect() -> sqlite3.Connection:
    path = agent_config.RUNS_DB_PATH
    Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _write(session: str, status: str, *, step: str | None = None,
           error: str | None = None, missing_skills: str | None = None,
           result: dict | None = None) -> None:
    """Upsert one row, preserving fields the caller does not touch.

    Failure is always non-fatal: the ledger must never break a run that
    the LLMs already paid for.
    """
    try:
        conn = _connect()
        try:
            with conn:
                existing = conn.execute(
                    "SELECT missing_skills, result_json FROM runs"
                    " WHERE session = ?", (session,)).fetchone()
                ms = (missing_skills if missing_skills is not None
                      else (existing["missing_skills"] if existing else ""))
                rj = (json.dumps(result, ensure_ascii=False)
                      if result is not None
                      else (existing["result_json"] if existing else None))
                now = time.time()
                conn.execute(
                    """INSERT INTO runs(session, status, step, error,
                                        missing_skills, result_json,
                                        created_at, updated_at)
                       VALUES(?,?,?,?,?,?,?,?)
                       ON CONFLICT(session) DO UPDATE SET
                           status=excluded.status, step=excluded.step,
                           error=excluded.error,
                           missing_skills=excluded.missing_skills,
                           result_json=excluded.result_json,
                           updated_at=excluded.updated_at""",
                    (session, status, step or "", error or "", ms, rj,
                     now, now))
        finally:
            conn.close()
    except Exception as exc:
        print(f"[Runs] ledger write failed (non-fatal): {exc}")


def start(session: str, step: str) -> None:
    """Opens (or reopens) a run row in the running state. Also purges rows
    older than the artifact retention window - every fresh run sweeps."""
    cutoff = time.time() - agent_config.OUTPUT_RETENTION_HOURS * 3600
    try:
        conn = _connect()
        try:
            with conn:
                conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,))
        finally:
            conn.close()
    except Exception as exc:
        print(f"[Runs] ledger purge failed (non-fatal): {exc}")
    _write(session, "running", step=step)


def step(session: str, node: str) -> None:
    """Records the last graph node that started (per-step progress)."""
    _write(session, "running", step=node)


def finish_scan(session: str, missing_skills: str) -> None:
    """Step 1 finished: the session now waits for the user's edited notes.

    The chips are stored here because the mid-pipeline recovery path needs
    them after the checkpoint's own copy was overwritten (see module
    docstring)."""
    _write(session, "awaiting_notes", step="step1",
           missing_skills=missing_skills)


def finish(session: str, result: dict) -> None:
    """Run completed: the full result payload is kept for reattach."""
    payload = {k: v for k, v in result.items() if k not in ("type", "session")}
    _write(session, "done", step="done", result=payload)


def fail(session: str, error: str) -> None:
    _write(session, "failed", error=error)


def supersede(session: str, message: str) -> None:
    """Marks a row superseded (the runner switched to a fresh session
    because this thread already held a completed run). A distinct status
    so startup recovery never mistakes it for an interrupted run."""
    _write(session, "superseded", error=message)


def get(session: str) -> dict | None:
    """Returns the run row (result payload parsed), or None."""
    try:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE session = ?",
                               (session,)).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[Runs] ledger read failed (non-fatal): {exc}")
        return None
    if row is None:
        return None
    data = dict(row)
    try:
        data["result"] = (json.loads(data.pop("result_json"))
                          if data.get("result_json") else None)
    except (TypeError, ValueError):
        data["result"] = None
    return data


def recover_interrupted() -> int:
    """Startup recovery: rows still 'running' belong to a dead process
    (their mid-run LangGraph checkpoints must not be resumed silently) ->
    marked failed with an actionable retry message. 'awaiting_notes' rows
    are left alone: the review checkpoint is durable by design."""
    try:
        conn = _connect()
        try:
            with conn:
                cur = conn.execute(
                    "UPDATE runs SET status = 'failed',"
                    " error = 'Interrupted by a backend restart - run Step 2 again.',"
                    " updated_at = ? WHERE status = 'running'",
                    (time.time(),))
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        finally:
            conn.close()
    except Exception as exc:
        print(f"[Runs] startup recovery failed (non-fatal): {exc}")
        return 0
