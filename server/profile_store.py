"""BB1 Phase A: the persistent profile store - applications + their runs.

The Dossier study's Application record, reimplemented on this repo's own
SQLite pattern (mirrors ``run_store.py``): the optimized resume becomes
*derived evidence* instead of the only truth. Every completed step-2 run is
recorded automatically (zero UI; the History tab reviews/deletes later) as
one attempt under an APPLICATION identified by the normalized (job text,
resume text) pair - re-optimizing the same posting against the same resume
attaches a new attempt to the existing application instead of forking a
duplicate, so history groups itself.

Differences from the runs ledger (both stores exist on purpose):
- PERSISTENT: no retention sweep. The resume/job text at rest IS the point
  (the artifacts under %TEMP% evaporate after 24h; the history does not).
  Set ``PROFILE_DB_PATH=off`` for the old ephemeral posture.
- Writes are non-fatal like the ledger: history must never break a run
  the LLMs already paid for.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from agent import config as agent_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    jd_hash     TEXT NOT NULL,
    resume_hash TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    job_text    TEXT NOT NULL DEFAULT '',
    resume_text TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_applications_pair
    ON applications(jd_hash, resume_hash);
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    session        TEXT NOT NULL DEFAULT '',
    score          REAL NOT NULL DEFAULT 0,
    ats_percentage REAL NOT NULL DEFAULT 0,
    llm_score      REAL NOT NULL DEFAULT 0,
    review_failed  INTEGER NOT NULL DEFAULT 0,
    result_json    TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_application
    ON runs(application_id, created_at);
"""


def _connect() -> sqlite3.Connection:
    path = agent_config.PROFILE_DB_PATH
    Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    for statement in _SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join((text or "").split()).encode()).hexdigest()


def _title_from(job_text: str) -> str:
    """First non-empty line of the JD, truncated - a stable human handle."""
    for line in (job_text or "").splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "(untitled posting)"


def record_run(job_text: str, resume_text: str, session: str,
               result: dict) -> int | None:
    """Records one completed optimization attempt.

    Dedupes the application on the normalized (JD, resume) pair; returns
    the run row id (or None when disabled / on failure - never raises).
    The full result payload (requirement table, prompt versions, token
    usage, artifacts list) rides in ``result_json`` exactly like the runs
    ledger's reattach payload.
    """
    if not agent_config.PROFILE_DB_ENABLED:
        return None
    try:
        job_text = job_text or ""
        resume_text = resume_text or ""
        jd_hash, resume_hash = _hash(job_text), _hash(resume_text)
        score = float(result.get("score") or 0)
        ats = float(result.get("ats_percentage") or 0)
        llm = float(result.get("llm_quality_score") or 0)
        failed = 1 if result.get("review_failed") else 0
        payload = {k: v for k, v in result.items()
                   if k not in ("type", "session")}
        now = time.time()
        conn = _connect()
        try:
            with conn:
                row = conn.execute(
                    "SELECT id, title FROM applications"
                    " WHERE jd_hash = ? AND resume_hash = ?",
                    (jd_hash, resume_hash)).fetchone()
                if row is None:
                    title = _title_from(job_text)
                    cur = conn.execute(
                        "INSERT INTO applications(jd_hash, resume_hash,"
                        " title, job_text, resume_text, status,"
                        " created_at, updated_at)"
                        " VALUES(?,?,?,?,?,'active',?,?)",
                        (jd_hash, resume_hash, title, job_text, resume_text,
                         now, now))
                    app_id = cur.lastrowid
                else:
                    app_id = row["id"]
                    conn.execute(
                        "UPDATE applications SET updated_at = ? WHERE id = ?",
                        (now, app_id))
                cur = conn.execute(
                    "INSERT INTO runs(application_id, session, score,"
                    " ats_percentage, llm_score, review_failed, result_json,"
                    " created_at, updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (app_id, session or "", score, ats, llm, failed,
                     json.dumps(payload, ensure_ascii=False), now, now))
                run_id = cur.lastrowid
        finally:
            conn.close()
        return run_id
    except Exception as exc:
        print(f"[Profile] recording failed (non-fatal): {exc}")
        return None


def list_applications(limit: int = 50) -> list[dict]:
    """Newest-first application summaries (id, title, status, timestamps,
    best/latest score, attempt count). Resume/job TEXT is NOT included
    here - the list endpoint stays cheap; detail comes from ``get``."""
    if not agent_config.PROFILE_DB_ENABLED:
        return []
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT a.id, a.title, a.status, a.created_at, a.updated_at,
                          COUNT(r.id)              AS attempts,
                          MAX(r.score)             AS best_score,
                          MAX(r.created_at)        AS last_run_at
                   FROM applications a LEFT JOIN runs r
                     ON r.application_id = a.id
                   GROUP BY a.id
                   ORDER BY a.updated_at DESC
                   LIMIT ?""", (int(limit),)).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[Profile] listing failed (non-fatal): {exc}")
        return []


def get_application(app_id: int) -> dict | None:
    """One application with all of its runs (newest first); each run's
    result payload parsed. ``None`` when unknown/disabled."""
    if not agent_config.PROFILE_DB_ENABLED:
        return None
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
            if row is None:
                return None
            app = dict(row)
            run_rows = conn.execute(
                "SELECT * FROM runs WHERE application_id = ?"
                " ORDER BY created_at DESC, id DESC", (app_id,)).fetchall()
        finally:
            conn.close()
        runs = []
        for r in run_rows:
            data = dict(r)
            try:
                data["result"] = (json.loads(data.pop("result_json"))
                                  if data.get("result_json") else None)
            except (TypeError, ValueError):
                data["result"] = None
            runs.append(data)
        return {**app, "runs": runs}
    except Exception as exc:
        print(f"[Profile] read failed (non-fatal): {exc}")
        return None


def delete_application(app_id: int) -> bool:
    """Removes one application and (via FK cascade) its runs."""
    if not agent_config.PROFILE_DB_ENABLED:
        return False
    try:
        conn = _connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM applications WHERE id = ?",
                                   (app_id,))
            return cur.rowcount > 0
        finally:
            conn.close()
    except Exception as exc:
        print(f"[Profile] delete failed (non-fatal): {exc}")
        return False
