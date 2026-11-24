"""BB1 Phase A: the persistent profile store - applications + their runs.

The Dossier study's Application record, reimplemented on this repo's own
SQLite pattern (mirrors ``run_store.py``): the optimized resume becomes
*derived evidence* instead of the only truth. Every completed step-2 run is
recorded automatically (zero UI; the History tab reviews/deletes later) as
one attempt under an APPLICATION identified by the normalized (job text,
resume text) pair - re-optimizing the same posting against the same resume
attaches a new attempt to the existing application instead of forking a
duplicate, so history groups itself.

Plus the submitted-materials archive (adapted from documents/applications/
in the MIT ai-job-search repo): every application also stores the shipped
cover letter text and the artifact manifest (PDF/DOCX/TEX/HTML paths +
API-download URLs). The texts and the manifest are permanent; the artifact
FILES themselves follow the usual retention (the exporter's 24h artifact
sweep applies).

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
    updated_at  REAL NOT NULL,
    -- Archive of the submitted materials (see module docstring): the final
    -- shipped cover letter, and the JSON list of {"kind", "filename", "url"}
    -- artifact rows (PDF/DOCX/TEX/HTML). '' / '[]' on rows written before
    -- the columns existed (migrated by _migrate_applications).
    cover_letter_text TEXT NOT NULL DEFAULT '',
    artifact_paths    TEXT NOT NULL DEFAULT '[]'
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


def _migrate_applications(conn: sqlite3.Connection) -> None:
    """Add the archive columns to an existing database.

    Fresh DBs get the columns from ``_SCHEMA``; rows written by older
    versions of this store get '' / '[]' defaults (the '' for artifact_paths
    is normalized by callers reading it via _parse_artifact_paths).
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(applications)")}
    defaults = {"cover_letter_text": "''", "artifact_paths": "'[]'"}
    for name in ("cover_letter_text", "artifact_paths"):
        if name in columns:
            continue
        try:
            conn.execute(
                f"ALTER TABLE applications ADD COLUMN {name}"
                f" TEXT NOT NULL DEFAULT {defaults[name]}")
        except sqlite3.Error:
            pass


def _connect() -> sqlite3.Connection:
    path = agent_config.PROFILE_DB_PATH
    Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    for statement in _SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    _migrate_applications(conn)
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
                   if k not in ("type", "session",
                                # PB3: a duplicate of this row's stored
                                # resume_text - no at-rest duplication.
                                "original_resume")}
        now = time.time()
        conn = _connect()
        try:
            # The submitted-materials archive (see module docstring): the
            # final shipped letter and the artifact manifest (kind/filename/
            # url, like the SSE result payload's "artifacts" rows).
            cover_letter_text = str(result.get("cover_letter") or "")
            artifact_paths = json.dumps([
                {"kind": a.get("kind"), "filename": a.get("filename"),
                 "url": a.get("url")}
                for a in (result.get("artifacts") or [])
                if isinstance(a, dict) and a.get("kind")
            ], ensure_ascii=False)
            # BEGIN IMMEDIATE (A1): same read-then-write pattern as
            # server/run_store.py::_write_once - acquire the write lock before
            # the dedupe SELECT so concurrent record_run calls serialize on
            # the lock (the UNIQUE index on (jd_hash, resume_hash) stays as a
            # backstop; on the race it raises IntegrityError, caught below).
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT id, title FROM applications"
                    " WHERE jd_hash = ? AND resume_hash = ?",
                    (jd_hash, resume_hash)).fetchone()
                if row is None:
                    title = _title_from(job_text)
                    cur = conn.execute(
                        "INSERT INTO applications(jd_hash, resume_hash,"
                        " title, job_text, resume_text, status,"
                        " cover_letter_text, artifact_paths,"
                        " created_at, updated_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (jd_hash, resume_hash, title, job_text, resume_text,
                         "active", cover_letter_text, artifact_paths, now, now))
                    app_id = cur.lastrowid
                else:
                    app_id = row["id"]
                    conn.execute(
                        "UPDATE applications SET updated_at = ?"
                        " , cover_letter_text = COALESCE(NULLIF(?, ''),"
                        " cover_letter_text)"
                        " , artifact_paths = COALESCE(NULLIF(?, ''),"
                        " artifact_paths)"
                        " WHERE id = ?",
                        (now, cover_letter_text, artifact_paths, app_id))
                cur = conn.execute(
                    "INSERT INTO runs(application_id, session, score,"
                    " ats_percentage, llm_score, review_failed, result_json,"
                    " created_at, updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (app_id, session or "", score, ats, llm, failed,
                     json.dumps(payload, ensure_ascii=False), now, now))
                run_id = cur.lastrowid
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
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
            try:
                app["artifact_paths"] = json.loads(
                    app.get("artifact_paths") or "[]")
            except (TypeError, ValueError):
                app["artifact_paths"] = []
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
    """Complete delete (Item 7): removes the application row, the
    profile.runs cascade, the run_store ledger rows for the app's sessions,
    the checkpointer threads and the artifact directories. WAL/SHM residue
    of the stores is checkpointed and unlinked best-effort."""
    if not agent_config.PROFILE_DB_ENABLED:
        return False
    try:
        # Sessions must be read BEFORE the DELETE: the FK cascade removes
        # the profile.runs rows that carry them.
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, artifact_paths FROM applications WHERE id = ?",
                (app_id,)).fetchone()
            if row is None:
                return False
            sessions = [r[0] for r in conn.execute(
                "SELECT session FROM runs WHERE application_id = ?", (app_id,))]
            artifact_paths = _parse_artifact_paths(row["artifact_paths"])
            with conn:
                conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
        finally:
            conn.close()
        # Item 7: ledger rows (run_store), checkpointer threads, artifacts.
        from server import run_store
        from agent import graphs
        run_store.delete_sessions(sessions)
        graphs.delete_checkpoint_threads(sessions)
        _delete_artifact_dirs(sessions, artifact_paths)
        return True
    except Exception as exc:
        print(f"[Profile] delete failed (non-fatal): {exc}")
        return False


def wipe() -> int:
    """Deletes every application (and its runs via the FK cascade) - the
    'delete everything' backend action (Item 7; ledger/threads/artifacts
    are wiped by the caller). Returns the number of applications removed."""
    if not agent_config.PROFILE_DB_ENABLED:
        return 0
    try:
        conn = _connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM applications")
            n = cur.rowcount
        finally:
            conn.close()
        from agent.helpers import remove_wal_residue
        remove_wal_residue(agent_config.PROFILE_DB_PATH)
        return n
    except Exception as exc:
        print(f"[Profile] wipe failed (non-fatal): {exc}")
        return 0


def _parse_artifact_paths(raw) -> list[dict]:
    """Parses the applications.artifact_paths JSON (Tolerant to bad JSON)."""
    try:
        rows = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _delete_artifact_dirs(sessions: list[str], artifact_paths: list[dict]) -> int:
    """Deletes the artifact directories under OUTPUT_ROOT for the deleted
    application (Item 7: complete delete). The artifact rows' "url" is
    api/v1/artifacts/<dir-name>/<kind>?file=... - the URL segment is the
    actual artifact dir name (the loader's request_id, independent of the
    session) - plus OUTPUT_ROOT/<session> dirs. Returns dirs removed."""
    import shutil
    dirs = {s for s in sessions if s}
    for row in artifact_paths:
        url = str(row.get("url") or "")
        parts = url.split("/")
        if "artifacts" in parts:
            dirs.add(parts[parts.index("artifacts") + 1])
    removed = 0
    for d in dirs:
        p = agent_config.OUTPUT_ROOT / d
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    return removed
