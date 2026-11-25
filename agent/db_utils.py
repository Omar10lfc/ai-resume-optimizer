"""SQLite maintenance helpers shared across pipeline stores and CLI scripts."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def remove_wal_residue(db_path) -> list[Path]:
    """Best-effort removal of -wal/-shm files of a closed SQLite store.

    Returns the paths that existed and were unlinked (errors suppressed).
    """
    removed = []
    for suffix in ("-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


def ensure_db_migrations(db_path, version: int = 1) -> None:
    """Create the versioned ``_migrations`` table on a SQLite store if
    missing and stamp the given schema version. Accepts a path or open Connection.
    """
    owns_conn = not isinstance(db_path, sqlite3.Connection)
    conn = sqlite3.connect(str(db_path)) if owns_conn else db_path
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS _migrations("
                     " version INTEGER PRIMARY KEY,"
                     " applied_at TEXT NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO _migrations(version, applied_at)"
                     " VALUES (?, ?)",
                     (version, time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


def db_schema_version(db_path) -> int:
    """Current ``_migrations`` version of a SQLite store (0 when absent)."""
    if not Path(db_path).exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "_migrations" not in tables:
            return 0
        row = conn.execute("SELECT MAX(version) FROM _migrations").fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()
