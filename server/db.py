"""Shared SQLite connection and one-time per-database schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from agent.helpers import ensure_db_migrations


def connect_sqlite(
    path: str | Path,
    schema_statements: tuple[str, ...] | list[str],
    *,
    schema_version: int = 1,
    migrate_fn: Callable[[sqlite3.Connection], None] | None = None,
    foreign_keys: bool = False,
) -> sqlite3.Connection:
    """Open a configured SQLite connection, initializing schema/WAL once per DB."""
    db_path = Path(path)
    db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")

    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if current_version < schema_version:
        conn.execute("PRAGMA journal_mode = WAL")
        for stmt in schema_statements:
            s = stmt.strip()
            if s:
                conn.execute(s)
        if migrate_fn is not None:
            migrate_fn(conn)
        ensure_db_migrations(conn, schema_version)
        conn.execute(f"PRAGMA user_version = {int(schema_version)}")
        conn.commit()
    return conn
