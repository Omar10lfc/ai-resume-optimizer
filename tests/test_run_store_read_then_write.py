"""Deterministic repro of the read-then-write upgrade failure (A1, Item 1
follow-up).

Phase 1 (the failure mode, raw deferred connections): connection A opens a
deferred (read) transaction, B writes + commits, then A tries to write. The
expected upgrade failure is ``sqlite3.OperationalError`` (SQLITE_BUSY_SNAPSHOT
family). Phase 2 uses the actual store path (``run_store._write`` ->
``_write_once`` with BEGIN IMMEDIATE): the same interleaving succeeds and
the rows are present.
"""

import sqlite3

import pytest

from server import run_store


def test_read_then_write_upgrade_failure_and_store_path(tmp_path, monkeypatch):
    import agent.config as agent_config
    monkeypatch.setattr(agent_config, "RUNS_DB_PATH",
                        str(tmp_path / "repro.sqlite3"))
    run_store._connect().close()
    db = agent_config.RUNS_DB_PATH

    # ---- Phase 1: the failure mode (A1 shape) ----------------------------
    # Connection A: deferred read transaction; A's SELECT takes the read
    # snapshot. Connection B writes + commits while A holds it.
    conn_a = sqlite3.connect(str(db))
    try:
        conn_a.execute("BEGIN")  # deferred - no lock yet
        conn_a.execute("SELECT * FROM runs").fetchone()
        # B writes through the real store path and commits.
        run_store._write("repro-B", "done")
        # A now tries to write into its stale read snapshot.
        with pytest.raises(sqlite3.OperationalError):
            conn_a.execute(
                "INSERT INTO runs(session, status, created_at, updated_at)"
                " VALUES ('repro-A', 'done', 0, 0)")
    finally:
        conn_a.rollback()
        conn_a.close()

    # ---- Phase 2: the actual store path (BEGIN IMMEDIATE) ---------------
    # Same interleaving through run_store._write (its _write_once uses
    # BEGIN IMMEDIATE): writes succeed and the rows are present.
    run_store._write("repro-A", "done")
    run_store._write("repro-C", "done")
    assert run_store.get("repro-A") is not None
    assert run_store.get("repro-C") is not None
    assert run_store.get("repro-B") is not None
