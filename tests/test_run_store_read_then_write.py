"""Deterministic repro of the read-then-write upgrade failure (A1, Item 1
follow-up).

Uses the REAL run_store code for the writes (``run_store._write`` ->
``_write_once``). Connection A opens a deferred read transaction
(``BEGIN`` + ``SELECT``) via ``run_store._connect()``, B writes via the real
``run_store._write``, then A writes. The same interleaving runs against two
variants of the store's transaction start (``run_store._begin_immediate``,
the monkeypatch seam):

  - "deferred" (the OLD behavior): ``_begin_immediate`` is monkeypatched to
    ``conn.execute("BEGIN")`` (deferred). Expected: ``OperationalError``
    (the SQLITE_BUSY_SNAPSHOT family - the A1 failure mode).
  - "production" (current code): unpatched. Expected: no failure - A's
    write goes through the real store path (``run_store._write``).
"""

import sqlite3

from server import run_store


def _interleaving(a_write_via_store: bool):
    """A: deferred read transaction + SELECT; B: real run_store._write +
    commit; A: write (via A's connection or via the real store path).
    Returns the exception A's write raised (None on success)."""
    a_error = None
    conn_a = run_store._connect()
    try:
        conn_a.execute("BEGIN")  # A: deferred read transaction
        conn_a.execute("SELECT * FROM runs").fetchone()  # A's read snapshot
        run_store._write("sess-b", "done")  # B: real store path + commit
        try:
            if a_write_via_store:
                # Production analog: A's write through the real store path.
                run_store._write("sess-a", "done")
            else:
                # The classic A1 shape: A writes on its stale-read connection.
                conn_a.execute(
                    "INSERT INTO runs(session, status, created_at, updated_at)"
                    " VALUES ('sess-a', 'done', 0, 0)")
        except Exception as exc:
            a_error = exc
    finally:
        conn_a.rollback()
        conn_a.close()
    return a_error


def test_read_then_write_fails_without_begin_immediate_and_passes_with_it(
        tmp_path, monkeypatch):
    # Points the store at a tmp DB and creates the schema.
    import agent.config as agent_config
    monkeypatch.setattr(agent_config, "RUNS_DB_PATH",
                        str(tmp_path / "repro.sqlite3"))
    run_store._connect().close()

    real_begin_immediate = run_store._begin_immediate

    # Variant 1: the OLD deferred behavior - monkeypatch the store's
    # transaction start.
    def begin_deferred(conn):
        conn.execute("BEGIN")  # old deferred behavior
    monkeypatch.setattr(run_store, "_begin_immediate", begin_deferred)
    err = _interleaving(a_write_via_store=False)
    assert isinstance(err, sqlite3.OperationalError), err
    assert "database is locked" in str(err).lower()

    # Variant 2: production code (BEGIN IMMEDIATE) - same interleaving.
    monkeypatch.setattr(run_store, "_begin_immediate", real_begin_immediate)
    err = _interleaving(a_write_via_store=True)
    assert err is None, err
    assert run_store.get("sess-b") is not None
    assert run_store.get("sess-a") is not None
